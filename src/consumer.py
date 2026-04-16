"""
Kafka consumer — reads sensor readings, runs the sliding-window detector,
writes raw readings and anomaly alerts to TimescaleDB, then fires async
LLM explanation tasks in the background.

Architecture
────────────
The event loop runs three coroutines concurrently:

  _kafka_loop   — polls Kafka via run_in_executor (non-blocking to asyncio),
                  feeds readings to the detector, enqueues alerts.

  _explain_loop — drains the alert queue, calls LLMExplainer, updates DB.
                  Multiple concurrent explanation tasks are allowed so that
                  a slow LLM response doesn't queue-starve downstream alerts.

  _stats_loop   — logs throughput and detection statistics every 30 s.

LLM calls never block the Kafka polling path. End-to-end latency from
anomaly detection to explanation available is dominated by LLM inference
(typically 2–8 s on CPU for a 7B model), not by any I/O in this codebase.

Run with:
    python -m src.consumer
"""

from __future__ import annotations

import asyncio
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import structlog
from confluent_kafka import Consumer, KafkaError

from src.config import KAFKA_BOOTSTRAP_SERVERS, KAFKA_GROUP_ID, KAFKA_TOPIC, LOG_LEVEL
from src.db import Database
from src.detector import SlidingWindowDetector
from src.explainer import LLMExplainer
from src.models import SensorReading

log = structlog.get_logger(__name__)

# Maximum concurrent LLM explanation tasks in flight simultaneously
MAX_CONCURRENT_EXPLANATIONS = 4


class Pipeline:
    def __init__(self) -> None:
        self.db        = Database()
        self.detector  = SlidingWindowDetector()
        self.explainer = LLMExplainer()
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="kafka")
        self._alert_queue: asyncio.Queue = asyncio.Queue(maxsize=512)
        self._running  = True

        # Stats
        self._total_readings  = 0
        self._total_anomalies = 0
        self._total_explained = 0
        self._t_start         = time.time()

    # ── Kafka polling (runs in thread pool to avoid blocking event loop) ──────

    def _poll_kafka(self, consumer: Consumer) -> list[SensorReading]:
        msgs = consumer.consume(num_messages=30, timeout=0.5)
        readings: list[SensorReading] = []
        for msg in msgs:
            if msg.error():
                if msg.error().code() != KafkaError._PARTITION_EOF:
                    log.error("consumer.kafka_error", error=str(msg.error()))
                continue
            try:
                readings.append(SensorReading.from_kafka_bytes(msg.value()))
            except Exception as exc:
                log.warning("consumer.parse_error", error=str(exc))
        return readings

    # ── Main coroutines ───────────────────────────────────────────────────────

    async def _kafka_loop(self, consumer: Consumer) -> None:
        loop = asyncio.get_running_loop()
        while self._running:
            readings = await loop.run_in_executor(
                self._executor, self._poll_kafka, consumer
            )
            if not readings:
                continue

            # Batch-insert raw readings
            await self.db.insert_readings_batch(readings)
            self._total_readings += len(readings)

            # Detect anomalies
            for reading in readings:
                alert = self.detector.process(reading)
                if alert:
                    alert.db_id = await self.db.insert_alert(alert)
                    self._total_anomalies += 1

                    # Snapshot the current window history for the LLM prompt
                    history = self.detector.window_history(
                        reading.machine_id, reading.signal
                    )
                    try:
                        self._alert_queue.put_nowait((alert, history))
                    except asyncio.QueueFull:
                        log.warning(
                            "consumer.alert_queue_full",
                            machine_id=alert.machine_id,
                        )

    async def _explain_one(self, alert, history) -> None:  # noqa: ANN001
        result = await self.explainer.explain(alert, history)
        if result is None:
            await self.db.mark_explanation_failed(alert.db_id)
            return

        explanation, latency_ms = result
        await self.db.update_explanation(
            alert.db_id, explanation, latency_ms, self.explainer.model
        )
        self._total_explained += 1

    async def _explain_loop(self) -> None:
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_EXPLANATIONS)

        async def _bounded(alert, history):  # noqa: ANN001
            async with semaphore:
                await self._explain_one(alert, history)

        while self._running:
            try:
                alert, history = await asyncio.wait_for(
                    self._alert_queue.get(), timeout=1.0
                )
                asyncio.create_task(_bounded(alert, history))
            except asyncio.TimeoutError:
                continue

    async def _stats_loop(self) -> None:
        while self._running:
            await asyncio.sleep(30)
            elapsed   = time.time() - self._t_start
            rate      = self._total_readings / max(elapsed, 1)
            det_rate  = (
                self._total_anomalies / self._total_readings * 100
                if self._total_readings else 0.0
            )
            log.info(
                "pipeline.stats",
                readings=self._total_readings,
                anomalies=self._total_anomalies,
                explained=self._total_explained,
                throughput_hz=round(rate, 1),
                detection_pct=round(det_rate, 2),
                active_streams=self.detector.active_streams,
            )

    # ── Entry point ───────────────────────────────────────────────────────────

    async def run(self) -> None:
        await self.db.connect()

        log.info("pipeline.checking_ollama")
        available = await self.explainer.ensure_model_available()
        if not available:
            log.warning(
                "pipeline.ollama_unavailable",
                note="Anomalies will be stored without LLM explanations",
            )

        consumer = Consumer({
            "bootstrap.servers":  KAFKA_BOOTSTRAP_SERVERS,
            "group.id":           KAFKA_GROUP_ID,
            "auto.offset.reset":  "latest",
            "enable.auto.commit": True,
            "fetch.min.bytes":    1,
            "fetch.wait.max.ms":  100,
        })
        consumer.subscribe([KAFKA_TOPIC])
        log.info("pipeline.started", topic=KAFKA_TOPIC)

        def _shutdown(sig, frame):  # noqa: ANN001
            log.info("pipeline.shutdown_signal")
            self._running = False

        signal.signal(signal.SIGINT,  _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

        try:
            await asyncio.gather(
                self._kafka_loop(consumer),
                self._explain_loop(),
                self._stats_loop(),
            )
        finally:
            consumer.close()
            await self.explainer.close()
            await self.db.close()
            self._executor.shutdown(wait=False)
            log.info(
                "pipeline.stopped",
                total_readings=self._total_readings,
                total_anomalies=self._total_anomalies,
                total_explained=self._total_explained,
            )


if __name__ == "__main__":
    import structlog as sl
    sl.configure(
        processors=[
            sl.stdlib.add_log_level,
            sl.dev.ConsoleRenderer(),
        ]
    )
    asyncio.run(Pipeline().run())
