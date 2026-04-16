"""
Kafka producer — reads from SensorGenerator and publishes to the
configured Kafka topic. Each message key is machine_id:signal, which
guarantees ordering within a stream and enables per-partition parallelism.

Run with:
    python -m src.producer
"""

from __future__ import annotations

import signal
import sys

import structlog
from confluent_kafka import Producer
from confluent_kafka.admin import AdminClient, NewTopic

from src.config import KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC
from src.sensor_generator import SensorGenerator

log = structlog.get_logger(__name__)


def _ensure_topic(bootstrap: str, topic: str, partitions: int = 10) -> None:
    admin = AdminClient({"bootstrap.servers": bootstrap})
    meta  = admin.list_topics(timeout=5)
    if topic not in meta.topics:
        fs = admin.create_topics([
            NewTopic(topic, num_partitions=partitions, replication_factor=1)
        ])
        for t, f in fs.items():
            try:
                f.result()
                log.info("producer.topic_created", topic=t, partitions=partitions)
            except Exception as exc:
                log.warning("producer.topic_exists_or_error", topic=t, error=str(exc))


def _delivery_report(err, msg) -> None:  # noqa: ANN001
    if err:
        log.error("producer.delivery_failed", error=str(err))


def run() -> None:
    _ensure_topic(KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC)

    producer = Producer({
        "bootstrap.servers":             KAFKA_BOOTSTRAP_SERVERS,
        "queue.buffering.max.messages":  100_000,
        "queue.buffering.max.kbytes":    65536,
        "linger.ms":                     10,
        "batch.num.messages":            500,
        "compression.type":              "lz4",
        "acks":                          "1",
    })

    generator  = SensorGenerator()
    total_sent = 0

    def _shutdown(sig, frame):  # noqa: ANN001
        log.info("producer.shutdown", total_sent=total_sent)
        producer.flush(timeout=10)
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    log.info("producer.started", topic=KAFKA_TOPIC, broker=KAFKA_BOOTSTRAP_SERVERS)

    for batch in generator.stream():
        for reading in batch:
            producer.produce(
                topic     = KAFKA_TOPIC,
                key       = f"{reading.machine_id}:{reading.signal}",
                value     = reading.to_kafka_bytes(),
                on_delivery = _delivery_report,
            )
        producer.poll(0)  # trigger delivery callbacks without blocking
        total_sent += len(batch)

        if total_sent % 300 == 0:
            log.info("producer.heartbeat", total_sent=total_sent)


if __name__ == "__main__":
    import structlog
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.dev.ConsoleRenderer(),
        ]
    )
    run()
