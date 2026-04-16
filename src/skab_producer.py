"""
SKAB (Skoltech Anomaly Benchmark) producer — replays real industrial sensor
data through the same Kafka pipeline as the simulated producer.

This replaces the synthetic Gaussian generator with ground-truth labelled
data from a physical water pump test bench, removing the biggest criticism
of simulated-data-only evaluation.

SKAB dataset
────────────
  Source : https://github.com/waico/SKAB
  Format : CSV per scenario, columns:
             datetime, Accelerometer1RMS, Accelerometer2RMS, Current,
             Pressure, Temperature, Thermocouple, Voltage,
             Volume Flow RateRMS, anomaly, changepoint

Signal mapping (3 chosen to match the simulated pipeline schema):
  Temperature       →  signal: "temperature"  (°C)
  Pressure          →  signal: "pressure"     (bar)
  Accelerometer1RMS →  signal: "vibration"    (mm/s)

Setup
─────
  1. Clone the SKAB repo:
       git clone https://github.com/waico/SKAB.git data/skab_repo
  2. Point SKAB_DATA_DIR at the data directory inside it:
       export SKAB_DATA_DIR=data/skab_repo/data
  3. Run:
       python -m src.skab_producer

  Or use run.sh:
       ./run.sh skab

Replay speed
────────────
  SKAB_REPLAY_SPEED=1.0  — real-time (1 second per row)
  SKAB_REPLAY_SPEED=10.0 — 10× faster (useful for quick eval runs)

The producer loops through all CSV files found in SKAB_DATA_DIR/**/*.csv
(sorted), so a single run covers all 34 scenarios.
"""

from __future__ import annotations

import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import structlog
from confluent_kafka import Producer
from confluent_kafka.admin import AdminClient, NewTopic

from src.config import KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC
from src.models import SensorReading

log = structlog.get_logger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
SKAB_DATA_DIR   = Path(os.environ.get("SKAB_DATA_DIR", "data/skab_repo/data"))
REPLAY_SPEED    = float(os.environ.get("SKAB_REPLAY_SPEED", "10.0"))
MACHINE_ID      = "skab_pump"
MACHINE_TYPE    = "water_pump"

# SKAB column → our signal name + unit
SIGNAL_MAP = {
    "Temperature":       ("temperature", "°C"),
    "Pressure":          ("pressure",    "bar"),
    "Accelerometer1RMS": ("vibration",   "mm/s"),
}


# ── Helpers ───────────────────────────────────────────────────────────────────

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
                log.info("producer.topic_created", topic=t)
            except Exception as exc:
                log.warning("producer.topic_error", topic=t, error=str(exc))


def _delivery_report(err, msg) -> None:  # noqa: ANN001
    if err:
        log.error("producer.delivery_failed", error=str(err))


def _load_skab_csvs(data_dir: Path) -> pd.DataFrame:
    """Load and concatenate all SKAB CSV files, sorted by path."""
    csvs = sorted(data_dir.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError(
            f"No CSV files found in {data_dir}.\n"
            "Run:  git clone https://github.com/waico/SKAB.git data/skab_repo\n"
            "Then: export SKAB_DATA_DIR=data/skab_repo/data"
        )

    frames = []
    for csv_path in csvs:
        try:
            df = pd.read_csv(csv_path, sep=";", parse_dates=["datetime"])
            df["_source"] = csv_path.stem
            frames.append(df)
            log.info("skab.loaded", file=str(csv_path.relative_to(data_dir)), rows=len(df))
        except Exception as exc:
            log.warning("skab.load_error", file=str(csv_path), error=str(exc))

    combined = pd.concat(frames, ignore_index=True)
    log.info("skab.total_rows", rows=len(combined))
    return combined


# ── Main ──────────────────────────────────────────────────────────────────────

def run() -> None:
    if not SKAB_DATA_DIR.exists():
        print(
            f"\nSKAB data directory not found: {SKAB_DATA_DIR}\n\n"
            "Download the dataset first:\n"
            "  git clone https://github.com/waico/SKAB.git data/skab_repo\n"
            "  export SKAB_DATA_DIR=data/skab_repo/data\n"
        )
        sys.exit(1)

    _ensure_topic(KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC)

    producer = Producer({
        "bootstrap.servers":            KAFKA_BOOTSTRAP_SERVERS,
        "queue.buffering.max.messages": 100_000,
        "linger.ms":                    10,
        "compression.type":             "lz4",
        "acks":                         "1",
    })

    df = _load_skab_csvs(SKAB_DATA_DIR)

    # Ensure required columns exist
    for col in [*SIGNAL_MAP.keys(), "anomaly"]:
        if col not in df.columns:
            raise KeyError(
                f"Column '{col}' missing from SKAB data. "
                "Check SKAB_DATA_DIR points to the correct directory."
            )

    total_sent = 0
    sleep_s    = 1.0 / REPLAY_SPEED  # delay between rows

    def _shutdown(sig, frame):  # noqa: ANN001
        log.info("skab_producer.shutdown", total_sent=total_sent)
        producer.flush(timeout=10)
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    log.info(
        "skab_producer.started",
        rows=len(df),
        replay_speed=REPLAY_SPEED,
        topic=KAFKA_TOPIC,
    )

    for _, row in df.iterrows():
        is_injected = bool(row.get("anomaly", 0))
        ts = datetime.now(timezone.utc)  # use wall-clock time for Grafana

        for skab_col, (signal_name, _unit) in SIGNAL_MAP.items():
            try:
                value = float(row[skab_col])
            except (ValueError, TypeError):
                continue  # skip NaN rows

            reading = SensorReading(
                time         = ts,
                machine_id   = MACHINE_ID,
                machine_type = MACHINE_TYPE,
                signal       = signal_name,
                value        = value,
                is_injected  = is_injected,
                anomaly_type = "skab_anomaly" if is_injected else None,
            )
            producer.produce(
                topic       = KAFKA_TOPIC,
                key         = f"{MACHINE_ID}:{signal_name}",
                value       = reading.to_kafka_bytes(),
                on_delivery = _delivery_report,
            )

        producer.poll(0)
        total_sent += len(SIGNAL_MAP)

        if total_sent % 300 == 0:
            log.info("skab_producer.heartbeat", total_sent=total_sent)

        time.sleep(sleep_s)

    producer.flush(timeout=10)
    log.info("skab_producer.done", total_sent=total_sent)


if __name__ == "__main__":
    import structlog
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.dev.ConsoleRenderer(),
        ]
    )
    run()
