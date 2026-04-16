"""
Central configuration — loaded once at import time.
All values are overridable via environment variables or a .env file.
"""

import os
from dotenv import load_dotenv

load_dotenv()


def _get(key: str, default: str) -> str:
    return os.environ.get(key, default)


# ── Kafka ────────────────────────────────────────────────────────────────────
KAFKA_BOOTSTRAP_SERVERS: str = _get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC: str             = _get("KAFKA_TOPIC", "sensor_readings")
KAFKA_GROUP_ID: str          = _get("KAFKA_GROUP_ID", "trace_consumer")

# ── TimescaleDB ──────────────────────────────────────────────────────────────
TSDB_DSN: str = _get(
    "TSDB_DSN",
    "postgresql://trace:trace@localhost:5432/trace",
)

# ── Ollama ───────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL: str = _get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str    = _get("OLLAMA_MODEL", "mistral")

# ── Sensor generator ─────────────────────────────────────────────────────────
READINGS_PER_SECOND: int = int(_get("READINGS_PER_SECOND", "10"))
ANOMALY_RATE: float      = float(_get("ANOMALY_RATE", "0.03"))
RANDOM_SEED: int         = int(_get("RANDOM_SEED", "42"))

# ── Detector ─────────────────────────────────────────────────────────────────
WINDOW_SIZE: int       = int(_get("WINDOW_SIZE", "60"))
ZSCORE_THRESHOLD: float = float(_get("ZSCORE_THRESHOLD", "2.5"))
WARMUP_READINGS: int   = int(_get("WARMUP_READINGS", "10"))

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL: str = _get("LOG_LEVEL", "INFO")

# ── Machine definitions ──────────────────────────────────────────────────────
# Each machine has distinct normal operating parameters — (mean, std) per signal
MACHINES: dict[str, dict] = {
    "machine_001": {
        "type": "hydraulic_pump",
        "signals": {
            "temperature": {"mean": 72.0, "std": 3.5, "unit": "°C"},
            "pressure":    {"mean": 4.8,  "std": 0.20, "unit": "bar"},
            "vibration":   {"mean": 1.8,  "std": 0.25, "unit": "mm/s"},
        },
    },
    "machine_002": {
        "type": "electric_motor",
        "signals": {
            "temperature": {"mean": 68.0, "std": 3.0, "unit": "°C"},
            "pressure":    {"mean": 3.2,  "std": 0.25, "unit": "bar"},
            "vibration":   {"mean": 1.2,  "std": 0.18, "unit": "mm/s"},
        },
    },
    "machine_003": {
        "type": "air_compressor",
        "signals": {
            "temperature": {"mean": 88.0, "std": 5.0,  "unit": "°C"},
            "pressure":    {"mean": 10.5, "std": 0.70, "unit": "bar"},
            "vibration":   {"mean": 2.8,  "std": 0.38, "unit": "mm/s"},
        },
    },
    "machine_004": {
        "type": "conveyor_drive",
        "signals": {
            "temperature": {"mean": 45.0, "std": 2.5, "unit": "°C"},
            "pressure":    {"mean": 2.1,  "std": 0.18, "unit": "bar"},
            "vibration":   {"mean": 0.9,  "std": 0.15, "unit": "mm/s"},
        },
    },
    "machine_005": {
        "type": "cnc_spindle",
        "signals": {
            "temperature": {"mean": 58.0, "std": 4.0,  "unit": "°C"},
            "pressure":    {"mean": 6.0,  "std": 0.40, "unit": "bar"},
            "vibration":   {"mean": 3.2,  "std": 0.45, "unit": "mm/s"},
        },
    },
    "machine_006": {
        "type": "hydraulic_pump",
        "signals": {
            "temperature": {"mean": 74.0, "std": 3.5, "unit": "°C"},
            "pressure":    {"mean": 4.6,  "std": 0.20, "unit": "bar"},
            "vibration":   {"mean": 1.9,  "std": 0.25, "unit": "mm/s"},
        },
    },
    "machine_007": {
        "type": "electric_motor",
        "signals": {
            "temperature": {"mean": 65.0, "std": 3.0, "unit": "°C"},
            "pressure":    {"mean": 3.5,  "std": 0.25, "unit": "bar"},
            "vibration":   {"mean": 1.3,  "std": 0.18, "unit": "mm/s"},
        },
    },
    "machine_008": {
        "type": "air_compressor",
        "signals": {
            "temperature": {"mean": 85.0, "std": 5.0,  "unit": "°C"},
            "pressure":    {"mean": 11.0, "std": 0.70, "unit": "bar"},
            "vibration":   {"mean": 2.6,  "std": 0.38, "unit": "mm/s"},
        },
    },
    "machine_009": {
        "type": "conveyor_drive",
        "signals": {
            "temperature": {"mean": 48.0, "std": 2.5, "unit": "°C"},
            "pressure":    {"mean": 2.3,  "std": 0.18, "unit": "bar"},
            "vibration":   {"mean": 1.0,  "std": 0.15, "unit": "mm/s"},
        },
    },
    "machine_010": {
        "type": "cnc_spindle",
        "signals": {
            "temperature": {"mean": 60.0, "std": 4.0,  "unit": "°C"},
            "pressure":    {"mean": 5.8,  "std": 0.40, "unit": "bar"},
            "vibration":   {"mean": 3.0,  "std": 0.45, "unit": "mm/s"},
        },
    },
}
