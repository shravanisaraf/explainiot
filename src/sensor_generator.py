"""
Sensor data generator — simulates 10 industrial machines emitting
temperature, pressure, and vibration readings at 1 Hz each.

Anomaly injection (3 % rate) produces three distinct fault signatures:
  spike  — single extreme deviation (|z| ~ 4–7)
  drift  — sustained gradual shift over 5–12 consecutive readings
  burst  — 2–4 rapid successive excursions

Each injected reading carries is_injected=True and the anomaly_type string,
providing ground truth for precision/recall evaluation.
"""

from __future__ import annotations

import random
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Iterator

import numpy as np

from src.config import ANOMALY_RATE, MACHINES, RANDOM_SEED
from src.models import SensorReading


class SensorGenerator:
    def __init__(self, seed: int = RANDOM_SEED) -> None:
        self._rng = np.random.default_rng(seed)
        self._py_rng = random.Random(seed)

        # Drift state: machine_id → {signal → remaining_drift_readings, drift_magnitude}
        self._drift_state: dict[str, dict[str, dict]] = defaultdict(dict)

        # Burst state: machine_id → {signal → remaining_burst_readings}
        self._burst_state: dict[str, dict[str, int]] = defaultdict(dict)

    def _normal_reading(self, machine_id: str, signal: str) -> float:
        params = MACHINES[machine_id]["signals"][signal]
        return float(self._rng.normal(params["mean"], params["std"]))

    def _inject_anomaly(
        self, machine_id: str, signal: str, anomaly_type: str
    ) -> float:
        params = MACHINES[machine_id]["signals"][signal]
        mean, std = params["mean"], params["std"]

        if anomaly_type == "spike":
            direction = self._py_rng.choice([-1, 1])
            magnitude = self._rng.uniform(4.0, 7.0)
            return float(mean + direction * magnitude * std)

        if anomaly_type == "drift":
            magnitude = float(self._rng.uniform(2.8, 5.0))
            direction = self._py_rng.choice([-1, 1])
            duration  = int(self._rng.integers(5, 13))
            self._drift_state[machine_id][signal] = {
                "remaining": duration,
                "magnitude": direction * magnitude,
            }
            return float(mean + direction * magnitude * std)

        if anomaly_type == "burst":
            duration = int(self._rng.integers(2, 5))
            self._burst_state[machine_id][signal] = duration
            direction = self._py_rng.choice([-1, 1])
            magnitude = self._rng.uniform(3.0, 5.5)
            return float(mean + direction * magnitude * std)

        return self._normal_reading(machine_id, signal)

    def _reading_for(
        self, machine_id: str, signal: str, ts: datetime
    ) -> SensorReading:
        machine_type = MACHINES[machine_id]["type"]
        is_injected  = False
        anomaly_type: str | None = None
        value: float

        # ── Continue ongoing drift ────────────────────────────────────────────
        drift = self._drift_state[machine_id].get(signal)
        if drift and drift["remaining"] > 0:
            params = MACHINES[machine_id]["signals"][signal]
            value = float(
                self._rng.normal(
                    params["mean"] + drift["magnitude"] * params["std"],
                    params["std"] * 0.3,
                )
            )
            drift["remaining"] -= 1
            is_injected  = True
            anomaly_type = "drift"
            return SensorReading(
                time=ts, machine_id=machine_id, machine_type=machine_type,
                signal=signal, value=value,
                is_injected=is_injected, anomaly_type=anomaly_type,
            )

        # ── Continue ongoing burst ────────────────────────────────────────────
        burst_rem = self._burst_state[machine_id].get(signal, 0)
        if burst_rem > 0:
            self._burst_state[machine_id][signal] = burst_rem - 1
            params = MACHINES[machine_id]["signals"][signal]
            direction = self._py_rng.choice([-1, 1])
            magnitude = self._rng.uniform(3.0, 5.5)
            value = float(params["mean"] + direction * magnitude * params["std"])
            return SensorReading(
                time=ts, machine_id=machine_id, machine_type=machine_type,
                signal=signal, value=value,
                is_injected=True, anomaly_type="burst",
            )

        # ── Probabilistic injection of new anomaly ────────────────────────────
        if self._rng.random() < ANOMALY_RATE:
            anomaly_type = self._py_rng.choice(["spike", "drift", "burst"])
            value        = self._inject_anomaly(machine_id, signal, anomaly_type)
            is_injected  = True
        else:
            value = self._normal_reading(machine_id, signal)

        return SensorReading(
            time=ts, machine_id=machine_id, machine_type=machine_type,
            signal=signal, value=value,
            is_injected=is_injected, anomaly_type=anomaly_type,
        )

    def stream(self) -> Iterator[list[SensorReading]]:
        """
        Yields a batch of readings (one per machine per signal) every second.
        Total throughput: 10 machines × 3 signals = 30 readings/s.
        Runs indefinitely; caller handles the loop.
        """
        while True:
            ts = datetime.now(timezone.utc)
            batch: list[SensorReading] = []
            for machine_id in MACHINES:
                for signal in MACHINES[machine_id]["signals"]:
                    batch.append(self._reading_for(machine_id, signal, ts))
            yield batch
            time.sleep(1.0)

    def single_batch(self) -> list[SensorReading]:
        """Return one batch without sleeping — useful for tests."""
        ts = datetime.now(timezone.utc)
        return [
            self._reading_for(machine_id, signal, ts)
            for machine_id in MACHINES
            for signal in MACHINES[machine_id]["signals"]
        ]
