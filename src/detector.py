"""
Sliding-window z-score anomaly detector.

Maintains a rolling buffer of WINDOW_SIZE readings per (machine_id, signal)
pair. Once the buffer has WARMUP_READINGS entries, any reading whose z-score
exceeds ZSCORE_THRESHOLD is flagged as anomalous.

Design notes:
  - Uses collections.deque for O(1) append/pop with a fixed maxlen.
  - Mean and std are recomputed from the window using numpy for accuracy.
  - The current reading is NOT included in the window statistics — this
    prevents self-normalisation and matches the standard streaming detector
    formulation used in the ICLR 2024 benchmark literature.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from src.config import WARMUP_READINGS, WINDOW_SIZE, ZSCORE_THRESHOLD
from src.models import AnomalyAlert, SensorReading


class SlidingWindowDetector:
    def __init__(
        self,
        window_size: int = WINDOW_SIZE,
        threshold: float = ZSCORE_THRESHOLD,
        warmup: int = WARMUP_READINGS,
    ) -> None:
        self.window_size = window_size
        self.threshold   = threshold
        self.warmup      = warmup

        # (machine_id, signal) → deque of float values
        self._windows: dict[tuple[str, str], deque[float]] = {}

    def _get_window(self, machine_id: str, signal: str) -> deque[float]:
        key = (machine_id, signal)
        if key not in self._windows:
            self._windows[key] = deque(maxlen=self.window_size)
        return self._windows[key]

    def process(self, reading: SensorReading) -> Optional[AnomalyAlert]:
        """
        Feed one reading into the detector.
        Returns an AnomalyAlert if the z-score exceeds the threshold,
        otherwise returns None.
        """
        window = self._get_window(reading.machine_id, reading.signal)

        alert: Optional[AnomalyAlert] = None

        if len(window) >= self.warmup:
            arr  = np.asarray(window, dtype=np.float64)
            mean = float(arr.mean())
            std  = float(arr.std(ddof=1))

            if std > 1e-9:  # guard against zero-variance window
                z = (reading.value - mean) / std
                if abs(z) >= self.threshold:
                    alert = AnomalyAlert(
                        time         = reading.time,
                        machine_id   = reading.machine_id,
                        machine_type = reading.machine_type,
                        signal       = reading.signal,
                        value        = reading.value,
                        z_score      = round(z, 4),
                        window_mean  = round(mean, 4),
                        window_std   = round(std, 4),
                        is_injected  = reading.is_injected,
                        anomaly_type = reading.anomaly_type,
                    )

        # Append AFTER computing statistics — current reading is future context
        window.append(reading.value)
        return alert

    def window_history(self, machine_id: str, signal: str) -> list[float]:
        """Return a copy of the current window (oldest → newest)."""
        return list(self._get_window(machine_id, signal))

    @property
    def active_streams(self) -> int:
        return len(self._windows)
