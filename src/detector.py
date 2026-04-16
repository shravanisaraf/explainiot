"""
Hybrid sliding-window anomaly detector: z-score + CUSUM.

Z-score (per-reading)
─────────────────────
Flags any reading whose deviation from the rolling mean exceeds ZSCORE_THRESHOLD.
Good for sudden spikes. Weak on slow drifts (window mean follows the drift).

CUSUM (cumulative sum)
──────────────────────
Accumulates normalised deviations over time. Triggers when the cumulative sum
exceeds threshold h, detecting sustained shifts that fall below the z-score
threshold individually but accumulate to a significant total.
Parameters: k=0.5 (slack = half the minimum detectable shift in σ units),
            h=4.0 (decision threshold, Basseville & Nikiforov standard value).

Hybrid label
────────────
detector_type = "zscore"   — only z-score fired
detector_type = "cusum"    — only CUSUM fired  (drift anomalies)
detector_type = "hybrid"   — both fired simultaneously

Persistent state
────────────────
Call detector.save_state(path) before shutdown and detector.load_state(path) on
startup to serialise rolling windows and CUSUM accumulators to JSON.
This eliminates the cold-start bias that depresses recall after pipeline restarts.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

from src.config import WARMUP_READINGS, WINDOW_SIZE, ZSCORE_THRESHOLD
from src.models import AnomalyAlert, SensorReading

# CUSUM parameters (Basseville & Nikiforov, "Detection of Abrupt Changes", 1993)
_CUSUM_K = 0.5   # slack — half the minimum detectable shift in σ units
_CUSUM_H = 4.0   # decision threshold

_DEFAULT_STATE_PATH = Path(".detector_state.json")


class SlidingWindowDetector:
    def __init__(
        self,
        window_size: int = WINDOW_SIZE,
        threshold: float = ZSCORE_THRESHOLD,
        warmup: int = WARMUP_READINGS,
        cusum_k: float = _CUSUM_K,
        cusum_h: float = _CUSUM_H,
    ) -> None:
        self.window_size = window_size
        self.threshold   = threshold
        self.warmup      = warmup
        self.cusum_k     = cusum_k
        self.cusum_h     = cusum_h

        # (machine_id, signal) → deque of float values
        self._windows: dict[tuple[str, str], deque[float]] = {}

        # CUSUM accumulators — both track positive magnitudes
        # s_up detects upward drift; s_dn detects downward drift
        self._cusum_up: dict[tuple[str, str], float] = {}
        self._cusum_dn: dict[tuple[str, str], float] = {}

    def _get_window(self, machine_id: str, signal: str) -> deque[float]:
        key = (machine_id, signal)
        if key not in self._windows:
            self._windows[key] = deque(maxlen=self.window_size)
        return self._windows[key]

    def process(self, reading: SensorReading) -> Optional[AnomalyAlert]:
        """
        Feed one reading into the detector.
        Returns an AnomalyAlert if z-score or CUSUM (or both) exceed their
        thresholds, otherwise returns None.
        """
        key    = (reading.machine_id, reading.signal)
        window = self._get_window(reading.machine_id, reading.signal)

        alert: Optional[AnomalyAlert] = None

        if len(window) >= self.warmup:
            arr  = np.asarray(window, dtype=np.float64)
            mean = float(arr.mean())
            std  = float(arr.std(ddof=1))

            if std > 1e-9:
                z = (reading.value - mean) / std

                # ── Z-score ───────────────────────────────────────────────────
                z_triggered = abs(z) >= self.threshold

                # ── CUSUM (one-sided upward + one-sided downward) ─────────────
                s_up = max(0.0, self._cusum_up.get(key, 0.0) + z        - self.cusum_k)
                s_dn = max(0.0, self._cusum_dn.get(key, 0.0) + (-z)     - self.cusum_k)
                cusum_triggered = s_up > self.cusum_h or s_dn > self.cusum_h
                cusum_score     = max(s_up, s_dn)

                # Reset CUSUM after trigger (standard CUSUM restart policy)
                if cusum_triggered:
                    self._cusum_up[key] = 0.0
                    self._cusum_dn[key] = 0.0
                else:
                    self._cusum_up[key] = s_up
                    self._cusum_dn[key] = s_dn

                if z_triggered or cusum_triggered:
                    if z_triggered and cusum_triggered:
                        detector_type = "hybrid"
                    elif z_triggered:
                        detector_type = "zscore"
                    else:
                        detector_type = "cusum"

                    alert = AnomalyAlert(
                        time          = reading.time,
                        machine_id    = reading.machine_id,
                        machine_type  = reading.machine_type,
                        signal        = reading.signal,
                        value         = reading.value,
                        z_score       = round(z, 4),
                        window_mean   = round(mean, 4),
                        window_std    = round(std, 4),
                        is_injected   = reading.is_injected,
                        anomaly_type  = reading.anomaly_type,
                        detector_type = detector_type,
                        cusum_score   = round(cusum_score, 4),
                    )
            else:
                # Zero-variance window — reset CUSUM
                self._cusum_up[key] = 0.0
                self._cusum_dn[key] = 0.0

        # Append AFTER computing statistics — current reading is future context
        window.append(reading.value)
        return alert

    def window_history(self, machine_id: str, signal: str) -> list[float]:
        """Return a copy of the current window (oldest → newest)."""
        return list(self._get_window(machine_id, signal))

    @property
    def active_streams(self) -> int:
        return len(self._windows)

    # ── Persistent state ──────────────────────────────────────────────────────

    def save_state(self, path: Path = _DEFAULT_STATE_PATH) -> None:
        """Serialise rolling windows and CUSUM accumulators to JSON."""
        state = {
            "windows":  {f"{k[0]}:{k[1]}": list(v) for k, v in self._windows.items()},
            "cusum_up": {f"{k[0]}:{k[1]}": v        for k, v in self._cusum_up.items()},
            "cusum_dn": {f"{k[0]}:{k[1]}": v        for k, v in self._cusum_dn.items()},
        }
        Path(path).write_text(json.dumps(state, indent=2))

    def load_state(self, path: Path = _DEFAULT_STATE_PATH) -> None:
        """Restore rolling windows and CUSUM accumulators from JSON."""
        p = Path(path)
        if not p.exists():
            return
        state = json.loads(p.read_text())
        for key_str, values in state.get("windows", {}).items():
            mid, sig = key_str.split(":", 1)
            self._windows[(mid, sig)] = deque(values, maxlen=self.window_size)
        for key_str, val in state.get("cusum_up", {}).items():
            mid, sig = key_str.split(":", 1)
            self._cusum_up[(mid, sig)] = float(val)
        for key_str, val in state.get("cusum_dn", {}).items():
            mid, sig = key_str.split(":", 1)
            self._cusum_dn[(mid, sig)] = float(val)
