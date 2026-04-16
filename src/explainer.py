"""
Async LLM explainer — calls Ollama with a structured prompt and forces
JSON output via Ollama's `format: "json"` parameter.

The prompt is engineered to elicit a four-field diagnostic object:
  probable_cause     — physics-grounded explanation
  severity           — low | medium | high | critical
  recommended_action — concrete operator instruction
  confidence         — 0.0–1.0

Low temperature (0.1) ensures consistent, deterministic JSON structure.
The async interface means LLM calls never block the Kafka consumer loop.
"""

from __future__ import annotations

import time
from typing import Optional

import httpx
import structlog

from src.config import MACHINES, OLLAMA_BASE_URL, OLLAMA_MODEL
from src.models import AnomalyAlert, LLMExplanation

log = structlog.get_logger(__name__)

# ── Prompt templates ─────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a senior industrial systems engineer and predictive maintenance specialist
with deep expertise in hydraulic systems, electric motors, air compressors,
conveyor drives, and CNC machine tools.

When given sensor anomaly data, you diagnose the most probable physical cause,
assess severity, and recommend an immediate corrective action for the plant operator.

You MUST respond with a valid JSON object containing EXACTLY these four fields:
  "probable_cause"     – one to two sentences describing the physical mechanism
  "severity"           – exactly one of: low, medium, high, critical
  "recommended_action" – one to two sentences of concrete operator instructions
  "confidence"         – a float in [0.0, 1.0] reflecting your diagnostic certainty

Return ONLY the JSON object. No preamble, no explanation, no markdown fencing.\
"""


def _build_user_prompt(
    alert: AnomalyAlert,
    recent_values: list[float],
) -> str:
    params     = MACHINES[alert.machine_id]["signals"][alert.signal]
    unit       = params["unit"]
    normal_lo  = round(params["mean"] - 2 * params["std"], 3)
    normal_hi  = round(params["mean"] + 2 * params["std"], 3)
    recent_str = ", ".join(f"{v:.3f}" for v in recent_values[-10:])

    # Direction of deviation — critical for accurate LLM diagnosis
    direction = "above" if alert.value > alert.window_mean else "below"
    z_abs     = abs(alert.z_score)
    detector  = (alert.detector_type or "zscore").upper()

    return f"""\
=== SENSOR ANOMALY REPORT ===

Machine ID   : {alert.machine_id}
Machine type : {alert.machine_type.replace("_", " ").title()}
Signal       : {alert.signal}  [{unit}]
Timestamp    : {alert.time.isoformat()}

Anomalous reading : {alert.value:.3f} {unit}
Normal range (±2σ): {normal_lo} – {normal_hi} {unit}
Rolling mean      : {alert.window_mean:.3f} {unit}
Rolling std dev   : {alert.window_std:.3f} {unit}
Direction         : {z_abs:.2f}σ {direction} the rolling mean
Z-score           : {alert.z_score:.2f}  (threshold: ±2.5)
Detector          : {detector}  (ZSCORE=single spike, CUSUM=persistent drift, HYBRID=both)

Recent readings (oldest → newest, last 10 values):
  [{recent_str}]

IMPORTANT: Direction is "{direction.upper()}" — this reading is a {direction} the normal range.
If "below", diagnose as a drop/loss (e.g. pressure loss, temperature drop) not a rise.

Provide your diagnosis as a JSON object with the four required fields.\
"""


# ── Explainer class ───────────────────────────────────────────────────────────

class LLMExplainer:
    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = OLLAMA_MODEL,
        timeout: float = 300.0,
    ) -> None:
        self.model   = model
        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout)

    async def close(self) -> None:
        await self._client.aclose()

    async def explain(
        self,
        alert: AnomalyAlert,
        recent_values: list[float],
    ) -> Optional[tuple[LLMExplanation, int]]:
        """
        Call the LLM and return a structured explanation.
        Returns None on failure so the caller can mark the alert as failed
        without crashing the pipeline.
        """
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": _build_user_prompt(alert, recent_values)},
        ]

        t0 = time.perf_counter()
        try:
            response = await self._client.post(
                "/api/chat",
                json={
                    "model":   self.model,
                    "messages": messages,
                    "format":  "json",
                    "stream":  False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 512,
                        "seed":        42,
                    },
                },
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            log.error(
                "explainer.http_error",
                machine_id=alert.machine_id,
                signal=alert.signal,
                error=str(exc),
            )
            return None

        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        raw_content: str = response.json()["message"]["content"]

        try:
            explanation = LLMExplanation.from_json_str(raw_content)
        except Exception as exc:
            log.error(
                "explainer.parse_error",
                machine_id=alert.machine_id,
                signal=alert.signal,
                raw=raw_content[:200],
                error=str(exc),
            )
            return None

        log.info(
            "explainer.done",
            machine_id=alert.machine_id,
            signal=alert.signal,
            severity=explanation.severity,
            confidence=round(explanation.confidence, 2),
            latency_ms=elapsed_ms,
        )
        return explanation, elapsed_ms

    async def ensure_model_available(self) -> bool:
        """Pull the model if it is not already cached in Ollama."""
        try:
            resp = await self._client.get("/api/tags")
            resp.raise_for_status()
            models = [m["name"].split(":")[0] for m in resp.json().get("models", [])]
            if self.model.split(":")[0] not in models:
                log.info("explainer.pulling_model", model=self.model)
                pull_resp = await self._client.post(
                    "/api/pull",
                    json={"name": self.model, "stream": False},
                    timeout=600.0,
                )
                pull_resp.raise_for_status()
                log.info("explainer.model_ready", model=self.model)
            return True
        except httpx.HTTPError as exc:
            log.error("explainer.ollama_unavailable", error=str(exc))
            return False
