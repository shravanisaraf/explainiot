"""
Pydantic data models shared across the pipeline.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


class SensorReading(BaseModel):
    time:         datetime
    machine_id:   str
    machine_type: str
    signal:       str
    value:        float
    is_injected:  bool  = False
    anomaly_type: Optional[str] = None  # "spike" | "drift" | "burst" | None

    def to_kafka_bytes(self) -> bytes:
        return self.model_dump_json().encode("utf-8")

    @classmethod
    def from_kafka_bytes(cls, data: bytes) -> "SensorReading":
        return cls.model_validate_json(data.decode("utf-8"))


class AnomalyAlert(BaseModel):
    time:         datetime
    machine_id:   str
    machine_type: str
    signal:       str
    value:        float
    z_score:      float
    window_mean:  float
    window_std:   float
    is_injected:  bool
    anomaly_type: Optional[str] = None

    # populated after DB insert
    db_id: Optional[int] = None


class LLMExplanation(BaseModel):
    probable_cause:     str
    severity:           Literal["low", "medium", "high", "critical"]
    recommended_action: str
    confidence:         float = Field(ge=0.0, le=1.0)

    @classmethod
    def from_json_str(cls, raw: str) -> "LLMExplanation":
        """Parse LLM JSON output, tolerating minor formatting issues."""
        # Strip markdown code fences if present
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return cls.model_validate(json.loads(raw.strip()))
