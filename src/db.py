"""
TimescaleDB interface — async via asyncpg.
All writes are fire-and-forget friendly; call await db.close() on shutdown.
"""

from __future__ import annotations

import asyncpg
import structlog

from src.config import TSDB_DSN
from src.models import AnomalyAlert, LLMExplanation, SensorReading

log = structlog.get_logger(__name__)


class Database:
    def __init__(self) -> None:
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        self._pool = await asyncpg.create_pool(
            TSDB_DSN,
            min_size=2,
            max_size=10,
            command_timeout=30,
        )
        log.info("db.connected", dsn=TSDB_DSN.split("@")[-1])

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()

    @property
    def pool(self) -> asyncpg.Pool:
        if not self._pool:
            raise RuntimeError("Database.connect() not called")
        return self._pool

    # ── Sensor readings ───────────────────────────────────────────────────────

    async def insert_reading(self, r: SensorReading) -> None:
        await self.pool.execute(
            """
            INSERT INTO sensor_readings
                (time, machine_id, machine_type, signal, value, is_injected, anomaly_type)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
            r.time, r.machine_id, r.machine_type,
            r.signal, r.value, r.is_injected, r.anomaly_type,
        )

    async def insert_readings_batch(self, readings: list[SensorReading]) -> None:
        rows = [
            (r.time, r.machine_id, r.machine_type,
             r.signal, r.value, r.is_injected, r.anomaly_type)
            for r in readings
        ]
        await self.pool.executemany(
            """
            INSERT INTO sensor_readings
                (time, machine_id, machine_type, signal, value, is_injected, anomaly_type)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
            rows,
        )

    # ── Anomaly alerts ────────────────────────────────────────────────────────

    async def insert_alert(self, alert: AnomalyAlert) -> int:
        """Insert alert and return the generated DB id."""
        row = await self.pool.fetchrow(
            """
            INSERT INTO anomaly_alerts
                (time, machine_id, machine_type, signal, value,
                 z_score, window_mean, window_std, is_injected, anomaly_type,
                 explanation_status)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,'pending')
            RETURNING id
            """,
            alert.time, alert.machine_id, alert.machine_type,
            alert.signal, alert.value,
            alert.z_score, alert.window_mean, alert.window_std,
            alert.is_injected, alert.anomaly_type,
        )
        return row["id"]

    async def update_explanation(
        self,
        alert_id: int,
        explanation: LLMExplanation,
        latency_ms: int,
        model: str,
    ) -> None:
        await self.pool.execute(
            """
            UPDATE anomaly_alerts SET
                explanation_status = 'done',
                probable_cause     = $2,
                severity           = $3,
                recommended_action = $4,
                confidence         = $5,
                llm_latency_ms     = $6,
                llm_model          = $7
            WHERE id = $1
            """,
            alert_id,
            explanation.probable_cause,
            explanation.severity,
            explanation.recommended_action,
            explanation.confidence,
            latency_ms,
            model,
        )

    async def mark_explanation_failed(self, alert_id: int) -> None:
        await self.pool.execute(
            "UPDATE anomaly_alerts SET explanation_status='failed' WHERE id=$1",
            alert_id,
        )

    # ── Eval helpers ──────────────────────────────────────────────────────────

    async def fetch_explained_alerts(self, limit: int = 50) -> list[dict]:
        rows = await self.pool.fetch(
            """
            SELECT
                aa.id, aa.time, aa.machine_id, aa.machine_type, aa.signal,
                aa.value, aa.z_score, aa.window_mean, aa.window_std,
                aa.is_injected, aa.anomaly_type,
                aa.probable_cause, aa.severity,
                aa.recommended_action, aa.confidence,
                aa.llm_latency_ms, aa.llm_model
            FROM anomaly_alerts aa
            LEFT JOIN llm_ratings lr ON lr.alert_id = aa.id
            WHERE aa.explanation_status = 'done'
              AND lr.id IS NULL
            ORDER BY aa.time DESC
            LIMIT $1
            """,
            limit,
        )
        return [dict(r) for r in rows]

    async def insert_rating(
        self,
        alert_id: int,
        correctness: int,
        actionability: int,
        hallucination: bool,
        notes: str | None = None,
    ) -> None:
        await self.pool.execute(
            """
            INSERT INTO llm_ratings
                (alert_id, correctness, actionability, hallucination, notes)
            VALUES ($1, $2, $3, $4, $5)
            """,
            alert_id, correctness, actionability, hallucination, notes,
        )

    async def fetch_all_alerts_for_metrics(self) -> list[dict]:
        rows = await self.pool.fetch(
            """
            SELECT
                aa.id, aa.is_injected, aa.z_score,
                aa.severity, aa.confidence, aa.llm_latency_ms,
                lr.correctness, lr.actionability, lr.hallucination
            FROM anomaly_alerts aa
            LEFT JOIN llm_ratings lr ON lr.alert_id = aa.id
            WHERE aa.explanation_status IN ('done','failed')
            ORDER BY aa.time
            """
        )
        return [dict(r) for r in rows]

    async def fetch_total_injected(self) -> int:
        row = await self.pool.fetchrow(
            "SELECT COUNT(*) AS n FROM sensor_readings WHERE is_injected = TRUE"
        )
        return row["n"]

    async def fetch_latency_percentiles(self) -> dict:
        row = await self.pool.fetchrow(
            """
            SELECT
                PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY llm_latency_ms) AS p50,
                PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY llm_latency_ms) AS p95,
                PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY llm_latency_ms) AS p99
            FROM anomaly_alerts
            WHERE llm_latency_ms IS NOT NULL
            """
        )
        return dict(row)
