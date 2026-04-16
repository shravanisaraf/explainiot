-- ExplainIoT — TimescaleDB schema
-- Executed automatically on first container start via docker-entrypoint-initdb.d

CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ── Raw sensor readings ──────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS sensor_readings (
    id               BIGSERIAL,
    time             TIMESTAMPTZ        NOT NULL,
    machine_id       TEXT               NOT NULL,
    machine_type     TEXT               NOT NULL,
    signal           TEXT               NOT NULL,
    value            DOUBLE PRECISION   NOT NULL,
    is_injected      BOOLEAN            NOT NULL DEFAULT FALSE,
    anomaly_type     TEXT               NULL      -- spike | drift | burst | NULL
);

SELECT create_hypertable('sensor_readings', 'time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_sr_machine_signal
    ON sensor_readings (machine_id, signal, time DESC);

-- ── Anomaly alerts with LLM explanations ────────────────────────────────────
CREATE TABLE IF NOT EXISTS anomaly_alerts (
    id                 BIGSERIAL,
    time               TIMESTAMPTZ        NOT NULL,
    machine_id         TEXT               NOT NULL,
    machine_type       TEXT               NOT NULL,
    signal             TEXT               NOT NULL,
    value              DOUBLE PRECISION   NOT NULL,
    z_score            DOUBLE PRECISION   NOT NULL,
    window_mean        DOUBLE PRECISION   NOT NULL,
    window_std         DOUBLE PRECISION   NOT NULL,
    is_injected        BOOLEAN            NOT NULL,
    anomaly_type       TEXT               NULL,

    -- LLM explanation fields (populated asynchronously)
    explanation_status TEXT               NOT NULL DEFAULT 'pending',
    probable_cause     TEXT               NULL,
    severity           TEXT               NULL,
    recommended_action TEXT               NULL,
    confidence         DOUBLE PRECISION   NULL,
    llm_latency_ms     INTEGER            NULL,
    llm_model          TEXT               NULL,

    created_at         TIMESTAMPTZ        NOT NULL DEFAULT NOW()
);

SELECT create_hypertable('anomaly_alerts', 'time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_aa_machine
    ON anomaly_alerts (machine_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_aa_severity
    ON anomaly_alerts (severity, time DESC)
    WHERE severity IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_aa_status
    ON anomaly_alerts (explanation_status);

-- ── Continuous aggregate: anomaly counts per minute ─────────────────────────
CREATE MATERIALIZED VIEW IF NOT EXISTS anomaly_rate_per_minute
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 minute', time) AS bucket,
    machine_id,
    signal,
    COUNT(*)                       AS anomaly_count,
    AVG(z_score)                   AS avg_z_score,
    MAX(z_score)                   AS max_z_score
FROM anomaly_alerts
GROUP BY bucket, machine_id, signal
WITH NO DATA;

SELECT add_continuous_aggregate_policy(
    'anomaly_rate_per_minute',
    start_offset  => INTERVAL '10 minutes',
    end_offset    => INTERVAL '1 minute',
    schedule_interval => INTERVAL '1 minute',
    if_not_exists => TRUE
);

-- ── LLM evaluation ratings (populated by eval/rate.py) ──────────────────────
CREATE TABLE IF NOT EXISTS llm_ratings (
    id                 BIGSERIAL PRIMARY KEY,
    alert_id           BIGINT             NOT NULL,
    rated_at           TIMESTAMPTZ        NOT NULL DEFAULT NOW(),
    correctness        SMALLINT           NOT NULL CHECK (correctness BETWEEN 1 AND 5),
    actionability      SMALLINT           NOT NULL CHECK (actionability BETWEEN 1 AND 5),
    hallucination      BOOLEAN            NOT NULL,  -- TRUE = hallucination present
    notes              TEXT               NULL
);

CREATE INDEX IF NOT EXISTS idx_ratings_alert ON llm_ratings (alert_id);
