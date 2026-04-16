#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# TRACE — single-command launcher
# Starts all infrastructure, waits for readiness, then runs the pipeline.
#
# Usage:
#   ./run.sh           — full stack (simulated producer + consumer)
#   ./run.sh producer  — simulated producer only
#   ./run.sh consumer  — consumer only
#   ./run.sh skab      — SKAB real-dataset producer + consumer
#   ./run.sh down      — stop all containers
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

MODE="${1:-all}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[trace]${NC} $*"; }
warn()  { echo -e "${YELLOW}[trace]${NC} $*"; }
error() { echo -e "${RED}[trace]${NC} $*" >&2; }

# ── Env ───────────────────────────────────────────────────────────────────────
if [[ ! -f .env ]]; then
    cp .env.example .env
    info "Created .env from .env.example — edit to override defaults."
fi
# shellcheck disable=SC1091
source .env

# ── Down ──────────────────────────────────────────────────────────────────────
if [[ "$MODE" == "down" ]]; then
    docker compose down
    info "All containers stopped."
    exit 0
fi

# ── Prerequisites ─────────────────────────────────────────────────────────────
for cmd in docker python3; do
    if ! command -v "$cmd" &>/dev/null; then
        error "Required command not found: $cmd"
        exit 1
    fi
done

PYTHON="python3"
if command -v conda &>/dev/null; then
    PYTHON="python"
fi

# ── Python dependencies ───────────────────────────────────────────────────────
info "Checking Python dependencies..."
$PYTHON -c "import confluent_kafka, asyncpg, httpx, pydantic, numpy" 2>/dev/null || {
    warn "Installing Python dependencies..."
    pip install -r requirements.txt -q
}

# ── Docker infrastructure ─────────────────────────────────────────────────────
info "Starting Docker services..."
docker compose up -d redpanda timescaledb grafana

# ── Wait for TimescaleDB ──────────────────────────────────────────────────────
info "Waiting for TimescaleDB..."
for i in $(seq 1 30); do
    if docker exec trace_tsdb pg_isready -U trace -d trace -q 2>/dev/null; then
        info "TimescaleDB ready."
        break
    fi
    [[ $i -eq 30 ]] && { error "TimescaleDB failed to start."; exit 1; }
    sleep 2
done

# ── Wait for Redpanda ─────────────────────────────────────────────────────────
info "Waiting for Redpanda..."
for i in $(seq 1 30); do
    if docker exec trace_redpanda rpk cluster health 2>/dev/null | grep -q "Healthy:.*true"; then
        info "Redpanda ready."
        break
    fi
    [[ $i -eq 30 ]] && { error "Redpanda failed to start."; exit 1; }
    sleep 2
done

# ── Check native Ollama ───────────────────────────────────────────────────────
if ! pgrep -x ollama &>/dev/null; then
    warn "Ollama not running. Start it with: brew services start ollama"
    warn "Then re-run ./run.sh"
    exit 1
fi

# ── Print URLs ────────────────────────────────────────────────────────────────
echo ""
info "Infrastructure ready."
echo "  Grafana        →  http://localhost:3000   (admin / trace)"
echo "  Redpanda UI    →  http://localhost:8080"
echo "  TimescaleDB    →  localhost:5432  (psql -U trace -d trace)"
echo ""

if [[ "$MODE" == "all" || "$MODE" == "producer" ]]; then
    info "Starting simulated producer..."
    $PYTHON -m src.producer &
    PRODUCER_PID=$!
fi

if [[ "$MODE" == "skab" ]]; then
    info "Starting SKAB real-dataset producer (replay speed: ${SKAB_REPLAY_SPEED:-10}×)..."
    $PYTHON -m src.skab_producer &
    PRODUCER_PID=$!
fi

if [[ "$MODE" == "all" || "$MODE" == "consumer" || "$MODE" == "skab" ]]; then
    info "Starting consumer (Kafka → hybrid detector → LLM → TimescaleDB)..."
    $PYTHON -m src.consumer &
    CONSUMER_PID=$!
fi

RUNTIME="${RUNTIME_MINUTES:-0}"
if [[ "$RUNTIME" -gt 0 ]]; then
    info "Pipeline will auto-stop in ${RUNTIME} minutes."
    (sleep $(( RUNTIME * 60 )); kill $$) &
fi

info "Pipeline running. Press Ctrl+C to stop."

cleanup() {
    info "Shutting down..."
    [[ -n "${PRODUCER_PID:-}" ]] && kill "$PRODUCER_PID" 2>/dev/null || true
    [[ -n "${CONSUMER_PID:-}" ]] && kill "$CONSUMER_PID" 2>/dev/null || true
    exit 0
}
trap cleanup INT TERM

wait
