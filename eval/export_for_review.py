"""
Export the 50 rated alerts to a plain-text file a second evaluator can read
without any database access.

Usage:
    python -m eval.export_for_review            # writes eval/review_sheet.txt
    python -m eval.export_for_review --csv      # also writes eval/review_sheet.csv

The second evaluator opens review_sheet.txt, fills in their scores on the
score sheet at the bottom, and sends it back. Then run:

    python -m eval.cohens_kappa  --their-scores "3 4 n, 2 3 n, ..."

or point it at their filled-in score sheet file.
"""

from __future__ import annotations

import asyncio
import csv
import sys
from pathlib import Path

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__file__)))
from src.db import Database

OUT_TXT = Path(__file__).parent / "review_sheet.txt"
OUT_CSV = Path(__file__).parent / "review_sheet.csv"

RUBRIC = """
SCORING RUBRIC
──────────────
Correctness (1–5)
  1 = Completely wrong — wrong signal, wrong direction, or impossible physics
  2 = Mostly wrong — misidentifies the mechanism
  3 = Plausible — reasonable guess given the data
  4 = Mostly correct — right mechanism, minor inaccuracies
  5 = Physically accurate — correct diagnosis with correct direction

Actionability (1–5)
  1 = Vague or unhelpful (e.g. "check the machine")
  2 = Somewhat useful but generic
  3 = Reasonable — correct type of action, not very specific
  4 = Specific and actionable
  5 = Specific, correctly targeted, and immediately executable

Hallucination (y/n)
  y = Explanation contains fabricated facts or physically impossible statements
      e.g. mentions fuel in an electric motor, cites non-existent components
  n = No hallucinations detected
"""


async def _fetch(db: Database) -> list[dict]:
    rows = await db.pool.fetch(
        """
        SELECT
            aa.id, aa.machine_id, aa.machine_type, aa.signal,
            aa.value, aa.z_score, aa.window_mean, aa.window_std,
            aa.is_injected, aa.anomaly_type, aa.detector_type,
            aa.probable_cause, aa.severity, aa.recommended_action,
            aa.confidence,
            lr.correctness, lr.actionability, lr.hallucination
        FROM anomaly_alerts aa
        JOIN llm_ratings lr ON lr.alert_id = aa.id
        ORDER BY lr.rated_at DESC
        LIMIT 50
        """
    )
    return [dict(r) for r in rows]


def _write_txt(alerts: list[dict]) -> None:
    lines = []
    lines.append("TRACE — LLM Explanation Quality Review")
    lines.append("=" * 60)
    lines.append(RUBRIC)
    lines.append("=" * 60)
    lines.append("ALERTS\n")

    for i, r in enumerate(alerts, 1):
        direction = "ABOVE" if r["value"] > r["window_mean"] else "BELOW"
        z_abs = abs(r["z_score"])
        lines.append(f"── Alert {i:02d} (id={r['id']}) ──────────────────────────")
        lines.append(f"  Machine     : {r['machine_id']}  ({r['machine_type']})")
        lines.append(f"  Signal      : {r['signal']}")
        lines.append(f"  Reading     : {r['value']:.3f}  →  {z_abs:.2f}σ {direction} rolling mean")
        lines.append(f"  Severity    : {r['severity'] or '?'}")
        lines.append(f"  Detector    : {r['detector_type'] or 'zscore'}")
        lines.append(f"  Cause       : {r['probable_cause'] or '–'}")
        lines.append(f"  Action      : {r['recommended_action'] or '–'}")
        lines.append("")

    lines.append("=" * 60)
    lines.append("SCORE SHEET  — fill in one line per alert: C  A  H")
    lines.append("Format: <correctness 1-5>  <actionability 1-5>  <hallucination y/n>")
    lines.append("Example:  3  4  n")
    lines.append("")
    for i in range(1, len(alerts) + 1):
        lines.append(f"  Alert {i:02d}: _  _  _")

    OUT_TXT.write_text("\n".join(lines))
    print(f"Written: {OUT_TXT}")


def _write_csv(alerts: list[dict]) -> None:
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "alert_num", "alert_id", "machine_id", "machine_type", "signal",
            "value", "z_score", "direction", "severity", "detector_type",
            "probable_cause", "recommended_action",
            "correctness_rater1", "actionability_rater1", "hallucination_rater1",
            "correctness_rater2", "actionability_rater2", "hallucination_rater2",
        ])
        for i, r in enumerate(alerts, 1):
            direction = "above" if r["value"] > r["window_mean"] else "below"
            writer.writerow([
                i, r["id"], r["machine_id"], r["machine_type"], r["signal"],
                round(r["value"], 3), round(r["z_score"], 2), direction,
                r["severity"], r["detector_type"] or "zscore",
                r["probable_cause"], r["recommended_action"],
                r["correctness"], r["actionability"],
                "y" if r["hallucination"] else "n",
                "", "", "",  # rater 2 columns blank
            ])
    print(f"Written: {OUT_CSV}")


async def _main(write_csv: bool) -> None:
    db = Database()
    await db.connect()
    alerts = await _fetch(db)
    await db.close()

    if not alerts:
        print("No rated alerts found. Run eval.rate first.")
        return

    print(f"Exporting {len(alerts)} rated alerts...")
    _write_txt(alerts)
    if write_csv:
        _write_csv(alerts)
    print("\nSend review_sheet.txt to your second evaluator.")
    print("When they return it, run:  python -m eval.cohens_kappa")


if __name__ == "__main__":
    asyncio.run(_main(write_csv="--csv" in sys.argv))
