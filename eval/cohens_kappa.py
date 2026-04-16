"""
Compute Cohen's κ between two raters on the 50 LLM explanation ratings.

Usage
─────
1. Generate the review sheet and send to your second evaluator:
       python -m eval.export_for_review

2. When they return their scores, either:

   a) Pass scores inline (space-separated C A H per alert):
       python -m eval.cohens_kappa --scores "3 4 n, 2 3 y, ..."

   b) Point at their filled review_sheet.txt:
       python -m eval.cohens_kappa --file eval/review_sheet_rater2.txt

The script prints κ for correctness, actionability, and hallucination,
and writes eval/kappa_results.json for the paper.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__file__)))
from src.db import Database

OUT = Path(__file__).parent / "kappa_results.json"


# ── Cohen's κ ─────────────────────────────────────────────────────────────────

def cohens_kappa(r1: list, r2: list) -> float:
    """Unweighted Cohen's κ for two equal-length rating lists."""
    assert len(r1) == len(r2), "Rating lists must be the same length"
    cats = sorted(set(r1) | set(r2))
    n = len(r1)

    # Observed agreement
    p_o = sum(a == b for a, b in zip(r1, r2)) / n

    # Expected agreement
    p_e = sum(
        (r1.count(c) / n) * (r2.count(c) / n)
        for c in cats
    )

    return (p_o - p_e) / (1 - p_e) if (1 - p_e) > 1e-9 else 1.0


def kappa_label(k: float) -> str:
    if k < 0:      return "poor"
    if k < 0.20:   return "slight"
    if k < 0.40:   return "fair"
    if k < 0.60:   return "moderate"
    if k < 0.80:   return "substantial"
    return "almost perfect"


# ── Parse rater 2 scores ──────────────────────────────────────────────────────

def _parse_inline(raw: str) -> list[tuple[int, int, bool]]:
    """Parse 'C A H, C A H, ...' inline string."""
    scores = []
    for token in re.split(r"[,\n]+", raw.strip()):
        parts = token.strip().split()
        if len(parts) != 3:
            continue
        try:
            c = int(parts[0])
            a = int(parts[1])
            h = parts[2].lower() in ("y", "yes", "1")
            scores.append((c, a, h))
        except ValueError:
            continue
    return scores


def _parse_file(path: Path) -> list[tuple[int, int, bool]]:
    """Parse a filled-in review_sheet.txt."""
    scores = []
    for line in path.read_text().splitlines():
        m = re.match(r"\s*Alert\s+\d+:\s*(\d)\s+(\d)\s+([yn])", line, re.IGNORECASE)
        if m:
            c, a, h = int(m.group(1)), int(m.group(2)), m.group(3).lower() == "y"
            scores.append((c, a, h))
    return scores


# ── Fetch rater 1 from DB ─────────────────────────────────────────────────────

async def _fetch_rater1(db: Database) -> list[tuple[int, int, bool]]:
    rows = await db.pool.fetch(
        """
        SELECT correctness, actionability, hallucination
        FROM llm_ratings
        ORDER BY rated_at DESC
        LIMIT 50
        """
    )
    return [(r["correctness"], r["actionability"], bool(r["hallucination"])) for r in rows]


# ── Main ──────────────────────────────────────────────────────────────────────

async def _main() -> None:
    args = sys.argv[1:]

    rater2: list[tuple[int, int, bool]] = []

    if "--scores" in args:
        idx = args.index("--scores")
        rater2 = _parse_inline(args[idx + 1])
    elif "--file" in args:
        idx = args.index("--file")
        rater2 = _parse_file(Path(args[idx + 1]))
    else:
        print("Usage:")
        print("  python -m eval.cohens_kappa --scores '3 4 n, 2 3 y, ...'")
        print("  python -m eval.cohens_kappa --file eval/review_sheet_rater2.txt")
        return

    db = Database()
    await db.connect()
    rater1 = await _fetch_rater1(db)
    await db.close()

    n = min(len(rater1), len(rater2))
    if n == 0:
        print("No scores to compare.")
        return

    r1 = rater1[:n]
    r2 = rater2[:n]

    c1 = [s[0] for s in r1];  c2 = [s[0] for s in r2]
    a1 = [s[1] for s in r1];  a2 = [s[1] for s in r2]
    h1 = [int(s[2]) for s in r1]; h2 = [int(s[2]) for s in r2]

    kc = cohens_kappa(c1, c2)
    ka = cohens_kappa(a1, a2)
    kh = cohens_kappa(h1, h2)

    print(f"\nCohen's κ  (n={n} alerts, 2 raters)\n")
    print(f"  Correctness    κ = {kc:.3f}  ({kappa_label(kc)})")
    print(f"  Actionability  κ = {ka:.3f}  ({kappa_label(ka)})")
    print(f"  Hallucination  κ = {kh:.3f}  ({kappa_label(kh)})")

    results = {
        "n": n,
        "correctness_kappa":   round(kc, 3),
        "actionability_kappa": round(ka, 3),
        "hallucination_kappa": round(kh, 3),
        "correctness_label":   kappa_label(kc),
        "actionability_label": kappa_label(ka),
        "hallucination_label": kappa_label(kh),
    }
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {OUT}")
    print("\nLaTeX snippet:")
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Inter-rater reliability (Cohen's $\kappa$, $n=50$).}")
    print(r"\begin{tabular}{lcc}")
    print(r"\toprule")
    print(r"Dimension & $\kappa$ & Agreement \\ \midrule")
    print(f"Correctness    & {kc:.2f} & {kappa_label(kc).title()} \\\\")
    print(f"Actionability  & {ka:.2f} & {kappa_label(ka).title()} \\\\")
    print(f"Hallucination  & {kh:.2f} & {kappa_label(kh).title()} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


if __name__ == "__main__":
    asyncio.run(_main())
