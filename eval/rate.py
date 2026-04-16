"""
Bulk LLM explanation rater.

Prints all unrated alerts as a numbered list, then collects all scores in
one fast pass: type "3 4 n" per alert (correctness, actionability, hallucination).

Usage:
    python -m eval.rate
"""

from __future__ import annotations

import asyncio
import sys

from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__file__)))

from src.db import Database

console = Console(width=120)

RUBRIC = (
    "[bold]Rubric[/bold]  "
    "[cyan]Correctness 1–5[/cyan]: 1=wrong 3=plausible 5=accurate   "
    "[cyan]Actionability 1–5[/cyan]: 1=vague 3=ok 5=specific   "
    "[cyan]Hallucination[/cyan]: y/n"
)

_SEV_COLOUR = {"low": "green", "medium": "yellow", "high": "red", "critical": "bold red"}


def _print_all(alerts: list[dict]) -> None:
    """Print every alert as a compact numbered block."""
    for i, r in enumerate(alerts, 1):
        sev   = r.get("severity") or "?"
        col   = _SEV_COLOUR.get(sev, "white")
        inj   = "[green]TP[/green]" if r["is_injected"] else "[red]FP[/red]"
        det   = r.get("detector_type") or "zscore"
        z     = r["z_score"]
        value = r["value"]
        sig   = r["signal"]
        mid   = r["machine_id"]

        # Direction
        direction = "▲ above" if value > r["window_mean"] else "▼ below"

        console.print(
            f"[bold white]── {i:02d}[/bold white]  "
            f"{mid}  {sig}  "
            f"val=[yellow]{value:.3f}[/yellow]  "
            f"z=[yellow]{z:+.2f}[/yellow]  {direction} mean  "
            f"det=[dim]{det}[/dim]  "
            f"[{col}]{sev.upper()}[/{col}]  {inj}"
        )
        console.print(f"       [bold]Cause :[/bold] {r['probable_cause'] or '–'}")
        console.print(f"       [bold]Action:[/bold] {r['recommended_action'] or '–'}")
        console.print()


def _parse_line(line: str) -> tuple[int, int, bool] | None:
    """Parse 'C A H' e.g. '3 4 n' → (3, 4, False). Returns None on bad input."""
    parts = line.strip().split()
    if len(parts) != 3:
        return None
    try:
        c = int(parts[0])
        a = int(parts[1])
        h = parts[2].lower() in ("y", "yes", "1", "true")
        if not (1 <= c <= 5 and 1 <= a <= 5):
            return None
        return c, a, h
    except ValueError:
        return None


async def _main() -> None:
    db = Database()
    await db.connect()

    console.rule("[bold blue]TRACE — LLM Explanation Rater[/bold blue]")
    console.print(RUBRIC + "\n")

    alerts = await db.fetch_explained_alerts(limit=50)
    if not alerts:
        console.print("[yellow]No unrated explained alerts. Run the pipeline first.[/yellow]")
        await db.close()
        return

    n = len(alerts)
    console.print(f"[green]{n} unrated alerts — read through, then score each one.[/green]\n")

    _print_all(alerts)

    console.rule("Enter scores")
    console.print(
        f"For each of the [bold]{n}[/bold] alerts above, type: "
        "[cyan]<correctness> <actionability> <hallucination>[/cyan]  "
        "e.g. [bold]3 4 n[/bold]  then Enter.\n"
        "Type [yellow]s[/yellow] to skip an alert, "
        "[yellow]r <number>[/yellow] to reprint that alert.\n"
    )

    scores: list[tuple[int, int, bool] | None] = []
    i = 0
    while i < n:
        r = alerts[i]
        try:
            raw = input(f"  [{i+1:02d}/{n}] (C A H) > ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Interrupted. Progress up to here will be saved.[/yellow]")
            break

        if not raw:
            continue

        if raw.lower() == "s":
            scores.append(None)
            i += 1
            continue

        if raw.lower().startswith("r "):
            try:
                idx = int(raw.split()[1]) - 1
                if 0 <= idx < n:
                    r2 = alerts[idx]
                    console.print(
                        f"\n[bold]── {idx+1:02d}[/bold]  {r2['machine_id']}  {r2['signal']}\n"
                        f"  Cause : {r2['probable_cause']}\n"
                        f"  Action: {r2['recommended_action']}\n"
                    )
            except (ValueError, IndexError):
                pass
            continue

        parsed = _parse_line(raw)
        if parsed is None:
            console.print("[red]  Bad input. Format: C A H  e.g.  3 4 n[/red]")
            continue

        scores.append(parsed)
        i += 1

    # Save all valid scores
    saved = 0
    for row, score in zip(alerts, scores):
        if score is None:
            continue
        c, a, h = score
        await db.insert_rating(
            alert_id      = row["id"],
            correctness   = c,
            actionability = a,
            hallucination = h,
        )
        saved += 1

    await db.close()

    console.print(f"\n[bold green]Done. Saved {saved} ratings.[/bold green]")
    if saved < n:
        console.print(f"[dim]{n - saved} skipped — run again to rate remaining alerts.[/dim]")
    console.print("Run [cyan]python -m eval.metrics[/cyan] to compute scores.")


if __name__ == "__main__":
    asyncio.run(_main())
