"""
Interactive LLM explanation rater.

Fetches up to 50 anomaly alerts that have an LLM explanation but no
human rating yet, then presents each one in a rich terminal UI and
prompts the evaluator for three scores:

  correctness    1–5  (does the probable_cause make physical sense?)
  actionability  1–5  (is the recommended_action specific and useful?)
  hallucination  y/n  (does the explanation contain fabricated facts?)

Ratings are written to the llm_ratings table in TimescaleDB.
Run when the pipeline has accumulated enough explanations (typically
1–2 hours of data).

Usage:
    python -m eval.rate
"""

from __future__ import annotations

import asyncio
import sys

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.prompt import IntPrompt, Prompt
from rich.table import Table
from rich.text import Text

# Add project root to path when run directly
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__file__)))

from src.db import Database

console = Console()


RUBRIC = """\
[bold]Scoring rubric[/bold]
  [cyan]Correctness 1–5[/cyan]  : 1 = completely wrong  3 = plausible  5 = physically accurate
  [cyan]Actionability 1–5[/cyan]: 1 = vague/unhelpful   3 = reasonable  5 = specific & actionable
  [cyan]Hallucination y/n[/cyan]: y = explanation contains fabricated or impossible facts
"""


def _severity_colour(severity: str | None) -> str:
    return {
        "low":      "green",
        "medium":   "yellow",
        "high":     "red",
        "critical": "bold red",
    }.get(severity or "", "white")


def _render_alert(idx: int, total: int, row: dict) -> Panel:
    t = Table.grid(padding=(0, 2))
    t.add_column(style="dim", no_wrap=True)
    t.add_column()

    t.add_row("Machine",  f"{row['machine_id']}  ({row['machine_type']})")
    t.add_row("Signal",   row["signal"])
    t.add_row("Value",    f"{row['value']:.3f}   z={row['z_score']:.2f}")
    t.add_row("Injected", "[green]yes[/green]" if row["is_injected"] else "[red]no[/red]")

    severity_str = f"[{_severity_colour(row['severity'])}]{row['severity']}[/]"
    t.add_row("Severity",    severity_str)
    t.add_row("Confidence",  f"{row['confidence']:.2f}" if row["confidence"] else "–")
    t.add_row("LLM latency", f"{row['llm_latency_ms']} ms" if row["llm_latency_ms"] else "–")
    t.add_row("Model",       row["llm_model"] or "–")
    t.add_row("", "")
    t.add_row("[bold]Probable cause[/bold]",     row["probable_cause"] or "–")
    t.add_row("[bold]Recommended action[/bold]", row["recommended_action"] or "–")

    return Panel(
        t,
        title=f"[bold]Alert {idx}/{total}[/bold]  [dim](id={row['id']})[/dim]",
        border_style="blue",
        expand=False,
    )


def _get_score(prompt: str, lo: int = 1, hi: int = 5) -> int:
    while True:
        val = IntPrompt.ask(prompt)
        if lo <= val <= hi:
            return val
        console.print(f"[red]Enter a number between {lo} and {hi}.[/red]")


def _get_bool(prompt: str) -> bool:
    ans = Prompt.ask(prompt, choices=["y", "n"], default="n")
    return ans == "y"


async def _main() -> None:
    db = Database()
    await db.connect()

    console.rule("[bold blue]TRACE — LLM Explanation Rater[/bold blue]")
    console.print(RUBRIC)

    alerts = await db.fetch_explained_alerts(limit=50)
    if not alerts:
        console.print("[yellow]No unrated alerts found. Run the pipeline for a while first.[/yellow]")
        await db.close()
        return

    console.print(f"[green]Loaded {len(alerts)} unrated alerts.[/green]\n")

    rated = 0
    for idx, row in enumerate(alerts, start=1):
        console.print(_render_alert(idx, len(alerts), row))

        try:
            correctness    = _get_score("  Correctness    (1–5)")
            actionability  = _get_score("  Actionability  (1–5)")
            hallucination  = _get_bool("  Hallucination? (y/n)")
            notes_str      = Prompt.ask("  Notes (optional, Enter to skip)", default="")
            notes          = notes_str.strip() or None
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Rating session interrupted. Progress saved.[/yellow]")
            break

        await db.insert_rating(
            alert_id      = row["id"],
            correctness   = correctness,
            actionability = actionability,
            hallucination = hallucination,
            notes         = notes,
        )
        rated += 1
        console.print(f"[dim]  ✓ Saved ({rated}/{len(alerts)})[/dim]\n")

    await db.close()
    console.print(f"\n[bold green]Done. Rated {rated} alerts.[/bold green]")
    console.print("Run [cyan]python -m eval.metrics[/cyan] to compute final scores.")


if __name__ == "__main__":
    asyncio.run(_main())
