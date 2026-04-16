"""
Evaluation metrics — computes and plots all quantitative results
needed for the paper.

Outputs
───────
  Console table  — precision, recall, F1, detection breakdown
  Console table  — LLM quality scores (mean ± std)
  Console table  — system latency percentiles (P50 / P95 / P99)
  metrics.json   — all numbers in one file for paper tables
  figures/
    detection_metrics.png      — bar chart: P / R / F1 vs baseline
    explanation_quality.png    — grouped bar: correctness & actionability by severity
    latency_distribution.png   — CDF of end-to-end LLM latency
    hallucination_rate.png     — hallucination rate by severity

Usage:
    python -m eval.metrics
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import numpy as np

# Lazy matplotlib import — avoids display server requirement on headless machines
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.db import Database

console = Console()
FIGURES = Path(__file__).parent / "figures"
FIGURES.mkdir(exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt(v: float | None, decimals: int = 3) -> str:
    return "–" if v is None else f"{v:.{decimals}f}"


# ── Detection metrics ─────────────────────────────────────────────────────────

def compute_detection_metrics(rows: list[dict], total_injected: int) -> dict:
    detected     = rows  # every row is a detected alert
    tp           = sum(1 for r in detected if r["is_injected"])
    fp           = sum(1 for r in detected if not r["is_injected"])
    fn           = total_injected - tp

    precision    = tp / (tp + fp) if (tp + fp) else 0.0
    recall       = tp / (tp + fn) if (tp + fn) else 0.0
    f1           = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) else 0.0
    )

    # Baseline: random classifier that flags at the same rate as our detector.
    # For a random classifier, precision = base rate of anomalies in the data,
    # recall = detection flag rate. This gives the performance of a naive system
    # with no signal, useful as a lower bound for comparison.
    total_readings  = total_injected + (len(detected) - tp) + fn  # approximate
    base_rate       = total_injected / max(total_readings, 1)
    flag_rate       = len(detected) / max(total_readings, 1)
    baseline_prec   = min(base_rate, 1.0)
    baseline_recall = min(flag_rate, 1.0)
    baseline_f1     = (
        2 * baseline_prec * baseline_recall / (baseline_prec + baseline_recall)
        if (baseline_prec + baseline_recall) else 0.0
    )

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "baseline_precision": baseline_prec,
        "baseline_recall":    baseline_recall,
        "baseline_f1":        baseline_f1,
    }


# ── LLM quality metrics ───────────────────────────────────────────────────────

def compute_quality_metrics(rows: list[dict]) -> dict:
    rated = [r for r in rows if r.get("correctness") is not None]
    if not rated:
        return {}

    corr  = [r["correctness"]   for r in rated]
    act   = [r["actionability"] for r in rated]
    hall  = [r["hallucination"] for r in rated]

    by_severity: dict[str, dict] = {}
    for r in rated:
        sev = r.get("severity") or "unknown"
        by_severity.setdefault(sev, {"corr": [], "act": [], "hall": []})
        by_severity[sev]["corr"].append(r["correctness"])
        by_severity[sev]["act"].append(r["actionability"])
        by_severity[sev]["hall"].append(int(r["hallucination"]))

    return {
        "n_rated":           len(rated),
        "correctness_mean":  float(np.mean(corr)),
        "correctness_std":   float(np.std(corr)),
        "actionability_mean": float(np.mean(act)),
        "actionability_std":  float(np.std(act)),
        "hallucination_rate": float(np.mean(hall)),
        "by_severity": {
            sev: {
                "n":                   len(v["corr"]),
                "correctness_mean":    float(np.mean(v["corr"])),
                "actionability_mean":  float(np.mean(v["act"])),
                "hallucination_rate":  float(np.mean(v["hall"])),
            }
            for sev, v in by_severity.items()
        },
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def _set_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "axes.grid":        True,
        "grid.alpha":       0.3,
        "font.size":        11,
    })


def plot_detection(det: dict) -> None:
    _set_style()
    labels = ["Precision", "Recall", "F1"]
    our    = [det["precision"],         det["recall"],         det["f1"]]
    base   = [det["baseline_precision"], det["baseline_recall"], det["baseline_f1"]]

    x = np.arange(len(labels))
    w = 0.32

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - w / 2, our,  w, label="Z-score detector (ours)", color="#2563EB")
    ax.bar(x + w / 2, base, w, label="Static threshold (baseline)", color="#94A3B8")

    for i, (o, b) in enumerate(zip(our, base)):
        ax.text(i - w / 2, o + 0.01, f"{o:.2f}", ha="center", fontsize=9, color="#1e3a8a")
        ax.text(i + w / 2, b + 0.01, f"{b:.2f}", ha="center", fontsize=9, color="#334155")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Detection Performance: Z-score Detector vs Baseline")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "detection_metrics.png", dpi=150)
    plt.close(fig)
    console.print(f"[dim]  → saved {FIGURES / 'detection_metrics.png'}[/dim]")


def plot_explanation_quality(qual: dict) -> None:
    if not qual or "by_severity" not in qual:
        return
    _set_style()

    sev_order  = ["low", "medium", "high", "critical"]
    sev_data   = qual["by_severity"]
    labels     = [s for s in sev_order if s in sev_data]
    corr_means = [sev_data[s]["correctness_mean"]   for s in labels]
    act_means  = [sev_data[s]["actionability_mean"]  for s in labels]

    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w / 2, corr_means, w, label="Correctness",    color="#2563EB")
    ax.bar(x + w / 2, act_means,  w, label="Actionability",  color="#16A34A")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 5.5)
    ax.set_ylabel("Mean score (1–5)")
    ax.set_title("LLM Explanation Quality by Anomaly Severity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "explanation_quality.png", dpi=150)
    plt.close(fig)
    console.print(f"[dim]  → saved {FIGURES / 'explanation_quality.png'}[/dim]")


def plot_latency_cdf(rows: list[dict]) -> None:
    latencies = [r["llm_latency_ms"] for r in rows if r.get("llm_latency_ms")]
    if not latencies:
        return
    _set_style()
    sorted_lat = np.sort(latencies)
    cdf        = np.arange(1, len(sorted_lat) + 1) / len(sorted_lat)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(sorted_lat, cdf, color="#2563EB", linewidth=2)

    for pct, label, colour in [
        (0.50, "P50", "#16A34A"),
        (0.95, "P95", "#EA580C"),
        (0.99, "P99", "#DC2626"),
    ]:
        val = float(np.percentile(sorted_lat, pct * 100))
        ax.axvline(val, linestyle="--", color=colour, alpha=0.7)
        ax.text(val + 50, pct - 0.05, f"{label}={val:.0f}ms", color=colour, fontsize=9)

    ax.set_xlabel("LLM latency (ms)")
    ax.set_ylabel("CDF")
    ax.set_title("End-to-End LLM Explanation Latency Distribution")
    fig.tight_layout()
    fig.savefig(FIGURES / "latency_distribution.png", dpi=150)
    plt.close(fig)
    console.print(f"[dim]  → saved {FIGURES / 'latency_distribution.png'}[/dim]")


def plot_hallucination(qual: dict) -> None:
    if not qual or "by_severity" not in qual:
        return
    _set_style()

    sev_order = ["low", "medium", "high", "critical"]
    sev_data  = qual["by_severity"]
    labels    = [s for s in sev_order if s in sev_data]
    rates     = [sev_data[s]["hallucination_rate"] * 100 for s in labels]

    colours = ["#22C55E", "#EAB308", "#F97316", "#EF4444"][:len(labels)]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(labels, rates, color=colours)
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.set_ylim(0, max(rates) * 1.3 + 5 if rates else 30)
    ax.set_ylabel("Hallucination rate (%)")
    ax.set_title("LLM Hallucination Rate by Anomaly Severity")
    fig.tight_layout()
    fig.savefig(FIGURES / "hallucination_rate.png", dpi=150)
    plt.close(fig)
    console.print(f"[dim]  → saved {FIGURES / 'hallucination_rate.png'}[/dim]")


# ── Console tables ────────────────────────────────────────────────────────────

def print_detection_table(det: dict) -> None:
    t = Table(title="Detection Performance", box=box.SIMPLE_HEAD)  # type: ignore[attr-defined]
    t.add_column("Metric",    style="bold")
    t.add_column("Z-score (ours)",        justify="right")
    t.add_column("Baseline",             justify="right")

    for label, key_ours, key_base in [
        ("Precision", "precision",         "baseline_precision"),
        ("Recall",    "recall",            "baseline_recall"),
        ("F1",        "f1",               "baseline_f1"),
    ]:
        t.add_row(label, _fmt(det[key_ours]), _fmt(det[key_base]))

    t.add_row("", "", "")
    t.add_row("True Positives",  str(det["tp"]), "")
    t.add_row("False Positives", str(det["fp"]), "")
    t.add_row("False Negatives", str(det["fn"]), "")
    console.print(t)


def print_quality_table(qual: dict) -> None:
    if not qual:
        console.print("[yellow]No rated alerts yet. Run eval/rate.py first.[/yellow]")
        return

    from rich import box as rbox
    t = Table(title="LLM Explanation Quality", box=rbox.SIMPLE_HEAD)
    t.add_column("Metric", style="bold")
    t.add_column("Value", justify="right")

    t.add_row("Rated alerts",        str(qual["n_rated"]))
    t.add_row("Correctness",         f"{qual['correctness_mean']:.2f} ± {qual['correctness_std']:.2f}")
    t.add_row("Actionability",       f"{qual['actionability_mean']:.2f} ± {qual['actionability_std']:.2f}")
    t.add_row("Hallucination rate",  f"{qual['hallucination_rate']*100:.1f} %")
    console.print(t)


def print_latency_table(latencies: dict) -> None:
    from rich import box as rbox
    t = Table(title="LLM Latency Percentiles", box=rbox.SIMPLE_HEAD)
    t.add_column("Percentile", style="bold")
    t.add_column("Latency (ms)", justify="right")

    for key, label in [("p50", "P50"), ("p95", "P95"), ("p99", "P99")]:
        val = latencies.get(key)
        t.add_row(label, _fmt(val, 0) if val else "–")
    console.print(t)


# ── Main ──────────────────────────────────────────────────────────────────────

async def _main() -> None:
    db = Database()
    await db.connect()

    console.rule("[bold blue]ExplainIoT — Evaluation Metrics[/bold blue]")

    rows           = await db.fetch_all_alerts_for_metrics()
    total_injected = await db.fetch_total_injected()
    latencies      = await db.fetch_latency_percentiles()
    await db.close()

    if not rows:
        console.print("[yellow]No alerts in database yet. Run the pipeline first.[/yellow]")
        return

    det  = compute_detection_metrics(rows, total_injected)
    qual = compute_quality_metrics(rows)

    print_detection_table(det)
    print_quality_table(qual)
    print_latency_table(latencies)

    console.print("\n[bold]Generating figures...[/bold]")
    plot_detection(det)
    plot_explanation_quality(qual)
    plot_latency_cdf(rows)
    plot_hallucination(qual)

    # Dump everything to JSON for paper tables
    output = {
        "detection":  det,
        "quality":    qual,
        "latency_ms": {k: (float(v) if v else None) for k, v in latencies.items()},
    }
    out_path = Path(__file__).parent / "metrics.json"
    out_path.write_text(json.dumps(output, indent=2))
    console.print(f"\n[green]All results saved to {out_path}[/green]")


if __name__ == "__main__":
    # rich.box is needed at module level for table styles
    from rich import box  # noqa: F401 (used in print_detection_table)
    asyncio.run(_main())
