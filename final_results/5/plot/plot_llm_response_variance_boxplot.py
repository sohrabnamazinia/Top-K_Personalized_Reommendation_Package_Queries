"""
Final Results - Section 5

Box plot showing response variance across repeated calls.

Input:
  final_results/5/llm_variance_boxplot_data.csv

Output:
  final_results/5/plot/llm_response_variance_boxplot.png

Run from project root:
  python3 final_results/5/plot/plot_llm_response_variance_boxplot.py
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Tuple

import matplotlib as mpl

mpl.use("Agg")  # headless/server-safe
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[3]
IN_CSV = PROJECT_ROOT / "final_results" / "5" / "llm_variance_boxplot_data.csv"
OUT_DIR = Path(__file__).resolve().parent


def _read() -> Tuple[List[str], List[List[float]]]:
    with open(IN_CSV, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"No rows found in {IN_CSV}")

    queries: List[str] = []
    series: List[List[float]] = []
    for r in rows:
        q = (r.get("query") or "").strip()
        vals: List[float] = []
        for i in range(1, 11):
            vals.append(float(r.get(f"resp{i}", "0") or 0))
        queries.append(q)
        series.append(vals)
    return queries, series


def _style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 220,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )


def main() -> None:
    if not IN_CSV.exists():
        raise SystemExit(f"Missing input CSV: {IN_CSV}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    _style()
    queries, series = _read()

    fig, ax = plt.subplots(figsize=(8.4, 3.8))

    bp = ax.boxplot(
        series,
        patch_artist=True,
        widths=0.55,
        showfliers=True,
        medianprops=dict(color="black", linewidth=1.2),
        boxprops=dict(facecolor="#e0e0e0", edgecolor="black", linewidth=1.0),
        whiskerprops=dict(color="black", linewidth=1.0),
        capprops=dict(color="black", linewidth=1.0),
        flierprops=dict(marker="o", markerfacecolor="white", markeredgecolor="black", markersize=4, alpha=1.0),
    )
    # Subtle hatch to be fully B/W distinguishable in print
    for b in bp.get("boxes", []):
        b.set_hatch("//")

    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("LLM score (0–1)")
    ax.set_xlabel("Query")
    ax.set_title("LLM response variability (10 repeated calls)")
    ax.set_xticks(list(range(1, len(queries) + 1)))
    ax.set_xticklabels([f"Q{i}" for i in range(1, len(queries) + 1)])

    fig.tight_layout()
    out_path = OUT_DIR / "llm_response_variance_boxplot.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

