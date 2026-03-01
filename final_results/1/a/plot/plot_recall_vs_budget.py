"""
Plot Section 1A: Recall vs Budget (one plot per scoring function).

Reads CSVs in final_results/1/a/ with filenames:
  recall_vs_budget_f{1..6}_k3_alpha0p5.csv

Writes PNGs to final_results/1/a/plot/
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
PLOT_DIR = Path(__file__).resolve().parent

CSV_TEMPLATE = "recall_vs_budget_{scoring}_k3_alpha0p5.csv"
SCORINGS = [f"f{i}" for i in range(1, 7)]
# Advisor feedback: omit ExactBaseline from final plots (still allowed in CSV).
METHODS = ["AQS", "GreedyLoose", "GreedyTight", "Random", "PCS"]

# Map raw budget values to "% of ExactBaseline probing" for the paper axis.
# (We intentionally cap at 70% so the last point doesn't imply full exact probing.)
BUDGET_TO_PERCENT = {50: 10, 100: 25, 200: 40, 300: 55, 500: 70}


def _read_csv(path: Path) -> Tuple[List[int], Dict[str, List[float]]]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    budgets = [int(float(r["budget"])) for r in rows]
    series: Dict[str, List[float]] = {m: [] for m in METHODS}
    for r in rows:
        for m in METHODS:
            series[m].append(float(r[m]))
    return budgets, series


def _style():
    mpl.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 200,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "-",
        }
    )


BARSTYLE = {
    # Black & white (grayscale) + hatches for distinction
    "AQS": ("#d9d9d9", "//"),
    "GreedyLoose": ("#bdbdbd", "xx"),
    "GreedyTight": ("#f0f0f0", "--"),
    "Random": ("#f7f7f7", ".."),
    "PCS": ("#969696", "\\\\"),
}


def _add_legend_below(ax: plt.Axes) -> None:
    """Legend below plot with clear box (no overlap)."""
    leg = ax.legend(
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        borderpad=0.6,
        handlelength=2.2,
        columnspacing=1.4,
    )
    if leg is not None:
        frame = leg.get_frame()
        frame.set_facecolor("white")
        frame.set_edgecolor("#bbbbbb")
        frame.set_linewidth(0.8)


def _plot_one_bar(scoring: str) -> None:
    csv_path = ROOT / CSV_TEMPLATE.format(scoring=scoring)
    budgets, series = _read_csv(csv_path)

    # Slightly taller aspect ratio for readability in paper layouts
    fig, ax = plt.subplots(figsize=(6.8, 4.3))

    n_methods = len(METHODS)
    bar_w = 0.82 / n_methods
    x = list(range(len(budgets)))

    for i, m in enumerate(METHODS):
        color, hatch = BARSTYLE[m]
        xs = [xi - 0.41 + (i + 0.5) * bar_w for xi in x]
        ax.bar(
            xs,
            series[m],
            width=bar_w,
            label=m,
            color=color,
            edgecolor="black",
            linewidth=0.8,
            hatch=hatch,
        )

    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Budget (% of ExactBaseline)")
    ax.set_ylabel("Recall")
    f_idx = scoring[1:] if scoring.lower().startswith("f") else scoring
    ax.set_title(rf"Recall - increasing Budget - $\mathcal{{F}}_{{{f_idx}}}$")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{BUDGET_TO_PERCENT.get(b, b)}%" for b in budgets])
    _add_legend_below(ax)

    out_path = PLOT_DIR / f"recall_vs_budget_{scoring}_bar.png"
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def _plot_one(scoring: str) -> None:
    _plot_one_bar(scoring)


def main() -> None:
    _style()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    for s in SCORINGS:
        _plot_one(s)
    print(f"Wrote 6 plots to {PLOT_DIR}")


if __name__ == "__main__":
    main()

