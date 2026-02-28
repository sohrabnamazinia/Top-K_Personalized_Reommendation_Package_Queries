"""
Final Results - Section 3
Two black & white line charts (Time vs Budget, Score vs Budget)

Input:
  final_results/3/budget_time_score_f1_n1000_k5.csv

Output:
  final_results/3/plot/budget_vs_time.png
  final_results/3/plot/budget_vs_score.png

Run from project root:
  python3 final_results/3/plot/plot_budget_time_score.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")  # headless/server-safe
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[3]
IN_CSV = PROJECT_ROOT / "final_results" / "3" / "budget_time_score_f1_n1000_k5.csv"
OUT_DIR = Path(__file__).resolve().parent


def _read_rows():
    with open(IN_CSV, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    budgets = [int(float(r["budget"])) for r in rows]
    pcs_time = [float(r["pcs_total_time_s"]) for r in rows]
    rnd_time = [float(r["random_approx_total_time_s"]) for r in rows]
    pcs_score = [float(r["pcs_package_score"]) for r in rows]
    rnd_score = [float(r["random_approx_package_score"]) for r in rows]
    return budgets, pcs_time, rnd_time, pcs_score, rnd_score


def _style():
    mpl.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 220,
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


def _plot_time(budgets, pcs_time, rnd_time) -> None:
    x = list(range(len(budgets)))
    fig, ax = plt.subplots(figsize=(6.8, 3.4))

    ax.plot(x, pcs_time, color="black", linestyle="-", marker="o", linewidth=1.6, label="PCS")
    ax.plot(
        x,
        rnd_time,
        color="black",
        linestyle="--",
        marker="s",
        linewidth=1.6,
        label="Random_Approx",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in budgets])
    ax.set_xlabel("Budget (#probes)")
    ax.set_ylabel("Total time (s)")
    ax.set_title("Time - increasing Budget")
    ax.legend(loc="upper left", frameon=True, framealpha=0.95)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "budget_vs_time.png", bbox_inches="tight")
    plt.close(fig)


def _plot_score(budgets, pcs_score, rnd_score) -> None:
    x = list(range(len(budgets)))
    fig, ax = plt.subplots(figsize=(6.8, 3.4))

    ax.plot(x, pcs_score, color="black", linestyle="-", marker="o", linewidth=1.6, label="PCS")
    ax.plot(
        x,
        rnd_score,
        color="black",
        linestyle="--",
        marker="s",
        linewidth=1.6,
        label="Random_Approx",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in budgets])
    ax.set_xlabel("Budget (#probes)")
    ax.set_ylabel("Package score")
    ax.set_title("Package score - increasing Budget")
    ax.legend(loc="upper left", frameon=True, framealpha=0.95)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "budget_vs_score.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not IN_CSV.exists():
        raise SystemExit(f"Missing input CSV: {IN_CSV}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _style()

    budgets, pcs_time, rnd_time, pcs_score, rnd_score = _read_rows()
    _plot_time(budgets, pcs_time, rnd_time)
    _plot_score(budgets, pcs_score, rnd_score)
    print(f"Wrote plots to {OUT_DIR}")


if __name__ == "__main__":
    main()

