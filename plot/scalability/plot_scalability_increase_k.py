"""
Plot scalability (increase k): grouped stacked bar chart (AQS vs PCS).

Input CSV:
  experiments_outputs/scalability_outputs/scalability_increase_k.csv

Expected columns:
  method,#entities,k,time_maintain_packages,time_ask_next_question,time_process_response,time_total
(time_total is optional; the plot uses the sum of the 3 components.)

Run from project root:
  python plot/scalability/plot_scalability_increase_k.py
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = Path(__file__).resolve().parent
CSV_PATH = ROOT / "experiments_outputs" / "scalability_outputs" / "scalability_increase_k.csv"

METHODS = ["aqs", "pcs"]  # left, right
STACK_KEYS = ["time_maintain_packages", "time_ask_next_question", "time_process_response"]
STACK_LABELS = ["Maintain packages", "Ask next question", "Process response"]
STACK_COLORS = ["#2ecc71", "#3498db", "#e74c3c"]
STACK_HATCHES = ["//", "xx", ".."]


def load_data(csv_path: Path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = str(row.get("method", "")).strip().lower()
            if method not in ("aqs", "pcs"):
                continue
            try:
                k = int(str(row.get("k", "")).strip())
            except (ValueError, TypeError):
                continue
            rows.append(
                {
                    "method": method,
                    "k": k,
                    "time_maintain_packages": float(row.get("time_maintain_packages", 0) or 0),
                    "time_ask_next_question": float(row.get("time_ask_next_question", 0) or 0),
                    "time_process_response": float(row.get("time_process_response", 0) or 0),
                }
            )
    return rows


def _total(row: dict) -> float:
    return sum(float(row.get(k, 0.0)) for k in STACK_KEYS)


def main():
    if not CSV_PATH.exists():
        raise SystemExit(f"Input CSV not found: {CSV_PATH}")
    data = load_data(CSV_PATH)
    if not data:
        raise SystemExit("No rows to plot")

    # Paper-ish defaults
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.linewidth": 0.8,
            "savefig.dpi": 300,
            "hatch.linewidth": 0.9,
        }
    )

    ks = sorted(set(r["k"] for r in data))
    by_key = {(r["k"], r["method"]): r for r in data}

    n = len(ks)
    x = np.arange(n)
    bar_width = 0.42
    offsets = [-bar_width / 2, bar_width / 2]  # touch

    pcs_totals = [_total(by_key.get((k, "pcs"), {})) for k in ks]
    y_max = (max(pcs_totals) * 1.12) if any(pcs_totals) else (max(_total(r) for r in data) * 1.12)

    fig, ax = plt.subplots(figsize=(max(8, n * 1.2), 5.8))
    bottoms = np.zeros((n, len(METHODS)))

    for stack_idx, key in enumerate(STACK_KEYS):
        for method_idx, method in enumerate(METHODS):
            vals = np.array([by_key.get((k, method), {}).get(key, 0.0) for k in ks], dtype=float)
            pos = x + offsets[method_idx]
            ax.bar(
                pos,
                vals,
                bar_width,
                bottom=bottoms[:, method_idx],
                color=STACK_COLORS[stack_idx],
                hatch=STACK_HATCHES[stack_idx],
                edgecolor="#222222",
                linewidth=0.35,
            )
            bottoms[:, method_idx] += vals

    ax.set_ylim(0, y_max)
    # Annotate truncated AQS bars
    for idx, k in enumerate(ks):
        aqs_total = _total(by_key.get((k, "aqs"), {}))
        if aqs_total <= y_max:
            continue
        x_center = idx + offsets[0] + bar_width / 2.0
        ax.annotate(
            "exceeds axis",
            xy=(x_center, y_max),
            xytext=(x_center, y_max * 0.92),
            ha="center",
            va="top",
            fontsize=9,
            annotation_clip=False,
            arrowprops=dict(arrowstyle="-|>", color="#444444", lw=1.0, mutation_scale=10),
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("k (package size)")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Scalability - increasing k")
    ax.yaxis.grid(True, color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    from matplotlib.patches import Patch

    component_handles = [
        Patch(facecolor=STACK_COLORS[i], hatch=STACK_HATCHES[i], edgecolor="#222222", label=STACK_LABELS[i])
        for i in range(3)
    ]
    method_note = Patch(facecolor="none", edgecolor="none", label="AQS = left bar, PCS = right bar")
    ax.legend(handles=component_handles + [method_note], loc="upper left", frameon=True, framealpha=0.95)

    plt.tight_layout()
    out_path = OUT_DIR / "scalability_increase_k.png"
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()

