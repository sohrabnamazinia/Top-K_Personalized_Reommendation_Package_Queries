"""
Plot scalability (increase entities): stacked bar chart.

Input CSV (required columns):
  method,#entities,time_maintain_packages,time_ask_next_question,time_process_response

The chart uses x = #entities. For each x there are two bars (AQS on the left, PCS on the right),
each stacked by the 3 time components.

Run from project root:
  python plot/scalability/plot_scalability_increase_entities.py
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = Path(__file__).resolve().parent
CSV_PATH = ROOT / "experiments_outputs" / "scalability_outputs" / "scalability_increase_entities.csv"

METHODS = ["aqs", "pcs"]  # left, right
STACK_KEYS = ["time_maintain_packages", "time_ask_next_question", "time_process_response"]
STACK_LABELS = ["Maintain packages", "Ask next question", "Process response"]
STACK_COLORS = ["#2ecc71", "#3498db", "#e74c3c"]  # green, blue, red
STACK_HATCHES = ["//", "xx", ".."]  # patterns for accessibility (color-vision friendly)


def load_data(csv_path: Path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                n = int(str(row.get("#entities", "")).strip())
                method = str(row.get("method", "")).strip().lower()
            except (ValueError, KeyError, TypeError):
                continue
            if method not in ("aqs", "pcs"):
                continue
            try:
                rows.append({
                    "method": method,
                    "#entities": n,
                    "time_maintain_packages": float(row.get("time_maintain_packages", 0)),
                    "time_ask_next_question": float(row.get("time_ask_next_question", 0)),
                    "time_process_response": float(row.get("time_process_response", 0)),
                })
            except (ValueError, TypeError):
                continue
    return rows


def _total(row: dict) -> float:
    return sum(float(row.get(k, 0.0)) for k in STACK_KEYS)


def plot_stacked_grouped(data, out_name: str) -> None:
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

    entities = sorted(set(r["#entities"] for r in data))
    by_key = {(r["#entities"], r["method"]): r for r in data}
    n_entities = len(entities)
    x = np.arange(n_entities)

    # "Stuck together" bars (AQS vs PCS)
    bar_width = 0.42
    offsets = [-bar_width / 2, bar_width / 2]  # centers separated by bar_width => edges touch

    # Set y-limit from PCS totals; truncate AQS if it exceeds
    pcs_totals = [_total(by_key.get((n, "pcs"), {})) for n in entities]
    y_max = (max(pcs_totals) * 1.12) if any(pcs_totals) else (max(_total(r) for r in data) * 1.12)

    fig, ax = plt.subplots(figsize=(max(9, n_entities * 1.35), 5.8))

    bottoms = np.zeros((n_entities, len(METHODS)))
    for stack_idx, key in enumerate(STACK_KEYS):
        for method_idx, method in enumerate(METHODS):
            vals = np.array([by_key.get((n, method), {}).get(key, 0.0) for n in entities], dtype=float)
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
    for idx, n in enumerate(entities):
        aqs_total = _total(by_key.get((n, "aqs"), {}))
        if aqs_total <= y_max:
            continue
        x_center = idx + offsets[0] + bar_width / 2.0
        y_text = y_max * 0.92
        ax.annotate(
            "exceeds axis",
            xy=(x_center, y_max),
            xytext=(x_center, y_text),
            ha="center",
            va="top",
            fontsize=9,
            annotation_clip=False,
            arrowprops=dict(arrowstyle="-|>", color="#444444", lw=1.0, mutation_scale=10),
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{n:,}" for n in entities])
    ax.set_xlabel("#entities")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Scalability - increasing #entities")
    ax.yaxis.grid(True, color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend (components with color + hatch) + note for methods
    from matplotlib.patches import Patch

    component_handles = [
        Patch(facecolor=STACK_COLORS[i], hatch=STACK_HATCHES[i], edgecolor="#222222", label=STACK_LABELS[i])
        for i in range(3)
    ]
    method_note = Patch(facecolor="none", edgecolor="none", label="AQS = left bar, PCS = right bar")
    ax.legend(handles=component_handles + [method_note], loc="upper left", frameon=True, framealpha=0.95)

    plt.tight_layout()
    out_path = OUT_DIR / out_name
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


def main():
    if not CSV_PATH.exists():
        raise SystemExit(f"Input CSV not found: {CSV_PATH}")
    data = load_data(CSV_PATH)
    plot_stacked_grouped(data, "scalability_increase_entities.png")


if __name__ == "__main__":
    main()
