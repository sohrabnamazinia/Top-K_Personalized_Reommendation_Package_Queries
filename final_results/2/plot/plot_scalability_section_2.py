"""
Final Results - Section 2:
Two stacked bar charts showing why AQS is not scalable.

Charts:
- Scalability - increasing #items
- Scalability - increasing k

Input CSVs (generated for paper-quality figures):
- final_results/2/scalability_increase_entities.csv
- final_results/2/scalability_increase_k.csv

Run from project root:
  python3 final_results/2/plot/plot_scalability_section_2.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")  # headless / server-safe
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent

CSV_ENTITIES = BASE_DIR / "scalability_increase_entities.csv"
CSV_K = BASE_DIR / "scalability_increase_k.csv"

METHODS = ["aqs", "pcs"]  # left, right
STACK_KEYS = ["time_maintain_packages", "time_ask_next_question", "time_process_response"]
STACK_LABELS = ["Maintain packages", "Ask next question", "Process response"]
# Black & white (grayscale) + hatches for accessibility (paper-safe)
STACK_COLORS = ["#d9d9d9", "#bdbdbd", "#969696"]
STACK_HATCHES = ["//", "xx", ".."]


def _total(row: dict) -> float:
    return sum(float(row.get(k, 0.0)) for k in STACK_KEYS)


def _paper_style() -> None:
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


def _load_entities(csv_path: Path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = str(row.get("method", "")).strip().lower()
            if method not in ("aqs", "pcs"):
                continue
            try:
                n = int(str(row.get("#entities", "")).strip())
            except (ValueError, TypeError):
                continue
            rows.append(
                {
                    "method": method,
                    "#entities": n,
                    "time_maintain_packages": float(row.get("time_maintain_packages", 0) or 0),
                    "time_ask_next_question": float(row.get("time_ask_next_question", 0) or 0),
                    "time_process_response": float(row.get("time_process_response", 0) or 0),
                }
            )
    return rows


def _load_k(csv_path: Path):
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


def _add_component_legend(ax):
    from matplotlib.patches import Patch

    component_handles = [
        Patch(facecolor=STACK_COLORS[i], hatch=STACK_HATCHES[i], edgecolor="#222222", label=STACK_LABELS[i])
        for i in range(3)
    ]
    method_note = Patch(facecolor="none", edgecolor="none", label="AQS = left bar, PCS = right bar")
    ax.legend(handles=component_handles + [method_note], loc="upper left", frameon=True, framealpha=0.95)


def _plot_stacked_grouped(
    *,
    x_values: list[int],
    by_key: dict[tuple[int, str], dict],
    x_label: str,
    title: str,
    out_path: Path,
) -> None:
    n = len(x_values)
    x = list(range(n))

    bar_width = 0.42
    offsets = [-bar_width / 2, bar_width / 2]  # touch

    # y-limit from PCS totals; AQS may be truncated (exceeds axis)
    pcs_totals = [_total(by_key.get((v, "pcs"), {})) for v in x_values]
    y_max = (max(pcs_totals) * 1.12) if any(pcs_totals) else (max(_total(r) for r in by_key.values()) * 1.12)

    fig, ax = plt.subplots(figsize=(max(9, n * 1.35), 5.8))
    bottoms = {m: [0.0] * n for m in METHODS}

    for stack_idx, key in enumerate(STACK_KEYS):
        for method_idx, method in enumerate(METHODS):
            vals = [float(by_key.get((v, method), {}).get(key, 0.0)) for v in x_values]
            pos = [xi + offsets[method_idx] for xi in x]
            ax.bar(
                pos,
                vals,
                bar_width,
                bottom=bottoms[method],
                color=STACK_COLORS[stack_idx],
                hatch=STACK_HATCHES[stack_idx],
                edgecolor="#222222",
                linewidth=0.35,
            )
            for i in range(n):
                bottoms[method][i] += vals[i]

    ax.set_ylim(0, y_max)

    # Annotate truncated AQS bars
    for idx, v in enumerate(x_values):
        aqs_total = _total(by_key.get((v, "aqs"), {}))
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

    # Label PCS bars for clarity
    for idx, v in enumerate(x_values):
        pcs_total = _total(by_key.get((v, "pcs"), {}))
        if pcs_total <= 0:
            continue
        x_center = idx + offsets[1] + bar_width / 2.0
        if pcs_total >= 0.18 * y_max:
            y = pcs_total * 0.96
            va = "top"
        else:
            y = min(pcs_total + 0.03 * y_max, y_max * 0.98)
            va = "bottom"
        ax.text(
            x_center,
            y,
            "PCS",
            ha="center",
            va=va,
            fontsize=10,
            color="black",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#cccccc", linewidth=0.6),
            clip_on=True,
        )

    ax.set_xticks(x)
    if x_label == "#items":
        ax.set_xticklabels([f"{v:,}" for v in x_values])
    else:
        ax.set_xticklabels([str(v) for v in x_values])
    ax.set_xlabel(x_label)
    ax.set_ylabel("Time (seconds)")
    ax.set_title(title)
    ax.yaxis.grid(True, color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    _add_component_legend(ax)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path.relative_to(ROOT)}")
    plt.close(fig)


def main() -> None:
    _paper_style()

    if not CSV_ENTITIES.exists():
        raise SystemExit(f"Missing CSV: {CSV_ENTITIES}")
    if not CSV_K.exists():
        raise SystemExit(f"Missing CSV: {CSV_K}")

    # Increasing entities
    data_e = _load_entities(CSV_ENTITIES)
    entities = sorted(set(r["#entities"] for r in data_e))
    by_key_e = {(r["#entities"], r["method"]): r for r in data_e}
    _plot_stacked_grouped(
        x_values=entities,
        by_key=by_key_e,
        x_label="#items",
        title="Scalability - increasing #items",
        out_path=OUT_DIR / "scalability_increase_entities.png",
    )

    # Increasing k
    data_k = _load_k(CSV_K)
    ks = sorted(set(r["k"] for r in data_k))
    by_key_k = {(r["k"], r["method"]): r for r in data_k}
    _plot_stacked_grouped(
        x_values=ks,
        by_key=by_key_k,
        x_label="k (package size)",
        title="Scalability - increasing k",
        out_path=OUT_DIR / "scalability_increase_k.png",
    )


if __name__ == "__main__":
    main()

