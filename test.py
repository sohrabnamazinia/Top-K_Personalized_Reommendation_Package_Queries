"""
Post-process alpha scalability CSV.

Input:
  experiments_outputs/scalability_outputs/scalability_increase_alpha_old.csv

Output:
  experiments_outputs/scalability_outputs/scalability_increase_alpha.csv

Changes:
  - Remove `time_total` column
  - Force k=3 for all rows
  - Add PCS rows for each alpha (same timings for all alphas since PCS doesn't use alpha)
"""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DIR = ROOT / "experiments_outputs" / "scalability_outputs"
IN_PATH = DIR / "scalability_increase_alpha_old.csv"
OUT_PATH = DIR / "scalability_increase_alpha.csv"

FORCED_K = 3

# PCS timings: (maintain, ask_next, process_response)
PCS_TIMES = {
    "time_maintain_packages": 0.0018174109973188024,
    "time_ask_next_question": 87.19930000000002,
    "time_process_response": 87.20111741099734,
}

OUT_FIELDNAMES = [
    "method",
    "#entities",
    "k",
    "alpha",
    "time_maintain_packages",
    "time_ask_next_question",
    "time_process_response",
]


def _f(row: dict, key: str) -> float:
    return float(str(row.get(key, "")).strip())


def main() -> None:
    if not IN_PATH.exists():
        raise SystemExit(f"Input CSV not found: {IN_PATH}")

    # Read AQS rows (alpha sweep)
    aqs_rows = []
    with open(IN_PATH, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            method = (r.get("method") or "").strip().lower()
            if method != "aqs":
                continue
            try:
                n_entities = int(str(r.get("#entities", "")).strip())
                alpha = float(str(r.get("alpha", "")).strip())
            except (ValueError, TypeError):
                continue
            aqs_rows.append(
                {
                    "method": "aqs",
                    "#entities": n_entities,
                    "k": FORCED_K,
                    "alpha": alpha,
                    "time_maintain_packages": _f(r, "time_maintain_packages"),
                    "time_ask_next_question": _f(r, "time_ask_next_question"),
                    "time_process_response": _f(r, "time_process_response"),
                }
            )

    if not aqs_rows:
        raise SystemExit("No AQS rows found in scalability_increase_alpha_old.csv")

    # Sort by alpha for stable output
    aqs_rows.sort(key=lambda r: r["alpha"])

    out_rows = []
    for r in aqs_rows:
        out_rows.append(r)
        out_rows.append(
            {
                "method": "pcs",
                "#entities": r["#entities"],
                "k": FORCED_K,
                "alpha": r["alpha"],
                **PCS_TIMES,
            }
        )

    DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OUT_FIELDNAMES)
        w.writeheader()
        w.writerows(out_rows)

    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()

