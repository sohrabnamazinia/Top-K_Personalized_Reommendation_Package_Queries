"""
Section 2 helper: convert #probes tables to estimated $ cost tables.

Input:
  experiments_outputs/quality_outputs/section_2/quality_section_2_num_probes_k*.csv

Output (next to each input file):
  quality_section_2_cost_usd_k*_....csv

Conversion rule:
  - f1..f4 are "text-only" probes
  - f5..f6 are "text+image" probes
  - cost per probe = measured mean INPUT-token cost per call (USD)
  - cell_usd = (#probes) * (usd_per_probe_for_that_type)

The usd_per_probe constants below come from your measurement output.
Run from project root:
  python experiments/quality/section_2/convert_num_probes_to_usd.py
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parent.parent.parent.parent
IN_DIR = ROOT / "experiments_outputs" / "quality_outputs" / "section_2"

# -----------------------------
# CONFIG (paste your measured values here)
# -----------------------------
# Mean $ cost per probe (INPUT tokens only), from your measurements.
USD_PER_PROBE_TEXT_ONLY_GPT4O_MINI = 8.0535e-05
USD_PER_PROBE_TEXT_PLUS_IMAGE_GPT4O_MINI = 0.0007603950000000001

# If you want to output costs for gpt-4o instead, set these and flip USE_GPT4O_MINI=False.
USD_PER_PROBE_TEXT_ONLY_GPT4O = 0.00134225
USD_PER_PROBE_TEXT_PLUS_IMAGE_GPT4O = 0.01267325

USE_GPT4O_MINI = True

TEXT_ONLY_SCORINGS = {"f1", "f2", "f3", "f4"}
TEXT_PLUS_IMAGE_SCORINGS = {"f5", "f6"}


def _usd_per_probe(scoring: str) -> float:
    s = scoring.lower().strip()
    if USE_GPT4O_MINI:
        if s in TEXT_ONLY_SCORINGS:
            return float(USD_PER_PROBE_TEXT_ONLY_GPT4O_MINI)
        if s in TEXT_PLUS_IMAGE_SCORINGS:
            return float(USD_PER_PROBE_TEXT_PLUS_IMAGE_GPT4O_MINI)
    else:
        if s in TEXT_ONLY_SCORINGS:
            return float(USD_PER_PROBE_TEXT_ONLY_GPT4O)
        if s in TEXT_PLUS_IMAGE_SCORINGS:
            return float(USD_PER_PROBE_TEXT_PLUS_IMAGE_GPT4O)
    raise ValueError(f"Unknown scoring column: {scoring}")


def _parse_int(x: str) -> int:
    x = (x or "").strip()
    if x == "":
        return 0
    return int(float(x))


def _format_usd(x: float) -> str:
    # Keep a fixed precision suitable for small per-probe costs.
    return f"{x:.6f}"


def main() -> None:
    if not IN_DIR.exists():
        raise SystemExit(f"Input folder not found: {IN_DIR}")

    inputs = sorted(IN_DIR.glob("quality_section_2_num_probes_k*.csv"))
    if not inputs:
        raise SystemExit(f"No input files found in {IN_DIR} matching quality_section_2_num_probes_k*.csv")

    model_label = "gpt-4o-mini" if USE_GPT4O_MINI else "gpt-4o"
    print(f"Found {len(inputs)} probe tables. Converting using model={model_label}.")

    for in_path in inputs:
        out_path = in_path.with_name(in_path.name.replace("num_probes", "cost_usd"))

        with open(in_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            rows = list(reader)

        if not fieldnames or "method" not in fieldnames:
            raise ValueError(f"Bad CSV header in {in_path}. Expected a 'method' column.")

        scorings = [c for c in fieldnames if c != "method"]
        for s in scorings:
            _ = _usd_per_probe(s)  # validate

        out_rows: List[Dict[str, str]] = []
        for r in rows:
            out_r: Dict[str, str] = {"method": (r.get("method") or "").strip()}
            for s in scorings:
                probes = _parse_int(r.get(s, "0"))
                usd = probes * _usd_per_probe(s)
                out_r[s] = _format_usd(usd)
            out_rows.append(out_r)

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(out_rows)

        print(f"Wrote {out_path.name}")


if __name__ == "__main__":
    main()

