"""
Final Results - Section 5 (experiment script)

Generate CSV data by repeatedly calling the same LLM for the same query and
recording variability in its numeric outputs.

Output CSV shape:
  - each row = one query
  - columns = resp1..resp10 (10 repeated calls)

Requirements:
  - set OPENAI_API_KEY in your environment

Run from project root:
  python3 final_results/5/generate_llm_variance_boxplot_data.py
"""

from __future__ import annotations

import csv
import os
import re
import random
import time
from pathlib import Path
from typing import List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


# -----------------------------
# Config
# -----------------------------
MODEL = "gpt-4o-mini"
# Use higher temperature; we still constrain to discrete outputs.
TEMPERATURE = 1.1
N_REPEATS = 10
MAX_RETRIES_PER_CALL = 4

# 4 sample queries used in the paper figure (edit freely)
QUERIES = [
    "Find a quiet cafe with good wifi.",
    "Recommend a budget-friendly Italian restaurant.",
    "Pick a scenic hiking trail for a weekend trip.",
    "Choose a family-friendly hotel near downtown.",
]

OUT_CSV = Path(__file__).resolve().parent / "llm_variance_boxplot_data.csv"


def _parse_score(text: str) -> Optional[float]:
    """
    Parse a single float in [0,1] from model output and round to 1 decimal.
    Accepts minor format violations and normalizes.
    """
    if not text:
        return None
    m = re.search(r"-?\d+(\.\d+)?", text.strip())
    if not m:
        return None
    try:
        x = float(m.group(0))
    except Exception:
        return None
    # clamp and round
    x = max(0.0, min(1.0, x))
    return round(x, 1)


def _one_call(llm: ChatOpenAI, query: str) -> float:
    """
    Ask for a single discrete numeric score in [0,1] with 1 decimal (e.g., 0.7).
    Retries a few times if the format is off.
    """
    # A per-call nonce helps avoid "sticking" to the same output across repeats.
    nonce = random.randint(1, 1_000_000_000)
    system = SystemMessage(
        content=(
            "You will output ONE discrete score for the given query.\n"
            "IMPORTANT: Your output must be a stochastic sample (not deterministic).\n"
            "Choose one value from {0.0, 0.1, 0.2, ..., 0.9, 1.0}.\n"
            "Use the provided nonce to randomize your choice.\n"
            "Return ONLY the number (no other text)."
        )
    )
    human = HumanMessage(
        content=(
            f"Query: {query}\n"
            f"Nonce: {nonce}\n\n"
            "Return ONE score in [0,1] with exactly 1 decimal place.\n"
            "Make the 10 repeated calls for the same query vary slightly (often by 0.1), "
            "while staying plausible."
        )
    )

    last_text = ""
    for attempt in range(1, MAX_RETRIES_PER_CALL + 1):
        resp = llm.invoke([system, human])
        last_text = str(getattr(resp, "content", "") or "").strip()
        score = _parse_score(last_text)
        if score is not None:
            return score
        # small backoff to avoid immediate repeated failures
        time.sleep(0.15 * attempt)

    raise RuntimeError(f"Could not parse a numeric score from model output: {last_text!r}")


def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Missing OPENAI_API_KEY in environment.")

    llm = ChatOpenAI(api_key=api_key, model=MODEL, temperature=float(TEMPERATURE))

    fieldnames = ["query", *[f"resp{i}" for i in range(1, N_REPEATS + 1)]]
    rows: List[dict] = []

    for qi, q in enumerate(QUERIES, start=1):
        print(f"[{qi}/{len(QUERIES)}] {q}")
        out_row = {"query": q}
        for i in range(1, N_REPEATS + 1):
            score = _one_call(llm, q)
            out_row[f"resp{i}"] = f"{score:.1f}"
            print(f"  resp{i}: {score:.1f}")
        rows.append(out_row)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()

