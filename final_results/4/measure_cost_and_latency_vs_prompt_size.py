"""
Final Results - Section 4

Measure how latency and $ cost vary with prompt size for gpt-4o-mini.

- Makes real API calls (requires OPENAI_API_KEY).
- Uses the same cost formula we've used elsewhere: INPUT cost only.
  cost_input_usd = (prompt_tokens / 1_000_000) * 0.15

Output:
  final_results/4/cost_latency_vs_prompt_size_gpt4o_mini.csv

Run from project root:
  python3 final_results/4/measure_cost_and_latency_vs_prompt_size.py
"""

from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


# -----------------------------
# Config
# -----------------------------
MODEL = "gpt-4o-mini"
PRICE_PER_1M_INPUT_TOKENS_USD = 0.15  # provided in your setup

# 5 increasing prompt sizes (approximate; we'll record actual tokens from API usage).
WORD_COUNTS = [200, 500, 1000, 2000, 4000]

# Repeat each size a few times to smooth out latency variance.
TRIALS_PER_SIZE = 3

OUT_PATH = (
    Path(__file__).resolve().parent
    / "cost_latency_vs_prompt_size_gpt4o_mini.csv"
)


# -----------------------------
# Helpers
# -----------------------------
@dataclass(frozen=True)
class Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


def _extract_usage(ai_message: Any) -> Optional[Usage]:
    """
    Extract token usage from LangChain AIMessage response metadata.
    Supports both common shapes seen in langchain_openai.
    """
    meta = getattr(ai_message, "response_metadata", None) or {}
    usage = meta.get("token_usage") or meta.get("usage") or {}
    if not isinstance(usage, dict):
        return None

    def _int(v: Any) -> int:
        try:
            return int(v)
        except Exception:
            return 0

    pt = _int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    ct = _int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    tt = _int(usage.get("total_tokens") or (pt + ct))
    if pt <= 0 and ct <= 0 and tt <= 0:
        return None
    return Usage(prompt_tokens=pt, completion_tokens=ct, total_tokens=tt)


def _cost_input_usd(prompt_tokens: int) -> float:
    return (float(prompt_tokens) / 1_000_000.0) * float(PRICE_PER_1M_INPUT_TOKENS_USD)


def _make_filler_words(n_words: int) -> str:
    # Deterministic filler so runs are comparable.
    # Using "wordX" style makes tokenization stable-ish.
    return " ".join(f"w{i%100}" for i in range(n_words))


def _invoke_once(llm: ChatOpenAI, *, n_words: int) -> Tuple[Optional[Usage], float]:
    filler = _make_filler_words(n_words)
    sys = SystemMessage(
        content="You are a helpful assistant. Reply with exactly: OK"
    )
    human = HumanMessage(
        content=(
            "Return exactly: OK\n\n"
            f"FILLER (ignore): {filler}"
        )
    )
    t0 = time.perf_counter()
    resp = llm.invoke([sys, human])
    elapsed = time.perf_counter() - t0
    return _extract_usage(resp), float(elapsed)


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Missing OPENAI_API_KEY in environment.")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model=MODEL, temperature=0)

    rows: list[Dict[str, Any]] = []
    for n_words in WORD_COUNTS:
        for trial in range(1, int(TRIALS_PER_SIZE) + 1):
            usage, elapsed_s = _invoke_once(llm, n_words=int(n_words))
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0
            total_tokens = usage.total_tokens if usage else 0

            rows.append(
                {
                    "model": MODEL,
                    "word_count_target": int(n_words),
                    "trial": int(trial),
                    "prompt_tokens": int(prompt_tokens),
                    "completion_tokens": int(completion_tokens),
                    "total_tokens": int(total_tokens),
                    "cost_input_usd": f"{_cost_input_usd(prompt_tokens):.8f}",
                    "latency_seconds": f"{elapsed_s:.4f}",
                }
            )
            print(
                f"size={n_words:>5} words  trial={trial}  "
                f"prompt_tokens={prompt_tokens}  "
                f"cost_input_usd={_cost_input_usd(prompt_tokens):.6f}  "
                f"latency={elapsed_s:.2f}s"
            )

    fieldnames = [
        "model",
        "word_count_target",
        "trial",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cost_input_usd",
        "latency_seconds",
    ]
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()

