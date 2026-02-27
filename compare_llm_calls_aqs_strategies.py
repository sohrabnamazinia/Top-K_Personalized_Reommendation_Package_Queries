"""
Quick experiment: compare number of "LLM calls" across AQS selection strategies.

Definition used here:
  "LLM calls" = number of NEW component evaluations (cache misses) performed by LLMEvaluator.evaluate_component.

We use mock_api=True (no external calls) so you can run fast locally.
Run from project root:
  python compare_llm_calls_aqs_strategies.py
"""

from __future__ import annotations

import random
import statistics
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from utils.components_storage import get_config, get_dataset_path
from preprocessing.load_data import load_entities_from_csv
from utils.llm_interface import LLMEvaluator
from AQS.algorithm import AQSAlgorithm, QuestionSelectionStrategy


# -----------------------------
# CONFIG (edit if you want)
# -----------------------------
SCORING = "f1"
N_ENTITIES = 10
K = 3
ALPHA = 0.8
TRIALS = 5

# If True, counts include unary preprocessing; if False, compares only selection loop.
INIT_DIM_1 = False

STRATEGIES = [
    QuestionSelectionStrategy.RANDOM,
    QuestionSelectionStrategy.GREEDY_LOOSE,
    QuestionSelectionStrategy.GREEDY_TIGHT,
]


def _entity_order_key(eid: str) -> int:
    if eid.startswith("e") and eid[1:].isdigit():
        return int(eid[1:])
    return 999999


def _summ(stats: List[int]) -> Dict[str, float]:
    if not stats:
        return {"mean": 0.0, "stdev": 0.0, "min": 0.0, "max": 0.0}
    if len(stats) == 1:
        return {"mean": float(stats[0]), "stdev": 0.0, "min": float(stats[0]), "max": float(stats[0])}
    return {
        "mean": statistics.mean(stats),
        "stdev": statistics.pstdev(stats),
        "min": float(min(stats)),
        "max": float(max(stats)),
    }


def main() -> None:
    cfg = get_config(SCORING)
    query = cfg["query"]
    components = cfg["components"]
    dataset_path = get_dataset_path(SCORING)

    entities_all = load_entities_from_csv(str(dataset_path))
    entity_ids = sorted(entities_all.keys(), key=_entity_order_key)[:N_ENTITIES]
    entities = {eid: entities_all[eid] for eid in entity_ids}

    results: Dict[str, List[int]] = {s.value: [] for s in STRATEGIES}

    for trial in range(TRIALS):
        # Fixed seeds so each strategy sees comparable randomness per trial.
        base_seed = 12345 + trial
        for strat in STRATEGIES:
            random.seed(base_seed)
            llm = LLMEvaluator(mock_api=True, use_MGT=False)
            llm.reset_component_evals_count()

            algo = AQSAlgorithm(
                entities=entities,
                components=components,
                k=K,
                alpha=ALPHA,
                query=query,
                llm_evaluator=llm,
                initial_packages=None,
                print_log=False,
                init_dim_1=INIT_DIM_1,
                selection_strategy=strat,
            )

            # Suppress AQS prints for clean output
            with redirect_stdout(StringIO()):
                pkg, meta = algo.run()

            calls = llm.get_component_evals_count()
            results[strat.value].append(int(calls))

    print("=" * 72)
    print("AQS selection strategy: cache-miss evaluation counts")
    print("=" * 72)
    print(f"scoring={SCORING}  n={N_ENTITIES}  k={K}  alpha={ALPHA}  trials={TRIALS}  init_dim_1={INIT_DIM_1}")
    print()
    print(f"{'strategy':<14} {'mean':>8} {'stdev':>8} {'min':>6} {'max':>6}   runs")
    for strat in STRATEGIES:
        s = strat.value
        agg = _summ(results[s])
        runs = ",".join(str(x) for x in results[s])
        print(f"{s:<14} {agg['mean']:>8.2f} {agg['stdev']:>8.2f} {agg['min']:>6.0f} {agg['max']:>6.0f}   {runs}")


if __name__ == "__main__":
    main()

