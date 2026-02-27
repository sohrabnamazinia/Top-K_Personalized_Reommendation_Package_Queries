"""
Smoke test for budget limiting (no assertions framework; prints results).

Runs tiny instances with very small budgets to ensure algorithms stop and return a package.
Run from project root:
  python experiments/quality/section_1/test_budget_limit_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from utils.components_storage import get_config, get_dataset_path
from preprocessing.load_data import load_entities_from_csv
from utils.llm_interface import LLMEvaluator
from PCS.algorithm import PCSAlgorithm
from AQS.algorithm import AQSAlgorithm, QuestionSelectionStrategy


def _entity_order_key(eid: str) -> int:
    if eid.startswith("e") and eid[1:].isdigit():
        return int(eid[1:])
    return 999999


def main() -> None:
    scoring = "f1"
    n = 6
    k = 3
    alpha = 0.8
    budget = 2

    cfg = get_config(scoring)
    query = cfg["query"]
    components = cfg["components"]
    dataset_path = get_dataset_path(scoring)
    entities_all = load_entities_from_csv(str(dataset_path))
    entity_ids = sorted(entities_all.keys(), key=_entity_order_key)[:n]
    entities = {eid: entities_all[eid] for eid in entity_ids}

    print(f"scoring={scoring} n={n} k={k} alpha={alpha} budget={budget}")

    llm1 = LLMEvaluator(use_MGT=True, entities_csv_path=str(dataset_path), components=components, output_dir=str(ROOT / "mgt_Results"), n=n)
    pcs = PCSAlgorithm(components=components, llm_evaluator=llm1, enable_budget_limit=True, budget=budget)
    pkg_pcs, meta_pcs = pcs.run(entities, k, query)
    print("PCS:", sorted(pkg_pcs.entities), meta_pcs.get("stop_reason"))

    llm2 = LLMEvaluator(use_MGT=True, entities_csv_path=str(dataset_path), components=components, output_dir=str(ROOT / "mgt_Results"), n=n)
    aqs = AQSAlgorithm(
        entities=entities,
        components=components,
        k=k,
        alpha=alpha,
        query=query,
        llm_evaluator=llm2,
        selection_strategy=QuestionSelectionStrategy.RANDOM,
        enable_budget_limit=True,
        budget=budget,
        print_log=False,
        init_dim_1=True,
    )
    pkg_aqs, meta_aqs = aqs.run()
    print("AQS(random):", sorted(pkg_aqs.entities) if pkg_aqs else None, meta_aqs["iterations"][-1].get("stop_reason") if meta_aqs.get("iterations") else None)


if __name__ == "__main__":
    main()

