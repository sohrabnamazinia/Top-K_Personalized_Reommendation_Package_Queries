"""
Compare PCS (approx) vs exact baseline: 1000 entities.
Reports actual top score (from exact baseline) vs score of the package PCS returns.
Run from project root: python compare_pcs_vs_exact_baseline.py [--scoring f1] [--k 2]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from utils.components_storage import get_config, get_dataset_path, CONFIGS
from utils.llm_interface import LLMEvaluator
from preprocessing.load_data import load_entities_from_csv
from BASELINE.algorithm import ExactBaseline
from PCS.algorithm import PCSAlgorithm
from PCS.scoring import ScoringFunction

N_ENTITIES = 1000


def _score_midpoint(interval):
    return (interval[0] + interval[1]) / 2.0


def main():
    parser = argparse.ArgumentParser(description="Compare PCS vs exact baseline (1000 entities)")
    parser.add_argument("--scoring", choices=list(CONFIGS.keys()), default="f1", help="Scoring config (default: f1)")
    parser.add_argument("--k", type=int, default=2, help="Package size (default: 2)")
    parser.add_argument("--no-mock", action="store_true", help="Use real LLM API (default: mock)")
    args = parser.parse_args()

    scoring = args.scoring.lower()
    k = args.k
    csv_path = get_dataset_path(scoring)
    if not csv_path.exists():
        print(f"Dataset not found: {csv_path}")
        sys.exit(1)

    config = get_config(scoring)
    query = config["query"]
    components = config["components"]

    entities_all = load_entities_from_csv(str(csv_path))
    entity_ids = list(entities_all.keys())[:N_ENTITIES]
    entities = {eid: entities_all[eid] for eid in entity_ids}
    n = len(entities)
    if n < k:
        print(f"Need at least {k} entities; got {n}")
        sys.exit(1)
    print(f"Loaded {n} entities, k={k}, scoring={scoring}")
    print(f"Query: {query[:80]}...")
    print()

    llm_evaluator = LLMEvaluator(mock_api=not args.no_mock, use_MGT=False)
    scoring_fn = ScoringFunction(components, llm_evaluator)

    # 1) Exact baseline: get actual top package and top score
    print("Running exact baseline (score all packages)...")
    baseline = ExactBaseline(components, llm_evaluator)
    best_package, scores_descending = baseline.run(entities, query, k)
    actual_top_score = scores_descending[0] if scores_descending else 0.0
    print(f"  Best package: {sorted(best_package.entities)}")
    print(f"  Actual top score (midpoint): {actual_top_score:.4f}")
    print(f"  Top-5 scores: {[round(s, 4) for s in scores_descending[:5]]}")
    print()

    # 2) PCS: run approx algorithm (reuses cache from baseline so same component values)
    print("Running PCS (approx)...")
    pcs = PCSAlgorithm(
        components=components,
        llm_evaluator=llm_evaluator,
        budget_rate=10,
        epsilon=0.01,
        smart_initial_package=True,
        return_timings=False,
    )
    pcs_package, pcs_metadata = pcs.run(entities, k, query)
    pcs_lb, pcs_ub, _ = scoring_fn.compute_package_score(
        pcs_package, entities, query, use_cache=True
    )
    pcs_score = _score_midpoint((pcs_lb, pcs_ub))
    n_iters = len(pcs_metadata.get("iterations", []))
    print(f"  PCS package: {sorted(pcs_package.entities)}")
    print(f"  PCS score (midpoint): {pcs_score:.4f}")
    print(f"  PCS iterations: {n_iters}")
    print()

    # 3) Compare
    gap = actual_top_score - pcs_score
    ratio = (pcs_score / actual_top_score * 100) if actual_top_score > 0 else 0
    print("=" * 60)
    print("Comparison (PCS vs actual top)")
    print("=" * 60)
    print(f"  Actual top score:    {actual_top_score:.4f}")
    print(f"  PCS package score:   {pcs_score:.4f}")
    print(f"  Gap (top - PCS):     {gap:.4f}")
    print(f"  PCS as %% of top:     {ratio:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
