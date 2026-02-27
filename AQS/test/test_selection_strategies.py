"""
Quick sanity test for question selection strategies:
  - heuristic (AQS)
  - random
  - greedy (tightest-bounds package, random unknown question affecting it)

Run:
  python AQS/test/test_selection_strategies.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from utils.models import Entity, Component
from utils.llm_interface import LLMEvaluator
from AQS.algorithm import AQSAlgorithm, QuestionSelectionStrategy
from BASELINE.algorithm import ExactBaseline


def build_tiny_entities(n: int = 12):
    return {
        f"e{i}": Entity(id=f"e{i}", name=f"Entity {i}", data=f"Synthetic entity {i}.")
        for i in range(1, n + 1)
    }


def main():
    entities = build_tiny_entities(15)
    components = [
        Component(name="c1", description="Relevance to query", dimension=1),
        Component(name="c2", description="Diversity between two entities", dimension=2),
    ]
    query = "Find a diverse set of good items."
    k = 3
    alpha = 0.8

    # Use one evaluator so all methods see the same cached component values
    llm = LLMEvaluator(mock_api=True, use_MGT=False)

    print("=" * 70)
    print("ExactBaseline")
    print("=" * 70)
    baseline = ExactBaseline(components, llm)
    best_pkg, scores = baseline.run(entities, query, k)
    best_score = scores[0] if scores else None
    print(f"best_package: {sorted(best_pkg.entities)}")
    print(f"best_score_midpoint: {best_score}")

    results = []
    for strat in [
        QuestionSelectionStrategy.HEURISTIC,
        QuestionSelectionStrategy.RANDOM,
        QuestionSelectionStrategy.GREEDY_TIGHT,
        QuestionSelectionStrategy.GREEDY_LOOSE,
    ]:
        print("=" * 70)
        print(f"Strategy: {strat.value}")
        algo = AQSAlgorithm(
            entities=entities,
            components=components,
            k=k,
            alpha=alpha,
            query=query,
            llm_evaluator=llm,
            init_dim_1=True,
            print_log=False,
            selection_strategy=strat,
            return_timings=True,
        )
        final_pkg, meta = algo.run()
        print(f"questions_asked: {meta.get('questions_asked')}")
        print(f"time_total: {meta.get('time_total')}")
        final_entities = sorted(final_pkg.entities) if final_pkg else None
        final_bounds = algo.package_manager.get_bounds(final_pkg) if final_pkg else None
        print(f"final_package: {final_entities}")
        print(f"final_bounds: {final_bounds}")

        final_score = None
        if final_pkg:
            lb, ub, _ = baseline.scoring.compute_package_score(final_pkg, entities, query, use_cache=True)
            final_score = (lb + ub) / 2.0

        results.append(
            {
                "strategy": strat.value,
                "questions_asked": meta.get("questions_asked"),
                "time_total": meta.get("time_total"),
                "final_package": final_entities,
                "final_bounds": final_bounds,
                "final_score_midpoint": final_score,
            }
        )

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"- exact: best_score_midpoint={best_score}, package={sorted(best_pkg.entities)}")
    for r in results:
        q = r["questions_asked"]
        t = r["time_total"]
        s = r["final_score_midpoint"]
        pkg = r["final_package"]
        b = r["final_bounds"]
        print(f"- {r['strategy']}: questions={q}, time_total={t}, score_midpoint={s}, package={pkg}, bounds={b}")


if __name__ == "__main__":
    main()

