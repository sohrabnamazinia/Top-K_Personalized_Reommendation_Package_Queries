from __future__ import annotations

import random

from PCS.algorithm import PCSAlgorithm
from utils.llm_interface import LLMEvaluator
from utils.models import Component, Entity


def _make_entities(n: int) -> dict[str, Entity]:
    out: dict[str, Entity] = {}
    for i in range(1, n + 1):
        eid = f"e{i}"
        out[eid] = Entity(id=eid, name=f"Entity {i}", data=f"Dummy data for entity {i}.")
    return out


def _run_one(title: str, *, swap_strategy: str) -> None:
    random.seed(7)
    entities = _make_entities(10)
    components = [
        Component(name="c1", description="Unary relevance to the query.", dimension=1),
        Component(name="c2", description="Pairwise diversity between two entities.", dimension=2),
    ]

    llm = LLMEvaluator(mock_api=True, use_MGT=False)
    alg = PCSAlgorithm(
        components=components,
        llm_evaluator=llm,
        budget_rate=4,
        epsilon=0.01,
        smart_initial_package=False,
        swap_strategy=swap_strategy,
        enable_budget_limit=True,
        budget=80,
        return_timings=True,
    )

    pkg, meta = alg.run(entities=entities, k=3, query="find good items")
    probes = alg.scoring_function.llm_evaluator.get_component_evals_count()

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print(f"swap_strategy: {swap_strategy}")
    print(f"final_package: {sorted(pkg.entities)}")
    print(f"final_score:   {meta.get('final_score')}")
    print(f"#probes:       {probes}")
    if alg.return_timings:
        print(
            "timings:      "
            f"ask={meta.get('time_ask_next_question'):.3f}s, "
            f"process={meta.get('time_process_response'):.3f}s, "
            f"total={meta.get('time_total'):.3f}s"
        )


def main() -> None:
    _run_one("PCS (tail-prob heuristic)", swap_strategy="tail_prob")
    _run_one("Random_Approx (PCS with random sampling/swaps)", swap_strategy="random")


if __name__ == "__main__":
    main()

