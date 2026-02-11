"""Exact baseline: probe all component values, score all packages, return the best."""
from typing import Dict, List, Optional, Tuple
import sys
from pathlib import Path
from itertools import combinations

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.models import Package, Component, Entity
from utils.llm_interface import LLMEvaluator
from PCS.scoring import ScoringFunction


def _score_midpoint(interval: Tuple[float, float]) -> float:
    return (interval[0] + interval[1]) / 2.0


class ExactBaseline:
    """Exact baseline: evaluates all component values, scores all packages, returns the best."""

    def __init__(self, components: List[Component], llm_evaluator: LLMEvaluator):
        self.components = components
        self.llm_evaluator = llm_evaluator
        self.scoring = ScoringFunction(components, llm_evaluator)

    def build_packages(self, entities: Dict[str, Entity], k: int) -> List[Package]:
        """Build all possible packages of size k from entity ids."""
        entity_ids = list(entities.keys())
        if k > len(entity_ids):
            return []
        return [Package(entities=set(c)) for c in combinations(entity_ids, k)]

    def _probe_all_component_values(
        self, entities: Dict[str, Entity], query: str
    ) -> None:
        """Probe every component value needed for any package (unary: per entity; binary: per pair)."""
        entity_ids = list(entities.keys())
        for component in self.components:
            if component.dimension == 1:
                for eid in entity_ids:
                    self.scoring.probe_question(
                        component, entities, [eid], query, use_cache=True
                    )
            elif component.dimension == 2:
                for i in range(len(entity_ids)):
                    for j in range(i + 1, len(entity_ids)):
                        pair = [entity_ids[i], entity_ids[j]]
                        self.scoring.probe_question(
                            component, entities, pair, query, use_cache=True
                        )

    def run(
        self,
        entities: Dict[str, Entity],
        query: str,
        k: int,
        packages: Optional[List[Package]] = None,
    ) -> Tuple[Package, List[float]]:
        """
        Probe all component values, score all packages, return best package and scores descending.
        If packages is None, build all C(n,k) packages first.
        """
        if packages is None:
            packages = self.build_packages(entities, k)
        if not packages:
            return (Package(entities=set()), [])

        self._probe_all_component_values(entities, query)

        scored = []
        for pkg in packages:
            lb, ub = self.scoring.compute_package_score(
                pkg, entities, query, use_cache=True
            )
            scored.append((pkg, _score_midpoint((lb, ub))))

        scored.sort(key=lambda x: x[1], reverse=True)
        best_package, best_score = scored[0]
        scores_descending = [s for _, s in scored]
        return (best_package, scores_descending)
