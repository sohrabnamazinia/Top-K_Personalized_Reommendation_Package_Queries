from typing import Dict, List, Set, Tuple
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.models import Package, Component, Entity
from utils.llm_interface import LLMEvaluator


# Default range for unknown component values (discrete uniform [MIN, MAX] inclusive)
DEFAULT_MIN = 0.0
DEFAULT_MAX = 1.0


class ScoringFunction:
    """Computes package scores based on components."""
    
    def __init__(self, components: List[Component], llm_evaluator: LLMEvaluator):
        self.components = components
        self.llm_evaluator = llm_evaluator
    
    def _get_component_value_key(self, component: Component, entity_ids: List[str], query: str) -> str:
        """Generate key for caching component values."""
        sorted_ids = tuple(sorted(entity_ids))
        return f"{component.name}:{sorted_ids}:{query}"

    def get_component_mean_variance(
        self,
        component: Component,
        entity_ids: List[str],
        query: str,
        min_val: float = DEFAULT_MIN,
        max_val: float = DEFAULT_MAX
    ) -> Tuple[float, float]:
        """
        Return (mean, variance) for a component value.
        If the value is cached (known), mean = midpoint of interval, variance = 0.
        If unknown, model as discrete uniform on [min_val, max_val] (inclusive):
          mean = (min_val + max_val) / 2
          variance = ((max_val - min_val + 1)**2 - 1) / 12
        """
        cached = self.llm_evaluator.get_component_value_if_cached(
            component, entity_ids, query
        )
        if cached is not None:
            lb, ub = cached
            mean = (lb + ub) / 2.0
            return (mean, 0.0)
        # Unknown: discrete uniform over [min_val, max_val] inclusive
        # mean = (MIN + MAX) / 2; variance = ((MAX - MIN + 1)^2 - 1) / 12
        mean = (min_val + max_val) / 2.0
        variance = ((max_val - min_val + 1) ** 2 - 1) / 12.0
        return (mean, variance)
    
    def probe_question(
        self,
        component: Component,
        entities: Dict[str, Entity],
        entity_ids: List[str],
        query: str,
        use_cache: bool = True
    ) -> Tuple[float, float]:
        """Probe a question (component value). Returns (lower_bound, upper_bound) interval."""
        value = self.llm_evaluator.evaluate_component(
            component, entities, entity_ids, query, use_cache
        )
        return value
    
    def compute_package_score(
        self,
        package: Package,
        entities: Dict[str, Entity],
        query: str,
        use_cache: bool = True
    ) -> Tuple[float, float]:
        """
        Compute total score of a package as an interval [total_lb, total_ub].
        Each component value is (lb, ub); intervals are summed.
        
        Returns:
            (total_lower_bound, total_upper_bound)
        """
        total_lb = 0.0
        total_ub = 0.0
        entity_list = list(package.entities)
        
        for component in self.components:
            if component.dimension == 1:
                for entity_id in entity_list:
                    lb, ub = self.probe_question(
                        component, entities, [entity_id], query, use_cache
                    )
                    total_lb += lb
                    total_ub += ub
            elif component.dimension == 2:
                for i in range(len(entity_list)):
                    for j in range(i + 1, len(entity_list)):
                        lb, ub = self.probe_question(
                            component, entities, [entity_list[i], entity_list[j]], query, use_cache
                        )
                        total_lb += lb
                        total_ub += ub
        
        return (total_lb, total_ub)
    
    def compute_contribution(
        self,
        entity_id: str,
        package: Package,
        entities: Dict[str, Entity],
        query: str,
        use_cache: bool = True
    ) -> Tuple[float, float]:
        """
        Compute contribution of an entity in a package as an interval [lb, ub].
        Contribution = sum of all component value intervals involving this entity.
        
        Returns:
            (contribution_lower_bound, contribution_upper_bound)
        """
        contrib_lb = 0.0
        contrib_ub = 0.0
        entity_list = list(package.entities)
        
        for component in self.components:
            if component.dimension == 1:
                if entity_id in package.entities:
                    lb, ub = self.probe_question(
                        component, entities, [entity_id], query, use_cache
                    )
                    contrib_lb += lb
                    contrib_ub += ub
            elif component.dimension == 2:
                for other_id in entity_list:
                    if other_id != entity_id:
                        sorted_pair = sorted([entity_id, other_id])
                        lb, ub = self.probe_question(
                            component, entities, sorted_pair, query, use_cache
                        )
                        contrib_lb += lb
                        contrib_ub += ub
        
        return (contrib_lb, contrib_ub)

