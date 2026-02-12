from typing import Dict, List, Set, Tuple
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.models import Package, Component, Entity
from utils.llm_interface import LLMEvaluator


class ScoringFunction:
    """Computes package scores with intervals (lower bound, upper bound) when some component values are unknown."""
    
    def __init__(self, components: List[Component], llm_evaluator: LLMEvaluator, entities: Dict[str, Entity] = None, query: str = None):
        self.components = components
        self.llm_evaluator = llm_evaluator
        self._unknown_questions: Set[str] = set()  # Track which component values have not been computed yet
        # Initialize unknown questions if entities and query are provided
        if entities is not None and query is not None:
            self._initialize_unknown_questions(entities, query)
    
    def _get_component_value_key(self, component: Component, entity_ids: List[str], query: str) -> str:
        """Generate key for caching component values."""
        sorted_ids = tuple(sorted(entity_ids))
        return f"{component.name}:{sorted_ids}:{query}"
    
    def _initialize_unknown_questions(self, entities: Dict[str, Entity], query: str):
        """Initialize unknown questions set with all possible component value keys."""
        entity_list = list(entities.keys())
        for component in self.components:
            if component.dimension == 1:
                # Unary component: one question per entity
                for entity_id in entity_list:
                    key = self._get_component_value_key(component, [entity_id], query)
                    self._unknown_questions.add(key)
            elif component.dimension == 2:
                # Binary component: one question per pair
                for i in range(len(entity_list)):
                    for j in range(i + 1, len(entity_list)):
                        sorted_pair = sorted([entity_list[i], entity_list[j]])
                        key = self._get_component_value_key(component, sorted_pair, query)
                        self._unknown_questions.add(key)
    
    def probe_question(
        self,
        component: Component,
        entities: Dict[str, Entity],
        entity_ids: List[str],
        query: str,
        use_cache: bool = True
    ) -> Tuple[float, float]:
        """Probe a question (component value) and get interval (lower_bound, upper_bound). Removes from unknown questions when computed."""
        # Initialize unknown questions if not done yet
        if not self._unknown_questions:
            self._initialize_unknown_questions(entities, query)
        
        key = self._get_component_value_key(component, entity_ids, query)
        # Get value from LLM evaluator (handles its own caching)
        lb, ub, _ = self.llm_evaluator.evaluate_component(
            component, entities, entity_ids, query, use_cache
        )
        
        # Remove from unknown questions (mark as answered)
        self._unknown_questions.discard(key)
        
        return (lb, ub)
    
    def is_component_value_unknown(
        self,
        component: Component,
        entity_ids: List[str],
        query: str
    ) -> bool:
        """Check if a component value has not been computed yet (is unknown)."""
        # Initialize unknown questions if not done yet
        if not self._unknown_questions:
            # Need entities and query - will be initialized on first probe_question call
            return True  # Assume unknown if not initialized
        
        key = self._get_component_value_key(component, entity_ids, query)
        return key in self._unknown_questions
    
    def compute_package_score_interval(
        self,
        package: Package,
        entities: Dict[str, Entity],
        query: str,
        use_cache: bool = True
    ) -> Tuple[float, float]:
        """
        Compute score interval [lower_bound, upper_bound] for a package.
        
        Some component values may be unknown. For unknown values:
        - Lower bound: assume value = 0.0
        - Upper bound: assume value = 1.0
        
        Args:
            package: The package to score
            entities: Dictionary mapping entity_id to Entity
            query: User query
            use_cache: Whether to use cached component values
            
        Returns:
            (lower_bound, upper_bound) tuple
        """
        # Initialize unknown questions if not done yet
        if not self._unknown_questions:
            self._initialize_unknown_questions(entities, query)
        
        lower_bound = 0.0
        upper_bound = 0.0
        entity_list = list(package.entities)
        
        for component in self.components:
            if component.dimension == 1:
                # Unary component: one value per entity
                for entity_id in entity_list:
                    key = self._get_component_value_key(component, [entity_id], query)
                    
                    if key not in self._unknown_questions:
                        # Known value: get from LLM evaluator (will use its cache)
                        value_lb, value_ub = self.probe_question(
                            component, entities, [entity_id], query, use_cache
                        )
                        lower_bound += value_lb
                        upper_bound += value_ub
                    else:
                        # Unknown value: use bounds [0, 1]
                        lower_bound += 0.0
                        upper_bound += 1.0
                        
            elif component.dimension == 2:
                # Binary component: one value per pair
                for i in range(len(entity_list)):
                    for j in range(i + 1, len(entity_list)):
                        sorted_pair = sorted([entity_list[i], entity_list[j]])
                        key = self._get_component_value_key(component, sorted_pair, query)
                        
                        if key not in self._unknown_questions:
                            # Known value: get from LLM evaluator (will use its cache)
                            value_lb, value_ub = self.probe_question(
                                component, entities, sorted_pair, query, use_cache
                            )
                            lower_bound += value_lb
                            upper_bound += value_ub
                        else:
                            # Unknown value: use bounds [0, 1]
                            lower_bound += 0.0
                            upper_bound += 1.0
        
        return (lower_bound, upper_bound)
