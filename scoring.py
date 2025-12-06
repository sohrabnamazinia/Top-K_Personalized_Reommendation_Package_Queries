from typing import Dict, List, Set
from models import Package, Component, Entity
from llm_interface import LLMEvaluator


class ScoringFunction:
    """Computes package scores based on components."""
    
    def __init__(self, components: List[Component], llm_evaluator: LLMEvaluator):
        self.components = components
        self.llm_evaluator = llm_evaluator
        self._component_cache: Dict[str, float] = {}
    
    def _get_component_value_key(self, component: Component, entity_ids: List[str], query: str) -> str:
        """Generate key for caching component values."""
        sorted_ids = tuple(sorted(entity_ids))
        return f"{component.name}:{sorted_ids}:{query}"
    
    def get_component_value(
        self,
        component: Component,
        entities: Dict[str, Entity],
        entity_ids: List[str],
        query: str,
        use_cache: bool = True
    ) -> float:
        """Get component value (with caching)."""
        key = self._get_component_value_key(component, entity_ids, query)
        if use_cache and key in self._component_cache:
            return self._component_cache[key]
        
        value = self.llm_evaluator.evaluate_component(
            component, entities, entity_ids, query, use_cache
        )
        
        if use_cache:
            self._component_cache[key] = value
        
        return value
    
    def compute_package_score(
        self,
        package: Package,
        entities: Dict[str, Entity],
        query: str,
        use_cache: bool = True
    ) -> float:
        """
        Compute total score of a package.
        
        Args:
            package: The package to score
            entities: Dictionary mapping entity_id to Entity
            query: User query
            use_cache: Whether to use cached component values
            
        Returns:
            Total score of the package
        """
        total_score = 0.0
        entity_list = list(package.entities)
        
        for component in self.components:
            if component.dimension == 1:
                # Unary component: one value per entity
                for entity_id in entity_list:
                    value = self.get_component_value(
                        component, entities, [entity_id], query, use_cache
                    )
                    total_score += value
            elif component.dimension == 2:
                # Binary component: one value per pair
                for i in range(len(entity_list)):
                    for j in range(i + 1, len(entity_list)):
                        value = self.get_component_value(
                            component, entities, [entity_list[i], entity_list[j]], query, use_cache
                        )
                        total_score += value
        
        return total_score
    
    def compute_contribution(
        self,
        entity_id: str,
        package: Package,
        entities: Dict[str, Entity],
        query: str,
        use_cache: bool = True
    ) -> float:
        """
        Compute contribution score of an entity in a package.
        
        Contribution = sum of all component values involving this entity.
        """
        contribution = 0.0
        entity_list = list(package.entities)
        
        for component in self.components:
            if component.dimension == 1:
                # Unary: only if this is the entity
                if entity_id in package.entities:
                    value = self.get_component_value(
                        component, entities, [entity_id], query, use_cache
                    )
                    contribution += value
            elif component.dimension == 2:
                # Binary: all pairs involving this entity
                for other_id in entity_list:
                    if other_id != entity_id:
                        sorted_pair = sorted([entity_id, other_id])
                        value = self.get_component_value(
                            component, entities, sorted_pair, query, use_cache
                        )
                        contribution += value
        
        return contribution

