from typing import Dict, List, Tuple, Optional
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import Package, Component, Entity
from llm_interface import LLMEvaluator
from scoring import ScoringFunction


class AQAAlgorithm:
    """Approximate Algorithm with Chebyshev-Guided Swaps for top-k package queries."""
    
    def __init__(
        self,
        components: List[Component],
        llm_evaluator: LLMEvaluator,
        budget_rate: int = 5,
        epsilon: float = 0.01
    ):
        """
        Initialize AQA algorithm.
        
        Args:
            components: List of components
            llm_evaluator: LLM evaluator instance
            budget_rate: Number of top candidates to evaluate exactly (s)
            epsilon: Convergence threshold
        """
        self.components = components
        self.scoring_function = ScoringFunction(components, llm_evaluator)
        self.budget_rate = budget_rate
        self.epsilon = epsilon
    
    def select_initial_package(
        self,
        entities: Dict[str, Entity],
        k: int,
        query: str
    ) -> Package:
        """
        Select initial package: top-k entities by sum of unary component values.
        
        Args:
            entities: Dictionary mapping entity_id to Entity
            k: Package size
            query: User query
            
        Returns:
            Initial package
        """
        # Get unary components
        unary_components = [c for c in self.components if c.dimension == 1]
        
        if not unary_components:
            # If no unary components, just take first k entities
            entity_ids = list(entities.keys())[:k]
            return Package(entities=set(entity_ids))
        
        # Compute unary scores for all entities
        entity_scores = {}
        for entity_id, entity in entities.items():
            score = 0.0
            for component in unary_components:
                value = self.scoring_function.probe_question(
                    component, entities, [entity_id], query, use_cache=True
                )
                score += value
            entity_scores[entity_id] = score
        
        # Select top-k
        sorted_entities = sorted(entity_scores.items(), key=lambda x: x[1], reverse=True)
        top_k_ids = [eid for eid, _ in sorted_entities[:k]]
        
        return Package(entities=set(top_k_ids))
    
    def estimate_contribution_stats(
        self,
        entity_id: str,
        package: Package,
        entities: Dict[str, Entity],
        query: str
    ) -> Tuple[float, float]:
        """
        Estimate expected value and variance of contribution for an external entity.
        
        This is a simplified estimation. In practice, you might use historical data
        or other heuristics. For now, we use a simple approximation.
        
        Returns:
            (expected_value, variance)
        """
        # Simple heuristic: estimate based on unary components
        unary_components = [c for c in self.components if c.dimension == 1]
        
        if unary_components:
            # Estimate based on unary component values
            estimated_value = 0.0
            for component in unary_components:
                # Use a simple estimate (could be improved)
                value = self.scoring_function.get_component_value(
                    component, entities, [entity_id], query, use_cache=True
                )
                estimated_value += value
            
            # Add estimated binary contributions (simplified)
            binary_components = [c for c in self.components if c.dimension == 2]
            if binary_components:
                # Rough estimate: average binary value * (package_size)
                estimated_value += len(binary_components) * len(package.entities) * 0.5
            
            # Simple variance estimation (could be improved)
            variance = abs(estimated_value) * 0.1  # 10% of value as variance
        else:
            # Fallback if no unary components
            estimated_value = 1.0
            variance = 0.5
        
        return estimated_value, variance
    
    def run(
        self,
        entities: Dict[str, Entity],
        k: int,
        query: str,
        initial_package: Optional[Package] = None
    ) -> Tuple[Package, Dict]:
        """
        Run AQA algorithm.
        
        Args:
            entities: Dictionary mapping entity_id to Entity
            k: Package size
            query: User query
            initial_package: Optional initial package (if None, will be computed)
            
        Returns:
            (final_package, metadata_dict)
        """
        # Initialize package
        if initial_package is None:
            package = self.select_initial_package(entities, k, query)
        else:
            package = initial_package.copy()
        
        # Ensure we have all necessary component values for initial package
        # Compute contribution scores for all entities in package
        contributions = {}
        for entity_id in package.entities:
            contributions[entity_id] = self.scoring_function.compute_contribution(
                entity_id, package, entities, query, use_cache=True
            )
        
        # Get external entities
        external_entities = set(entities.keys()) - package.entities
        
        iteration = 0
        metadata = {
            'iterations': [],
            'final_score': 0.0
        }
        
        while True:
            iteration += 1
            iter_info = {'iteration': iteration}
            
            # Step 1: Find entity with minimum contribution (e⁻)
            if not contributions:
                break
            
            min_entity = min(contributions.items(), key=lambda x: x[1])
            e_minus = min_entity[0]
            delta_min = min_entity[1]
            
            iter_info['e_minus'] = e_minus
            iter_info['delta_min'] = delta_min
            
            # Step 2: For each external entity, estimate and compute TP(e)
            tp_scores = {}
            for ext_entity_id in external_entities:
                E_X, Var_X = self.estimate_contribution_stats(
                    ext_entity_id, package, entities, query
                )
                
                # Compute tail probability: TP(e) = 1 - Var(X_e) / (E[X_e] - δ_min)²
                if E_X > delta_min:
                    denominator = (E_X - delta_min) ** 2
                    if denominator > 0:
                        tp = 1 - (Var_X / denominator)
                        tp = max(0.0, min(1.0, tp))  # Clamp to [0, 1]
                    else:
                        tp = 0.0
                else:
                    tp = 0.0
                
                tp_scores[ext_entity_id] = tp
            
            # Step 3: Select top-s candidates
            sorted_tp = sorted(tp_scores.items(), key=lambda x: x[1], reverse=True)
            top_s_candidates = [eid for eid, _ in sorted_tp[:self.budget_rate]]
            
            iter_info['top_s_candidates'] = top_s_candidates
            iter_info['max_tp'] = max(tp_scores.values()) if tp_scores else 0.0
            
            # Step 4: Check convergence (max TP < epsilon)
            if not tp_scores or max(tp_scores.values()) < self.epsilon:
                iter_info['stop_reason'] = 'max_tp_below_epsilon'
                metadata['iterations'].append(iter_info)
                break
            
            # Step 5: Query LLM for exact contribution of top-s candidates
            # (in hypothetical package where e_minus is replaced)
            exact_contributions = {}
            for candidate_id in top_s_candidates:
                # Create hypothetical package
                hyp_package = package.copy()
                hyp_package.remove(e_minus)
                hyp_package.add(candidate_id)
                
                # Compute exact contribution
                exact_contrib = self.scoring_function.compute_contribution(
                    candidate_id, hyp_package, entities, query, use_cache=True
                )
                exact_contributions[candidate_id] = exact_contrib
            
            iter_info['exact_contributions'] = exact_contributions
            
            # Step 6: Find best candidate
            if not exact_contributions:
                iter_info['stop_reason'] = 'no_candidates'
                metadata['iterations'].append(iter_info)
                break
            
            max_contrib = max(exact_contributions.items(), key=lambda x: x[1])
            e_star = max_contrib[0]
            max_contrib_value = max_contrib[1]
            
            iter_info['e_star'] = e_star
            iter_info['max_contrib_value'] = max_contrib_value
            
            # Step 7: Check if swap is beneficial
            if max_contrib_value <= delta_min:
                iter_info['stop_reason'] = 'no_beneficial_swap'
                metadata['iterations'].append(iter_info)
                break
            
            # Step 8: Perform swap
            package.remove(e_minus)
            package.add(e_star)
            
            # Recompute contributions for all entities in package (they may have changed due to binary components)
            contributions = {}
            for entity_id in package.entities:
                contributions[entity_id] = self.scoring_function.compute_contribution(
                    entity_id, package, entities, query, use_cache=True
                )
            
            # Update external entities
            external_entities.remove(e_star)
            external_entities.add(e_minus)
            
            iter_info['swap_performed'] = True
            iter_info['stop_reason'] = 'continue'
            metadata['iterations'].append(iter_info)
        
        # Compute final score
        final_score = self.scoring_function.compute_package_score(
            package, entities, query, use_cache=True
        )
        metadata['final_score'] = final_score
        metadata['final_package'] = list(package.entities)
        
        return package, metadata

