from typing import Dict, List, Tuple, Optional
import random
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.models import Package, Component, Entity
from utils.llm_interface import LLMEvaluator
from PCS.scoring import ScoringFunction


class PCSAlgorithm:
    """Probabilistic Candidate Swap algorithm for top-k package queries."""
    
    def __init__(
        self,
        components: List[Component],
        llm_evaluator: LLMEvaluator,
        budget_rate: int = 5,
        epsilon: float = 0.01,
        smart_initial_package: bool = True,
        exceed_number_of_chance: Optional[int] = None,
    ):
        """
        Initialize PCS algorithm.

        Args:
            components: List of components
            llm_evaluator: LLM evaluator instance
            budget_rate: Number of top candidates to evaluate exactly (s)
            epsilon: Convergence threshold
            smart_initial_package: If True, initial package is top-k by unary scores; if False, k entities chosen at random.
            exceed_number_of_chance: When no_beneficial_swap, try next-s-by-TP up to this many times across the run.
                If None, set at run() to max(1, int(sqrt(n))) where n = number of entities.
        """
        self.components = components
        self.scoring_function = ScoringFunction(components, llm_evaluator)
        self.budget_rate = budget_rate
        self.epsilon = epsilon
        self.smart_initial_package = smart_initial_package
        self.exceed_number_of_chance = exceed_number_of_chance

    def select_initial_package(
        self,
        entities: Dict[str, Entity],
        k: int,
        query: str
    ) -> Package:
        """
        Select initial package: either smart (top-k by unary scores) or naive (random k entities),
        depending on self.smart_initial_package.
        """
        if self.smart_initial_package:
            return self.select_initial_package_smart(entities, k, query)
        return self.select_initial_package_naive(entities, k)

    def select_initial_package_smart(
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
        
        # Compute unary scores for all entities (use midpoint of interval)
        entity_scores = {}
        for entity_id, entity in entities.items():
            score = 0.0
            for component in unary_components:
                lb, ub = self.scoring_function.probe_question(
                    component, entities, [entity_id], query, use_cache=True
                )
                score += (lb + ub) / 2.0
            entity_scores[entity_id] = score
        
        # Select top-k
        sorted_entities = sorted(entity_scores.items(), key=lambda x: x[1], reverse=True)
        top_k_ids = [eid for eid, _ in sorted_entities[:k]]
        
        return Package(entities=set(top_k_ids))

    def select_initial_package_naive(
        self,
        entities: Dict[str, Entity],
        k: int
    ) -> Package:
        """
        Select initial package: k entities chosen uniformly at random.
        
        Args:
            entities: Dictionary mapping entity_id to Entity
            k: Package size
            
        Returns:
            Initial package
        """
        entity_ids = list(entities.keys())
        if k >= len(entity_ids):
            return Package(entities=set(entity_ids))
        chosen = set(random.sample(entity_ids, k))
        return Package(entities=chosen)
    
    def estimate_contribution_stats(
        self,
        entity_id: str,
        package: Package,
        entities: Dict[str, Entity],
        query: str,
        e_minus: str
    ) -> Tuple[float, float]:
        """
        Estimate expected value and variance of contribution for an external entity
        in the hypothetical package p_t = (package \\ {e_minus}) ∪ {entity_id}.

        E[X_i] = sum over j sum over p ⊆ p_t, i ∈ p, |p|=d_j  of μ_{j,p}
        Var(X_i) = sum over j sum over p ⊆ p_t, i ∈ p, |p|=d_j  of σ²_{j,p}

        For known component values: μ = midpoint of cached interval, σ² = 0.
        For unknown: model as discrete uniform on [MIN, MAX]: μ = (MIN+MAX)/2,
        σ² = ((MAX-MIN+1)² - 1) / 12.

        Returns:
            (expected_contribution, variance_of_contribution)
        """
        # Hypothetical package: current package with e_minus removed and entity_id added
        hyp_entities = (package.entities - {e_minus}) | {entity_id}
        hyp_list = list(hyp_entities)

        expected = 0.0
        variance = 0.0

        for component in self.components:
            d_j = component.dimension
            # All subsets p of hyp_entities with entity_id in p and |p| = d_j
            if d_j == 1:
                if entity_id not in hyp_entities:
                    continue
                p_ids = [entity_id]
                mu, sigma_sq = self.scoring_function.get_component_mean_variance(
                    component, p_ids, query
                )
                expected += mu
                variance += sigma_sq
            elif d_j == 2:
                for other_id in hyp_list:
                    if other_id == entity_id:
                        continue
                    p_ids = sorted([entity_id, other_id])
                    mu, sigma_sq = self.scoring_function.get_component_mean_variance(
                        component, p_ids, query
                    )
                    expected += mu
                    variance += sigma_sq

        #print(f"Estimated contribution stats for entity {entity_id} in hypothetical package: E[X] = {expected}, Var(X) = {variance}")
        return (expected, variance)

    def _try_beneficial_swap_or_exceed(
        self,
        sorted_tp: List[Tuple[str, float]],
        e_minus: str,
        package: Package,
        entities: Dict[str, Entity],
        query: str,
        delta_min_mid: float,
        iter_info: dict,
        exceed_chances_remaining: int,
        e_star: str,
        max_contrib_value: Tuple[float, float],
        max_contrib_mid: float
    ) -> Tuple[bool, str, Tuple[float, float], int]:
        """If current best is beneficial, return (True, e_star, max_contrib_value, exceed_chances_remaining).
        Else try next-s-by-TP up to exceed_chances_remaining times; return (swap_done, e_star, max_contrib_value, updated_remaining).
        """
        def _mid(interval):
            return (interval[0] + interval[1]) / 2.0
        if max_contrib_mid > delta_min_mid:
            return (True, e_star, max_contrib_value, exceed_chances_remaining)
        offset = 1
        while exceed_chances_remaining > 0:
            next_s = [eid for eid, _ in sorted_tp[self.budget_rate * offset : self.budget_rate * (offset + 1)]]
            if not next_s:
                break
            exceed_chances_remaining -= 1
            exact_next = {}
            for candidate_id in next_s:
                hyp = package.copy()
                hyp.remove(e_minus)
                hyp.add(candidate_id)
                exact_next[candidate_id] = self.scoring_function.compute_contribution(
                    candidate_id, hyp, entities, query, use_cache=True
                )
            if exact_next:
                best = max(exact_next.items(), key=lambda x: _mid(x[1]))
                e_star, max_contrib_value = best[0], best[1]
                max_contrib_mid = _mid(max_contrib_value)
                if max_contrib_mid > delta_min_mid:
                    iter_info['e_star'] = e_star
                    iter_info['max_contrib_value'] = max_contrib_value
                    iter_info['exceed_used'] = offset
                    return (True, e_star, max_contrib_value, exceed_chances_remaining)
            offset += 1
        return (False, e_star, max_contrib_value, exceed_chances_remaining)

    def run(
        self,
        entities: Dict[str, Entity],
        k: int,
        query: str,
        initial_package: Optional[Package] = None
    ) -> Tuple[Package, Dict]:
        """
        Run PCS algorithm.
        
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

        # Default exceed_number_of_chance from n if not set (scale with sqrt(n))
        if self.exceed_number_of_chance is None:
            n = len(entities)
            self.exceed_number_of_chance = max(1, int(np.sqrt(n)))

        iteration = 0
        exceed_chances_remaining = self.exceed_number_of_chance
        metadata = {
            'iterations': [],
            'final_score': (0.0, 0.0)  # (lb, ub) interval
        }
        
        while True:
            iteration += 1
            iter_info = {'iteration': iteration}

            pkg_score = self.scoring_function.compute_package_score(
                package, entities, query, use_cache=True
            )
            print(f"  Iteration {iteration}: package = {sorted(package.entities)}  total score = {pkg_score}")

            # Step 1: Find entity with minimum contribution (e⁻); contributions are (lb, ub)
            if not contributions:
                break
            
            def _midpoint(interval):
                return (interval[0] + interval[1]) / 2.0

            min_entity = min(contributions.items(), key=lambda x: _midpoint(x[1]))
            e_minus = min_entity[0]
            delta_min = min_entity[1]  # (lb, ub) range
            delta_min_mid = _midpoint(delta_min)
            
            iter_info['e_minus'] = e_minus
            iter_info['delta_min'] = delta_min
            
            # Step 2: For each external entity, estimate and compute TP(e)
            tp_scores = {}
            for ext_entity_id in external_entities:
                E_X, Var_X = self.estimate_contribution_stats(
                    ext_entity_id, package, entities, query, e_minus
                )
                
                # TP(e) = 1 - Var(X_e) / (E[X_e] - δ_min)²; use midpoint of δ_min
                denominator = (E_X - delta_min_mid) ** 2
                if denominator > 0:
                    tp = 1 - (Var_X / denominator)
                    tp = abs(tp)  # Clamp to [0, 1]
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
            
            max_contrib = max(
                exact_contributions.items(),
                key=lambda x: _midpoint(x[1])
            )
            e_star = max_contrib[0]
            max_contrib_value = max_contrib[1]  # (lb, ub)
            max_contrib_mid = _midpoint(max_contrib_value)
            
            iter_info['e_star'] = e_star
            iter_info['max_contrib_value'] = max_contrib_value

            # Step 7: Check if swap is beneficial; if not, try next-s by TP up to exceed_number_of_chance times
            swap_done, e_star, max_contrib_value, exceed_chances_remaining = self._try_beneficial_swap_or_exceed(
                sorted_tp, e_minus, package, entities, query, delta_min_mid, iter_info,
                exceed_chances_remaining, e_star, max_contrib_value, max_contrib_mid
            )
            if not swap_done:
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
        print(f"  Done: package = {sorted(package.entities)}  total score = {final_score}")

        return package, metadata

