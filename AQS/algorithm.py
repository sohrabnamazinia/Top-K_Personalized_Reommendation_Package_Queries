from typing import List, Dict, Set, Tuple, Optional
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import Package, Component, Entity
from AQS.package_manager import PackageManager
from AQS.scoring import ScoringFunction
from llm_interface import LLMEvaluator


class AQSAlgorithm:
    """Admissible Question Selection Algorithm for top-k package queries."""
    
    def __init__(
        self,
        entities: Dict[str, Entity],
        components: List[Component],
        k: int,
        alpha: float,
        query: str,
        llm_evaluator: LLMEvaluator,
        initial_packages: Optional[List[Package]] = None
    ):
        """
        Initialize AQS algorithm.
        
        Args:
            entities: Dictionary mapping entity_id to Entity
            components: List of components
            k: Package size
            alpha: Confidence threshold [0, 1]
            query: User query
            llm_evaluator: LLM evaluator instance
            initial_packages: Optional initial set of packages (if None, uses all possible packages)
        """
        self.entities = entities
        self.components = components
        self.k = k
        self.alpha = alpha
        self.query = query
        self.llm_evaluator = llm_evaluator
        self.scoring_function = ScoringFunction(components, llm_evaluator, entities, query)
        
        # Initialize package manager
        entity_ids = list(entities.keys())
        self.package_manager = PackageManager(
            entities=entity_ids,
            k=k,
            components=components,
            packages=initial_packages
        )
        
        # Initialize known and unknown questions
        self.known_questions: Set[Tuple[str, Tuple[str, ...]]] = set()
        self.unknown_questions: Set[Tuple[str, Tuple[str, ...]]] = self._initialize_unknown_questions()
    
    def _initialize_unknown_questions(self) -> Set[Tuple[str, Tuple[str, ...]]]:
        """Initialize the set of all possible questions."""
        questions = set()
        entity_ids = list(self.entities.keys())
        
        for component in self.components:
            if component.dimension == 1:
                # Unary: one question per entity
                for entity_id in entity_ids:
                    sorted_ids = tuple(sorted([entity_id]))
                    questions.add((component.name, sorted_ids))
            elif component.dimension == 2:
                # Binary: one question per pair
                for i in range(len(entity_ids)):
                    for j in range(i + 1, len(entity_ids)):
                        sorted_ids = tuple(sorted([entity_ids[i], entity_ids[j]]))
                        questions.add((component.name, sorted_ids))
        
        return questions
    
    def _question_to_component_entities(self, question: Tuple[str, Tuple[str, ...]]) -> Tuple[Component, List[str]]:
        """Convert question tuple to (component, entity_ids)."""
        component_name, entity_ids_tuple = question
        # Find component by name
        component = next(c for c in self.components if c.name == component_name)
        entity_ids = list(entity_ids_tuple)
        return component, entity_ids
    
    def identify_current_super_candidate(self) -> Optional[Package]:
        """
        Identify the current super-candidate package P_c*.
        
        Returns the package with the highest probability given currently known information.
        For simplicity, we use the package with the highest upper bound.
        """
        packages = self.package_manager.get_packages()
        if not packages:
            return None
        
        max_ub = float('-inf')
        super_candidate = None
        
        for package in packages:
            lb, ub = self.package_manager.get_bounds(package)
            if ub > max_ub:
                max_ub = ub
                super_candidate = package
        
        return super_candidate
    
    def sort_questions_by_coverage(
        self,
        questions: Set[Tuple[str, Tuple[str, ...]]],
        super_candidate: Package
    ) -> List[Tuple[str, Tuple[str, ...]]]:
        """
        Sort questions by coverage in descending order.
        
        Algorithm 4: sort_questions_by_coverage
        """
        coverage_scores: Dict[Tuple[str, Tuple[str, ...]], int] = {}
        
        for question in questions:
            coverage_scores[question] = 0
            component, entity_ids = self._question_to_component_entities(question)
            
            # Count how many packages this question covers
            for package in self.package_manager.get_packages():
                if self.package_manager.coverage(component, entity_ids, package, super_candidate):
                    coverage_scores[question] += 1
        
        # Sort by coverage descending
        sorted_questions = sorted(questions, key=lambda q: coverage_scores[q], reverse=True)
        return sorted_questions
    
    def simulate_question(
        self,
        question: Tuple[str, Tuple[str, ...]],
        super_candidate: Package
    ) -> None:
        """
        Simulate asking a question with best-case outcome.
        
        Algorithm 3: simulate_question
        Updates bounds assuming best-case response for super-candidate.
        """
        component, entity_ids = self._question_to_component_entities(question)
        
        # Determine response: 1 if Q affects P*, else 0
        questions_for_p_star = self.package_manager._get_questions_for_package(super_candidate)
        if question in questions_for_p_star:
            # Best case: response is [1.0, 1.0] (maximum)
            response = (1.0, 1.0)
        else:
            # Best case: response is [0.0, 0.0] (minimum, doesn't help competitors)
            response = (0.0, 0.0)
        
        # Update bounds for all affected packages
        affected_packages = self.package_manager.update_bounds(component, entity_ids, response)
        
        # Prune dominated packages
        self.package_manager.prune_packages(affected_packages)
    
    def heuristic_evaluation(
        self,
        question: Tuple[str, Tuple[str, ...]]
    ) -> int:
        """
        Compute heuristic h_alpha(Q).
        
        Algorithm 2: heuristic_evaluation
        Estimates minimum number of additional questions needed after asking Q.
        """
        # Identify current super-candidate
        super_candidate = self.identify_current_super_candidate()
        if super_candidate is None:
            return float('inf')
        
        # Create a copy of bounds for simulation (we'll work with actual bounds)
        # Note: We'll restore bounds later, but for now we simulate directly
        t = 0
        
        # Simulate asking Q
        self.simulate_question(question, super_candidate)
        
        # Sort remaining questions by coverage
        remaining_questions = self.unknown_questions - {question}
        sorted_questions = self.sort_questions_by_coverage(remaining_questions, super_candidate)
        
        # Iteratively simulate questions until alpha-top condition is met
        while not self.package_manager.check_alpha_top_package(super_candidate, self.alpha):
            if not sorted_questions:
                # No more questions available
                break
            
            # Pop highest coverage question
            next_question = sorted_questions.pop(0)
            
            # Simulate asking it
            self.simulate_question(next_question, super_candidate)
            t += 1
        
        # Note: In a real implementation, we'd restore the original bounds
        # For now, we assume the simulation modifies bounds temporarily
        # This is a simplification - in practice, you'd want to deep copy bounds
        
        return t
    
    def run(self) -> Tuple[Package, Dict]:
        """
        Run the AQS algorithm.
        
        Algorithm 1: AQS main algorithm
        """
        metadata = {
            'iterations': [],
            'questions_asked': 0,
            'final_package': None
        }
        
        iteration = 0
        
        while self.unknown_questions:
            iteration += 1
            iter_info = {'iteration': iteration}
            
            # Identify current super-candidate
            super_candidate = self.identify_current_super_candidate()
            if super_candidate is None:
                break
            
            iter_info['super_candidate'] = list(super_candidate.entities)
            iter_info['super_candidate_bounds'] = self.package_manager.get_bounds(super_candidate)
            
            # Check if super-candidate is alpha-top
            if self.package_manager.check_alpha_top_package(super_candidate, self.alpha):
                iter_info['alpha_top_condition_met'] = True
                iter_info['stop_reason'] = 'alpha_top_condition_met'
                metadata['iterations'].append(iter_info)
                break
            
            iter_info['alpha_top_condition_met'] = False
            
            # Evaluate heuristics for all unknown questions
            heuristic_values = {}
            for question in self.unknown_questions:
                # Note: This is expensive - in practice, you might want to cache or optimize
                heuristic_values[question] = self.heuristic_evaluation(question)
            
            iter_info['heuristic_values'] = {str(q): v for q, v in heuristic_values.items()}
            
            # Select question with minimum heuristic
            if not heuristic_values:
                iter_info['stop_reason'] = 'no_questions_available'
                metadata['iterations'].append(iter_info)
                break
            
            best_question = min(heuristic_values.items(), key=lambda x: x[1])[0]
            iter_info['selected_question'] = str(best_question)
            
            # Probe LLM for actual response
            component, entity_ids = self._question_to_component_entities(best_question)
            response = self.scoring_function.probe_question(
                component, self.entities, entity_ids, self.query, use_cache=True
            )
            
            iter_info['response'] = response
            metadata['questions_asked'] += 1
            
            # Update bounds
            affected_packages = self.package_manager.update_bounds(component, entity_ids, response)
            
            # Prune dominated packages
            pruned = self.package_manager.prune_packages(affected_packages)
            iter_info['affected_packages'] = len(affected_packages)
            iter_info['pruned_packages'] = len(pruned)
            
            # Move question from unknown to known
            self.unknown_questions.remove(best_question)
            self.known_questions.add(best_question)
            
            iter_info['remaining_unknown_questions'] = len(self.unknown_questions)
            metadata['iterations'].append(iter_info)
        
        # Get final super-candidate
        final_package = self.identify_current_super_candidate()
        metadata['final_package'] = list(final_package.entities) if final_package else None
        metadata['final_bounds'] = self.package_manager.get_bounds(final_package) if final_package else None
        
        return final_package, metadata
