from typing import List, Dict, Set, Tuple, Optional
import sys
import random
import math
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.models import Package, Component, Entity
from AQS.package_manager import PackageManager
from AQS.scoring import ScoringFunction
from utils.llm_interface import LLMEvaluator


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
        initial_packages: Optional[List[Package]] = None,
        print_log: bool = False,
        init_dim_1: bool = True,
        is_next_q_random: bool = False,
        heuristic_top_packages_pct: float = 0.1,
        return_timings: bool = False,
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
            print_log: If True, print detailed logs for each iteration
            init_dim_1: If True, preprocess by asking all dimension 1 (unary) questions and updating bounds
            is_next_q_random: If True, randomly select next question instead of using heuristic evaluation
            heuristic_top_packages_pct: Only compute heuristic for questions that affect the top x%% of packages by lower bound (default 0.1).
        """
        self.entities = entities
        self.components = components
        self.k = k
        self.alpha = alpha
        self.query = query
        self.llm_evaluator = llm_evaluator
        self.print_log = print_log
        self.is_next_q_random = is_next_q_random
        self.heuristic_top_packages_pct = heuristic_top_packages_pct
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
        
        # Track initial package count for pruning statistics
        self.initial_package_count = len(self.package_manager.get_packages())

        self.return_timings = return_timings
        self._timings: Optional[Dict[str, float]] = None
        if return_timings:
            self._timings = {
                "time_maintain_packages": 0.0,
                "time_ask_next_question": 0.0,
                "time_process_response": 0.0,
            }

        # Preprocessing: Ask all dimension 1 (unary) questions if enabled
        if init_dim_1:
            self._preprocess_dimension_1(self._timings)
    
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
    
    def _preprocess_dimension_1(self, timings: Optional[Dict[str, float]] = None):
        """Preprocess by asking all dimension 1 (unary) questions, updating bounds, and pruning packages."""
        # Find all dimension 1 questions
        print("Preprocessing: asking all dimension 1 questions...")
        dim_1_questions = []
        for question in list(self.unknown_questions):
            component_name, entity_ids_tuple = question
            component = next(c for c in self.components if c.name == component_name)
            if component.dimension == 1:
                dim_1_questions.append(question)
        # If all components are dimension 1, only preprocess one of them
        if all(c.dimension == 1 for c in self.components) and dim_1_questions:
            first_component_name = self.components[0].name
            dim_1_questions = [q for q in dim_1_questions if q[0] == first_component_name]
        # Ask each dimension 1 question
        for question in dim_1_questions:

            component, entity_ids = self._question_to_component_entities(question)

            lb, ub, time_taken = self.scoring_function.probe_question(
                component, self.entities, entity_ids, self.query, use_cache=True
            )
            if timings is not None:
                timings["time_process_response"] += time_taken

            t0 = time.perf_counter()
            affected_packages = self.package_manager.update_bounds(component, entity_ids, (lb, ub))
            self.package_manager.prune_packages(affected_packages)
            if timings is not None:
                timings["time_maintain_packages"] += time.perf_counter() - t0

            # Move question from unknown to known
            self.unknown_questions.remove(question)
            self.known_questions.add(question)
        
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
        Uses the average (midpoint) of lower and upper bounds as the selection criterion.
        """
        packages = self.package_manager.get_packages()
        if not packages:
            return None
        
        max_midpoint = float('-inf')
        super_candidate = None
        
        for package in packages:
            lb, ub = self.package_manager.get_bounds(package)
            midpoint = (lb + ub) / 2.0  # Average of lower and upper bound
            if midpoint > max_midpoint:
                max_midpoint = midpoint
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
        super_candidate: Package,
        bounds_copy: Dict,
        packages_copy: List[Package]
    ) -> Tuple[Dict, List[Package]]:
        """
        Simulate asking a question with best-case outcome.
        
        Algorithm 3: simulate_question
        Updates bounds assuming best-case response for super-candidate.
        Works on copies of bounds and packages, returns updated copies.
        
        Args:
            question: The question to simulate
            super_candidate: The super-candidate package
            bounds_copy: Copy of bounds dictionary to update
            packages_copy: Copy of packages list to update
            
        Returns:
            (updated_bounds, updated_packages) tuple
        """
        import copy
        
        component, entity_ids = self._question_to_component_entities(question)
        
        # Determine response: 1 if Q affects P*, else 0
        questions_for_p_star = self.package_manager._get_questions_for_package(super_candidate)
        if question in questions_for_p_star:
            # Best case: response is [1.0, 1.0] (maximum)
            response = (1.0, 1.0)
        else:
            # Best case: response is [0.0, 0.0] (minimum, doesn't help competitors)
            response = (0.0, 0.0)
        
        # Update bounds for all affected packages (on copy)
        affected_packages = []
        for package in packages_copy:
            if self.package_manager._is_package_affected(package, entity_ids):
                affected_packages.append(package)
                # Get current bounds
                key = self.package_manager._package_key(package)
                if key in bounds_copy:
                    current_lb, current_ub = bounds_copy[key]
                    # Update bounds: replace unknown assumption [0, 1] with actual response
                    new_lb = current_lb + response[0]  # was assuming 0.0, now have response_lb
                    new_ub = current_ub - 1.0 + response[1]  # was assuming 1.0, now have response_ub
                    bounds_copy[key] = (new_lb, new_ub)
        
        # Prune dominated packages (on copy)
        # Check ALL packages against ALL other packages, not just affected ones
        packages_to_remove = set()
        
        for package in packages_copy:
            key = self.package_manager._package_key(package)
            if key not in bounds_copy:
                continue
            
            lb, ub = bounds_copy[key]
            
            # Check against all other packages
            for other_package in packages_copy:
                if package == other_package:
                    continue
                
                other_key = self.package_manager._package_key(other_package)
                if other_key not in bounds_copy:
                    continue
                
                other_lb, other_ub = bounds_copy[other_key]
                
                # Case 1: This package's upper bound <= other's lower bound
                # This package can never be better, prune it
                if ub <= other_lb:
                    packages_to_remove.add(key)
                    break  # No need to check further for this package
                
                # Case 2: Other package's upper bound <= this package's lower bound
                # Other package can never be better, prune it
                if other_ub <= lb:
                    packages_to_remove.add(other_key)
        
        # Remove pruned packages from copy
        packages_copy = [pkg for pkg in packages_copy 
                        if self.package_manager._package_key(pkg) not in packages_to_remove]
        
        # Remove their bounds from copy
        for key in packages_to_remove:
            bounds_copy.pop(key, None)
        
        return bounds_copy, packages_copy

    def heuristic_evaluation(
        self,
        question: Tuple[str, Tuple[str, ...]]
    ) -> int:
        """
        Compute heuristic h_alpha(Q).
        
        Algorithm 2: heuristic_evaluation
        Estimates minimum number of additional questions needed after asking Q.
        Works on copies of bounds and packages to avoid modifying actual state.
        """
        import copy
        
        # Identify current super-candidate (using actual state)
        super_candidate = self.identify_current_super_candidate()
        if super_candidate is None:
            return float('inf')
        
        # Create deep copies of bounds and packages for simulation
        bounds_copy = copy.deepcopy(self.package_manager.bounds)
        packages_copy = copy.deepcopy(self.package_manager.packages)
        
        t = 0
        
        # Simulate asking Q (on copies)
        bounds_copy, packages_copy = self.simulate_question(
            question, super_candidate, bounds_copy, packages_copy
        )
        
        # Helper function to compute prob(p >= other) for discrete values
        def compute_prob_exceeds(lb_p, ub_p, lb_other, ub_other):
            """Compute probability that p >= other using discrete enumeration."""
            # Generate all possible discrete values (1 decimal place)
            p_values = []
            current = lb_p
            while current <= ub_p + 0.05:  # Add small epsilon for floating point
                p_values.append(round(current, 1))
                current += 0.1
            
            other_values = []
            current = lb_other
            while current <= ub_other + 0.05:
                other_values.append(round(current, 1))
                current += 0.1
            
            # Count cases where p >= other
            favorable_cases = 0
            total_cases = len(p_values) * len(other_values)
            
            for p_val in p_values:
                for other_val in other_values:
                    if p_val >= other_val:
                        favorable_cases += 1
            
            if total_cases == 0:
                return 0.0
            
            return favorable_cases / total_cases
        
        # Helper function to check alpha-top on copies
        def check_alpha_top_on_copy(package, bounds, packages, alpha):
            """Check alpha-top condition using copied bounds and packages."""
            key = self.package_manager._package_key(package)
            if key not in bounds:
                return False
            
            lb_p, ub_p = bounds[key]
            
            # If no other packages, definitely best
            other_packages = [p for p in packages if p != package]
            if not other_packages:
                return True
            
            # Compute probability that this package >= each other package
            probabilities = []
            for other_package in other_packages:
                other_key = self.package_manager._package_key(other_package)
                if other_key not in bounds:
                    continue
                
                other_lb, other_ub = bounds[other_key]
                
                # Compute prob(p >= other)
                prob_exceeds = compute_prob_exceeds(lb_p, ub_p, other_lb, other_ub)
                probabilities.append(prob_exceeds)
            
            if not probabilities:
                return False
            
            # Total probability = product of all individual probabilities
            total_probability = 1.0
            for prob in probabilities:
                total_probability *= prob
            
            return total_probability >= alpha
        
        # Sort remaining questions by coverage (using actual package manager for coverage calculation)
        remaining_questions = self.unknown_questions - {question}
        sorted_questions = self.sort_questions_by_coverage(remaining_questions, super_candidate)
        
        # Iteratively simulate questions until alpha-top condition is met
        while not check_alpha_top_on_copy(super_candidate, bounds_copy, packages_copy, self.alpha):
            if not sorted_questions:
                # No more questions available
                break
            
            # Pop highest coverage question
            next_question = sorted_questions.pop(0)
            
            # Simulate asking it (on copies)
            bounds_copy, packages_copy = self.simulate_question(
                next_question, super_candidate, bounds_copy, packages_copy
            )
            t += 1
        
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
            n_unknown = len(self.unknown_questions)
            if self.print_log:
                print(f"  Iteration {iteration} (unknown questions: {n_unknown})...")
            iter_info = {'iteration': iteration}
            
            # Identify current super-candidate
            super_candidate = self.identify_current_super_candidate()
            if super_candidate is None:
                break

            print(f"  Super candidate: {list(super_candidate.entities)}")
            
            iter_info['super_candidate'] = list(super_candidate.entities)
            iter_info['super_candidate_bounds'] = self.package_manager.get_bounds(super_candidate)
            
            # Check if super-candidate is alpha-top
            if self.package_manager.check_alpha_top_package(super_candidate, self.alpha):
                iter_info['alpha_top_condition_met'] = True
                iter_info['stop_reason'] = 'alpha_top_condition_met'
                metadata['iterations'].append(iter_info)
                break
            
            print(f"  Alpha top condition not met")
            iter_info['alpha_top_condition_met'] = False
            
            # Select next question based on mode
            t0_ask = time.perf_counter()
            if self.is_next_q_random:
                # Random selection: choose randomly from unknown questions
                if not self.unknown_questions:
                    iter_info['stop_reason'] = 'no_questions_available'
                    metadata['iterations'].append(iter_info)
                    break
                best_question = random.choice(list(self.unknown_questions))
                iter_info['selected_question'] = str(best_question)
                iter_info['selection_mode'] = 'random'
            else:
                # Heuristic-based selection: only evaluate heuristic for questions affecting top x% packages (by lower bound)
                packages = self.package_manager.get_packages()
                n_pkg = len(packages)
                top_n = max(1, math.ceil(self.heuristic_top_packages_pct * n_pkg))
                sorted_by_lb = sorted(
                    packages,
                    key=lambda p: self.package_manager.get_bounds(p)[0],
                    reverse=True,
                )
                top_packages = sorted_by_lb[:top_n]
                questions_affecting_top = set()
                for pkg in top_packages:
                    questions_affecting_top |= self.package_manager._get_questions_for_package(pkg)
                questions_to_eval = self.unknown_questions & questions_affecting_top
                if not questions_to_eval:
                    questions_to_eval = self.unknown_questions
                if self.print_log:
                    print(f"  Heuristic: evaluating {len(questions_to_eval)} questions (top {top_n}/{n_pkg} packages)")
                heuristic_values = {q: self.heuristic_evaluation(q) for q in questions_to_eval}
                iter_info['heuristic_values'] = {str(q): v for q, v in heuristic_values.items()}
                # Select question with minimum heuristic
                if not heuristic_values:
                    iter_info['stop_reason'] = 'no_questions_available'
                    metadata['iterations'].append(iter_info)
                    break

                best_question = min(heuristic_values.items(), key=lambda x: x[1])[0]
                iter_info['selected_question'] = str(best_question)
                iter_info['selection_mode'] = 'heuristic'
            if self._timings is not None:
                self._timings["time_ask_next_question"] += time.perf_counter() - t0_ask

            # Probe LLM for actual response (returns lb, ub, time_taken; time_taken is API/MGT/mock time)
            component, entity_ids = self._question_to_component_entities(best_question)
            lb, ub, time_taken = self.scoring_function.probe_question(
                component, self.entities, entity_ids, self.query, use_cache=True
            )
            if self._timings is not None:
                self._timings["time_process_response"] += time_taken

            response = (lb, ub)
            iter_info['response'] = response
            metadata['questions_asked'] += 1

            # Update bounds
            t0_maintain = time.perf_counter()
            affected_packages = self.package_manager.update_bounds(component, entity_ids, response)

            # Prune dominated packages
            pruned = self.package_manager.prune_packages(affected_packages)
            if self._timings is not None:
                self._timings["time_maintain_packages"] += time.perf_counter() - t0_maintain
            iter_info['affected_packages'] = len(affected_packages)
            iter_info['pruned_packages'] = len(pruned)
            
            # Print log if enabled
            if self.print_log:
                print("=" * 60)
                print(f"Iteration {iteration}")
                print("=" * 60)
                print("\nPackages with bounds:")
                for pkg in self.package_manager.get_packages():
                    lb, ub = self.package_manager.get_bounds(pkg)
                    print(f"  Package {list(pkg.entities)}: bounds=({lb:.2f}, {ub:.2f})")
                print(f"\nSelected question: {best_question}")
                print(f"Response: {response}")
                if pruned:
                    print(f"\nPruned packages ({len(pruned)}):")
                    for pkg in pruned:
                        print(f"  {list(pkg.entities)}")
                else:
                    print("\nNo packages pruned in this iteration")
                print()
            
            # Move question from unknown to known
            self.unknown_questions.remove(best_question)
            self.known_questions.add(best_question)
            
            iter_info['remaining_unknown_questions'] = len(self.unknown_questions)
            metadata['iterations'].append(iter_info)
        
        # Get final super-candidate
        final_package = self.identify_current_super_candidate()
        metadata['final_package'] = list(final_package.entities) if final_package else None
        metadata['final_bounds'] = self.package_manager.get_bounds(final_package) if final_package else None
        
        # Calculate total packages pruned (including preprocessing)
        final_package_count = len(self.package_manager.get_packages())
        total_packages_pruned = self.initial_package_count - final_package_count

        if self._timings is not None:
            metadata["time_maintain_packages"] = self._timings["time_maintain_packages"]
            metadata["time_ask_next_question"] = self._timings["time_ask_next_question"]
            metadata["time_process_response"] = self._timings["time_process_response"]
            metadata["time_total"] = (
                self._timings["time_maintain_packages"]
                + self._timings["time_ask_next_question"]
                + self._timings["time_process_response"]
            )

        # Print number of packages pruned
        print(f"\nNumber of packages pruned: {total_packages_pruned}")

        return final_package, metadata
