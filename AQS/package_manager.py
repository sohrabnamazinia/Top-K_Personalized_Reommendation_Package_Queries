from typing import List, Optional, Set, Dict, Tuple
import sys
from pathlib import Path
from itertools import combinations

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import Package, Entity, Component


class PackageManager:
    """Manages a set of candidate packages for the AQS algorithm."""
    
    def __init__(
        self,
        entities: List[str],
        k: int,
        components: List[Component],
        packages: Optional[List[Package]] = None
    ):
        """
        Initialize PackageManager.
        
        Args:
            entities: List of all entity IDs
            k: Size of each package
            components: List of components (needed to calculate maximum possible score)
            packages: Optional list of initial packages. If None, builds all possible packages of size k.
        """
        self.entities = entities
        self.k = k
        self.components = components
        self.packages: List[Package] = []
        
        if packages is not None:
            # Use provided packages
            self.packages = packages
        else:
            # Build all possible packages of size k
            self.packages = self.build_all_packages(entities, k)
        
        # Initialize bounds for each package
        # Key: tuple of sorted entity IDs (hashable representation of package)
        # Value: (lower_bound, upper_bound) tuple
        self.bounds: Dict[Tuple[str, ...], Tuple[float, float]] = {}
        self._initialize_bounds()
    
    def build_all_packages(self, entities: List[str], k: int) -> List[Package]:
        """
        Build all possible packages of size k from the given entities.
        
        Args:
            entities: List of entity IDs
            k: Size of each package
            
        Returns:
            List of all possible Package objects of size k
        """
        if k > len(entities):
            return []  # Cannot build packages larger than available entities
        
        if k <= 0:
            return []  # Invalid package size
        
        # Generate all combinations of size k
        all_packages = []
        for combo in combinations(entities, k):
            package = Package(entities=set(combo))
            all_packages.append(package)
        
        return all_packages
    
    def _package_key(self, package: Package) -> Tuple[str, ...]:
        """Get hashable key for a package (sorted tuple of entity IDs)."""
        return tuple(sorted(package.entities))
    
    def _calculate_maximum_score(self) -> float:
        """
        Calculate maximum possible package score.
        
        Maximum = sum of all component values when each is at maximum (1.0).
        For unary components: k values per component
        For binary components: k*(k-1)/2 values per component
        """
        max_score = 0.0
        
        for component in self.components:
            if component.dimension == 1:
                # Unary: k values, each max 1.0
                max_score += self.k * 1.0
            elif component.dimension == 2:
                # Binary: k choose 2 = k*(k-1)/2 values, each max 1.0
                num_pairs = self.k * (self.k - 1) // 2
                max_score += num_pairs * 1.0
        
        return max_score
    
    def _initialize_bounds(self):
        """Initialize bounds for all packages: lower=0.0, upper=maximum possible score."""
        max_score = self._calculate_maximum_score()
        
        for package in self.packages:
            key = self._package_key(package)
            # Initially: lower bound = 0.0, upper bound = maximum possible score
            self.bounds[key] = (0.0, max_score)
    
    def get_packages(self) -> List[Package]:
        """Get the list of managed packages."""
        return self.packages
    
    def get_package_count(self) -> int:
        """Get the number of packages currently managed."""
        return len(self.packages)
    
    def get_bounds(self, package: Package) -> Tuple[float, float]:
        """
        Get the score bounds for a package.
        
        Args:
            package: The package to get bounds for
            
        Returns:
            (lower_bound, upper_bound) tuple
        """
        key = self._package_key(package)
        return self.bounds.get(key, (0.0, 0.0))
    
    def set_bounds(self, package: Package, lower_bound: float, upper_bound: float):
        """
        Set the score bounds for a package.
        
        Args:
            package: The package to set bounds for
            lower_bound: Lower bound of the score interval
            upper_bound: Upper bound of the score interval
        """
        key = self._package_key(package)
        self.bounds[key] = (lower_bound, upper_bound)
    
    def _is_package_affected(self, package: Package, entity_ids: List[str]) -> bool:
        """Check if a package is affected by a question (all entities in question exist in package)."""
        return all(entity_id in package.entities for entity_id in entity_ids)
    
    def update_bounds(
        self,
        component: Component,
        entity_ids: List[str],
        response: Tuple[float, float]
    ) -> List[Package]:
        """
        Update bounds for all packages affected by a question response.
        
        A package is affected if all entities in the question exist in that package.
        Updates the bounds by replacing the unknown value assumption [0, 1] with the actual response.
        
        Args:
            component: The component that was evaluated
            entity_ids: List of entity IDs in the question (1 for unary, 2 for binary)
            response: (lower_bound, upper_bound) tuple from LLM response
            
        Returns:
            List of packages that were affected and updated
        """
        response_lb, response_ub = response
        affected_packages = []
        
        # Find all packages that contain all entities in the question
        for package in self.packages:
            if self._is_package_affected(package, entity_ids):
                affected_packages.append(package)
                
                # Get current bounds
                key = self._package_key(package)
                current_lb, current_ub = self.bounds[key]
                
                # Update bounds: replace unknown assumption [0, 1] with actual response
                # Before: we assumed this component value was [0, 1]
                # After: we know it's [response_lb, response_ub]
                # So: new_lb = current_lb - 0.0 + response_lb
                #     new_ub = current_ub - 1.0 + response_ub
                new_lb = current_lb + response_lb  # was assuming 0.0, now have response_lb
                new_ub = current_ub - 1.0 + response_ub  # was assuming 1.0, now have response_ub
                
                self.bounds[key] = (new_lb, new_ub)
        
        return affected_packages
    
    def prune_packages(self, packages_to_check: List[Package]) -> List[Package]:
        """
        Prune packages that are dominated by others based on bounds.
        
        A package is pruned if:
        - Its upper bound <= lower bound of any other package (this package can never be better)
        - OR any other package's lower bound >= this package's upper bound (that other package can never be better)
        
        Args:
            packages_to_check: List of packages to check for pruning (typically recently updated packages)
            
        Returns:
            List of packages that were pruned
        """
        pruned = []
        packages_to_remove = set()
        
        # Check each package in the input list
        for package in packages_to_check:
            if package not in self.packages:
                continue  # Already removed or not in manager
            
            key = self._package_key(package)
            if key not in self.bounds:
                continue  # No bounds for this package
            
            lb, ub = self.bounds[key]
            
            # Check against all other packages in the manager
            for other_package in self.packages:
                if package == other_package:
                    continue  # Skip self
                
                other_key = self._package_key(other_package)
                if other_key not in self.bounds:
                    continue
                
                other_lb, other_ub = self.bounds[other_key]
                
                # Case 1: This package's upper bound <= other's lower bound
                # This package can never be better, prune it
                if ub <= other_lb:
                    packages_to_remove.add(key)
                    pruned.append(package)
                    break  # No need to check further for this package
                
                # Case 2: Other package's lower bound >= this package's upper bound
                # Other package can never be better, prune it
                if other_lb >= ub:
                    packages_to_remove.add(other_key)
                    if other_package not in pruned:
                        pruned.append(other_package)
        
        # Remove pruned packages from manager
        self.packages = [pkg for pkg in self.packages if self._package_key(pkg) not in packages_to_remove]
        
        # Also remove their bounds
        for key in packages_to_remove:
            self.bounds.pop(key, None)
        
        return pruned
    
    def _compute_prob_package_exceeds(self, lb_p: float, ub_p: float, lb_other: float, ub_other: float) -> float:
        """
        Compute probability that package p (with bounds [lb_p, ub_p]) >= package other (with bounds [lb_other, ub_other]).
        
        Since scores are discrete (1 decimal place), we enumerate all possible values.
        
        Args:
            lb_p: Lower bound of package p
            ub_p: Upper bound of package p
            lb_other: Lower bound of other package
            ub_other: Upper bound of other package
            
        Returns:
            Probability that p >= other
        """
        # Generate all possible discrete values (1 decimal place)
        p_values = []
        current = lb_p
        while current <= ub_p + 0.05:  # Add small epsilon for floating point
            p_values.append(round(current, 1))
            current += 0.1
        
        other_values = []
        current = lb_other
        while current <= ub_other + 0.05:  # Add small epsilon for floating point
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
        
        result = favorable_cases / total_cases 
        return result
    
    def check_alpha_top_package(self, package: Package, alpha: float) -> bool:
        """
        Check if the probability of a package being the best (top) package is >= alpha.
        
        Probability = product of probabilities that this package >= each other package.
        
        Args:
            package: The package to check
            alpha: Probability threshold in [0, 1]
            
        Returns:
            True if probability >= alpha, False otherwise
        """
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError(f"Alpha must be in [0, 1], got {alpha}")
        
        key = self._package_key(package)
        if key not in self.bounds:
            return False  # Package not in manager or has no bounds
        
        lb_p, ub_p = self.bounds[key]
        
        # If package has zero-width interval and is not in manager, return False
        if package not in self.packages:
            return False
        
        # If no other packages, this is definitely the best
        other_packages = [p for p in self.packages if p != package]
        if not other_packages:
            return True  # Only one package, definitely best
        
        # Compute probability that this package >= each other package
        probabilities = []
        for other_package in other_packages:
            other_key = self._package_key(other_package)
            if other_key not in self.bounds:
                continue
            
            other_lb, other_ub = self.bounds[other_key]
            
            # Compute prob(p >= other)
            prob_exceeds = self._compute_prob_package_exceeds(lb_p, ub_p, other_lb, other_ub)
            probabilities.append(prob_exceeds)
        
        if not probabilities:
            return False
        
        # Total probability = product of all individual probabilities
        # (probability that p >= all other packages)
        total_probability = 1.0
        for prob in probabilities:
            total_probability *= prob
        
        return total_probability >= alpha
    
    def _get_questions_for_package(self, package: Package) -> Set[Tuple[str, Tuple[str, ...]]]:
        """
        Get the set of all questions (component, entity_ids) needed for a package.
        
        F(P, q) - the set of component value questions needed to compute package P's score.
        
        Args:
            package: The package
            
        Returns:
            Set of (component_name, sorted_entity_ids_tuple) tuples
        """
        questions = set()
        entity_list = list(package.entities)
        
        for component in self.components:
            if component.dimension == 1:
                # Unary: one question per entity
                for entity_id in entity_list:
                    sorted_ids = tuple(sorted([entity_id]))
                    questions.add((component.name, sorted_ids))
            elif component.dimension == 2:
                # Binary: one question per pair
                for i in range(len(entity_list)):
                    for j in range(i + 1, len(entity_list)):
                        sorted_ids = tuple(sorted([entity_list[i], entity_list[j]]))
                        questions.add((component.name, sorted_ids))
        
        return questions
    
    def coverage(
        self,
        component: Component,
        entity_ids: List[str],
        package: Package,
        reference_package: Package
    ) -> bool:
        """
        Check if a question covers a package according to the coverage definition.
        
        Cover(Q, P) = {
            1[Q in F(P*, q)] if P = P*,
            1[Q in F(P, q)] XOR 1[Q in F(P*, q)] otherwise
        }
        
        where F(P, q) is the set of questions needed for package P.
        
        Args:
            component: The component of the question
            entity_ids: The entity IDs of the question
            package: The package P to check coverage for
            reference_package: The reference package P*
            
        Returns:
            True if question covers the package, False otherwise
        """
        # Create question representation (component name, sorted entity IDs)
        sorted_entity_ids = tuple(sorted(entity_ids))
        question = (component.name, sorted_entity_ids)
        
        # Case 1: P = P*
        if package == reference_package:
            questions_for_p_star = self._get_questions_for_package(reference_package)
            return question in questions_for_p_star
        
        # Case 2: P != P*
        # Coverage = 1[Q in F(P, q)] XOR 1[Q in F(P*, q)]
        questions_for_p = self._get_questions_for_package(package)
        questions_for_p_star = self._get_questions_for_package(reference_package)
        
        in_f_p = question in questions_for_p
        in_f_p_star = question in questions_for_p_star
        
        # XOR: True if exactly one is True
        return in_f_p != in_f_p_star