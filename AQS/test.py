"""Test for simulate_question method."""
import sys
from pathlib import Path
import copy

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import Entity, Component, Package
from llm_interface import LLMEvaluator
from AQS.algorithm import AQSAlgorithm


def test_simulate_question():
    """Test simulate_question with 10 packages, 5 entities, k=3."""
    print("=" * 60)
    print("Test: simulate_question")
    print("=" * 60)
    
    entities = {
        'e1': Entity(id='e1', name='Entity 1', data='data1'),
        'e2': Entity(id='e2', name='Entity 2', data='data2'),
        'e3': Entity(id='e3', name='Entity 3', data='data3'),
        'e4': Entity(id='e4', name='Entity 4', data='data4'),
        'e5': Entity(id='e5', name='Entity 5', data='data5'),
    }
    
    components = [
        Component(name='relevance', description='Relevance', dimension=1),
        Component(name='diversity', description='Diversity', dimension=2)
    ]
    
    k = 3
    llm_evaluator = LLMEvaluator(mock_api=True)
    algorithm = AQSAlgorithm(
        entities=entities,
        components=components,
        k=k,
        alpha=0.8,
        query="test query",
        llm_evaluator=llm_evaluator
    )
    
    # Get all packages (should be C(5,3) = 10 packages)
    packages = algorithm.package_manager.get_packages()
    print(f"Total packages: {len(packages)}")
    print(f"Entities: {list(entities.keys())}")
    print(f"k: {k}")
    print()
    
    # Set initial bounds for packages
    print("Setting initial bounds for packages:")
    for i, pkg in enumerate(packages):
        # Set bounds with some variation: (i*0.5, i*0.5 + 2.0)
        lb = i * 0.5
        ub = lb + 2.0
        algorithm.package_manager.set_bounds(pkg, lb, ub)
        print(f"  Package {i}: {list(pkg.entities)} -> bounds=({lb:.1f}, {ub:.1f})")
    print()
    
    # Identify super-candidate
    super_candidate = algorithm.identify_current_super_candidate()
    if super_candidate:
        sc_lb, sc_ub = algorithm.package_manager.get_bounds(super_candidate)
        sc_midpoint = (sc_lb + sc_ub) / 2.0
        print(f"Super-candidate: {list(super_candidate.entities)}")
        print(f"  Bounds: ({sc_lb:.1f}, {sc_ub:.1f}), midpoint: {sc_midpoint:.1f}")
        print()
    else:
        print("No super-candidate found!")
        return
    
    # Create copies for simulation
    bounds_copy = copy.deepcopy(algorithm.package_manager.bounds)
    packages_copy = copy.deepcopy(algorithm.package_manager.packages)
    
    print(f"Before simulation:")
    print(f"  Number of packages: {len(packages_copy)}")
    print(f"  Number of bounds: {len(bounds_copy)}")
    print()
    
    # Choose a question to simulate
    # Let's use a relevance question for e1 (unary component)
    question = ('relevance', ('e1',))
    print(f"Simulating question: {question}")
    print(f"  Component: relevance (unary)")
    print(f"  Entity: e1")
    print()
    
    # Check if question affects super-candidate
    questions_for_p_star = algorithm.package_manager._get_questions_for_package(super_candidate)
    affects_super = question in questions_for_p_star
    print(f"Question affects super-candidate: {affects_super}")
    if affects_super:
        print("  -> Will simulate best-case response: (1.0, 1.0)")
    else:
        print("  -> Will simulate best-case response: (0.0, 0.0)")
    print()
    
    # Simulate the question
    updated_bounds, updated_packages = algorithm.simulate_question(
        question, super_candidate, bounds_copy, packages_copy
    )
    
    print(f"After simulation:")
    print(f"  Number of packages: {len(updated_packages)}")
    print(f"  Number of bounds: {len(updated_bounds)}")
    print(f"  Packages pruned: {len(packages_copy) - len(updated_packages)}")
    print()
    
    # Show affected packages and their updated bounds
    print("Updated bounds for affected packages:")
    affected_count = 0
    for pkg in updated_packages:
        key = algorithm.package_manager._package_key(pkg)
        if key in updated_bounds:
            # Check if this package was affected by the question
            if algorithm.package_manager._is_package_affected(pkg, ['e1']):
                affected_count += 1
                old_lb, old_ub = bounds_copy.get(key, (0.0, 0.0))
                new_lb, new_ub = updated_bounds[key]
                print(f"  Package {list(pkg.entities)}:")
                print(f"    Old bounds: ({old_lb:.1f}, {old_ub:.1f})")
                print(f"    New bounds: ({new_lb:.1f}, {new_ub:.1f})")
    
    print(f"\nTotal affected packages: {affected_count}")
    print()
    
    # Show pruned packages
    pruned_keys = set(bounds_copy.keys()) - set(updated_bounds.keys())
    if pruned_keys:
        print("Pruned packages:")
        for key in pruned_keys:
            # Find the package
            for pkg in packages_copy:
                if algorithm.package_manager._package_key(pkg) == key:
                    print(f"  {list(pkg.entities)}")
                    break
    else:
        print("No packages were pruned.")
    print()
    
    print("=" * 60)
    print("Test completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_simulate_question()
