"""Test file for AQS algorithm."""
import sys
from pathlib import Path
import json
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import Entity, Component
from llm_interface import LLMEvaluator
from AQS.algorithm import AQSAlgorithm


def test_algorithm():
    """Test AQS algorithm with sample data."""
    # Sample entities
    entities = {
        'e1': Entity(id='e1', name='Entity 1', data='Great product with excellent features'),
        'e2': Entity(id='e2', name='Entity 2', data='Good quality and fast delivery'),
        'e3': Entity(id='e3', name='Entity 3', data='Average product, could be better'),
        'e4': Entity(id='e4', name='Entity 4', data='Outstanding service and support'),
        'e5': Entity(id='e5', name='Entity 5', data='Decent product at reasonable price'),
    }
    
    # Components
    components = [
        Component(
            name='relevance',
            description='Relevance of the entity with the user query',
            dimension=1
        ),
        Component(
            name='diversity',
            description='Diversity between two entities',
            dimension=2
        )
    ]
    
    # Test parameters
    k = 3
    alpha = 0.8
    query = "Find products with good quality"
    
    print("=" * 60)
    print("Testing AQS Algorithm")
    print("=" * 60)
    print(f"Entities: {list(entities.keys())}")
    print(f"k (package size): {k}")
    print(f"Alpha: {alpha}")
    print(f"Query: {query}")
    print(f"Components: {[c.name for c in components]}")
    print()
    
    # Initialize LLM evaluator (use mock for testing)
    llm_evaluator = LLMEvaluator(mock_api=True)
    
    # Initialize algorithm
    print("Initializing AQS algorithm...")
    algorithm = AQSAlgorithm(
        entities=entities,
        components=components,
        k=k,
        alpha=alpha,
        query=query,
        llm_evaluator=llm_evaluator,
        initial_packages=None  # Will build all possible packages
    )
    
    print(f"  Initial packages: {algorithm.package_manager.get_package_count()}")
    print(f"  Unknown questions: {len(algorithm.unknown_questions)}")
    print()
    
    # Run algorithm
    print("Running AQS algorithm...")
    print("-" * 60)
    
    final_package, metadata = algorithm.run()
    
    # Display results
    print()
    print("=" * 60)
    print("Algorithm Results")
    print("=" * 60)
    print(f"Final Package: {list(final_package.entities) if final_package else 'None'}")
    if final_package:
        lb, ub = algorithm.package_manager.get_bounds(final_package)
        print(f"Final Bounds: [{lb}, {ub}]")
    print(f"Questions Asked: {metadata['questions_asked']}")
    print(f"Number of Iterations: {len(metadata['iterations'])}")
    print()
    
    # Print iteration details
    print("Iteration Details:")
    for iter_info in metadata['iterations']:
        print(f"  Iteration {iter_info['iteration']}:")
        print(f"    Super-candidate: {iter_info.get('super_candidate', 'N/A')}")
        print(f"    Bounds: {iter_info.get('super_candidate_bounds', 'N/A')}")
        print(f"    Alpha-top condition met: {iter_info.get('alpha_top_condition_met', 'N/A')}")
        if 'selected_question' in iter_info:
            print(f"    Selected question: {iter_info['selected_question']}")
            print(f"    Response: {iter_info.get('response', 'N/A')}")
            print(f"    Affected packages: {iter_info.get('affected_packages', 'N/A')}")
            print(f"    Pruned packages: {iter_info.get('pruned_packages', 'N/A')}")
        print(f"    Remaining unknown questions: {iter_info.get('remaining_unknown_questions', 'N/A')}")
        print()
    
    # Save results to file
    log_filename = f"aqs_algorithm_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    log_data = {
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'k': k,
            'alpha': alpha,
            'query': query,
            'num_entities': len(entities),
            'components': [{'name': c.name, 'dimension': c.dimension} for c in components]
        },
        'result': metadata
    }
    
    with open(log_filename, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    print(f"Detailed log saved to: {log_filename}")
    print("=" * 60)


if __name__ == "__main__":
    test_algorithm()
