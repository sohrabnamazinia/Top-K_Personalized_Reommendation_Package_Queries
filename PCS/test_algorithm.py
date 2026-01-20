"""Test file 2: Test full AQA algorithm with logging."""
import json
from datetime import datetime
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import Entity, Package, Component
from llm_interface import LLMEvaluator
from algorithm import AQAAlgorithm


def test_algorithm():
    """Test AQA algorithm with sample data."""
    # Sample entities
    entities = {
        'e1': Entity(id='e1', name='Entity 1', data='Great product with excellent features'),
        'e2': Entity(id='e2', name='Entity 2', data='Good quality and fast delivery'),
        'e3': Entity(id='e3', name='Entity 3', data='Average product, could be better'),
        'e4': Entity(id='e4', name='Entity 4', data='Outstanding service and support'),
        'e5': Entity(id='e5', name='Entity 5', data='Decent product at reasonable price'),
    }
    
    # Sample components
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
    n = len(entities)
    query = "Find products with good quality"
    budget_rate = 2
    epsilon = 0.01
    
    # Optional: provide initial packages (list of Package objects)
    # If None, algorithm will compute initial package
    initial_packages = None
    # Example: initial_packages = [Package(entities={'e1', 'e2', 'e3'})]
    
    print(f"Test Parameters:")
    print(f"  k (package size): {k}")
    print(f"  n (number of entities): {n}")
    print(f"  Query: {query}")
    print(f"  Budget rate (s): {budget_rate}")
    print(f"  Epsilon: {epsilon}")
    print(f"  Components: {[c.name for c in components]}")
    print()
    
    # Initialize
    llm_evaluator = LLMEvaluator()
    algorithm = AQAAlgorithm(
        components=components,
        llm_evaluator=llm_evaluator,
        budget_rate=budget_rate,
        epsilon=epsilon
    )
    
    # Run algorithm
    if initial_packages:
        # If multiple initial packages provided, run for each
        results = []
        for init_pkg in initial_packages:
            final_package, metadata = algorithm.run(
                entities, k, query, initial_package=init_pkg
            )
            results.append((final_package, metadata))
        
        # Use first result for logging (or log all)
        final_package, metadata = results[0]
    else:
        final_package, metadata = algorithm.run(entities, k, query)
    
    # Log results
    log_data = {
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'k': k,
            'n': n,
            'query': query,
            'budget_rate': budget_rate,
            'epsilon': epsilon,
            'components': [{'name': c.name, 'description': c.description, 'dimension': c.dimension} 
                          for c in components]
        },
        'result': {
            'final_package': list(final_package.entities),
            'final_score': metadata['final_score'],
            'iterations': metadata['iterations']
        }
    }
    
    # Write to log file
    log_filename = f"algorithm_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_filename, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    print(f"Algorithm Results:")
    print(f"  Final Package: {list(final_package.entities)}")
    print(f"  Final Score: {metadata['final_score']}")
    print(f"  Number of Iterations: {len(metadata['iterations'])}")
    print()
    print(f"Detailed log saved to: {log_filename}")
    print()
    
    # Print iteration details
    print("Iteration Details:")
    for iter_info in metadata['iterations']:
        print(f"  Iteration {iter_info['iteration']}:")
        print(f"    e⁻ (min contribution entity): {iter_info.get('e_minus', 'N/A')}")
        print(f"    δ_min: {iter_info.get('delta_min', 'N/A')}")
        print(f"    Max TP: {iter_info.get('max_tp', 'N/A')}")
        print(f"    Top-s candidates: {iter_info.get('top_s_candidates', [])}")
        if 'e_star' in iter_info:
            print(f"    e* (selected): {iter_info['e_star']}")
            print(f"    Max contribution: {iter_info.get('max_contrib_value', 'N/A')}")
        print(f"    Stop reason: {iter_info.get('stop_reason', 'N/A')}")
        print()
    
    return final_package, metadata


if __name__ == "__main__":
    test_algorithm()

