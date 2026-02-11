"""Test file for AQS algorithm."""
import sys
from pathlib import Path
import json
from datetime import datetime

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.models import Entity, Component
from utils.llm_interface import LLMEvaluator
from preprocessing.load_data import load_entities_from_csv
from BASELINE.algorithm import ExactBaseline
from AQS.algorithm import AQSAlgorithm


def test_algorithm():
    """Test AQS algorithm with sample data."""
    csv_path = Path(__file__).parent.parent.parent / "data" / "sample_data.csv"
    entities = dict(list(load_entities_from_csv(str(csv_path)).items())[:10])

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

    # Exact baseline
    llm_evaluator = LLMEvaluator(mock_api=False, use_MGT=True, components=components, entities_csv_path=csv_path)
    baseline = ExactBaseline(components, llm_evaluator)
    baseline_package, baseline_scores = baseline.run(entities, query, k)
    print("Exact baseline:")
    print(f"  Best package: {sorted(baseline_package.entities)}")
    print(f"  Best score (midpoint): {baseline_scores[0] if baseline_scores else 'N/A'}")
    print(f"  All scores (top 5): {baseline_scores[:5]}")
    print()

    # Initialize algorithm
    print("Initializing AQS algorithm...")
    algorithm = AQSAlgorithm(
        entities=entities,
        components=components,
        k=k,
        alpha=alpha,
        query=query,
        llm_evaluator=llm_evaluator,
        initial_packages=None,  # Will build all possible packages
        print_log=True,  # Enable detailed logging during execution
        is_next_q_random=False,
    )
    
    print(f"  Initial packages: {algorithm.package_manager.get_package_count()}")
    print(f"  Unknown questions: {len(algorithm.unknown_questions)}")
    print()
    
    # Run algorithm
    print("Running AQS algorithm...")
    print("-" * 60)
    print()
    
    final_package, metadata = algorithm.run()
    
    # Display final results summary
    print()
    print("=" * 60)
    print("Algorithm Results Summary")
    print("=" * 60)
    print(f"Final Package: {list(final_package.entities) if final_package else 'None'}")
    if final_package:
        lb, ub = algorithm.package_manager.get_bounds(final_package)
        print(f"Final Bounds: [{lb}, {ub}]")
    print(f"Questions Asked: {metadata['questions_asked']}")
    print(f"Number of Iterations: {len(metadata['iterations'])}")
    print()
    
    # Save results to file
    import os
    outputs_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)
    
    log_filename = os.path.join(outputs_dir, f"aqs_algorithm_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
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
