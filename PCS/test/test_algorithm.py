"""Test file for PCS algorithm with logging."""
import json
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.models import Package, Component
from utils.llm_interface import LLMEvaluator
from preprocessing.load_data import load_entities_from_csv
from BASELINE.algorithm import ExactBaseline
from PCS.algorithm import PCSAlgorithm


def test_algorithm():
    """Test PCS algorithm with sample data."""
    csv_path = Path(__file__).parent.parent.parent / "data" / "sample_data.csv"
    entities = dict(list(load_entities_from_csv(str(csv_path)).items())[:10])


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
    epsilon = 0
    # Initial package: False = random k entities, True = top-k by unary scores
    smart_initial_package = True

    print(f"Test Parameters:")
    print(f"  k (package size): {k}")
    print(f"  n (number of entities): {n}")
    print(f"  Query: {query}")
    print(f"  Budget rate (s): {budget_rate}")
    print(f"  Epsilon: {epsilon}")
    print(f"  Smart initial package: {smart_initial_package}")
    print(f"  Components: {[c.name for c in components]}")
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

    # Initialize
    algorithm = PCSAlgorithm(
        components=components,
        llm_evaluator=llm_evaluator,
        budget_rate=budget_rate,
        epsilon=epsilon,
        smart_initial_package=smart_initial_package
    )

    # Run algorithm
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
    import os
    outputs_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)
    log_filename = os.path.join(outputs_dir, f"pcs_algorithm_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
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
