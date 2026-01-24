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
        'e1': Entity(id='e1', name='Product A', data='Great product with excellent features and high quality'),
        'e2': Entity(id='e2', name='Product B', data='Good quality and fast delivery service'),
        'e3': Entity(id='e3', name='Product C', data='Average product, could be better but affordable'),
        'e4': Entity(id='e4', name='Product D', data='Outstanding service and customer support'),
        'e5': Entity(id='e5', name='Product E', data='Decent product at reasonable price point'),
        # 'e6': Entity(id='e6', name='Product F', data='Premium product with advanced features and warranty'),
        # 'e7': Entity(id='e7', name='Product G', data='Budget-friendly option with basic functionality'),
        # 'e8': Entity(id='e8', name='Product H', data='Innovative design with modern technology integration'),
        # 'e9': Entity(id='e9', name='Product I', data='Reliable product with consistent performance'),
        # 'e10': Entity(id='e10', name='Product J', data='Versatile product suitable for multiple use cases'),
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
    alpha = 0.05
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
        initial_packages=None,  # Will build all possible packages
        print_log=True  # Enable detailed logging during execution
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
