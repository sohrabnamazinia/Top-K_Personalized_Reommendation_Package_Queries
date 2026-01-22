"""Simple test scenario: 5 entities, 3 packages, ask 3 questions and update bounds."""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import Entity, Package, Component
from llm_interface import LLMEvaluator
from package_manager import PackageManager
from scoring import ScoringFunction


def test_scenario():
    """Test scenario: 5 entities, 3 packages, ask 3 questions."""
    # Create 5 entities
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
    
    # Create 3 specific packages
    package1 = Package(entities={'e1', 'e2', 'e3'})
    package2 = Package(entities={'e1', 'e2', 'e4'})
    package3 = Package(entities={'e3', 'e4', 'e5'})
    
    packages = [package1, package2, package3]
    
    # Test parameters
    k = 3
    query = "Find products with good quality"
    
    print("=" * 60)
    print("Test Scenario: 5 entities, 3 packages, 3 questions")
    print("=" * 60)
    print(f"Entities: {list(entities.keys())}")
    print(f"Packages:")
    print(f"  Package 1: {list(package1.entities)}")
    print(f"  Package 2: {list(package2.entities)}")
    print(f"  Package 3: {list(package3.entities)}")
    print(f"Query: {query}")
    print()
    
    # Initialize LLM evaluator (use mock for testing)
    llm_evaluator = LLMEvaluator(mock_api=True)
    
    # Initialize scoring function
    scoring_function = ScoringFunction(components, llm_evaluator, entities, query)
    
    # Initialize package manager with the 3 packages
    entity_ids = list(entities.keys())
    package_manager = PackageManager(
        entities=entity_ids,
        k=k,
        components=components,
        packages=packages
    )
    
    # Print initial bounds
    print("Initial Bounds:")
    print("-" * 60)
    for i, package in enumerate(packages, 1):
        lb, ub = package_manager.get_bounds(package)
        print(f"Package {i} {list(package.entities)}: [{lb}, {ub}]")
    print()
    
    # Define 3 questions to ask
    questions = [
        (components[0], ['e1']),  # Question 1: relevance of e1
        (components[0], ['e2']),  # Question 2: relevance of e2
        (components[1], ['e1', 'e2']),  # Question 3: diversity between e1 and e2
    ]
    
    # Ask each question and update bounds
    for question_num, (component, entity_ids) in enumerate(questions, 1):
        print(f"Question {question_num}: {component.name} for entities {entity_ids}")
        print("-" * 60)
        
        # Probe LLM for response
        response = scoring_function.probe_question(
            component, entities, entity_ids, query, use_cache=True
        )
        print(f"  Response: [{response[0]}, {response[1]}]")
        
        # Update bounds
        affected_packages = package_manager.update_bounds(component, entity_ids, response)
        print(f"  Affected packages: {len(affected_packages)}")
        
        # Print bounds after update
        print(f"\nBounds after Question {question_num}:")
        for i, package in enumerate(packages, 1):
            lb, ub = package_manager.get_bounds(package)
            print(f"  Package {i} {list(package.entities)}: [{lb}, {ub}]")
        print()
    
    print("=" * 60)
    print("Test completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_scenario()
