"""Test file: Test data model and scoring function calculation."""
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.models import Entity, Package, Component
from utils.llm_interface import LLMEvaluator
from PCS.scoring import ScoringFunction


def test_scoring(n, k, package, entities, components=None, query="Find products with good quality"):
    """
    Test scoring function with sample data.

    Args:
        n: Number of entities
        k: Package size
        package: Package object containing entity IDs
        entities: Dictionary mapping entity_id to Entity
        components: List of Component objects (default: relevance + diversity)
        query: User query string
    """
    # Default components if not provided
    if components is None:
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

    # Initialize LLM evaluator (requires OPENAI_API_KEY in environment)
    llm_evaluator = LLMEvaluator()
    scoring_function = ScoringFunction(components, llm_evaluator)

    print(f"Test Parameters:")
    print(f"  k (package size): {k}")
    print(f"  n (number of entities): {n}")
    print(f"  Package: {list(package.entities)}")
    print(f"  Components: {[c.name for c in components]}")
    print(f"  Query: {query}")
    print()

    # Compute package score
    total_score = scoring_function.compute_package_score(
        package, entities, query, use_cache=True
    )

    print(f"Result:")
    print(f"  Total Package Score: {total_score}")
    print()

    # Show contribution of each entity
    print("Entity Contributions:")
    for entity_id in package.entities:
        contrib = scoring_function.compute_contribution(
            entity_id, package, entities, query, use_cache=True
        )
        print(f"  {entity_id}: {contrib}")
    print()

    # Print all component values (from cache; no extra LLM calls)
    print("Component values (cached):")
    entity_list = list(package.entities)
    for component in components:
        if component.dimension == 1:
            for entity_id in entity_list:
                value = scoring_function.probe_question(
                    component, entities, [entity_id], query, use_cache=True
                )
                print(f"  {component.name}({entity_id}): {value}")
        elif component.dimension == 2:
            for i in range(len(entity_list)):
                for j in range(i + 1, len(entity_list)):
                    pair = [entity_list[i], entity_list[j]]
                    value = scoring_function.probe_question(
                        component, entities, pair, query, use_cache=True
                    )
                    print(f"  {component.name}({entity_list[i]}, {entity_list[j]}): {value}")
    print()

    return total_score


if __name__ == "__main__":
    # Sample entities
    entities = {
        'e1': Entity(id='e1', name='Entity 1', data='Great product with excellent features'),
        'e2': Entity(id='e2', name='Entity 2', data='Good quality and fast delivery'),
        'e3': Entity(id='e3', name='Entity 3', data='Average product, could be better'),
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
    n = len(entities)
    k = 3
    package = Package(entities=entities.keys())
    query = "Find products with good quality"

    test_scoring(n, k, package, entities, components, query)
