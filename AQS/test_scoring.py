"""Test file for AQS scoring function with interval computation."""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import Entity, Package, Component
from llm_interface import LLMEvaluator
from AQS.scoring import ScoringFunction


def test_scoring():
    """Test AQS scoring function with known and unknown component values."""
    # Sample entities
    entities = {
        'e1': Entity(id='e1', name='Entity 1', data='Great product with excellent features'),
        'e2': Entity(id='e2', name='Entity 2', data='Good quality and fast delivery'),
        'e3': Entity(id='e3', name='Entity 3', data='Average product, could be better'),
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
    
    # Create package of size 3
    package = Package(entities={'e1', 'e2', 'e3'})
    query = "Find products with good quality"
    
    # Initialize
    llm_evaluator = LLMEvaluator(mock_api=False)
    scoring_function = ScoringFunction(components, llm_evaluator)
    
    print(f"Test Parameters:")
    print(f"  Package size: {package.size()}")
    print(f"  Package entities: {list(package.entities)}")
    print(f"  Components: {[c.name for c in components]}")
    print(f"  Query: {query}")
    print()
    
    # Package of size 3 with relevance (unary) and diversity (binary):
    # - Relevance: 3 component values (one per entity)
    # - Diversity: 3 choose 2 = 3 component values (one per pair)
    # Total: 6 component values
    
    # Compute some component values (known)
    print("Computing some component values...")
    
    # Compute 2 relevance values (now returns intervals)
    rel_e1_lb, rel_e1_ub = scoring_function.probe_question(
        components[0], entities, ['e1'], query, use_cache=True
    )
    rel_e2_lb, rel_e2_ub = scoring_function.probe_question(
        components[0], entities, ['e2'], query, use_cache=True
    )
    print(f"  Known - Rel(q, e1): [{rel_e1_lb}, {rel_e1_ub}]")
    print(f"  Known - Rel(q, e2): [{rel_e2_lb}, {rel_e2_ub}]")
    
    # Compute 2 diversity values (now returns intervals)
    div_e1_e2_lb, div_e1_e2_ub = scoring_function.probe_question(
        components[1], entities, ['e1', 'e2'], query, use_cache=True
    )
    div_e1_e3_lb, div_e1_e3_ub = scoring_function.probe_question(
        components[1], entities, ['e1', 'e3'], query, use_cache=True
    )
    print(f"  Known - Div(e1, e2): [{div_e1_e2_lb}, {div_e1_e2_ub}]")
    print(f"  Known - Div(e1, e3): [{div_e1_e3_lb}, {div_e1_e3_ub}]")
    print()
    
    # Now we have 4 known values, 2 unknown values:
    # Unknown: Rel(q, e3) and Div(e2, e3)
    
    # Compute score interval
    lower_bound, upper_bound = scoring_function.compute_package_score_interval(
        package, entities, query, use_cache=True
    )

    
    print(f"Score Interval:")
    print(f"  Lower bound: {lower_bound}")
    print(f"  Upper bound: {upper_bound}")
    print(f"  Interval: [{lower_bound}, {upper_bound}]")
    print()
    
    # Verify: lower = sum of lower bounds of known + 0*2, upper = sum of upper bounds of known + 1*2
    known_sum_lb = rel_e1_lb + rel_e2_lb + div_e1_e2_lb + div_e1_e3_lb
    known_sum_ub = rel_e1_ub + rel_e2_ub + div_e1_e2_ub + div_e1_e3_ub
    print(f"Verification:")
    print(f"  Sum of lower bounds of known values: {known_sum_lb}")
    print(f"  Sum of upper bounds of known values: {known_sum_ub}")
    print(f"  Lower bound (known_lb + 0*2 unknown): {known_sum_lb + 0.0 * 2} = {lower_bound}")
    print(f"  Upper bound (known_ub + 1*2 unknown): {known_sum_ub + 1.0 * 2} = {upper_bound}")
    print()
    
    # Note: With interval-based component values, package scores are always intervals
    # Even when all component values are computed, the final score remains an interval
    
    return (lower_bound, upper_bound)


if __name__ == "__main__":
    test_scoring()
