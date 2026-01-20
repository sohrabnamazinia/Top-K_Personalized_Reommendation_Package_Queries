"""Test file for PackageManager."""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import Entity, Package, Component
from AQS.package_manager import PackageManager


def test_package_manager():
    """Test PackageManager with all its methods."""
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
    entity_ids = list(entities.keys())
    
    print("=" * 60)
    print("Testing PackageManager")
    print("=" * 60)
    print(f"Entities: {entity_ids}")
    print(f"k (package size): {k}")
    print(f"Components: {[c.name for c in components]}")
    print()
    
    # Test 1: Initialize PackageManager (builds all packages)
    print("Test 1: Initialize PackageManager")
    package_manager = PackageManager(
        entities=entity_ids,
        k=k,
        components=components,
        packages=None  # Will build all packages
    )
    print(f"  Number of packages created: {package_manager.get_package_count()}")
    print(f"  Expected: C({len(entity_ids)}, {k}) = {len(entity_ids) * (len(entity_ids) - 1) * (len(entity_ids) - 2) // 6}")
    print()
    
    # Test 2: Get bounds for a package
    print("Test 2: Get initial bounds for a package")
    test_package = package_manager.get_packages()[0]
    lb, ub = package_manager.get_bounds(test_package)
    print(f"  Package: {list(test_package.entities)}")
    print(f"  Bounds: [{lb}, {ub}]")
    print(f"  Lower bound (should be 0.0): {lb}")
    print(f"  Upper bound (should be max score): {ub}")
    print()
    
    # Test 3: Set bounds
    print("Test 3: Set bounds for a package")
    package_manager.set_bounds(test_package, 2.5, 5.0)
    lb, ub = package_manager.get_bounds(test_package)
    print(f"  Updated bounds: [{lb}, {ub}]")
    print()
    
    # Test 4: Update bounds with a question response
    print("Test 4: Update bounds with question response")
    component = components[0]  # relevance
    entity_ids_question = ['e1']
    response = (0.7, 0.9)  # (lower, upper)
    
    affected = package_manager.update_bounds(component, entity_ids_question, response)
    print(f"  Question: {component.name} for entities {entity_ids_question}")
    print(f"  Response: {response}")
    print(f"  Number of affected packages: {len(affected)}")
    if affected:
        sample_affected = affected[0]
        lb, ub = package_manager.get_bounds(sample_affected)
        print(f"  Sample affected package: {list(sample_affected.entities)}")
        print(f"  Updated bounds: [{lb}, {ub}]")
    print()
    
    # Test 5: Check if package is affected
    print("Test 5: Check if package is affected by question")
    test_pkg = package_manager.get_packages()[0]
    is_affected = package_manager._is_package_affected(test_pkg, ['e1'])
    print(f"  Package: {list(test_pkg.entities)}")
    print(f"  Question entities: ['e1']")
    print(f"  Is affected: {is_affected}")
    print()
    
    # Test 6: Prune packages
    print("Test 6: Prune dominated packages")
    packages_before = package_manager.get_package_count()
    
    # Create a package with very low bounds to test pruning
    low_package = Package(entities={'e1', 'e2', 'e3'})
    package_manager.set_bounds(low_package, 0.0, 1.0)
    
    # Create a package with high bounds
    high_package = Package(entities={'e4', 'e5', 'e1'})
    package_manager.set_bounds(high_package, 10.0, 15.0)
    
    # Add these to packages if not already there
    if low_package not in package_manager.get_packages():
        package_manager.packages.append(low_package)
    if high_package not in package_manager.get_packages():
        package_manager.packages.append(high_package)
    
    pruned = package_manager.prune_packages([low_package])
    packages_after = package_manager.get_package_count()
    print(f"  Packages before pruning: {packages_before}")
    print(f"  Packages after pruning: {packages_after}")
    print(f"  Packages pruned: {len(pruned)}")
    print()
    
    # Test 7: Check alpha-top package
    print("Test 7: Check alpha-top package")
    # Set up a package with high bounds
    candidate_package = package_manager.get_packages()[0]
    package_manager.set_bounds(candidate_package, 8.0, 10.0)
    
    # Set other packages with lower bounds
    for pkg in package_manager.get_packages()[1:5]:  # Just test a few
        package_manager.set_bounds(pkg, 2.0, 4.0)
    
    is_alpha_top = package_manager.check_alpha_top_package(candidate_package, alpha=0.8)
    lb, ub = package_manager.get_bounds(candidate_package)
    print(f"  Candidate package: {list(candidate_package.entities)}")
    print(f"  Bounds: [{lb}, {ub}]")
    print(f"  Alpha: 0.8")
    print(f"  Is alpha-top: {is_alpha_top}")
    print()
    
    # Test 8: Coverage
    print("Test 8: Check coverage")
    reference_package = package_manager.get_packages()[0]
    test_package = package_manager.get_packages()[1] if len(package_manager.get_packages()) > 1 else reference_package
    
    component = components[0]  # relevance
    entity_ids_q = ['e1']
    
    covers = package_manager.coverage(component, entity_ids_q, test_package, reference_package)
    print(f"  Question: {component.name} for entities {entity_ids_q}")
    print(f"  Package: {list(test_package.entities)}")
    print(f"  Reference package: {list(reference_package.entities)}")
    print(f"  Covers: {covers}")
    print()
    
    print("=" * 60)
    print("All PackageManager tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_package_manager()
