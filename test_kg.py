#!/usr/bin/env python3
"""
Comprehensive test suite for hashable knowledge_graph.py

Tests:
1. Entity hashability
2. Relationship hashability
3. Set operations
4. Dictionary operations
5. Graph database operations with hashable entities
6. No functionality regression
"""

import sys
from pathlib import Path

# Import the knowledge graph module
from knowledge_graph import (
    Entity, Relationship, EntityType, RelationType, KnowledgeGraph
)


def test_entity_hashability():
    """Test Entity class hashability"""
    print("\n" + "="*70)
    print("TEST 1: Entity Hashability")
    print("="*70)
    
    # Create test entities
    e1 = Entity(
        id="proc_1",
        type=EntityType.PROCEDURE,
        name="calculate_payment",
        qualified_name="payments::calculate_payment",
        file_path="payments.tal",
        start_line=100,
        metadata={"params": 3}
    )
    
    e2 = Entity(
        id="proc_1",  # Same ID as e1
        type=EntityType.PROCEDURE,
        name="calculate_payment",
        qualified_name="payments::calculate_payment",
        file_path="different_path.tal",  # Different path
        metadata={"params": 5}  # Different metadata
    )
    
    e3 = Entity(
        id="proc_2",  # Different ID
        type=EntityType.PROCEDURE,
        name="validate_account",
        qualified_name="validation::validate_account"
    )
    
    # Test 1.1: Hashing works
    try:
        h1 = hash(e1)
        h2 = hash(e2)
        h3 = hash(e3)
        print(f"✓ 1.1: Hashing works")
        print(f"    hash(e1) = {h1}")
        print(f"    hash(e2) = {h2}")
        print(f"    hash(e3) = {h3}")
        print(f"    e1 and e2 have same hash: {h1 == h2}")
    except TypeError as e:
        print(f"✗ 1.1: Hashing failed: {e}")
        return False
    
    # Test 1.2: Equality works correctly
    try:
        assert e1 == e2, "Entities with same ID should be equal"
        assert e1 != e3, "Entities with different IDs should not be equal"
        assert e2 != e3, "Entities with different IDs should not be equal"
        print(f"✓ 1.2: Equality works correctly")
        print(f"    e1 == e2: {e1 == e2} (same ID)")
        print(f"    e1 == e3: {e1 == e3} (different ID)")
    except AssertionError as e:
        print(f"✗ 1.2: Equality failed: {e}")
        return False
    
    # Test 1.3: Hash stability after metadata changes
    try:
        original_hash = hash(e1)
        e1.metadata['new_field'] = 'new_value'
        new_hash = hash(e1)
        assert original_hash == new_hash, "Hash should remain stable despite metadata changes"
        print(f"✓ 1.3: Hash stability verified")
        print(f"    Hash before metadata change: {original_hash}")
        print(f"    Hash after metadata change:  {new_hash}")
    except AssertionError as e:
        print(f"✗ 1.3: Hash stability failed: {e}")
        return False
    
    print("\n✅ All Entity hashability tests passed!\n")
    return True


def test_entity_in_sets():
    """Test Entity usage in sets"""
    print("\n" + "="*70)
    print("TEST 2: Entity Set Operations")
    print("="*70)
    
    # Create test entities
    e1 = Entity(id="1", type=EntityType.PROCEDURE, name="proc1", qualified_name="file::proc1")
    e2 = Entity(id="2", type=EntityType.PROCEDURE, name="proc2", qualified_name="file::proc2")
    e3 = Entity(id="1", type=EntityType.PROCEDURE, name="proc1", qualified_name="file::proc1")  # Same as e1
    e4 = Entity(id="3", type=EntityType.VARIABLE, name="var1", qualified_name="file::var1")
    
    # Test 2.1: Creating sets
    try:
        entity_set = {e1, e2, e3, e4}
        assert len(entity_set) == 3, f"Expected 3 unique entities, got {len(entity_set)}"
        print(f"✓ 2.1: Set creation and deduplication works")
        print(f"    Created set with 4 entities (2 identical)")
        print(f"    Set size: {len(entity_set)} (correctly deduplicated)")
    except (TypeError, AssertionError) as e:
        print(f"✗ 2.1: Set creation failed: {e}")
        return False
    
    # Test 2.2: Set membership
    try:
        assert e1 in entity_set, "e1 should be in set"
        assert e3 in entity_set, "e3 should be in set (equal to e1)"
        assert e2 in entity_set, "e2 should be in set"
        assert e4 in entity_set, "e4 should be in set"
        print(f"✓ 2.2: Set membership works correctly")
    except AssertionError as e:
        print(f"✗ 2.2: Set membership failed: {e}")
        return False
    
    # Test 2.3: Set operations (union, intersection, difference)
    try:
        set1 = {e1, e2}
        set2 = {e2, e4}
        
        union = set1 | set2
        assert len(union) == 3, f"Union should have 3 entities, got {len(union)}"
        
        intersection = set1 & set2
        assert len(intersection) == 1, f"Intersection should have 1 entity, got {len(intersection)}"
        assert e2 in intersection, "e2 should be in intersection"
        
        difference = set1 - set2
        assert len(difference) == 1, f"Difference should have 1 entity, got {len(difference)}"
        assert e1 in difference, "e1 should be in difference"
        
        print(f"✓ 2.3: Set operations (union, intersection, difference) work")
        print(f"    Union: {len(union)} entities")
        print(f"    Intersection: {len(intersection)} entities")
        print(f"    Difference: {len(difference)} entities")
    except (TypeError, AssertionError) as e:
        print(f"✗ 2.3: Set operations failed: {e}")
        return False
    
    print("\n✅ All Entity set operation tests passed!\n")
    return True


def test_entity_as_dict_keys():
    """Test Entity usage as dictionary keys"""
    print("\n" + "="*70)
    print("TEST 3: Entity Dictionary Key Operations")
    print("="*70)
    
    # Create test entities
    e1 = Entity(id="1", type=EntityType.PROCEDURE, name="proc1", qualified_name="file::proc1")
    e2 = Entity(id="2", type=EntityType.PROCEDURE, name="proc2", qualified_name="file::proc2")
    e3 = Entity(id="1", type=EntityType.PROCEDURE, name="proc1", qualified_name="file::proc1")  # Same as e1
    
    # Test 3.1: Using entities as dict keys
    try:
        entity_dict = {
            e1: "payment logic",
            e2: "validation logic"
        }
        assert len(entity_dict) == 2, f"Expected 2 entries, got {len(entity_dict)}"
        print(f"✓ 3.1: Entities work as dictionary keys")
        print(f"    Created dict with 2 entity keys")
    except (TypeError, AssertionError) as e:
        print(f"✗ 3.1: Dict key creation failed: {e}")
        return False
    
    # Test 3.2: Retrieving values with equal keys
    try:
        value = entity_dict[e3]  # e3 equals e1
        assert value == "payment logic", f"Expected 'payment logic', got '{value}'"
        print(f"✓ 3.2: Retrieving values with equal keys works")
        print(f"    Retrieved value using equivalent entity: '{value}'")
    except (KeyError, AssertionError) as e:
        print(f"✗ 3.2: Dict retrieval failed: {e}")
        return False
    
    # Test 3.3: Key existence checks
    try:
        assert e1 in entity_dict, "e1 should be in dict"
        assert e3 in entity_dict, "e3 should be in dict (equal to e1)"
        assert e2 in entity_dict, "e2 should be in dict"
        
        e4 = Entity(id="3", type=EntityType.VARIABLE, name="var1", qualified_name="file::var1")
        assert e4 not in entity_dict, "e4 should not be in dict"
        
        print(f"✓ 3.3: Key existence checks work correctly")
    except AssertionError as e:
        print(f"✗ 3.3: Key existence check failed: {e}")
        return False
    
    print("\n✅ All Entity dictionary operation tests passed!\n")
    return True


def test_relationship_hashability():
    """Test Relationship class hashability"""
    print("\n" + "="*70)
    print("TEST 4: Relationship Hashability")
    print("="*70)
    
    # Create test relationships
    r1 = Relationship(
        source_id="proc1",
        target_id="proc2",
        type=RelationType.CALLS,
        weight=1.0,
        metadata={"line": 42}
    )
    
    r2 = Relationship(
        source_id="proc1",
        target_id="proc2",
        type=RelationType.CALLS,
        weight=2.0,  # Different weight
        metadata={"line": 100}  # Different metadata
    )
    
    r3 = Relationship(
        source_id="proc2",
        target_id="proc3",
        type=RelationType.CALLS
    )
    
    # Test 4.1: Hashing works
    try:
        h1 = hash(r1)
        h2 = hash(r2)
        h3 = hash(r3)
        print(f"✓ 4.1: Relationship hashing works")
        print(f"    hash(r1) = {h1}")
        print(f"    hash(r2) = {h2}")
        print(f"    hash(r3) = {h3}")
        print(f"    r1 and r2 have same hash: {h1 == h2}")
    except TypeError as e:
        print(f"✗ 4.1: Relationship hashing failed: {e}")
        return False
    
    # Test 4.2: Equality works correctly
    try:
        assert r1 == r2, "Relationships with same source/target/type should be equal"
        assert r1 != r3, "Relationships with different source/target should not be equal"
        print(f"✓ 4.2: Relationship equality works correctly")
        print(f"    r1 == r2: {r1 == r2} (same source/target/type)")
        print(f"    r1 == r3: {r1 == r3} (different source/target)")
    except AssertionError as e:
        print(f"✗ 4.2: Relationship equality failed: {e}")
        return False
    
    # Test 4.3: Relationships in sets
    try:
        rel_set = {r1, r2, r3}
        assert len(rel_set) == 2, f"Expected 2 unique relationships, got {len(rel_set)}"
        print(f"✓ 4.3: Relationship set operations work")
        print(f"    Created set with 3 relationships (2 identical)")
        print(f"    Set size: {len(rel_set)} (correctly deduplicated)")
    except (TypeError, AssertionError) as e:
        print(f"✗ 4.3: Relationship set operations failed: {e}")
        return False
    
    print("\n✅ All Relationship hashability tests passed!\n")
    return True


def test_knowledge_graph_operations():
    """Test that graph operations work with hashable entities"""
    print("\n" + "="*70)
    print("TEST 5: Knowledge Graph Operations with Hashable Entities")
    print("="*70)
    
    # Create knowledge graph
    kg = KnowledgeGraph(backend="networkx")
    
    # Create test entities
    file_entity = Entity(
        id="file_1",
        type=EntityType.FILE,
        name="payment_system.tal",
        qualified_name="payment_system.tal"
    )
    
    proc1 = Entity(
        id="proc_1",
        type=EntityType.PROCEDURE,
        name="process_payment",
        qualified_name="payment_system::process_payment",
        metadata={"params": 3}
    )
    
    proc2 = Entity(
        id="proc_2",
        type=EntityType.PROCEDURE,
        name="validate_account",
        qualified_name="payment_system::validate_account"
    )
    
    proc3 = Entity(
        id="proc_3",
        type=EntityType.PROCEDURE,
        name="check_balance",
        qualified_name="payment_system::check_balance"
    )
    
    # Test 5.1: Adding entities
    try:
        kg.add_entity(file_entity)
        kg.add_entity(proc1)
        kg.add_entity(proc2)
        kg.add_entity(proc3)
        
        stats = kg.get_statistics()
        assert stats['total_entities'] == 4, f"Expected 4 entities, got {stats['total_entities']}"
        print(f"✓ 5.1: Adding entities works")
        print(f"    Added 4 entities")
        print(f"    Total entities: {stats['total_entities']}")
    except AssertionError as e:
        print(f"✗ 5.1: Adding entities failed: {e}")
        return False
    
    # Test 5.2: Adding relationships
    try:
        kg.add_relationship(Relationship(
            source_id=file_entity.id,
            target_id=proc1.id,
            type=RelationType.DEFINES
        ))
        
        kg.add_relationship(Relationship(
            source_id=proc1.id,
            target_id=proc2.id,
            type=RelationType.CALLS
        ))
        
        kg.add_relationship(Relationship(
            source_id=proc1.id,
            target_id=proc3.id,
            type=RelationType.CALLS
        ))
        
        kg.add_relationship(Relationship(
            source_id=proc2.id,
            target_id=proc3.id,
            type=RelationType.CALLS
        ))
        
        stats = kg.get_statistics()
        assert stats['total_relationships'] == 4, f"Expected 4 relationships, got {stats['total_relationships']}"
        print(f"✓ 5.2: Adding relationships works")
        print(f"    Added 4 relationships")
        print(f"    Total relationships: {stats['total_relationships']}")
    except AssertionError as e:
        print(f"✗ 5.2: Adding relationships failed: {e}")
        return False
    
    # Test 5.3: Get neighbors (uses set() internally - requires hashable entities)
    try:
        neighbors = kg.get_neighbors(proc1.id, rel_type=RelationType.CALLS, direction="outgoing")
        assert len(neighbors) == 2, f"Expected 2 neighbors, got {len(neighbors)}"
        
        neighbor_names = {n.name for n in neighbors}
        assert "validate_account" in neighbor_names, "validate_account should be a neighbor"
        assert "check_balance" in neighbor_names, "check_balance should be a neighbor"
        
        print(f"✓ 5.3: Get neighbors works (requires hashable entities)")
        print(f"    Found {len(neighbors)} neighbors for process_payment")
        print(f"    Neighbors: {', '.join(neighbor_names)}")
    except (TypeError, AssertionError) as e:
        print(f"✗ 5.3: Get neighbors failed: {e}")
        return False
    
    # Test 5.4: Query entities
    try:
        procedures = kg.query_entities(entity_type=EntityType.PROCEDURE)
        assert len(procedures) == 3, f"Expected 3 procedures, got {len(procedures)}"
        print(f"✓ 5.4: Query entities works")
        print(f"    Found {len(procedures)} procedures")
    except AssertionError as e:
        print(f"✗ 5.4: Query entities failed: {e}")
        return False
    
    # Test 5.5: Export subgraph
    try:
        subgraph = kg.db.export_subgraph([proc1.id, proc2.id, proc3.id])
        assert subgraph['entity_count'] == 3, f"Expected 3 entities, got {subgraph['entity_count']}"
        print(f"✓ 5.5: Export subgraph works")
        print(f"    Exported subgraph with {subgraph['entity_count']} entities")
    except AssertionError as e:
        print(f"✗ 5.5: Export subgraph failed: {e}")
        return False
    
    print("\n✅ All Knowledge Graph operation tests passed!\n")
    return True


def test_edge_cases():
    """Test edge cases and corner scenarios"""
    print("\n" + "="*70)
    print("TEST 6: Edge Cases and Corner Scenarios")
    print("="*70)
    
    # Test 6.1: Empty entities in sets
    try:
        e1 = Entity(id="", type=EntityType.PROCEDURE, name="", qualified_name="")
        e2 = Entity(id="", type=EntityType.PROCEDURE, name="", qualified_name="")
        
        entity_set = {e1, e2}
        assert len(entity_set) == 1, "Empty entities should deduplicate"
        print(f"✓ 6.1: Empty entities handled correctly")
    except Exception as e:
        print(f"✗ 6.1: Empty entities failed: {e}")
        return False
    
    # Test 6.2: None values in optional fields
    try:
        e1 = Entity(
            id="test",
            type=EntityType.PROCEDURE,
            name="test",
            qualified_name="test",
            file_path=None,
            start_line=None
        )
        _ = hash(e1)
        print(f"✓ 6.2: None values in optional fields handled correctly")
    except Exception as e:
        print(f"✗ 6.2: None values failed: {e}")
        return False
    
    # Test 6.3: Large sets of entities (performance test)
    try:
        entities = []
        for i in range(1000):
            e = Entity(
                id=f"entity_{i}",
                type=EntityType.PROCEDURE,
                name=f"proc_{i}",
                qualified_name=f"file::proc_{i}"
            )
            entities.append(e)
        
        entity_set = set(entities)
        assert len(entity_set) == 1000, f"Expected 1000 unique entities, got {len(entity_set)}"
        
        # Add duplicates
        entity_set.update(entities[:100])  # Add first 100 again
        assert len(entity_set) == 1000, "Duplicates should not increase set size"
        
        print(f"✓ 6.3: Large sets handled correctly (1000 entities)")
    except Exception as e:
        print(f"✗ 6.3: Large sets failed: {e}")
        return False
    
    # Test 6.4: Mixed entity types in same set
    try:
        entities = [
            Entity(id="1", type=EntityType.PROCEDURE, name="p", qualified_name="p"),
            Entity(id="2", type=EntityType.VARIABLE, name="v", qualified_name="v"),
            Entity(id="3", type=EntityType.STRUCTURE, name="s", qualified_name="s"),
            Entity(id="4", type=EntityType.FILE, name="f", qualified_name="f"),
        ]
        
        entity_set = set(entities)
        assert len(entity_set) == 4, "Should handle mixed entity types"
        print(f"✓ 6.4: Mixed entity types in sets handled correctly")
    except Exception as e:
        print(f"✗ 6.4: Mixed entity types failed: {e}")
        return False
    
    print("\n✅ All edge case tests passed!\n")
    return True


def test_no_functionality_regression():
    """Ensure no functionality was lost"""
    print("\n" + "="*70)
    print("TEST 7: No Functionality Regression")
    print("="*70)
    
    kg = KnowledgeGraph(backend="networkx")
    
    # Test 7.1: Entity to_dict works
    try:
        e = Entity(
            id="test",
            type=EntityType.PROCEDURE,
            name="test_proc",
            qualified_name="file::test_proc",
            metadata={"key": "value"}
        )
        
        entity_dict = e.to_dict()
        assert entity_dict['id'] == "test", "to_dict should preserve id"
        assert entity_dict['type'] == "procedure", "to_dict should convert enum to value"
        assert entity_dict['metadata']['key'] == "value", "to_dict should preserve metadata"
        print(f"✓ 7.1: Entity.to_dict() works correctly")
    except AssertionError as e:
        print(f"✗ 7.1: Entity.to_dict() failed: {e}")
        return False
    
    # Test 7.2: Relationship to_dict works
    try:
        r = Relationship(
            source_id="src",
            target_id="tgt",
            type=RelationType.CALLS,
            metadata={"line": 42}
        )
        
        rel_dict = r.to_dict()
        assert rel_dict['source_id'] == "src", "to_dict should preserve source_id"
        assert rel_dict['type'] == "calls", "to_dict should convert enum to value"
        print(f"✓ 7.2: Relationship.to_dict() works correctly")
    except AssertionError as e:
        print(f"✗ 7.2: Relationship.to_dict() failed: {e}")
        return False
    
    # Test 7.3: __repr__ methods work
    try:
        e = Entity(id="test", type=EntityType.PROCEDURE, name="test", qualified_name="test")
        r = Relationship(source_id="s", target_id="t", type=RelationType.CALLS)
        
        entity_repr = repr(e)
        rel_repr = repr(r)
        
        assert "Entity" in entity_repr, "__repr__ should include class name"
        assert "Relationship" in rel_repr, "__repr__ should include class name"
        print(f"✓ 7.3: __repr__ methods work correctly")
        print(f"    Entity repr: {entity_repr}")
        print(f"    Relationship repr: {rel_repr}")
    except AssertionError as e:
        print(f"✗ 7.3: __repr__ methods failed: {e}")
        return False
    
    # Test 7.4: Statistics work
    try:
        # Add some entities
        for i in range(5):
            kg.add_entity(Entity(
                id=f"proc_{i}",
                type=EntityType.PROCEDURE,
                name=f"proc_{i}",
                qualified_name=f"file::proc_{i}"
            ))
        
        stats = kg.get_statistics()
        assert 'total_entities' in stats, "Statistics should include total_entities"
        assert 'entity_counts' in stats, "Statistics should include entity_counts"
        assert stats['total_entities'] == 5, f"Expected 5 entities, got {stats['total_entities']}"
        print(f"✓ 7.4: Statistics calculation works correctly")
    except AssertionError as e:
        print(f"✗ 7.4: Statistics failed: {e}")
        return False
    
    print("\n✅ All functionality regression tests passed!\n")
    return True


def main():
    """Run all tests"""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║       Comprehensive Test Suite for Hashable Knowledge Graph         ║
╚══════════════════════════════════════════════════════════════════════╝
""")
    
    tests = [
        ("Entity Hashability", test_entity_hashability),
        ("Entity Set Operations", test_entity_in_sets),
        ("Entity Dictionary Operations", test_entity_as_dict_keys),
        ("Relationship Hashability", test_relationship_hashability),
        ("Knowledge Graph Operations", test_knowledge_graph_operations),
        ("Edge Cases", test_edge_cases),
        ("No Functionality Regression", test_no_functionality_regression),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n" + "="*70)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("="*70)
        print("\nThe knowledge_graph.py module is fully hashable and functional!")
        print("\nYou can now:")
        print("  • Use Entity objects in sets")
        print("  • Use Entity objects as dictionary keys")
        print("  • Use Relationship objects in sets")
        print("  • Parse TAL directories without 'unhashable type' errors")
        print("  • Perform all graph operations with full confidence")
        print("\nRun your TAL parser with:")
        print("  python parsers.py ./your_tal_directory\n")
        return 0
    else:
        print("\n" + "="*70)
        print("❌ SOME TESTS FAILED")
        print("="*70)
        print("\nPlease review the failures above and ensure:")
        print("  • __hash__ and __eq__ are properly implemented")
        print("  • All functionality is preserved")
        print("  • Edge cases are handled correctly\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
