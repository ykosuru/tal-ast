#!/usr/bin/env python3
"""
Export ONLY entities related to a search term (focused subgraph)
Shows the search matches + their direct relationships only
"""

from knowledge_graph import KnowledgeGraph, EntityType, RelationType
import json
from pathlib import Path
import sys
import re

def export_focused_subgraph(kg: KnowledgeGraph, 
                           search_term: str,
                           output_dir: str = "./output",
                           depth: int = 1,
                           include_variables: bool = True):
    """
    Export only entities matching search term and their relationships
    
    Args:
        kg: Knowledge graph
        search_term: Term to search for
        output_dir: Output directory
        depth: How many relationship hops to include (1 = direct only)
        include_variables: Include variables in subgraph
    
    Returns:
        Path to exported file
    """
    print(f"\n{'='*70}")
    print(f"EXPORTING FOCUSED SUBGRAPH: '{search_term}'")
    print(f"{'='*70}\n")
    
    all_entities = kg.query_entities()
    
    # Step 1: Find all entities matching search term
    pattern = re.compile(re.escape(search_term), re.IGNORECASE)
    matching_entities = []
    
    for entity in all_entities:
        matches = (
            pattern.search(entity.name) or 
            pattern.search(entity.qualified_name) or
            any(pattern.search(str(v)) for v in entity.metadata.values() if isinstance(v, str))
        )
        
        if matches:
            matching_entities.append(entity)
    
    print(f"Step 1: Found {len(matching_entities)} entities matching '{search_term}'")
    
    # Show what we found
    type_counts = {}
    for entity in matching_entities:
        entity_type = entity.type.value
        type_counts[entity_type] = type_counts.get(entity_type, 0) + 1
        print(f"  • {entity.name} ({entity.type.value})")
    
    print(f"\nMatching entity types:")
    for entity_type, count in sorted(type_counts.items()):
        print(f"  {entity_type}: {count}")
    
    if not matching_entities:
        print(f"\n❌ No entities found matching '{search_term}'")
        sys.exit(1)
    
    # Step 2: Build subgraph by traversing relationships
    subgraph_entities = {e.id: e for e in matching_entities}
    subgraph_relationships = []
    
    print(f"\nStep 2: Finding related entities (depth={depth})...")
    
    for current_depth in range(depth):
        current_entity_ids = set(subgraph_entities.keys())
        new_entities = {}
        
        for entity_id in current_entity_ids:
            # Get all relationships involving this entity
            outgoing = kg.query_relationships(source_id=entity_id)
            incoming = kg.query_relationships(target_id=entity_id)
            
            for rel in outgoing + incoming:
                # Add the relationship
                if rel not in subgraph_relationships:
                    subgraph_relationships.append(rel)
                
                # Add the other entity
                other_id = rel.target_id if rel.source_id == entity_id else rel.source_id
                
                if other_id not in subgraph_entities and other_id not in new_entities:
                    other_entity = kg.get_entity(other_id)
                    if other_entity:
                        # Filter out files and optionally variables
                        if other_entity.type == EntityType.FILE:
                            continue
                        if not include_variables and other_entity.type == EntityType.VARIABLE:
                            continue
                        
                        new_entities[other_id] = other_entity
        
        print(f"  Depth {current_depth + 1}: Added {len(new_entities)} related entities")
        subgraph_entities.update(new_entities)
    
    print(f"\nStep 3: Building export...")
    print(f"  Total entities in subgraph: {len(subgraph_entities)}")
    print(f"  Total relationships: {len(subgraph_relationships)}")
    
    # Step 3: Filter relationships to only those within subgraph
    entity_ids_set = set(subgraph_entities.keys())
    filtered_relationships = [
        rel for rel in subgraph_relationships
        if rel.source_id in entity_ids_set and rel.target_id in entity_ids_set
    ]
    
    print(f"  Filtered relationships: {len(filtered_relationships)}")
    
    # Step 4: Convert to visualization format
    nodes = []
    for entity in subgraph_entities.values():
        # Mark if this is a primary match
        is_match = entity in matching_entities
        
        node = {
            'id': entity.id,
            'name': entity.name,
            'type': entity.type.value,
            'qualified_name': entity.qualified_name,
            'file_path': entity.file_path,
            'start_line': entity.start_line,
            'end_line': entity.end_line,
            'language': entity.language,
            'metadata': {
                **entity.metadata,
                'is_search_match': is_match  # Mark primary matches
            }
        }
        nodes.append(node)
    
    edges = []
    for rel in filtered_relationships:
        edge = {
            'source': rel.source_id,
            'target': rel.target_id,
            'type': rel.type.value,
            'weight': rel.weight,
            'metadata': rel.metadata
        }
        edges.append(edge)
    
    vis_data = {
        'nodes': nodes,
        'edges': edges,
        'statistics': kg.get_statistics(),
        'metadata': {
            'search_term': search_term,
            'primary_matches': len(matching_entities),
            'total_entities_in_subgraph': len(subgraph_entities),
            'relationships': len(filtered_relationships),
            'depth': depth,
            'is_focused_subgraph': True
        }
    }
    
    # Step 5: Save
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    safe_search_term = re.sub(r'[^\w\-]', '_', search_term)
    output_file = output_path / f"subgraph_{safe_search_term}.json"
    
    with open(output_file, 'w') as f:
        json.dump(vis_data, f, indent=2)
    
    print(f"\n{'='*70}")
    print("EXPORT COMPLETE")
    print(f"{'='*70}")
    print(f"\n✓ Saved to: {output_file}")
    
    print(f"\nSubgraph contents:")
    print(f"  Primary matches (with '{search_term}'): {len(matching_entities)}")
    
    # Count types in subgraph
    subgraph_type_counts = {}
    for node in nodes:
        node_type = node['type']
        subgraph_type_counts[node_type] = subgraph_type_counts.get(node_type, 0) + 1
    
    print(f"  Total entities: {len(nodes)}")
    for entity_type, count in sorted(subgraph_type_counts.items()):
        print(f"    • {entity_type}: {count}")
    
    print(f"  Relationships: {len(edges)}")
    
    # Count relationship types
    rel_type_counts = {}
    for edge in edges:
        rel_type = edge['type']
        rel_type_counts[rel_type] = rel_type_counts.get(rel_type, 0) + 1
    
    for rel_type, count in sorted(rel_type_counts.items()):
        print(f"    • {rel_type}: {count}")
    
    print(f"\n{'='*70}")
    print("NEXT STEP")
    print(f"{'='*70}")
    print(f"\nGenerate visualization:")
    print(f"  python graph_visualizer.py {output_file} -t '{search_term} Subgraph'")
    print()
    
    return str(output_file)


def list_matching_entities(kg: KnowledgeGraph, search_term: str):
    """List all entities matching a search term"""
    all_entities = kg.query_entities()
    pattern = re.compile(re.escape(search_term), re.IGNORECASE)
    
    matching = []
    for entity in all_entities:
        if (pattern.search(entity.name) or 
            pattern.search(entity.qualified_name)):
            matching.append(entity)
    
    print(f"\n{'='*70}")
    print(f"ENTITIES MATCHING: '{search_term}'")
    print(f"{'='*70}\n")
    
    if not matching:
        print(f"No entities found matching '{search_term}'")
        return
    
    # Group by type
    by_type = {}
    for entity in matching:
        entity_type = entity.type.value
        if entity_type not in by_type:
            by_type[entity_type] = []
        by_type[entity_type].append(entity)
    
    for entity_type, entities in sorted(by_type.items()):
        print(f"\n{entity_type.upper()} ({len(entities)}):")
        for entity in entities:
            print(f"  • {entity.name}")
            print(f"    File: {Path(entity.file_path).name if entity.file_path else 'N/A'}")
            print(f"    Line: {entity.start_line}")
            if entity.type.value == 'procedure' and 'parameters' in entity.metadata:
                params = entity.metadata.get('parameters', [])
                print(f"    Params: {', '.join(params) if params else 'none'}")
    
    print(f"\n{'='*70}")
    print(f"Total: {len(matching)} entities")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Export focused subgraph for a search term',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export subgraph for "drawdown" (direct connections only)
  python export_subgraph.py knowledge_graph.json drawdown
  
  # Include 2 levels of relationships
  python export_subgraph.py knowledge_graph.json drawdown --depth 2
  
  # Exclude variables from subgraph
  python export_subgraph.py knowledge_graph.json drawdown --no-variables
  
  # Just list what matches (don't export)
  python export_subgraph.py knowledge_graph.json drawdown --list-only
        """
    )
    
    parser.add_argument('graph_file', help='Path to knowledge_graph.json')
    parser.add_argument('search_term', help='Search term (e.g., "drawdown")')
    parser.add_argument('--depth', type=int, default=1, 
                       help='Relationship depth (default: 1 = direct connections only)')
    parser.add_argument('--no-variables', action='store_true',
                       help='Exclude variables from subgraph')
    parser.add_argument('--list-only', action='store_true',
                       help='Just list matching entities, don\'t export')
    parser.add_argument('-o', '--output', default='./output',
                       help='Output directory (default: ./output)')
    
    args = parser.parse_args()
    
    # Load knowledge graph
    print(f"\nLoading graph from: {args.graph_file}")
    
    if not Path(args.graph_file).exists():
        print(f"Error: File not found: {args.graph_file}")
        sys.exit(1)
    
    kg = KnowledgeGraph()
    kg.load_from_json(args.graph_file)
    
    stats = kg.get_statistics()
    print(f"Loaded graph:")
    print(f"  Entities: {stats['total_entities']}")
    print(f"  Relationships: {stats['total_relationships']}")
    
    # List or export
    if args.list_only:
        list_matching_entities(kg, args.search_term)
    else:
        output_file = export_focused_subgraph(
            kg,
            search_term=args.search_term,
            output_dir=args.output,
            depth=args.depth,
            include_variables=not args.no_variables
        )
        
        print(f"Run this to visualize:")
        print(f"  python graph_visualizer.py {output_file} -t '{args.search_term} Analysis'")
