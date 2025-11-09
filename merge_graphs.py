#!/usr/bin/env python3
"""
JSON Graph Merger - Merge two graph JSON files (OUTER JOIN on node IDs)

Merges nodes and edges from two JSON graph files:
- Keeps ALL nodes from both files (OUTER JOIN)
- For matching IDs: merges node data
- Removes duplicate edges
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Set
from copy import deepcopy


def load_json_file(filepath: str) -> Dict[str, Any]:
    """Load JSON file and detect format"""
    print(f"Loading: {filepath}")
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Normalize to standard format
    if 'entities' in data and 'relationships' in data:
        # KnowledgeGraph format
        result = {
            'nodes': data['entities'],
            'edges': data['relationships'],
            'metadata': data.get('metadata', {}),
            'statistics': data.get('statistics', {})
        }
    elif 'nodes' in data:
        # Already in standard format
        result = {
            'nodes': data['nodes'],
            'edges': data.get('edges', data.get('links', [])),
            'metadata': data.get('metadata', {}),
            'statistics': data.get('statistics', {})
        }
    else:
        raise ValueError(f"Unknown format. Keys: {list(data.keys())}")
    
    print(f"  Nodes: {len(result['nodes'])}")
    print(f"  Edges: {len(result['edges'])}")
    
    return result


def normalize_metadata(metadata: Any) -> Dict:
    """Normalize metadata to dict format"""
    if metadata is None:
        return {}
    if isinstance(metadata, dict):
        return metadata
    if isinstance(metadata, str):
        try:
            return json.loads(metadata)
        except:
            return {'raw': metadata}
    return {'value': metadata}


def merge_node_data(node1: Dict, node2: Dict) -> Dict:
    """
    Deep merge two nodes with the same ID
    
    Strategy:
    - Keep all fields from both nodes
    - For conflicts, prefer non-empty values
    - For metadata: deep merge dictionaries, combine lists
    """
    # Start with deep copy of first node
    merged = deepcopy(node1)
    
    # Merge each field from node2
    for key, value2 in node2.items():
        if key == 'id':
            # ID must be the same, skip
            continue
        
        if key not in merged:
            # New field from node2, add it
            merged[key] = deepcopy(value2)
        else:
            value1 = merged[key]
            
            # Both have this field - merge intelligently
            if key == 'metadata':
                # Special handling for metadata
                meta1 = normalize_metadata(value1)
                meta2 = normalize_metadata(value2)
                merged[key] = deep_merge_dicts(meta1, meta2)
            
            elif isinstance(value1, dict) and isinstance(value2, dict):
                # Merge dictionaries
                merged[key] = deep_merge_dicts(value1, value2)
            
            elif isinstance(value1, list) and isinstance(value2, list):
                # Combine lists, remove duplicates (for simple types)
                try:
                    merged[key] = list(set(value1 + value2))
                except TypeError:
                    # Not hashable, just concatenate
                    merged[key] = value1 + value2
            
            elif value2 is not None and (value1 is None or value1 == '' or value1 == 0):
                # Prefer non-empty value from node2
                merged[key] = value2
            
            elif value1 != value2:
                # Conflict - keep both in a special field if different
                if value2 is not None and value2 != '':
                    # For most fields, prefer node2 (newer data)
                    if key not in ['name', 'type', 'qualified_name']:
                        merged[key] = value2
                    # Keep node1 value for core identity fields
    
    return merged


def deep_merge_dicts(dict1: Dict, dict2: Dict) -> Dict:
    """
    Deep merge two dictionaries
    
    For conflicts:
    - Lists: combine and deduplicate
    - Dicts: recursively merge
    - Scalars: prefer non-empty value from dict2
    """
    merged = deepcopy(dict1)
    
    for key, value2 in dict2.items():
        if key not in merged:
            merged[key] = deepcopy(value2)
        else:
            value1 = merged[key]
            
            if isinstance(value1, dict) and isinstance(value2, dict):
                # Recursively merge dicts
                merged[key] = deep_merge_dicts(value1, value2)
            
            elif isinstance(value1, list) and isinstance(value2, list):
                # Combine lists
                try:
                    # Try to deduplicate
                    merged[key] = list(set(value1 + value2))
                except TypeError:
                    # Not hashable, concatenate
                    merged[key] = value1 + value2
            
            elif value2 is not None and (value1 is None or value1 == ''):
                # Prefer non-empty
                merged[key] = value2
            
            else:
                # Keep value2 (newer data)
                merged[key] = value2
    
    return merged


def merge_nodes_outer_join(nodes1: List[Dict], nodes2: List[Dict]) -> tuple[List[Dict], Dict[str, str]]:
    """
    Merge nodes using OUTER JOIN semantics
    
    - Keeps ALL nodes from both lists
    - For matching IDs: merges node data
    - Returns merged nodes and statistics
    
    Returns:
        (merged_nodes, stats_dict)
    """
    nodes_by_id = {}
    
    # Track where nodes came from
    node_sources = {}  # id -> 'file1', 'file2', or 'both'
    
    # Add all nodes from file1
    for node in nodes1:
        node_id = node.get('id')
        if node_id:
            nodes_by_id[node_id] = deepcopy(node)
            node_sources[node_id] = 'file1'
    
    file1_count = len(nodes_by_id)
    
    # Process nodes from file2
    new_from_file2 = 0
    merged_count = 0
    
    for node in nodes2:
        node_id = node.get('id')
        if not node_id:
            continue
        
        if node_id in nodes_by_id:
            # MATCH - merge the two nodes
            existing = nodes_by_id[node_id]
            merged_node = merge_node_data(existing, node)
            nodes_by_id[node_id] = merged_node
            node_sources[node_id] = 'both'
            merged_count += 1
        else:
            # NEW from file2 - add it
            nodes_by_id[node_id] = deepcopy(node)
            node_sources[node_id] = 'file2'
            new_from_file2 += 1
    
    # Create stats
    stats = {
        'only_in_file1': sum(1 for s in node_sources.values() if s == 'file1'),
        'only_in_file2': new_from_file2,
        'in_both': merged_count,
        'node_sources': node_sources
    }
    
    return list(nodes_by_id.values()), stats


def get_edge_key(edge: Dict[str, Any]) -> tuple:
    """Generate a unique key for an edge"""
    source_id = edge.get('source') or edge.get('source_id')
    target_id = edge.get('target') or edge.get('target_id')
    edge_type = edge.get('type', 'unknown')
    
    # Handle case where source/target might be objects
    if isinstance(source_id, dict):
        source_id = source_id.get('id')
    if isinstance(target_id, dict):
        target_id = target_id.get('id')
    
    return (source_id, target_id, edge_type)


def merge_edges(edges1: List[Dict], edges2: List[Dict], 
                valid_node_ids: Set[str]) -> tuple[List[Dict], Dict[str, int]]:
    """
    Merge edges from both files, removing duplicates
    
    Only keeps edges where both source and target nodes exist
    
    Returns:
        (merged_edges, stats_dict)
    """
    edges_by_key = {}
    
    # Track stats
    from_file1 = 0
    from_file2 = 0
    duplicates = 0
    orphaned = 0
    
    # Process first list
    for edge in edges1:
        edge_key = get_edge_key(edge)
        source_id, target_id, _ = edge_key
        
        # Only keep edge if both nodes exist
        if source_id in valid_node_ids and target_id in valid_node_ids:
            if edge_key not in edges_by_key:
                edges_by_key[edge_key] = {
                    'source': source_id,
                    'target': target_id,
                    'type': edge.get('type', 'unknown'),
                    'weight': edge.get('weight', 1.0),
                    'metadata': normalize_metadata(edge.get('metadata', {}))
                }
                from_file1 += 1
        else:
            orphaned += 1
    
    # Process second list
    for edge in edges2:
        edge_key = get_edge_key(edge)
        source_id, target_id, _ = edge_key
        
        # Only keep edge if both nodes exist
        if source_id in valid_node_ids and target_id in valid_node_ids:
            if edge_key not in edges_by_key:
                edges_by_key[edge_key] = {
                    'source': source_id,
                    'target': target_id,
                    'type': edge.get('type', 'unknown'),
                    'weight': edge.get('weight', 1.0),
                    'metadata': normalize_metadata(edge.get('metadata', {}))
                }
                from_file2 += 1
            else:
                # Duplicate - merge metadata if needed
                existing_meta = edges_by_key[edge_key]['metadata']
                new_meta = normalize_metadata(edge.get('metadata', {}))
                edges_by_key[edge_key]['metadata'] = deep_merge_dicts(existing_meta, new_meta)
                duplicates += 1
        else:
            orphaned += 1
    
    stats = {
        'unique_from_file1': from_file1,
        'unique_from_file2': from_file2,
        'duplicates': duplicates,
        'orphaned': orphaned
    }
    
    return list(edges_by_key.values()), stats


def merge_json_graphs(file1: str, file2: str, output: str) -> Dict[str, Any]:
    """
    Merge two JSON graph files using OUTER JOIN semantics
    
    Args:
        file1: Path to first JSON file
        file2: Path to second JSON file
        output: Path to output merged JSON file
    
    Returns:
        Merged graph data with statistics
    """
    print(f"\n{'='*70}")
    print("MERGING JSON GRAPH FILES (OUTER JOIN)")
    print(f"{'='*70}\n")
    
    # Load both files
    graph1 = load_json_file(file1)
    graph2 = load_json_file(file2)
    
    print(f"\nMerging nodes (OUTER JOIN on id)...")
    
    # Merge nodes with OUTER JOIN semantics
    merged_nodes, node_stats = merge_nodes_outer_join(graph1['nodes'], graph2['nodes'])
    
    print(f"  Total merged nodes: {len(merged_nodes)}")
    print(f"    Only in file1: {node_stats['only_in_file1']}")
    print(f"    Only in file2: {node_stats['only_in_file2']}")
    print(f"    In both (merged): {node_stats['in_both']}")
    
    # Get valid node IDs for edge filtering
    valid_node_ids = {node['id'] for node in merged_nodes}
    
    print(f"\nMerging edges...")
    
    # Merge edges
    merged_edges, edge_stats = merge_edges(graph1['edges'], graph2['edges'], valid_node_ids)
    
    print(f"  Total merged edges: {len(merged_edges)}")
    print(f"    Unique from file1: {edge_stats['unique_from_file1']}")
    print(f"    Unique from file2: {edge_stats['unique_from_file2']}")
    print(f"    Duplicates merged: {edge_stats['duplicates']}")
    if edge_stats['orphaned'] > 0:
        print(f"    Orphaned (filtered): {edge_stats['orphaned']}")
    
    # Merge metadata
    merged_metadata = deep_merge_dicts(graph1['metadata'], graph2['metadata'])
    
    # Build result
    result = {
        'nodes': merged_nodes,
        'edges': merged_edges,
        'metadata': {
            **merged_metadata,
            'merge_info': {
                'source_files': [file1, file2],
                'merge_type': 'outer_join',
                'file1_nodes': len(graph1['nodes']),
                'file2_nodes': len(graph2['nodes']),
                'merged_nodes': len(merged_nodes),
                'nodes_only_in_file1': node_stats['only_in_file1'],
                'nodes_only_in_file2': node_stats['only_in_file2'],
                'nodes_merged': node_stats['in_both'],
                'file1_edges': len(graph1['edges']),
                'file2_edges': len(graph2['edges']),
                'merged_edges': len(merged_edges),
                'edge_duplicates': edge_stats['duplicates']
            }
        }
    }
    
    # Calculate statistics
    node_types = {}
    for node in merged_nodes:
        node_type = node.get('type', 'unknown')
        node_types[node_type] = node_types.get(node_type, 0) + 1
    
    edge_types = {}
    for edge in merged_edges:
        edge_type = edge.get('type', 'unknown')
        edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
    
    result['statistics'] = {
        'total_nodes': len(merged_nodes),
        'total_edges': len(merged_edges),
        'node_types': node_types,
        'edge_types': edge_types
    }
    
    # Save to file
    output_path = Path(output)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n{'='*70}")
    print("MERGE COMPLETE")
    print(f"{'='*70}\n")
    print(f"Output: {output_path}")
    print(f"\nFinal statistics:")
    print(f"  Total nodes: {len(merged_nodes)}")
    print(f"    From file1 only: {node_stats['only_in_file1']}")
    print(f"    From file2 only: {node_stats['only_in_file2']}")
    print(f"    Merged (in both): {node_stats['in_both']}")
    print(f"\n  Total edges: {len(merged_edges)}")
    print(f"\n  Node types:")
    for node_type, count in sorted(node_types.items()):
        print(f"    {node_type}: {count}")
    print(f"\n  Edge types:")
    for edge_type, count in sorted(edge_types.items()):
        print(f"    {edge_type}: {count}")
    print(f"\n{'='*70}\n")
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Merge two JSON graph files using OUTER JOIN on node IDs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
OUTER JOIN Behavior:
  - Keeps ALL nodes from both files
  - For nodes with matching IDs: deep merges their data
  - Edges are deduplicated by (source, target, type)

Examples:
  # Merge two subgraphs
  python3 merge_graphs.py subgraph1.json subgraph2.json -o combined.json
  
  # Merge subgraph into full graph
  python3 merge_graphs.py full_graph.json new_subgraph.json -o updated.json
  
  # Combine multiple incremental results
  python3 merge_graphs.json parse1.json parse2.json -o temp.json
  python3 merge_graphs.json temp.json parse3.json -o final.json

Merge Strategy:
  Nodes:
    - All nodes from file1 are kept
    - All nodes from file2 are kept
    - Matching IDs: metadata is deeply merged
  
  Edges:
    - All unique edges from both files
    - Duplicates (same source+target+type) are removed
    - Orphaned edges (missing nodes) are filtered out
        """
    )
    
    parser.add_argument('file1', help='First JSON graph file')
    parser.add_argument('file2', help='Second JSON graph file')
    parser.add_argument('-o', '--output', required=True,
                       help='Output file path')
    
    args = parser.parse_args()
    
    # Validate input files
    if not Path(args.file1).exists():
        print(f"Error: File not found: {args.file1}")
        sys.exit(1)
    
    if not Path(args.file2).exists():
        print(f"Error: File not found: {args.file2}")
        sys.exit(1)
    
    try:
        result = merge_json_graphs(args.file1, args.file2, args.output)
        print(f"✓ Success! Merged graph saved to: {args.output}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
