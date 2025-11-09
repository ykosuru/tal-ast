#!/usr/bin/env python3
"""
JSON Graph Merger - Merge two graph JSON files based on node IDs

Merges nodes and edges from two JSON graph files, deduplicating by ID.
Useful for combining knowledge graphs, subgraphs, or incremental updates.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Set


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


def get_edge_key(edge: Dict[str, Any]) -> tuple:
    """Generate a unique key for an edge based on source, target, and type"""
    source_id = edge.get('source') or edge.get('source_id')
    target_id = edge.get('target') or edge.get('target_id')
    edge_type = edge.get('type', 'unknown')
    
    # Handle case where source/target might be objects
    if isinstance(source_id, dict):
        source_id = source_id.get('id')
    if isinstance(target_id, dict):
        target_id = target_id.get('id')
    
    return (source_id, target_id, edge_type)


def merge_nodes(nodes1: List[Dict], nodes2: List[Dict], 
                strategy: str = 'prefer_first') -> List[Dict]:
    """
    Merge two lists of nodes, deduplicating by ID
    
    Args:
        nodes1: First list of nodes
        nodes2: Second list of nodes
        strategy: How to handle duplicates:
            - 'prefer_first': Keep node from first list
            - 'prefer_second': Keep node from second list (overwrite)
            - 'merge': Merge metadata from both (second overwrites first)
    
    Returns:
        Merged list of nodes
    """
    nodes_by_id = {}
    
    # Add nodes from first list
    for node in nodes1:
        node_id = node.get('id')
        if node_id:
            nodes_by_id[node_id] = node.copy()
    
    # Process nodes from second list
    for node in nodes2:
        node_id = node.get('id')
        if not node_id:
            continue
        
        if node_id in nodes_by_id:
            # Duplicate found - apply strategy
            if strategy == 'prefer_first':
                # Keep existing, skip this one
                pass
            elif strategy == 'prefer_second':
                # Replace with new one
                nodes_by_id[node_id] = node.copy()
            elif strategy == 'merge':
                # Merge metadata
                existing = nodes_by_id[node_id]
                new_node = node.copy()
                
                # Merge metadata fields
                existing_meta = existing.get('metadata', {})
                new_meta = new_node.get('metadata', {})
                
                if isinstance(existing_meta, str):
                    try:
                        existing_meta = json.loads(existing_meta)
                    except:
                        existing_meta = {}
                
                if isinstance(new_meta, str):
                    try:
                        new_meta = json.loads(new_meta)
                    except:
                        new_meta = {}
                
                # Merge: new overwrites existing
                merged_meta = {**existing_meta, **new_meta}
                new_node['metadata'] = merged_meta
                
                # Keep other fields from new node
                nodes_by_id[node_id] = new_node
        else:
            # New node, add it
            nodes_by_id[node_id] = node.copy()
    
    return list(nodes_by_id.values())


def merge_edges(edges1: List[Dict], edges2: List[Dict], 
                valid_node_ids: Set[str]) -> List[Dict]:
    """
    Merge two lists of edges, deduplicating and filtering
    
    Args:
        edges1: First list of edges
        edges2: Second list of edges
        valid_node_ids: Set of valid node IDs (edges referencing missing nodes are filtered)
    
    Returns:
        Merged list of edges
    """
    edges_by_key = {}
    
    # Process first list
    for edge in edges1:
        edge_key = get_edge_key(edge)
        source_id, target_id, _ = edge_key
        
        # Only keep edge if both nodes exist
        if source_id in valid_node_ids and target_id in valid_node_ids:
            if edge_key not in edges_by_key:
                # Normalize edge format
                edges_by_key[edge_key] = {
                    'source': source_id,
                    'target': target_id,
                    'type': edge.get('type', 'unknown'),
                    'weight': edge.get('weight', 1.0),
                    'metadata': edge.get('metadata', {})
                }
    
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
                    'metadata': edge.get('metadata', {})
                }
            # If duplicate exists, could merge metadata here if needed
    
    return list(edges_by_key.values())


def merge_metadata(meta1: Dict, meta2: Dict) -> Dict:
    """Merge metadata from two graphs"""
    merged = {}
    
    # Combine metadata fields, prefer arrays for lists
    all_keys = set(meta1.keys()) | set(meta2.keys())
    
    for key in all_keys:
        val1 = meta1.get(key)
        val2 = meta2.get(key)
        
        if val1 is None:
            merged[key] = val2
        elif val2 is None:
            merged[key] = val1
        else:
            # Both exist - try to combine
            if isinstance(val1, list) and isinstance(val2, list):
                # Combine lists, remove duplicates
                merged[key] = list(set(val1 + val2))
            elif isinstance(val1, dict) and isinstance(val2, dict):
                # Recursively merge dicts
                merged[key] = {**val1, **val2}
            else:
                # Take second value (or could store both)
                merged[key] = val2
    
    return merged


def merge_json_graphs(file1: str, file2: str, output: str,
                      strategy: str = 'prefer_first') -> Dict[str, Any]:
    """
    Merge two JSON graph files
    
    Args:
        file1: Path to first JSON file
        file2: Path to second JSON file
        output: Path to output merged JSON file
        strategy: Merge strategy for duplicate nodes
    
    Returns:
        Merged graph data with statistics
    """
    print(f"\n{'='*70}")
    print("MERGING JSON GRAPH FILES")
    print(f"{'='*70}\n")
    
    # Load both files
    graph1 = load_json_file(file1)
    graph2 = load_json_file(file2)
    
    print(f"\nMerge strategy: {strategy}")
    print(f"\nMerging nodes...")
    
    # Merge nodes
    merged_nodes = merge_nodes(graph1['nodes'], graph2['nodes'], strategy=strategy)
    print(f"  Result: {len(merged_nodes)} nodes")
    print(f"    From file1: {len(graph1['nodes'])}")
    print(f"    From file2: {len(graph2['nodes'])}")
    print(f"    Duplicates removed: {len(graph1['nodes']) + len(graph2['nodes']) - len(merged_nodes)}")
    
    # Get valid node IDs for edge filtering
    valid_node_ids = {node['id'] for node in merged_nodes}
    
    print(f"\nMerging edges...")
    
    # Merge edges
    merged_edges = merge_edges(graph1['edges'], graph2['edges'], valid_node_ids)
    print(f"  Result: {len(merged_edges)} edges")
    print(f"    From file1: {len(graph1['edges'])}")
    print(f"    From file2: {len(graph2['edges'])}")
    print(f"    Duplicates removed: {len(graph1['edges']) + len(graph2['edges']) - len(merged_edges)}")
    
    # Merge metadata
    merged_metadata = merge_metadata(graph1['metadata'], graph2['metadata'])
    
    # Build result
    result = {
        'nodes': merged_nodes,
        'edges': merged_edges,
        'metadata': {
            **merged_metadata,
            'merge_info': {
                'source_files': [file1, file2],
                'merge_strategy': strategy,
                'file1_nodes': len(graph1['nodes']),
                'file2_nodes': len(graph2['nodes']),
                'merged_nodes': len(merged_nodes),
                'file1_edges': len(graph1['edges']),
                'file2_edges': len(graph2['edges']),
                'merged_edges': len(merged_edges)
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
    print(f"  Total edges: {len(merged_edges)}")
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
        description='Merge two JSON graph files based on node IDs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic merge (prefer nodes from first file on conflict)
  python3 merge_graphs.py graph1.json graph2.json -o merged.json
  
  # Prefer nodes from second file
  python3 merge_graphs.py graph1.json graph2.json -o merged.json --prefer-second
  
  # Merge metadata from both files
  python3 merge_graphs.py graph1.json graph2.json -o merged.json --merge
  
Use cases:
  - Combine multiple subgraphs into one
  - Add new nodes/edges to existing graph
  - Update graph with new information
  - Merge incremental parsing results
        """
    )
    
    parser.add_argument('file1', help='First JSON graph file')
    parser.add_argument('file2', help='Second JSON graph file')
    parser.add_argument('-o', '--output', required=True,
                       help='Output file path')
    parser.add_argument('--strategy', '--prefer-second', action='store_const',
                       const='prefer_second', default='prefer_first',
                       dest='strategy',
                       help='Prefer nodes from second file on conflict')
    parser.add_argument('--merge', action='store_const',
                       const='merge', dest='strategy',
                       help='Merge metadata from both files')
    
    args = parser.parse_args()
    
    # Validate input files
    if not Path(args.file1).exists():
        print(f"Error: File not found: {args.file1}")
        sys.exit(1)
    
    if not Path(args.file2).exists():
        print(f"Error: File not found: {args.file2}")
        sys.exit(1)
    
    try:
        result = merge_json_graphs(
            args.file1,
            args.file2,
            args.output,
            strategy=args.strategy
        )
        print(f"Success! Merged graph saved to: {args.output}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
