#!/usr/bin/env python3
"""
STANDALONE Subgraph Exporter – NO DEPENDENCIES
Works directly with JSON files – doesn’t need knowledge_graph.py
"""

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Any

def load_graph_json(filepath: str) -> Dict[str, Any]:
    """Load graph data from any JSON format"""
    print(f"Loading: {filepath}")
    with open(filepath, 'r') as f:
        data = json.load(f)

    # Detect and convert format
    if 'entities' in data and 'relationships' in data:
        print("Format: KnowledgeGraph (entities/relationships)")
        nodes = data['entities']
        edges = data['relationships']
    elif 'nodes' in data:
        print("Format: D3.js (nodes/edges)")
        nodes = data['nodes']
        edges = data.get('edges') or data.get('links', [])
    else:
        raise ValueError(f"Unknown format. Keys: {list(data.keys())}")

    print(f"  Total nodes: {len(nodes)}")
    print(f"  Total edges: {len(edges)}")

    return {
        'nodes': nodes,
        'edges': edges,
        'metadata': data.get('metadata', {}),
        'statistics': data.get('statistics', {})
    }


def find_matching_nodes(nodes: List[Dict], search_term: str) -> List[Dict]:
    """Find all nodes matching the search term"""
    pattern = re.compile(re.escape(search_term), re.IGNORECASE)
    matching = []

    for node in nodes:
        # Check name
        if pattern.search(node.get('name', '')):
            matching.append(node)
            continue
        # Check qualified_name
        if pattern.search(node.get('qualified_name', '')):
            matching.append(node)
            continue
        # Check metadata (handle both dict and string)
        metadata = node.get('metadata', {})
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except:
                metadata = {}
        if isinstance(metadata, dict):
            for value in metadata.values():
                if isinstance(value, str) and pattern.search(value):
                    matching.append(node)
                    break
    return matching


def should_include_node(node: Dict, include_variables: bool) -> bool:
    """Check if a node should be included in subgraph"""
    node_type = node.get('type', '')
    # Always exclude files
    if node_type == 'file':
        return False
    # Optionally exclude variables
    if not include_variables and node_type == 'variable':
        return False
    return True


def build_subgraph(
    nodes: List[Dict],
    edges: List[Dict],
    matching_nodes: List[Dict],
    depth: int = 1,
    include_variables: bool = True
) -> Dict[str, Any]:
    """Build a subgraph around matching nodes"""
    # Create node lookup
    node_by_id = {n['id']: n for n in nodes}
    matching_ids = {n['id'] for n in matching_nodes}

    # Start with matching nodes
    subgraph_ids = set(matching_ids)
    print(f"\nBuilding subgraph (depth={depth})...")

    # Expand by depth
    for d in range(depth):
        new_ids = set()
        for edge in edges:
            # Get source and target IDs (handle multiple formats)
            source_id = edge.get('source') or edge.get('source_id')
            target_id = edge.get('target') or edge.get('target_id')

            # Handle case where source/target might be objects
            if isinstance(source_id, dict):
                source_id = source_id.get('id')
            if isinstance(target_id, dict):
                target_id = target_id.get('id')

            if not source_id or not target_id:
                continue

            # If edge connects to subgraph, add the other node
            if source_id in subgraph_ids and target_id not in subgraph_ids:
                other_node = node_by_id.get(target_id)
                if other_node and should_include_node(other_node, include_variables):
                    new_ids.add(target_id)
            elif target_id in subgraph_ids and source_id not in subgraph_ids:
                other_node = node_by_id.get(source_id)
                if other_node and should_include_node(other_node, include_variables):
                    new_ids.add(source_id)

        subgraph_ids.update(new_ids)
        print(f"  Depth {d+1}: Added {len(new_ids)} related nodes")

    # Build final subgraph
    subgraph_nodes = []
    for nid in subgraph_ids:
        if nid in node_by_id:
            node = node_by_id[nid].copy()
            # Ensure metadata is a dict
            metadata = node.get('metadata', {})
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except:
                    metadata = {}
            # Mark search matches
            metadata['is_search_match'] = nid in matching_ids
            node['metadata'] = metadata
            subgraph_nodes.append(node)

    # Filter edges
    subgraph_edges = []
    for edge in edges:
        source_id = edge.get('source') or edge.get('source_id')
        target_id = edge.get('target') or edge.get('target_id')
        if isinstance(source_id, dict):
            source_id = source_id.get('id')
        if isinstance(target_id, dict):
            target_id = target_id.get('id')
        if source_id in subgraph_ids and target_id in subgraph_ids:
            # Normalize edge format
            normalized_edge = {
                'source': source_id,
                'target': target_id,
                'type': edge.get('type', 'unknown'),
                'weight': edge.get('weight', 1.0),
                'metadata': edge.get('metadata', {})
            }
            subgraph_edges.append(normalized_edge)

    return {
        'nodes': subgraph_nodes,
        'edges': subgraph_edges,
        'metadata': {
            'search_term': None,  # Will be set by caller
            'primary_matches': len(matching_nodes),
            'total_nodes': len(subgraph_nodes),
            'total_edges': len(subgraph_edges),
            'depth': depth
        }
    }


def export_subgraph(
    json_file: str,
    search_term: str,
    output_dir: str = "./output",
    depth: int = 1,
    include_variables: bool = True
) -> str:
    """Main export function"""
    print(f"\n{'='*70}")
    print(f"EXPORTING SUBGRAPH: '{search_term}'")
    print(f"{'='*70}\n")

    # Load graph
    graph_data = load_graph_json(json_file)

    # Find matching nodes
    print(f"\nSearching for '{search_term}'...")
    matching_nodes = find_matching_nodes(graph_data['nodes'], search_term)
    print(f"Found {len(matching_nodes)} matching nodes:")

    # Group by type
    by_type = {}
    for node in matching_nodes:
        node_type = node.get('type', 'unknown')
        if node_type not in by_type:
            by_type[node_type] = []
        by_type[node_type].append(node)

    for node_type, nodes in sorted(by_type.items()):
        print(f"\n  {node_type.upper()} ({len(nodes)}):")
        for node in nodes[:10]:  # Show first 10
            print(f"    • {node.get('name')}")
        if len(nodes) > 10:
            print(f"    ... and {len(nodes) - 10} more")

    if not matching_nodes:
        print(f"\n❌ No nodes found matching '{search_term}'")
        sys.exit(1)

    # Build subgraph
    subgraph = build_subgraph(
        graph_data['nodes'],
        graph_data['edges'],
        matching_nodes,
        depth=depth,
        include_variables=include_variables
    )

    # Add search term to metadata
    subgraph['metadata']['search_term'] = search_term

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    safe_term = re.sub(r'[\W\-]+', '_', search_term)
    output_file = output_path / f"subgraph_{safe_term}.json"

    with open(output_file, 'w') as f:
        json.dump(subgraph, f, indent=2)

    print(f"\n{'='*70}")
    print("EXPORT COMPLETE!")
    print(f"{'='*70}")
    print(f"\nSaved to: {output_file}")
    print(f"\nSubgraph summary:")
    print(f"  Primary matches: {subgraph['metadata']['primary_matches']}")
    print(f"  Total nodes: {subgraph['metadata']['total_nodes']}")
    print(f"  Total edges: {subgraph['metadata']['total_edges']}")

    # Count node types
    type_counts = {}
    for node in subgraph['nodes']:
        node_type = node.get('type', 'unknown')
        type_counts[node_type] = type_counts.get(node_type, 0) + 1

    print(f"\n  Node types:")
    for node_type, count in sorted(type_counts.items()):
        print(f"    • {node_type}: {count}")

    print(f"\n{'='*70}")
    print("NEXT STEP")
    print(f"{'='*70}")
    print(f"\nVisualize:")
    print(f"  python3 tal_graph_viz.py {output_file} -t '{search_term}' Analyze")
    print()

    return str(output_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Export focused subgraph (standalone – no dependencies)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export drawdown subgraph
  python3 subgraph.py ./output/knowledge_graph.json drawdown

  # Include 2 levels of relationships
  python3 subgraph.py ./output/knowledge_graph.json drawdown --depth 2

  # Exclude variables
  python3 subgraph.py ./output/knowledge_graph.json drawdown --no-variables
        """
    )

    parser.add_argument('json_file', help='Path to JSON file')
    parser.add_argument('search_term', help='Search term (e.g., "drawdown")')
    parser.add_argument('--depth', type=int, default=1,
                        help='Relationship depth (default: 1)')
    parser.add_argument('--no-variables', action='store_true',
                        help='Exclude variables from subgraph')
    parser.add_argument('-o', '--output', default='./output',
                        help='Output directory (default: ./output)')

    args = parser.parse_args()

    # Check file exists
    if not Path(args.json_file).exists():
        print(f"Error: File not found: {args.json_file}")
        sys.exit(1)

    try:
        output_file = export_subgraph(
            args.json_file,
            search_term=args.search_term,
            output_dir=args.output,
            depth=args.depth,
            include_variables=not args.no_variables
        )
        print(f"Success! Now run:")
        print(f"  python3 tal_graph_viz.py {output_file} -t '{args.search_term}' Analyze")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
