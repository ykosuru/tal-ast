#!/usr/bin/env python3
"""
STANDALONE Subgraph Exporter – Call Graph Extraction
Works directly with JSON files – doesn't need knowledge_graph.py

Extracts subgraphs starting from entry point procedures, following CALLS relationships
"""

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Any, Set, Optional
from collections import deque

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


def find_procedures_by_name(nodes: List[Dict], procedure_names: List[str]) -> List[Dict]:
    """Find procedure nodes by name (exact or partial match)"""
    procedures = [n for n in nodes if n.get('type') == 'procedure']
    found = []
    
    for proc_name in procedure_names:
        # Try exact match first
        exact_matches = [p for p in procedures if p.get('name') == proc_name]
        if exact_matches:
            found.extend(exact_matches)
            continue
        
        # Try case-insensitive match
        pattern = re.compile(re.escape(proc_name), re.IGNORECASE)
        partial_matches = [p for p in procedures if pattern.search(p.get('name', ''))]
        if partial_matches:
            found.extend(partial_matches)
    
    return found


def build_call_graph_index(edges: List[Dict]) -> Dict[str, Dict[str, List[str]]]:
    """
    Build an index of call relationships for fast traversal
    
    Returns:
        Dict with 'outgoing' and 'incoming' mappings from node_id to list of connected node_ids
    """
    outgoing = {}  # node_id -> [called_node_ids]
    incoming = {}  # node_id -> [caller_node_ids]
    
    for edge in edges:
        edge_type = edge.get('type', '')
        if edge_type != 'calls':
            continue
        
        source_id = edge.get('source') or edge.get('source_id')
        target_id = edge.get('target') or edge.get('target_id')
        
        # Handle case where source/target might be objects
        if isinstance(source_id, dict):
            source_id = source_id.get('id')
        if isinstance(target_id, dict):
            target_id = target_id.get('id')
        
        if not source_id or not target_id:
            continue
        
        # Build outgoing index (source calls target)
        if source_id not in outgoing:
            outgoing[source_id] = []
        outgoing[source_id].append(target_id)
        
        # Build incoming index (target is called by source)
        if target_id not in incoming:
            incoming[target_id] = []
        incoming[target_id].append(source_id)
    
    return {'outgoing': outgoing, 'incoming': incoming}


def extract_call_subgraph(
    nodes: List[Dict],
    edges: List[Dict],
    entry_procedures: List[Dict],
    exclude_procedures: Set[str],
    max_depth: int = 10,
    include_variables: bool = True,
    include_structures: bool = True
) -> Dict[str, Any]:
    """
    Extract subgraph by traversing CALLS relationships from entry points
    
    Args:
        nodes: All nodes in graph
        edges: All edges in graph
        entry_procedures: List of entry point procedure nodes
        exclude_procedures: Set of procedure names to exclude
        max_depth: Maximum call depth to traverse
        include_variables: Include variables contained in procedures
        include_structures: Include structure definitions
    
    Returns:
        Subgraph with nodes and edges
    """
    node_by_id = {n['id']: n for n in nodes}
    call_index = build_call_graph_index(edges)
    
    # Track visited procedures and subgraph
    visited_procedures = set()
    subgraph_node_ids = set()
    call_depth_map = {}  # node_id -> depth from entry
    
    # BFS from entry points
    queue = deque()
    for entry_proc in entry_procedures:
        queue.append((entry_proc['id'], 0))
        call_depth_map[entry_proc['id']] = 0
        subgraph_node_ids.add(entry_proc['id'])
    
    print(f"\nTraversing call graph from {len(entry_procedures)} entry points...")
    depth_stats = {}
    
    while queue:
        current_id, depth = queue.popleft()
        
        if current_id in visited_procedures:
            continue
        
        if depth > max_depth:
            continue
        
        visited_procedures.add(current_id)
        depth_stats[depth] = depth_stats.get(depth, 0) + 1
        
        current_node = node_by_id.get(current_id)
        if not current_node:
            continue
        
        # Skip excluded procedures
        if current_node.get('name') in exclude_procedures:
            continue
        
        # Get procedures called by this one
        called_ids = call_index['outgoing'].get(current_id, [])
        
        for called_id in called_ids:
            called_node = node_by_id.get(called_id)
            if not called_node:
                continue
            
            # Skip excluded procedures
            if called_node.get('name') in exclude_procedures:
                continue
            
            # Skip if not a procedure
            if called_node.get('type') != 'procedure':
                continue
            
            # Add to subgraph
            if called_id not in subgraph_node_ids:
                subgraph_node_ids.add(called_id)
                call_depth_map[called_id] = depth + 1
                queue.append((called_id, depth + 1))
    
    print(f"  Found {len(visited_procedures)} procedures in call chain")
    for d in sorted(depth_stats.keys()):
        print(f"    Depth {d}: {depth_stats[d]} procedures")
    
    # Add variables if requested
    variable_count = 0
    if include_variables:
        for proc_id in list(subgraph_node_ids):
            # Find CONTAINS relationships from this procedure
            for edge in edges:
                if edge.get('type') != 'contains':
                    continue
                
                source_id = edge.get('source') or edge.get('source_id')
                target_id = edge.get('target') or edge.get('target_id')
                
                if isinstance(source_id, dict):
                    source_id = source_id.get('id')
                if isinstance(target_id, dict):
                    target_id = target_id.get('id')
                
                if source_id == proc_id:
                    target_node = node_by_id.get(target_id)
                    if target_node and target_node.get('type') == 'variable':
                        if target_id not in subgraph_node_ids:
                            subgraph_node_ids.add(target_id)
                            variable_count += 1
    
    if variable_count > 0:
        print(f"  Added {variable_count} variables")
    
    # Add structures if requested
    structure_count = 0
    if include_structures:
        # Get files from procedures
        proc_files = set()
        for proc_id in visited_procedures:
            proc_node = node_by_id.get(proc_id)
            if proc_node and proc_node.get('file_path'):
                proc_files.add(proc_node['file_path'])
        
        # Add structures from same files
        for node in nodes:
            if node.get('type') == 'structure':
                if node.get('file_path') in proc_files:
                    if node['id'] not in subgraph_node_ids:
                        subgraph_node_ids.add(node['id'])
                        structure_count += 1
    
    if structure_count > 0:
        print(f"  Added {structure_count} structures")
    
    # Build subgraph nodes
    subgraph_nodes = []
    for node_id in subgraph_node_ids:
        if node_id in node_by_id:
            node = node_by_id[node_id].copy()
            
            # Ensure metadata is a dict
            metadata = node.get('metadata', {})
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except:
                    metadata = {}
            
            # Add metadata
            metadata['is_entry_point'] = any(e['id'] == node_id for e in entry_procedures)
            if node_id in call_depth_map:
                metadata['call_depth'] = call_depth_map[node_id]
            
            node['metadata'] = metadata
            subgraph_nodes.append(node)
    
    # Filter edges to only those in subgraph
    subgraph_edges = []
    for edge in edges:
        source_id = edge.get('source') or edge.get('source_id')
        target_id = edge.get('target') or edge.get('target_id')
        
        if isinstance(source_id, dict):
            source_id = source_id.get('id')
        if isinstance(target_id, dict):
            target_id = target_id.get('id')
        
        if source_id in subgraph_node_ids and target_id in subgraph_node_ids:
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
            'extraction_type': 'call_chain',
            'entry_points': [e.get('name') for e in entry_procedures],
            'excluded_procedures': list(exclude_procedures),
            'max_depth': max_depth,
            'procedure_count': len(visited_procedures),
            'variable_count': variable_count,
            'structure_count': structure_count,
            'total_nodes': len(subgraph_nodes),
            'total_edges': len(subgraph_edges)
        }
    }


def find_common_procedures(nodes: List[Dict], edges: List[Dict], min_callers: int = 5) -> List[Dict[str, Any]]:
    """
    Find procedures that are called by many other procedures (likely utilities)
    
    Args:
        nodes: All nodes
        edges: All edges
        min_callers: Minimum number of callers to report
    
    Returns:
        List of common procedures with caller counts
    """
    procedures = [n for n in nodes if n.get('type') == 'procedure']
    call_index = build_call_graph_index(edges)
    node_by_id = {n['id']: n for n in nodes}
    
    common_procs = []
    
    for proc in procedures:
        callers = call_index['incoming'].get(proc['id'], [])
        caller_count = len(callers)
        
        if caller_count >= min_callers:
            caller_names = []
            for caller_id in callers[:10]:
                caller_node = node_by_id.get(caller_id)
                if caller_node:
                    caller_names.append(caller_node.get('name', 'unknown'))
            
            common_procs.append({
                'name': proc.get('name'),
                'caller_count': caller_count,
                'sample_callers': caller_names
            })
    
    common_procs.sort(key=lambda x: x['caller_count'], reverse=True)
    return common_procs


def display_common_procedures(common_procs: List[Dict[str, Any]], limit: int = 20):
    """Display common utility procedures"""
    print(f"\n{'='*70}")
    print(f"COMMON UTILITY PROCEDURES (potential exclusions)")
    print(f"{'='*70}\n")
    
    for proc in common_procs[:limit]:
        print(f"  {proc['name']:40} - called by {proc['caller_count']:3} procedures")
        print(f"    Sample callers: {', '.join(proc['sample_callers'][:5])}")
        print()
    
    if len(common_procs) > limit:
        print(f"  ... and {len(common_procs) - limit} more\n")


def analyze_call_chains(nodes: List[Dict], edges: List[Dict], 
                       entry_procedures: List[Dict],
                       exclude_procedures: Set[str],
                       max_paths: int = 10) -> List[List[str]]:
    """
    Find interesting call chains from entry procedures
    
    Args:
        nodes: All nodes
        edges: All edges
        entry_procedures: Entry point procedures
        exclude_procedures: Procedures to exclude
        max_paths: Maximum number of paths to show
    
    Returns:
        List of call paths (each path is a list of procedure names)
    """
    node_by_id = {n['id']: n for n in nodes}
    call_index = build_call_graph_index(edges)
    
    # Find interesting paths (avoiding excluded procedures)
    interesting_paths = []
    
    for entry_proc in entry_procedures:
        # DFS to find paths
        def dfs_paths(current_id: str, path: List[str], visited: Set[str], depth: int):
            if depth > 5 or len(interesting_paths) >= max_paths:
                return
            
            current_node = node_by_id.get(current_id)
            if not current_node:
                return
            
            current_name = current_node.get('name')
            
            # Skip excluded
            if current_name in exclude_procedures:
                return
            
            # Add to path
            new_path = path + [current_name]
            new_visited = visited | {current_id}
            
            # Get callees
            callees = call_index['outgoing'].get(current_id, [])
            
            if not callees:
                # Leaf node - save path if interesting (length > 2)
                if len(new_path) > 2:
                    interesting_paths.append(new_path)
            else:
                # Continue exploring
                for callee_id in callees:
                    if callee_id not in new_visited:
                        dfs_paths(callee_id, new_path, new_visited, depth + 1)
        
        dfs_paths(entry_proc['id'], [], set(), 0)
    
    return interesting_paths[:max_paths]


def export_call_subgraph(
    json_file: str,
    entry_points: List[str],
    output_dir: str = "./output",
    exclude: Optional[List[str]] = None,
    max_depth: int = 10,
    include_variables: bool = True,
    include_structures: bool = True,
    show_common: bool = False
) -> str:
    """
    Main export function for call chain subgraphs
    
    Args:
        json_file: Path to full knowledge graph JSON
        entry_points: List of entry point procedure names
        output_dir: Output directory
        exclude: List of procedure names to exclude
        max_depth: Maximum call depth
        include_variables: Include variables
        include_structures: Include structures
        show_common: Show common utility procedures
    
    Returns:
        Path to output file
    """
    print(f"\n{'='*70}")
    print(f"EXTRACTING CALL SUBGRAPH")
    print(f"{'='*70}\n")
    print(f"Entry points: {', '.join(entry_points)}")
    if exclude:
        print(f"Excluding: {', '.join(exclude)}")
    print(f"Max depth: {max_depth}")
    print(f"{'='*70}\n")
    
    # Load graph
    graph_data = load_graph_json(json_file)
    
    # Show common procedures if requested
    if show_common:
        common_procs = find_common_procedures(graph_data['nodes'], graph_data['edges'])
        display_common_procedures(common_procs)
        print("\nSuggested exclusions for next run:")
        print(f"  --exclude {' '.join([p['name'] for p in common_procs[:10]])}")
        print()
        return None
    
    # Find entry procedure nodes
    entry_proc_nodes = find_procedures_by_name(graph_data['nodes'], entry_points)
    
    if not entry_proc_nodes:
        print(f"\n❌ No procedures found matching: {', '.join(entry_points)}")
        
        # Show available procedures
        all_procs = [n for n in graph_data['nodes'] if n.get('type') == 'procedure']
        print(f"\nAvailable procedures ({len(all_procs)} total):")
        for proc in all_procs[:20]:
            print(f"  • {proc.get('name')}")
        if len(all_procs) > 20:
            print(f"  ... and {len(all_procs) - 20} more")
        
        sys.exit(1)
    
    print(f"Found {len(entry_proc_nodes)} entry procedures:")
    for proc in entry_proc_nodes:
        file_name = Path(proc.get('file_path', '')).name
        print(f"  ✓ {proc.get('name'):40} ({file_name})")
    
    # Build exclusion set
    exclude_set = set(exclude) if exclude else set()
    
    # Extract subgraph
    subgraph = extract_call_subgraph(
        graph_data['nodes'],
        graph_data['edges'],
        entry_proc_nodes,
        exclude_set,
        max_depth=max_depth,
        include_variables=include_variables,
        include_structures=include_structures
    )
    
    # Find interesting call chains
    print(f"\n{'='*70}")
    print("SAMPLE CALL CHAINS")
    print(f"{'='*70}\n")
    
    call_paths = analyze_call_chains(
        graph_data['nodes'],
        graph_data['edges'],
        entry_proc_nodes,
        exclude_set
    )
    
    for i, path in enumerate(call_paths[:10], 1):
        print(f"{i}. {' → '.join(path)}")
    
    if len(call_paths) > 10:
        print(f"\n... and {len(call_paths) - 10} more call paths")
    
    # Save to file
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Create filename from entry points
    safe_name = '_'.join([re.sub(r'[\W\-]+', '_', ep) for ep in entry_points[:3]])
    output_file = output_path / f"callgraph_{safe_name}.json"
    
    with open(output_file, 'w') as f:
        json.dump(subgraph, f, indent=2)
    
    # Display summary
    print(f"\n{'='*70}")
    print("EXPORT COMPLETE!")
    print(f"{'='*70}\n")
    print(f"Saved to: {output_file}\n")
    print("Subgraph summary:")
    meta = subgraph['metadata']
    print(f"  Entry points: {len(meta['entry_points'])}")
    print(f"  Procedures: {meta['procedure_count']}")
    print(f"  Variables: {meta['variable_count']}")
    print(f"  Structures: {meta['structure_count']}")
    print(f"  Total nodes: {meta['total_nodes']}")
    print(f"  Total edges: {meta['total_edges']}")
    
    # Node type breakdown
    type_counts = {}
    for node in subgraph['nodes']:
        node_type = node.get('type', 'unknown')
        type_counts[node_type] = type_counts.get(node_type, 0) + 1
    
    print(f"\n  Node types:")
    for node_type, count in sorted(type_counts.items()):
        print(f"    • {node_type}: {count}")
    
    print(f"\n{'='*70}\n")
    
    return str(output_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Extract call chain subgraph from entry procedures',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show common utility procedures (to know what to exclude)
  python3 subgraph.py graph.json --show-common
  
  # Extract subgraph from main procedure
  python3 subgraph.py graph.json --entry PAYMENT_MAIN
  
  # Multiple entry points, excluding common utilities
  python3 subgraph.py graph.json \\
    --entry PAYMENT_MAIN DRAWDOWN_PROCESS \\
    --exclude keyposition utility_log format_date
  
  # Deeper traversal
  python3 subgraph.py graph.json --entry MAIN_PROC --depth 15
  
  # Exclude variables for cleaner graph
  python3 subgraph.py graph.json --entry MAIN_PROC --no-variables
        """
    )

    parser.add_argument('json_file', help='Path to full knowledge graph JSON')
    parser.add_argument('--entry', nargs='+', metavar='PROC',
                       help='Entry point procedure names')
    parser.add_argument('--exclude', nargs='+', metavar='PROC',
                       help='Exclude common utility procedures')
    parser.add_argument('--depth', type=int, default=10,
                       help='Maximum call depth (default: 10)')
    parser.add_argument('--no-variables', action='store_true',
                       help='Exclude variables from subgraph')
    parser.add_argument('--no-structures', action='store_true',
                       help='Exclude structures from subgraph')
    parser.add_argument('-o', '--output', default='./output',
                       help='Output directory (default: ./output)')
    parser.add_argument('--show-common', action='store_true',
                       help='Show common utility procedures and exit')

    args = parser.parse_args()

    # Check file exists
    if not Path(args.json_file).exists():
        print(f"Error: File not found: {args.json_file}")
        sys.exit(1)

    # Validate arguments
    if not args.show_common and not args.entry:
        print("Error: Must specify --entry or --show-common")
        parser.print_help()
        sys.exit(1)

    try:
        output_file = export_call_subgraph(
            args.json_file,
            entry_points=args.entry or [],
            output_dir=args.output,
            exclude=args.exclude,
            max_depth=args.depth,
            include_variables=not args.no_variables,
            include_structures=not args.no_structures,
            show_common=args.show_common
        )
        
        if output_file:
            print(f"✓ Success! Visualize with:")
            print(f"  python3 tal_graph_viz.py {output_file}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
