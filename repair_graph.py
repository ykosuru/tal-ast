#!/usr/bin/env python3
"""
Diagnostic and Repair Tool for Missing Procedure File Entities

Identifies procedures with missing/incorrect file_path and attempts to resolve them
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Set
from collections import defaultdict


def load_graph(filepath: str) -> Dict[str, Any]:
    """Load knowledge graph JSON"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    if 'entities' in data:
        return {'nodes': data['entities'], 'edges': data['relationships']}
    return {'nodes': data['nodes'], 'edges': data['edges']}


def diagnose_missing_file_entities(graph: Dict[str, Any]) -> Dict[str, Any]:
    """
    Diagnose procedures with missing or invalid file_path
    
    Returns diagnostic report
    """
    nodes = graph['nodes']
    edges = graph['edges']
    
    # Separate procedures by status
    procedures = [n for n in nodes if n.get('type') == 'procedure']
    
    missing_file = []
    external_refs = []
    valid_procs = []
    
    for proc in procedures:
        file_path = proc.get('file_path')
        is_external = proc.get('metadata', {}).get('is_external', False)
        
        if is_external or not file_path or file_path == '':
            if 'external::' in proc.get('qualified_name', ''):
                external_refs.append(proc)
            else:
                missing_file.append(proc)
        else:
            valid_procs.append(proc)
    
    # Build call graph statistics
    node_by_id = {n['id']: n for n in nodes}
    
    # Count how many procedures call the problematic ones
    callers_of_missing = defaultdict(list)
    callers_of_external = defaultdict(list)
    
    for edge in edges:
        if edge.get('type') != 'calls':
            continue
        
        source_id = edge.get('source') or edge.get('source_id')
        target_id = edge.get('target') or edge.get('target_id')
        
        target = node_by_id.get(target_id)
        source = node_by_id.get(source_id)
        
        if target and source:
            target_name = target.get('name')
            source_name = source.get('name')
            
            # Check if target is problematic
            if target in missing_file:
                callers_of_missing[target_name].append(source_name)
            elif target in external_refs:
                callers_of_external[target_name].append(source_name)
    
    # Find procedures defined in same files as callers
    proc_by_file = defaultdict(list)
    for proc in valid_procs:
        if proc.get('file_path'):
            proc_by_file[proc['file_path']].append(proc['name'])
    
    return {
        'total_procedures': len(procedures),
        'valid_procedures': len(valid_procs),
        'missing_file_path': len(missing_file),
        'external_references': len(external_refs),
        'missing_procedures': missing_file,
        'external_procedures': external_refs,
        'callers_of_missing': dict(callers_of_missing),
        'callers_of_external': dict(callers_of_external),
        'procedures_by_file': dict(proc_by_file),
        'valid_procedures_list': valid_procs
    }


def find_procedure_in_graph(proc_name: str, graph: Dict[str, Any]) -> List[Dict]:
    """
    Search for a procedure by name in the knowledge graph
    
    Returns all matching procedure nodes
    """
    nodes = graph['nodes']
    matches = []
    
    for node in nodes:
        if node.get('type') != 'procedure':
            continue
        
        if node.get('name', '').lower() == proc_name.lower():
            matches.append(node)
    
    return matches


def suggest_file_locations(proc_name: str, 
                           callers: List[str],
                           proc_by_file: Dict[str, List[str]]) -> List[str]:
    """
    Suggest possible file locations based on callers
    
    If callers are in the same file, the missing procedure might be there too
    """
    suggestions = []
    
    for file_path, procs_in_file in proc_by_file.items():
        # Check if any caller is in this file
        for caller in callers:
            if caller in procs_in_file:
                suggestions.append(file_path)
                break
    
    return list(set(suggestions))


def repair_external_references(graph: Dict[str, Any]) -> Dict[str, Any]:
    """
    Attempt to repair external references by matching them to real procedures
    
    Returns repaired graph
    """
    nodes = graph['nodes']
    edges = graph['edges']
    
    # Build lookup tables
    node_by_id = {n['id']: n for n in nodes}
    
    # Find all real procedures (non-external, with file_path)
    real_procs = {}
    for node in nodes:
        if node.get('type') == 'procedure':
            is_external = node.get('metadata', {}).get('is_external', False)
            has_file = node.get('file_path') and node['file_path'] != ''
            
            if not is_external and has_file:
                name = node['name']
                if name not in real_procs:  # Keep first occurrence
                    real_procs[name] = node
    
    # Find external references
    external_nodes = []
    for node in nodes:
        if node.get('type') == 'procedure':
            is_external = node.get('metadata', {}).get('is_external', False)
            if is_external or 'external::' in node.get('qualified_name', ''):
                external_nodes.append(node)
    
    # Track repairs
    repairs = []
    id_mapping = {}  # old_id -> new_id
    
    for ext_node in external_nodes:
        ext_name = ext_node['name']
        ext_id = ext_node['id']
        
        if ext_name in real_procs:
            real_node = real_procs[ext_name]
            repairs.append({
                'external_name': ext_name,
                'external_id': ext_id,
                'resolved_to_id': real_node['id'],
                'resolved_file': real_node['file_path']
            })
            id_mapping[ext_id] = real_node['id']
    
    # Update edges to point to real procedures
    updated_edges = []
    for edge in edges:
        source_id = edge.get('source') or edge.get('source_id')
        target_id = edge.get('target') or edge.get('target_id')
        
        # Remap IDs
        if source_id in id_mapping:
            source_id = id_mapping[source_id]
        if target_id in id_mapping:
            target_id = id_mapping[target_id]
        
        updated_edge = {
            'source': source_id,
            'target': target_id,
            'type': edge.get('type', 'unknown'),
            'weight': edge.get('weight', 1.0),
            'metadata': edge.get('metadata', {})
        }
        
        # Mark as resolved
        if edge.get('target') in id_mapping or edge.get('target_id') in id_mapping:
            updated_edge['metadata']['resolved'] = True
            updated_edge['metadata']['originally_external'] = True
        
        updated_edges.append(updated_edge)
    
    # Remove duplicate edges after remapping
    edge_keys = set()
    unique_edges = []
    
    for edge in updated_edges:
        key = (edge['source'], edge['target'], edge['type'])
        if key not in edge_keys:
            edge_keys.add(key)
            unique_edges.append(edge)
    
    # Remove external nodes that were resolved
    resolved_ids = set(id_mapping.keys())
    updated_nodes = [n for n in nodes if n['id'] not in resolved_ids]
    
    return {
        'nodes': updated_nodes,
        'edges': unique_edges,
        'repairs': repairs,
        'removed_external_nodes': len(resolved_ids),
        'duplicate_edges_removed': len(updated_edges) - len(unique_edges)
    }


def display_diagnostic_report(report: Dict[str, Any]):
    """Display comprehensive diagnostic report"""
    print(f"\n{'='*70}")
    print("DIAGNOSTIC REPORT - Missing File Entities")
    print(f"{'='*70}\n")
    
    print(f"📊 PROCEDURE STATISTICS:")
    print(f"  Total procedures: {report['total_procedures']}")
    print(f"  Valid (with file_path): {report['valid_procedures']}")
    print(f"  Missing file_path: {report['missing_file_path']}")
    print(f"  External references: {report['external_references']}")
    
    # Missing file paths
    if report['missing_procedures']:
        print(f"\n❌ PROCEDURES MISSING FILE_PATH ({len(report['missing_procedures'])}):")
        for proc in report['missing_procedures'][:10]:
            print(f"\n  • {proc['name']}")
            print(f"    ID: {proc['id']}")
            print(f"    Qualified name: {proc.get('qualified_name', 'N/A')}")
            
            # Show who calls it
            callers = report['callers_of_missing'].get(proc['name'], [])
            if callers:
                print(f"    Called by: {', '.join(callers[:5])}")
                
                # Suggest file locations
                suggestions = suggest_file_locations(
                    proc['name'],
                    callers,
                    report['procedures_by_file']
                )
                if suggestions:
                    print(f"    Likely in files:")
                    for sugg in suggestions[:3]:
                        print(f"      → {Path(sugg).name}")
        
        if len(report['missing_procedures']) > 10:
            print(f"\n  ... and {len(report['missing_procedures']) - 10} more")
    
    # External references
    if report['external_procedures']:
        print(f"\n🌐 EXTERNAL REFERENCES ({len(report['external_procedures'])}):")
        
        # Check which ones might be resolvable
        resolvable = []
        unresolvable = []
        
        valid_proc_names = {p['name'] for p in report['valid_procedures_list']}
        
        for ext_proc in report['external_procedures']:
            if ext_proc['name'] in valid_proc_names:
                resolvable.append(ext_proc)
            else:
                unresolvable.append(ext_proc)
        
        print(f"  Resolvable (found in graph): {len(resolvable)}")
        print(f"  Unresolvable (truly external): {len(unresolvable)}")
        
        if resolvable:
            print(f"\n  ✓ Can be resolved:")
            for proc in resolvable[:5]:
                print(f"    • {proc['name']}")
                callers = report['callers_of_external'].get(proc['name'], [])
                if callers:
                    print(f"      Called by: {', '.join(callers[:3])}")
            if len(resolvable) > 5:
                print(f"    ... and {len(resolvable) - 5} more")
        
        if unresolvable:
            print(f"\n  ⚠ Truly external (not in any file):")
            for proc in unresolvable[:10]:
                print(f"    • {proc['name']}")
                callers = report['callers_of_external'].get(proc['name'], [])
                if callers:
                    print(f"      Called by: {', '.join(callers[:3])}")
            if len(unresolvable) > 10:
                print(f"    ... and {len(unresolvable) - 10} more")
    
    print(f"\n{'='*70}\n")


def main():
    """Main diagnostic and repair tool"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Diagnose and repair missing file entities in knowledge graph',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Diagnose issues
  python3 diagnose_graph.py knowledge_graph.json
  
  # Repair external references
  python3 diagnose_graph.py knowledge_graph.json --repair -o repaired.json
  
  # Search for specific procedure
  python3 diagnose_graph.py knowledge_graph.json --find PROC_NAME
        """
    )
    
    parser.add_argument('graph_file', help='Knowledge graph JSON file')
    parser.add_argument('--repair', action='store_true',
                       help='Attempt to repair external references')
    parser.add_argument('--find', metavar='PROC_NAME',
                       help='Search for a specific procedure')
    parser.add_argument('-o', '--output',
                       help='Output file for repaired graph')
    
    args = parser.parse_args()
    
    # Validate input
    if not Path(args.graph_file).exists():
        print(f"Error: File not found: {args.graph_file}")
        sys.exit(1)
    
    try:
        # Load graph
        print(f"Loading: {args.graph_file}")
        graph = load_graph(args.graph_file)
        print(f"  Nodes: {len(graph['nodes'])}")
        print(f"  Edges: {len(graph['edges'])}")
        
        # Search for specific procedure
        if args.find:
            print(f"\n{'='*70}")
            print(f"SEARCHING FOR: {args.find}")
            print(f"{'='*70}\n")
            
            matches = find_procedure_in_graph(args.find, graph)
            
            if not matches:
                print(f"❌ No procedure found matching '{args.find}'")
            else:
                print(f"✓ Found {len(matches)} match(es):\n")
                for i, proc in enumerate(matches, 1):
                    print(f"{i}. {proc['name']}")
                    print(f"   ID: {proc['id']}")
                    print(f"   Qualified name: {proc.get('qualified_name', 'N/A')}")
                    print(f"   File: {proc.get('file_path', 'MISSING')}")
                    print(f"   Line: {proc.get('start_line', 'N/A')}")
                    
                    metadata = proc.get('metadata', {})
                    if metadata.get('is_external'):
                        print(f"   Status: EXTERNAL REFERENCE")
                    
                    # Show callers
                    callers = []
                    for edge in graph['edges']:
                        if edge.get('type') == 'calls':
                            target_id = edge.get('target') or edge.get('target_id')
                            if target_id == proc['id']:
                                source_id = edge.get('source') or edge.get('source_id')
                                node_by_id = {n['id']: n for n in graph['nodes']}
                                caller = node_by_id.get(source_id)
                                if caller:
                                    callers.append(caller['name'])
                    
                    if callers:
                        print(f"   Called by: {', '.join(callers[:5])}")
                        if len(callers) > 5:
                            print(f"   ... and {len(callers) - 5} more")
                    print()
            
            sys.exit(0)
        
        # Run diagnostics
        print(f"\nRunning diagnostics...")
        report = diagnose_missing_file_entities(graph)
        display_diagnostic_report(report)
        
        # Repair if requested
        if args.repair:
            print(f"\n{'='*70}")
            print("REPAIRING EXTERNAL REFERENCES")
            print(f"{'='*70}\n")
            
            result = repair_external_references(graph)
            
            print(f"Repair summary:")
            print(f"  Resolved external refs: {len(result['repairs'])}")
            print(f"  Removed external nodes: {result['removed_external_nodes']}")
            print(f"  Duplicate edges removed: {result['duplicate_edges_removed']}")
            
            if result['repairs']:
                print(f"\n✓ Resolved references:")
                for repair in result['repairs'][:10]:
                    print(f"    • {repair['external_name']}")
                    print(f"      → {Path(repair['resolved_file']).name}")
                if len(result['repairs']) > 10:
                    print(f"    ... and {len(result['repairs']) - 10} more")
            
            # Save repaired graph
            if args.output:
                output_path = Path(args.output)
                output_path.parent.mkdir(exist_ok=True, parents=True)
                
                output_data = {
                    'nodes': result['nodes'],
                    'edges': result['edges'],
                    'metadata': {
                        'repair_info': {
                            'source_file': args.graph_file,
                            'repairs_made': len(result['repairs']),
                            'nodes_removed': result['removed_external_nodes'],
                            'edges_deduplicated': result['duplicate_edges_removed']
                        }
                    }
                }
                
                with open(output_path, 'w') as f:
                    json.dump(output_data, f, indent=2)
                
                print(f"\n✓ Repaired graph saved to: {output_path}")
            else:
                print(f"\n⚠ Use --output to save repaired graph")
        
        # Recommendations
        print(f"\n{'='*70}")
        print("RECOMMENDATIONS")
        print(f"{'='*70}\n")
        
        if report['external_references'] > 0:
            print("To repair external references:")
            print(f"  python3 diagnose_graph.py {args.graph_file} --repair -o repaired.json\n")
        
        if report['missing_file_path'] > 0:
            print("To find a specific procedure:")
            print(f"  python3 diagnose_graph.py {args.graph_file} --find PROC_NAME\n")
        
        print("To use the repaired graph with subgraph.py:")
        print(f"  python3 subgraph.py repaired.json --entry MAIN_PROC\n")
        
        print(f"{'='*70}\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
