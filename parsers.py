"""
AST Parser Interface for Knowledge Graph - TAL Parser
Modified to export graph data to files for visualization

Features:
- Parse TAL AST into knowledge graph entities and relationships
- Handle external procedure references
- Export graph data to JSON for visualization
- Support for multiple file parsing
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Set, Tuple
from pathlib import Path
import re
from dataclasses import dataclass
import logging
import sys
import json

from knowledge_graph import (
    Entity, Relationship, EntityType, RelationType, KnowledgeGraph
)

# Import the TAL parser modules
try:
    import tal_proc_parser
    from enhanced_tal_parser import EnhancedTALParser
    TAL_PARSERS_AVAILABLE = True
except ImportError:
    TAL_PARSERS_AVAILABLE = False
    logging.warning("TAL parser modules not available")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Graph Export Functions (NEW)
# ============================================================================

def export_for_visualization(kg: KnowledgeGraph, 
                             include_files: bool = False,
                             include_variables: bool = False,
                             max_nodes: int = 500) -> Dict[str, Any]:
    """
    Export knowledge graph in visualization-ready format
    
    Args:
        kg: Knowledge graph to export
        include_files: Include file entities
        include_variables: Include variable entities
        max_nodes: Maximum number of nodes to export
    
    Returns:
        Dict with nodes and edges ready for visualization
    """
    all_entities = kg.query_entities()
    
    # Filter entities
    filtered_entities = []
    for entity in all_entities:
        if entity.type == EntityType.FILE and not include_files:
            continue
        if entity.type == EntityType.VARIABLE and not include_variables:
            continue
        filtered_entities.append(entity)
    
    # Limit nodes
    if len(filtered_entities) > max_nodes:
        logger.info(f"Limiting to {max_nodes} nodes (found {len(filtered_entities)})")
        procedures = [e for e in filtered_entities if e.type == EntityType.PROCEDURE]
        others = [e for e in filtered_entities if e.type != EntityType.PROCEDURE]
        filtered_entities = procedures[:int(max_nodes * 0.8)] + others[:int(max_nodes * 0.2)]
    
    # Get relationships
    all_relationships = kg.query_relationships()
    entity_ids = {e.id for e in filtered_entities}
    filtered_relationships = [r for r in all_relationships 
                             if r.source_id in entity_ids and r.target_id in entity_ids]
    
    # Convert to serializable format
    nodes = []
    for entity in filtered_entities:
        node = {
            'id': entity.id,
            'name': entity.name,
            'type': entity.type.value,
            'qualified_name': entity.qualified_name,
            'file_path': entity.file_path,
            'start_line': entity.start_line,
            'end_line': entity.end_line,
            'language': entity.language,
            'metadata': entity.metadata
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
    
    return {
        'nodes': nodes,
        'edges': edges,
        'statistics': kg.get_statistics(),
        'metadata': {
            'total_entities': len(all_entities),
            'filtered_entities': len(filtered_entities),
            'total_relationships': len(all_relationships),
            'filtered_relationships': len(filtered_relationships),
            'include_files': include_files,
            'include_variables': include_variables
        }
    }


def export_knowledge_graph(kg: KnowledgeGraph, output_dir: str = "./output") -> str:
    """
    Export knowledge graph in multiple formats including visualization-ready JSON
    
    Args:
        kg: Knowledge graph to export
        output_dir: Output directory for export files
    
    Returns:
        Path to visualization data file
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    print(f"\n{'='*70}")
    print("EXPORTING KNOWLEDGE GRAPH")
    print(f"{'='*70}\n")
    
    # Export full graph to JSON
    json_file = output_path / "knowledge_graph.json"
    kg.save_to_json(str(json_file))
    print(f"✓ Full graph: {json_file}")
    
    # Export visualization-ready format (THIS IS THE KEY ONE)
    vis_data = export_for_visualization(kg, include_files=False, include_variables=False)
    vis_file = output_path / "graph_data.json"
    with open(vis_file, 'w') as f:
        json.dump(vis_data, f, indent=2)
    print(f"✓ Visualization data: {vis_file}")
    
    # Export procedures summary
    procedures = kg.query_entities(entity_type=EntityType.PROCEDURE)
    procedures_data = []
    for proc in procedures:
        proc_info = {
            'name': proc.name,
            'file': proc.file_path,
            'line': proc.start_line,
            'return_type': proc.metadata.get('return_type'),
            'parameters': proc.metadata.get('parameters', []),
            'is_main': proc.metadata.get('is_main', False),
            'is_external': proc.metadata.get('is_external', False),
            'statement_count': proc.metadata.get('statement_count', 0)
        }
        procedures_data.append(proc_info)
    
    proc_file = output_path / "procedures.json"
    with open(proc_file, 'w') as f:
        json.dump(procedures_data, f, indent=2)
    print(f"✓ Procedures: {proc_file}")
    
    # Export call graph
    call_relationships = kg.query_relationships(rel_type=RelationType.CALLS)
    call_graph_data = []
    for rel in call_relationships:
        caller = kg.get_entity(rel.source_id)
        callee = kg.get_entity(rel.target_id)
        if caller and callee:
            call_graph_data.append({
                'caller': caller.name,
                'caller_file': caller.file_path,
                'callee': callee.name,
                'callee_file': callee.file_path,
                'line': rel.metadata.get('line'),
                'external': rel.metadata.get('external', False)
            })
    
    call_file = output_path / "call_graph.json"
    with open(call_file, 'w') as f:
        json.dump(call_graph_data, f, indent=2)
    print(f"✓ Call graph: {call_file}")
    
    # Export statistics
    stats = kg.get_statistics()
    stats_file = output_path / "statistics.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"✓ Statistics: {stats_file}")
    
    print(f"\n{'='*70}\n")
    
    return str(vis_file)


# Note: Include all the other parser code from the original file here
# (TALParser, KnowledgeGraphSearch, etc.)
# For brevity, I'm showing just the new/modified functions

if __name__ == "__main__":
    print("Updated parsers.py with graph export functionality")
    print("Use export_knowledge_graph(kg, output_dir) to export graph data")
