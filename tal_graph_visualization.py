"""
GraphViz Visualization for Knowledge Graph
Generates visual representations of the parsed TAL code structure

Features:
- Full codebase visualization showing all procedures and relationships
- Filtered subgraph visualization for specific functionality
- Multiple layout algorithms (hierarchical, radial, etc.)
- Color-coded by entity type
- Export to PNG, SVG, PDF formats
"""

from typing import Dict, List, Optional, Set, Any
from pathlib import Path
import logging

try:
    import graphviz
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False
    logging.warning("graphviz not installed. Install with: pip install graphviz")

from knowledge_graph import (
    Entity, Relationship, EntityType, RelationType, KnowledgeGraph
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# GraphViz Visualizer
# ============================================================================

class KnowledgeGraphVisualizer:
    """Generate GraphViz visualizations of the knowledge graph"""
    
    # Color schemes for different entity types
    ENTITY_COLORS = {
        EntityType.FILE: '#E8F4F8',
        EntityType.PROCEDURE: '#C3E6CB',
        EntityType.FUNCTION: '#C3E6CB',
        EntityType.VARIABLE: '#FFF3CD',
        EntityType.STRUCTURE: '#D4EDDA',
        EntityType.CONSTANT: '#F8D7DA',
    }
    
    ENTITY_SHAPES = {
        EntityType.FILE: 'folder',
        EntityType.PROCEDURE: 'box',
        EntityType.FUNCTION: 'box',
        EntityType.VARIABLE: 'ellipse',
        EntityType.STRUCTURE: 'component',
        EntityType.CONSTANT: 'diamond',
    }
    
    RELATIONSHIP_STYLES = {
        RelationType.CALLS: {'color': 'blue', 'style': 'solid', 'arrowhead': 'normal'},
        RelationType.CONTAINS: {'color': 'gray', 'style': 'dashed', 'arrowhead': 'empty'},
        RelationType.DEFINES: {'color': 'green', 'style': 'solid', 'arrowhead': 'empty'},
        RelationType.REFERENCES: {'color': 'orange', 'style': 'dotted', 'arrowhead': 'vee'},
        RelationType.USES: {'color': 'purple', 'style': 'dotted', 'arrowhead': 'vee'},
    }
    
    def __init__(self, kg: KnowledgeGraph):
        """Initialize visualizer with knowledge graph"""
        if not GRAPHVIZ_AVAILABLE:
            raise ImportError("graphviz is not installed. Install with: pip install graphviz")
        
        self.kg = kg
    
    def visualize_full_graph(self, 
                            output_file: str = "knowledge_graph",
                            format: str = "png",
                            layout: str = "dot",
                            max_nodes: int = 200,
                            include_files: bool = False,
                            include_variables: bool = False) -> str:
        """
        Visualize the entire knowledge graph
        
        Args:
            output_file: Output filename (without extension)
            format: Output format (png, svg, pdf)
            layout: GraphViz layout engine (dot, neato, fdp, circo, twopi)
            max_nodes: Maximum nodes to include (for large graphs)
            include_files: Include file entities
            include_variables: Include variable entities
        
        Returns:
            Path to generated visualization file
        """
        dot = graphviz.Digraph(
            name='Knowledge Graph',
            comment='TAL Code Knowledge Graph',
            engine=layout,
            format=format
        )
        
        # Set graph attributes
        dot.attr(rankdir='TB', ranksep='1.0', nodesep='0.5')
        dot.attr('node', fontname='Arial', fontsize='10')
        dot.attr('edge', fontname='Arial', fontsize='8')
        
        # Get all entities
        all_entities = self.kg.query_entities()
        
        # Filter entities based on options
        filtered_entities = []
        for entity in all_entities:
            if entity.type == EntityType.FILE and not include_files:
                continue
            if entity.type == EntityType.VARIABLE and not include_variables:
                continue
            filtered_entities.append(entity)
        
        # Limit to max_nodes
        if len(filtered_entities) > max_nodes:
            logger.warning(f"Graph has {len(filtered_entities)} nodes, limiting to {max_nodes}")
            # Prioritize procedures
            procedures = [e for e in filtered_entities if e.type == EntityType.PROCEDURE]
            others = [e for e in filtered_entities if e.type != EntityType.PROCEDURE]
            filtered_entities = procedures[:int(max_nodes * 0.8)] + others[:int(max_nodes * 0.2)]
        
        # Create nodes
        entity_ids = set()
        for entity in filtered_entities:
            self._add_node(dot, entity)
            entity_ids.add(entity.id)
        
        # Get relationships for included entities
        all_relationships = self.kg.query_relationships()
        
        # Add edges
        for rel in all_relationships:
            if rel.source_id in entity_ids and rel.target_id in entity_ids:
                self._add_edge(dot, rel)
        
        # Add legend
        self._add_legend(dot, include_files, include_variables)
        
        # Render
        output_path = dot.render(output_file, cleanup=True)
        logger.info(f"Generated visualization: {output_path}")
        
        return output_path
    
    def visualize_procedure_subgraph(self,
                                    procedure_name: str,
                                    output_file: str = "procedure_graph",
                                    format: str = "png",
                                    layout: str = "dot",
                                    depth: int = 2,
                                    include_variables: bool = True) -> str:
        """
        Visualize a procedure and its immediate context
        
        Args:
            procedure_name: Name of the procedure to visualize
            output_file: Output filename (without extension)
            format: Output format (png, svg, pdf)
            layout: GraphViz layout engine
            depth: Relationship depth to include
            include_variables: Include local variables
        
        Returns:
            Path to generated visualization file
        """
        # Find the procedure
        procedures = self.kg.query_entities(entity_type=EntityType.PROCEDURE)
        target_proc = None
        for proc in procedures:
            if proc.name == procedure_name:
                target_proc = proc
                break
        
        if not target_proc:
            raise ValueError(f"Procedure '{procedure_name}' not found")
        
        dot = graphviz.Digraph(
            name=f'Procedure: {procedure_name}',
            comment=f'Context for {procedure_name}',
            engine=layout,
            format=format
        )
        
        # Set graph attributes
        dot.attr(rankdir='TB', ranksep='0.8', nodesep='0.5')
        dot.attr('node', fontname='Arial', fontsize='10')
        dot.attr('edge', fontname='Arial', fontsize='8')
        
        # Get related entities
        visited_entities = {target_proc.id: target_proc}
        visited_relationships = set()
        
        self._traverse_procedure_context(
            target_proc, 
            depth, 
            visited_entities, 
            visited_relationships,
            include_variables
        )
        
        # Add nodes
        for entity in visited_entities.values():
            self._add_node(dot, entity, highlight=(entity.id == target_proc.id))
        
        # Add edges
        for rel in visited_relationships:
            self._add_edge(dot, rel)
        
        # Add title
        with dot.subgraph(name='cluster_title') as c:
            c.attr(label=f'Procedure: {procedure_name}', fontsize='14', fontname='Arial Bold')
            c.attr(style='invis')
        
        # Render
        output_path = dot.render(output_file, cleanup=True)
        logger.info(f"Generated procedure visualization: {output_path}")
        
        return output_path
    
    def visualize_call_graph(self,
                           output_file: str = "call_graph",
                           format: str = "png",
                           layout: str = "dot",
                           main_only: bool = True,
                           max_depth: int = 5) -> str:
        """
        Visualize procedure call graph
        
        Args:
            output_file: Output filename (without extension)
            format: Output format (png, svg, pdf)
            layout: GraphViz layout engine
            main_only: Start from main procedures only
            max_depth: Maximum call depth to show
        
        Returns:
            Path to generated visualization file
        """
        dot = graphviz.Digraph(
            name='Call Graph',
            comment='TAL Procedure Call Graph',
            engine=layout,
            format=format
        )
        
        # Set graph attributes for call graph
        dot.attr(rankdir='LR', ranksep='1.5', nodesep='0.8')
        dot.attr('node', fontname='Arial', fontsize='10', shape='box')
        dot.attr('edge', fontname='Arial', fontsize='8')
        
        # Get procedures
        procedures = self.kg.query_entities(entity_type=EntityType.PROCEDURE)
        
        # Filter to main procedures if requested
        if main_only:
            start_procedures = [p for p in procedures if p.metadata.get('is_main')]
        else:
            start_procedures = procedures
        
        if not start_procedures:
            start_procedures = procedures[:10]  # Fallback
        
        # Build call graph
        visited_procs = set()
        proc_depths = {}
        
        for proc in start_procedures:
            self._traverse_calls(
                proc, 
                0, 
                max_depth, 
                visited_procs, 
                proc_depths
            )
        
        # Add nodes grouped by depth
        for depth in range(max_depth + 1):
            procs_at_depth = [pid for pid, d in proc_depths.items() if d == depth]
            if procs_at_depth:
                with dot.subgraph(name=f'cluster_{depth}') as c:
                    c.attr(label=f'Depth {depth}', style='dashed', color='gray')
                    for proc_id in procs_at_depth:
                        proc = self.kg.get_entity(proc_id)
                        if proc:
                            color = '#FFE6E6' if proc.metadata.get('is_external') else '#C3E6CB'
                            c.node(
                                proc.id,
                                label=self._get_node_label(proc),
                                fillcolor=color,
                                style='filled'
                            )
        
        # Add call edges
        for proc_id in visited_procs:
            rels = self.kg.query_relationships(
                source_id=proc_id,
                rel_type=RelationType.CALLS
            )
            for rel in rels:
                if rel.target_id in visited_procs:
                    style = 'dashed' if rel.metadata.get('external') else 'solid'
                    dot.edge(rel.source_id, rel.target_id, style=style, color='blue')
        
        # Render
        output_path = dot.render(output_file, cleanup=True)
        logger.info(f"Generated call graph: {output_path}")
        
        return output_path
    
    def visualize_file_structure(self,
                                output_file: str = "file_structure",
                                format: str = "png",
                                layout: str = "dot") -> str:
        """
        Visualize file structure with procedure definitions
        
        Args:
            output_file: Output filename (without extension)
            format: Output format (png, svg, pdf)
            layout: GraphViz layout engine
        
        Returns:
            Path to generated visualization file
        """
        dot = graphviz.Digraph(
            name='File Structure',
            comment='TAL File and Procedure Structure',
            engine=layout,
            format=format
        )
        
        # Set graph attributes
        dot.attr(rankdir='TB', ranksep='1.0', nodesep='0.5')
        dot.attr('node', fontname='Arial', fontsize='10')
        
        # Get files
        files = self.kg.query_entities(entity_type=EntityType.FILE)
        
        # Create a cluster for each file
        for file_entity in files:
            file_name = Path(file_entity.file_path).name if file_entity.file_path else file_entity.name
            
            with dot.subgraph(name=f'cluster_{file_entity.id}') as c:
                c.attr(label=file_name, style='rounded', color='lightblue')
                
                # Get procedures defined in this file
                procedures = self.kg.query_entities(entity_type=EntityType.PROCEDURE)
                file_procedures = [p for p in procedures if p.file_path == file_entity.file_path]
                
                # Add procedure nodes
                for proc in file_procedures:
                    color = '#FFE6E6' if proc.metadata.get('is_external') else '#C3E6CB'
                    c.node(
                        proc.id,
                        label=self._get_node_label(proc),
                        fillcolor=color,
                        style='filled',
                        shape='box'
                    )
        
        # Add call relationships between procedures
        rels = self.kg.query_relationships(rel_type=RelationType.CALLS)
        for rel in rels:
            style = 'dashed' if rel.metadata.get('external') else 'solid'
            dot.edge(rel.source_id, rel.target_id, style=style, color='blue', constraint='false')
        
        # Render
        output_path = dot.render(output_file, cleanup=True)
        logger.info(f"Generated file structure: {output_path}")
        
        return output_path
    
    def _add_node(self, dot, entity: Entity, highlight: bool = False):
        """Add a node to the graph"""
        label = self._get_node_label(entity)
        color = self.ENTITY_COLORS.get(entity.type, '#F0F0F0')
        shape = self.ENTITY_SHAPES.get(entity.type, 'box')
        
        # Highlight node if requested
        if highlight:
            dot.node(
                entity.id,
                label=label,
                fillcolor='#FFD700',
                style='filled,bold',
                shape=shape,
                penwidth='2'
            )
        elif entity.metadata.get('is_external'):
            dot.node(
                entity.id,
                label=label,
                fillcolor='#FFE6E6',
                style='filled,dashed',
                shape=shape
            )
        else:
            dot.node(
                entity.id,
                label=label,
                fillcolor=color,
                style='filled',
                shape=shape
            )
    
    def _add_edge(self, dot, relationship: Relationship):
        """Add an edge to the graph"""
        style_attrs = self.RELATIONSHIP_STYLES.get(
            relationship.type,
            {'color': 'black', 'style': 'solid', 'arrowhead': 'normal'}
        )
        
        # Add label if metadata available
        label = ''
        if relationship.metadata.get('line'):
            label = f"line {relationship.metadata['line']}"
        
        dot.edge(
            relationship.source_id,
            relationship.target_id,
            label=label,
            **style_attrs
        )
    
    def _get_node_label(self, entity: Entity) -> str:
        """Generate label for entity node"""
        label_parts = [entity.name]
        
        # Add metadata
        if entity.type == EntityType.PROCEDURE:
            if entity.metadata.get('is_main'):
                label_parts.append('★ MAIN')
            
            params = entity.metadata.get('parameters', [])
            if params:
                param_str = ', '.join(params[:3])
                if len(params) > 3:
                    param_str += '...'
                label_parts.append(f'({param_str})')
            
            ret_type = entity.metadata.get('return_type')
            if ret_type and ret_type != 'void':
                label_parts.append(f'→ {ret_type}')
        
        elif entity.type == EntityType.VARIABLE:
            data_type = entity.metadata.get('data_type', '')
            if data_type:
                label_parts.append(f': {data_type}')
        
        elif entity.type == EntityType.STRUCTURE:
            field_count = entity.metadata.get('field_count', 0)
            if field_count:
                label_parts.append(f'{field_count} fields')
        
        return '\n'.join(label_parts)
    
    def _add_legend(self, dot, include_files: bool, include_variables: bool):
        """Add legend to the graph"""
        with dot.subgraph(name='cluster_legend') as c:
            c.attr(label='Legend', style='rounded', color='gray')
            
            # Entity types
            if include_files:
                c.node('legend_file', label='File', fillcolor=self.ENTITY_COLORS[EntityType.FILE],
                      style='filled', shape=self.ENTITY_SHAPES[EntityType.FILE])
            
            c.node('legend_proc', label='Procedure', fillcolor=self.ENTITY_COLORS[EntityType.PROCEDURE],
                  style='filled', shape=self.ENTITY_SHAPES[EntityType.PROCEDURE])
            
            if include_variables:
                c.node('legend_var', label='Variable', fillcolor=self.ENTITY_COLORS[EntityType.VARIABLE],
                      style='filled', shape=self.ENTITY_SHAPES[EntityType.VARIABLE])
            
            c.node('legend_struct', label='Structure', fillcolor=self.ENTITY_COLORS[EntityType.STRUCTURE],
                  style='filled', shape=self.ENTITY_SHAPES[EntityType.STRUCTURE])
            
            c.node('legend_external', label='External', fillcolor='#FFE6E6',
                  style='filled,dashed', shape='box')
    
    def _traverse_procedure_context(self, 
                                   entity: Entity,
                                   depth: int,
                                   visited_entities: Dict[str, Entity],
                                   visited_relationships: Set[Relationship],
                                   include_variables: bool):
        """Recursively traverse procedure context"""
        if depth <= 0:
            return
        
        # Get relationships
        outgoing = self.kg.query_relationships(source_id=entity.id)
        incoming = self.kg.query_relationships(target_id=entity.id)
        
        for rel in outgoing + incoming:
            visited_relationships.add(rel)
            
            # Get the other entity
            other_id = rel.target_id if rel.source_id == entity.id else rel.source_id
            if other_id not in visited_entities:
                other = self.kg.get_entity(other_id)
                if other:
                    # Filter variables if needed
                    if not include_variables and other.type == EntityType.VARIABLE:
                        continue
                    
                    visited_entities[other_id] = other
                    
                    # Continue traversal for procedures
                    if other.type == EntityType.PROCEDURE and rel.type == RelationType.CALLS:
                        self._traverse_procedure_context(
                            other,
                            depth - 1,
                            visited_entities,
                            visited_relationships,
                            include_variables
                        )
    
    def _traverse_calls(self,
                       entity: Entity,
                       current_depth: int,
                       max_depth: int,
                       visited_procs: Set[str],
                       proc_depths: Dict[str, int]):
        """Recursively traverse call graph"""
        if current_depth > max_depth or entity.id in visited_procs:
            return
        
        visited_procs.add(entity.id)
        
        # Record depth (keep minimum depth)
        if entity.id not in proc_depths or current_depth < proc_depths[entity.id]:
            proc_depths[entity.id] = current_depth
        
        # Get called procedures
        rels = self.kg.query_relationships(
            source_id=entity.id,
            rel_type=RelationType.CALLS
        )
        
        for rel in rels:
            callee = self.kg.get_entity(rel.target_id)
            if callee and callee.type == EntityType.PROCEDURE:
                self._traverse_calls(
                    callee,
                    current_depth + 1,
                    max_depth,
                    visited_procs,
                    proc_depths
                )


# ============================================================================
# Convenience Functions
# ============================================================================

def visualize_knowledge_graph(kg: KnowledgeGraph,
                             output_dir: str = "./visualizations",
                             format: str = "png") -> Dict[str, str]:
    """
    Generate all standard visualizations
    
    Args:
        kg: Knowledge graph to visualize
        output_dir: Output directory for visualizations
        format: Output format (png, svg, pdf)
    
    Returns:
        Dict mapping visualization type to file path
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    visualizer = KnowledgeGraphVisualizer(kg)
    
    results = {}
    
    print(f"\n{'='*70}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*70}\n")
    
    try:
        # Full graph
        print("Creating full graph visualization...")
        full_path = visualizer.visualize_full_graph(
            output_file=str(output_path / "full_graph"),
            format=format,
            max_nodes=200,
            include_files=False,
            include_variables=False
        )
        results['full_graph'] = full_path
        print(f"  ✓ {full_path}")
    except Exception as e:
        logger.error(f"Failed to create full graph: {e}")
    
    try:
        # Call graph
        print("\nCreating call graph...")
        call_path = visualizer.visualize_call_graph(
            output_file=str(output_path / "call_graph"),
            format=format,
            main_only=True,
            max_depth=4
        )
        results['call_graph'] = call_path
        print(f"  ✓ {call_path}")
    except Exception as e:
        logger.error(f"Failed to create call graph: {e}")
    
    try:
        # File structure
        print("\nCreating file structure...")
        file_path = visualizer.visualize_file_structure(
            output_file=str(output_path / "file_structure"),
            format=format
        )
        results['file_structure'] = file_path
        print(f"  ✓ {file_path}")
    except Exception as e:
        logger.error(f"Failed to create file structure: {e}")
    
    print(f"\n{'='*70}\n")
    
    return results


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║              Knowledge Graph Visualizer                              ║
╚══════════════════════════════════════════════════════════════════════╝

Generate GraphViz visualizations of TAL code knowledge graphs.

USAGE:
  from graph_visualizer import visualize_knowledge_graph
  from knowledge_graph import KnowledgeGraph
  
  kg = KnowledgeGraph()
  # ... populate kg ...
  
  visualize_knowledge_graph(kg, output_dir="./visualizations")

VISUALIZATION TYPES:
  • Full Graph      - Complete knowledge graph structure
  • Call Graph      - Procedure call relationships
  • File Structure  - File organization and definitions
  • Procedure Graph - Context for specific procedure

REQUIREMENTS:
  pip install graphviz
  
  Also requires graphviz binary installed on system:
  - Ubuntu/Debian: sudo apt-get install graphviz
  - macOS: brew install graphviz
  - Windows: Download from https://graphviz.org/download/
    """)
