"""
HTML-Based Knowledge Graph Visualizer
Interactive browser-based visualizations using D3.js - NO system binaries required!

Features:
- Interactive force-directed graphs
- Zoom and pan
- Node filtering and search
- Export to PNG/SVG (browser-based)
- No GraphViz system binary needed
- Works entirely in the browser
"""

from typing import Dict, List, Optional, Set, Any
from pathlib import Path
import json
import logging

from knowledge_graph import (
    Entity, Relationship, EntityType, RelationType, KnowledgeGraph
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# HTML Visualizer (No System Dependencies)
# ============================================================================

class HTMLGraphVisualizer:
    """Generate interactive HTML visualizations using D3.js"""
    
    # Color schemes for different entity types
    ENTITY_COLORS = {
        EntityType.FILE: '#E8F4F8',
        EntityType.PROCEDURE: '#90EE90',
        EntityType.FUNCTION: '#90EE90',
        EntityType.VARIABLE: '#FFD700',
        EntityType.STRUCTURE: '#98FB98',
        EntityType.CONSTANT: '#FFB6C1',
    }
    
    def __init__(self, kg: KnowledgeGraph):
        """Initialize visualizer with knowledge graph"""
        self.kg = kg
    
    def visualize_full_graph(self,
                            output_file: str = "knowledge_graph.html",
                            max_nodes: int = 200,
                            include_files: bool = False,
                            include_variables: bool = False) -> str:
        """
        Create interactive HTML visualization of the full graph
        
        Args:
            output_file: Output HTML filename
            max_nodes: Maximum nodes to include
            include_files: Include file entities
            include_variables: Include variable entities
        
        Returns:
            Path to generated HTML file
        """
        # Get entities
        all_entities = self.kg.query_entities()
        
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
            logger.warning(f"Limiting to {max_nodes} nodes (found {len(filtered_entities)})")
            procedures = [e for e in filtered_entities if e.type == EntityType.PROCEDURE]
            others = [e for e in filtered_entities if e.type != EntityType.PROCEDURE]
            filtered_entities = procedures[:int(max_nodes * 0.8)] + others[:int(max_nodes * 0.2)]
        
        # Get relationships
        all_relationships = self.kg.query_relationships()
        entity_ids = {e.id for e in filtered_entities}
        filtered_relationships = [r for r in all_relationships 
                                 if r.source_id in entity_ids and r.target_id in entity_ids]
        
        # Convert to D3 format
        graph_data = self._convert_to_d3_format(filtered_entities, filtered_relationships)
        
        # Generate HTML
        html_content = self._generate_html_template(
            graph_data,
            title="Knowledge Graph - Full View",
            description=f"Showing {len(filtered_entities)} entities and {len(filtered_relationships)} relationships"
        )
        
        # Write file
        output_path = Path(output_file)
        output_path.write_text(html_content, encoding='utf-8')
        
        logger.info(f"Generated HTML visualization: {output_path}")
        return str(output_path)
    
    def visualize_call_graph(self,
                            output_file: str = "call_graph.html",
                            main_only: bool = True,
                            max_depth: int = 5) -> str:
        """
        Create interactive HTML visualization of call graph
        
        Args:
            output_file: Output HTML filename
            main_only: Start from main procedures only
            max_depth: Maximum call depth
        
        Returns:
            Path to generated HTML file
        """
        # Get procedures
        procedures = self.kg.query_entities(entity_type=EntityType.PROCEDURE)
        
        # Filter to main procedures if requested
        if main_only:
            start_procedures = [p for p in procedures if p.metadata.get('is_main')]
        else:
            start_procedures = procedures
        
        if not start_procedures:
            start_procedures = procedures[:10]
        
        # Build call graph
        visited_procs = set()
        proc_depths = {}
        
        for proc in start_procedures:
            self._traverse_calls(proc, 0, max_depth, visited_procs, proc_depths)
        
        # Get entities and relationships
        entities = [self.kg.get_entity(pid) for pid in visited_procs]
        entities = [e for e in entities if e]
        
        relationships = []
        for proc_id in visited_procs:
            rels = self.kg.query_relationships(
                source_id=proc_id,
                rel_type=RelationType.CALLS
            )
            for rel in rels:
                if rel.target_id in visited_procs:
                    relationships.append(rel)
        
        # Convert to D3 format
        graph_data = self._convert_to_d3_format(entities, relationships)
        
        # Add depth information
        for node in graph_data['nodes']:
            node['depth'] = proc_depths.get(node['id'], 0)
        
        # Generate HTML
        html_content = self._generate_html_template(
            graph_data,
            title="Call Graph",
            description=f"Procedure call relationships (max depth: {max_depth})",
            layout_type="hierarchical"
        )
        
        output_path = Path(output_file)
        output_path.write_text(html_content, encoding='utf-8')
        
        logger.info(f"Generated call graph: {output_path}")
        return str(output_path)
    
    def visualize_file_structure(self,
                                output_file: str = "file_structure.html") -> str:
        """
        Create interactive HTML visualization of file structure
        
        Args:
            output_file: Output HTML filename
        
        Returns:
            Path to generated HTML file
        """
        # Get files and procedures
        files = self.kg.query_entities(entity_type=EntityType.FILE)
        procedures = self.kg.query_entities(entity_type=EntityType.PROCEDURE)
        
        # Group procedures by file
        file_procedures = {}
        for proc in procedures:
            if proc.file_path:
                if proc.file_path not in file_procedures:
                    file_procedures[proc.file_path] = []
                file_procedures[proc.file_path].append(proc)
        
        # Create entities list
        entities = list(files)
        for proc_list in file_procedures.values():
            entities.extend(proc_list)
        
        # Get call relationships
        relationships = self.kg.query_relationships(rel_type=RelationType.CALLS)
        
        # Convert to D3 format
        graph_data = self._convert_to_d3_format(entities, relationships)
        
        # Add file grouping
        for node in graph_data['nodes']:
            entity = next((e for e in entities if e.id == node['id']), None)
            if entity and entity.file_path:
                node['file'] = Path(entity.file_path).name
        
        # Generate HTML
        html_content = self._generate_html_template(
            graph_data,
            title="File Structure",
            description=f"Code organization across {len(files)} files",
            layout_type="grouped"
        )
        
        output_path = Path(output_file)
        output_path.write_text(html_content, encoding='utf-8')
        
        logger.info(f"Generated file structure: {output_path}")
        return str(output_path)
    
    def visualize_procedure_subgraph(self,
                                    procedure_name: str,
                                    output_file: str = "procedure_graph.html",
                                    depth: int = 2) -> str:
        """
        Create interactive HTML visualization of procedure context
        
        Args:
            procedure_name: Name of the procedure
            output_file: Output HTML filename
            depth: Relationship depth
        
        Returns:
            Path to generated HTML file
        """
        # Find procedure
        procedures = self.kg.query_entities(entity_type=EntityType.PROCEDURE)
        target_proc = None
        for proc in procedures:
            if proc.name == procedure_name:
                target_proc = proc
                break
        
        if not target_proc:
            raise ValueError(f"Procedure '{procedure_name}' not found")
        
        # Get related entities
        visited_entities = {target_proc.id: target_proc}
        visited_relationships = set()
        
        self._traverse_procedure_context(
            target_proc,
            depth,
            visited_entities,
            visited_relationships,
            include_variables=True
        )
        
        # Convert to D3 format
        entities = list(visited_entities.values())
        relationships = list(visited_relationships)
        graph_data = self._convert_to_d3_format(entities, relationships)
        
        # Mark the target procedure
        for node in graph_data['nodes']:
            if node['id'] == target_proc.id:
                node['is_target'] = True
        
        # Generate HTML
        html_content = self._generate_html_template(
            graph_data,
            title=f"Procedure: {procedure_name}",
            description=f"Context and relationships for {procedure_name}",
            layout_type="radial",
            center_node=target_proc.id
        )
        
        output_path = Path(output_file)
        output_path.write_text(html_content, encoding='utf-8')
        
        logger.info(f"Generated procedure graph: {output_path}")
        return str(output_path)
    
    def _convert_to_d3_format(self, entities: List[Entity], 
                              relationships: List[Relationship]) -> Dict[str, Any]:
        """Convert entities and relationships to D3.js format"""
        nodes = []
        for entity in entities:
            node = {
                'id': entity.id,
                'name': entity.name,
                'type': entity.type.value,
                'qualified_name': entity.qualified_name,
                'color': self.ENTITY_COLORS.get(entity.type, '#CCCCCC'),
                'size': self._get_node_size(entity),
                'metadata': {}
            }
            
            # Add relevant metadata
            if entity.type == EntityType.PROCEDURE:
                node['metadata']['is_main'] = entity.metadata.get('is_main', False)
                node['metadata']['is_external'] = entity.metadata.get('is_external', False)
                node['metadata']['param_count'] = entity.metadata.get('parameter_count', 0)
                node['metadata']['return_type'] = entity.metadata.get('return_type', 'void')
            
            nodes.append(node)
        
        links = []
        for rel in relationships:
            link = {
                'source': rel.source_id,
                'target': rel.target_id,
                'type': rel.type.value,
                'weight': rel.weight
            }
            links.append(link)
        
        return {'nodes': nodes, 'links': links}
    
    def _get_node_size(self, entity: Entity) -> int:
        """Calculate node size based on entity type and metadata"""
        base_size = 10
        
        if entity.type == EntityType.PROCEDURE:
            if entity.metadata.get('is_main'):
                return base_size * 2
            stmt_count = entity.metadata.get('statement_count', 0)
            return base_size + min(stmt_count / 10, 20)
        elif entity.type == EntityType.FILE:
            return base_size * 1.5
        
        return base_size
    
    def _generate_html_template(self, graph_data: Dict[str, Any],
                                title: str,
                                description: str,
                                layout_type: str = "force",
                                center_node: Optional[str] = None) -> str:
        """Generate complete HTML file with embedded D3.js visualization"""
        
        graph_json = json.dumps(graph_data, indent=2)
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f5f7fa;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .header h1 {{
            margin: 0 0 10px 0;
            font-size: 2em;
            font-weight: 300;
        }}
        
        .header p {{
            margin: 0;
            opacity: 0.9;
        }}
        
        .controls {{
            background: white;
            padding: 15px;
            border-bottom: 1px solid #e0e0e0;
            display: flex;
            gap: 15px;
            align-items: center;
            flex-wrap: wrap;
        }}
        
        .control-group {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .control-group label {{
            font-weight: 500;
            color: #333;
        }}
        
        input[type="text"], input[type="range"], select {{
            padding: 6px 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
        }}
        
        button {{
            padding: 8px 16px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            transition: background 0.3s;
        }}
        
        button:hover {{
            background: #5568d3;
        }}
        
        #graph {{
            width: 100vw;
            height: calc(100vh - 180px);
            background: white;
        }}
        
        .node {{
            stroke: #fff;
            stroke-width: 2px;
            cursor: pointer;
        }}
        
        .node:hover {{
            stroke: #667eea;
            stroke-width: 3px;
        }}
        
        .node.highlighted {{
            stroke: #ff6b6b;
            stroke-width: 4px;
        }}
        
        .link {{
            stroke: #999;
            stroke-opacity: 0.6;
        }}
        
        .link.call {{
            stroke: #4285f4;
        }}
        
        .link.contains {{
            stroke: #999;
            stroke-dasharray: 5,5;
        }}
        
        .node-label {{
            font-size: 11px;
            pointer-events: none;
            text-anchor: middle;
            fill: #333;
        }}
        
        .tooltip {{
            position: absolute;
            padding: 10px;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            border-radius: 4px;
            font-size: 12px;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.3s;
            max-width: 300px;
        }}
        
        .legend {{
            position: absolute;
            top: 200px;
            right: 20px;
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            font-size: 12px;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 5px 0;
        }}
        
        .legend-color {{
            width: 20px;
            height: 20px;
            border-radius: 50%;
            margin-right: 8px;
            border: 2px solid #fff;
        }}
        
        .stats {{
            position: absolute;
            top: 80px;
            left: 20px;
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            font-size: 12px;
        }}
        
        .stat-item {{
            margin: 5px 0;
        }}
        
        .stat-label {{
            font-weight: 600;
            color: #666;
        }}
        
        .stat-value {{
            color: #333;
            font-weight: 500;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{title}</h1>
        <p>{description}</p>
    </div>
    
    <div class="controls">
        <div class="control-group">
            <label>Search:</label>
            <input type="text" id="search" placeholder="Search nodes...">
        </div>
        
        <div class="control-group">
            <label>Filter Type:</label>
            <select id="filter-type">
                <option value="all">All</option>
                <option value="procedure">Procedures</option>
                <option value="variable">Variables</option>
                <option value="structure">Structures</option>
                <option value="file">Files</option>
            </select>
        </div>
        
        <div class="control-group">
            <label>Link Distance:</label>
            <input type="range" id="link-distance" min="30" max="200" value="100">
        </div>
        
        <div class="control-group">
            <label>Charge Strength:</label>
            <input type="range" id="charge-strength" min="-500" max="-50" value="-200">
        </div>
        
        <button onclick="resetZoom()">Reset Zoom</button>
        <button onclick="exportSVG()">Export SVG</button>
    </div>
    
    <div class="stats" id="stats">
        <div class="stat-item">
            <span class="stat-label">Nodes:</span>
            <span class="stat-value" id="node-count">0</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">Links:</span>
            <span class="stat-value" id="link-count">0</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">Visible:</span>
            <span class="stat-value" id="visible-count">0</span>
        </div>
    </div>
    
    <div class="legend">
        <strong>Entity Types</strong>
        <div class="legend-item">
            <div class="legend-color" style="background: #90EE90;"></div>
            <span>Procedure</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #FFD700;"></div>
            <span>Variable</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #98FB98;"></div>
            <span>Structure</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: #E8F4F8;"></div>
            <span>File</span>
        </div>
    </div>
    
    <svg id="graph"></svg>
    <div class="tooltip" id="tooltip"></div>
    
    <script>
        // Graph data
        const graphData = {graph_json};
        
        // Dimensions
        const width = window.innerWidth;
        const height = window.innerHeight - 180;
        
        // Create SVG
        const svg = d3.select("#graph")
            .attr("viewBox", [0, 0, width, height]);
        
        // Add zoom
        const g = svg.append("g");
        const zoom = d3.zoom()
            .scaleExtent([0.1, 10])
            .on("zoom", (event) => {{
                g.attr("transform", event.transform);
            }});
        
        svg.call(zoom);
        
        // Create simulation
        let linkDistance = 100;
        let chargeStrength = -200;
        
        const simulation = d3.forceSimulation(graphData.nodes)
            .force("link", d3.forceLink(graphData.links).id(d => d.id).distance(linkDistance))
            .force("charge", d3.forceManyBody().strength(chargeStrength))
            .force("center", d3.forceCenter(width / 2, height / 2))
            .force("collision", d3.forceCollide().radius(d => d.size + 5));
        
        // Create links
        const link = g.append("g")
            .selectAll("line")
            .data(graphData.links)
            .join("line")
            .attr("class", d => `link ${{d.type}}`)
            .attr("stroke-width", d => Math.sqrt(d.weight || 1));
        
        // Create nodes
        const node = g.append("g")
            .selectAll("circle")
            .data(graphData.nodes)
            .join("circle")
            .attr("class", "node")
            .attr("r", d => d.size)
            .attr("fill", d => d.color)
            .call(drag(simulation))
            .on("mouseover", showTooltip)
            .on("mouseout", hideTooltip)
            .on("click", highlightConnections);
        
        // Create labels
        const label = g.append("g")
            .selectAll("text")
            .data(graphData.nodes)
            .join("text")
            .attr("class", "node-label")
            .text(d => d.name)
            .attr("dy", d => d.size + 12);
        
        // Update stats
        updateStats();
        
        // Simulation tick
        simulation.on("tick", () => {{
            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);
            
            node
                .attr("cx", d => d.x)
                .attr("cy", d => d.y);
            
            label
                .attr("x", d => d.x)
                .attr("y", d => d.y);
        }});
        
        // Drag behavior
        function drag(simulation) {{
            function dragstarted(event) {{
                if (!event.active) simulation.alphaTarget(0.3).restart();
                event.subject.fx = event.subject.x;
                event.subject.fy = event.subject.y;
            }}
            
            function dragged(event) {{
                event.subject.fx = event.x;
                event.subject.fy = event.y;
            }}
            
            function dragended(event) {{
                if (!event.active) simulation.alphaTarget(0);
                event.subject.fx = null;
                event.subject.fy = null;
            }}
            
            return d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended);
        }}
        
        // Tooltip
        function showTooltip(event, d) {{
            const tooltip = d3.select("#tooltip");
            let html = `<strong>${{d.name}}</strong><br>`;
            html += `Type: ${{d.type}}<br>`;
            
            if (d.metadata.is_main) {{
                html += `<strong style="color: #ffd700;">★ MAIN PROCEDURE</strong><br>`;
            }}
            if (d.metadata.is_external) {{
                html += `<em>External Reference</em><br>`;
            }}
            if (d.metadata.param_count) {{
                html += `Parameters: ${{d.metadata.param_count}}<br>`;
            }}
            if (d.metadata.return_type) {{
                html += `Returns: ${{d.metadata.return_type}}<br>`;
            }}
            
            tooltip.html(html)
                .style("left", (event.pageX + 10) + "px")
                .style("top", (event.pageY - 10) + "px")
                .style("opacity", 1);
        }}
        
        function hideTooltip() {{
            d3.select("#tooltip").style("opacity", 0);
        }}
        
        // Highlight connections
        function highlightConnections(event, d) {{
            const connected = new Set();
            
            graphData.links.forEach(link => {{
                if (link.source.id === d.id) connected.add(link.target.id);
                if (link.target.id === d.id) connected.add(link.source.id);
            }});
            
            node.classed("highlighted", n => n.id === d.id || connected.has(n.id));
            link.attr("stroke-opacity", l => 
                (l.source.id === d.id || l.target.id === d.id) ? 1 : 0.1
            );
        }}
        
        // Search functionality
        document.getElementById("search").addEventListener("input", function(e) {{
            const searchTerm = e.target.value.toLowerCase();
            
            node.classed("highlighted", d => 
                d.name.toLowerCase().includes(searchTerm) ||
                d.qualified_name.toLowerCase().includes(searchTerm)
            );
            
            if (searchTerm) {{
                node.attr("opacity", d =>
                    d.name.toLowerCase().includes(searchTerm) ||
                    d.qualified_name.toLowerCase().includes(searchTerm) ? 1 : 0.2
                );
                label.attr("opacity", d =>
                    d.name.toLowerCase().includes(searchTerm) ||
                    d.qualified_name.toLowerCase().includes(searchTerm) ? 1 : 0.2
                );
            }} else {{
                node.attr("opacity", 1);
                label.attr("opacity", 1);
                link.attr("stroke-opacity", 0.6);
            }}
        }});
        
        // Filter by type
        document.getElementById("filter-type").addEventListener("change", function(e) {{
            const filterType = e.target.value;
            
            node.attr("opacity", d => 
                filterType === "all" || d.type === filterType ? 1 : 0.2
            );
            label.attr("opacity", d =>
                filterType === "all" || d.type === filterType ? 1 : 0.2
            );
            
            updateStats();
        }});
        
        // Link distance control
        document.getElementById("link-distance").addEventListener("input", function(e) {{
            linkDistance = +e.target.value;
            simulation.force("link").distance(linkDistance);
            simulation.alpha(0.3).restart();
        }});
        
        // Charge strength control
        document.getElementById("charge-strength").addEventListener("input", function(e) {{
            chargeStrength = +e.target.value;
            simulation.force("charge").strength(chargeStrength);
            simulation.alpha(0.3).restart();
        }});
        
        // Reset zoom
        function resetZoom() {{
            svg.transition().duration(750).call(
                zoom.transform,
                d3.zoomIdentity
            );
            
            node.classed("highlighted", false);
            link.attr("stroke-opacity", 0.6);
        }}
        
        // Export SVG
        function exportSVG() {{
            const svgData = document.getElementById("graph").outerHTML;
            const blob = new Blob([svgData], {{type: "image/svg+xml"}});
            const url = URL.createObjectURL(blob);
            const link = document.createElement("a");
            link.href = url;
            link.download = "{title.replace(' ', '_').lower()}.svg";
            link.click();
        }}
        
        // Update statistics
        function updateStats() {{
            document.getElementById("node-count").textContent = graphData.nodes.length;
            document.getElementById("link-count").textContent = graphData.links.length;
            
            const visibleNodes = graphData.nodes.filter(n => 
                parseFloat(d3.select(`circle[cx="${{n.x}}"]`).attr("opacity") || 1) > 0.5
            ).length;
            document.getElementById("visible-count").textContent = visibleNodes;
        }}
    </script>
</body>
</html>"""
        
        return html
    
    def _traverse_calls(self, entity: Entity, current_depth: int, max_depth: int,
                       visited_procs: Set[str], proc_depths: Dict[str, int]):
        """Recursively traverse call graph"""
        if current_depth > max_depth or entity.id in visited_procs:
            return
        
        visited_procs.add(entity.id)
        
        if entity.id not in proc_depths or current_depth < proc_depths[entity.id]:
            proc_depths[entity.id] = current_depth
        
        rels = self.kg.query_relationships(
            source_id=entity.id,
            rel_type=RelationType.CALLS
        )
        
        for rel in rels:
            callee = self.kg.get_entity(rel.target_id)
            if callee and callee.type == EntityType.PROCEDURE:
                self._traverse_calls(callee, current_depth + 1, max_depth,
                                   visited_procs, proc_depths)
    
    def _traverse_procedure_context(self, entity: Entity, depth: int,
                                   visited_entities: Dict[str, Entity],
                                   visited_relationships: Set[Relationship],
                                   include_variables: bool):
        """Recursively traverse procedure context"""
        if depth <= 0:
            return
        
        outgoing = self.kg.query_relationships(source_id=entity.id)
        incoming = self.kg.query_relationships(target_id=entity.id)
        
        for rel in outgoing + incoming:
            visited_relationships.add(rel)
            
            other_id = rel.target_id if rel.source_id == entity.id else rel.source_id
            if other_id not in visited_entities:
                other = self.kg.get_entity(other_id)
                if other:
                    if not include_variables and other.type == EntityType.VARIABLE:
                        continue
                    
                    visited_entities[other_id] = other
                    
                    if other.type == EntityType.PROCEDURE and rel.type == RelationType.CALLS:
                        self._traverse_procedure_context(
                            other, depth - 1, visited_entities,
                            visited_relationships, include_variables
                        )


# ============================================================================
# Convenience Functions
# ============================================================================

def visualize_knowledge_graph_html(kg: KnowledgeGraph,
                                   output_dir: str = "./visualizations",
                                   format: str = "html") -> Dict[str, str]:
    """
    Generate all standard HTML visualizations (no GraphViz binary needed!)
    
    Args:
        kg: Knowledge graph to visualize
        output_dir: Output directory for visualizations
        format: Output format (html only)
    
    Returns:
        Dict mapping visualization type to file path
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    visualizer = HTMLGraphVisualizer(kg)
    
    results = {}
    
    print(f"\n{'='*70}")
    print("GENERATING HTML VISUALIZATIONS (No System Dependencies!)")
    print(f"{'='*70}\n")
    
    try:
        # Full graph
        print("Creating full graph visualization...")
        full_path = visualizer.visualize_full_graph(
            output_file=str(output_path / "full_graph.html"),
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
            output_file=str(output_path / "call_graph.html"),
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
            output_file=str(output_path / "file_structure.html")
        )
        results['file_structure'] = file_path
        print(f"  ✓ {file_path}")
    except Exception as e:
        logger.error(f"Failed to create file structure: {e}")
    
    print(f"\n{'='*70}")
    print("All visualizations are interactive HTML files!")
    print("Open them in any web browser - no additional software needed.")
    print(f"{'='*70}\n")
    
    return results


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║          HTML Knowledge Graph Visualizer (No GraphViz!)              ║
╚══════════════════════════════════════════════════════════════════════╝

Generate interactive browser-based visualizations using D3.js.

✓ NO system binaries required
✓ NO GraphViz installation needed
✓ Works entirely in the browser
✓ Interactive: zoom, pan, search
✓ Export SVG directly from browser

USAGE:
  from html_graph_visualizer import visualize_knowledge_graph_html
  from knowledge_graph import KnowledgeGraph
  
  kg = KnowledgeGraph()
  # ... populate kg ...
  
  visualize_knowledge_graph_html(kg, output_dir="./visualizations")

FEATURES:
  • Interactive force-directed graphs
  • Search and filter nodes
  • Highlight connections
  • Zoom and pan
  • Export to SVG
  • Responsive design
    """)
