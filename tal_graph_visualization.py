"""
HTML-Based Knowledge Graph Visualizer - Updated to read from JSON files
Interactive browser-based visualizations using D3.js - NO system binaries required!

Features:
- Read graph data from JSON files exported by parsers.py
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
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Graph Data Loader (NEW)
# ============================================================================

class GraphDataLoader:
    """Load graph data from JSON files"""
    
    @staticmethod
    def load_from_file(json_file: str) -> Dict[str, Any]:
        """
        Load graph data from JSON file
        
        Args:
            json_file: Path to graph_data.json file
        
        Returns:
            Dict with nodes and edges
        """
        file_path = Path(json_file)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Graph data file not found: {json_file}")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        logger.info(f"Loaded graph data from {json_file}")
        logger.info(f"  Nodes: {len(data.get('nodes', []))}")
        logger.info(f"  Edges: {len(data.get('edges', []))}")
        
        return data
    
    @staticmethod
    def convert_to_d3_format(graph_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert loaded graph data to D3.js format
        
        Args:
            graph_data: Data from graph_data.json
        
        Returns:
            Dict with nodes and links in D3 format
        """
        # Color schemes for different entity types
        ENTITY_COLORS = {
            'file': '#E8F4F8',
            'procedure': '#90EE90',
            'function': '#90EE90',
            'variable': '#FFD700',
            'structure': '#98FB98',
            'constant': '#FFB6C1',
        }
        
        nodes = []
        for node_data in graph_data.get('nodes', []):
            node = {
                'id': node_data['id'],
                'name': node_data['name'],
                'type': node_data['type'],
                'qualified_name': node_data.get('qualified_name', ''),
                'color': ENTITY_COLORS.get(node_data['type'], '#CCCCCC'),
                'size': GraphDataLoader._get_node_size(node_data),
                'metadata': node_data.get('metadata', {}),
                'file_path': node_data.get('file_path', ''),
                'start_line': node_data.get('start_line', 0)
            }
            nodes.append(node)
        
        links = []
        for edge_data in graph_data.get('edges', []):
            link = {
                'source': edge_data['source'],
                'target': edge_data['target'],
                'type': edge_data['type'],
                'weight': edge_data.get('weight', 1.0),
                'metadata': edge_data.get('metadata', {})
            }
            links.append(link)
        
        return {'nodes': nodes, 'links': links}
    
    @staticmethod
    def _get_node_size(node_data: Dict[str, Any]) -> int:
        """Calculate node size based on entity type and metadata"""
        base_size = 10
        
        if node_data['type'] == 'procedure':
            metadata = node_data.get('metadata', {})
            if metadata.get('is_main'):
                return base_size * 2
            stmt_count = metadata.get('statement_count', 0)
            return base_size + min(stmt_count / 10, 20)
        elif node_data['type'] == 'file':
            return base_size * 1.5
        
        return base_size


# ============================================================================
# Standalone HTML Generator (NEW)
# ============================================================================

def generate_standalone_html(json_file: str, 
                            output_file: str = "graph_visualization.html",
                            title: str = "Knowledge Graph Visualization") -> str:
    """
    Generate a standalone HTML file from graph JSON data
    
    Args:
        json_file: Path to graph_data.json
        output_file: Output HTML file path
        title: Page title
    
    Returns:
        Path to generated HTML file
    """
    # Load graph data
    loader = GraphDataLoader()
    graph_data = loader.load_from_file(json_file)
    d3_data = loader.convert_to_d3_format(graph_data)
    
    # Get statistics
    stats = graph_data.get('statistics', {})
    metadata = graph_data.get('metadata', {})
    
    description = f"Showing {len(d3_data['nodes'])} entities and {len(d3_data['links'])} relationships"
    
    # Generate HTML
    html_content = _generate_html_template(
        d3_data,
        title=title,
        description=description,
        statistics=stats
    )
    
    # Write file
    output_path = Path(output_file)
    output_path.write_text(html_content, encoding='utf-8')
    
    logger.info(f"Generated HTML visualization: {output_path}")
    print(f"\n✓ HTML visualization created: {output_path}")
    print(f"  Open this file in your web browser to view the interactive graph")
    
    return str(output_path)


def _generate_html_template(graph_data: Dict[str, Any],
                            title: str,
                            description: str,
                            statistics: Dict[str, Any] = None) -> str:
    """Generate complete HTML file with embedded D3.js visualization"""
    
    graph_json = json.dumps(graph_data, indent=2)
    stats_json = json.dumps(statistics or {}, indent=2)
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f5f7fa;
            overflow: hidden;
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
            font-size: 14px;
        }}
        
        input[type="text"], input[type="range"], select {{
            padding: 6px 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
        }}
        
        input[type="text"] {{
            min-width: 200px;
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
            transition: all 0.3s;
        }}
        
        .node:hover {{
            stroke: #667eea;
            stroke-width: 3px;
        }}
        
        .node.highlighted {{
            stroke: #ff6b6b;
            stroke-width: 4px;
        }}
        
        .node.main-procedure {{
            stroke: #ffd700;
            stroke-width: 3px;
        }}
        
        .link {{
            stroke: #999;
            stroke-opacity: 0.6;
        }}
        
        .link.calls {{
            stroke: #4285f4;
        }}
        
        .link.contains {{
            stroke: #999;
            stroke-dasharray: 5,5;
        }}
        
        .link.highlighted {{
            stroke: #ff6b6b;
            stroke-width: 2px;
            stroke-opacity: 1;
        }}
        
        .node-label {{
            font-size: 11px;
            pointer-events: none;
            text-anchor: middle;
            fill: #333;
            font-weight: 500;
        }}
        
        .tooltip {{
            position: absolute;
            padding: 12px;
            background: rgba(0, 0, 0, 0.85);
            color: white;
            border-radius: 6px;
            font-size: 12px;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.3s;
            max-width: 350px;
            z-index: 1000;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }}
        
        .tooltip strong {{
            color: #ffd700;
            font-size: 14px;
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
            box-shadow: 0 1px 3px rgba(0,0,0,0.2);
        }}
        
        .stats {{
            position: absolute;
            top: 200px;
            left: 20px;
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            font-size: 12px;
            max-width: 250px;
        }}
        
        .stat-item {{
            margin: 8px 0;
            padding: 5px 0;
            border-bottom: 1px solid #eee;
        }}
        
        .stat-item:last-child {{
            border-bottom: none;
        }}
        
        .stat-label {{
            font-weight: 600;
            color: #666;
            display: block;
        }}
        
        .stat-value {{
            color: #333;
            font-weight: 500;
            font-size: 16px;
            margin-top: 3px;
            display: block;
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
            <label>🔍 Search:</label>
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
            <input type="range" id="link-distance" min="30" max="200" value="100" step="10">
            <span id="link-distance-value">100</span>
        </div>
        
        <div class="control-group">
            <label>Charge:</label>
            <input type="range" id="charge-strength" min="-500" max="-50" value="-200" step="10">
            <span id="charge-value">-200</span>
        </div>
        
        <button onclick="resetZoom()">🔄 Reset</button>
        <button onclick="centerGraph()">📍 Center</button>
        <button onclick="exportSVG()">💾 Export SVG</button>
    </div>
    
    <div class="stats" id="stats">
        <h3 style="margin-bottom: 10px; color: #667eea;">Graph Statistics</h3>
        <div class="stat-item">
            <span class="stat-label">Total Nodes</span>
            <span class="stat-value" id="node-count">0</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">Total Links</span>
            <span class="stat-value" id="link-count">0</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">Visible Nodes</span>
            <span class="stat-value" id="visible-count">0</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">Selected</span>
            <span class="stat-value" id="selected-node">None</span>
        </div>
    </div>
    
    <div class="legend">
        <h4 style="margin-bottom: 10px; color: #667eea;">Entity Types</h4>
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
        <hr style="margin: 10px 0;">
        <div class="legend-item">
            <div class="legend-color" style="background: #90EE90; border-color: #ffd700; border-width: 3px;"></div>
            <span>★ Main Proc</span>
        </div>
    </div>
    
    <svg id="graph"></svg>
    <div class="tooltip" id="tooltip"></div>
    
    <script>
        // Graph data embedded from JSON
        const graphData = {graph_json};
        const statistics = {stats_json};
        
        console.log('Loaded graph data:', graphData);
        console.log('Statistics:', statistics);
        
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
            .attr("class", d => {{
                let classes = "node";
                if (d.metadata && d.metadata.is_main) classes += " main-procedure";
                return classes;
            }})
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
            html += `<div style="margin-top: 5px;">`;
            html += `Type: <strong>${{d.type}}</strong><br>`;
            
            if (d.file_path) {{
                html += `File: ${{d.file_path.split('/').pop()}}<br>`;
            }}
            
            if (d.start_line) {{
                html += `Line: ${{d.start_line}}<br>`;
            }}
            
            if (d.metadata) {{
                if (d.metadata.is_main) {{
                    html += `<strong style="color: #ffd700;">★ MAIN PROCEDURE</strong><br>`;
                }}
                if (d.metadata.is_external) {{
                    html += `<em style="color: #ff9999;">External Reference</em><br>`;
                }}
                if (d.metadata.parameter_count !== undefined) {{
                    html += `Parameters: ${{d.metadata.parameter_count}}<br>`;
                }}
                if (d.metadata.return_type) {{
                    html += `Returns: ${{d.metadata.return_type}}<br>`;
                }}
                if (d.metadata.statement_count) {{
                    html += `Statements: ${{d.metadata.statement_count}}<br>`;
                }}
            }}
            
            html += `</div>`;
            
            tooltip.html(html)
                .style("left", (event.pageX + 10) + "px")
                .style("top", (event.pageY - 10) + "px")
                .style("opacity", 1);
        }}
        
        function hideTooltip() {{
            d3.select("#tooltip").style("opacity", 0);
        }}
        
        // Highlight connections
        let highlightedNode = null;
        
        function highlightConnections(event, d) {{
            if (highlightedNode === d.id) {{
                // Deselect
                highlightedNode = null;
                node.classed("highlighted", false);
                link.classed("highlighted", false)
                    .attr("stroke-opacity", 0.6);
                document.getElementById("selected-node").textContent = "None";
                return;
            }}
            
            highlightedNode = d.id;
            const connected = new Set();
            const connectedLinks = new Set();
            
            graphData.links.forEach((link, idx) => {{
                if (link.source.id === d.id) {{
                    connected.add(link.target.id);
                    connectedLinks.add(idx);
                }}
                if (link.target.id === d.id) {{
                    connected.add(link.source.id);
                    connectedLinks.add(idx);
                }}
            }});
            
            node.classed("highlighted", n => n.id === d.id || connected.has(n.id));
            
            link.classed("highlighted", (l, i) => connectedLinks.has(i))
                .attr("stroke-opacity", (l, i) => connectedLinks.has(i) ? 1 : 0.1);
            
            document.getElementById("selected-node").textContent = d.name;
        }}
        
        // Search functionality
        document.getElementById("search").addEventListener("input", function(e) {{
            const searchTerm = e.target.value.toLowerCase();
            
            if (searchTerm) {{
                node.classed("highlighted", d => 
                    d.name.toLowerCase().includes(searchTerm) ||
                    (d.qualified_name && d.qualified_name.toLowerCase().includes(searchTerm))
                );
                
                node.attr("opacity", d =>
                    d.name.toLowerCase().includes(searchTerm) ||
                    (d.qualified_name && d.qualified_name.toLowerCase().includes(searchTerm)) ? 1 : 0.2
                );
                label.attr("opacity", d =>
                    d.name.toLowerCase().includes(searchTerm) ||
                    (d.qualified_name && d.qualified_name.toLowerCase().includes(searchTerm)) ? 1 : 0.2
                );
            }} else {{
                node.classed("highlighted", false)
                    .attr("opacity", 1);
                label.attr("opacity", 1);
                link.attr("stroke-opacity", 0.6);
            }}
            
            updateStats();
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
            document.getElementById("link-distance-value").textContent = linkDistance;
            simulation.force("link").distance(linkDistance);
            simulation.alpha(0.3).restart();
        }});
        
        // Charge strength control
        document.getElementById("charge-strength").addEventListener("input", function(e) {{
            chargeStrength = +e.target.value;
            document.getElementById("charge-value").textContent = chargeStrength;
            simulation.force("charge").strength(chargeStrength);
            simulation.alpha(0.3).restart();
        }});
        
        // Reset zoom
        function resetZoom() {{
            svg.transition().duration(750).call(
                zoom.transform,
                d3.zoomIdentity
            );
            
            node.classed("highlighted", false)
                .attr("opacity", 1);
            link.classed("highlighted", false)
                .attr("stroke-opacity", 0.6);
            label.attr("opacity", 1);
            highlightedNode = null;
            document.getElementById("selected-node").textContent = "None";
        }}
        
        // Center graph
        function centerGraph() {{
            const bounds = g.node().getBBox();
            const fullWidth = bounds.width;
            const fullHeight = bounds.height;
            const midX = bounds.x + fullWidth / 2;
            const midY = bounds.y + fullHeight / 2;
            
            const scale = 0.8 / Math.max(fullWidth / width, fullHeight / height);
            const translate = [width / 2 - scale * midX, height / 2 - scale * midY];
            
            svg.transition().duration(750).call(
                zoom.transform,
                d3.zoomIdentity.translate(translate[0], translate[1]).scale(scale)
            );
        }}
        
        // Export SVG
        function exportSVG() {{
            const svgData = document.getElementById("graph").outerHTML;
            const blob = new Blob([svgData], {{type: "image/svg+xml"}});
            const url = URL.createObjectURL(blob);
            const link = document.createElement("a");
            link.href = url;
            link.download = "knowledge_graph.svg";
            link.click();
            URL.revokeObjectURL(url);
        }}
        
        // Update statistics
        function updateStats() {{
            document.getElementById("node-count").textContent = graphData.nodes.length;
            document.getElementById("link-count").textContent = graphData.links.length;
            
            const visibleCount = graphData.nodes.filter(n => {{
                const nodeElement = node.filter(d => d.id === n.id);
                const opacity = nodeElement.attr("opacity");
                return !opacity || parseFloat(opacity) > 0.5;
            }}).length;
            
            document.getElementById("visible-count").textContent = visibleCount;
        }}
        
        // Initialize - center after a delay to let simulation settle
        setTimeout(centerGraph, 1000);
    </script>
</body>
</html>"""
    
    return html


# ============================================================================
# Command Line Interface
# ============================================================================

def main():
    """Command line interface for generating visualizations from JSON"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate interactive HTML visualizations from graph JSON data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate visualization from graph data
  python graph_visualizer.py graph_data.json
  
  # Custom output file
  python graph_visualizer.py graph_data.json -o my_graph.html
  
  # Custom title
  python graph_visualizer.py graph_data.json -t "Payment System Architecture"
        """
    )
    
    parser.add_argument('json_file', help='Path to graph_data.json file')
    parser.add_argument('-o', '--output', default='graph_visualization.html',
                       help='Output HTML file (default: graph_visualization.html)')
    parser.add_argument('-t', '--title', default='Knowledge Graph Visualization',
                       help='Visualization title')
    
    args = parser.parse_args()
    
    try:
        # Validate input file
        if not Path(args.json_file).exists():
            print(f"Error: File '{args.json_file}' not found")
            sys.exit(1)
        
        # Generate visualization
        print(f"\n{'='*70}")
        print("GENERATING INTERACTIVE HTML VISUALIZATION")
        print(f"{'='*70}\n")
        print(f"Input:  {args.json_file}")
        print(f"Output: {args.output}")
        print(f"Title:  {args.title}\n")
        
        output_path = generate_standalone_html(
            json_file=args.json_file,
            output_file=args.output,
            title=args.title
        )
        
        print(f"\n{'='*70}")
        print("SUCCESS!")
        print(f"{'='*70}")
        print(f"\nOpen the following file in your web browser:")
        print(f"  {Path(output_path).absolute()}")
        print(f"\nFeatures:")
        print(f"  • Interactive force-directed graph")
        print(f"  • Search and filter nodes")
        print(f"  • Click nodes to highlight connections")
        print(f"  • Zoom and pan")
        print(f"  • Export to SVG")
        print(f"\n{'='*70}\n")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("""
╔══════════════════════════════════════════════════════════════════════╗
║     HTML Knowledge Graph Visualizer - Standalone File Version       ║
╚══════════════════════════════════════════════════════════════════════╝

Generate interactive browser-based visualizations from JSON graph data.

✓ NO system binaries required
✓ NO GraphViz installation needed  
✓ Works entirely in the browser
✓ Interactive: zoom, pan, search
✓ Export SVG directly from browser

USAGE:
  python graph_visualizer.py <graph_data.json> [options]

WORKFLOW:
  1. Parse TAL files:
     python parsers.py ./tal_source --export ./output
     
  2. Visualize graph:
     python graph_visualizer.py ./output/graph_data.json
     
  3. Open graph_visualization.html in your browser

Run with --help for more options.
        """)
    else:
        main()
