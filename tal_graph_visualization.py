#!/usr/bin/env python3
"""
Minimal HTML Graph Visualizer - Single File Version
Copy this entire file and save as: simple_html_viz.py

NO GraphViz needed! Creates interactive HTML visualizations.

Usage:
    from simple_html_viz import create_html_visualization
    from knowledge_graph import KnowledgeGraph
    
    kg = KnowledgeGraph()
    # ... populate your kg ...
    
    create_html_visualization(kg, "my_graph.html")
    # Then open my_graph.html in your browser!
"""

from knowledge_graph import KnowledgeGraph, EntityType, RelationType
from pathlib import Path
import json


def create_html_visualization(kg: KnowledgeGraph, output_file: str = "graph.html", max_nodes: int = 200):
    """
    Create interactive HTML visualization - NO GraphViz needed!
    
    Args:
        kg: Your populated KnowledgeGraph
        output_file: Output filename (e.g., "my_graph.html")
        max_nodes: Maximum nodes to display
    """
    # Get data
    entities = kg.query_entities()[:max_nodes]
    relationships = kg.query_relationships()
    
    # Filter relationships to included nodes
    entity_ids = {e.id for e in entities}
    relationships = [r for r in relationships if r.source_id in entity_ids and r.target_id in entity_ids]
    
    # Convert to D3 format
    nodes = []
    for e in entities:
        nodes.append({
            'id': e.id,
            'name': e.name,
            'type': e.type.value,
            'color': '#90EE90' if e.type == EntityType.PROCEDURE else '#FFD700'
        })
    
    links = []
    for r in relationships:
        links.append({
            'source': r.source_id,
            'target': r.target_id,
            'type': r.type.value
        })
    
    graph_data = {'nodes': nodes, 'links': links}
    
    # Generate HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Knowledge Graph</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{ margin: 0; font-family: Arial; background: #f5f7fa; }}
        #graph {{ width: 100vw; height: 100vh; }}
        .node {{ stroke: #fff; stroke-width: 2px; cursor: pointer; }}
        .node:hover {{ stroke: #667eea; stroke-width: 3px; }}
        .link {{ stroke: #999; stroke-opacity: 0.6; }}
        .tooltip {{
            position: absolute; padding: 10px; background: rgba(0,0,0,0.8);
            color: white; border-radius: 4px; pointer-events: none; opacity: 0;
        }}
    </style>
</head>
<body>
    <svg id="graph"></svg>
    <div class="tooltip" id="tooltip"></div>
    
    <script>
        const data = {json.dumps(graph_data)};
        const width = window.innerWidth;
        const height = window.innerHeight;
        
        const svg = d3.select("#graph").attr("viewBox", [0, 0, width, height]);
        const g = svg.append("g");
        
        const zoom = d3.zoom()
            .scaleExtent([0.1, 10])
            .on("zoom", (e) => g.attr("transform", e.transform));
        svg.call(zoom);
        
        const simulation = d3.forceSimulation(data.nodes)
            .force("link", d3.forceLink(data.links).id(d => d.id).distance(100))
            .force("charge", d3.forceManyBody().strength(-200))
            .force("center", d3.forceCenter(width / 2, height / 2));
        
        const link = g.append("g").selectAll("line")
            .data(data.links).join("line")
            .attr("class", "link")
            .attr("stroke-width", 2);
        
        const node = g.append("g").selectAll("circle")
            .data(data.nodes).join("circle")
            .attr("class", "node")
            .attr("r", 10)
            .attr("fill", d => d.color)
            .call(d3.drag()
                .on("start", (e) => {{
                    if (!e.active) simulation.alphaTarget(0.3).restart();
                    e.subject.fx = e.subject.x;
                    e.subject.fy = e.subject.y;
                }})
                .on("drag", (e) => {{
                    e.subject.fx = e.x;
                    e.subject.fy = e.y;
                }})
                .on("end", (e) => {{
                    if (!e.active) simulation.alphaTarget(0);
                    e.subject.fx = null;
                    e.subject.fy = null;
                }}))
            .on("mouseover", (event, d) => {{
                d3.select("#tooltip")
                    .html(`<strong>${{d.name}}</strong><br>Type: ${{d.type}}`)
                    .style("left", (event.pageX + 10) + "px")
                    .style("top", (event.pageY - 10) + "px")
                    .style("opacity", 1);
            }})
            .on("mouseout", () => {{
                d3.select("#tooltip").style("opacity", 0);
            }});
        
        const label = g.append("g").selectAll("text")
            .data(data.nodes).join("text")
            .text(d => d.name)
            .attr("font-size", 10)
            .attr("text-anchor", "middle")
            .attr("dy", 15);
        
        simulation.on("tick", () => {{
            link.attr("x1", d => d.source.x).attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
            node.attr("cx", d => d.x).attr("cy", d => d.y);
            label.attr("x", d => d.x).attr("y", d => d.y);
        }});
    </script>
</body>
</html>"""
    
    # Write file
    Path(output_file).write_text(html, encoding='utf-8')
    print(f"✓ Created: {output_file}")
    print(f"  Open this file in your browser!")
    return output_file


# Quick test
if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║        Simple HTML Graph Visualizer - NO GraphViz Needed!           ║
╚══════════════════════════════════════════════════════════════════════╝

USAGE:
    from simple_html_viz import create_html_visualization
    from knowledge_graph import KnowledgeGraph
    
    kg = KnowledgeGraph()
    # ... populate kg with your TAL code ...
    
    # Create visualization
    create_html_visualization(kg, "my_graph.html")
    
    # Open my_graph.html in your browser!

FEATURES:
✓ NO system binaries needed
✓ Interactive: zoom, pan, drag nodes
✓ Hover to see node details
✓ Works in any browser
    """)
