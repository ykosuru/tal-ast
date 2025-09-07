#!/usr/bin/env python3
"""
Generate a self-contained HTML visualization with embedded data
"""

import json
import os
from pathlib import Path

def create_embedded_visualization():
    """Create HTML file with embedded graph data"""
    
    # Load the graph data
    graph_file = "capability_mapping_output/capability_keyword_procedure_graph.json"
    
    if not os.path.exists(graph_file):
        print(f"Error: {graph_file} not found!")
        print("Make sure you've run the keywordslinker.py script first.")
        return
    
    with open(graph_file, 'r') as f:
        graph_data = json.load(f)
    
    # Create HTML with embedded data
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Business Capability Graph Explorer</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f7fa;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            text-align: center;
        }}
        
        .header h1 {{
            margin: 0;
            font-size: 2.2em;
            font-weight: 300;
        }}
        
        .navigation {{
            background: #2c3e50;
            color: white;
            padding: 15px 25px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
        }}
        
        .breadcrumb {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .breadcrumb-item {{
            padding: 8px 15px;
            background: rgba(255,255,255,0.1);
            border-radius: 20px;
            cursor: pointer;
            transition: all 0.3s ease;
        }}
        
        .breadcrumb-item:hover {{
            background: rgba(255,255,255,0.2);
        }}
        
        .breadcrumb-separator {{
            color: #bdc3c7;
        }}
        
        .stats {{
            display: flex;
            gap: 20px;
            font-size: 14px;
        }}
        
        .stat-item {{
            background: rgba(255,255,255,0.1);
            padding: 5px 12px;
            border-radius: 15px;
        }}
        
        .visualization {{
            padding: 30px;
            min-height: 600px;
        }}
        
        .level-title {{
            font-size: 1.5em;
            margin-bottom: 25px;
            color: #2c3e50;
            text-align: center;
        }}
        
        .node-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            max-height: 70vh;
            overflow-y: auto;
        }}
        
        .node-card {{
            background: white;
            border: 2px solid #e0e6ed;
            border-radius: 10px;
            padding: 20px;
            cursor: pointer;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }}
        
        .node-card:hover {{
            border-color: #667eea;
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.15);
        }}
        
        .node-card.capability {{
            border-left: 5px solid #3498db;
        }}
        
        .node-card.keyword {{
            border-left: 5px solid #f39c12;
        }}
        
        .node-card.procedure {{
            border-left: 5px solid #27ae60;
        }}
        
        .node-title {{
            font-weight: 600;
            font-size: 1.1em;
            margin-bottom: 10px;
            color: #2c3e50;
            word-wrap: break-word;
        }}
        
        .node-meta {{
            font-size: 0.9em;
            color: #7f8c8d;
            margin-bottom: 15px;
        }}
        
        .confidence-bar {{
            width: 100%;
            height: 8px;
            background: #ecf0f1;
            border-radius: 4px;
            overflow: hidden;
            margin: 10px 0;
        }}
        
        .confidence-fill {{
            height: 100%;
            border-radius: 4px;
            transition: all 0.3s ease;
        }}
        
        .confidence-high {{ background: #27ae60; }}
        .confidence-medium {{ background: #f39c12; }}
        .confidence-low {{ background: #e74c3c; }}
        
        .tags {{
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
            margin-top: 10px;
        }}
        
        .tag {{
            background: #ecf0f1;
            color: #2c3e50;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 0.8em;
        }}
        
        .tag.match-exact {{ background: #d5f4e6; color: #27ae60; }}
        .tag.match-partial {{ background: #ffeaa7; color: #d63031; }}
        .tag.match-semantic {{ background: #a29bfe; color: #6c5ce7; }}
        
        .procedure-details {{
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid #ecf0f1;
        }}
        
        .detail-row {{
            display: flex;
            justify-content: space-between;
            margin: 5px 0;
            font-size: 0.9em;
        }}
        
        .detail-label {{
            font-weight: 500;
            color: #2c3e50;
        }}
        
        .detail-value {{
            color: #7f8c8d;
            text-align: right;
            max-width: 60%;
            word-wrap: break-word;
        }}
        
        .search-box {{
            margin-bottom: 20px;
            position: relative;
        }}
        
        .search-input {{
            width: 100%;
            padding: 12px 20px;
            border: 2px solid #e0e6ed;
            border-radius: 25px;
            font-size: 16px;
            outline: none;
            transition: border-color 0.3s ease;
        }}
        
        .search-input:focus {{
            border-color: #667eea;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Business Capability Graph Explorer</h1>
        </div>
        
        <div class="navigation">
            <div class="breadcrumb" id="breadcrumb">
                <div class="breadcrumb-item" onclick="showCapabilities()">Capabilities</div>
            </div>
            <div class="stats" id="stats"></div>
        </div>
        
        <div class="visualization">
            <div class="search-box">
                <input type="text" class="search-input" id="searchInput" placeholder="Search capabilities, keywords, or procedures...">
            </div>
            <div id="content">
                <div class="loading">Loading data...</div>
            </div>
        </div>
    </div>

    <script>
        // Embedded data
        const graphData = {json.dumps(graph_data, indent=8)};
        
        // Application state
        let currentView = 'capabilities';
        let currentFilter = '';
        let breadcrumbPath = [];
        
        function updateStats() {{
            const capabilities = Object.keys(graphData.capabilities_to_keywords || {{}}).length;
            const keywords = Object.keys(graphData.keywords_to_procedures || {{}}).length;
            const procedures = Object.keys(graphData.procedures_to_capabilities || {{}}).length;
            
            document.getElementById('stats').innerHTML = `
                <div class="stat-item">${{capabilities}} Capabilities</div>
                <div class="stat-item">${{keywords}} Keywords</div>
                <div class="stat-item">${{procedures}} Procedures</div>
            `;
        }}
        
        function showCapabilities() {{
            currentView = 'capabilities';
            breadcrumbPath = [];
            updateBreadcrumb();
            
            const capabilities = Object.keys(graphData.capabilities_to_keywords || {{}});
            const filtered = filterItems(capabilities);
            
            const html = `
                <h2 class="level-title">Business Capabilities (${{filtered.length}})</h2>
                <div class="node-grid">
                    ${{filtered.map(capability => {{
                        const keywordCount = graphData.capabilities_to_keywords[capability]?.length || 0;
                        return `
                            <div class="node-card capability" onclick="showKeywords('${{escapeHtml(capability)}}')">
                                <div class="node-title">${{escapeHtml(capability)}}</div>
                                <div class="node-meta">${{keywordCount}} keywords mapped</div>
                                <div class="tags">
                                    <div class="tag">Click to explore</div>
                                </div>
                            </div>
                        `;
                    }}).join('')}}
                </div>
            `;
            
            document.getElementById('content').innerHTML = html;
        }}
        
        function showKeywords(capability) {{
            currentView = 'keywords';
            breadcrumbPath = [capability];
            updateBreadcrumb();
            
            const keywords = graphData.capabilities_to_keywords[capability] || [];
            const filtered = filterItems(keywords);
            
            const html = `
                <h2 class="level-title">Keywords for "${{capability}}" (${{filtered.length}})</h2>
                <div class="node-grid">
                    ${{filtered.map(keyword => {{
                        const procedures = graphData.keywords_to_procedures[keyword] || [];
                        const procedureCount = procedures.length;
                        const avgConfidence = procedures.length > 0 
                            ? procedures.reduce((sum, p) => sum + p.confidence, 0) / procedures.length 
                            : 0;
                        
                        return `
                            <div class="node-card keyword" onclick="showProcedures('${{escapeHtml(keyword)}}')">
                                <div class="node-title">${{escapeHtml(keyword)}}</div>
                                <div class="node-meta">${{procedureCount}} procedures mapped</div>
                                ${{avgConfidence > 0 ? `
                                    <div class="confidence-bar">
                                        <div class="confidence-fill ${{getConfidenceClass(avgConfidence)}}" 
                                             style="width: ${{avgConfidence * 100}}%"></div>
                                    </div>
                                    <div style="font-size: 0.8em; color: #7f8c8d;">
                                        Avg Confidence: ${{(avgConfidence * 100).toFixed(1)}}%
                                    </div>
                                ` : ''}}
                                <div class="tags">
                                    <div class="tag">Click to view procedures</div>
                                </div>
                            </div>
                        `;
                    }}).join('')}}
                </div>
            `;
            
            document.getElementById('content').innerHTML = html;
        }}
        
        function showProcedures(keyword) {{
            currentView = 'procedures';
            breadcrumbPath.push(keyword);
            updateBreadcrumb();
            
            const procedures = graphData.keywords_to_procedures[keyword] || [];
            const filtered = procedures.filter(proc => 
                proc.procedure.toLowerCase().includes(currentFilter.toLowerCase())
            );
            
            const html = `
                <h2 class="level-title">Procedures for keyword "${{keyword}}" (${{filtered.length}})</h2>
                <div class="node-grid">
                    ${{filtered.map(procInfo => {{
                        return `
                            <div class="node-card procedure">
                                <div class="node-title">${{escapeHtml(procInfo.procedure)}}</div>
                                <div class="confidence-bar">
                                    <div class="confidence-fill ${{getConfidenceClass(procInfo.confidence)}}" 
                                         style="width: ${{procInfo.confidence * 100}}%"></div>
                                </div>
                                <div class="procedure-details">
                                    <div class="detail-row">
                                        <span class="detail-label">Confidence:</span>
                                        <span class="detail-value">${{(procInfo.confidence * 100).toFixed(1)}}%</span>
                                    </div>
                                    <div class="detail-row">
                                        <span class="detail-label">Match Type:</span>
                                        <span class="detail-value">
                                            <span class="tag match-${{procInfo.match_type}}">${{procInfo.match_type}}</span>
                                        </span>
                                    </div>
                                    <div class="detail-row">
                                        <span class="detail-label">File:</span>
                                        <span class="detail-value">${{escapeHtml(procInfo.file_path)}}</span>
                                    </div>
                                </div>
                            </div>
                        `;
                    }}).join('')}}
                </div>
            `;
            
            document.getElementById('content').innerHTML = html;
        }}
        
        function updateBreadcrumb() {{
            const breadcrumb = document.getElementById('breadcrumb');
            let html = '<div class="breadcrumb-item" onclick="showCapabilities()">Capabilities</div>';
            
            if (breadcrumbPath.length > 0) {{
                html += '<span class="breadcrumb-separator">→</span>';
                html += `<div class="breadcrumb-item" onclick="showKeywords('${{escapeHtml(breadcrumbPath[0])}}')">${{escapeHtml(breadcrumbPath[0])}}</div>`;
                
                if (breadcrumbPath.length > 1) {{
                    html += '<span class="breadcrumb-separator">→</span>';
                    html += `<div class="breadcrumb-item">${{escapeHtml(breadcrumbPath[1])}}</div>`;
                }}
            }}
            
            breadcrumb.innerHTML = html;
        }}
        
        function filterItems(items) {{
            if (!currentFilter) return items;
            return items.filter(item => 
                item.toString().toLowerCase().includes(currentFilter.toLowerCase())
            );
        }}
        
        function getConfidenceClass(confidence) {{
            if (confidence >= 0.7) return 'confidence-high';
            if (confidence >= 0.4) return 'confidence-medium';
            return 'confidence-low';
        }}
        
        function escapeHtml(text) {{
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }}
        
        // Search functionality
        document.getElementById('searchInput').addEventListener('input', function(e) {{
            currentFilter = e.target.value;
            
            if (currentView === 'capabilities') {{
                showCapabilities();
            }} else if (currentView === 'keywords' && breadcrumbPath.length > 0) {{
                showKeywords(breadcrumbPath[0]);
            }} else if (currentView === 'procedures' && breadcrumbPath.length > 1) {{
                showProcedures(breadcrumbPath[1]);
            }}
        }});
        
        // Initialize
        updateStats();
        showCapabilities();
    </script>
</body>
</html>"""
    
    # Write the HTML file
    output_file = "capability_graph_viewer.html"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Created self-contained visualization: {output_file}")
    print(f"📊 Embedded data: {len(graph_data.get('capabilities_to_keywords', {}))} capabilities")
    print(f"🌐 Open {output_file} directly in your browser (no web server needed)")

if __name__ == "__main__":
    create_embedded_visualization()

