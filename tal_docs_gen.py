#!/usr/bin/env python3
"""
TAL DDG with Source Code from Graph File Paths
Uses file_path attribute from graph nodes to load actual TAL source code
No separate source directory needed - reads from paths in the graph!
"""

import json
import networkx as nx
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ArchitectureComponent:
    """Represents a component in the architecture"""
    name: str
    type: str
    description: str
    dependencies: List[str]
    call_count: int
    complexity: str
    business_capability: str
    source_code: Optional[str] = None
    file_path: Optional[str] = None


@dataclass
class ProcessFlow:
    """Represents a process flow through the system"""
    name: str
    entry_point: str
    steps: List[Dict[str, Any]]
    data_flow: List[str]
    decision_points: List[str]


class TALDocumentationGenerator:
    """Generate documentation from TAL knowledge graph with source code from file paths"""
    
    def __init__(self, graph_path: Optional[str] = None):
        """
        Initialize the documentation generator
        
        Args:
            graph_path: Path to knowledge graph (GraphML or JSON)
        """
        self.graph = None
        self.graph_data = None
        self.source_cache = {}  # Cache loaded source files by file path
        self.components = []
        self.process_flows = []
        self.statistics = {}
        self.base_path = None  # Base path for resolving relative paths
        
        if graph_path:
            self.base_path = Path(graph_path).parent
            self.load_graph(graph_path)
    
    def load_graph(self, graph_path: str):
        """Load the TAL knowledge graph"""
        logger.info(f"Loading knowledge graph from {graph_path}")
        
        path = Path(graph_path)
        if path.suffix == '.graphml':
            self.graph = nx.read_graphml(graph_path)
            self.graph_data = nx.node_link_data(self.graph)
        elif path.suffix == '.json':
            with open(graph_path, 'r') as f:
                data = json.load(f)
                self.graph_data = data
                
                edge_key = None
                for key in ['links', 'edges', 'relationships']:
                    if key in data:
                        edge_key = key
                        break
                
                if edge_key is None:
                    raise ValueError("JSON must contain 'links', 'edges', or 'relationships' key")
                
                if 'nodes' not in data:
                    raise ValueError("JSON must contain 'nodes' key")
                
                clean_data = {
                    'directed': data.get('directed', True),
                    'multigraph': data.get('multigraph', False),
                    'graph': data.get('graph', {}),
                    'nodes': data['nodes'],
                    'links': data[edge_key]
                }
                
                if clean_data['directed'] is None:
                    clean_data['directed'] = True
                    logger.warning("'directed' key was null, assuming directed graph")
                
                self.graph = nx.node_link_graph(clean_data)
        else:
            raise ValueError(f"Unsupported graph format: {path.suffix}")
        
        logger.info(f"Loaded graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        
        # Pre-load source files from file_path attributes
        self._load_source_files_from_graph()
    
    def _load_source_files_from_graph(self):
        """Load TAL source files using file_path from graph nodes"""
        logger.info("Loading TAL source files from graph file_path attributes...")
        
        # Collect unique file paths from nodes
        file_paths = set()
        for node, data in self.graph.nodes(data=True):
            file_path = data.get('file_path')
            if file_path:
                file_paths.add(file_path)
        
        logger.info(f"Found {len(file_paths)} unique file paths in graph")
        
        # Load each file
        loaded_count = 0
        for file_path_str in file_paths:
            file_path = Path(file_path_str)
            
            # Try absolute path first
            if file_path.exists():
                actual_path = file_path
            # Try relative to graph file
            elif self.base_path and (self.base_path / file_path).exists():
                actual_path = self.base_path / file_path
            # Try just the filename in base directory
            elif self.base_path and (self.base_path / file_path.name).exists():
                actual_path = self.base_path / file_path.name
            else:
                logger.debug(f"File not found: {file_path_str}")
                continue
            
            try:
                with open(actual_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    # Store by original path string for matching
                    self.source_cache[file_path_str] = content
                    loaded_count += 1
                    
            except Exception as e:
                logger.debug(f"Could not read {actual_path}: {e}")
        
        logger.info(f"Successfully loaded {loaded_count}/{len(file_paths)} source files")
        
        if loaded_count == 0:
            logger.warning("No source files loaded! Make sure file paths in graph are correct.")
            logger.warning(f"Base path: {self.base_path}")
            logger.warning(f"Sample file paths from graph: {list(file_paths)[:5]}")
    
    def _get_source_code_for_node(self, node_data: Dict, max_lines: int = 200) -> Optional[str]:
        """
        Get source code for a node using its file_path attribute
        
        Args:
            node_data: Node data dict with file_path
            max_lines: Maximum lines to return (for token efficiency)
        
        Returns:
            Source code or None if not found
        """
        file_path = node_data.get('file_path')
        if not file_path:
            return None
        
        # Get full file content
        content = self.source_cache.get(file_path)
        if not content:
            return None
        
        # For procedures, try to extract just that procedure
        if node_data.get('type') == 'procedure':
            proc_name = node_data.get('name')
            if proc_name:
                # Try to find PROC procname ... END pattern
                import re
                # Match PROC name ... END or PROC name(...) ... END
                pattern = rf'PROC\s+{re.escape(proc_name)}\s*[(\[].*?(?:END|ENDPROC)'
                match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
                if match:
                    proc_code = match.group(0)
                    lines = proc_code.split('\n')
                    if len(lines) > max_lines:
                        truncated = '\n'.join(lines[:max_lines])
                        return f"{truncated}\n\n... [truncated - {len(lines)} total lines]"
                    return proc_code
        
        # Return full file (truncated if needed)
        lines = content.split('\n')
        if len(lines) > max_lines:
            truncated = '\n'.join(lines[:max_lines])
            return f"{truncated}\n\n... [truncated - {len(lines)} total lines]"
        return content
    
    def _extract_graph_context(self, nodes_limit: int = 50, edges_limit: int = 100) -> Dict[str, Any]:
        """Extract graph structure as rich context for LLM"""
        edge_key = 'edges' if 'edges' in self.graph_data else 'links'
        
        context = {
            'total_nodes': len(self.graph_data.get('nodes', [])),
            'total_edges': len(self.graph_data.get(edge_key, [])),
            'nodes': self.graph_data.get('nodes', [])[:nodes_limit],
            'edges': self.graph_data.get(edge_key, [])[:edges_limit],
            'graph_metadata': self.graph_data.get('graph', {})
        }
        
        return context
    
    def analyze_graph(self):
        """Analyze the knowledge graph and extract architectural information"""
        logger.info("Analyzing knowledge graph...")
        
        self.components = self._extract_components()
        self.process_flows = self._identify_process_flows()
        self.statistics = self._calculate_statistics()
        
        logger.info(f"Found {len(self.components)} components and {len(self.process_flows)} process flows")
        
        # Count how many have source code
        with_code = sum(1 for c in self.components if c.source_code)
        logger.info(f"Found source code for {with_code}/{len(self.components)} components ({with_code/max(len(self.components),1)*100:.1f}%)")
    
    def _extract_components(self) -> List[ArchitectureComponent]:
        """Extract architectural components from the graph"""
        components = []
        
        for node, data in self.graph.nodes(data=True):
            node_type = data.get('type', 'unknown')
            
            if node_type in ['intrinsic', 'system', 'utility']:
                continue
            
            dependencies = list(self.graph.successors(node))
            call_count = self.graph.in_degree(node)
            out_degree = self.graph.out_degree(node)
            
            if out_degree > 10 or call_count > 20:
                complexity = "high"
            elif out_degree > 5 or call_count > 10:
                complexity = "medium"
            else:
                complexity = "low"
            
            # Load source code using file_path from node
            source_code = self._get_source_code_for_node(data)
            
            component = ArchitectureComponent(
                name=node,
                type=node_type,
                description=data.get('description', ''),
                dependencies=dependencies[:10],
                call_count=call_count,
                complexity=complexity,
                business_capability=data.get('business_capability', 'unknown'),
                source_code=source_code,
                file_path=data.get('file_path')
            )
            components.append(component)
        
        components.sort(key=lambda x: x.call_count, reverse=True)
        return components
    
    def _identify_process_flows(self) -> List[ProcessFlow]:
        """Identify key process flows through the system"""
        flows = []
        
        entry_points = [
            node for node, in_deg in self.graph.in_degree()
            if in_deg <= 2 and self.graph.out_degree(node) >= 3
        ]
        
        for entry_point in entry_points[:10]:
            steps = []
            visited = set()
            queue = [(entry_point, 0)]
            
            while queue and len(steps) < 20:
                current, depth = queue.pop(0)
                if current in visited:
                    continue
                
                visited.add(current)
                node_data = self.graph.nodes[current]
                
                steps.append({
                    'node': current,
                    'type': node_data.get('type', 'unknown'),
                    'depth': depth,
                    'description': node_data.get('description', '')
                })
                
                for successor in self.graph.successors(current):
                    if successor not in visited:
                        queue.append((successor, depth + 1))
            
            if len(steps) >= 3:
                flow = ProcessFlow(
                    name=f"Flow from {entry_point}",
                    entry_point=entry_point,
                    steps=steps,
                    data_flow=[s['node'] for s in steps],
                    decision_points=[s['node'] for s in steps if self.graph.out_degree(s['node']) > 2]
                )
                flows.append(flow)
        
        return flows
    
    def _calculate_statistics(self) -> Dict[str, Any]:
        """Calculate various statistics about the codebase"""
        stats = {
            'total_procedures': self.graph.number_of_nodes(),
            'total_calls': self.graph.number_of_edges(),
            'avg_dependencies': sum(d for n, d in self.graph.out_degree()) / max(self.graph.number_of_nodes(), 1),
            'max_dependencies': max(d for n, d in self.graph.out_degree()) if self.graph.number_of_nodes() > 0 else 0,
            'most_called': max(self.graph.in_degree(), key=lambda x: x[1])[0] if self.graph.number_of_nodes() > 0 else None,
            'complexity_distribution': {}
        }
        
        complexity_counts = defaultdict(int)
        for comp in self.components:
            complexity_counts[comp.complexity] += 1
        
        stats['complexity_distribution'] = dict(complexity_counts)
        
        strongly_connected = list(nx.strongly_connected_components(self.graph))
        stats['circular_dependency_groups'] = len([c for c in strongly_connected if len(c) > 1])
        
        # Add source code statistics
        stats['procedures_with_source'] = sum(1 for c in self.components if c.source_code)
        stats['source_coverage'] = stats['procedures_with_source'] / max(len(self.components), 1)
        
        return stats
    
    def call_llm(self, prompt: str, system_prompt: str = "", temperature: float = 0.7, max_tokens: int = 2000) -> str:
        """Call LLM using OpenAI-compatible API"""
        try:
            # PLACEHOLDER - Replace with actual API call
            logger.warning("LLM call placeholder - implement actual API call")
            return f"[LLM Response Placeholder]\nPrompt length: {len(prompt)} chars"
            
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            return f"Error: {e}"
    
    def generate_architecture_overview(self) -> str:
        """Generate high-level architecture overview using LLM with code context"""
        logger.info("Generating architecture overview with source code context...")
        
        graph_context = self._extract_graph_context(nodes_limit=30, edges_limit=50)
        
        component_types = defaultdict(int)
        for comp in self.components:
            component_types[comp.type] += 1
        
        capabilities = defaultdict(list)
        for comp in self.components[:30]:
            capabilities[comp.business_capability].append(comp.name)
        
        # Include sample source code snippets from top procedures
        code_samples = []
        for comp in self.components[:3]:
            if comp.source_code:
                lines = comp.source_code.split('\n')[:30]
                code_samples.append({
                    'name': comp.name,
                    'file_path': comp.file_path,
                    'snippet': '\n'.join(lines)
                })
        
        prompt = f"""
Analyze this TAL payment processing system with actual source code.

=== GRAPH STRUCTURE ===
Total Nodes: {graph_context['total_nodes']}
Total Edges: {graph_context['total_edges']}

Sample Nodes (with file paths):
{json.dumps(graph_context['nodes'][:15], indent=2)}

Sample Edges (call relationships):
{json.dumps(graph_context['edges'][:20], indent=2)}

=== STATISTICS ===
- Total Procedures: {self.statistics['total_procedures']}
- Total Calls: {self.statistics['total_calls']}
- Average Dependencies: {self.statistics['avg_dependencies']:.2f}
- Most Called: {self.statistics.get('most_called', 'N/A')}
- Circular Dependencies: {self.statistics.get('circular_dependency_groups', 0)}
- Source Code Coverage: {self.statistics.get('source_coverage', 0):.1%}

=== COMPONENT TYPES ===
{json.dumps(dict(component_types), indent=2)}

=== TOP CRITICAL COMPONENTS ===
{json.dumps([asdict(c) for c in self.components[:10]], indent=2, default=str)}

=== BUSINESS CAPABILITIES ===
{json.dumps({k: v[:5] for k, v in capabilities.items()}, indent=2)}

=== ACTUAL TAL SOURCE CODE SAMPLES ===
{json.dumps(code_samples, indent=2)}

Provide comprehensive, code-informed architecture overview:

1. **Architecture Overview** (3-4 paragraphs)
   - System design from actual code
   - Architectural patterns observed
   - Component organization

2. **Key Subsystems**
   - Subsystems from capabilities
   - Implementation patterns from code
   - Key procedures

3. **Integration Points**
   - How subsystems communicate (from code)
   - Data flow patterns
   - Critical integration procedures

4. **Code Quality & Complexity**
   - Code structure observations
   - Complexity hotspots
   - Technical debt indicators

5. **Architectural Assessment**
   - Strengths (with code examples)
   - Weaknesses (with code examples)
   - Modernization opportunities

Reference actual code patterns and procedures.
"""
        
        system_prompt = """You are a senior software architect analyzing legacy TAL payment systems. You have access to both the call graph AND actual source code from file paths. Provide specific, code-informed insights."""
        
        return self.call_llm(prompt, system_prompt, temperature=0.3, max_tokens=3500)
    
    def generate_component_documentation(self, component: ArchitectureComponent) -> str:
        """Generate detailed documentation for component with actual source code"""
        
        callers = [n for n in self.graph.nodes() if component.name in list(self.graph.successors(n))]
        
        dependency_details = []
        for dep in component.dependencies[:8]:
            if dep in self.graph.nodes:
                dep_data = self.graph.nodes[dep]
                dependency_details.append({
                    'name': dep,
                    'type': dep_data.get('type', 'unknown'),
                    'capability': dep_data.get('business_capability', 'unknown'),
                    'file_path': dep_data.get('file_path')
                })
        
        # Include dependency source code for context
        dependency_code = []
        for dep in component.dependencies[:3]:
            if dep in self.graph.nodes:
                dep_data = self.graph.nodes[dep]
                dep_code = self._get_source_code_for_node(dep_data, max_lines=50)
                if dep_code:
                    dependency_code.append({
                        'name': dep,
                        'file_path': dep_data.get('file_path'),
                        'code': dep_code
                    })
        
        prompt = f"""
Analyze this TAL component with actual source code:

=== COMPONENT DETAILS ===
Name: {component.name}
Type: {component.type}
File: {component.file_path}
Business Capability: {component.business_capability}
Complexity: {component.complexity}
Times Called: {component.call_count}
Dependencies: {len(component.dependencies)}

=== ACTUAL SOURCE CODE ===
```tal
{component.source_code if component.source_code else '[Source code not available - file not found]'}
```

=== CALLERS (who calls this) ===
{json.dumps(callers[:10], indent=2)}

=== DEPENDENCIES (what this calls) ===
{json.dumps(dependency_details, indent=2)}

=== SAMPLE DEPENDENCY CODE ===
{json.dumps(dependency_code, indent=2)}

Provide code-informed documentation:

1. **Purpose and Functionality**
   - What does the code do?
   - Key algorithms and logic
   - Business rules implemented

2. **Code Structure Analysis**
   - Variables and data structures
   - Control flow patterns
   - Error handling

3. **Role in Architecture**
   - Position in call hierarchy ({component.call_count} callers)
   - Integration patterns
   - Criticality

4. **Dependencies Analysis**
   - Why each dependency (from code)
   - Data passed
   - Integration patterns

5. **Callers Analysis**
   - Who relies on this
   - Usage patterns
   - Impact if fails

6. **Code Quality**
   - Complexity indicators
   - Maintainability
   - Technical debt

7. **Modernization**
   - Refactoring opportunities
   - API design
   - Migration strategy
   - Microservice extraction

Reference actual code patterns, variables, and logic.
"""
        
        return self.call_llm(prompt, "You are documenting legacy TAL payment code with access to actual source.", temperature=0.5, max_tokens=2000)
    
    def generate_process_flow_documentation(self, flow: ProcessFlow) -> str:
        """Generate documentation for process flow with actual code"""
        
        detailed_steps = []
        for step in flow.steps[:12]:
            node_name = step['node']
            if node_name in self.graph.nodes:
                node_data = dict(self.graph.nodes[node_name])
                code = self._get_source_code_for_node(node_data, max_lines=30)
                detailed_steps.append({
                    'node': node_name,
                    'type': step['type'],
                    'depth': step['depth'],
                    'file_path': node_data.get('file_path'),
                    'out_degree': self.graph.out_degree(node_name),
                    'in_degree': self.graph.in_degree(node_name),
                    'code_snippet': code[:500] if code else None
                })
        
        prompt = f"""
Document this process flow with actual TAL code:

=== FLOW OVERVIEW ===
Name: {flow.name}
Entry Point: {flow.entry_point}
Steps: {len(flow.steps)}
Decision Points: {len(flow.decision_points)}

=== DETAILED FLOW WITH CODE ===
{json.dumps(detailed_steps, indent=2)}

=== DECISION POINTS ===
{json.dumps(flow.decision_points[:5], indent=2)}

=== DATA FLOW PATH ===
{' -> '.join(flow.data_flow[:15])}

Provide code-informed process documentation:

1. **Business Process**
   - What operation (from code)?
   - When triggered?
   - Expected outcomes

2. **Step-by-Step Flow**
   - Each step with code insights
   - Data transformations
   - State changes

3. **Decision Points**
   - Branching conditions (from code)
   - Alternative paths
   - Error handling

4. **Data Flow**
   - Input data (variables)
   - Transformations
   - Output data

5. **Integration Points**
   - External calls
   - Database operations
   - Message passing

6. **Error Handling**
   - Detection in code
   - Rollback mechanisms
   - Recovery procedures

7. **Performance**
   - Bottlenecks from code
   - Optimization opportunities

Reference actual code patterns and logic.
"""
        
        return self.call_llm(prompt, "You are documenting payment workflows with access to actual TAL code.", temperature=0.5, max_tokens=2500)
    
    def identify_microservices_candidates(self) -> Dict[str, Any]:
        """Identify microservice candidates with code analysis"""
        logger.info("Identifying microservice candidates with code context...")
        
        capabilities = defaultdict(list)
        for comp in self.components:
            capabilities[comp.business_capability].append(comp)
        
        candidates = []
        
        for capability, comps in capabilities.items():
            if len(comps) < 3:
                continue
            
            internal_calls = 0
            external_calls = 0
            external_deps = defaultdict(int)
            
            comp_names = {c.name for c in comps}
            for comp in comps:
                for dep in self.graph.successors(comp.name):
                    if dep in comp_names:
                        internal_calls += 1
                    else:
                        external_calls += 1
                        if dep in self.graph.nodes:
                            dep_cap = self.graph.nodes[dep].get('business_capability', 'unknown')
                            external_deps[dep_cap] += 1
            
            cohesion_ratio = internal_calls / max(internal_calls + external_calls, 1)
            
            # Get code samples
            code_samples = []
            for comp in comps[:3]:
                if comp.source_code:
                    lines = comp.source_code.split('\n')[:40]
                    code_samples.append({
                        'name': comp.name,
                        'file_path': comp.file_path,
                        'snippet': '\n'.join(lines)
                    })
            
            context = {
                'capability': capability,
                'component_count': len(comps),
                'total_calls': sum(c.call_count for c in comps),
                'components': [asdict(c) for c in comps[:10]],
                'internal_calls': internal_calls,
                'external_calls': external_calls,
                'cohesion_ratio': round(cohesion_ratio, 2),
                'external_dependencies': dict(external_deps),
                'code_available': sum(1 for c in comps if c.source_code)
            }
            
            prompt = f"""
Evaluate as microservice with actual code:

=== METRICS ===
Capability: {capability}
Components: {len(comps)}
Calls: {context['total_calls']}
Cohesion: {cohesion_ratio:.1%}
Internal: {internal_calls} | External: {external_calls}
Code Available: {context['code_available']}/{len(comps)}

=== COMPONENTS ===
{json.dumps(context['components'], indent=2, default=str)}

=== EXTERNAL DEPENDENCIES ===
{json.dumps(context['external_dependencies'], indent=2)}

=== ACTUAL CODE SAMPLES ===
{json.dumps(code_samples, indent=2)}

Assess with code analysis:

1. **Recommendation** (Yes/No/Maybe)
2. **Cohesion Analysis** (code patterns)
3. **Coupling Analysis** (dependencies from code)
4. **Service Design** (API from code)
5. **Migration Strategy**
6. **Benefits & Risks**

Reference actual code patterns.
"""
            
            response = self.call_llm(prompt, "You are analyzing legacy code for microservice extraction.", temperature=0.3, max_tokens=2000)
            
            candidates.append({
                'capability': capability,
                'components': [asdict(c) for c in comps[:10]],
                'metrics': context,
                'analysis': response
            })
        
        return {'candidates': candidates}
    
    def generate_data_flow_analysis(self) -> str:
        """Analyze data flows with code context"""
        logger.info("Analyzing data flows with code...")
        
        entry_points = [n for n in self.graph.nodes() if self.graph.in_degree(n) <= 1]
        
        sample_paths = []
        for entry in entry_points[:5]:
            paths = nx.single_source_shortest_path(self.graph, entry, cutoff=8)
            if paths:
                longest = max(paths.values(), key=len)
                if len(longest) > 3:
                    path_details = []
                    for node in longest:
                        node_data = self.graph.nodes[node]
                        path_details.append({
                            'node': node,
                            'type': node_data.get('type', 'unknown'),
                            'file_path': node_data.get('file_path')
                        })
                    sample_paths.append(path_details)
        
        # Data hubs with code
        hubs = []
        for node, data in self.graph.nodes(data=True):
            in_deg = self.graph.in_degree(node)
            out_deg = self.graph.out_degree(node)
            if in_deg >= 3 and out_deg >= 3:
                code = self._get_source_code_for_node(data, max_lines=40)
                hubs.append({
                    'node': node,
                    'file_path': data.get('file_path'),
                    'in_degree': in_deg,
                    'out_degree': out_deg,
                    'has_code': code is not None,
                    'code_snippet': code[:300] if code else None
                })
        
        prompt = f"""
Analyze data flows with code:

=== OVERVIEW ===
Procedures: {self.statistics['total_procedures']}
Paths: {self.statistics['total_calls']}
Avg Dependencies: {self.statistics['avg_dependencies']:.2f}

=== SAMPLE PATHS ===
{json.dumps(sample_paths, indent=2)}

=== DATA HUBS (high connectivity) ===
{json.dumps(sorted(hubs, key=lambda x: x['in_degree'] + x['out_degree'], reverse=True)[:8], indent=2)}

=== TOP DATA CONSUMERS ===
{json.dumps([{'name': c.name, 'file_path': c.file_path, 'call_count': c.call_count} for c in self.components[:10]], indent=2)}

Provide code-informed analysis:

1. **Data Flow Patterns** (from code)
2. **Data Transformations** (stages, structures)
3. **Critical Hubs** (why from code)
4. **Data Consistency** (mechanisms in code)
5. **Data Issues** (coupling, shared state)
6. **Modernization** (events, APIs, CQRS)

Reference actual code patterns.
"""
        
        return self.call_llm(prompt, "You are analyzing payment system data flows with code.", temperature=0.4, max_tokens=2800)
    
    def generate_modernization_roadmap(self) -> str:
        """Generate modernization roadmap with code insights"""
        logger.info("Generating modernization roadmap...")
        
        quick_wins = [c for c in self.components if c.complexity == 'low' and len(c.dependencies) <= 3][:10]
        risky = [c for c in self.components if c.complexity == 'high' and len(c.dependencies) > 8][:10]
        
        prompt = f"""
Create code-informed modernization roadmap:

=== CURRENT STATE ===
Procedures: {self.statistics['total_procedures']}
Avg Dependencies: {self.statistics['avg_dependencies']:.2f}
Circular Dependencies: {self.statistics.get('circular_dependency_groups', 0)}
Complexity: High={self.statistics['complexity_distribution'].get('high', 0)}, Medium={self.statistics['complexity_distribution'].get('medium', 0)}, Low={self.statistics['complexity_distribution'].get('low', 0)}
Source Coverage: {self.statistics.get('source_coverage', 0):.1%}

=== QUICK WINS ===
{json.dumps([{'name': c.name, 'file_path': c.file_path, 'has_code': c.source_code is not None} for c in quick_wins], indent=2)}

=== HIGH RISK ===
{json.dumps([{'name': c.name, 'file_path': c.file_path, 'dependencies': len(c.dependencies)} for c in risky], indent=2)}

Provide detailed roadmap:

## Phase 1: Foundation (0-6 months)
### Activities (specific procedures)
### Success Metrics
### Risks & Mitigation

## Phase 2: Core (6-18 months)
### Activities (major refactoring)
### Technology Stack
### Success Metrics
### Risks & Mitigation

## Phase 3: Migration (18-36 months)
### Activities
### Success Metrics
### Risks & Mitigation

## Investment Analysis
### Effort
### Benefits
### ROI

Reference actual procedures and code patterns.
"""
        
        return self.call_llm(prompt, "You are a technology transformation consultant with TAL expertise.", temperature=0.4, max_tokens=4000)
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Recursively convert objects to JSON-serializable types"""
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, defaultdict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif hasattr(obj, '__dataclass_fields__'):
            return asdict(obj)
        else:
            return str(obj)
    
    def generate_documentation_report(self, output_file: str = "tal_architecture_report.json"):
        """Generate complete documentation with source code from file paths"""
        logger.info("Generating documentation with source code from graph file paths...")
        
        report = {
            'metadata': {
                'generated_at': str(Path.cwd()),
                'graph_stats': self.statistics,
                'source_files_found': len(self.source_cache),
                'source_coverage': f"{self.statistics.get('source_coverage', 0):.1%}"
            },
            'architecture_overview': self.generate_architecture_overview(),
            'components': [],
            'process_flows': [],
            'data_flow_analysis': self.generate_data_flow_analysis(),
            'microservices_candidates': self.identify_microservices_candidates(),
            'modernization_roadmap': self.generate_modernization_roadmap()
        }
        
        logger.info("Documenting critical components with source code...")
        for comp in self.components[:10]:
            comp_doc = self.generate_component_documentation(comp)
            report['components'].append({
                'component': asdict(comp),
                'documentation': comp_doc
            })
        
        logger.info("Documenting process flows with source code...")
        for flow in self.process_flows[:5]:
            flow_doc = self.generate_process_flow_documentation(flow)
            report['process_flows'].append({
                'flow': asdict(flow),
                'documentation': flow_doc
            })
        
        report = self._make_json_serializable(report)
        
        try:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Documentation saved to {output_file}")
        except TypeError as e:
            logger.error(f"JSON serialization error: {e}")
            for key, value in report.items():
                try:
                    json.dumps({key: value})
                except TypeError as field_error:
                    logger.error(f"Field '{key}' not serializable: {field_error}")
            raise
        
        return report


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate TAL documentation with source code from graph file paths'
    )
    parser.add_argument('graph_file', help='Path to knowledge graph (JSON/GraphML)')
    parser.add_argument('--output', default='tal_architecture_report.json', help='Output JSON file')
    
    args = parser.parse_args()
    
    # Initialize generator (source code loaded from file_path in graph)
    generator = TALDocumentationGenerator(args.graph_file)
    
    # Analyze graph
    generator.analyze_graph()
    
    # Generate documentation
    report = generator.generate_documentation_report(args.output)
    
    logger.info("="*70)
    logger.info("DOCUMENTATION COMPLETE!")
    logger.info("="*70)
    logger.info(f"Report: {args.output}")
    logger.info(f"Nodes: {len(generator.graph.nodes())}")
    logger.info(f"Edges: {len(generator.graph.edges())}")
    logger.info(f"Source files loaded: {len(generator.source_cache)}")
    logger.info(f"Components with code: {sum(1 for c in generator.components if c.source_code)}")
    logger.info(f"Source coverage: {generator.statistics.get('source_coverage', 0):.1%}")
    logger.info("="*70)


if __name__ == '__main__':
    main()
