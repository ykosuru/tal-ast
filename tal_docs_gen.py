#!/usr/bin/env python3
"""
TAL DDG with Source Code Analysis
Includes actual TAL procedure code in LLM prompts for deep analysis
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
    source_code: Optional[str] = None  # NEW: actual code


@dataclass
class ProcessFlow:
    """Represents a process flow through the system"""
    name: str
    entry_point: str
    steps: List[Dict[str, Any]]
    data_flow: List[str]
    decision_points: List[str]


class TALDocumentationGenerator:
    """Generate documentation from TAL knowledge graph with source code analysis"""
    
    def __init__(self, graph_path: Optional[str] = None, source_dir: Optional[str] = None):
        """
        Initialize the documentation generator
        
        Args:
            graph_path: Path to knowledge graph (GraphML or JSON)
            source_dir: Path to directory containing TAL source files
        """
        self.graph = None
        self.graph_data = None
        self.source_dir = Path(source_dir) if source_dir else None
        self.source_cache = {}  # Cache loaded source files
        self.components = []
        self.process_flows = []
        self.statistics = {}
        
        if graph_path:
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
        
        if self.source_dir:
            logger.info(f"Source directory: {self.source_dir}")
            self._load_source_files()
    
    def _load_source_files(self):
        """Load TAL source files from source directory"""
        if not self.source_dir or not self.source_dir.exists():
            logger.warning(f"Source directory not found: {self.source_dir}")
            return
        
        logger.info("Loading TAL source files...")
        
        # Find all .tal files
        tal_files = list(self.source_dir.rglob("*.tal")) + list(self.source_dir.rglob("*.TAL"))
        logger.info(f"Found {len(tal_files)} TAL source files")
        
        # Load each file
        for tal_file in tal_files:
            try:
                with open(tal_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    # Store by filename (without extension)
                    name = tal_file.stem.upper()
                    self.source_cache[name] = content
                    
            except Exception as e:
                logger.warning(f"Could not read {tal_file}: {e}")
        
        logger.info(f"Loaded {len(self.source_cache)} source files into cache")
    
    def _get_source_code(self, procedure_name: str, max_lines: int = 200) -> Optional[str]:
        """
        Get source code for a procedure
        
        Args:
            procedure_name: Name of the procedure
            max_lines: Maximum lines to return (for token efficiency)
        
        Returns:
            Source code or None if not found
        """
        # Try exact match
        if procedure_name in self.source_cache:
            code = self.source_cache[procedure_name]
            lines = code.split('\n')
            if len(lines) > max_lines:
                truncated = '\n'.join(lines[:max_lines])
                return f"{truncated}\n\n... [truncated - {len(lines)} total lines]"
            return code
        
        # Try case-insensitive match
        for name, code in self.source_cache.items():
            if name.upper() == procedure_name.upper():
                lines = code.split('\n')
                if len(lines) > max_lines:
                    truncated = '\n'.join(lines[:max_lines])
                    return f"{truncated}\n\n... [truncated - {len(lines)} total lines]"
                return code
        
        return None
    
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
        logger.info(f"Found source code for {with_code}/{len(self.components)} components")
    
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
            
            # Load source code
            source_code = self._get_source_code(node)
            
            component = ArchitectureComponent(
                name=node,
                type=node_type,
                description=data.get('description', ''),
                dependencies=dependencies[:10],
                call_count=call_count,
                complexity=complexity,
                business_capability=data.get('business_capability', 'unknown'),
                source_code=source_code
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
        """
        Call LLM using OpenAI-compatible API
        
        Args:
            prompt: The user prompt
            system_prompt: System prompt for context
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        
        Returns:
            LLM response text
        """
        try:
            # PLACEHOLDER - Replace with actual API call at work
            # Example implementation:
            """
            import openai
            
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content
            """
            
            logger.warning("LLM call placeholder - implement actual API call")
            return f"[LLM Response Placeholder]\nPrompt length: {len(prompt)} chars\nFirst 200 chars: {prompt[:200]}..."
            
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            return f"Error: {e}"
    
    def generate_architecture_overview(self) -> str:
        """Generate high-level architecture overview using LLM with full graph context"""
        logger.info("Generating architecture overview with graph context...")
        
        graph_context = self._extract_graph_context(nodes_limit=30, edges_limit=50)
        
        component_types = defaultdict(int)
        for comp in self.components:
            component_types[comp.type] += 1
        
        capabilities = defaultdict(list)
        for comp in self.components[:30]:
            capabilities[comp.business_capability].append(comp.name)
        
        # Include sample source code snippets
        code_samples = []
        for comp in self.components[:3]:
            if comp.source_code:
                # Get first 30 lines as sample
                lines = comp.source_code.split('\n')[:30]
                code_samples.append({
                    'name': comp.name,
                    'snippet': '\n'.join(lines)
                })
        
        prompt = f"""
Analyze this TAL payment processing system and provide a comprehensive architecture overview.

=== GRAPH STRUCTURE ===
Total Nodes: {graph_context['total_nodes']}
Total Edges: {graph_context['total_edges']}

Sample Nodes (with attributes):
{json.dumps(graph_context['nodes'][:15], indent=2)}

Sample Edges (call relationships):
{json.dumps(graph_context['edges'][:20], indent=2)}

=== STATISTICS ===
- Total Procedures: {self.statistics['total_procedures']}
- Total Call Relationships: {self.statistics['total_calls']}
- Average Dependencies: {self.statistics['avg_dependencies']:.2f}
- Maximum Dependencies: {self.statistics['max_dependencies']}
- Most Called: {self.statistics.get('most_called', 'N/A')}
- Circular Dependencies: {self.statistics.get('circular_dependency_groups', 0)}
- Source Code Coverage: {self.statistics.get('source_coverage', 0):.1%}

=== COMPONENT TYPES ===
{json.dumps(dict(component_types), indent=2)}

=== TOP CRITICAL COMPONENTS ===
{json.dumps([asdict(c) for c in self.components[:10]], indent=2, default=str)}

=== BUSINESS CAPABILITIES ===
{json.dumps({k: v[:5] for k, v in capabilities.items()}, indent=2)}

=== SAMPLE TAL CODE (for context) ===
{json.dumps(code_samples, indent=2)}

Based on this TAL codebase structure and actual source code, provide:

1. **Architecture Overview** (3-4 paragraphs)
   - System purpose and design
   - Architectural patterns observed in code
   - Component organization

2. **Key Subsystems**
   - Identify subsystems from capabilities
   - Purpose and implementation approach
   - Key procedures

3. **Integration Points**
   - How subsystems communicate
   - Data flow patterns seen in code
   - Critical integration procedures

4. **Code Quality & Complexity**
   - Code structure observations
   - Complexity hotspots
   - Technical debt indicators

5. **Architectural Assessment**
   - Strengths
   - Weaknesses
   - Modernization opportunities

Format as clear markdown with headers.
"""
        
        system_prompt = """You are a senior software architect with 20+ years analyzing legacy TAL (Transaction Application Language) payment systems. You have access to both the call graph AND actual source code. Provide specific, code-informed insights referencing actual procedures and patterns you observe."""
        
        return self.call_llm(prompt, system_prompt, temperature=0.3, max_tokens=3500)
    
    def generate_component_documentation(self, component: ArchitectureComponent) -> str:
        """Generate detailed documentation for a component using actual source code"""
        
        callers = [n for n in self.graph.nodes() if component.name in list(self.graph.successors(n))]
        
        dependency_details = []
        for dep in component.dependencies[:8]:
            if dep in self.graph.nodes:
                dep_data = self.graph.nodes[dep]
                dependency_details.append({
                    'name': dep,
                    'type': dep_data.get('type', 'unknown'),
                    'capability': dep_data.get('business_capability', 'unknown'),
                    'description': dep_data.get('description', '')
                })
        
        # Include dependency source code for deeper analysis
        dependency_code = []
        for dep in component.dependencies[:3]:
            dep_code = self._get_source_code(dep, max_lines=50)
            if dep_code:
                dependency_code.append({
                    'name': dep,
                    'code': dep_code
                })
        
        prompt = f"""
Analyze this TAL component in detail with access to actual source code:

=== COMPONENT DETAILS ===
Name: {component.name}
Type: {component.type}
Business Capability: {component.business_capability}
Complexity: {component.complexity}
Times Called: {component.call_count}
Dependencies: {len(component.dependencies)}

=== ACTUAL SOURCE CODE ===
```tal
{component.source_code if component.source_code else '[Source code not available]'}
```

=== CALLERS (who calls this) ===
{json.dumps(callers[:10], indent=2)}

=== DEPENDENCIES (what this calls) ===
{json.dumps(dependency_details, indent=2)}

=== SAMPLE DEPENDENCY CODE (for context) ===
{json.dumps(dependency_code, indent=2)}

=== NODE ATTRIBUTES ===
{json.dumps(dict(self.graph.nodes[component.name]), indent=2)}

Provide comprehensive, code-informed documentation:

1. **Purpose and Functionality**
   - What does the code actually do?
   - Key algorithms and logic
   - Business rules implemented

2. **Code Structure Analysis**
   - Variables and data structures used
   - Control flow patterns
   - Error handling approach

3. **Role in Architecture**
   - Position in call hierarchy
   - Integration patterns
   - Criticality (called {component.call_count} times)

4. **Dependencies Analysis**
   - Why each dependency is called (based on code)
   - Data passed to dependencies
   - Integration patterns

5. **Callers Analysis**
   - Who relies on this
   - How it's used
   - Impact if modified/fails

6. **Code Quality Assessment**
   - Complexity indicators
   - Maintainability concerns
   - Technical debt

7. **Modernization Recommendations**
   - Refactoring opportunities
   - API design suggestions
   - Migration strategy
   - Microservice extraction approach

Be specific, reference actual code patterns, variable names, and logic.
"""
        
        system_prompt = "You are documenting legacy TAL payment processing code for modernization. Analyze actual source code to provide specific, actionable insights."
        
        return self.call_llm(prompt, system_prompt, temperature=0.5, max_tokens=2000)
    
    def generate_process_flow_documentation(self, flow: ProcessFlow) -> str:
        """Generate documentation for a process flow with actual code"""
        
        detailed_steps = []
        for step in flow.steps[:12]:
            node_name = step['node']
            if node_name in self.graph.nodes:
                node_data = dict(self.graph.nodes[node_name])
                # Get code snippet for this step
                code = self._get_source_code(node_name, max_lines=30)
                detailed_steps.append({
                    'node': node_name,
                    'type': step['type'],
                    'depth': step['depth'],
                    'attributes': node_data,
                    'out_degree': self.graph.out_degree(node_name),
                    'in_degree': self.graph.in_degree(node_name),
                    'code_snippet': code[:500] if code else None  # First 500 chars
                })
        
        prompt = f"""
Document this process flow with access to actual TAL source code:

=== FLOW OVERVIEW ===
Name: {flow.name}
Entry Point: {flow.entry_point}
Total Steps: {len(flow.steps)}
Decision Points: {len(flow.decision_points)}

=== DETAILED FLOW WITH CODE ===
{json.dumps(detailed_steps, indent=2)}

=== DECISION POINTS ===
{json.dumps(flow.decision_points[:5], indent=2)}

=== DATA FLOW PATH ===
{' -> '.join(flow.data_flow[:15])}

Provide comprehensive, code-informed process documentation:

1. **Business Process Description**
   - What business operation based on code?
   - When triggered?
   - Expected outcomes

2. **Step-by-Step Flow**
   - Describe each step with code insights
   - Data transformations (based on actual code)
   - State changes

3. **Decision Points Analysis**
   - Branching conditions (from code)
   - Alternative paths
   - Error handling logic

4. **Data Flow**
   - Input data (variables, structures)
   - Transformations at each step
   - Output data produced

5. **Integration Points**
   - External system calls
   - Database operations
   - Message passing

6. **Error Handling**
   - Error detection in code
   - Rollback mechanisms
   - Recovery procedures

7. **Performance & Optimization**
   - Potential bottlenecks from code
   - Optimization opportunities
   - Scalability concerns

Reference actual code patterns, variables, and logic flow.
"""
        
        return self.call_llm(prompt, "You are a business analyst with deep TAL expertise documenting payment workflows.", temperature=0.5, max_tokens=2500)
    
    def identify_microservices_candidates(self) -> Dict[str, Any]:
        """Identify microservice candidates using graph and code analysis"""
        logger.info("Identifying microservice candidates with code analysis...")
        
        capabilities = defaultdict(list)
        for comp in self.components:
            capabilities[comp.business_capability].append(comp)
        
        candidates = []
        
        for capability, comps in capabilities.items():
            if len(comps) < 3:
                continue
            
            # Calculate metrics
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
Evaluate this group as a microservice candidate with actual code analysis:

=== SERVICE METRICS ===
Capability: {capability}
Components: {len(comps)}
Incoming Calls: {context['total_calls']}
Cohesion: {cohesion_ratio:.1%}
Internal: {internal_calls} | External: {external_calls}
Code Available: {context['code_available']}/{len(comps)}

=== COMPONENTS ===
{json.dumps(context['components'], indent=2, default=str)}

=== EXTERNAL DEPENDENCIES ===
{json.dumps(context['external_dependencies'], indent=2)}

=== SAMPLE TAL CODE ===
{json.dumps(code_samples, indent=2)}

Assess based on code and structure:

1. **Recommendation** (Yes/No/Maybe)
   - Clear decision with code-based reasoning

2. **Cohesion Analysis**
   - Code patterns showing cohesion
   - Shared data structures
   - Related business logic

3. **Coupling Analysis**
   - Dependencies from code
   - Data sharing patterns
   - Transaction boundaries

4. **Service Design**
   - Service name
   - API operations (from code)
   - Data ownership
   - Events to publish

5. **Migration Strategy**
   - Extraction difficulty
   - Code refactoring needed
   - Phased approach

6. **Benefits & Risks**
   - Specific benefits
   - Code-based challenges
   - Mitigation

Reference actual code patterns and structures.
"""
            
            response = self.call_llm(prompt, "You are a microservices architect analyzing legacy code for extraction.", temperature=0.3, max_tokens=2000)
            
            candidates.append({
                'capability': capability,
                'components': [asdict(c) for c in comps[:10]],
                'metrics': context,
                'analysis': response
            })
        
        return {'candidates': candidates}
    
    def generate_data_flow_analysis(self) -> str:
        """Analyze data flows using graph and code"""
        logger.info("Analyzing data flows with code context...")
        
        entry_points = [n for n in self.graph.nodes() if self.graph.in_degree(n) <= 1]
        
        sample_paths = []
        for entry in entry_points[:5]:
            paths = nx.single_source_shortest_path(self.graph, entry, cutoff=8)
            if paths:
                longest = max(paths.values(), key=len)
                if len(longest) > 3:
                    path_details = []
                    for node in longest:
                        path_details.append({
                            'node': node,
                            'type': self.graph.nodes[node].get('type', 'unknown'),
                            'capability': self.graph.nodes[node].get('business_capability', 'unknown')
                        })
                    sample_paths.append(path_details)
        
        # Data hubs
        hubs = []
        for node, data in self.graph.nodes(data=True):
            in_deg = self.graph.in_degree(node)
            out_deg = self.graph.out_degree(node)
            if in_deg >= 3 and out_deg >= 3:
                code = self._get_source_code(node, max_lines=40)
                hubs.append({
                    'node': node,
                    'type': data.get('type', 'unknown'),
                    'capability': data.get('business_capability', 'unknown'),
                    'in_degree': in_deg,
                    'out_degree': out_deg,
                    'has_code': code is not None,
                    'code_snippet': code[:300] if code else None
                })
        
        prompt = f"""
Analyze data flows with actual code insights:

=== SYSTEM OVERVIEW ===
Total Procedures: {self.statistics['total_procedures']}
Total Paths: {self.statistics['total_calls']}
Avg Dependencies: {self.statistics['avg_dependencies']:.2f}

=== SAMPLE DATA FLOW PATHS ===
{json.dumps(sample_paths, indent=2)}

=== DATA TRANSFORMATION HUBS ===
{json.dumps(sorted(hubs, key=lambda x: x['in_degree'] + x['out_degree'], reverse=True)[:8], indent=2)}

=== TOP DATA CONSUMERS ===
{json.dumps([{'name': c.name, 'call_count': c.call_count, 'has_code': c.source_code is not None} for c in self.components[:10]], indent=2)}

Provide code-informed data flow analysis:

1. **Data Flow Patterns**
   - Patterns from code
   - Hub-spoke, pipeline, etc.
   - Specific procedures

2. **Data Transformations**
   - Transformation stages
   - Data structures used
   - Validation points

3. **Critical Hubs**
   - Why they're hubs (from code)
   - Data operations
   - Failure impact

4. **Data Consistency**
   - Consistency mechanisms in code
   - Transaction boundaries
   - Compensation patterns

5. **Data Issues**
   - Coupling from code
   - Shared state
   - Quality concerns

6. **Modernization**
   - Event-driven opportunities
   - Data ownership clarity
   - API design
   - Database decomposition
   - CQRS patterns

Reference actual code patterns and data structures.
"""
        
        return self.call_llm(prompt, "You are a data architect analyzing payment system code.", temperature=0.4, max_tokens=2800)
    
    def generate_modernization_roadmap(self) -> str:
        """Generate modernization roadmap with code insights"""
        logger.info("Generating modernization roadmap with code analysis...")
        
        quick_wins = [c for c in self.components if c.complexity == 'low' and len(c.dependencies) <= 3][:10]
        risky = [c for c in self.components if c.complexity == 'high' and len(c.dependencies) > 8][:10]
        
        prompt = f"""
Create code-informed modernization roadmap:

=== CURRENT STATE ===
Total Procedures: {self.statistics['total_procedures']}
Avg Dependencies: {self.statistics['avg_dependencies']:.2f}
Circular Dependencies: {self.statistics.get('circular_dependency_groups', 0)}
Complexity: High={self.statistics['complexity_distribution'].get('high', 0)}, Medium={self.statistics['complexity_distribution'].get('medium', 0)}, Low={self.statistics['complexity_distribution'].get('low', 0)}
Source Coverage: {self.statistics.get('source_coverage', 0):.1%}

=== QUICK WINS (low complexity, low coupling) ===
{json.dumps([{'name': c.name, 'has_code': c.source_code is not None} for c in quick_wins], indent=2)}

=== HIGH RISK (high complexity, high coupling) ===
{json.dumps([{'name': c.name, 'dependencies': len(c.dependencies), 'has_code': c.source_code is not None} for c in risky], indent=2)}

=== PATTERNS ===
Component Types: {len(set(c.type for c in self.components))}
Capabilities: {len(set(c.business_capability for c in self.components))}
Process Flows: {len(self.process_flows)}

Provide detailed, code-informed roadmap:

## Phase 1: Foundation (0-6 months)

### Activities
- Specific procedures to modernize
- Code refactoring priorities
- Technology decisions

### Success Metrics
### Risks & Mitigation

## Phase 2: Core (6-18 months)

### Activities
- Major refactoring
- Service extraction
- Data changes
- Specific components

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
### ROI Timeline

Be specific, reference actual procedures and code patterns.
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
        """Generate complete documentation report with source code analysis"""
        logger.info("Generating comprehensive documentation with source code analysis...")
        
        report = {
            'metadata': {
                'generated_at': str(Path.cwd()),
                'graph_stats': self.statistics,
                'source_directory': str(self.source_dir) if self.source_dir else None,
                'source_files_loaded': len(self.source_cache)
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
            logger.info(f"Documentation report saved to {output_file}")
        except TypeError as e:
            logger.error(f"JSON serialization error: {e}")
            for key, value in report.items():
                try:
                    json.dumps({key: value})
                except TypeError as field_error:
                    logger.error(f"Field '{key}' is not serializable: {field_error}")
            raise
        
        return report


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate TAL documentation with source code analysis'
    )
    parser.add_argument('graph_file', help='Path to knowledge graph (GraphML/JSON)')
    parser.add_argument('--source-dir', help='Directory containing TAL source files', default=None)
    parser.add_argument('--output', default='tal_architecture_report.json', help='Output JSON file')
    
    args = parser.parse_args()
    
    # Initialize generator with source directory
    generator = TALDocumentationGenerator(args.graph_file, args.source_dir)
    
    # Analyze graph
    generator.analyze_graph()
    
    # Generate documentation
    report = generator.generate_documentation_report(args.output)
    
    logger.info("="*70)
    logger.info("Documentation generation complete!")
    logger.info("="*70)
    logger.info(f"Report: {args.output}")
    logger.info(f"Nodes: {len(generator.graph.nodes())}")
    logger.info(f"Edges: {len(generator.graph.edges())}")
    logger.info(f"Source files: {len(generator.source_cache)}")
    logger.info(f"Components with code: {sum(1 for c in generator.components if c.source_code)}")
    logger.info(f"Source coverage: {generator.statistics.get('source_coverage', 0):.1%}")


if __name__ == '__main__':
    main()
