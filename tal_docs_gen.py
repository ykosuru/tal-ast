#!/usr/bin/env python3
"""
TAL DDG (Documentation Generation) Tool
Walks through TAL knowledge graph and generates comprehensive architecture documentation
using LLM analysis.
"""

import json
import networkx as nx
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ArchitectureComponent:
    """Represents a component in the architecture"""
    name: str
    type: str  # procedure, module, service, etc.
    description: str
    dependencies: List[str]
    call_count: int
    complexity: str  # low, medium, high
    business_capability: str


@dataclass
class ProcessFlow:
    """Represents a process flow through the system"""
    name: str
    entry_point: str
    steps: List[Dict[str, Any]]
    data_flow: List[str]
    decision_points: List[str]


class TALDocumentationGenerator:
    """Main class for generating architecture documentation from TAL knowledge graph"""
    
    def __init__(self, graph_path: Optional[str] = None):
        """
        Initialize the documentation generator
        
        Args:
            graph_path: Path to the knowledge graph file (GraphML or JSON)
        """
        self.graph = None
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
        elif path.suffix == '.json':
            with open(graph_path, 'r') as f:
                data = json.load(f)
                self.graph = nx.node_link_graph(data)
        else:
            raise ValueError(f"Unsupported graph format: {path.suffix}")
        
        logger.info(f"Loaded graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
    
    def analyze_graph(self):
        """Analyze the knowledge graph and extract architectural information"""
        logger.info("Analyzing knowledge graph...")
        
        # Extract components
        self.components = self._extract_components()
        
        # Identify process flows
        self.process_flows = self._identify_process_flows()
        
        # Calculate statistics
        self.statistics = self._calculate_statistics()
        
        logger.info(f"Found {len(self.components)} components and {len(self.process_flows)} process flows")
    
    def _extract_components(self) -> List[ArchitectureComponent]:
        """Extract architectural components from the graph"""
        components = []
        
        for node, data in self.graph.nodes(data=True):
            # Get node attributes
            node_type = data.get('type', 'unknown')
            
            # Skip system intrinsics or utility nodes
            if node_type in ['intrinsic', 'system', 'utility']:
                continue
            
            # Get dependencies (outgoing edges)
            dependencies = list(self.graph.successors(node))
            
            # Calculate in-degree (how many times this is called)
            call_count = self.graph.in_degree(node)
            
            # Determine complexity based on out-degree and call patterns
            out_degree = self.graph.out_degree(node)
            if out_degree > 10 or call_count > 20:
                complexity = "high"
            elif out_degree > 5 or call_count > 10:
                complexity = "medium"
            else:
                complexity = "low"
            
            component = ArchitectureComponent(
                name=node,
                type=node_type,
                description=data.get('description', ''),
                dependencies=dependencies[:10],  # Limit for readability
                call_count=call_count,
                complexity=complexity,
                business_capability=data.get('business_capability', 'unknown')
            )
            components.append(component)
        
        # Sort by call count (most called first)
        components.sort(key=lambda x: x.call_count, reverse=True)
        return components
    
    def _identify_process_flows(self) -> List[ProcessFlow]:
        """Identify key process flows through the system"""
        flows = []
        
        # Find entry points (nodes with high out-degree and low in-degree)
        entry_points = [
            node for node, in_deg in self.graph.in_degree()
            if in_deg <= 2 and self.graph.out_degree(node) >= 3
        ]
        
        for entry_point in entry_points[:10]:  # Limit to top 10 entry points
            # Perform BFS to trace the flow
            steps = []
            visited = set()
            queue = [(entry_point, 0)]
            
            while queue and len(steps) < 20:  # Limit flow depth
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
                
                # Add successors to queue
                for successor in self.graph.successors(current):
                    if successor not in visited:
                        queue.append((successor, depth + 1))
            
            if len(steps) >= 3:  # Only include meaningful flows
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
            'complexity_distribution': defaultdict(int)
        }
        
        # Calculate complexity distribution
        for comp in self.components:
            stats['complexity_distribution'][comp.complexity] += 1
        
        # Identify strongly connected components (circular dependencies)
        strongly_connected = list(nx.strongly_connected_components(self.graph))
        stats['circular_dependency_groups'] = len([c for c in strongly_connected if len(c) > 1])
        
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
        # This will be replaced with actual OpenAI API call at work
        # For now, return a placeholder
        
        try:
            # PLACEHOLDER - Replace with actual API call
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
            return f"[LLM Response Placeholder]\nPrompt: {prompt[:100]}..."
            
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            return f"Error: {e}"
    
    def generate_architecture_overview(self) -> str:
        """Generate high-level architecture overview using LLM"""
        logger.info("Generating architecture overview...")
        
        # Prepare context for LLM
        context = {
            'statistics': self.statistics,
            'top_components': [asdict(c) for c in self.components[:20]],
            'component_types': defaultdict(int)
        }
        
        for comp in self.components:
            context['component_types'][comp.type] += 1
        
        prompt = f"""
Based on the following TAL codebase analysis, provide a comprehensive architecture overview:

STATISTICS:
- Total Procedures: {self.statistics['total_procedures']}
- Total Call Relationships: {self.statistics['total_calls']}
- Average Dependencies per Procedure: {self.statistics['avg_dependencies']:.2f}
- Maximum Dependencies: {self.statistics['max_dependencies']}
- Circular Dependency Groups: {self.statistics.get('circular_dependency_groups', 0)}

COMPONENT TYPES:
{json.dumps(dict(context['component_types']), indent=2)}

TOP 20 MOST CRITICAL COMPONENTS:
{json.dumps(context['top_components'][:10], indent=2)}

Please provide:
1. Overall architecture description (2-3 paragraphs)
2. Key architectural patterns observed
3. Main subsystems and their purposes
4. Integration points and dependencies
5. Architectural strengths and weaknesses
"""
        
        system_prompt = """You are a senior software architect analyzing a legacy TAL (Transaction Application Language) 
payment processing system. Provide clear, technical, and actionable insights."""
        
        return self.call_llm(prompt, system_prompt, temperature=0.3, max_tokens=2000)
    
    def generate_component_documentation(self, component: ArchitectureComponent) -> str:
        """Generate detailed documentation for a component"""
        
        prompt = f"""
Analyze this TAL component and provide detailed documentation:

COMPONENT: {component.name}
TYPE: {component.type}
COMPLEXITY: {component.complexity}
TIMES CALLED: {component.call_count}
DEPENDENCIES: {len(component.dependencies)}
BUSINESS CAPABILITY: {component.business_capability}

KEY DEPENDENCIES: {', '.join(component.dependencies[:5])}

Provide:
1. Purpose and functionality (2-3 sentences)
2. Role in the overall architecture
3. Key dependencies and why they're needed
4. Potential modernization recommendations
"""
        
        system_prompt = "You are documenting legacy TAL payment processing code. Be specific and technical."
        
        return self.call_llm(prompt, system_prompt, temperature=0.5, max_tokens=800)
    
    def generate_process_flow_documentation(self, flow: ProcessFlow) -> str:
        """Generate documentation for a process flow"""
        
        flow_steps = "\n".join([f"{i+1}. {step['node']} ({step['type']})" 
                                for i, step in enumerate(flow.steps[:10])])
        
        prompt = f"""
Document this process flow from the TAL payment system:

FLOW NAME: {flow.name}
ENTRY POINT: {flow.entry_point}
NUMBER OF STEPS: {len(flow.steps)}
DECISION POINTS: {len(flow.decision_points)}

FLOW SEQUENCE:
{flow_steps}

DECISION POINTS: {', '.join(flow.decision_points[:5])}

Provide:
1. Business process description (what does this flow accomplish?)
2. Step-by-step explanation of the main path
3. Key decision points and their impact
4. Data transformations in this flow
5. Error handling considerations
"""
        
        system_prompt = "You are a business analyst documenting payment processing workflows."
        
        return self.call_llm(prompt, system_prompt, temperature=0.5, max_tokens=1000)
    
    def identify_microservices_candidates(self) -> List[Dict[str, Any]]:
        """Use LLM to identify potential microservice boundaries"""
        logger.info("Identifying microservice candidates...")
        
        # Group components by business capability
        capabilities = defaultdict(list)
        for comp in self.components:
            capabilities[comp.business_capability].append(comp)
        
        candidates = []
        for capability, comps in capabilities.items():
            if len(comps) < 3:  # Skip small groups
                continue
            
            context = {
                'capability': capability,
                'component_count': len(comps),
                'total_calls': sum(c.call_count for c in comps),
                'components': [c.name for c in comps[:10]]
            }
            
            prompt = f"""
Evaluate this group of TAL components as a potential microservice:

BUSINESS CAPABILITY: {capability}
NUMBER OF COMPONENTS: {len(comps)}
TOTAL INCOMING CALLS: {context['total_calls']}

KEY COMPONENTS:
{json.dumps(context['components'], indent=2)}

Assess:
1. Is this a good microservice candidate? (Yes/No)
2. Cohesion level (High/Medium/Low)
3. Suggested service name
4. API boundaries needed
5. Data ownership scope
"""
            
            response = self.call_llm(prompt, "You are a microservices architect.", temperature=0.3)
            
            candidates.append({
                'capability': capability,
                'components': comps[:10],
                'analysis': response
            })
        
        return candidates
    
    def generate_data_flow_analysis(self) -> str:
        """Analyze data flows through the system"""
        logger.info("Analyzing data flows...")
        
        # Find longest paths (representing major data flows)
        try:
            # For directed graphs, we look at paths from entry points
            entry_points = [n for n in self.graph.nodes() if self.graph.in_degree(n) <= 1]
            
            sample_paths = []
            for entry in entry_points[:5]:
                # Get paths from this entry point
                paths = nx.single_source_shortest_path(self.graph, entry, cutoff=8)
                if paths:
                    longest = max(paths.values(), key=len)
                    if len(longest) > 3:
                        sample_paths.append(longest)
        except:
            sample_paths = []
        
        prompt = f"""
Analyze data flows in this TAL payment processing system:

STATISTICS:
- Total Procedures: {self.statistics['total_procedures']}
- Total Data Flow Paths: {self.statistics['total_calls']}

SAMPLE DATA FLOW PATHS:
{json.dumps([' -> '.join(path) for path in sample_paths[:5]], indent=2)}

TOP DATA TRANSFORMATION POINTS:
{json.dumps([c.name for c in self.components[:10]], indent=2)}

Provide:
1. Main data flow patterns
2. Data transformation stages
3. Critical data processing nodes
4. Data consistency mechanisms needed
5. Recommended data architecture for modernization
"""
        
        return self.call_llm(prompt, "You are a data architect analyzing payment systems.", temperature=0.4)
    
    def generate_modernization_roadmap(self) -> str:
        """Generate modernization recommendations"""
        logger.info("Generating modernization roadmap...")
        
        prompt = f"""
Create a modernization roadmap for this TAL legacy system:

CURRENT STATE:
- Total Procedures: {self.statistics['total_procedures']}
- Average Dependencies: {self.statistics['avg_dependencies']:.2f}
- Circular Dependencies: {self.statistics.get('circular_dependency_groups', 0)}
- High Complexity Components: {self.statistics['complexity_distribution'].get('high', 0)}
- Medium Complexity: {self.statistics['complexity_distribution'].get('medium', 0)}
- Low Complexity: {self.statistics['complexity_distribution'].get('low', 0)}

ARCHITECTURE PATTERNS:
- Component Types Identified: {len(set(c.type for c in self.components))}
- Process Flows Mapped: {len(self.process_flows)}

Provide a phased modernization roadmap:
1. **Phase 1** (Quick Wins - 0-6 months)
2. **Phase 2** (Core Modernization - 6-18 months)
3. **Phase 3** (Complete Migration - 18-36 months)

For each phase include:
- Key activities
- Technologies to adopt
- Risks and mitigation
- Success metrics
"""
        
        return self.call_llm(prompt, "You are a technology transformation consultant.", temperature=0.4, max_tokens=2500)
    
    def generate_documentation_report(self, output_file: str = "tal_architecture_report.json"):
        """Generate complete documentation report"""
        logger.info("Generating comprehensive documentation report...")
        
        report = {
            'metadata': {
                'generated_at': str(Path.cwd()),
                'graph_stats': self.statistics
            },
            'architecture_overview': self.generate_architecture_overview(),
            'components': [],
            'process_flows': [],
            'data_flow_analysis': self.generate_data_flow_analysis(),
            'microservices_candidates': self.identify_microservices_candidates(),
            'modernization_roadmap': self.generate_modernization_roadmap()
        }
        
        # Document top 10 critical components
        logger.info("Documenting critical components...")
        for comp in self.components[:10]:
            report['components'].append({
                'component': asdict(comp),
                'documentation': self.generate_component_documentation(comp)
            })
        
        # Document top 5 process flows
        logger.info("Documenting process flows...")
        for flow in self.process_flows[:5]:
            report['process_flows'].append({
                'flow': asdict(flow),
                'documentation': self.generate_process_flow_documentation(flow)
            })
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Documentation report saved to {output_file}")
        return report
    
    def create_word_document(self, report: Dict[str, Any], output_file: str = "tal_architecture_guide.docx"):
        """Create a Word document from the report"""
        logger.info("Creating Word document...")
        
        try:
            from docx import Document
            from docx.shared import Pt, RGBColor, Inches
            from docx.enum.text import WD_ALIGN_PARAGRAPH
            
            doc = Document()
            
            # Title
            title = doc.add_heading('TAL Architecture & Modernization Guide', 0)
            title.alignment = WD_ALIGN_PARAGRAPH.CENTER
            
            # Table of Contents placeholder
            doc.add_heading('Table of Contents', 1)
            doc.add_paragraph('[Auto-generated in Word]')
            doc.add_page_break()
            
            # 1. Architecture Overview
            doc.add_heading('1. Architecture Overview', 1)
            doc.add_paragraph(report['architecture_overview'])
            
            # Statistics
            doc.add_heading('1.1 System Statistics', 2)
            stats = report['metadata']['graph_stats']
            table = doc.add_table(rows=len(stats), cols=2)
            table.style = 'Light Grid Accent 1'
            for i, (key, value) in enumerate(stats.items()):
                table.rows[i].cells[0].text = str(key).replace('_', ' ').title()
                table.rows[i].cells[1].text = str(value)
            
            doc.add_page_break()
            
            # 2. Components
            doc.add_heading('2. Key Components', 1)
            for i, comp_data in enumerate(report['components'][:10]):
                comp = comp_data['component']
                doc.add_heading(f"2.{i+1} {comp['name']}", 2)
                doc.add_paragraph(f"Type: {comp['type']} | Complexity: {comp['complexity']} | Called: {comp['call_count']} times")
                doc.add_paragraph(comp_data['documentation'])
            
            doc.add_page_break()
            
            # 3. Process Flows
            doc.add_heading('3. Process Flows', 1)
            for i, flow_data in enumerate(report['process_flows'][:5]):
                flow = flow_data['flow']
                doc.add_heading(f"3.{i+1} {flow['name']}", 2)
                doc.add_paragraph(flow_data['documentation'])
            
            doc.add_page_break()
            
            # 4. Data Flow Analysis
            doc.add_heading('4. Data Flow Analysis', 1)
            doc.add_paragraph(report['data_flow_analysis'])
            
            doc.add_page_break()
            
            # 5. Microservices Candidates
            doc.add_heading('5. Microservices Candidates', 1)
            for candidate in report['microservices_candidates'][:5]:
                doc.add_heading(f"5.{candidate['capability']}", 2)
                doc.add_paragraph(f"Components: {len(candidate['components'])}")
                doc.add_paragraph(candidate['analysis'])
            
            doc.add_page_break()
            
            # 6. Modernization Roadmap
            doc.add_heading('6. Modernization Roadmap', 1)
            doc.add_paragraph(report['modernization_roadmap'])
            
            # Save document
            doc.save(output_file)
            logger.info(f"Word document saved to {output_file}")
            
        except ImportError:
            logger.warning("python-docx not installed. Skipping Word document generation.")
            logger.info("Install with: pip install python-docx")


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate TAL architecture documentation from knowledge graph')
    parser.add_argument('graph_file', help='Path to knowledge graph file (GraphML or JSON)')
    parser.add_argument('--output-json', default='tal_architecture_report.json', help='Output JSON report file')
    parser.add_argument('--output-docx', default='tal_architecture_guide.docx', help='Output Word document file')
    parser.add_argument('--skip-docx', action='store_true', help='Skip Word document generation')
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = TALDocumentationGenerator(args.graph_file)
    
    # Analyze graph
    generator.analyze_graph()
    
    # Generate documentation report
    report = generator.generate_documentation_report(args.output_json)
    
    # Create Word document
    if not args.skip_docx:
        generator.create_word_document(report, args.output_docx)
    
    logger.info("Documentation generation complete!")


if __name__ == '__main__':
    main()
