#!/usr/bin/env python3
"""
TAL DDG - Functionality-Focused Documentation Generator
Generates documentation for specific business functionalities (drawdown, nostro, wire transfers, etc.)
Uses domain terminology and includes actual source code
"""

import json
import networkx as nx
from typing import Dict, List, Any, Optional, Set
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict
import logging
import re
import asyncio

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Domain-specific keywords for financial systems
DOMAIN_KEYWORDS = {
    'drawdown': [
        'drawdown', 'loan', 'advance', 'disbursement', 'facility', 'tranche',
        'commitment', 'utilization', 'borrower', 'principal', 'interest'
    ],
    'nostro': [
        'nostro', 'vostro', 'correspondent', 'account', 'reconciliation',
        'balance', 'position', 'settlement', 'clearing'
    ],
    'wire_transfer': [
        'wire', 'swift', 'mt103', 'mt202', 'fedwire', 'chips',
        'beneficiary', 'ordering', 'remittance', 'payment'
    ],
    'ach': [
        'ach', 'batch', 'nacha', 'sec_code', 'addenda', 'prenote',
        'return', 'reversal', 'iat'
    ],
    'compliance': [
        'ofac', 'aml', 'kyc', 'cip', 'sanctions', 'screening',
        'suspicious_activity', 'sar', 'ctr', 'fraud'
    ],
    'ledger': [
        'ledger', 'general_ledger', 'gl', 'posting', 'journal',
        'debit', 'credit', 'balance', 'account', 'chart_of_accounts'
    ],
    'settlement': [
        'settlement', 'clearing', 'netting', 'finality', 'dvp',
        'rtgs', 'value_date', 'payment_date'
    ],
    'forex': [
        'forex', 'fx', 'exchange_rate', 'spot', 'forward', 'swap',
        'currency', 'conversion', 'hedging'
    ]
}


@dataclass
class ArchitectureComponent:
    """Component in the architecture"""
    name: str
    type: str
    description: str
    dependencies: List[str]
    call_count: int
    complexity: str
    business_capability: str
    source_code: Optional[str] = None
    file_path: Optional[str] = None
    relevance_score: float = 0.0  # How relevant to target functionality


@dataclass
class ProcessFlow:
    """Process flow through the system"""
    name: str
    entry_point: str
    steps: List[Dict[str, Any]]
    data_flow: List[str]
    decision_points: List[str]


class TALFunctionalityDocGenerator:
    """Generate focused documentation for specific business functionality"""
    
    def __init__(self, graph_path: Optional[str] = None, functionality: Optional[str] = None):
        """
        Initialize the documentation generator
        
        Args:
            graph_path: Path to knowledge graph (JSON)
            functionality: Target functionality (e.g., 'drawdown', 'nostro', 'wire_transfer')
        """
        self.graph = None
        self.graph_data = None
        self.source_cache = {}
        self.components = []
        self.process_flows = []
        self.statistics = {}
        self.base_path = None
        self.functionality = functionality
        self.functionality_keywords = []
        
        if functionality:
            # Safe conversion to lowercase with fallback
            func_key = functionality.lower() if functionality else ''
            self.functionality_keywords = DOMAIN_KEYWORDS.get(func_key, [])
            if self.functionality_keywords:
                logger.info(f"Target functionality: {functionality}")
                logger.info(f"Domain keywords: {', '.join(self.functionality_keywords[:10])}")
            else:
                logger.warning(f"No domain keywords found for functionality: {functionality}")
        
        if graph_path:
            self.base_path = Path(graph_path).parent
            self.load_graph(graph_path)
    
    def load_graph(self, graph_path: str):
        """Load the TAL knowledge graph"""
        logger.info(f"Loading knowledge graph from {graph_path}")
        
        path = Path(graph_path)
        with open(path, 'r') as f:
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
            
            self.graph = nx.node_link_graph(clean_data)
        
        logger.info(f"Loaded graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        self._load_source_files_from_graph()
    
    def _load_source_files_from_graph(self):
        """Load TAL source files using file_path from graph nodes"""
        logger.info("Loading TAL source files from graph file_path attributes...")
        
        file_paths = set()
        for node, data in self.graph.nodes(data=True):
            file_path = data.get('file_path')
            if file_path:
                file_paths.add(file_path)
        
        logger.info(f"Found {len(file_paths)} unique file paths in graph")
        
        loaded_count = 0
        for file_path_str in file_paths:
            file_path = Path(file_path_str)
            
            if file_path.exists():
                actual_path = file_path
            elif self.base_path and (self.base_path / file_path).exists():
                actual_path = self.base_path / file_path
            elif self.base_path and (self.base_path / file_path.name).exists():
                actual_path = self.base_path / file_path.name
            else:
                continue
            
            try:
                with open(actual_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    self.source_cache[file_path_str] = content
                    loaded_count += 1
            except Exception as e:
                logger.debug(f"Could not read {actual_path}: {e}")
        
        logger.info(f"Successfully loaded {loaded_count}/{len(file_paths)} source files")
    
    def _calculate_relevance_score(self, node_data: Dict, source_code: Optional[str]) -> float:
        """
        Calculate how relevant a procedure is to the target functionality
        
        Args:
            node_data: Node data from graph
            source_code: Source code if available
        
        Returns:
            Relevance score (0.0 to 1.0)
        """
        if not self.functionality_keywords:
            return 1.0  # No filtering if no functionality specified
        
        score = 0.0
        text_to_search = []
        
        # Check procedure name (with None check)
        name = node_data.get('name')
        if name:
            text_to_search.append(name.lower())
        
        # Check description (with None check)
        desc = node_data.get('description')
        if desc:
            text_to_search.append(desc.lower())
        
        # Check file path (with None check)
        file_path = node_data.get('file_path')
        if file_path:
            text_to_search.append(file_path.lower())
        
        # Check source code (with None check)
        if source_code:
            text_to_search.append(source_code.lower())
        
        # Count keyword matches
        combined_text = ' '.join(text_to_search)
        matches = 0
        for keyword in self.functionality_keywords:
            if keyword in combined_text:
                matches += 1
        
        # Score based on matches
        if matches > 0:
            score = min(1.0, matches / 5.0)  # Cap at 1.0
        
        return score
    
    def _get_source_code_for_node(self, node_data: Dict, max_lines: int = 200) -> Optional[str]:
        """Get source code for a node using its file_path attribute"""
        file_path = node_data.get('file_path')
        if not file_path:
            return None
        
        content = self.source_cache.get(file_path)
        if not content:
            return None
        
        if node_data.get('type') == 'procedure':
            proc_name = node_data.get('name')
            if proc_name:
                pattern = rf'PROC\s+{re.escape(proc_name)}\s*[(\[].*?(?:END|ENDPROC)'
                match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
                if match:
                    proc_code = match.group(0)
                    lines = proc_code.split('\n')
                    if len(lines) > max_lines:
                        truncated = '\n'.join(lines[:max_lines])
                        return f"{truncated}\n\n... [truncated - {len(lines)} total lines]"
                    return proc_code
        
        lines = content.split('\n')
        if len(lines) > max_lines:
            truncated = '\n'.join(lines[:max_lines])
            return f"{truncated}\n\n... [truncated - {len(lines)} total lines]"
        return content
    
    def analyze_graph(self, min_relevance: float = 0.0):
        """
        Analyze the knowledge graph and extract components relevant to functionality
        
        Args:
            min_relevance: Minimum relevance score to include (0.0 to 1.0)
        """
        logger.info(f"Analyzing graph for {self.functionality} functionality...")
        
        self.components = self._extract_components(min_relevance)
        self.process_flows = self._identify_process_flows()
        self.statistics = self._calculate_statistics()
        
        logger.info(f"Found {len(self.components)} relevant components")
        
        with_code = sum(1 for c in self.components if c.source_code)
        logger.info(f"Source code available for {with_code}/{len(self.components)} components")
        
        if self.functionality:
            avg_relevance = sum(c.relevance_score for c in self.components) / max(len(self.components), 1)
            logger.info(f"Average relevance score: {avg_relevance:.2f}")
    
    def _extract_components(self, min_relevance: float = 0.0) -> List[ArchitectureComponent]:
        """Extract architectural components relevant to functionality"""
        components = []
        
        for node, data in self.graph.nodes(data=True):
            node_type = data.get('type', 'unknown')
            
            if node_type in ['intrinsic', 'system']:
                continue
            
            # Load source code
            source_code = self._get_source_code_for_node(data)
            
            # Calculate relevance
            relevance_score = self._calculate_relevance_score(data, source_code)
            
            # Filter by relevance
            if relevance_score < min_relevance:
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
            
            component = ArchitectureComponent(
                name=node,
                type=node_type,
                description=data.get('description', ''),
                dependencies=dependencies[:10],
                call_count=call_count,
                complexity=complexity,
                business_capability=data.get('business_capability', 'unknown'),
                source_code=source_code,
                file_path=data.get('file_path'),
                relevance_score=relevance_score
            )
            components.append(component)
        
        # Sort by relevance first, then call count
        components.sort(key=lambda x: (x.relevance_score, x.call_count), reverse=True)
        return components
    
    def _identify_process_flows(self) -> List[ProcessFlow]:
        """Identify key process flows"""
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
        """Calculate statistics"""
        stats = {
            'total_procedures': len(self.components),
            'total_calls': self.graph.number_of_edges(),
            'avg_dependencies': sum(d for n, d in self.graph.out_degree()) / max(self.graph.number_of_nodes(), 1),
            'max_dependencies': max(d for n, d in self.graph.out_degree()) if self.graph.number_of_nodes() > 0 else 0,
            'procedures_with_source': sum(1 for c in self.components if c.source_code),
            'source_coverage': sum(1 for c in self.components if c.source_code) / max(len(self.components), 1)
        }
        
        complexity_counts = defaultdict(int)
        for comp in self.components:
            complexity_counts[comp.complexity] += 1
        stats['complexity_distribution'] = dict(complexity_counts)
        
        if self.functionality:
            stats['avg_relevance'] = sum(c.relevance_score for c in self.components) / max(len(self.components), 1)
        
        return stats
    
    async def llm_wrapper(self, system_prompt: str, user_prompt: str) -> str:
        """
        Async LLM wrapper - replace with actual API call
        
        Args:
            system_prompt: System prompt for context
            user_prompt: User prompt with query
        
        Returns:
            LLM response text (max 1500 words)
        """
        try:
            # PLACEHOLDER - Replace with actual async API call
            # Example with OpenAI:
            """
            import openai
            
            response = await openai.ChatCompletion.acreate(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=2000  # ~1500 words
            )
            
            return response.choices[0].message.content
            """
            
            logger.warning("LLM wrapper placeholder - implement actual async API call")
            return f"[LLM Response Placeholder]\nPrompt length: {len(user_prompt)} chars"
            
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            return f"Error: {e}"
    
    def generate_functionality_overview(self) -> str:
        """Generate architecture documentation with FULL procedure source code"""
        logger.info(f"Generating {self.functionality} functionality overview...")
        
        # Get top relevant components with FULL code (not snippets)
        top_components = [c for c in self.components[:15] if c.relevance_score > 0.3 and c.source_code]
        
        # Build procedures context with FULL source code (like tal_to_java_translator)
        procedures_context = []
        for comp in top_components:
            procedures_context.append({
                'name': comp.name,
                'file_path': comp.file_path,
                'relevance': comp.relevance_score,
                'complexity': comp.complexity,
                'called_by': comp.call_count,
                'code': comp.source_code  # FULL CODE, not snippet
            })
        
        functionality_context = f"""
This is a {self.functionality.upper()} system focused on {', '.join(self.functionality_keywords[:8])}.

Common industry terminology for {self.functionality}:
- Use proper financial/payment terms
- Reference standard message formats (SWIFT, ISO 20022, etc.)
- Follow industry best practices
"""
        
        # TAL Language Primer for LLM
        tal_primer = """
=== TAL (Transaction Application Language) PRIMER ===

TAL is HP's systems programming language for NonStop servers. Key constructs:

**Procedure Declaration:**
```tal
PROC procedure_name(param1, param2);
  ! Procedure body
END;
```

**Data Types:**
- INT, INT(32) - Integer types
- STRING - String type  
- REAL, FIXED - Floating point
- Structures defined with STRUCT

**Control Flow:**
- IF...THEN...ELSE
- WHILE...DO
- FOR...TO...DO
- CASE...OF

**Procedure Calls:**
```tal
CALL procedure_name(args);
result := procedure_name(args);
```

**Common Patterns:**
- Error handling via return codes (0 = success, non-zero = error)
- Database operations via SQL or file system calls
- Interprocess communication
- Transaction processing

Focus on understanding business logic and data flow, not low-level TAL syntax details.
"""
        
        user_prompt = f"""
Analyze this TAL {self.functionality.upper()} system implementation with COMPLETE source code.

{tal_primer}

=== FUNCTIONALITY CONTEXT ===
{functionality_context}

=== SYSTEM SCOPE ===
Total Procedures Analyzed: {len(procedures_context)}
Total Procedures in System: {len(self.components)}
Procedures with Source Code: {self.statistics['procedures_with_source']}
Average Relevance to {self.functionality}: {self.statistics.get('avg_relevance', 0):.1%}

=== COMPLETE PROCEDURE SOURCE CODE ===

"""
        
        # Add ALL procedures with FULL source code
        for i, proc in enumerate(procedures_context, 1):
            user_prompt += f"""
## Procedure {i}: {proc['name']}

**File:** `{proc['file_path']}`  
**Relevance:** {proc['relevance']:.1%}  
**Complexity:** {proc['complexity']}  
**Called by:** {proc['called_by']} procedures

**FULL SOURCE CODE:**
```tal
{proc['code']}
```

---

"""
        
        user_prompt += f"""

=== ANALYSIS REQUIREMENTS ===

Based on the COMPLETE source code above, produce a comprehensive architecture and design document in MARKDOWN format.

**LIMIT: 1500 WORDS**

# {self.functionality.title()} System - Architecture & Design

## 1. System Overview (200 words)
   - Purpose and business scope
   - Key capabilities provided
   - Integration points with other systems
   - Industry standards compliance (SWIFT, ISO 20022, etc.)

## 2. Architecture Diagram
   Create a Mermaid diagram showing:
   - Entry point procedures
   - Core processing procedures  
   - Data flow between procedures
   - External system interactions
   
   Use Mermaid syntax:
   ```mermaid
   graph TD
       A[Entry Point] --> B[Validation]
       B --> C[Processing]
       C --> D[Persistence]
   ```

## 3. Component Analysis (400 words)
   For each major procedure analyzed:
   - Business purpose and responsibility
   - Key algorithms and business logic
   - Data transformations performed
   - Error handling approach
   - Dependencies on other procedures

## 4. Data Flow (300 words)
   - Input data sources and formats
   - Data structures used (reference actual code)
   - Transformation pipeline through procedures
   - Output destinations
   - State management approach

## 5. Design Patterns & Practices (200 words)
   - Architectural patterns observed in code
   - Error handling and transaction strategy
   - Code organization and modularity
   - Reusability patterns

## 6. Technical Debt & Risks (200 words)
   - Code quality issues identified
   - Potential failure points
   - Maintainability concerns
   - Performance bottlenecks
   - Tight coupling issues

## 7. Modernization Recommendations (200 words)
   - RESTful API design suggestions
   - Microservice extraction opportunities
   - Event-driven architecture potential
   - Cloud-native migration path
   - Modern technology stack recommendations (Java/Spring Boot, etc.)

**CRITICAL INSTRUCTIONS:**
- Base ALL analysis on the ACTUAL source code provided above
- Reference specific procedures, variables, and logic from the code
- Use proper {self.functionality} industry terminology throughout
- Include Mermaid diagrams where helpful
- Focus on business logic and architecture, not TAL syntax
- Keep total response to 1500 words maximum
"""
        
        system_prompt = f"""You are a senior enterprise architect and payment systems expert specializing in legacy system modernization. You have deep expertise in {self.functionality} processing systems and understand both business requirements and technical implementation.

Your task is to analyze the COMPLETE TAL source code provided and produce a comprehensive architecture and design document. Focus on:
1. Business logic and data flow (not TAL syntax)
2. System architecture and design patterns
3. Integration points and dependencies
4. Modernization opportunities

Use proper {self.functionality} industry terminology throughout. Output in clean markdown with Mermaid diagrams where helpful. Limit to 1500 words."""
        
        return asyncio.run(self.llm_wrapper(system_prompt, user_prompt))
    
    def generate_component_documentation(self, component: ArchitectureComponent) -> str:
        """Generate component architecture documentation with FULL source code"""
        
        callers = [n for n in self.graph.nodes() if component.name in list(self.graph.successors(n))]
        
        user_prompt = f"""
Document this {self.functionality.upper()} component with COMPLETE source code:

=== TAL LANGUAGE CONTEXT ===
TAL is HP's systems programming language. Focus on business logic, not syntax.
Key patterns: PROC/END for procedures, CALL for invocation, return codes for errors.

=== FUNCTIONALITY CONTEXT ===
This component is part of the {self.functionality.upper()} system.
Relevance Score: {component.relevance_score:.1%}

=== COMPONENT DETAILS ===
**Name:** {component.name}
**File:** `{component.file_path}`
**Business Capability:** {component.business_capability}
**Complexity:** {component.complexity}
**Times Called:** {component.call_count} procedures call this

=== COMPLETE SOURCE CODE ===
```tal
{component.source_code if component.source_code else '[Source code not available]'}
```

=== DEPENDENCIES ===
**Called by:** {', '.join(callers[:8]) if callers else 'None'}
{f'... and {len(callers) - 8} more' if len(callers) > 8 else ''}

=== DOCUMENTATION REQUIREMENTS ===

Produce focused architecture documentation in MARKDOWN format (limit: 1500 words):

# {component.name} Component

## 1. Purpose in {self.functionality.title()} Processing (200 words)
   - Specific business responsibility
   - Role in overall {self.functionality} workflow
   - Business rules implemented
   - Industry standards followed (SWIFT, ISO 20022, etc.)

## 2. Component Flow Diagram
   Create Mermaid flowchart showing:
   - Input parameters/data
   - Key processing steps
   - Decision points
   - Output/return values
   
   Example:
   ```mermaid
   graph LR
       A[Input] --> B{{Validate}}
       B -->|Valid| C[Process]
       B -->|Invalid| D[Error]
       C --> E[Output]
   ```

## 3. Code Analysis (400 words)
   - Key algorithms and business logic
   - Data structures and variables used (reference actual code)
   - Validation rules and error handling
   - Transaction processing approach
   - State management

## 4. Integration Points (300 words)
   - Procedures this calls (dependencies)
   - Procedures that call this (consumers)
   - Data transformations performed
   - External system interactions
   - Database operations

## 5. Design Assessment (300 words)
   - Code quality indicators
   - Adherence to patterns
   - Potential failure points
   - Performance considerations
   - Maintainability score

## 6. Modernization Strategy (300 words)
   - Refactoring opportunities
   - Suggested API design (REST/gRPC)
   - Microservice extraction approach
   - Event-driven alternatives
   - Modern language equivalent (Java/Python)
   - Industry best practices to adopt

**CRITICAL:**
- Base analysis on ACTUAL source code provided
- Use proper {self.functionality} industry terminology
- Reference specific code lines, variables, logic
- Include Mermaid diagram
- Focus on architecture, not TAL syntax
- Limit to 1500 words maximum
"""
        
        system_prompt = f"""You are documenting a {self.functionality} system component. You are an expert in payment systems architecture and legacy modernization. Analyze the COMPLETE TAL source code and provide architectural documentation using proper {self.functionality} industry terminology. Output in clean markdown with diagrams. Limit to 1500 words."""
        
        return asyncio.run(self.llm_wrapper(system_prompt, user_prompt))
    
    def generate_functionality_report(self, output_file: str):
        """Generate complete functionality-focused documentation in Markdown"""
        logger.info(f"Generating {self.functionality} documentation report...")
        
        # Generate overview
        overview = self.generate_functionality_overview()
        
        # Generate component documentation
        component_docs = []
        logger.info(f"Documenting top {min(10, len(self.components))} components...")
        for comp in self.components[:10]:
            if comp.relevance_score > 0.2:  # Only document relevant ones
                comp_doc = self.generate_component_documentation(comp)
                component_docs.append({
                    'name': comp.name,
                    'file_path': comp.file_path,
                    'relevance': comp.relevance_score,
                    'complexity': comp.complexity,
                    'call_count': comp.call_count,
                    'documentation': comp_doc
                })
        
        # Build markdown report
        md_report = f"""# {self.functionality.title()} System Documentation

**Generated:** {Path.cwd()}  
**Functionality:** {self.functionality}  
**Domain Keywords:** {', '.join(self.functionality_keywords)}

## Metadata

- **Total Procedures:** {len(self.components)}
- **Procedures with Source Code:** {self.statistics['procedures_with_source']}
- **Source Coverage:** {self.statistics['source_coverage']:.1%}
- **Average Relevance:** {self.statistics.get('avg_relevance', 0):.1%}
- **Procedures Documented:** {len(component_docs)}

---

{overview}

---

# Component Documentation

"""
        
        # Add component documentation
        for i, comp_doc in enumerate(component_docs, 1):
            md_report += f"""
---

## Component {i}: {comp_doc['name']}

**File:** `{comp_doc['file_path']}`  
**Relevance:** {comp_doc['relevance']:.1%}  
**Complexity:** {comp_doc['complexity']}  
**Called By:** {comp_doc['call_count']} procedures

{comp_doc['documentation']}

"""
        
        # Save markdown report
        with open(output_file, 'w') as f:
            f.write(md_report)
        
        logger.info(f"Markdown documentation saved to {output_file}")
        
        return {
            'metadata': {
                'functionality': self.functionality,
                'total_procedures': len(self.components),
                'procedures_documented': len(component_docs),
                'source_coverage': f"{self.statistics['source_coverage']:.1%}",
                'avg_relevance': f"{self.statistics.get('avg_relevance', 0):.1%}"
            },
            'output_file': output_file
        }
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert to JSON-serializable types"""
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


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate focused documentation for specific business functionality',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported Functionalities:
  drawdown        - Loan drawdown/disbursement processing
  nostro          - Nostro account management and reconciliation
  wire_transfer   - Wire transfer processing (SWIFT, Fedwire)
  ach             - ACH/batch payment processing
  compliance      - OFAC/AML/KYC screening and compliance
  ledger          - General ledger and accounting
  settlement      - Payment settlement and clearing
  forex           - Foreign exchange and currency conversion

Examples:
  # Document drawdown functionality from subgraph
  python3 tal_ddg_auto_source.py callgraph_DRAWDOWN.json \\
    --functionality drawdown

  # Document nostro reconciliation
  python3 tal_ddg_auto_source.py callgraph_NOSTRO_RECON.json \\
    --functionality nostro --output nostro_docs.json
  
  # Document wire transfer processing
  python3 tal_ddg_auto_source.py callgraph_WIRE_MAIN.json \\
    --functionality wire_transfer
        """
    )
    
    parser.add_argument('graph_file', help='Path to subgraph JSON (from subgraph.py)')
    parser.add_argument('--functionality', required=True,
                       choices=['drawdown', 'nostro', 'wire_transfer', 'ach', 
                               'compliance', 'ledger', 'settlement', 'forex'],
                       help='Business functionality to document')
    parser.add_argument('--output', help='Output markdown file (default: <functionality>_documentation.md)')
    parser.add_argument('--min-relevance', type=float, default=0.0,
                       help='Minimum relevance score (0.0-1.0, default: 0.0)')
    
    args = parser.parse_args()
    
    if not Path(args.graph_file).exists():
        print(f"Error: File not found: {args.graph_file}")
        return 1
    
    # Default output filename
    if not args.output:
        args.output = f"{args.functionality}_documentation.md"
    
    try:
        # Initialize generator
        generator = TALFunctionalityDocGenerator(args.graph_file, args.functionality)
        
        # Analyze graph
        generator.analyze_graph(min_relevance=args.min_relevance)
        
        # Generate documentation
        report = generator.generate_functionality_report(args.output)
        
        logger.info("="*70)
        logger.info(f"{args.functionality.upper()} DOCUMENTATION COMPLETE!")
        logger.info("="*70)
        logger.info(f"Output: {args.output}")
        logger.info(f"Procedures documented: {report['metadata']['procedures_documented']}")
        logger.info(f"Source coverage: {report['metadata']['source_coverage']}")
        logger.info(f"Avg relevance: {report['metadata']['avg_relevance']}")
        logger.info("="*70)
        
        return 0
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
