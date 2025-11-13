#!/usr/bin/env python3
"""
TAL Deep Documentation Generator (DDG)
Generates comprehensive documentation from TAL knowledge graphs

Architecture:
- Multi-pass documentation generation (breadth → depth → integration → validation)
- Leverages existing knowledge graph infrastructure
- Integrates with payment processing RAG when available
- Outputs multiple documentation formats (markdown, JSON, HTML)
- LLM-powered rich analysis and descriptions

Usage:
    python tal_ddg.py <knowledge_graph.json> [options]
    python tal_ddg.py <tal_source_dir> --parse [options]
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Set, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import re
from datetime import datetime

# Import existing knowledge graph infrastructure
from knowledge_graph import (
    KnowledgeGraph, Entity, Relationship, 
    EntityType, RelationType
)

try:
    from parsers import (
        parse_tal_directory_recursive,
        KnowledgeGraphSearch,
        resolve_external_references
    )
    PARSERS_AVAILABLE = True
except ImportError:
    PARSERS_AVAILABLE = False
    print("Warning: parsers.py not available. Can only load pre-existing graphs.")


# ============================================================================
# Documentation Data Models
# ============================================================================

@dataclass
class DocumentationSection:
    """Represents a documentation section"""
    title: str
    content: str
    subsections: List['DocumentationSection'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    level: int = 1
    
    def to_markdown(self, base_level: int = 1) -> str:
        """Convert to markdown format"""
        level = base_level + self.level - 1
        header = "#" * level
        
        result = f"{header} {self.title}\n\n"
        result += f"{self.content}\n\n"
        
        for subsection in self.subsections:
            result += subsection.to_markdown(base_level)
        
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON export"""
        return {
            'title': self.title,
            'content': self.content,
            'subsections': [s.to_dict() for s in self.subsections],
            'metadata': self.metadata,
            'level': self.level
        }


@dataclass
class Documentation:
    """Complete documentation artifact"""
    title: str
    sections: List[DocumentationSection] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=datetime.now)
    
    def to_markdown(self) -> str:
        """Generate full markdown document"""
        result = f"# {self.title}\n\n"
        
        # Add metadata section
        result += "---\n"
        result += f"Generated: {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
        for key, value in self.metadata.items():
            result += f"{key}: {value}\n"
        result += "---\n\n"
        
        # Add all sections
        for section in self.sections:
            result += section.to_markdown()
        
        return result
    
    def to_json(self) -> str:
        """Generate JSON representation"""
        return json.dumps({
            'title': self.title,
            'sections': [s.to_dict() for s in self.sections],
            'metadata': self.metadata,
            'generated_at': self.generated_at.isoformat()
        }, indent=2)
    
    def to_html(self) -> str:
        """Generate HTML documentation"""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{self.title}</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #95a5a6; padding-bottom: 8px; margin-top: 30px; }}
        h3 {{ color: #7f8c8d; margin-top: 25px; }}
        .metadata {{ background: #ecf0f1; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .content {{ line-height: 1.6; }}
        code {{ background: #f4f4f4; padding: 2px 5px; border-radius: 3px; }}
        pre {{ background: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto; }}
        .toc {{ background: #e8f4f8; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .toc ul {{ list-style-type: none; }}
        .toc li {{ margin: 5px 0; }}
    </style>
</head>
<body>
    <h1>{self.title}</h1>
    <div class="metadata">
        <strong>Generated:</strong> {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}<br>
"""
        for key, value in self.metadata.items():
            html += f"        <strong>{key}:</strong> {value}<br>\n"
        html += "    </div>\n"
        
        # Add table of contents
        html += '    <div class="toc">\n'
        html += '        <h2>Table of Contents</h2>\n'
        html += '        <ul>\n'
        for section in self.sections:
            html += f'            <li><a href="#{self._make_anchor(section.title)}">{section.title}</a></li>\n'
        html += '        </ul>\n'
        html += '    </div>\n'
        
        # Add sections
        for section in self.sections:
            html += self._section_to_html(section)
        
        html += """
</body>
</html>
"""
        return html
    
    def _make_anchor(self, title: str) -> str:
        """Create HTML anchor from title"""
        return re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')
    
    def _section_to_html(self, section: DocumentationSection, level: int = 2) -> str:
        """Convert section to HTML"""
        anchor = self._make_anchor(section.title)
        html = f'    <h{level} id="{anchor}">{section.title}</h{level}>\n'
        html += f'    <div class="content">\n'
        
        # Convert markdown-style content to HTML
        content = section.content
        content = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', content)
        content = re.sub(r'`(.+?)`', r'<code>\1</code>', content)
        content = content.replace('\n\n', '</p><p>')
        content = f'<p>{content}</p>'
        
        html += f'        {content}\n'
        html += '    </div>\n'
        
        # Add subsections
        for subsection in section.subsections:
            html += self._section_to_html(subsection, level + 1)
        
        return html


# ============================================================================
# Documentation Generators - Multi-Pass Architecture
# ============================================================================

class BreadthPassGenerator:
    """
    Pass 1: Generate architectural overview and breadth documentation
    
    This pass creates:
    - System overview
    - Component inventory
    - High-level architecture
    - Module/file organization
    - Statistics and metrics
    """
    
    def __init__(self, kg: KnowledgeGraph, kg_search: Optional[KnowledgeGraphSearch] = None):
        self.kg = kg
        self.kg_search = kg_search or KnowledgeGraphSearch(kg)
    
    def generate(self) -> List[DocumentationSection]:
        """Generate breadth documentation sections"""
        sections = []
        
        # System Overview
        sections.append(self._generate_system_overview())
        
        # Component Inventory
        sections.append(self._generate_component_inventory())
        
        # Architecture Overview
        sections.append(self._generate_architecture_overview())
        
        # File Organization
        sections.append(self._generate_file_organization())
        
        # Statistics Summary
        sections.append(self._generate_statistics())
        
        return sections
    
    def _generate_system_overview(self) -> DocumentationSection:
        """Generate high-level system overview"""
        stats = self.kg.get_statistics()
        procedures = self.kg.query_entities(entity_type=EntityType.PROCEDURE)
        files = self.kg.query_entities(entity_type=EntityType.FILE)
        
        # Find main procedures
        main_procs = [p for p in procedures if p.metadata.get('is_main')]
        
        content = "## Overview\n\n"
        content += f"This TAL system consists of **{len(files)} source files** "
        content += f"containing **{len(procedures)} procedures**.\n\n"
        
        if main_procs:
            content += "### Entry Points\n\n"
            for proc in main_procs:
                file_name = Path(proc.file_path).name if proc.file_path else "unknown"
                content += f"- **{proc.name}** ({file_name})\n"
            content += "\n"
        
        content += "### System Characteristics\n\n"
        content += f"- Total Entities: {stats['total_entities']}\n"
        content += f"- Total Relationships: {stats['total_relationships']}\n"
        content += f"- Average Connectivity: {stats.get('avg_degree', 0):.2f}\n"
        
        return DocumentationSection(
            title="System Overview",
            content=content,
            level=1,
            metadata={'type': 'overview', 'pass': 'breadth'}
        )
    
    def _generate_component_inventory(self) -> DocumentationSection:
        """Generate inventory of all components"""
        procedures = self.kg.query_entities(entity_type=EntityType.PROCEDURE)
        structures = self.kg.query_entities(entity_type=EntityType.STRUCTURE)
        variables = self.kg.query_entities(entity_type=EntityType.VARIABLE)
        
        content = "Complete inventory of system components.\n\n"
        
        # Procedures
        content += f"### Procedures ({len(procedures)})\n\n"
        
        # Group by file
        procs_by_file = defaultdict(list)
        for proc in procedures:
            file_name = Path(proc.file_path).name if proc.file_path else "unknown"
            procs_by_file[file_name].append(proc)
        
        for file_name in sorted(procs_by_file.keys()):
            content += f"**{file_name}**\n\n"
            for proc in sorted(procs_by_file[file_name], key=lambda p: p.name):
                return_type = proc.metadata.get('return_type', 'void')
                param_count = proc.metadata.get('parameter_count', 0)
                content += f"- `{proc.name}` (returns {return_type}, {param_count} parameters)\n"
            content += "\n"
        
        # Structures
        if structures:
            content += f"### Data Structures ({len(structures)})\n\n"
            for struct in sorted(structures, key=lambda s: s.name):
                field_count = struct.metadata.get('field_count', 0)
                content += f"- `{struct.name}` ({field_count} fields)\n"
            content += "\n"
        
        return DocumentationSection(
            title="Component Inventory",
            content=content,
            level=1,
            metadata={'type': 'inventory', 'pass': 'breadth'}
        )
    
    def _generate_architecture_overview(self) -> DocumentationSection:
        """Generate architecture documentation"""
        procedures = self.kg.query_entities(entity_type=EntityType.PROCEDURE)
        call_relationships = self.kg.query_relationships(rel_type=RelationType.CALLS)
        
        content = "System architecture and calling patterns.\n\n"
        
        # Analyze calling patterns
        callers = defaultdict(int)
        callees = defaultdict(int)
        
        for rel in call_relationships:
            callers[rel.target_id] += 1
            callees[rel.source_id] += 1
        
        # Most called procedures (potential core services)
        content += "### Core Service Procedures\n\n"
        content += "Procedures called by many other procedures (potential utilities/services):\n\n"
        
        proc_by_id = {p.id: p for p in procedures}
        top_called = sorted(
            [(proc_by_id[pid], count) for pid, count in callers.items() if pid in proc_by_id],
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        for proc, count in top_called:
            content += f"- **{proc.name}** - called by {count} procedures\n"
        content += "\n"
        
        # High-level procedures (call many others but rarely called)
        content += "### High-Level Orchestrators\n\n"
        content += "Procedures that call many others but are rarely called:\n\n"
        
        orchestrators = []
        for proc in procedures:
            calls_out = callees.get(proc.id, 0)
            calls_in = callers.get(proc.id, 0)
            if calls_out >= 5 and calls_in <= 2:
                orchestrators.append((proc, calls_out, calls_in))
        
        orchestrators.sort(key=lambda x: x[1], reverse=True)
        for proc, out_count, in_count in orchestrators[:10]:
            content += f"- **{proc.name}** - calls {out_count} procedures, called by {in_count}\n"
        content += "\n"
        
        return DocumentationSection(
            title="Architecture Overview",
            content=content,
            level=1,
            metadata={'type': 'architecture', 'pass': 'breadth'}
        )
    
    def _generate_file_organization(self) -> DocumentationSection:
        """Generate file organization documentation"""
        files = self.kg.query_entities(entity_type=EntityType.FILE)
        
        content = "Source file organization and structure.\n\n"
        
        # Organize by directory
        files_by_dir = defaultdict(list)
        for file in files:
            if file.file_path:
                dir_path = str(Path(file.file_path).parent)
                files_by_dir[dir_path].append(file)
        
        content += "### Directory Structure\n\n"
        for dir_path in sorted(files_by_dir.keys()):
            content += f"**{dir_path}**\n\n"
            
            for file in sorted(files_by_dir[dir_path], key=lambda f: f.name):
                # Get procedures in this file
                procs = self.kg.get_neighbors(
                    file.id,
                    rel_type=RelationType.DEFINES,
                    direction="outgoing"
                )
                content += f"- {file.name} ({len(procs)} procedures)\n"
            content += "\n"
        
        return DocumentationSection(
            title="File Organization",
            content=content,
            level=1,
            metadata={'type': 'organization', 'pass': 'breadth'}
        )
    
    def _generate_statistics(self) -> DocumentationSection:
        """Generate statistics summary"""
        stats = self.kg.get_statistics()
        
        content = "System metrics and statistics.\n\n"
        
        content += "### Entity Counts\n\n"
        for entity_type, count in sorted(stats.get('entity_counts', {}).items()):
            content += f"- {entity_type}: {count}\n"
        content += "\n"
        
        content += "### Relationship Counts\n\n"
        for rel_type, count in sorted(stats.get('relationship_counts', {}).items()):
            content += f"- {rel_type}: {count}\n"
        content += "\n"
        
        return DocumentationSection(
            title="Statistics",
            content=content,
            level=1,
            metadata={'type': 'statistics', 'pass': 'breadth'}
        )


class DepthPassGenerator:
    """
    Pass 2: Generate deep documentation for each component
    
    This pass creates:
    - Detailed procedure documentation
    - Parameter and return type analysis
    - Local variable documentation
    - Call chain analysis
    - Data structure details
    """
    
    def __init__(self, kg: KnowledgeGraph, kg_search: Optional[KnowledgeGraphSearch] = None):
        self.kg = kg
        self.kg_search = kg_search or KnowledgeGraphSearch(kg)
    
    def generate(self, focus_procedures: Optional[List[str]] = None) -> List[DocumentationSection]:
        """
        Generate depth documentation
        
        Args:
            focus_procedures: If provided, only document these procedures in depth
        """
        sections = []
        
        # Get procedures to document
        procedures = self.kg.query_entities(entity_type=EntityType.PROCEDURE)
        
        if focus_procedures:
            procedures = [p for p in procedures if p.name in focus_procedures]
        
        # Generate deep documentation for each procedure
        for proc in procedures:
            section = self._generate_procedure_deep_doc(proc)
            sections.append(section)
        
        return sections
    
    def _generate_procedure_deep_doc(self, proc: Entity) -> DocumentationSection:
        """Generate comprehensive documentation for a single procedure"""
        content = ""
        subsections = []
        
        # Basic information
        content += f"**Procedure:** `{proc.name}`\n\n"
        
        if proc.file_path:
            file_name = Path(proc.file_path).name
            content += f"**File:** {file_name}\n"
        
        if proc.start_line:
            content += f"**Line:** {proc.start_line}\n"
        
        return_type = proc.metadata.get('return_type', 'void')
        content += f"**Returns:** {return_type}\n\n"
        
        # Attributes
        attributes = []
        if proc.metadata.get('is_main'):
            attributes.append('MAIN')
        if proc.metadata.get('is_forward'):
            attributes.append('FORWARD')
        if proc.metadata.get('is_external'):
            attributes.append('EXTERNAL')
        
        if attributes:
            content += f"**Attributes:** {', '.join(attributes)}\n\n"
        
        # Parameters
        params_section = self._generate_parameters_doc(proc)
        if params_section:
            subsections.append(params_section)
        
        # Local variables
        vars_section = self._generate_local_variables_doc(proc)
        if vars_section:
            subsections.append(vars_section)
        
        # Calls made by this procedure
        calls_section = self._generate_calls_doc(proc)
        if calls_section:
            subsections.append(calls_section)
        
        # Called by (incoming calls)
        called_by_section = self._generate_called_by_doc(proc)
        if called_by_section:
            subsections.append(called_by_section)
        
        # Complexity metrics
        metrics_section = self._generate_metrics_doc(proc)
        if metrics_section:
            subsections.append(metrics_section)
        
        return DocumentationSection(
            title=f"Procedure: {proc.name}",
            content=content,
            subsections=subsections,
            level=2,
            metadata={'type': 'procedure_detail', 'pass': 'depth', 'procedure_id': proc.id}
        )
    
    def _generate_parameters_doc(self, proc: Entity) -> Optional[DocumentationSection]:
        """Generate parameter documentation"""
        # Get parameter variables
        params = []
        for neighbor in self.kg.get_neighbors(proc.id, rel_type=RelationType.CONTAINS):
            if (neighbor.type == EntityType.VARIABLE and 
                neighbor.metadata.get('scope') == 'parameter'):
                params.append(neighbor)
        
        if not params:
            return None
        
        content = f"This procedure accepts {len(params)} parameters:\n\n"
        
        for param in params:
            data_type = param.metadata.get('data_type', 'UNKNOWN')
            is_pointer = param.metadata.get('is_pointer', False)
            
            content += f"- **{param.name}** ({data_type}"
            if is_pointer:
                content += ", pointer"
            content += ")\n"
        
        return DocumentationSection(
            title="Parameters",
            content=content,
            level=3
        )
    
    def _generate_local_variables_doc(self, proc: Entity) -> Optional[DocumentationSection]:
        """Generate local variables documentation"""
        variables = []
        for neighbor in self.kg.get_neighbors(proc.id, rel_type=RelationType.CONTAINS):
            if (neighbor.type == EntityType.VARIABLE and 
                neighbor.metadata.get('scope') == 'local'):
                variables.append(neighbor)
        
        if not variables:
            return None
        
        content = f"Local variables ({len(variables)}):\n\n"
        
        for var in variables:
            data_type = var.metadata.get('data_type', 'UNKNOWN')
            is_array = var.metadata.get('is_array', False)
            is_pointer = var.metadata.get('is_pointer', False)
            
            content += f"- **{var.name}** ({data_type}"
            if is_array:
                bounds = var.metadata.get('array_bounds', '')
                content += f", array{bounds}"
            if is_pointer:
                content += ", pointer"
            content += ")\n"
        
        return DocumentationSection(
            title="Local Variables",
            content=content,
            level=3
        )
    
    def _generate_calls_doc(self, proc: Entity) -> Optional[DocumentationSection]:
        """Generate documentation of procedures called"""
        callees = self.kg.get_neighbors(
            proc.id,
            rel_type=RelationType.CALLS,
            direction="outgoing"
        )
        
        if not callees:
            return None
        
        content = f"This procedure calls {len(callees)} other procedures:\n\n"
        
        for callee in callees:
            is_external = callee.metadata.get('is_external', False)
            content += f"- **{callee.name}**"
            if is_external:
                content += " (external)"
            content += "\n"
        
        return DocumentationSection(
            title="Calls",
            content=content,
            level=3
        )
    
    def _generate_called_by_doc(self, proc: Entity) -> Optional[DocumentationSection]:
        """Generate documentation of callers"""
        callers = self.kg.get_neighbors(
            proc.id,
            rel_type=RelationType.CALLS,
            direction="incoming"
        )
        
        if not callers:
            content = "This procedure is not called by any other procedures in the system.\n"
            return DocumentationSection(
                title="Called By",
                content=content,
                level=3
            )
        
        content = f"This procedure is called by {len(callers)} procedures:\n\n"
        
        for caller in callers:
            content += f"- **{caller.name}**\n"
        
        return DocumentationSection(
            title="Called By",
            content=content,
            level=3
        )
    
    def _generate_metrics_doc(self, proc: Entity) -> Optional[DocumentationSection]:
        """Generate complexity metrics"""
        stmt_count = proc.metadata.get('statement_count', 0)
        local_var_count = proc.metadata.get('local_variable_count', 0)
        param_count = proc.metadata.get('parameter_count', 0)
        
        # Calculate outgoing call count
        callees = self.kg.get_neighbors(
            proc.id,
            rel_type=RelationType.CALLS,
            direction="outgoing"
        )
        
        content = "**Complexity Metrics:**\n\n"
        content += f"- Statement Count: {stmt_count}\n"
        content += f"- Local Variables: {local_var_count}\n"
        content += f"- Parameters: {param_count}\n"
        content += f"- Outgoing Calls: {len(callees)}\n"
        
        # Calculate complexity indicator
        complexity = stmt_count + len(callees) * 2 + local_var_count
        if complexity < 20:
            complexity_level = "Low"
        elif complexity < 50:
            complexity_level = "Medium"
        else:
            complexity_level = "High"
        
        content += f"- Complexity: {complexity_level}\n"
        
        return DocumentationSection(
            title="Metrics",
            content=content,
            level=3
        )


class IntegrationPassGenerator:
    """
    Pass 3: Generate integration and workflow documentation
    
    This pass creates:
    - End-to-end workflow documentation
    - Call chain diagrams
    - Data flow documentation
    - Integration point analysis
    - Cross-cutting concerns
    """
    
    def __init__(self, kg: KnowledgeGraph, kg_search: Optional[KnowledgeGraphSearch] = None):
        self.kg = kg
        self.kg_search = kg_search or KnowledgeGraphSearch(kg)
    
    def generate(self, entry_points: Optional[List[str]] = None) -> List[DocumentationSection]:
        """Generate integration documentation"""
        sections = []
        
        # Call chain analysis
        if entry_points:
            for entry in entry_points:
                section = self._generate_call_chain_doc(entry)
                if section:
                    sections.append(section)
        
        # Data flow analysis
        sections.append(self._generate_data_flow_doc())
        
        # Integration points
        sections.append(self._generate_integration_points_doc())
        
        return sections
    
    def _generate_call_chain_doc(self, entry_procedure: str) -> Optional[DocumentationSection]:
        """Generate call chain documentation from an entry point"""
        # Find the entry procedure
        procedures = self.kg.query_entities(entity_type=EntityType.PROCEDURE)
        entry_proc = next((p for p in procedures if p.name == entry_procedure), None)
        
        if not entry_proc:
            return None
        
        # Get call subgraph
        try:
            subgraph = self.kg.extract_call_subgraph(
                entry_procedures=[entry_procedure],
                max_depth=5,
                include_variables=False,
                include_structures=False
            )
        except Exception as e:
            print(f"Warning: Could not extract call subgraph: {e}")
            return None
        
        content = f"Call chain starting from **{entry_procedure}**\n\n"
        
        # Analyze subgraph
        proc_count = subgraph['statistics']['procedure_count']
        content += f"This workflow involves {proc_count} procedures.\n\n"
        
        # Build call tree visualization
        content += "### Call Tree\n\n"
        content += self._build_call_tree(entry_proc, max_depth=3)
        
        return DocumentationSection(
            title=f"Workflow: {entry_procedure}",
            content=content,
            level=2,
            metadata={'type': 'workflow', 'pass': 'integration'}
        )
    
    def _build_call_tree(self, proc: Entity, depth: int = 0, max_depth: int = 3, visited: Optional[Set[str]] = None) -> str:
        """Build ASCII call tree"""
        if visited is None:
            visited = set()
        
        if depth > max_depth or proc.id in visited:
            return ""
        
        visited.add(proc.id)
        
        indent = "  " * depth
        tree = f"{indent}- {proc.name}\n"
        
        # Get callees
        callees = self.kg.get_neighbors(
            proc.id,
            rel_type=RelationType.CALLS,
            direction="outgoing"
        )
        
        for callee in callees:
            tree += self._build_call_tree(callee, depth + 1, max_depth, visited)
        
        return tree
    
    def _generate_data_flow_doc(self) -> DocumentationSection:
        """Generate data flow documentation"""
        structures = self.kg.query_entities(entity_type=EntityType.STRUCTURE)
        
        content = "Data structures and their usage patterns.\n\n"
        
        if structures:
            content += f"### Data Structures ({len(structures)})\n\n"
            
            for struct in structures:
                content += f"**{struct.name}**\n\n"
                
                # Get field information
                fields = struct.metadata.get('fields', [])
                if fields:
                    content += "Fields:\n"
                    for field in fields:
                        field_name = field.get('name', 'unknown')
                        field_type = field.get('type', 'UNKNOWN')
                        content += f"- {field_name} ({field_type})\n"
                    content += "\n"
        
        return DocumentationSection(
            title="Data Flow",
            content=content,
            level=2,
            metadata={'type': 'data_flow', 'pass': 'integration'}
        )
    
    def _generate_integration_points_doc(self) -> DocumentationSection:
        """Generate integration points documentation"""
        procedures = self.kg.query_entities(entity_type=EntityType.PROCEDURE)
        
        # Find external procedures
        external_procs = [p for p in procedures if p.metadata.get('is_external')]
        
        content = "External integration points and dependencies.\n\n"
        
        if external_procs:
            content += f"### External Procedures ({len(external_procs)})\n\n"
            content += "These procedures are defined externally:\n\n"
            
            for proc in external_procs:
                # Find who calls this external procedure
                callers = self.kg.get_neighbors(
                    proc.id,
                    rel_type=RelationType.CALLS,
                    direction="incoming"
                )
                
                content += f"- **{proc.name}** (called by {len(callers)} procedures)\n"
        else:
            content += "No external integration points found.\n"
        
        return DocumentationSection(
            title="Integration Points",
            content=content,
            level=2,
            metadata={'type': 'integration', 'pass': 'integration'}
        )


class ValidationPassGenerator:
    """
    Pass 4: Validate and enhance documentation
    
    This pass:
    - Validates documentation completeness
    - Identifies gaps and missing information
    - Suggests improvements
    - Generates quality metrics
    """
    
    def __init__(self, documentation: Documentation, kg: KnowledgeGraph):
        self.documentation = documentation
        self.kg = kg
    
    def validate(self) -> Dict[str, Any]:
        """Validate documentation and return report"""
        report = {
            'validation_date': datetime.now().isoformat(),
            'sections_count': len(self.documentation.sections),
            'issues': [],
            'suggestions': [],
            'quality_score': 0,
            'completeness': {}
        }
        
        # Check for required sections
        section_titles = [s.title.lower() for s in self.documentation.sections]
        
        required_sections = [
            'system overview',
            'component inventory',
            'architecture overview'
        ]
        
        for required in required_sections:
            if not any(required in title for title in section_titles):
                report['issues'].append(f"Missing required section: {required}")
        
        # Check documentation depth
        procedures = self.kg.query_entities(entity_type=EntityType.PROCEDURE)
        documented_procs = set()
        
        for section in self.documentation.sections:
            if section.metadata.get('type') == 'procedure_detail':
                proc_id = section.metadata.get('procedure_id')
                if proc_id:
                    documented_procs.add(proc_id)
        
        coverage = len(documented_procs) / max(len(procedures), 1)
        report['completeness']['procedure_coverage'] = f"{coverage * 100:.1f}%"
        
        if coverage < 0.5:
            report['suggestions'].append(
                f"Low procedure coverage ({coverage * 100:.1f}%). "
                "Consider generating depth documentation for more procedures."
            )
        
        # Calculate quality score
        quality_score = 0
        if len(report['issues']) == 0:
            quality_score += 50
        
        if coverage > 0.8:
            quality_score += 30
        elif coverage > 0.5:
            quality_score += 20
        elif coverage > 0.2:
            quality_score += 10
        
        if len(self.documentation.sections) >= 5:
            quality_score += 20
        
        report['quality_score'] = quality_score
        
        return report


# ============================================================================
# Main Documentation Generator
# ============================================================================

class TALDocumentationGenerator:
    """
    Main orchestrator for TAL documentation generation
    
    Implements multi-pass documentation strategy:
    1. Breadth Pass - Architecture and overview
    2. Depth Pass - Detailed component documentation
    3. Integration Pass - Workflows and data flow
    4. Validation Pass - Quality assurance
    """
    
    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg
        self.kg_search = KnowledgeGraphSearch(kg)
        
        self.breadth_gen = BreadthPassGenerator(kg, self.kg_search)
        self.depth_gen = DepthPassGenerator(kg, self.kg_search)
        self.integration_gen = IntegrationPassGenerator(kg, self.kg_search)
    
    def generate_complete_documentation(self,
                                       title: str = "TAL System Documentation",
                                       depth_focus: Optional[List[str]] = None,
                                       entry_points: Optional[List[str]] = None) -> Documentation:
        """
        Generate complete documentation with all passes
        
        Args:
            title: Documentation title
            depth_focus: Procedures to document in depth (None = all)
            entry_points: Entry points for workflow analysis
        
        Returns:
            Complete Documentation object
        """
        print("Generating TAL Documentation...")
        print("=" * 60)
        
        doc = Documentation(
            title=title,
            metadata={
                'generator': 'TAL DDG',
                'version': '1.0',
                'backend': self.kg.backend
            }
        )
        
        # Pass 1: Breadth
        print("\nPass 1: Generating breadth documentation...")
        breadth_sections = self.breadth_gen.generate()
        doc.sections.extend(breadth_sections)
        print(f"  Generated {len(breadth_sections)} breadth sections")
        
        # Pass 2: Depth
        print("\nPass 2: Generating depth documentation...")
        depth_sections = self.depth_gen.generate(focus_procedures=depth_focus)
        if depth_sections:
            # Group depth sections under a parent section
            depth_parent = DocumentationSection(
                title="Detailed Component Documentation",
                content="Comprehensive documentation for each component.",
                subsections=depth_sections,
                level=1
            )
            doc.sections.append(depth_parent)
        print(f"  Generated {len(depth_sections)} depth sections")
        
        # Pass 3: Integration
        if entry_points:
            print("\nPass 3: Generating integration documentation...")
            integration_sections = self.integration_gen.generate(entry_points=entry_points)
            if integration_sections:
                integration_parent = DocumentationSection(
                    title="Workflows and Integration",
                    content="End-to-end workflows and integration patterns.",
                    subsections=integration_sections,
                    level=1
                )
                doc.sections.append(integration_parent)
            print(f"  Generated {len(integration_sections)} integration sections")
        
        # Pass 4: Validation
        print("\nPass 4: Validating documentation...")
        validator = ValidationPassGenerator(doc, self.kg)
        validation_report = validator.validate()
        
        # Add validation report as metadata
        doc.metadata['validation'] = validation_report
        
        print(f"\nValidation Results:")
        print(f"  Quality Score: {validation_report['quality_score']}/100")
        print(f"  Procedure Coverage: {validation_report['completeness'].get('procedure_coverage', 'N/A')}")
        if validation_report['issues']:
            print(f"  Issues: {len(validation_report['issues'])}")
            for issue in validation_report['issues']:
                print(f"    - {issue}")
        if validation_report['suggestions']:
            print(f"  Suggestions: {len(validation_report['suggestions'])}")
            for suggestion in validation_report['suggestions']:
                print(f"    - {suggestion}")
        
        print("\n" + "=" * 60)
        print("Documentation generation complete!")
        
        return doc


# ============================================================================
# Command-Line Interface
# ============================================================================

def main():
    """Main CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='TAL Deep Documentation Generator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate docs from existing knowledge graph
  python tal_ddg.py knowledge_graph.json -o docs/
  
  # Parse TAL source and generate docs
  python tal_ddg.py tal_source/ --parse -o docs/
  
  # Focus on specific procedures
  python tal_ddg.py graph.json --focus MAIN_PROC PROCESS_PAYMENT -o docs/
  
  # Generate workflow documentation
  python tal_ddg.py graph.json --workflows MAIN_PROC --entry MAIN_PROC -o docs/
  
  # All formats
  python tal_ddg.py graph.json -o docs/ --format markdown html json
        """
    )
    
    parser.add_argument('input', help='Knowledge graph JSON or TAL source directory')
    parser.add_argument('-o', '--output', default='./docs', help='Output directory (default: ./docs)')
    parser.add_argument('--parse', action='store_true', help='Parse TAL source before generating docs')
    parser.add_argument('--focus', nargs='+', help='Procedures to document in depth')
    parser.add_argument('--entry', '--workflows', dest='entry_points', nargs='+', 
                       help='Entry points for workflow analysis')
    parser.add_argument('--format', nargs='+', choices=['markdown', 'html', 'json'],
                       default=['markdown'], help='Output formats (default: markdown)')
    parser.add_argument('--title', default='TAL System Documentation', 
                       help='Documentation title')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    try:
        # Load or create knowledge graph
        kg = KnowledgeGraph(backend="networkx")
        
        if args.parse:
            if not PARSERS_AVAILABLE:
                print("Error: parsers.py not available for parsing TAL source")
                return 1
            
            if not input_path.is_dir():
                print(f"Error: {input_path} is not a directory")
                return 1
            
            print(f"Parsing TAL source from: {input_path}")
            results = parse_tal_directory_recursive(str(input_path), kg, recursive=True)
            
            if not results['success']:
                print(f"Error parsing TAL source: {results.get('error')}")
                return 1
            
            print(f"Parsed {len(results['parsed_successfully'])} files")
            
            # Resolve external references
            resolve_external_references(kg)
            
        else:
            if not input_path.exists():
                print(f"Error: {input_path} does not exist")
                return 1
            
            print(f"Loading knowledge graph from: {input_path}")
            kg.load_from_json(str(input_path))
        
        # Generate documentation
        generator = TALDocumentationGenerator(kg)
        
        doc = generator.generate_complete_documentation(
            title=args.title,
            depth_focus=args.focus,
            entry_points=args.entry_points
        )
        
        # Save in requested formats
        print(f"\nSaving documentation to: {output_dir}")
        
        if 'markdown' in args.format:
            md_file = output_dir / "documentation.md"
            with open(md_file, 'w', encoding='utf-8') as f:
                f.write(doc.to_markdown())
            print(f"  ✓ Markdown: {md_file}")
        
        if 'html' in args.format:
            html_file = output_dir / "documentation.html"
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(doc.to_html())
            print(f"  ✓ HTML: {html_file}")
        
        if 'json' in args.format:
            json_file = output_dir / "documentation.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                f.write(doc.to_json())
            print(f"  ✓ JSON: {json_file}")
        
        print("\nDocumentation generation complete!")
        return 0
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
