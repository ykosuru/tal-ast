"""
Enhanced LLM Context Builder for Code Rewriting
Specialized context generation for TAL-to-Java migration

Features:
- Include actual procedure code snippets
- Business logic descriptions
- Call chain analysis
- Data structure mappings
- Migration hints and patterns
- Comprehensive context for LLM understanding
"""

from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from pathlib import Path
import json

from knowledge_graph import Entity, Relationship, EntityType, RelationType, KnowledgeGraph
from parsers import KnowledgeGraphSearch


# ============================================================================
# Code Rewriting Context Builder
# ============================================================================

@dataclass
class CodeRewritingContext:
    """Container for code rewriting context"""
    search_term: str
    target_language: str
    source_procedures: List[Dict[str, Any]]
    data_structures: List[Dict[str, Any]]
    call_chains: List[Dict[str, Any]]
    dependencies: List[Dict[str, Any]]
    business_logic_summary: str
    migration_hints: List[str]
    code_snippets: Dict[str, str]


class CodeRewritingContextBuilder:
    """Build comprehensive LLM context for code rewriting tasks"""
    
    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg
        self.search = KnowledgeGraphSearch(kg)
    
    def build_rewriting_context(self,
                               search_term: str,
                               target_language: str = "Java",
                               include_code: bool = True,
                               max_procedures: int = 20) -> CodeRewritingContext:
        """
        Build comprehensive context for rewriting TAL code to target language
        
        Args:
            search_term: Functionality to search for (e.g., "drawdown", "payment")
            target_language: Target language for rewriting (default: Java)
            include_code: Include actual code snippets
            max_procedures: Maximum procedures to include
        
        Returns:
            CodeRewritingContext with all necessary information
        """
        # Search for relevant procedures
        search_results = self.search.find_by_functionality(search_term)
        
        # Extract source procedures
        procedures = search_results.get('procedures', [])[:max_procedures]
        source_procedures = []
        
        for proc in procedures:
            proc_info = self._extract_procedure_info(proc, include_code)
            source_procedures.append(proc_info)
        
        # Extract data structures
        structures = search_results.get('structures', [])
        data_structures = []
        for struct in structures:
            struct_info = self._extract_structure_info(struct)
            data_structures.append(struct_info)
        
        # Analyze call chains
        call_chains = self._analyze_call_chains(procedures)
        
        # Extract dependencies
        dependencies = self._extract_dependencies(procedures)
        
        # Generate business logic summary
        business_logic_summary = self._generate_business_logic_summary(
            search_term,
            procedures,
            call_chains
        )
        
        # Generate migration hints
        migration_hints = self._generate_migration_hints(
            procedures,
            data_structures,
            target_language
        )
        
        # Collect code snippets
        code_snippets = {}
        if include_code:
            code_snippets = self._collect_code_snippets(procedures)
        
        return CodeRewritingContext(
            search_term=search_term,
            target_language=target_language,
            source_procedures=source_procedures,
            data_structures=data_structures,
            call_chains=call_chains,
            dependencies=dependencies,
            business_logic_summary=business_logic_summary,
            migration_hints=migration_hints,
            code_snippets=code_snippets
        )
    
    def format_for_llm(self, context: CodeRewritingContext, format: str = "markdown") -> str:
        """
        Format context for LLM consumption
        
        Args:
            context: CodeRewritingContext to format
            format: Output format (markdown, json, xml)
        
        Returns:
            Formatted string ready for LLM
        """
        if format == "markdown":
            return self._format_markdown(context)
        elif format == "json":
            return self._format_json(context)
        elif format == "xml":
            return self._format_xml(context)
        else:
            return self._format_markdown(context)
    
    def _extract_procedure_info(self, procedure: Entity, include_code: bool) -> Dict[str, Any]:
        """Extract detailed procedure information"""
        info = {
            'name': procedure.name,
            'qualified_name': procedure.qualified_name,
            'file': str(Path(procedure.file_path).name) if procedure.file_path else None,
            'file_path': procedure.file_path,
            'line': procedure.start_line,
            'return_type': procedure.metadata.get('return_type', 'void'),
            'parameters': procedure.metadata.get('parameters', []),
            'parameter_count': procedure.metadata.get('parameter_count', 0),
            'is_main': procedure.metadata.get('is_main', False),
            'is_external': procedure.metadata.get('is_external', False),
            'statement_count': procedure.metadata.get('statement_count', 0),
            'local_variable_count': procedure.metadata.get('local_variable_count', 0)
        }
        
        # Get calls made by this procedure
        callees = self.kg.get_neighbors(
            procedure.id,
            rel_type=RelationType.CALLS,
            direction="outgoing"
        )
        info['calls'] = [c.name for c in callees]
        info['calls_count'] = len(callees)
        
        # Get procedures that call this one
        callers = self.kg.get_neighbors(
            procedure.id,
            rel_type=RelationType.CALLS,
            direction="incoming"
        )
        info['called_by'] = [c.name for c in callers]
        info['called_by_count'] = len(callers)
        
        # Get local variables
        variables = self.kg.get_neighbors(
            procedure.id,
            rel_type=RelationType.CONTAINS,
            direction="outgoing"
        )
        local_vars = [v for v in variables if v.type == EntityType.VARIABLE]
        info['local_variables'] = []
        for var in local_vars:
            info['local_variables'].append({
                'name': var.name,
                'type': var.metadata.get('data_type', 'UNKNOWN'),
                'is_array': var.metadata.get('is_array', False),
                'is_pointer': var.metadata.get('is_pointer', False)
            })
        
        return info
    
    def _extract_structure_info(self, structure: Entity) -> Dict[str, Any]:
        """Extract structure information"""
        info = {
            'name': structure.name,
            'qualified_name': structure.qualified_name,
            'field_count': structure.metadata.get('field_count', 0),
            'fields': structure.metadata.get('fields', []),
            'is_nested': structure.metadata.get('is_nested', False)
        }
        return info
    
    def _analyze_call_chains(self, procedures: List[Entity]) -> List[Dict[str, Any]]:
        """Analyze call chains between procedures"""
        call_chains = []
        
        for proc in procedures:
            # Find who calls this procedure
            callers = self.kg.get_neighbors(
                proc.id,
                rel_type=RelationType.CALLS,
                direction="incoming"
            )
            
            # Find what this procedure calls
            callees = self.kg.get_neighbors(
                proc.id,
                rel_type=RelationType.CALLS,
                direction="outgoing"
            )
            
            if callers or callees:
                chain_info = {
                    'procedure': proc.name,
                    'entry_point': len(callers) == 0 and proc.metadata.get('is_main', False),
                    'callers': [c.name for c in callers],
                    'callees': [c.name for c in callees],
                    'depth': self._calculate_call_depth(proc)
                }
                call_chains.append(chain_info)
        
        # Sort by depth (entry points first)
        call_chains.sort(key=lambda x: x['depth'])
        
        return call_chains
    
    def _calculate_call_depth(self, procedure: Entity) -> int:
        """Calculate call depth from entry point"""
        # BFS to find shortest path to an entry point
        visited = {procedure.id}
        queue = [(procedure.id, 0)]
        
        while queue:
            current_id, depth = queue.pop(0)
            current = self.kg.get_entity(current_id)
            
            if current and current.metadata.get('is_main'):
                return depth
            
            # Get callers
            callers = self.kg.get_neighbors(
                current_id,
                rel_type=RelationType.CALLS,
                direction="incoming"
            )
            
            if not callers:  # No callers, this is an entry point
                return depth
            
            for caller in callers:
                if caller.id not in visited:
                    visited.add(caller.id)
                    queue.append((caller.id, depth + 1))
        
        return 999  # No path to entry point found
    
    def _extract_dependencies(self, procedures: List[Entity]) -> List[Dict[str, Any]]:
        """Extract external dependencies and references"""
        dependencies = []
        seen_external = set()
        
        for proc in procedures:
            # Get external calls
            callees = self.kg.get_neighbors(
                proc.id,
                rel_type=RelationType.CALLS,
                direction="outgoing"
            )
            
            for callee in callees:
                if callee.metadata.get('is_external') and callee.name not in seen_external:
                    seen_external.add(callee.name)
                    dependencies.append({
                        'name': callee.name,
                        'type': 'external_procedure',
                        'called_by': [proc.name],
                        'resolved': callee.metadata.get('resolved', False)
                    })
        
        return dependencies
    
    def _generate_business_logic_summary(self,
                                        search_term: str,
                                        procedures: List[Entity],
                                        call_chains: List[Dict[str, Any]]) -> str:
        """Generate high-level business logic summary"""
        lines = []
        
        lines.append(f"Business Logic Summary for '{search_term}'")
        lines.append("=" * 70)
        lines.append("")
        
        # Entry points
        entry_points = [c for c in call_chains if c['entry_point'] or not c['callers']]
        if entry_points:
            lines.append("Entry Points:")
            for ep in entry_points:
                lines.append(f"  • {ep['procedure']}")
            lines.append("")
        
        # Main processing flow
        lines.append("Processing Flow:")
        for chain in call_chains[:10]:  # Top 10
            if chain['callees']:
                lines.append(f"  {chain['procedure']}:")
                for callee in chain['callees'][:5]:
                    lines.append(f"    → {callee}")
        lines.append("")
        
        # Statistics
        total_procs = len(procedures)
        external_deps = len([p for p in procedures if p.metadata.get('is_external')])
        avg_complexity = sum(p.metadata.get('statement_count', 0) for p in procedures) / max(total_procs, 1)
        
        lines.append("Statistics:")
        lines.append(f"  • Total procedures: {total_procs}")
        lines.append(f"  • External dependencies: {external_deps}")
        lines.append(f"  • Average complexity: {avg_complexity:.1f} statements/procedure")
        
        return '\n'.join(lines)
    
    def _generate_migration_hints(self,
                                  procedures: List[Entity],
                                  data_structures: List[Dict[str, Any]],
                                  target_language: str) -> List[str]:
        """Generate language-specific migration hints"""
        hints = []
        
        if target_language.lower() == "java":
            hints.extend(self._generate_java_hints(procedures, data_structures))
        elif target_language.lower() == "python":
            hints.extend(self._generate_python_hints(procedures, data_structures))
        else:
            hints.extend(self._generate_generic_hints(procedures, data_structures))
        
        return hints
    
    def _generate_java_hints(self,
                            procedures: List[Entity],
                            data_structures: List[Dict[str, Any]]) -> List[str]:
        """Generate Java-specific migration hints"""
        hints = []
        
        # Procedure to method hints
        hints.append("TAL procedures should be converted to Java methods in appropriate classes")
        hints.append("Consider grouping related procedures into service classes")
        
        # Main procedure handling
        main_procs = [p for p in procedures if p.metadata.get('is_main')]
        if main_procs:
            hints.append(f"Main procedures ({', '.join(p.name for p in main_procs)}) should become entry points or controller methods")
        
        # Parameter handling
        for proc in procedures:
            params = proc.metadata.get('parameters', [])
            if len(params) > 5:
                hints.append(f"{proc.name} has {len(params)} parameters - consider using a parameter object pattern in Java")
        
        # Data structure hints
        if data_structures:
            hints.append("TAL structures should be converted to Java classes or records")
            hints.append("Consider using Builder pattern for structures with many fields")
        
        # Pointer handling
        has_pointers = any(
            any(v.get('is_pointer') for v in p.metadata.get('local_variables', []))
            for p in procedures
        )
        if has_pointers:
            hints.append("TAL pointers should be converted to Java object references")
        
        # Array handling
        has_arrays = any(
            any(v.get('is_array') for v in p.metadata.get('local_variables', []))
            for p in procedures
        )
        if has_arrays:
            hints.append("TAL arrays should be converted to Java arrays or Collections (List, ArrayList)")
        
        # Error handling
        hints.append("Implement proper exception handling for error conditions")
        hints.append("Consider using Optional<T> for nullable return values")
        
        return hints
    
    def _generate_python_hints(self,
                               procedures: List[Entity],
                               data_structures: List[Dict[str, Any]]) -> List[str]:
        """Generate Python-specific migration hints"""
        hints = []
        
        hints.append("TAL procedures should be converted to Python functions or class methods")
        hints.append("Consider using dataclasses for TAL structures")
        hints.append("Use type hints for better code documentation and IDE support")
        hints.append("TAL arrays can become Python lists")
        hints.append("Consider using None for optional values instead of special sentinels")
        
        return hints
    
    def _generate_generic_hints(self,
                                procedures: List[Entity],
                                data_structures: List[Dict[str, Any]]) -> List[str]:
        """Generate generic migration hints"""
        hints = []
        
        hints.append("Review procedure call chains to understand control flow")
        hints.append("Pay attention to data structure definitions and field types")
        hints.append("External dependencies may need to be stubbed or mocked")
        hints.append("Consider transaction boundaries and error handling patterns")
        
        return hints
    
    def _collect_code_snippets(self, procedures: List[Entity]) -> Dict[str, str]:
        """Collect actual code snippets for procedures"""
        snippets = {}
        
        for proc in procedures:
            if proc.file_path and proc.start_line:
                try:
                    # Read the file and extract procedure code
                    with open(proc.file_path, 'r', encoding='latin-1') as f:
                        lines = f.readlines()
                    
                    # Get procedure code (start_line to end_line or next 50 lines)
                    end_line = proc.end_line if proc.end_line else min(
                        proc.start_line + 50,
                        len(lines)
                    )
                    
                    proc_lines = lines[proc.start_line - 1:end_line]
                    snippets[proc.name] = ''.join(proc_lines)
                    
                except Exception as e:
                    snippets[proc.name] = f"# Could not read code: {str(e)}"
        
        return snippets
    
    def _format_markdown(self, context: CodeRewritingContext) -> str:
        """Format context as Markdown for LLM"""
        lines = []
        
        # Header
        lines.append(f"# TAL to {context.target_language} Migration Context")
        lines.append(f"## Functionality: `{context.search_term}`")
        lines.append("")
        
        # Business logic summary
        lines.append("## Business Logic Overview")
        lines.append(context.business_logic_summary)
        lines.append("")
        
        # Procedures
        lines.append(f"## Source Procedures ({len(context.source_procedures)})")
        lines.append("")
        
        for idx, proc in enumerate(context.source_procedures, 1):
            lines.append(f"### {idx}. `{proc['name']}`")
            lines.append("")
            
            # Metadata
            lines.append("| Property | Value |")
            lines.append("|----------|-------|")
            lines.append(f"| File | `{proc['file']}` |")
            lines.append(f"| Line | {proc['line']} |")
            lines.append(f"| Return Type | `{proc['return_type']}` |")
            lines.append(f"| Parameters | {proc['parameter_count']} |")
            
            if proc['parameters']:
                lines.append(f"| Parameter Names | `{', '.join(proc['parameters'])}` |")
            
            lines.append(f"| Statements | {proc['statement_count']} |")
            lines.append(f"| Local Variables | {proc['local_variable_count']} |")
            lines.append(f"| Calls | {proc['calls_count']} procedures |")
            lines.append(f"| Called By | {proc['called_by_count']} procedures |")
            
            if proc['is_main']:
                lines.append(f"| Entry Point | ⭐ Main |")
            if proc['is_external']:
                lines.append(f"| External | Yes |")
            
            lines.append("")
            
            # Calls
            if proc['calls']:
                lines.append("**Calls:**")
                for callee in proc['calls'][:10]:
                    lines.append(f"- `{callee}`")
                if len(proc['calls']) > 10:
                    lines.append(f"- ... and {len(proc['calls']) - 10} more")
                lines.append("")
            
            # Called by
            if proc['called_by']:
                lines.append("**Called by:**")
                for caller in proc['called_by'][:10]:
                    lines.append(f"- `{caller}`")
                if len(proc['called_by']) > 10:
                    lines.append(f"- ... and {len(proc['called_by']) - 10} more")
                lines.append("")
            
            # Local variables
            if proc['local_variables']:
                lines.append("**Local Variables:**")
                for var in proc['local_variables'][:15]:
                    var_desc = f"{var['name']}: {var['type']}"
                    if var['is_array']:
                        var_desc += " (array)"
                    if var['is_pointer']:
                        var_desc += " (pointer)"
                    lines.append(f"- `{var_desc}`")
                if len(proc['local_variables']) > 15:
                    lines.append(f"- ... and {len(proc['local_variables']) - 15} more")
                lines.append("")
            
            # Code snippet
            if proc['name'] in context.code_snippets:
                lines.append("**Source Code:**")
                lines.append("```tal")
                lines.append(context.code_snippets[proc['name']].strip())
                lines.append("```")
                lines.append("")
            
            lines.append("---")
            lines.append("")
        
        # Data structures
        if context.data_structures:
            lines.append(f"## Data Structures ({len(context.data_structures)})")
            lines.append("")
            
            for struct in context.data_structures:
                lines.append(f"### `{struct['name']}`")
                lines.append("")
                lines.append(f"**Fields:** {struct['field_count']}")
                lines.append("")
                
                if struct['fields']:
                    lines.append("| Field Name | Type |")
                    lines.append("|------------|------|")
                    for field in struct['fields']:
                        field_name = field.get('name', 'unknown')
                        field_type = field.get('type', 'UNKNOWN')
                        lines.append(f"| `{field_name}` | `{field_type}` |")
                
                lines.append("")
        
        # Call chains
        if context.call_chains:
            lines.append("## Call Chain Analysis")
            lines.append("")
            
            for chain in context.call_chains[:15]:
                lines.append(f"**`{chain['procedure']}`** (depth {chain['depth']})")
                if chain['entry_point']:
                    lines.append("- ⭐ Entry point")
                if chain['callers']:
                    lines.append(f"- Called by: {', '.join(chain['callers'][:5])}")
                if chain['callees']:
                    lines.append(f"- Calls: {', '.join(chain['callees'][:5])}")
                lines.append("")
        
        # Dependencies
        if context.dependencies:
            lines.append("## External Dependencies")
            lines.append("")
            
            for dep in context.dependencies:
                status = "✓ Resolved" if dep.get('resolved') else "⚠ Unresolved"
                lines.append(f"- `{dep['name']}` - {status}")
            lines.append("")
        
        # Migration hints
        lines.append(f"## Migration Hints for {context.target_language}")
        lines.append("")
        
        for hint in context.migration_hints:
            lines.append(f"- {hint}")
        lines.append("")
        
        return '\n'.join(lines)
    
    def _format_json(self, context: CodeRewritingContext) -> str:
        """Format context as JSON"""
        data = {
            'search_term': context.search_term,
            'target_language': context.target_language,
            'source_procedures': context.source_procedures,
            'data_structures': context.data_structures,
            'call_chains': context.call_chains,
            'dependencies': context.dependencies,
            'business_logic_summary': context.business_logic_summary,
            'migration_hints': context.migration_hints,
            'code_snippets': context.code_snippets
        }
        return json.dumps(data, indent=2)
    
    def _format_xml(self, context: CodeRewritingContext) -> str:
        """Format context as XML"""
        lines = ['<?xml version="1.0" encoding="UTF-8"?>']
        lines.append(f'<migration_context search_term="{context.search_term}" target_language="{context.target_language}">')
        
        # Procedures
        lines.append('  <procedures>')
        for proc in context.source_procedures:
            lines.append(f'    <procedure name="{proc["name"]}" file="{proc["file"]}" line="{proc["line"]}">')
            lines.append(f'      <return_type>{proc["return_type"]}</return_type>')
            lines.append(f'      <parameter_count>{proc["parameter_count"]}</parameter_count>')
            lines.append(f'      <statement_count>{proc["statement_count"]}</statement_count>')
            lines.append('    </procedure>')
        lines.append('  </procedures>')
        
        lines.append('</migration_context>')
        return '\n'.join(lines)


# ============================================================================
# Convenience Functions
# ============================================================================

def create_rewriting_context(kg: KnowledgeGraph,
                            search_term: str,
                            target_language: str = "Java",
                            output_file: Optional[str] = None) -> str:
    """
    Quick function to create and format rewriting context
    
    Args:
        kg: Knowledge graph
        search_term: Functionality to search for
        target_language: Target language (Java, Python, etc.)
        output_file: Optional file to save context
    
    Returns:
        Formatted context string
    """
    builder = CodeRewritingContextBuilder(kg)
    context = builder.build_rewriting_context(search_term, target_language)
    formatted = builder.format_for_llm(context, format="markdown")
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(formatted)
        print(f"Context saved to: {output_file}")
    
    return formatted


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║         Enhanced LLM Context Builder for Code Rewriting              ║
╚══════════════════════════════════════════════════════════════════════╝

Generate comprehensive context for LLM-assisted code migration.

USAGE:
  from code_rewriting_context import create_rewriting_context
  from knowledge_graph import KnowledgeGraph
  
  kg = KnowledgeGraph()
  # ... populate kg ...
  
  context = create_rewriting_context(
      kg,
      search_term="drawdown",
      target_language="Java",
      output_file="migration_context.md"
  )

FEATURES:
  • Procedure signatures and metadata
  • Actual source code snippets
  • Call chain analysis
  • Data structure definitions
  • External dependencies
  • Language-specific migration hints
  • Business logic summary

TARGET LANGUAGES:
  • Java (with specific patterns and hints)
  • Python
  • Generic (extensible to other languages)
    """)
