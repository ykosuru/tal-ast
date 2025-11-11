"""
Hybrid Modernization System
Uses YOUR Knowledge Graph for analysis, Aider/Cursor/OpenHands for code generation

Architecture:
    Your KG (analysis) → Context Builder → Coding Agent (generation)
"""

import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass
import sys

# Import your existing KG
# Adjust the import path based on your project structure
try:
    from knowledge_graph import KnowledgeGraph, Entity, EntityType, RelationType
except ImportError:
    print("Note: Import your knowledge_graph module")
    # Mock for standalone usage
    class KnowledgeGraph: pass
    class EntityType: pass
    class RelationType: pass


@dataclass
class ModernizationContext:
    """Rich context extracted from KG for agent consumption"""
    procedure_name: str
    procedure_id: str
    source_code: str
    language: str
    file_path: str
    
    # From KG analysis
    direct_calls: List[str]
    called_by: List[str]
    call_depth: int
    coupling_score: float
    data_dependencies: List[str]
    
    # Risk factors
    warnings: List[str]
    recommendation: str
    
    # Business context (from RAG if available)
    business_context: Optional[str] = None
    domain_docs: Optional[str] = None


class ContextBuilder:
    """
    Builds rich context from your Knowledge Graph for coding agents
    This is the KEY component - uses your KG for intelligence
    """
    
    def __init__(self, kg: KnowledgeGraph):
        """
        Args:
            kg: Your existing KnowledgeGraph instance
        """
        self.kg = kg
    
    def build_context(
        self, 
        procedure_name: str,
        rag_system: Optional[Any] = None
    ) -> ModernizationContext:
        """
        Extract comprehensive context for a procedure
        
        Args:
            procedure_name: Name of procedure to modernize
            rag_system: Optional RAG system for business docs
            
        Returns:
            ModernizationContext with all relevant information
        """
        # Find procedure entity
        procedures = self.kg.query_entities(entity_type=EntityType.PROCEDURE)
        entity = next((p for p in procedures if p.name == procedure_name), None)
        
        if not entity:
            raise ValueError(f"Procedure '{procedure_name}' not found in KG")
        
        # Get direct calls (what this procedure calls)
        callees = self.kg.get_neighbors(
            entity.id,
            rel_type=RelationType.CALLS,
            direction="outgoing"
        )
        direct_calls = [c.name for c in callees]
        
        # Get callers (who calls this procedure)
        callers = self.kg.get_neighbors(
            entity.id,
            rel_type=RelationType.CALLS,
            direction="incoming"
        )
        called_by = [c.name for c in callers]
        
        # Get data dependencies
        data_deps = []
        for rel_type in [RelationType.REFERENCES, RelationType.USES]:
            deps = self.kg.get_neighbors(
                entity.id,
                rel_type=rel_type,
                direction="outgoing"
            )
            data_deps.extend([d.name for d in deps])
        
        # Compute coupling score
        coupling_score = self._compute_coupling(entity.id)
        
        # Compute call depth
        call_depth = self._compute_call_depth(entity.id)
        
        # Generate warnings
        warnings = self._generate_warnings(
            len(direct_calls),
            len(called_by),
            coupling_score,
            len(data_deps)
        )
        
        # Get recommendation
        recommendation = self._get_recommendation(coupling_score)
        
        # Query RAG for business context (if available)
        business_context = None
        domain_docs = None
        if rag_system:
            business_context = self._query_rag(rag_system, procedure_name, entity)
            domain_docs = self._extract_domain_docs(rag_system, entity)
        
        return ModernizationContext(
            procedure_name=entity.name,
            procedure_id=entity.id,
            source_code=entity.metadata.get('source_code', ''),
            language=entity.language or 'TAL',
            file_path=entity.file_path or '',
            direct_calls=direct_calls,
            called_by=called_by,
            call_depth=call_depth,
            coupling_score=coupling_score,
            data_dependencies=data_deps,
            warnings=warnings,
            recommendation=recommendation,
            business_context=business_context,
            domain_docs=domain_docs
        )
    
    def _compute_coupling(self, entity_id: str) -> float:
        """Compute coupling score (0-1)"""
        outgoing = len(self.kg.get_neighbors(
            entity_id,
            rel_type=RelationType.CALLS,
            direction="outgoing"
        ))
        incoming = len(self.kg.get_neighbors(
            entity_id,
            rel_type=RelationType.CALLS,
            direction="incoming"
        ))
        total = outgoing + incoming
        return min(total / 20.0, 1.0)  # Normalize to 0-1
    
    def _compute_call_depth(self, entity_id: str, max_depth: int = 5) -> int:
        """Compute maximum call depth from this procedure"""
        visited = set()
        
        def dfs_depth(current_id: str, depth: int) -> int:
            if depth >= max_depth or current_id in visited:
                return depth
            
            visited.add(current_id)
            callees = self.kg.get_neighbors(
                current_id,
                rel_type=RelationType.CALLS,
                direction="outgoing"
            )
            
            if not callees:
                return depth
            
            max_child_depth = depth
            for callee in callees:
                child_depth = dfs_depth(callee.id, depth + 1)
                max_child_depth = max(max_child_depth, child_depth)
            
            return max_child_depth
        
        return dfs_depth(entity_id, 0)
    
    def _generate_warnings(
        self, 
        num_calls: int, 
        num_callers: int, 
        coupling: float,
        num_data_deps: int
    ) -> List[str]:
        """Generate risk warnings"""
        warnings = []
        
        if coupling > 0.7:
            warnings.append(
                f"⚠️  HIGH COUPLING (score: {coupling:.2f}). "
                "Consider modernizing dependent procedures together."
            )
        
        if num_calls > 15:
            warnings.append(
                f"⚠️  HIGH FAN-OUT: Calls {num_calls} procedures. "
                "Ensure all dependencies modernized or have compatibility layers."
            )
        
        if num_callers > 15:
            warnings.append(
                f"⚠️  HIGH FAN-IN: Called by {num_callers} procedures. "
                "Interface changes will impact many callers."
            )
        
        if num_data_deps > 8:
            warnings.append(
                f"⚠️  SHARED DATA: Accesses {num_data_deps} data structures. "
                "Data migration strategy required."
            )
        
        return warnings
    
    def _get_recommendation(self, coupling: float) -> str:
        """Get modernization recommendation"""
        if coupling < 0.3:
            return "LOW COUPLING: Good candidate for independent modernization."
        elif coupling < 0.7:
            return "MODERATE COUPLING: Modernize incrementally with interface management."
        else:
            return "HIGH COUPLING: Modernize as part of a group to avoid interface issues."
    
    def _query_rag(self, rag_system: Any, proc_name: str, entity: Entity) -> str:
        """Query RAG system for business context"""
        try:
            # Extract business terms from procedure
            terms = [proc_name]
            if 'payment' in proc_name.lower():
                terms.append('payment processing')
            if 'wire' in proc_name.lower():
                terms.append('wire transfer')
            
            # Query RAG
            query = ' '.join(terms)
            results = rag_system.search(query, top_k=3)
            
            # Combine results
            context_parts = []
            for result in results:
                context_parts.append(f"From {result.source}:")
                context_parts.append(result.content[:300])
                context_parts.append("")
            
            return '\n'.join(context_parts)
        except:
            return None
    
    def _extract_domain_docs(self, rag_system: Any, entity: Entity) -> str:
        """Extract domain-specific documentation"""
        # Similar to _query_rag but focused on domain concepts
        return None


class AgentPromptGenerator:
    """
    Generates rich prompts for coding agents (Aider/Cursor/etc.)
    Uses context from your KG
    """
    
    def generate_context_file(
        self,
        context: ModernizationContext,
        target_language: str = "Python",
        output_path: Path = Path("./agent_context")
    ) -> Path:
        """
        Generate comprehensive context file for coding agent
        
        Args:
            context: Context from KG
            target_language: Target language
            output_path: Where to save context files
            
        Returns:
            Path to generated context file
        """
        output_path.mkdir(exist_ok=True, parents=True)
        
        context_file = output_path / f"{context.procedure_name}_context.md"
        
        content = self._build_context_markdown(context, target_language)
        
        with open(context_file, 'w') as f:
            f.write(content)
        
        return context_file
    
    def _build_context_markdown(
        self,
        context: ModernizationContext,
        target_language: str
    ) -> str:
        """Build rich markdown context for agent"""
        
        sections = []
        
        # Header
        sections.append(f"# Modernization Context: {context.procedure_name}")
        sections.append(f"\n**Task**: Translate from {context.language} to {target_language}")
        sections.append("")
        
        # Source code
        sections.append("## Source Code")
        sections.append(f"```{context.language}")
        sections.append(context.source_code)
        sections.append("```")
        sections.append("")
        
        # Dependency graph
        sections.append("## Dependency Graph (from Knowledge Graph)")
        sections.append("")
        sections.append(f"**This procedure calls** ({len(context.direct_calls)} procedures):")
        if context.direct_calls:
            for call in context.direct_calls[:15]:
                sections.append(f"  - `{call}`")
            if len(context.direct_calls) > 15:
                sections.append(f"  - ... and {len(context.direct_calls) - 15} more")
        else:
            sections.append("  - (none - leaf procedure)")
        sections.append("")
        
        sections.append(f"**Called by** ({len(context.called_by)} procedures):")
        if context.called_by:
            for caller in context.called_by[:15]:
                sections.append(f"  - `{caller}`")
            if len(context.called_by) > 15:
                sections.append(f"  - ... and {len(context.called_by) - 15} more")
        else:
            sections.append("  - (none - entry point)")
        sections.append("")
        
        # Data dependencies
        if context.data_dependencies:
            sections.append(f"**Data Dependencies** ({len(context.data_dependencies)} items):")
            for dep in context.data_dependencies[:10]:
                sections.append(f"  - `{dep}`")
            if len(context.data_dependencies) > 10:
                sections.append(f"  - ... and {len(context.data_dependencies) - 10} more")
            sections.append("")
        
        # Call depth
        sections.append("## Call Depth Analysis")
        sections.append(f"- **Max call depth**: {context.call_depth} levels")
        sections.append(f"- **Coupling score**: {context.coupling_score:.2f} (0=loose, 1=tight)")
        sections.append("")
        
        # Risk factors
        if context.warnings:
            sections.append("## ⚠️  Risk Factors")
            for warning in context.warnings:
                sections.append(f"{warning}")
            sections.append("")
        
        # Recommendation
        sections.append("## Recommendation")
        sections.append(context.recommendation)
        sections.append("")
        
        # Business context from RAG
        if context.business_context:
            sections.append("## Business Documentation (from RAG)")
            sections.append(context.business_context)
            sections.append("")
        
        if context.domain_docs:
            sections.append("## Domain Knowledge")
            sections.append(context.domain_docs)
            sections.append("")
        
        # Requirements
        sections.append("## Modernization Requirements")
        sections.append("")
        sections.append(f"Generate modern {target_language} code that:")
        sections.append("")
        sections.append("1. **Preserves exact business logic**")
        sections.append("   - Every condition, calculation, and side effect must be maintained")
        sections.append("   - All edge cases must be handled identically")
        sections.append("")
        sections.append("2. **Maintains interface compatibility**")
        sections.append("   - Same inputs and outputs")
        sections.append("   - Called procedures must remain compatible")
        sections.append("")
        sections.append("3. **Uses modern patterns**")
        sections.append(f"   - Idiomatic {target_language} code")
        sections.append("   - Type hints/annotations")
        sections.append("   - Proper error handling")
        sections.append("   - Clear documentation")
        sections.append("")
        sections.append("4. **Handles dependencies**")
        if context.direct_calls:
            sections.append(f"   - Calls to {len(context.direct_calls)} procedures must be handled")
            sections.append("   - Use stubs/mocks if dependencies not yet modernized")
        sections.append("")
        sections.append("5. **Respects data dependencies**")
        if context.data_dependencies:
            sections.append(f"   - Access to {len(context.data_dependencies)} data structures")
            sections.append("   - Maintain data integrity")
        sections.append("")
        
        # Special considerations for payment processing
        if any(term in context.procedure_name.lower() 
               for term in ['payment', 'wire', 'transfer', 'ofac', 'transaction']):
            sections.append("## 🔒 Payment Processing Specific")
            sections.append("")
            sections.append("**CRITICAL:**")
            sections.append("- Maintain transaction boundaries")
            sections.append("- Preserve audit trail functionality")
            sections.append("- Keep compliance checks (OFAC, AML, etc.)")
            sections.append("- Ensure data validation is comprehensive")
            sections.append("- Error handling must be robust")
            sections.append("")
        
        # Output format
        sections.append("## Expected Output")
        sections.append("")
        sections.append("Provide:")
        sections.append(f"1. Complete {target_language} implementation")
        sections.append("2. Docstring explaining business logic")
        sections.append("3. Comments on key decisions")
        sections.append("4. Unit test suggestions")
        sections.append("")
        
        return '\n'.join(sections)
    
    def generate_short_prompt(self, context: ModernizationContext, target_language: str) -> str:
        """Generate a short prompt for command-line use"""
        return f"""Modernize {context.language} procedure to {target_language}.

Procedure: {context.procedure_name}
Calls: {len(context.direct_calls)} procedures
Called by: {len(context.called_by)} procedures
Coupling: {context.coupling_score:.2f}

{context.warnings[0] if context.warnings else ''}

See {context.procedure_name}_context.md for full requirements.

Requirements:
- Preserve exact business logic
- Use {target_language} best practices
- Add type hints and documentation
- Handle all edge cases
"""


class HybridModernizer:
    """
    Main orchestrator: Your KG for analysis, Aider/Cursor for generation
    """
    
    def __init__(
        self,
        kg: KnowledgeGraph,
        agent: str = "aider",
        rag_system: Optional[Any] = None
    ):
        """
        Args:
            kg: Your KnowledgeGraph instance
            agent: Coding agent to use ('aider', 'cursor', or command)
            rag_system: Optional RAG system
        """
        self.kg = kg
        self.agent = agent
        self.rag_system = rag_system
        self.context_builder = ContextBuilder(kg)
        self.prompt_generator = AgentPromptGenerator()
    
    def modernize_procedure(
        self,
        procedure_name: str,
        target_language: str = "Python",
        output_dir: Path = Path("./modernized"),
        context_dir: Path = Path("./agent_context")
    ) -> Dict[str, Any]:
        """
        Modernize a single procedure using hybrid approach
        
        Args:
            procedure_name: Name of procedure to modernize
            target_language: Target language
            output_dir: Where to save modernized code
            context_dir: Where to save context files
            
        Returns:
            Dict with results and paths
        """
        print(f"\n{'='*70}")
        print(f"MODERNIZING: {procedure_name}")
        print(f"{'='*70}\n")
        
        # Step 1: Extract context from KG
        print("📊 Extracting context from Knowledge Graph...")
        context = self.context_builder.build_context(
            procedure_name,
            self.rag_system
        )
        
        print(f"  ✓ Calls: {len(context.direct_calls)} procedures")
        print(f"  ✓ Called by: {len(context.called_by)} procedures")
        print(f"  ✓ Coupling: {context.coupling_score:.2f}")
        print(f"  ✓ Call depth: {context.call_depth}")
        
        if context.warnings:
            print(f"\n⚠️  Warnings:")
            for warning in context.warnings:
                print(f"  {warning}")
        
        # Step 2: Generate context file for agent
        print(f"\n📝 Generating context file...")
        context_file = self.prompt_generator.generate_context_file(
            context,
            target_language,
            context_dir
        )
        print(f"  ✓ Saved to: {context_file}")
        
        # Step 3: Save source code to file
        output_dir.mkdir(exist_ok=True, parents=True)
        source_ext = self._get_extension(context.language)
        source_file = output_dir / f"{procedure_name}{source_ext}"
        
        with open(source_file, 'w') as f:
            f.write(context.source_code)
        
        print(f"  ✓ Source saved to: {source_file}")
        
        # Step 4: Call coding agent
        print(f"\n🤖 Calling {self.agent} for code generation...")
        
        result = self._call_agent(
            context,
            target_language,
            source_file,
            context_file,
            output_dir
        )
        
        return result
    
    def modernize_subsystem(
        self,
        entry_procedures: List[str],
        target_language: str = "Python",
        exclude_utilities: Optional[Set[str]] = None,
        max_depth: int = 10,
        output_dir: Path = Path("./modernized"),
        context_dir: Path = Path("./agent_context")
    ) -> Dict[str, Any]:
        """
        Modernize entire subsystem using KG subgraph extraction + agent
        
        Args:
            entry_procedures: Entry point procedure names
            target_language: Target language
            exclude_utilities: Common utilities to exclude
            max_depth: Max call depth
            output_dir: Output directory
            context_dir: Context directory
            
        Returns:
            Dict with results
        """
        print(f"\n{'='*70}")
        print(f"SUBSYSTEM MODERNIZATION")
        print(f"{'='*70}")
        print(f"Entry points: {', '.join(entry_procedures)}")
        print(f"{'='*70}\n")
        
        # Step 1: Use YOUR excellent extract_call_subgraph
        print("🔍 Extracting call subgraph from Knowledge Graph...")
        subgraph = self.kg.extract_call_subgraph(
            entry_procedures=entry_procedures,
            max_depth=max_depth,
            exclude_procedures=exclude_utilities or set(),
            include_variables=True,
            include_structures=True
        )
        
        procedures = [e for e in subgraph['entities'] if e.type == EntityType.PROCEDURE]
        print(f"  ✓ Found {len(procedures)} procedures")
        print(f"  ✓ Total entities: {subgraph['statistics']['total_entities']}")
        
        # Step 2: Sequence by coupling (low coupling first)
        print(f"\n📋 Sequencing procedures by coupling...")
        sequenced_procedures = self._sequence_by_coupling(procedures)
        
        for i, (proc, coupling) in enumerate(sequenced_procedures[:10], 1):
            print(f"  {i}. {proc.name:40} (coupling: {coupling:.2f})")
        if len(sequenced_procedures) > 10:
            print(f"  ... and {len(sequenced_procedures) - 10} more")
        
        # Step 3: Modernize each procedure
        print(f"\n🔄 Modernizing {len(sequenced_procedures)} procedures...\n")
        
        results = []
        for i, (proc_entity, coupling) in enumerate(sequenced_procedures, 1):
            print(f"\n[{i}/{len(sequenced_procedures)}] {proc_entity.name}")
            print(f"  Coupling: {coupling:.2f}")
            
            try:
                result = self.modernize_procedure(
                    proc_entity.name,
                    target_language,
                    output_dir,
                    context_dir
                )
                results.append({
                    'procedure': proc_entity.name,
                    'status': 'success',
                    'result': result
                })
            except Exception as e:
                print(f"  ❌ Error: {e}")
                results.append({
                    'procedure': proc_entity.name,
                    'status': 'failed',
                    'error': str(e)
                })
        
        # Summary
        successes = sum(1 for r in results if r['status'] == 'success')
        failures = len(results) - successes
        
        print(f"\n{'='*70}")
        print("SUBSYSTEM MODERNIZATION COMPLETE")
        print(f"{'='*70}")
        print(f"  Total procedures: {len(results)}")
        print(f"  Successful: {successes}")
        print(f"  Failed: {failures}")
        print(f"{'='*70}\n")
        
        return {
            'subgraph': subgraph,
            'results': results,
            'summary': {
                'total': len(results),
                'success': successes,
                'failed': failures
            }
        }
    
    def _sequence_by_coupling(self, procedures: List[Entity]) -> List[tuple]:
        """Sequence procedures by coupling score (low first)"""
        scored = []
        for proc in procedures:
            coupling = self.context_builder._compute_coupling(proc.id)
            scored.append((proc, coupling))
        
        # Sort by coupling (low first = easier to modernize)
        scored.sort(key=lambda x: x[1])
        return scored
    
    def _get_extension(self, language: str) -> str:
        """Get file extension for language"""
        extensions = {
            'TAL': '.tal',
            'COBOL': '.cbl',
            'Python': '.py',
            'Java': '.java',
            'TypeScript': '.ts'
        }
        return extensions.get(language, '.txt')
    
    def _call_agent(
        self,
        context: ModernizationContext,
        target_language: str,
        source_file: Path,
        context_file: Path,
        output_dir: Path
    ) -> Dict[str, Any]:
        """
        Call the coding agent (Aider/Cursor/etc.)
        
        Returns:
            Dict with agent results
        """
        if self.agent == "aider":
            return self._call_aider(context, target_language, source_file, context_file, output_dir)
        elif self.agent == "cursor":
            return self._call_cursor(context, target_language, source_file, context_file)
        else:
            return self._call_custom(context, target_language, source_file, context_file)
    
    def _call_aider(
        self,
        context: ModernizationContext,
        target_language: str,
        source_file: Path,
        context_file: Path,
        output_dir: Path
    ) -> Dict[str, Any]:
        """Call Aider with context"""
        
        # Generate short prompt
        prompt = self.prompt_generator.generate_short_prompt(context, target_language)
        
        # Aider command
        cmd = [
            "aider",
            "--yes",  # Auto-accept
            "--message", prompt,
            str(source_file),
            str(context_file)
        ]
        
        print(f"  Running: {' '.join(cmd[:3])}...")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=output_dir
            )
            
            return {
                'agent': 'aider',
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'source_file': str(source_file),
                'context_file': str(context_file)
            }
        except FileNotFoundError:
            print("\n  ⚠️  Aider not installed. Install with: pip install aider-chat")
            print("  ℹ️  Or run manually:")
            print(f"     aider --message '{prompt}' {source_file}")
            
            return {
                'agent': 'aider',
                'success': False,
                'error': 'Aider not installed',
                'manual_command': f"aider --message '{prompt}' {source_file}"
            }
    
    def _call_cursor(
        self,
        context: ModernizationContext,
        target_language: str,
        source_file: Path,
        context_file: Path
    ) -> Dict[str, Any]:
        """Instructions for Cursor"""
        print("\n  ℹ️  Open in Cursor IDE:")
        print(f"     1. Open file: {source_file}")
        print(f"     2. Open context: {context_file}")
        print(f"     3. Ask Cursor: 'Modernize this {context.language} to {target_language} following context.md requirements'")
        
        return {
            'agent': 'cursor',
            'success': True,
            'instructions': f"Open {source_file} and {context_file} in Cursor",
            'source_file': str(source_file),
            'context_file': str(context_file)
        }
    
    def _call_custom(
        self,
        context: ModernizationContext,
        target_language: str,
        source_file: Path,
        context_file: Path
    ) -> Dict[str, Any]:
        """Instructions for custom agent"""
        print(f"\n  ℹ️  Manual modernization:")
        print(f"     1. Source code: {source_file}")
        print(f"     2. Full context: {context_file}")
        print(f"     3. Use your preferred coding agent")
        
        return {
            'agent': 'custom',
            'success': True,
            'source_file': str(source_file),
            'context_file': str(context_file)
        }


def main():
    """Example usage"""
    print("""
Hybrid Modernization System
Uses your Knowledge Graph for analysis, Aider/Cursor for generation

Usage:
    from hybrid_modernizer import HybridModernizer
    from knowledge_graph import KnowledgeGraph
    
    # Initialize your KG
    kg = KnowledgeGraph(backend="networkx")
    kg.load_from_json("tal_system.json")
    
    # Create hybrid modernizer
    modernizer = HybridModernizer(kg, agent="aider")
    
    # Modernize single procedure
    result = modernizer.modernize_procedure("PROCESS_WIRE_TRANSFER", "Python")
    
    # Or modernize subsystem
    result = modernizer.modernize_subsystem(
        entry_procedures=["PROCESS_WIRE_TRANSFER"],
        exclude_utilities={"LOG_MESSAGE", "FORMAT_DATE"}
    )
""")


if __name__ == "__main__":
    main()
