#!/usr/bin/env python3
"""
TAL to Java Translation Assistant using LLM + RAG
Now with comprehensive prompt generation for complete translations

This program:
1. Uses knowledge graph to find functionality (e.g., drawdown) TAL code
2. Gathers all relevant context (procedures, data structures, dependencies)
3. Reads actual TAL source code
4. Builds comprehensive LLM context
5. Generates comprehensive prompts ensuring complete line-by-line translation
6. Optional direct LLM API integration

Features:
- Automatic code discovery using knowledge graph search
- Full dependency tracking with call graph analysis
- Data structure extraction
- Built-in diagnostics for missing procedures
- Comprehensive 4-phase prompt methodology
- Financial code precision requirements
- Verification checklists

Usage:
    python tal_to_java_translator.py <tal_source_dir> <functionality> [options]
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Set, Optional
import re
import tal_proc_parser

# Import our existing tools
from knowledge_graph import KnowledgeGraph, EntityType, RelationType
from parsers import (
    parse_tal_directory_recursive,
    KnowledgeGraphSearch,
    resolve_external_references
)

# Import the comprehensive prompt generator
from comprehensive_prompt_generator import ComprehensivePromptGenerator


class TALCodeExtractor:
    """Extract actual TAL source code from files using tal_proc_parser for accuracy."""
    
    def __init__(self, tal_source_dir: str):
        self.source_dir = Path(tal_source_dir)
        self.file_cache = {}
        self.proc_boundaries_cache = {}
    
    def _get_file_contents(self, file_path: Path) -> Optional[List[str]]:
        """Get file contents with caching."""
        file_path_str = str(file_path)
        
        if file_path_str in self.file_cache:
            return self.file_cache[file_path_str]
        
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            self.file_cache[file_path_str] = lines
            return lines
        except Exception as e:
            print(f"Warning: Error reading file {file_path}: {e}")
            return None
    
    def _get_procedure_boundaries(self, file_path: Path) -> Dict[str, Dict[str, int]]:
        """
        Get all procedure boundaries in a file using tal_proc_parser.
        
        This uses tal_proc_parser.find_procedure_declarations() to get accurate
        procedure start positions, then calculates where each procedure ends.
        """
        file_path_str = str(file_path)
        
        # Check cache first
        if file_path_str in self.proc_boundaries_cache:
            return self.proc_boundaries_cache[file_path_str]
        
        lines = self._get_file_contents(file_path)
        if not lines:
            return {}
        
        # Use tal_proc_parser to find all procedure declarations
        content = ''.join(lines)
        proc_declarations = tal_proc_parser.find_procedure_declarations(content)
        
        print(f"proc_declarations from tal_proc_parser {proc_declarations}")
        
        boundaries = {}
        for i, (start_line, proc_name, declaration) in enumerate(proc_declarations):
            # Calculate how many lines the declaration spans
            declaration_line_count = declaration.count('\n') + 1
            declaration_end_line = start_line + declaration_line_count - 1
            
            # Find where this procedure ends
            # It ends at the start of the next procedure, or at end of file
            if i + 1 < len(proc_declarations):
                end_line = proc_declarations[i + 1][0] - 1
            else:
                end_line = len(lines)
            
            # Store boundaries (case-insensitive lookup)
            boundaries[proc_name.upper()] = {
                'start': start_line,
                'declaration_end': declaration_end_line,
                'end': end_line,
                'original_name': proc_name
            }
        
        # Cache the results
        self.proc_boundaries_cache[file_path_str] = boundaries
        return boundaries
    
    def find_missing_procedure(self, name: str):
        """Find and display detailed info about a missing procedure using repair_graph."""
        import repair_graph
        graph = repair_graph.load_graph('./output/knowledge_graph.json')
        matches = repair_graph.find_procedure_in_graph(name, graph)
        
        if not matches:
            print(f"✗ No procedure found matching '{name}!'")
        else:
            print(f"✓ Found {len(matches)} match(es):\n")
            for i, proc in enumerate(matches, 1):
                print(f"{i}. {proc['name']}")
                print(f"      ID: {proc['id']}")
                print(f"      Qualified name: {proc.get('qualified_name', 'N/A')}")
                print(f"      File: {proc.get('file_path', 'MISSING')}")
                print(f"      Line: {proc.get('start_line', 'N/A')}")
                
                metadata = proc.get('metadata', {})
                if metadata.get('is_external'):
                    print(f"      Status: EXTERNAL REFERENCE")
                
                # Show callers
                callers = []
                for edge in graph['edges']:
                    if edge.get('type') == 'calls':
                        target_id = edge.get('target') or edge.get('target_id')
                        if target_id == proc['id']:
                            source_id = edge.get('source') or edge.get('source_id')
                            node_by_id = {n['id']: n for n in graph['nodes']}
                            caller = node_by_id.get(source_id)
                            if caller:
                                callers.append(caller['name'])
                
                if callers:
                    print(f"      Called by: {', '.join(callers[:5])}")
                    if len(callers) > 5:
                        print(f"      ... and {len(callers) - 5} more")
                print()
    
    def extract_procedure_code(self, entity) -> Optional[str]:
        """Extract the actual source code for a procedure."""
        if not entity.file_path:
            print(f"Warning: No file_path for entity {entity.name}")
            self.find_missing_procedure(entity.name)
            return None
        
        file_path = Path(entity.file_path)
        
        # NOTE: Commented out to use paths as-is from knowledge graph
        # Your version uses paths directly without making them absolute
        #if not file_path.is_absolute():
        #    file_path = self.source_dir / file_path
        
        lines = self._get_file_contents(file_path)
        if not lines:
            print(f"Warning: Could not read file {file_path}")
            return None
        
        boundaries = self._get_procedure_boundaries(file_path)
        
        proc_name_upper = entity.name.upper()
        
        # Primary path: Use pre-calculated boundaries (RELIABLE)
        if proc_name_upper in boundaries:
            boundary = boundaries[proc_name_upper]
            start = boundary['start'] - 1  # Convert to 0-indexed
            end = boundary['end']
            
            code = ''.join(lines[start:end])
            return code
        
        # Fallback path: Calculate boundaries manually (WITH PROPER BEGIN/END TRACKING)
        elif entity.start_line:
            print(f"Warning: Using fallback extraction for {entity.name}")
            start = entity.start_line - 1
            
            if entity.end_line and entity.end_line > entity.start_line:
                end = entity.end_line
            else:
                # FIXED: Track BEGIN/END nesting to find procedure end
                end = start + 1
                begin_count = 0
                end_count = 0
                found_begin = False
                
                for i in range(start, min(len(lines), start + 1000)):
                    line = lines[i].strip().upper()
                    
                    # Skip comments
                    comment_pos = line.find('!')
                    if comment_pos >= 0:
                        line = line[:comment_pos].strip()
                    
                    # Count BEGIN keywords
                    if 'BEGIN' in line:
                        begin_count += line.count('BEGIN')
                        found_begin = True
                    
                    # Count END keywords (as whole words only)
                    if re.search(r'\bEND\b', line):
                        end_count += 1
                    
                    # Stop at next PROC (different procedure)
                    if i > start and re.search(r'\bPROC\b', line):
                        end = i
                        break
                    
                    # If we found BEGIN and all BEGINs are closed
                    if found_begin and begin_count > 0 and begin_count == end_count:
                        end = i + 1
                        break
                    
                    # If no BEGIN found but we hit END, assume simple procedure
                    if not found_begin and end_count > 0:
                        end = i + 1
                        break
                
                # Safety fallback
                if end == start + 1:
                    end = min(start + 200, len(lines))
            
            code = ''.join(lines[start:end])
            return code
        
        print(f"Warning: Could not find procedure {entity.name} in {file_path}")
        return None
    
    def extract_structure_code(self, entity) -> Optional[str]:
        """Extract structure definition from source file."""
        if not entity.file_path:
            return None
        
        file_path = Path(entity.file_path)
        
        # NOTE: Commented out to use paths as-is from knowledge graph
        #if not file_path.is_absolute():
        #    file_path = self.source_dir / file_path
        
        lines = self._get_file_contents(file_path)
        if not lines:
            return None
        
        if not entity.start_line:
            return None
        
        start = entity.start_line - 1
        
        # For structures, look for the closing pattern
        # STRUCT can have BEGIN...END blocks or simple semicolon termination
        end = start + 1
        found_begin = False
        begin_count = 0
        end_count = 0
        
        for i in range(start, min(len(lines), start + 200)):
            line = lines[i].strip().upper()
            
            # Track BEGIN/END for nested structures
            if 'BEGIN' in line:
                found_begin = True
                begin_count += line.count('BEGIN')
            if 'END' in line:
                end_count += line.count('END')
            
            # If we have nested BEGIN/END, wait for all to close
            if found_begin:
                if begin_count > 0 and begin_count == end_count:
                    end = i + 1
                    break
            # Otherwise look for semicolon
            elif line.endswith(';'):
                end = i + 1
                break
        
        code = ''.join(lines[start:end])
        return code
    
    def extract_file_section(self, file_path: str, start_line: int, end_line: int) -> Optional[str]:
        """Extract a section of a file by line numbers."""
        path = Path(file_path)
        
        # NOTE: Commented out to use paths as-is from knowledge graph
        #if not path.is_absolute():
        #    path = self.source_dir / path
        
        lines = self._get_file_contents(path)
        if not lines:
            return None
        
        start = max(0, start_line - 1)
        end = min(len(lines), end_line)
        
        return ''.join(lines[start:end])


class TranslationContextBuilder:
    """Build comprehensive context for LLM translation"""
    
    def __init__(self, kg: KnowledgeGraph, code_extractor: TALCodeExtractor):
        self.kg = kg
        self.extractor = code_extractor
        self.search = KnowledgeGraphSearch(kg)
    
    def build_context_for_functionality(self, 
                                       functionality: str, 
                                       depth: int = 2) -> Dict[str, Any]:
        """
        Build comprehensive translation context for a functionality.
        
        Args:
            functionality: Search term (e.g., "drawdown")
            depth: How many levels of dependencies to include
        
        Returns:
            Dict with all context needed for translation
        """
        print(f"\n{'='*70}")
        print(f"BUILDING TRANSLATION CONTEXT: '{functionality}'")
        print(f"{'='*70}\n")
        
        # Step 1: Find all related entities
        print("Step 1: Searching knowledge graph...")
        search_results = self.search.find_by_functionality(functionality)
        
        primary_procedures = search_results.get('procedures', [])
        related_vars = search_results.get('variables', [])
        related_structs = search_results.get('structures', [])
        
        print(f"  Found {len(primary_procedures)} primary procedures")
        print(f"  Found {len(related_vars)} related variables")
        print(f"  Found {len(related_structs)} related structures")
        
        if not primary_procedures:
            print(f"\n⚠️  WARNING: No procedures found for '{functionality}'")
            print("    Try a different search term or check knowledge graph")
        
        # Step 2: Build dependency graph
        print(f"\nStep 2: Building dependency graph (depth={depth})...")
        all_procedures = set(primary_procedures)
        all_structures = set(related_structs)
        all_variables = set(related_vars)
        
        procedure_dependencies = {}
        
        # Expand by depth
        current_procs = list(primary_procedures)
        for d in range(depth):
            next_procs = []
            
            for proc in current_procs:
                if proc.name not in procedure_dependencies:
                    procedure_dependencies[proc.name] = {
                        'calls': [],
                        'called_by': [],
                        'uses_structures': [],
                        'uses_variables': []
                    }
                
                # Find what this proc calls
                callees = self.kg.get_neighbors(
                    proc.id,
                    rel_type=RelationType.CALLS,
                    direction="outgoing"
                )
                
                # Find what calls this proc
                callers = self.kg.get_neighbors(
                    proc.id,
                    rel_type=RelationType.CALLS,
                    direction="incoming"
                )
                
                # Find structures used
                uses_structs = self.kg.get_neighbors(
                    proc.id,
                    rel_type=RelationType.USES,
                    direction="outgoing"
                )
                
                # Find variables contained in this proc
                contained = self.kg.get_neighbors(
                    proc.id,
                    rel_type=RelationType.CONTAINS,
                    direction="outgoing"
                )
                
                for callee in callees:
                    if callee.type == EntityType.PROCEDURE:
                        procedure_dependencies[proc.name]['calls'].append(callee.name)
                        if callee not in all_procedures:
                            all_procedures.add(callee)
                            next_procs.append(callee)
                
                for caller in callers:
                    if caller.type == EntityType.PROCEDURE:
                        procedure_dependencies[proc.name]['called_by'].append(caller.name)
                        if caller not in all_procedures:
                            all_procedures.add(caller)
                            next_procs.append(caller)
                
                for struct in uses_structs:
                    if struct.type == EntityType.STRUCTURE:
                        procedure_dependencies[proc.name]['uses_structures'].append(struct.name)
                        all_structures.add(struct)
                
                for var in contained:
                    if var.type == EntityType.VARIABLE:
                        procedure_dependencies[proc.name]['uses_variables'].append(var.name)
                        all_variables.add(var)
            
            print(f"  Depth {d+1}: Added {len(next_procs)} procedures")
            current_procs = next_procs
        
        print(f"\nTotal context:")
        print(f"  Procedures: {len(all_procedures)}")
        print(f"  Variables: {len(all_variables)}")
        print(f"  Structures: {len(all_structures)}")
        
        # Step 3: Extract source code
        print(f"\nStep 3: Extracting source code...")
        procedures_with_code = []
        code_extraction_stats = {
            'success': 0,
            'failed': 0,
            'empty': 0,
            'total_chars': 0
        }
        
        for proc in all_procedures:
            code = self.extractor.extract_procedure_code(proc)
            code_length = len(code) if code else 0
            
            if code:
                code_extraction_stats['success'] += 1
                code_extraction_stats['total_chars'] += code_length
                
                if code_length < 10:
                    code_extraction_stats['empty'] += 1
                    print(f"  ⚠️  Very short code for {proc.name} ({code_length} chars)")
            else:
                code_extraction_stats['failed'] += 1
                print(f"  ✗ Failed to extract code for {proc.name}")
            
            procedures_with_code.append({
                'entity': proc,
                'code': code,
                'code_length': code_length,  # Required by comprehensive prompt generator
                'is_primary': proc in primary_procedures
            })
        
        print(f"\n  Code extraction summary:")
        print(f"    ✓ Success: {code_extraction_stats['success']}")
        print(f"    ✗ Failed: {code_extraction_stats['failed']}")
        print(f"    ⚠  Empty: {code_extraction_stats['empty']}")
        print(f"    Total characters: {code_extraction_stats['total_chars']:,}")
        
        structures_with_code = []
        for struct in all_structures:
            code = self.extractor.extract_structure_code(struct)
            structures_with_code.append({
                'entity': struct,
                'code': code
            })
        
        # Step 4: Build context dictionary (compatible with comprehensive prompt generator)
        context = {
            'functionality': functionality,
            'summary': {
                'primary_procedures': len(primary_procedures),
                'total_procedures': len(all_procedures),
                'total_variables': len(all_variables),
                'total_structures': len(all_structures),
                'depth': depth,
                'code_extraction': code_extraction_stats  # Required by comprehensive generator
            },
            'primary_procedures': [
                {
                    'name': p['entity'].name,
                    'file': p['entity'].file_path,
                    'line': p['entity'].start_line,
                    'parameters': p['entity'].metadata.get('parameters', []),
                    'return_type': p['entity'].metadata.get('return_type'),
                    'code': p['code'],
                    'code_length': p['code_length'],  # Required by comprehensive generator
                    'dependencies': procedure_dependencies.get(p['entity'].name, {})
                }
                for p in procedures_with_code if p['is_primary']
            ],
            'dependency_procedures': [
                {
                    'name': p['entity'].name,
                    'file': p['entity'].file_path,
                    'line': p['entity'].start_line,
                    'parameters': p['entity'].metadata.get('parameters', []),
                    'return_type': p['entity'].metadata.get('return_type'),
                    'code': p['code'],
                    'code_length': p['code_length'],  # Required by comprehensive generator
                    'is_external': p['entity'].metadata.get('is_external', False),
                    'dependencies': procedure_dependencies.get(p['entity'].name, {})
                }
                for p in procedures_with_code if not p['is_primary']
            ],
            'structures': [
                {
                    'name': s['entity'].name,
                    'fields': s['entity'].metadata.get('fields', []),
                    'code': s['code']
                }
                for s in structures_with_code
            ],
            'variables': [
                {
                    'name': v.name,
                    'type': v.metadata.get('data_type'),
                    'scope': v.metadata.get('scope')
                }
                for v in all_variables
            ],
            'call_graph': procedure_dependencies
        }
        
        # Validation
        primary_with_code = [p for p in context['primary_procedures'] if p['code']]
        if len(primary_with_code) < len(primary_procedures):
            print(f"\n  ⚠️  WARNING: Only {len(primary_with_code)}/{len(primary_procedures)} "
                  f"primary procedures have source code!")
            print("      Translation quality may be affected.")
        
        print(f"\n✓ Context built successfully")
        
        return context


class TALToJavaTranslator:
    """Main orchestrator for TAL to Java translation with comprehensive prompts"""
    
    def __init__(self, tal_source_dir: str, use_api: bool = False, api_key: Optional[str] = None):
        self.source_dir = tal_source_dir
        self.kg = None
        self.code_extractor = None
        self.context_builder = None
        self.use_api = use_api
        self.api_key = api_key
    
    def initialize(self):
        """Initialize knowledge graph by parsing TAL source"""
        print(f"\n{'='*70}")
        print("INITIALIZING TAL TO JAVA TRANSLATOR")
        print(f"{'='*70}\n")
        
        print("Step 1: Parsing TAL source code...")
        self.kg = KnowledgeGraph(backend="networkx")
        
        results = parse_tal_directory_recursive(
            self.source_dir,
            self.kg,
            recursive=True
        )
        
        if not results['success']:
            raise Exception(f"Failed to parse TAL code: {results.get('error')}")
        
        print(f"  ✓ Parsed {results.get('files_processed', 0)} files")
        
        print(f"\nStep 2: Resolving external references...")
        resolution = resolve_external_references(self.kg)
        print(f"  ✓ Resolved: {len(resolution['resolved'])}")
        if resolution['unresolved']:
            print(f"  ⚠  Unresolved: {len(resolution['unresolved'])}")
        
        print(f"\nStep 3: Initializing code extractor...")
        self.code_extractor = TALCodeExtractor(self.source_dir)
        print(f"  ✓ Code extractor ready")
        
        print(f"\nStep 4: Initializing context builder...")
        self.context_builder = TranslationContextBuilder(self.kg, self.code_extractor)
        print(f"  ✓ Context builder ready")
        
        stats = self.kg.get_statistics()
        print(f"\n✓ Initialization complete")
        print(f"  Total entities: {stats['total_entities']}")
        print(f"  Total relationships: {stats['total_relationships']}")
        print(f"  Procedures: {stats['entity_counts'].get('PROCEDURE', 0)}")
        print(f"  Structures: {stats['entity_counts'].get('STRUCTURE', 0)}")
        print(f"  Variables: {stats['entity_counts'].get('VARIABLE', 0)}")
    
    def translate_functionality(self, 
                               functionality: str,
                               output_dir: str = "./llm_context",
                               depth: int = 2,
                               strict_mode: bool = True,
                               financial_code: bool = True,
                               include_validation: bool = True):
        """
        Translate a specific functionality from TAL to Java using comprehensive prompts.
        
        Args:
            functionality: Search term for functionality (e.g., "drawdown")
            output_dir: Directory to save prompts
            depth: Dependency depth to include
            strict_mode: Enable strict translation requirements
            financial_code: Enable financial code specific requirements
            include_validation: Include validation checklists
        """
        if not self.kg:
            self.initialize()
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Build context
        context = self.context_builder.build_context_for_functionality(
            functionality,
            depth=depth
        )
        
        # Validate context
        print(f"\n{'='*70}")
        print("CONTEXT VALIDATION")
        print(f"{'='*70}")
        
        primary_procs = context['primary_procedures']
        print(f"\nPrimary Procedures ({len(primary_procs)}):")
        for proc in primary_procs:
            status = "✓" if proc.get('code') else "✗"
            code_len = proc.get('code_length', 0)
            print(f"  {status} {proc['name']}: {code_len:,} chars")
            if not proc.get('code'):
                print(f"      Location: {proc['file']}:{proc['line']}")
        
        dep_procs = context['dependency_procedures']
        procs_with_code = [p for p in dep_procs if p.get('code')]
        print(f"\nDependency Procedures ({len(dep_procs)} total, {len(procs_with_code)} with code)")
        
        # Save context JSON
        context_file = output_path / f"{functionality}_context.json"
        with open(context_file, 'w', encoding='utf-8') as f:
            # Create serializable version (without code to keep file size reasonable)
            json_context = {
                'functionality': context['functionality'],
                'summary': context['summary'],
                'primary_procedures': [
                    {k: v for k, v in proc.items() if k != 'code'}
                    for proc in context['primary_procedures']
                ],
                'dependency_procedures': [
                    {k: v for k, v in proc.items() if k != 'code'}
                    for proc in context['dependency_procedures']
                ],
                'structures': context['structures'],
                'variables': context['variables'],
                'call_graph': context['call_graph']
            }
            json.dump(json_context, f, indent=2, default=str)
        
        print(f"\n✓ Saved context metadata to: {context_file}")
        
        # Generate comprehensive prompt using the imported generator
        print(f"\nGenerating comprehensive translation prompt...")
        prompt = ComprehensivePromptGenerator.generate_translation_prompt(
            context,
            strict_mode=strict_mode,
            include_validation=include_validation,
            financial_code=financial_code
        )
        
        # Save prompt
        prompt_file = output_path / f"{functionality}_comprehensive_translation_prompt.md"
        ComprehensivePromptGenerator.save_prompt(prompt, str(prompt_file))
        
        # Call API if configured
        java_file = None
        if self.use_api and self.api_key:
            print(f"\nCalling LLM API for translation...")
            java_code = self._call_llm_api(prompt)
            
            if java_code:
                java_file = output_path / f"{functionality.capitalize()}Service.java"
                with open(java_file, 'w', encoding='utf-8') as f:
                    f.write(java_code)
                print(f"✓ Saved Java translation to: {java_file}")
            else:
                print(f"✗ Failed to get LLM response")
        else:
            print(f"\n{'='*70}")
            print("NEXT STEPS FOR TRANSLATION")
            print(f"{'='*70}")
            print(f"\nThe comprehensive prompt ensures:")
            print(f"  ✓ Complete logic translation (no placeholders)")
            print(f"  ✓ 4-phase methodology (Analyze → Map → Implement → Verify)")
            print(f"  ✓ Financial code precision (BigDecimal for amounts)")
            print(f"  ✓ Line-by-line traceability")
            print(f"  ✓ Verification checklist for each procedure")
            print(f"\nTo translate:")
            print(f"  1. Copy the comprehensive prompt from: {prompt_file}")
            print(f"  2. Paste into Claude/GPT-4")
            print(f"  3. Review the generated Java code carefully")
            print(f"  4. Verify completeness using the built-in checklists")
            print(f"\nOr run with --use-api to translate automatically")
        
        return {
            'context_file': str(context_file),
            'prompt_file': str(prompt_file),
            'java_file': str(java_file) if java_file else None,
            'output_dir': str(output_path),
            'validation': {
                'primary_procedures': len(primary_procs),
                'procedures_with_code': len([p for p in primary_procs if p.get('code')]),
                'total_code_chars': sum(p.get('code_length', 0) for p in primary_procs),
                'dependency_procedures': len(dep_procs),
                'structures': len(context['structures'])
            }
        }
    
    def _call_llm_api(self, prompt: str) -> str:
        """Call LLM API (Claude) for translation"""
        try:
            import anthropic
            
            client = anthropic.Anthropic(api_key=self.api_key)
            
            print("  Sending to Claude API...")
            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=16000,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            print("  ✓ Received response")
            return message.content[0].text
            
        except ImportError:
            print("  ✗ Error: anthropic package not installed")
            print("    Install with: pip install anthropic")
            return ""
        except Exception as e:
            print(f"  ✗ Error calling API: {e}")
            return ""


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='TAL to Java Translation Assistant with Comprehensive Prompts',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic translation with comprehensive prompt
    python tal_to_java_translator.py ./tal_source drawdown --depth 3
    
    # With API call
    python tal_to_java_translator.py ./tal_source payment --use-api --api-key sk-...
    
    # Custom output directory
    python tal_to_java_translator.py ./tal_source wire_transfer --output ./output --depth 4
    
    # Non-financial code (skip BigDecimal requirements)
    python tal_to_java_translator.py ./tal_source utility --no-financial

The comprehensive prompt ensures:
  ✓ Complete logic translation (no placeholders)
  ✓ 4-phase methodology (Analyze → Map → Implement → Verify)
  ✓ Financial code precision (BigDecimal)
  ✓ Line-by-line traceability  
  ✓ Verification checklist for each procedure
"""
    )
    
    parser.add_argument('tal_source_dir', help='Directory containing TAL source code')
    parser.add_argument('functionality', help='Functionality to translate (e.g., "drawdown", "payment")')
    parser.add_argument('--depth', type=int, default=2, 
                       help='Dependency depth (default: 2)')
    parser.add_argument('--output', default='./llm_context',
                       help='Output directory (default: ./llm_context)')
    parser.add_argument('--use-api', action='store_true',
                       help='Call LLM API directly')
    parser.add_argument('--api-key', help='API key for LLM service')
    parser.add_argument('--no-strict', dest='strict_mode', action='store_false',
                       help='Disable strict translation requirements')
    parser.add_argument('--no-financial', dest='financial_code', action='store_false',
                       help='Disable financial code specific requirements')
    parser.add_argument('--no-validation', dest='include_validation', action='store_false',
                       help='Disable validation checklists')
    parser.set_defaults(strict_mode=True, financial_code=True, include_validation=True)
    
    args = parser.parse_args()
    
    # Create translator
    translator = TALToJavaTranslator(
        tal_source_dir=args.tal_source_dir,
        use_api=args.use_api,
        api_key=args.api_key
    )
    
    # Translate
    result = translator.translate_functionality(
        functionality=args.functionality,
        output_dir=args.output,
        depth=args.depth,
        strict_mode=args.strict_mode,
        financial_code=args.financial_code,
        include_validation=args.include_validation
    )
    
    # Summary
    print(f"\n{'='*70}")
    print("TRANSLATION PREPARATION COMPLETE")
    print(f"{'='*70}")
    print(f"Context: {result['context_file']}")
    print(f"Prompt: {result['prompt_file']}")
    if result['java_file']:
        print(f"Java: {result['java_file']}")
    print(f"\nValidation:")
    print(f"  Primary procedures: {result['validation']['primary_procedures']}")
    print(f"  With source code: {result['validation']['procedures_with_code']}")
    print(f"  Total code: {result['validation']['total_code_chars']:,} characters")
    print(f"  Dependencies: {result['validation']['dependency_procedures']}")
    print(f"  Structures: {result['validation']['structures']}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
