#!/usr/bin/env python3
"""
TAL to Java Translation Assistant using LLM + RAG

This program:
1. Uses knowledge graph to find functionality-related TAL code
2. Gathers all relevant context (procedures, data structures, dependencies)
3. Reads actual TAL source code using tal_proc_parser
4. Builds comprehensive LLM context
5. Generates Java translation using LLM (Claude, GPT, etc.)

Features:
- Automatic code discovery using knowledge graph search
- Accurate code extraction using tal_proc_parser
- Full dependency tracking
- Data structure extraction
- Call graph analysis
- LLM-ready prompt generation
- Optional direct LLM API integration
- Comprehensive validation and diagnostics

Usage:
    python tal_to_java_translator.py <tal_source_dir> <functionality> [options]

Example:
    python tal_to_java_translator.py ./tal_source drawdown --depth 3
    python tal_to_java_translator.py ./tal_source payment --use-api --api-key sk-...
"""

import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Set, Optional
import re

from knowledge_graph import KnowledgeGraph, EntityType, RelationType
from parsers import (
    parse_tal_directory_recursive,
    KnowledgeGraphSearch,
    resolve_external_references
)
import tal_proc_parser


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
        """Get all procedure boundaries in a file using tal_proc_parser."""
        file_path_str = str(file_path)
        
        if file_path_str in self.proc_boundaries_cache:
            return self.proc_boundaries_cache[file_path_str]
        
        lines = self._get_file_contents(file_path)
        if not lines:
            return {}
        
        content = ''.join(lines)
        proc_declarations = tal_proc_parser.find_procedure_declarations(content)
        
        boundaries = {}
        for i, (start_line, proc_name, declaration) in enumerate(proc_declarations):
            declaration_line_count = declaration.count('\n') + 1
            declaration_end_line = start_line + declaration_line_count - 1
            
            if i + 1 < len(proc_declarations):
                end_line = proc_declarations[i + 1][0] - 1
            else:
                end_line = len(lines)
            
            boundaries[proc_name.upper()] = {
                'start': start_line,
                'declaration_end': declaration_end_line,
                'end': end_line,
                'original_name': proc_name
            }
        
        self.proc_boundaries_cache[file_path_str] = boundaries
        return boundaries
    
    def extract_procedure_code(self, entity) -> Optional[str]:
        """Extract the actual source code for a procedure."""
        if not entity.file_path:
            print(f"Warning: No file_path for entity {entity.name}")
            return None
        
        file_path = Path(entity.file_path)
        
        if not file_path.is_absolute():
            file_path = self.source_dir / file_path
        
        lines = self._get_file_contents(file_path)
        if not lines:
            print(f"Warning: Could not read file {file_path}")
            return None
        
        boundaries = self._get_procedure_boundaries(file_path)
        
        proc_name_upper = entity.name.upper()
        
        if proc_name_upper in boundaries:
            boundary = boundaries[proc_name_upper]
            start = boundary['start'] - 1
            end = boundary['end']
            
            code = ''.join(lines[start:end])
            return code
        
        elif entity.start_line:
            print(f"Warning: Using fallback extraction for {entity.name}")
            start = entity.start_line - 1
            
            if entity.end_line and entity.end_line > entity.start_line:
                end = entity.end_line
            else:
                end = start + 100
                for i in range(start + 1, min(len(lines), start + 500)):
                    line = lines[i].strip().upper()
                    if line.startswith('END') or line.startswith('PROC '):
                        end = i + 1
                        break
            
            code = ''.join(lines[start:end])
            return code
        
        print(f"Warning: Could not find procedure {entity.name} in {file_path}")
        return None
    
    def extract_structure_code(self, entity) -> Optional[str]:
        """Extract structure definition from source file."""
        if not entity.file_path:
            return None
        
        file_path = Path(entity.file_path)
        
        if not file_path.is_absolute():
            file_path = self.source_dir / file_path
        
        lines = self._get_file_contents(file_path)
        if not lines:
            return None
        
        if not entity.start_line:
            return None
        
        start = entity.start_line - 1
        
        end = start + 1
        found_begin = False
        begin_count = 0
        end_count = 0
        
        for i in range(start, min(len(lines), start + 200)):
            line = lines[i].strip().upper()
            
            if 'BEGIN' in line:
                found_begin = True
                begin_count += line.count('BEGIN')
            if 'END' in line:
                end_count += line.count('END')
            
            if found_begin:
                if begin_count > 0 and begin_count == end_count:
                    end = i + 1
                    break
            elif line.endswith(';'):
                end = i + 1
                break
        
        code = ''.join(lines[start:end])
        return code
    
    def extract_file_section(self, file_path: str, start_line: int, end_line: int) -> Optional[str]:
        """Extract a section of a file by line numbers."""
        path = Path(file_path)
        
        if not path.is_absolute():
            path = self.source_dir / path
        
        lines = self._get_file_contents(path)
        if not lines:
            return None
        
        start = max(0, start_line - 1)
        end = min(len(lines), end_line)
        
        return ''.join(lines[start:end])


class TranslationContextBuilder:
    """Build comprehensive context for LLM translation."""
    
    def __init__(self, kg: KnowledgeGraph, code_extractor: TALCodeExtractor):
        self.kg = kg
        self.extractor = code_extractor
        self.search = KnowledgeGraphSearch(kg)
    
    def build_context_for_functionality(self, 
                                       functionality: str, 
                                       depth: int = 2,
                                       include_full_dependencies: bool = True) -> Dict[str, Any]:
        """Build comprehensive translation context for a functionality."""
        print(f"\n{'='*70}")
        print(f"BUILDING TRANSLATION CONTEXT: '{functionality}'")
        print(f"{'='*70}\n")
        
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
        
        print(f"\nStep 2: Building dependency graph (depth={depth})...")
        all_procedures = set(primary_procedures)
        all_structures = set(related_structs)
        all_variables = set(related_vars)
        
        procedure_dependencies = {}
        
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
                
                callees = self.kg.get_neighbors(
                    proc.id,
                    rel_type=RelationType.CALLS,
                    direction="outgoing"
                )
                
                callers = self.kg.get_neighbors(
                    proc.id,
                    rel_type=RelationType.CALLS,
                    direction="incoming"
                )
                
                uses_structs = self.kg.get_neighbors(
                    proc.id,
                    rel_type=RelationType.USES,
                    direction="outgoing"
                )
                
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
                print(f"      File: {proc.file_path}, Line: {proc.start_line}")
            
            procedures_with_code.append({
                'entity': proc,
                'code': code,
                'is_primary': proc in primary_procedures,
                'code_length': code_length
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
        
        context = {
            'functionality': functionality,
            'summary': {
                'primary_procedures': len(primary_procedures),
                'total_procedures': len(all_procedures),
                'total_variables': len(all_variables),
                'total_structures': len(all_structures),
                'depth': depth,
                'code_extraction': code_extraction_stats
            },
            'primary_procedures': [
                {
                    'name': p['entity'].name,
                    'file': p['entity'].file_path,
                    'line': p['entity'].start_line,
                    'parameters': p['entity'].metadata.get('parameters', []),
                    'return_type': p['entity'].metadata.get('return_type'),
                    'code': p['code'],
                    'code_length': p['code_length'],
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
                    'code': p['code'] if include_full_dependencies else (
                        p['code'][:500] + "\n... (truncated)" if p['code'] and len(p['code']) > 500 else p['code']
                    ),
                    'code_length': p['code_length'],
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
        
        primary_with_code = [p for p in context['primary_procedures'] if p['code']]
        if len(primary_with_code) < len(primary_procedures):
            print(f"\n  ⚠️  WARNING: Only {len(primary_with_code)}/{len(primary_procedures)} "
                  f"primary procedures have source code!")
            print("      Translation quality may be affected.")
        
        print(f"\n✓ Context built successfully")
        
        return context


class LLMPromptGenerator:
    """Generate LLM prompts for TAL to Java translation."""
    
    @staticmethod
    def generate_translation_prompt(context: Dict[str, Any], 
                                    target_procedure: Optional[str] = None,
                                    verbose: bool = False) -> str:
        """Generate a comprehensive prompt for LLM translation."""
        
        issues = []
        primary_procs = context['primary_procedures']
        procs_without_code = [p for p in primary_procs if not p.get('code')]
        
        if procs_without_code:
            issues.append(f"⚠️  {len(procs_without_code)} primary procedures missing source code:")
            for p in procs_without_code:
                issues.append(f"   - {p['name']} (file: {p['file']}, line: {p['line']})")
        
        if verbose and issues:
            print("\nPrompt Generation Issues:")
            for issue in issues:
                print(issue)
        
        prompt = f"""# TAL to Java Translation Task

## Objective
Translate the following TAL (Transaction Application Language) code implementing **{context['functionality']}** functionality into modern, idiomatic Java code.

## Context Summary
- Primary procedures: {context['summary']['primary_procedures']}
- Total procedures in context: {context['summary']['total_procedures']}
- Data structures: {context['summary']['total_structures']}
- Variables: {context['summary']['total_variables']}
- Code successfully extracted: {context['summary']['code_extraction']['success']} procedures
- Total source code: {context['summary']['code_extraction']['total_chars']:,} characters

"""
        
        if issues:
            prompt += "## ⚠️ Translation Notes\n\n"
            for issue in issues:
                prompt += f"{issue}\n"
            prompt += "\n"
        
        prompt += """---

## TAL Language Overview

TAL (Transaction Application Language) is a block-structured language designed for HP NonStop systems. Key characteristics:

### Syntax Differences from Java:
- **Procedure Declaration**: `PROC procedure_name(params);` instead of `public void method()`
- **Variable Types**: `INT`, `STRING`, `FIXED` instead of `int`, `String`, `double`
- **Pointers**: `.EXT` for extended pointers (64-bit addresses)
- **Arrays**: `ARRAY[0:9]` instead of `[]`
- **String Operations**: `@` for address-of, `':='` for string assignment
- **Control Flow**: `IF...THEN`, `FOR...TO...DO`, `WHILE...DO`
- **Procedure Calls**: `CALL procedure_name(args)` or just `procedure_name(args)`
- **Block Delimiters**: `BEGIN...END` instead of `{{}}`
- **Assignment**: `:=` instead of `=`
- **Comments**: `!` instead of `//` or `/* */`

### Common TAL Patterns:
```tal
! Comment
INT variable;
STRING .str;
PROC my_proc(param1, param2);
  BEGIN
    IF condition THEN
      statement;
  END;
```

---

## Primary Procedures ({context['functionality']} functionality)

"""
        
        for i, proc in enumerate(context['primary_procedures'], 1):
            code_display = proc['code'] if proc['code'] else "// ⚠️ CODE NOT AVAILABLE - Manual extraction required"
            code_length = proc.get('code_length', 0)
            
            prompt += f"""
### {i}. {proc['name']}

**Location**: `{proc['file']}:{proc['line']}`
**Parameters**: {', '.join(proc['parameters']) if proc['parameters'] else 'none'}
**Returns**: {proc['return_type'] or 'void'}
**Code Size**: {code_length:,} characters
```tal
{code_display}
```

"""
            
            deps = proc.get('dependencies', {})
            if deps.get('calls'):
                prompt += f"**Calls**: {', '.join(deps['calls'][:10])}"
                if len(deps['calls']) > 10:
                    prompt += f" (and {len(deps['calls']) - 10} more)"
                prompt += "\n"
            if deps.get('called_by'):
                prompt += f"**Called By**: {', '.join(deps['called_by'][:10])}"
                if len(deps['called_by']) > 10:
                    prompt += f" (and {len(deps['called_by']) - 10} more)"
                prompt += "\n"
            if deps.get('uses_structures'):
                prompt += f"**Uses Structures**: {', '.join(deps['uses_structures'])}\n"
            prompt += "\n"
        
        if context['structures']:
            prompt += f"""
---

## Data Structures

These structures are used by the {context['functionality']} functionality:

"""
            for struct in context['structures']:
                prompt += f"""
### {struct['name']}

**Fields**: {len(struct['fields'])}
```tal
{struct['code'] or '// Structure definition not available'}
```

"""
                if struct['fields']:
                    prompt += "Fields:\n"
                    for field in struct['fields']:
                        prompt += f"- `{field['name']}`: {field.get('type', 'unknown')}\n"
                prompt += "\n"
        
        if context['dependency_procedures']:
            prompt += f"""
---

## Dependency Procedures

These procedures are called by or call the {context['functionality']} functionality.
Showing first 10 dependencies:

"""
            for proc in context['dependency_procedures'][:10]:
                is_external = proc.get('is_external', False)
                external_marker = " (EXTERNAL)" if is_external else ""
                
                prompt += f"""
### {proc['name']}{external_marker}

**Location**: `{proc['file']}:{proc['line']}`
**Parameters**: {', '.join(proc['parameters']) if proc['parameters'] else 'none'}
**Returns**: {proc['return_type'] or 'void'}
```tal
{proc['code'][:500] if proc['code'] else '// Code not available'}...
```

"""
        
        prompt += """
---

## Call Graph

The following shows the procedure call relationships:
"""
for proc in context['primary_procedures']:
proc_name = proc['name']
if proc_name in context['call_graph']:
graph = context['call_graph'][proc_name]
            prompt += f"{proc_name}:\n"
            
            if graph['calls']:
                prompt += f"  Calls:\n"
                for callee in graph['calls'][:5]:
                    prompt += f"    → {callee}\n"
                if len(graph['calls']) > 5:
                    prompt += f"    → ... and {len(graph['calls']) - 5} more\n"
            
            if graph['called_by']:
                prompt += f"  Called by:\n"
                for caller in graph['called_by'][:5]:
                    prompt += f"    ← {caller}\n"
                if len(graph['called_by']) > 5:
                    prompt += f"    ← ... and {len(graph['called_by']) - 5} more\n"
            
            prompt += "\n"
    
    prompt += "```\n\n"
    
    prompt += """

Translation Requirements
Java Translation Guidelines:

Package Structure:

Create appropriate package: com.company.payment.<functionality>
Separate concerns: services, models, utils
Use standard Java directory structure


Class Design:

Convert TAL procedures to Java methods in appropriate classes
Use object-oriented design principles
Primary procedures → Public service methods
Helper procedures → Private helper methods
External procedures → Interface definitions
Consider using dependency injection


Data Structures:

TAL STRUCT → Java class or record (Java 14+)
Use appropriate Java types:

INT → int or Integer
STRING → String
FIXED/REAL → BigDecimal for financial data, or double


Add proper encapsulation (getters/setters or records)
Use immutable objects where appropriate


Error Handling:

Replace TAL error codes with Java exceptions
Create custom exception types where appropriate
Use try-catch-finally blocks
Document exceptions in Javadoc


Best Practices:

Use Java naming conventions (camelCase for methods/variables, PascalCase for classes)
Add comprehensive Javadoc comments
Include logging (SLF4J/Log4j2)
Use modern Java features:

Streams API for collections
Optional for nullable values
Try-with-resources for resource management


Add input validation
Make thread-safe where appropriate
Follow SOLID principles


Financial Data (if applicable):

Use BigDecimal for all monetary amounts
Use LocalDateTime / ZonedDateTime for timestamps
Maintain precision in calculations
Round appropriately for display


Testing:

Design code to be testable
Consider what unit tests would be needed
Identify integration points



Expected Output:
For each TAL procedure, provide:

Java Class with proper package and imports
Method signatures with types and parameters
Full implementation with inline comments
Javadoc for all public methods and classes
Unit test skeleton (JUnit 5) showing key test cases


Translation Task
Please translate the {context['functionality']} functionality from TAL to Java, following the guidelines above.
Provide:

Main service class with primary methods
Supporting classes (DTOs, models, etc.)
Interface definitions for external dependencies
Brief design overview explaining your approach
Migration notes (things to watch out for, gotchas, assumptions)

Start with the main class implementing the core functionality, then provide supporting classes.
Focus on creating production-quality, maintainable Java code that preserves the business logic while following modern Java best practices.
"""
    return prompt

@staticmethod
def save_prompt(prompt: str, output_file: str):
    """Save prompt to a file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(prompt)
    print(f"✓ Saved prompt to: {output_file}")
class TALToJavaTranslator:
"""Main orchestrator for TAL to Java translation."""
def __init__(self, tal_source_dir: str, use_api: bool = False, api_key: Optional[str] = None):
    self.source_dir = tal_source_dir
    self.kg = None
    self.code_extractor = None
    self.context_builder = None
    self.use_api = use_api
    self.api_key = api_key

def initialize(self):
    """Initialize knowledge graph by parsing TAL source."""
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
                           output_dir: str = "./translation_output",
                           depth: int = 2,
                           verbose: bool = True):
    """Translate a specific functionality from TAL to Java."""
    if not self.kg:
        self.initialize()
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    context = self.context_builder.build_context_for_functionality(
        functionality,
        depth=depth,
        include_full_dependencies=True
    )
    
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
    
    context_file = output_path / f"{functionality}_context.json"
    with open(context_file, 'w', encoding='utf-8') as f:
        json_context = {
            'functionality': context['functionality'],
            'summary': context['summary'],
            'primary_procedures': context['primary_procedures'],
            'dependency_procedures': [
                {k: v for k, v in proc.items() if k != 'code'}
                for proc in context['dependency_procedures']
            ],
            'structures': context['structures'],
            'variables': context['variables'],
            'call_graph': context['call_graph']
        }
        json.dump(json_context, f, indent=2, default=str)
    
    print(f"\n✓ Saved context to: {context_file}")
    
    print(f"\nGenerating LLM prompt...")
    prompt = LLMPromptGenerator.generate_translation_prompt(context, verbose=verbose)
    
    prompt_file = output_path / f"{functionality}_translation_prompt.md"
    LLMPromptGenerator.save_prompt(prompt, str(prompt_file))
    
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
        print(f"\nTo translate with LLM:")
        print(f"  1. Copy the prompt from: {prompt_file}")
        print(f"  2. Paste into Claude/GPT-4")
        print(f"  3. Save the Java code to {output_path}/")
    
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
    """Call LLM API (Claude) for translation."""
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
"""Main entry point for the TAL to Java translator."""
import argparse
parser = argparse.ArgumentParser(
    description='TAL to Java Translation Assistant with LLM + RAG',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
