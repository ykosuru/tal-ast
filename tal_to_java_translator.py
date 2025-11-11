#!/usr/bin/env python3
"""
TAL to Java Translation Assistant using LLM + RAG

This program:
1. Uses knowledge graph to find functionality (e.g., drawdown) TAL code
2. Gathers all relevant context (procedures, data structures, dependencies)
3. Reads actual TAL source code
4. Builds comprehensive LLM context
5. Generates Java translation using LLM (Claude, GPT, etc.)

Features:
- Automatic code discovery using knowledge graph search
- Full dependency tracking
- Data structure extraction
- Call graph analysis
- LLM-ready prompt generation
- Optional direct LLM API integration
- This keeps the full graph in memory, it walks through it finding the functionality and deps

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
        import repair_graph
        graph = repair_graph.load_graph('./output/knowledge_graph.json')
        matches = repair_graph.find_procedure_in_graph(name, graph)
        
        if not matches:
            print(f"✗ No procedure found matching '{args.find}!'")
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
    
    def build_context_for_functionality(self, functionality: str, 
                                       depth: int = 2) -> Dict[str, Any]:
        """
        Build comprehensive translation context for a functionality
        
        Args:
            functionality: Search term (e.g., "drawdown")
            depth: How many levels of dependencies to include
        
        Returns:
            Dict with all context needed for translation
        """
        print(f"\n{'='*70}")
        print(f"BUILDING LLM CONTEXT: '{functionality}'")
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
        
        # Step 2: Build dependency graph
        print(f"\nStep 2: Building dependency graph (depth={depth})...")
        all_procedures = set(primary_procedures)
        all_structures = set(related_structs)
        all_variables = set(related_vars)
        
        # Expand by depth
        current_procs = list(primary_procedures)
        for d in range(depth):
            next_procs = []
            
            for proc in current_procs:
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
                
                # Find variables contained in this proc
                contained = self.kg.get_neighbors(
                    proc.id,
                    rel_type=RelationType.CONTAINS,
                    direction="outgoing"
                )
                
                for callee in callees:
                    if callee not in all_procedures and callee.type == EntityType.PROCEDURE:
                        all_procedures.add(callee)
                        next_procs.append(callee)
                
                for caller in callers:
                    if caller not in all_procedures and caller.type == EntityType.PROCEDURE:
                        all_procedures.add(caller)
                        next_procs.append(caller)
                
                for var in contained:
                    if var.type == EntityType.VARIABLE:
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
        for proc in all_procedures:
            code = self.extractor.extract_procedure_code(proc)
            procedures_with_code.append({
                'entity': proc,
                'code': code,
                'is_primary': proc in primary_procedures
            })
        
        structures_with_code = []
        for struct in all_structures:
            code = self.extractor.extract_structure_code(struct)
            structures_with_code.append({
                'entity': struct,
                'code': code
            })
        
        # Step 4: Analyze call relationships
        print(f"\nStep 4: Analyzing call graph...")
        call_graph = {}
        for proc in all_procedures:
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
            
            call_graph[proc.name] = {
                'calls': [c.name for c in callees],
                'called_by': [c.name for c in callers]
            }
        
        # Step 5: Build context dictionary
        context = {
            'functionality': functionality,
            'summary': {
                'primary_procedures': len(primary_procedures),
                'total_procedures': len(all_procedures),
                'total_variables': len(all_variables),
                'total_structures': len(all_structures),
                'depth': depth
            },
            'primary_procedures': [
                {
                    'name': p['entity'].name,
                    'file': p['entity'].file_path,
                    'line': p['entity'].start_line,
                    'parameters': p['entity'].metadata.get('parameters', []),
                    'return_type': p['entity'].metadata.get('return_type'),
                    'code': p['code']
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
                    'is_external': p['entity'].metadata.get('is_external', False)
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
            'call_graph': call_graph
        }
        
        print(f"\n✓ Context built successfully")
        
        return context


class LLMPromptGenerator:
    """Generate LLM prompts for TAL to Java translation"""
    
    @staticmethod
    def generate_translation_prompt(context: Dict[str, Any],
                                    target_procedure: Optional[str] = None) -> str:
        """
        Generate a comprehensive prompt for LLM translation
        
        Args:
            context: Translation context from TranslationContextBuilder
            target_procedure: Optional specific procedure to translate
        
        Returns:
            Complete LLM prompt
        """
        
        prompt = f"""# TAL to Java Translation Task

## Objective

Convert the following TAL (Transaction Application Language) code implementing **{context['functionality']}** functionality
that aligns with Wells Fargo's current technology standards.

---

## Key Requirements

1. **Modern Java Practices**:
   - Use **Java 17+** features where appropriate.
   - Apply **Java best principles**, clean code practices, and proper exception handling.
   - Ensure thread safety and scalability for enterprise-grade systems.

2. **Frameworks & Architecture**:
   - Implement using **Spring Boot** for application structure.
   - Integrate **Apache Kafka** for messaging where applicable.
   - Use **ISO 20022 message types** for financial messaging standards:
     - Consider **pain** (Payment Initiation), **pacs** (Payments Clearing and Settlement), and **camt** (Cash Management)
     - Select the appropriate message type (e.g., `pain.001`, `funcs.003`, `camt.054`) based on the drawdown context.
   - Assume older TAL/HPE systems are **deprecated** and unavailable.

3. **Testing**:
   - Generate **unit tests** (JUnit 5 + Mockito).
   - Provide **functional/integration tests** using Spring Boot Test.
   - Ensure high code coverage and test readability.

4. **System Context**:
   - This code is part of a **larger distributed system**.
   - Design for **modularity**, **reusability**, and **future extensibility**.
   - Include **logging** (SLF4J/Logback) and **error handling** consistent with enterprise standards.

5. **Output Expectations**:
   - Fully functional **Java class(es)** implementing the drawdown logic.
   - **DTOs**, **service layer**, and **Kafka producer/consumer** if messaging is involved.
   - **ISO 20022-compliant message mapping** where applicable.
   - Complete **unit and functional tests**.

## Context Summary

- Primary procedures: {context['summary']['primary_procedures']}
- Total procedures in context: {context['summary']['total_procedures']}
- Data structures: {context['summary']['total_structures']}
- Variables: {context['summary']['total_variables']}
- db or dbtr refers to debtor
- cr or crdtr refers to creditors
- acct refers to account, trn is payment transaction
- ^ represents delimiter or word separators in TAL language as naming convention
- FAIN is the older payment message type
- Host messages and MQs are no longer relevant
- Prioritize understanding how the functionality is implemented
- GSMOS refers to OFAC/sanctions

---

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
- **send and receieve** are OS (Guardian) functions
- **$len** returns length of a string

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
        
        # Add primary procedures
        for i, proc in enumerate(context['primary_procedures'], 1):
            prompt += f"""
### {i}. {proc['name']}

**Location**: `{proc['file']}:{proc['line']}`
**Parameters**: {', '.join(proc['parameters']) if proc['parameters'] else 'none'}
**Returns**: {proc['return_type'] or 'void'}

```tal
{proc['code'] or '// Code not available'}
```

"""
        
        # Add data structures
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
        
        # Add dependency procedures (summary)
        if context['dependency_procedures']:
            prompt += f"""

---

## Dependency Procedures

These procedures are called by or call the {context['functionality']} functionality:

"""
            for proc in context['dependency_procedures'][:10]:  # Limit to first 10
                
                print(f"----> {proc}")
                is_external = proc.get('is_external', False)
                external_marker = " (EXTERNAL)" if is_external else ""
                prompt += f"""

### {proc['name']}{external_marker}

**Location**: `{proc['file']}:{proc['line']}`
**Parameters**: {', '.join(proc['parameters']) if proc['parameters'] else 'none'}
**Returns**: {proc['return_type'] or 'void'}

```tal
{proc['code'][:1000] if proc['code'] else '// Code not available'}...
```

"""
        
        # Add call graph
        prompt += """

---

## Call Graph

The following shows the procedure call relationships:

```
"""
        # Show call graph for primary procedures
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
        
        # Add translation requirements
        prompt += """

---

## Translation Requirements

### Java Translation Guidelines:

1. **Package Structure**:
   - Create appropriate package: `com.wellsfargo.epe.<functionality>`
   - Separate concerns: services, models, utils

2. **Class Design**:
   - Convert TAL procedures to Java methods in appropriate classes
   - Use object-oriented design principles
   - Primary procedures → Public service methods
   - Helper procedures → Private helper methods
   - External procedures → Interface definitions

3. **Data Structures**:
   - TAL STRUCT → Java class(es) and/or database structures
   - Use appropriate Java types (String, int, BigDecimal for financial data)
   - Add proper encapsulation (getters/setters or records)

4. **Error Handling**:
   - Replace TAL error codes with Java exceptions
   - Create custom exception types where appropriate
   - Use try-catch-finally blocks

5. **Best Practices**:
   - Use Java naming conventions (camelCase for methods/variables)
   - Add comprehensive Javadoc comments
   - Include logging (Log4j)
   - Use modern Java features (streams, optionals, etc.)
   - Add input validation
   - Make thread-safe where appropriate
   - Check for security vulnerabilities

6. **Financial Data**:
   - Use `BigDecimal` for all monetary amounts
   - Use `LocalDateTime` for timestamps
   - Maintain precision in calculations

### Expected Output:

For each TAL procedure, provide:

1. **Java Class** with proper package and imports
2. **Method signatures** with types and parameters
3. **Full implementation** with comments
4. **Unit test skeleton** (JUnit 5)

---

## Translation Task

Please translate the {context['functionality']} functionality from TAL to Java, following the guidelines above.

Provide:

1. Main service class with primary methods
2. Supporting classes (DTOs, models, etc.)
3. Interface definitions for external dependencies
4. Brief explanation of design decisions
5. Migration notes (things to watch out for)

Start with the main class implementing the core functionality.

Focus on creating production-quality, maintainable Java code that preserves the business logic
while following modern Java best practices.
"""
        
        return prompt
    
    @staticmethod
    def save_prompt(prompt: str, output_file: str):
        """Save prompt to a file"""
        with open(output_file, 'w') as f:
            f.write(prompt)
        print(f"✓ Saved prompt to: {output_file}")


class TALToJavaTranslator:
    """Main orchestrator for TAL to Java translation"""
    
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
        print(f"  Resolved: {len(resolution['resolved'])}")
        print(f"  Unresolved: {len(resolution['unresolved'])}")
        
        print(f"\nStep 3: Initializing code extractor...")
        self.code_extractor = TALCodeExtractor(self.source_dir)
        print(f"  ✓ Code extractor ready")
        
        print(f"\nStep 4: Initializing context builder...")
        self.context_builder = TranslationContextBuilder(self.kg, self.code_extractor)
        
        print(f"\n✓ Initialization complete")
        print(f"  Total entities: {self.kg.get_statistics()['total_entities']}")
        print(f"  Total relationships: {self.kg.get_statistics()['total_relationships']}")
    
    def translate_functionality(self, functionality: str,
                               output_dir: str = "./llm_context",
                               depth: int = 2):
        """
        Translate a specific functionality from TAL to Java
        
        Args:
            functionality: Search term for functionality (e.g., "drawdown")
            output_dir: Directory to save compressed prompts
            depth: Dependency depth to include
        """
        if not self.kg:
            self.initialize()
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Step 1: Build context
        context = self.context_builder.build_context_for_functionality(
            functionality,
            depth=depth
        )
        
        # Step 2: Save context as JSON
        context_file = output_path / f"{functionality}_context.json"
        with open(context_file, 'w') as f:
            # Make entities serializable
            serializable_context = {
                **context,
                'primary_procedures': context['primary_procedures'],
                'dependency_procedures': context['dependency_procedures'],
                'structures': context['structures'],
                'variables': context['variables'],
                'call_graph': context['call_graph']
            }
            json.dump(serializable_context, f, indent=2, default=str)
        
        print(f"\n✓\n Saved context to: {context_file}")
        
        # Step 3: Generate LLM prompt
        prompt = LLMPromptGenerator.generate_translation_prompt(context)
        
        prompt_file = output_path / f"{functionality}_coding_agent_prompt.md"
        LLMPromptGenerator.save_prompt(prompt, str(prompt_file))
        
        # Step 4: If using API, call LLM
        if self.use_api and self.api_key:
            print(f"\nStep 5: Calling LLM API for translation...")
            java_code = self._call_llm_api(prompt)
            
            java_file = output_path / f"{functionality.capitalize()}Service.java"
            with open(java_file, 'w') as f:
                f.write(java_code)
            
            print(f"✓ Saved Java translation to: {java_file}")
        else:
            print(f"\nNext steps:")
            print(f"  1. Review the prompt: {prompt_file}")
            print(f"  2. Copy the prompt to your LLM (Claude, GPT-4, etc.)")
            print(f"  3. Save the generated Java code")
            print(f"\n  Or run with --use-api to translate automatically")
        
        return {
            'context_file': str(context_file),
            'prompt_file': str(prompt_file),
            'output_dir': str(output_path)
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
        description='TAL to Java Translation Assistant with LLM + RAG',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Build translation context for drawdown functionality
    python tal_to_java_translator.py ./tal_source drawdown
    
    # Include deeper dependency chain
    python tal_to_java_translator.py ./tal_source drawdown --depth 3
    
    # Use Claude API directly for translation
    python tal_to_java_translator.py ./tal_source drawdown --use-api --api-key sk-ant-...
    
    # Custom output directory
    python tal_to_java_translator.py ./tal_source payment --output ./prompts
"""
    )
    
    parser.add_argument('tal_source_dir', help='Directory containing TAL source code')
    parser.add_argument('functionality', help='Functionality to translate (e.g., "drawdown", "payment")')
    parser.add_argument('--depth', type=int, default=2, 
                       help='Dependency depth to include (default: 2)')
    parser.add_argument('--output', default='./llm_context',
                       help='Output directory for prompts (default: ./llm_context)')
    parser.add_argument('--use-api', action='store_true',
                       help='Use Claude API for direct translation')
    parser.add_argument('--api-key', help='API key for Claude (or set ANTHROPIC_API_KEY env var)')
    
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
        depth=args.depth
    )
    
    print(f"\n{'='*70}")
    print("TRANSLATION CONTEXT READY")
    print(f"{'='*70}")
    print(f"Context: {result['context_file']}")
    print(f"Prompt: {result['prompt_file']}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
