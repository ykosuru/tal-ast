#!/usr/bin/env python3
"""
TAL to Java Translation Assistant using LLM + RAG

This program:
1. Uses knowledge graph to find drawdown-related TAL code
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

Usage:
    python tal_to_java_translator.py <tal_source_dir> <functionality> [options]
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Set, Optional
import re

# Import our existing tools
from knowledge_graph import KnowledgeGraph, EntityType, RelationType
from parsers import (
    parse_tal_directory_recursive,
    KnowledgeGraphSearch,
    resolve_external_references
)


class TALCodeExtractor:
    """Extract actual TAL source code from files"""
    
    def __init__(self, tal_source_dir: str):
        self.source_dir = Path(tal_source_dir)
    
    def extract_procedure_code(self, entity) -> Optional[str]:
        """Extract the actual source code for a procedure"""
        if not entity.file_path or not entity.start_line:
            return None
        
        file_path = Path(entity.file_path)
        
        # Handle relative paths
        if not file_path.is_absolute():
            file_path = self.source_dir / file_path
        
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Extract from start_line to end_line (or estimate end)
            start = entity.start_line - 1  # 0-indexed
            
            # If we have end_line, use it
            if entity.end_line and entity.end_line > entity.start_line:
                end = entity.end_line
            else:
                # Estimate: look for END or next PROC
                end = start + 100  # Default max
                for i in range(start + 1, min(len(lines), start + 500)):
                    line = lines[i].strip().upper()
                    if line.startswith('END') or line.startswith('PROC '):
                        end = i + 1
                        break
            
            code = ''.join(lines[start:end])
            return code
            
        except Exception as e:
            print(f"Warning: Could not extract code from {file_path}: {e}")
            return None
    
    def extract_structure_code(self, entity) -> Optional[str]:
        """Extract structure definition"""
        if not entity.file_path or not entity.start_line:
            return None
        
        file_path = Path(entity.file_path)
        
        if not file_path.is_absolute():
            file_path = self.source_dir / file_path
        
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            start = entity.start_line - 1
            # Structures are usually shorter
            end = start + 50
            
            for i in range(start + 1, min(len(lines), start + 200)):
                line = lines[i].strip()
                if line.endswith(';'):  # End of structure
                    end = i + 1
                    break
            
            code = ''.join(lines[start:end])
            return code
            
        except Exception as e:
            return None


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
Translate the following TAL (Transaction Application Language) code implementing **{context['functionality']}** functionality into modern, idiomatic Java code.

## Context Summary
- Primary procedures: {context['summary']['primary_procedures']}
- Total procedures in context: {context['summary']['total_procedures']}
- Data structures: {context['summary']['total_structures']}
- Variables: {context['summary']['total_variables']}

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
{struct['code'] or '// Code not available'}
```

Fields:
"""
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
                is_external = proc.get('is_external', False)
                external_marker = " (EXTERNAL)" if is_external else ""
                prompt += f"""
### {proc['name']}{external_marker}

**Parameters**: {', '.join(proc['parameters']) if proc['parameters'] else 'none'}
**Returns**: {proc['return_type'] or 'void'}

```tal
{proc['code'][:500] if proc['code'] else '// Code not available'}...
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
                
                if graph['called_by']:
                    prompt += f"  Called by:\n"
                    for caller in graph['called_by'][:5]:
                        prompt += f"    ← {caller}\n"
                
                prompt += "\n"
        
        prompt += "```\n\n"
        
        # Add translation requirements
        prompt += """
---

## Translation Requirements

### Java Translation Guidelines:

1. **Package Structure**:
   - Create appropriate package: `com.company.payment.drawdown`
   - Separate concerns: services, models, utils

2. **Class Design**:
   - Convert TAL procedures to Java methods in appropriate classes
   - Use object-oriented design principles
   - Primary procedures → Public service methods
   - Helper procedures → Private helper methods
   - External procedures → Interface definitions

3. **Data Structures**:
   - TAL STRUCT → Java class or record
   - Use appropriate Java types (String, int, BigDecimal for financial data)
   - Add proper encapsulation (getters/setters or records)

4. **Error Handling**:
   - Replace TAL error codes with Java exceptions
   - Create custom exception types where appropriate
   - Use try-catch-finally blocks

5. **Best Practices**:
   - Use Java naming conventions (camelCase for methods/variables)
   - Add comprehensive Javadoc comments
   - Include logging (SLF4J/Log4j2)
   - Use modern Java features (streams, optionals, etc.)
   - Add input validation
   - Make thread-safe where appropriate

6. **Financial Data**:
   - Use `BigDecimal` for all monetary amounts
   - Use `LocalDateTime` for timestamps
   - Maintain precision in calculations

### Expected Output:

For each TAL procedure, provide:
1. **Java Class** with proper package and imports
2. **Method signature** with types and parameters
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
        
        print(f"\nStep 2: Resolving external references...")
        resolution = resolve_external_references(self.kg)
        print(f"  Resolved: {len(resolution['resolved'])}")
        print(f"  Unresolved: {len(resolution['unresolved'])}")
        
        print(f"\nStep 3: Initializing code extractor...")
        self.code_extractor = TALCodeExtractor(self.source_dir)
        
        print(f"\nStep 4: Initializing context builder...")
        self.context_builder = TranslationContextBuilder(self.kg, self.code_extractor)
        
        print(f"\n✓ Initialization complete")
        print(f"  Total entities: {self.kg.get_statistics()['total_entities']}")
        print(f"  Total relationships: {self.kg.get_statistics()['total_relationships']}")
    
    def translate_functionality(self, functionality: str, 
                               output_dir: str = "./translation_output",
                               depth: int = 2):
        """
        Translate a specific functionality from TAL to Java
        
        Args:
            functionality: Search term for functionality (e.g., "drawdown")
            output_dir: Directory to save translation artifacts
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
        
        print(f"\n✓ Saved context to: {context_file}")
        
        # Step 3: Generate LLM prompt
        prompt = LLMPromptGenerator.generate_translation_prompt(context)
        
        prompt_file = output_path / f"{functionality}_translation_prompt.md"
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
  
  # Include more dependencies (depth=3)
  python tal_to_java_translator.py ./tal_source drawdown --depth 3
  
  # Use Claude API for automatic translation
  python tal_to_java_translator.py ./tal_source drawdown --use-api --api-key sk-...
  
  # Custom output directory
  python tal_to_java_translator.py ./tal_source payment --output ./translations
  
The program will:
1. Parse TAL source code into knowledge graph
2. Find all procedures/structures related to the functionality
3. Extract actual source code
4. Build comprehensive translation context
5. Generate LLM-ready prompt
6. Optionally call LLM API for translation
        """
    )
    
    parser.add_argument('tal_source', help='Directory containing TAL source files')
    parser.add_argument('functionality', help='Functionality to translate (e.g., "drawdown")')
    parser.add_argument('--depth', type=int, default=2,
                       help='Dependency depth to include (default: 2)')
    parser.add_argument('--output', default='./translation_output',
                       help='Output directory (default: ./translation_output)')
    parser.add_argument('--use-api', action='store_true',
                       help='Use LLM API for automatic translation')
    parser.add_argument('--api-key', help='API key for LLM (or set ANTHROPIC_API_KEY env var)')
    
    args = parser.parse_args()
    
    # Get API key from args or environment
    api_key = args.api_key
    if args.use_api and not api_key:
        import os
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            print("Error: --api-key required or set ANTHROPIC_API_KEY environment variable")
            sys.exit(1)
    
    # Check source directory exists
    if not Path(args.tal_source).exists():
        print(f"Error: TAL source directory not found: {args.tal_source}")
        sys.exit(1)
    
    try:
        # Create translator
        translator = TALToJavaTranslator(
            args.tal_source,
            use_api=args.use_api,
            api_key=api_key
        )
        
        # Translate
        result = translator.translate_functionality(
            args.functionality,
            output_dir=args.output,
            depth=args.depth
        )
        
        print(f"\n{'='*70}")
        print("TRANSLATION COMPLETE")
        print(f"{'='*70}\n")
        print(f"Output directory: {result['output_dir']}")
        print(f"\nGenerated files:")
        print(f"  • Context JSON: {result['context_file']}")
        print(f"  • LLM Prompt: {result['prompt_file']}")
        
        if args.use_api:
            print(f"  • Java Code: {result['output_dir']}/{args.functionality.capitalize()}Service.java")
        else:
            print(f"\nTo translate:")
            print(f"  1. Open: {result['prompt_file']}")
            print(f"  2. Copy to Claude/GPT-4")
            print(f"  3. Save Java code to {result['output_dir']}/")
        
        print(f"\n{'='*70}\n")
        
    except KeyboardInterrupt:
        print("\n\nTranslation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
