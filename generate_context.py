"""
Practical Integration Example: Query → Search → Compress → LLM Code Generation

This shows the complete workflow for your use case:
1. Query for functionality (e.g., "drawdown processing")
2. Search both indexers
3. Combine and deduplicate results
4. Compress context for LLM
5. Generate code using LLM

Assumes you've applied the critical patches from code_patches.py
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import json

# Import your indexers (adjust paths as needed)
# from universal_file_indexer import UniversalFileSearcher
# from lsi_indexer import HybridSearcher
from unified_code_search import UnifiedCodeSearch, SearchResult


class CodeGenerationPipeline:
    """
    End-to-end pipeline: Query → Search → Compress → Generate Code
    
    Example:
        pipeline = CodeGenerationPipeline(
            universal_index="./universal_index",
            lsi_index="./hybrid_index"
        )
        
        code = pipeline.generate_code(
            functionality="drawdown processing",
            language="tal",
            requirements="Must handle ACE repair codes and OFAC screening"
        )
    """
    
    def __init__(
        self,
        universal_index: str,
        lsi_index: str,
        universal_weight: float = 0.6,  # Prefer docs slightly
        lsi_weight: float = 0.4
    ):
        # Initialize unified search
        self.search = UnifiedCodeSearch(
            universal_index_path=universal_index,
            lsi_index_path=lsi_index,
            universal_weight=universal_weight,
            lsi_weight=lsi_weight
        )
        
        print("✓ Code generation pipeline initialized")
    
    def generate_code(
        self,
        functionality: str,
        language: str = "python",
        requirements: str = "",
        context_tokens: int = 4000,
        top_k_results: int = 20,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Generate code for a given functionality
        
        Args:
            functionality: What to implement (e.g., "drawdown processing")
            language: Target language (python, tal, java, etc.)
            requirements: Additional requirements or constraints
            context_tokens: Max tokens for context
            top_k_results: Number of search results to consider
            verbose: Show progress
        
        Returns:
            {
                'generated_code': str,
                'explanation': str,
                'sources': List[str],
                'context_used': str,
                'search_results': List[SearchResult]
            }
        """
        if verbose:
            print(f"\n{'='*80}")
            print(f"Generating {language} code for: {functionality}")
            print(f"{'='*80}\n")
        
        # Step 1: Search for relevant code/docs
        if verbose:
            print("[Step 1/4] Searching codebase...")
        
        results, compressed_context = self.search.search_and_compress(
            query=functionality,
            top_k=top_k_results,
            max_tokens=context_tokens,
            verbose=verbose
        )
        
        if not results:
            return {
                'generated_code': '',
                'explanation': 'No relevant examples found in codebase',
                'sources': [],
                'context_used': compressed_context,
                'search_results': []
            }
        
        # Step 2: Build LLM prompt
        if verbose:
            print("\n[Step 2/4] Building LLM prompt...")
        
        prompt = self._build_code_generation_prompt(
            functionality=functionality,
            language=language,
            requirements=requirements,
            context=compressed_context,
            results=results
        )
        
        if verbose:
            print(f"  Prompt length: {len(prompt)} chars")
            print(f"  Using {len(results)} code examples")
        
        # Step 3: Call LLM (placeholder - use your actual LLM)
        if verbose:
            print("\n[Step 3/4] Calling LLM...")
        
        generated_code, explanation = self._call_llm(prompt, language)
        
        # Step 4: Post-process and return
        if verbose:
            print("\n[Step 4/4] Post-processing...")
        
        sources = list(set(r.source_file for r in results[:10]))
        
        result = {
            'generated_code': generated_code,
            'explanation': explanation,
            'sources': sources,
            'context_used': compressed_context,
            'search_results': results,
            'prompt': prompt  # For debugging
        }
        
        if verbose:
            print(f"\n{'='*80}")
            print("✓ Code generation complete!")
            print(f"  Lines of code: {len(generated_code.splitlines())}")
            print(f"  Sources used: {len(sources)}")
            print(f"{'='*80}\n")
        
        return result
    
    def _build_code_generation_prompt(
        self,
        functionality: str,
        language: str,
        requirements: str,
        context: str,
        results: List[SearchResult]
    ) -> str:
        """Build comprehensive prompt for LLM"""
        
        # Extract key patterns from results
        key_functions = set()
        key_imports = set()
        key_variables = set()
        
        for result in results[:10]:
            if result.metadata.get('functions'):
                key_functions.update(result.metadata['functions'][:5])
            if result.metadata.get('imports'):
                key_imports.update(result.metadata['imports'][:3])
        
        # Build prompt
        prompt_parts = []
        
        # Header
        prompt_parts.append(f"# Code Generation Task: {functionality}")
        prompt_parts.append(f"Target Language: {language.upper()}")
        if requirements:
            prompt_parts.append(f"Requirements: {requirements}")
        prompt_parts.append("")
        
        # Context from codebase
        prompt_parts.append("## Relevant Code Examples from Codebase")
        prompt_parts.append(context)
        prompt_parts.append("")
        
        # Key patterns observed
        if key_functions:
            prompt_parts.append("## Key Functions/Procedures Found")
            for func in list(key_functions)[:10]:
                prompt_parts.append(f"- {func}")
            prompt_parts.append("")
        
        if key_imports:
            prompt_parts.append("## Common Imports/Dependencies")
            for imp in list(key_imports)[:5]:
                prompt_parts.append(f"- {imp}")
            prompt_parts.append("")
        
        # Instructions
        prompt_parts.append("## Instructions")
        prompt_parts.append(f"Based on the code examples above, generate {language} code that implements: {functionality}")
        prompt_parts.append("")
        prompt_parts.append("Requirements:")
        prompt_parts.append("1. Follow the patterns and style from the examples")
        prompt_parts.append("2. Reuse existing functions/procedures where applicable")
        prompt_parts.append("3. Include appropriate error handling")
        prompt_parts.append("4. Add comments explaining key logic")
        
        if language.lower() == 'tal':
            prompt_parts.append("5. Use TAL-specific patterns (PROC, SUBPROC, system calls)")
            prompt_parts.append("6. Handle error codes appropriately")
        elif language.lower() == 'python':
            prompt_parts.append("5. Use type hints")
            prompt_parts.append("6. Follow PEP 8 style guidelines")
        
        if requirements:
            prompt_parts.append(f"7. {requirements}")
        
        prompt_parts.append("")
        prompt_parts.append("## Generated Code")
        prompt_parts.append(f"```{language}")
        
        return "\n".join(prompt_parts)
    
    def _call_llm(self, prompt: str, language: str) -> Tuple[str, str]:
        """
        Call LLM to generate code
        
        This is a placeholder - replace with your actual LLM call
        """
        # PLACEHOLDER: Replace with your actual LLM integration
        # Examples:
        # - OpenAI API: openai.ChatCompletion.create(...)
        # - Anthropic Claude: anthropic.messages.create(...)
        # - Local model: your_model.generate(...)
        
        # For demonstration, return a template
        template_code = f"""
# Generated {language} code for functionality
# This is a PLACEHOLDER - replace _call_llm() with actual LLM integration

def placeholder_implementation():
    \"\"\"
    TODO: Implement the actual functionality here
    
    Based on the context provided in the prompt, this should:
    1. Follow patterns from the codebase
    2. Reuse existing functions
    3. Handle errors appropriately
    \"\"\"
    pass
"""
        
        explanation = """
This is a placeholder response. To use this pipeline with a real LLM:

1. Integrate with OpenAI:
   import openai
   response = openai.ChatCompletion.create(
       model="gpt-4",
       messages=[{"role": "user", "content": prompt}]
   )
   code = response.choices[0].message.content

2. Or integrate with Anthropic Claude:
   import anthropic
   client = anthropic.Anthropic(api_key="...")
   response = client.messages.create(
       model="claude-3-5-sonnet-20241022",
       messages=[{"role": "user", "content": prompt}]
   )
   code = response.content[0].text

3. Or use a local model via API
"""
        
        print("\n⚠️  Using placeholder LLM response")
        print("    Replace _call_llm() method with your actual LLM integration\n")
        
        return template_code, explanation
    
    def batch_generate(
        self,
        functionalities: List[str],
        language: str = "python",
        output_dir: str = "./generated_code"
    ) -> List[Dict[str, Any]]:
        """
        Generate code for multiple functionalities at once
        
        Useful for generating a complete module or set of related functions
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        results = []
        
        for i, functionality in enumerate(functionalities, 1):
            print(f"\n{'='*80}")
            print(f"[{i}/{len(functionalities)}] Processing: {functionality}")
            print(f"{'='*80}")
            
            result = self.generate_code(
                functionality=functionality,
                language=language,
                verbose=True
            )
            
            results.append(result)
            
            # Save to file
            safe_name = "".join(c if c.isalnum() else "_" for c in functionality)
            file_ext = self._get_file_extension(language)
            output_file = output_path / f"{safe_name}{file_ext}"
            
            with open(output_file, 'w') as f:
                f.write(f"# Generated code for: {functionality}\n")
                f.write(f"# Sources: {', '.join(result['sources'][:3])}\n\n")
                f.write(result['generated_code'])
            
            print(f"✓ Saved to: {output_file}")
        
        # Save summary
        summary_file = output_path / "generation_summary.json"
        with open(summary_file, 'w') as f:
            summary = [
                {
                    'functionality': functionalities[i],
                    'sources': r['sources'],
                    'lines_of_code': len(r['generated_code'].splitlines())
                }
                for i, r in enumerate(results)
            ]
            json.dump(summary, f, indent=2)
        
        print(f"\n{'='*80}")
        print(f"✓ Batch generation complete!")
        print(f"  Generated {len(results)} files")
        print(f"  Output directory: {output_path}")
        print(f"  Summary: {summary_file}")
        print(f"{'='*80}\n")
        
        return results
    
    def _get_file_extension(self, language: str) -> str:
        """Get file extension for language"""
        extensions = {
            'python': '.py',
            'tal': '.tal',
            'java': '.java',
            'c': '.c',
            'cpp': '.cpp',
            'javascript': '.js',
            'typescript': '.ts',
            'go': '.go',
            'rust': '.rs',
            'sql': '.sql'
        }
        return extensions.get(language.lower(), '.txt')


# ============================================================================
# Example Usage
# ============================================================================

def example_single_function():
    """Example: Generate a single function"""
    
    pipeline = CodeGenerationPipeline(
        universal_index="./universal_index",
        lsi_index="./hybrid_index",
        universal_weight=0.6,
        lsi_weight=0.4
    )
    
    result = pipeline.generate_code(
        functionality="payment drawdown processing with ACE repair",
        language="tal",
        requirements="Must validate OFAC screening before processing",
        context_tokens=4000,
        verbose=True
    )
    
    print("\n" + "="*80)
    print("GENERATED CODE:")
    print("="*80)
    print(result['generated_code'])
    print("\n" + "="*80)
    print("EXPLANATION:")
    print("="*80)
    print(result['explanation'])
    print("\n" + "="*80)
    print("SOURCES USED:")
    print("="*80)
    for source in result['sources']:
        print(f"  - {source}")


def example_batch_generation():
    """Example: Generate multiple related functions"""
    
    pipeline = CodeGenerationPipeline(
        universal_index="./universal_index",
        lsi_index="./hybrid_index"
    )
    
    functionalities = [
        "payment drawdown initiation",
        "OFAC party screening validation",
        "ACE repair code determination",
        "wire transfer cutoff checking",
        "credit party validation",
        "ISO 20022 message formatting"
    ]
    
    results = pipeline.batch_generate(
        functionalities=functionalities,
        language="python",
        output_dir="./generated_payment_processing"
    )
    
    print(f"\n✓ Generated {len(results)} functions")


def example_with_context_analysis():
    """Example: Analyze search results before generating"""
    
    pipeline = CodeGenerationPipeline(
        universal_index="./universal_index",
        lsi_index="./hybrid_index"
    )
    
    # Step 1: Search first
    print("Step 1: Searching codebase...")
    results = pipeline.search.search(
        query="drawdown processing",
        top_k=20,
        verbose=True
    )
    
    # Step 2: Analyze what was found
    print("\n" + "="*80)
    print("SEARCH ANALYSIS:")
    print("="*80)
    
    print("\nTop 5 Results:")
    for i, result in enumerate(results[:5], 1):
        print(f"\n[{i}] {result.source_file} (score: {result.score:.3f})")
        print(f"    Type: {result.file_type}")
        if result.capabilities:
            print(f"    Capabilities: {', '.join(result.capabilities[:2])}")
        if result.keywords:
            print(f"    Keywords: {', '.join(result.keywords[:5])}")
        print(f"    Snippet: {result.text[:150]}...")
    
    # Step 3: Generate if satisfied with results
    decision = input("\nProceed with code generation? (y/n): ")
    
    if decision.lower() == 'y':
        compressed = pipeline.search.get_compressed_context(
            results,
            query="drawdown processing",
            max_tokens=4000
        )
        
        print("\n" + "="*80)
        print("COMPRESSED CONTEXT (for LLM):")
        print("="*80)
        print(compressed)
        
        # Now generate
        result = pipeline.generate_code(
            functionality="drawdown processing",
            language="python",
            verbose=True
        )


def example_compare_indexers():
    """Example: Compare results from both indexers"""
    
    pipeline = CodeGenerationPipeline(
        universal_index="./universal_index",
        lsi_index="./hybrid_index"
    )
    
    query = "OFAC screening validation"
    
    print(f"Query: {query}\n")
    
    # Get results from both
    results = pipeline.search.search(query, top_k=10, verbose=False)
    
    # Separate by source
    universal_results = [r for r in results if r.source_indexer == 'universal']
    lsi_results = [r for r in results if r.source_indexer == 'lsi']
    
    print("="*80)
    print("RESULTS COMPARISON:")
    print("="*80)
    
    print(f"\nFrom Universal Indexer (PDFs/Docs): {len(universal_results)}")
    for r in universal_results[:3]:
        print(f"  - {r.source_file} (score: {r.score:.3f})")
    
    print(f"\nFrom LSI Indexer (Code): {len(lsi_results)}")
    for r in lsi_results[:3]:
        print(f"  - {r.source_file} (score: {r.score:.3f})")
    
    print(f"\nTotal unique results: {len(results)}")
    print(f"Coverage: {len(set(r.source_file for r in results))} unique files")


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Code Generation Pipeline using Unified Search"
    )
    parser.add_argument(
        "--universal-index",
        default="./universal_index",
        help="Path to universal file index"
    )
    parser.add_argument(
        "--lsi-index",
        default="./hybrid_index",
        help="Path to LSI/hybrid index"
    )
    parser.add_argument(
        "--functionality",
        required=True,
        help="What to implement (e.g., 'payment drawdown processing')"
    )
    parser.add_argument(
        "--language",
        default="python",
        choices=['python', 'tal', 'java', 'c', 'cpp', 'sql'],
        help="Target programming language"
    )
    parser.add_argument(
        "--requirements",
        default="",
        help="Additional requirements or constraints"
    )
    parser.add_argument(
        "--output",
        help="Output file for generated code"
    )
    parser.add_argument(
        "--batch",
        nargs='+',
        help="List of functionalities for batch generation"
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = CodeGenerationPipeline(
        universal_index=args.universal_index,
        lsi_index=args.lsi_index
    )
    
    if args.batch:
        # Batch generation
        results = pipeline.batch_generate(
            functionalities=args.batch,
            language=args.language,
            output_dir="./generated_code"
        )
    else:
        # Single function generation
        result = pipeline.generate_code(
            functionality=args.functionality,
            language=args.language,
            requirements=args.requirements,
            verbose=True
        )
        
        # Save if output specified
        if args.output:
            with open(args.output, 'w') as f:
                f.write(result['generated_code'])
            print(f"\n✓ Saved to: {args.output}")
        else:
            print("\n" + "="*80)
            print("GENERATED CODE:")
            print("="*80)
            print(result['generated_code'])


print("\n" + "="*80)
print("Code Generation Pipeline Ready!")
print("="*80)
print("\nUsage examples:")
print("  python integration_example.py --functionality 'drawdown processing' --language tal")
print("  python integration_example.py --batch 'payment init' 'ofac screen' --language python")
print("\nOr use in Python:")
print("  from integration_example import CodeGenerationPipeline")
print("  pipeline = CodeGenerationPipeline(...)")
print("  result = pipeline.generate_code('drawdown processing', language='tal')")
