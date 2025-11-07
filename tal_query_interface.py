"""
TAL Code Query Interface - Interactive search and RAG integration

This module provides:
1. Interactive command-line query interface
2. Context retrieval for RAG systems
3. Multi-mode search (TF-IDF, capability-based, graph traversal)
4. Result formatting for LLM consumption
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import tal_code_indexer


class TALQueryInterface:
    """
    Interactive query interface for indexed TAL code.
    Provides multiple search modes optimized for different use cases.
    """
    
    def __init__(self, indexer: tal_code_indexer.TALCodeIndexer):
        """
        Initialize the query interface.
        
        Args:
            indexer: Populated TAL code indexer instance
        """
        self.indexer = indexer
        self.query_history = []
    
    def search_by_keywords(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        TF-IDF keyword search for code elements.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            
        Returns:
            List of formatted search results
        """
        results = self.indexer.search(query, top_k=top_k)
        
        formatted_results = []
        for rank, (name, score, element) in enumerate(results, 1):
            formatted_results.append({
                'rank': rank,
                'score': score,
                'name': element.name,
                'type': element.element_type,
                'file': element.file_path,
                'line': element.line_number,
                'return_type': element.return_type,
                'parameters': element.parameters,
                'calls': element.calls,
                'capabilities': list(element.business_capabilities),
                'content': element.content
            })
        
        return formatted_results
    
    def search_by_capability(self, capability_id: str) -> Dict[str, Any]:
        """
        Find all code implementing a business capability.
        
        Args:
            capability_id: Business capability identifier
            
        Returns:
            Dictionary with capability info and implementing functions
        """
        if capability_id not in self.indexer.capabilities:
            return {
                'error': f"Capability '{capability_id}' not found",
                'available': list(self.indexer.capabilities.keys())
            }
        
        capability = self.indexer.capabilities[capability_id]
        functions = self.indexer.get_capability_functions(capability_id)
        
        return {
            'capability_id': capability_id,
            'name': capability.name,
            'description': capability.description,
            'keywords': list(capability.keywords),
            'function_count': len(functions),
            'functions': [
                {
                    'name': func.name,
                    'type': func.element_type,
                    'file': func.file_path,
                    'line': func.line_number,
                    'return_type': func.return_type,
                    'parameters': func.parameters,
                    'calls': func.calls[:10],  # Limit for readability
                }
                for func in functions
            ]
        }
    
    def get_function_context(self, function_name: str, include_callers: bool = True,
                           include_callees: bool = True, max_depth: int = 2) -> Dict[str, Any]:
        """
        Get complete context for a function (for RAG).
        
        Args:
            function_name: Function to get context for
            include_callers: Include functions that call this function
            include_callees: Include functions called by this function
            max_depth: Maximum depth for call chain traversal
            
        Returns:
            Comprehensive function context
        """
        if function_name not in self.indexer.elements:
            return {'error': f"Function '{function_name}' not found"}
        
        element = self.indexer.elements[function_name]
        
        context = {
            'function': {
                'name': element.name,
                'type': element.element_type,
                'file': element.file_path,
                'line': element.line_number,
                'return_type': element.return_type,
                'parameters': element.parameters,
                'local_variables': element.local_variables,
                'content': element.content,
                'business_capabilities': list(element.business_capabilities)
            }
        }
        
        # Add callers
        if include_callers:
            callers = self.indexer.get_callers(function_name)
            context['callers'] = [
                {
                    'name': c.name,
                    'file': c.file_path,
                    'line': c.line_number,
                    'capabilities': list(c.business_capabilities)
                }
                for c in callers
            ]
        
        # Add callees
        if include_callees:
            context['callees'] = [
                {
                    'name': callee,
                    'exists': callee in self.indexer.elements,
                    'capabilities': list(self.indexer.elements[callee].business_capabilities)
                    if callee in self.indexer.elements else []
                }
                for callee in element.calls
            ]
        
        # Add call chains
        chains = self.indexer.get_call_chain(function_name, max_depth=max_depth)
        context['call_chains'] = [
            ' → '.join(chain) for chain in chains[:10]  # Limit for readability
        ]
        
        # Add related capabilities
        if element.business_capabilities:
            related_caps = []
            for cap_id in element.business_capabilities:
                if cap_id in self.indexer.capabilities:
                    cap = self.indexer.capabilities[cap_id]
                    related_caps.append({
                        'id': cap_id,
                        'name': cap.name,
                        'description': cap.description
                    })
            context['capabilities'] = related_caps
        
        return context
    
    def find_related_functions(self, function_name: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Find functions related to a given function.
        Uses capability overlap and call graph proximity.
        
        Args:
            function_name: Source function
            top_k: Number of related functions to return
            
        Returns:
            List of related functions with relationship explanations
        """
        if function_name not in self.indexer.elements:
            return []
        
        source = self.indexer.elements[function_name]
        related = []
        
        for name, element in self.indexer.elements.items():
            if name == function_name:
                continue
            
            relationship_score = 0
            reasons = []
            
            # Capability overlap
            cap_overlap = source.business_capabilities & element.business_capabilities
            if cap_overlap:
                relationship_score += len(cap_overlap) * 3
                reasons.append(f"shares {len(cap_overlap)} capability(ies)")
            
            # Call relationship
            if name in source.calls:
                relationship_score += 5
                reasons.append("called by source")
            
            if function_name in element.calls:
                relationship_score += 5
                reasons.append("calls source")
            
            # Same file
            if element.file_path == source.file_path:
                relationship_score += 1
                reasons.append("same file")
            
            # Keyword overlap
            keyword_overlap = source.keywords & element.keywords
            if keyword_overlap:
                relationship_score += len(keyword_overlap) * 0.5
            
            if relationship_score > 0:
                related.append({
                    'name': name,
                    'type': element.element_type,
                    'file': element.file_path,
                    'line': element.line_number,
                    'score': relationship_score,
                    'relationship': ', '.join(reasons),
                    'capabilities': list(element.business_capabilities)
                })
        
        # Sort by score and return top-k
        related.sort(key=lambda x: x['score'], reverse=True)
        return related[:top_k]
    
    def format_for_llm_context(self, query_results: List[Dict[str, Any]], 
                               max_tokens: int = 4000) -> str:
        """
        Format search results as context for LLM consumption.
        
        Args:
            query_results: Results from any search method
            max_tokens: Approximate token limit for context
            
        Returns:
            Formatted context string
        """
        context_parts = []
        estimated_tokens = 0
        
        for result in query_results:
            # Build function description
            func_desc = f"Function: {result['name']}\n"
            func_desc += f"Type: {result['type']}\n"
            func_desc += f"File: {Path(result['file']).name} (line {result['line']})\n"
            
            if result.get('return_type'):
                func_desc += f"Returns: {result['return_type']}\n"
            
            if result.get('parameters'):
                params = ', '.join(result['parameters'])
                func_desc += f"Parameters: {params}\n"
            
            if result.get('capabilities'):
                caps = ', '.join(result['capabilities'])
                func_desc += f"Business Capabilities: {caps}\n"
            
            if result.get('calls'):
                calls = ', '.join(result['calls'][:5])
                func_desc += f"Calls: {calls}\n"
            
            func_desc += "\n"
            
            # Rough token estimate (1 token ≈ 4 characters)
            estimated_tokens += len(func_desc) // 4
            
            if estimated_tokens > max_tokens:
                break
            
            context_parts.append(func_desc)
        
        return "=== CODE CONTEXT ===\n\n" + "\n---\n\n".join(context_parts)
    
    def interactive_search(self):
        """
        Interactive command-line search interface.
        """
        print("\n" + "=" * 60)
        print("TAL CODE QUERY INTERFACE")
        print("=" * 60)
        print("\nCommands:")
        print("  search <query>         - Keyword search")
        print("  capability <cap_id>    - Search by business capability")
        print("  context <function>     - Get complete function context")
        print("  related <function>     - Find related functions")
        print("  list capabilities      - List all capabilities")
        print("  stats                  - Show index statistics")
        print("  help                   - Show this help")
        print("  quit                   - Exit")
        print()
        
        while True:
            try:
                command = input("\nQuery> ").strip()
                
                if not command:
                    continue
                
                parts = command.split(maxsplit=1)
                cmd = parts[0].lower()
                arg = parts[1] if len(parts) > 1 else ""
                
                if cmd == 'quit' or cmd == 'exit':
                    print("Goodbye!")
                    break
                
                elif cmd == 'help':
                    print("\nCommands:")
                    print("  search <query>         - Keyword search")
                    print("  capability <cap_id>    - Search by business capability")
                    print("  context <function>     - Get complete function context")
                    print("  related <function>     - Find related functions")
                    print("  list capabilities      - List all capabilities")
                    print("  stats                  - Show index statistics")
                
                elif cmd == 'search':
                    if not arg:
                        print("Usage: search <query>")
                        continue
                    
                    print(f"\nSearching for: '{arg}'")
                    results = self.search_by_keywords(arg, top_k=10)
                    
                    if results:
                        for r in results:
                            print(f"\n{r['rank']}. {r['name']} (score: {r['score']:.3f})")
                            print(f"   Type: {r['type']}, File: {Path(r['file']).name}:{r['line']}")
                            if r['capabilities']:
                                print(f"   Capabilities: {', '.join(r['capabilities'])}")
                    else:
                        print("No results found")
                
                elif cmd == 'capability':
                    if not arg:
                        print("Usage: capability <capability_id>")
                        continue
                    
                    result = self.search_by_capability(arg)
                    
                    if 'error' in result:
                        print(f"\n{result['error']}")
                        if 'available' in result:
                            print("\nAvailable capabilities:")
                            for cap_id in result['available'][:10]:
                                print(f"  - {cap_id}")
                    else:
                        print(f"\n{result['name']} ({result['capability_id']})")
                        print(f"{result['description']}")
                        print(f"\nImplemented by {result['function_count']} function(s):")
                        for func in result['functions'][:10]:
                            print(f"  - {func['name']} ({Path(func['file']).name}:{func['line']})")
                
                elif cmd == 'context':
                    if not arg:
                        print("Usage: context <function_name>")
                        continue
                    
                    context = self.get_function_context(arg)
                    
                    if 'error' in context:
                        print(f"\n{context['error']}")
                    else:
                        func = context['function']
                        print(f"\n{func['name']} ({func['type']})")
                        print(f"File: {func['file']}:{func['line']}")
                        
                        if func['return_type']:
                            print(f"Returns: {func['return_type']}")
                        
                        if func['parameters']:
                            print(f"Parameters: {', '.join(func['parameters'])}")
                        
                        if func['business_capabilities']:
                            print(f"Capabilities: {', '.join(func['business_capabilities'])}")
                        
                        if context.get('callers'):
                            print(f"\nCalled by {len(context['callers'])} function(s):")
                            for caller in context['callers'][:5]:
                                print(f"  ← {caller['name']}")
                        
                        if context.get('callees'):
                            print(f"\nCalls {len(context['callees'])} function(s):")
                            for callee in context['callees'][:5]:
                                print(f"  → {callee['name']}")
                
                elif cmd == 'related':
                    if not arg:
                        print("Usage: related <function_name>")
                        continue
                    
                    related = self.find_related_functions(arg, top_k=10)
                    
                    if related:
                        print(f"\nFunctions related to '{arg}':")
                        for r in related:
                            print(f"\n  {r['name']} (score: {r['score']:.1f})")
                            print(f"    {r['relationship']}")
                            print(f"    File: {Path(r['file']).name}:{r['line']}")
                            if r['capabilities']:
                                print(f"    Capabilities: {', '.join(r['capabilities'])}")
                    else:
                        print(f"\nNo related functions found for '{arg}'")
                
                elif cmd == 'list' and arg == 'capabilities':
                    print("\nAvailable Business Capabilities:")
                    for cap_id, cap in self.indexer.capabilities.items():
                        func_count = len(cap.implementing_functions)
                        print(f"\n  {cap.name}")
                        print(f"    ID: {cap_id}")
                        print(f"    {cap.description}")
                        print(f"    Implemented by {func_count} function(s)")
                
                elif cmd == 'stats':
                    self.indexer.print_statistics()
                
                else:
                    print(f"Unknown command: '{cmd}'. Type 'help' for available commands.")
                
            except KeyboardInterrupt:
                print("\n\nInterrupted. Type 'quit' to exit.")
                continue
            except Exception as e:
                print(f"\nError: {e}")
                continue


def main():
    """Main entry point for interactive query interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Interactive query interface for indexed TAL code'
    )
    
    parser.add_argument('index_file', help='Path to binary index file (.pkl)')
    parser.add_argument('--query', help='Execute single query and exit')
    parser.add_argument('--format', choices=['text', 'json'], default='text',
                       help='Output format')
    
    args = parser.parse_args()
    
    # Load the index
    try:
        indexer = tal_code_indexer.TALCodeIndexer.load_binary(args.index_file)
    except Exception as e:
        print(f"Error loading index: {e}")
        return 1
    
    # Create query interface
    interface = TALQueryInterface(indexer)
    
    # Execute single query if provided
    if args.query:
        results = interface.search_by_keywords(args.query, top_k=10)
        
        if args.format == 'json':
            print(json.dumps(results, indent=2))
        else:
            for r in results:
                print(f"\n{r['rank']}. {r['name']} (score: {r['score']:.3f})")
                print(f"   Type: {r['type']}, File: {Path(r['file']).name}:{r['line']}")
                if r['capabilities']:
                    print(f"   Capabilities: {', '.join(r['capabilities'])}")
        
        return 0
    
    # Start interactive mode
    interface.interactive_search()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
