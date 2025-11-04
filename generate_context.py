"""
Quick Context Extractor - Works with existing indexes
No re-indexing needed!

Extracts 200 lines before/after matches even if indexes don't store line numbers.
Optimized for speed with caching.

Usage:
    from quick_context_extractor import QuickContextExtractor
    
    extractor = QuickContextExtractor()
    context = extractor.extract(
        query="payment drawdown",
        lines_before=200,
        lines_after=200
    )
    print(context)
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict
import pickle
import re


class QuickContextExtractor:
    """
    Extract context without requiring line numbers in index
    Works with existing indexes!
    """
    
    def __init__(
        self,
        universal_index: str = "./universal_index",
        hybrid_index: str = "./hybrid_index"
    ):
        self.universal_index = Path(universal_index)
        self.hybrid_index = Path(hybrid_index)
        self.file_cache = {}  # Cache file contents
        
        print("✓ Quick context extractor ready")
    
    def extract(
        self,
        query: str,
        max_matches: int = 5,
        lines_before: int = 200,
        lines_after: int = 200,
        index_type: str = "both"  # "both", "universal", or "hybrid"
    ) -> str:
        """
        Extract context for query
        
        Args:
            query: Search query
            max_matches: Number of matches
            lines_before: Lines before match
            lines_after: Lines after match  
            index_type: Which index to search
        
        Returns:
            Formatted context string
        """
        print(f"\nQuery: {query}")
        print(f"Extracting {lines_before} lines before, {lines_after} lines after")
        
        # Search indexes
        results = self._search_indexes(query, max_matches, index_type)
        
        if not results:
            return "No results found."
        
        print(f"\nFound {len(results)} matches")
        
        # Extract context from each result
        contexts = []
        for i, result in enumerate(results, 1):
            print(f"[{i}/{len(results)}] {result['source_file']}")
            
            context = self._extract_context_for_result(
                result,
                lines_before,
                lines_after
            )
            
            if context:
                contexts.append(context)
        
        # Format for LLM
        return self._format_contexts(contexts, query)
    
    def _search_indexes(
        self,
        query: str,
        max_matches: int,
        index_type: str
    ) -> List[Dict[str, Any]]:
        """Search one or both indexes"""
        from universal_indexer_v2 import UniversalFileSearcher
        from hybrid_indexer_v2 import HybridSearcher
        
        results = []
        
        # Search universal index
        if index_type in ["both", "universal"] and self.universal_index.exists():
            try:
                searcher = UniversalFileSearcher(str(self.universal_index))
                uni_results = searcher.search(query, top_k=max_matches, verbose=False)
                
                for r in uni_results:
                    results.append({
                        'text': r['text'],
                        'source_file': r['source_file'],
                        'file_type': r['file_type'],
                        'chunk_index': r.get('chunk_index', 0),
                        'score': r['score'],
                        'index': 'universal'
                    })
            except Exception as e:
                print(f"⚠ Universal index error: {e}")
        
        # Search hybrid index
        if index_type in ["both", "hybrid"] and self.hybrid_index.exists():
            try:
                searcher = HybridSearcher(str(self.hybrid_index))
                hyb_results = searcher.search(query, top_k=max_matches, verbose=False)
                
                for r in hyb_results:
                    results.append({
                        'text': r['text'],
                        'source_file': r['source_file'],
                        'file_type': r['file_type'],
                        'chunk_index': 0,
                        'score': r.get('combined_score', r.get('bm25_score', 0)),
                        'index': 'hybrid'
                    })
            except Exception as e:
                print(f"⚠ Hybrid index error: {e}")
        
        # Sort by score and deduplicate
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # Remove duplicates
        seen = set()
        unique_results = []
        for r in results:
            key = f"{r['source_file']}:{r['text'][:100]}"
            if key not in seen:
                seen.add(key)
                unique_results.append(r)
        
        return unique_results[:max_matches]
    
    def _extract_context_for_result(
        self,
        result: Dict[str, Any],
        lines_before: int,
        lines_after: int
    ) -> Optional[Dict[str, Any]]:
        """Extract context for a single result"""
        
        # Find source file
        source_path = self._find_source_file(result)
        
        if not source_path or not source_path.exists():
            print(f"  ⚠ Source file not found")
            return None
        
        # Read file (with caching)
        lines = self._read_file_cached(source_path)
        
        if not lines:
            print(f"  ⚠ Could not read file")
            return None
        
        # Find match location in file
        match_start, match_end = self._find_match_in_lines(
            lines,
            result['text']
        )
        
        # Extract context
        before_start = max(0, match_start - lines_before)
        before_lines = lines[before_start:match_start]
        
        after_end = min(len(lines), match_end + lines_after)
        after_lines = lines[match_end:after_end]
        
        match_lines = lines[match_start:match_end]
        
        print(f"  ✓ Lines {match_start}-{match_end} (+{len(before_lines)} before, +{len(after_lines)} after)")
        
        return {
            'source_file': result['source_file'],
            'source_path': source_path,
            'match_start': match_start,
            'match_end': match_end,
            'before_lines': before_lines,
            'match_lines': match_lines,
            'after_lines': after_lines,
            'score': result['score'],
            'file_type': result['file_type']
        }
    
    def _find_source_file(self, result: Dict[str, Any]) -> Optional[Path]:
        """Find the source file path"""
        filename = result['source_file']
        
        # Try common locations
        search_paths = [
            Path(filename),
            Path("./your_docs") / filename,
            Path("./your_code") / filename,
            Path("./docs") / filename,
            Path("./code") / filename,
            Path("./src") / filename,
        ]
        
        # Also search metadata from index
        if result['index'] == 'universal':
            try:
                with open(self.universal_index / "metadata.pkl", 'rb') as f:
                    metadata_list = pickle.load(f)
                for meta in metadata_list:
                    if meta.get('source_file') == filename:
                        source_path = meta.get('source_path')
                        if source_path:
                            search_paths.insert(0, Path(source_path))
                        break
            except:
                pass
        
        for path in search_paths:
            if path.exists():
                return path
        
        return None
    
    def _read_file_cached(self, file_path: Path) -> List[str]:
        """Read file with caching"""
        file_path_str = str(file_path)
        
        if file_path_str in self.file_cache:
            return self.file_cache[file_path_str]
        
        try:
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        lines = [line.rstrip('\n\r') for line in f.readlines()]
                    
                    self.file_cache[file_path_str] = lines
                    return lines
                except UnicodeDecodeError:
                    continue
        except Exception as e:
            print(f"  Error reading file: {e}")
        
        return []
    
    def _find_match_in_lines(
        self,
        lines: List[str],
        match_text: str
    ) -> tuple:
        """Find where match_text appears in lines"""
        
        # Clean match text
        match_lines = [l.strip() for l in match_text.split('\n') if l.strip()]
        
        if not match_lines:
            return (0, min(10, len(lines)))
        
        # Search for first few lines
        search_lines = match_lines[:min(3, len(match_lines))]
        
        for i in range(len(lines)):
            # Check if we have a match starting at line i
            matches = 0
            for j, search_line in enumerate(search_lines):
                if i + j >= len(lines):
                    break
                
                file_line = lines[i + j].strip()
                
                # Fuzzy match
                if (search_line in file_line or 
                    file_line in search_line or
                    self._similarity(search_line, file_line) > 0.7):
                    matches += 1
            
            # If we matched most of the search lines
            if matches >= len(search_lines) * 0.7:
                start_line = i
                end_line = min(i + len(match_lines), len(lines))
                return (start_line, end_line)
        
        # Fallback: estimate based on line count
        estimated_lines = len(match_lines)
        return (0, min(estimated_lines, len(lines)))
    
    def _similarity(self, s1: str, s2: str) -> float:
        """Compute similarity between two strings"""
        words1 = set(s1.lower().split())
        words2 = set(s2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _format_contexts(
        self,
        contexts: List[Dict[str, Any]],
        query: str
    ) -> str:
        """Format contexts for LLM"""
        if not contexts:
            return "No contexts found."
        
        parts = []
        
        # Header
        parts.append(f"# Code Context for: {query}\n")
        parts.append(f"Found {len(contexts)} relevant sections\n")
        
        # Each context
        for i, ctx in enumerate(contexts, 1):
            parts.append(f"\n{'='*70}")
            parts.append(f"## Match {i}: {ctx['source_file']}")
            parts.append(f"Lines {ctx['match_start']}-{ctx['match_end']}, Score: {ctx['score']:.3f}")
            parts.append(f"{'='*70}\n")
            
            # Determine language for syntax highlighting
            ext = Path(ctx['source_file']).suffix.lower()
            lang_map = {
                '.py': 'python', '.java': 'java', '.c': 'c', '.cpp': 'cpp',
                '.js': 'javascript', '.tal': 'tal', '.cbl': 'cobol',
                '.sql': 'sql', '.sh': 'bash'
            }
            lang = lang_map.get(ext, 'text')
            
            parts.append(f"```{lang}")
            
            # Show limited context
            before = ctx['before_lines'][-50:] if len(ctx['before_lines']) > 50 else ctx['before_lines']
            after = ctx['after_lines'][:50] if len(ctx['after_lines']) > 50 else ctx['after_lines']
            
            # Lines before
            start_num = ctx['match_start'] - len(before)
            for j, line in enumerate(before):
                parts.append(f"{start_num + j:4d} | {line}")
            
            # Match (highlighted)
            if before:
                parts.append("")
            parts.append(">>> RELEVANT MATCH <<<")
            parts.append("")
            
            for j, line in enumerate(ctx['match_lines'][:30]):  # Limit match display
                parts.append(f"{ctx['match_start'] + j:4d} | {line}")
            
            parts.append("")
            parts.append(">>> END MATCH <<<")
            if after:
                parts.append("")
            
            # Lines after
            for j, line in enumerate(after):
                parts.append(f"{ctx['match_end'] + j:4d} | {line}")
            
            parts.append("```\n")
        
        # Summary
        parts.append(f"\n{'='*70}")
        parts.append("## Summary")
        parts.append(f"Total matches: {len(contexts)}")
        parts.append(f"Total lines: {sum(len(c['before_lines']) + len(c['match_lines']) + len(c['after_lines']) for c in contexts)}")
        
        return '\n'.join(parts)


# Quick usage function
def quick_extract(
    query: str,
    lines_before: int = 200,
    lines_after: int = 200,
    max_matches: int = 5
) -> str:
    """
    One-liner: extract context for query
    
    Usage:
        context = quick_extract("payment drawdown", lines_before=200, lines_after=200)
        print(context)
    """
    extractor = QuickContextExtractor()
    return extractor.extract(query, max_matches, lines_before, lines_after)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        query = ' '.join(sys.argv[1:])
    else:
        query = "payment drawdown processing"
    
    print("="*70)
    print("QUICK CONTEXT EXTRACTION")
    print("="*70)
    
    context = quick_extract(query, lines_before=200, lines_after=200, max_matches=5)
    
    print("\n" + "="*70)
    print("RESULT")
    print("="*70)
    print(context)
    
    # Save to file
    with open("extracted_context.txt", 'w') as f:
        f.write(context)
    
    print(f"\n✓ Saved to: extracted_context.txt")
    print(f"  Length: {len(context):,} chars")
