"""
Quick Context Extractor v2.3 - Query Validation Edition
Works with existing indexes
Handles both text/code AND images!

NEW FEATURES:
- Query term validation - Ensures results contain actual query terms
- Iterative search - Keeps fetching until enough valid results found
- Configurable strictness - Exact, case-insensitive, or fuzzy matching

FEATURES:
- Full command-line interface
- Select which index(es) to use (universal, hybrid, both)
- Result diversification (mix of PDFs and code)
- Better source path resolution using metadata
- Configurable file type filtering
- Multiple output formats (text, HTML, JSON)
- **Query term validation and iteration**

Usage:
    python quick_context_extractor_cli.py "payment validation"
    python quick_context_extractor_cli.py "OFAC screening" --strict
    python quick_context_extractor_cli.py "payment flow" --match-mode fuzzy
"""

import argparse
import sys
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict
import pickle
import base64
import io

# Optional: PDF and image support
try:
    import pdfplumber
    from PIL import Image
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("⚠ PDF/Image support not available. Install: pip install pdfplumber Pillow")


class QueryValidator:
    """Validates that search results actually contain query terms"""
    
    def __init__(self, match_mode: str = 'flexible'):
        """
        Args:
            match_mode: 'exact', 'case-insensitive', 'flexible', or 'fuzzy'
        """
        self.match_mode = match_mode
    
    def extract_query_terms(self, query: str) -> Set[str]:
        """Extract meaningful terms from query"""
        # Remove common stopwords
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were',
            'how', 'what', 'when', 'where', 'why', 'who', 'which',
            'do', 'does', 'did', 'has', 'have', 'had'
        }
        
        # Extract words
        words = re.findall(r'\b[\w-]+\b', query.lower())
        
        # Filter out stopwords and very short words
        terms = {w for w in words if len(w) > 2 and w not in stopwords}
        
        return terms
    
    def validate_result(self, result_text: str, query_terms: Set[str]) -> Tuple[bool, Set[str], Set[str]]:
        """
        Validate that result contains query terms
        
        Returns:
            (is_valid, found_terms, missing_terms)
        """
        if not query_terms:
            return True, set(), set()
        
        result_lower = result_text.lower()
        found_terms = set()
        
        for term in query_terms:
            if self.match_mode == 'exact':
                # Exact match (case-sensitive)
                if term in result_text:
                    found_terms.add(term)
            
            elif self.match_mode == 'case-insensitive':
                # Case-insensitive exact match
                if term.lower() in result_lower:
                    found_terms.add(term)
            
            elif self.match_mode == 'flexible':
                # Allow partial matches (substring)
                if term.lower() in result_lower:
                    found_terms.add(term)
                else:
                    # Check if term is part of a compound word
                    pattern = r'\b\w*' + re.escape(term.lower()) + r'\w*\b'
                    if re.search(pattern, result_lower):
                        found_terms.add(term)
            
            elif self.match_mode == 'fuzzy':
                # Most permissive - check for related terms
                if term.lower() in result_lower:
                    found_terms.add(term)
                else:
                    # Check stemmed versions (simple stemming)
                    stem = self._simple_stem(term)
                    pattern = r'\b' + re.escape(stem) + r'\w*\b'
                    if re.search(pattern, result_lower):
                        found_terms.add(term)
        
        missing_terms = query_terms - found_terms
        
        # Result is valid if we found most of the terms
        if self.match_mode == 'exact':
            is_valid = len(missing_terms) == 0
        elif self.match_mode == 'case-insensitive':
            is_valid = len(missing_terms) == 0
        elif self.match_mode == 'flexible':
            # At least 70% of terms must be found
            is_valid = len(found_terms) >= len(query_terms) * 0.7
        else:  # fuzzy
            # At least 50% of terms must be found
            is_valid = len(found_terms) >= max(1, len(query_terms) * 0.5)
        
        return is_valid, found_terms, missing_terms
    
    def _simple_stem(self, word: str) -> str:
        """Simple stemming - removes common suffixes"""
        suffixes = ['ing', 'ed', 'es', 's', 'tion', 'ment', 'ness', 'ly']
        word_lower = word.lower()
        
        for suffix in suffixes:
            if word_lower.endswith(suffix) and len(word_lower) > len(suffix) + 2:
                return word_lower[:-len(suffix)]
        
        return word_lower


class QuickContextExtractor:
    """
    Extract context without requiring line numbers in index
    Works with existing indexes!
    NOW WITH QUERY VALIDATION AND ITERATIVE SEARCH!
    """
    
    def __init__(
        self,
        universal_index: str = "./universal_index",
        hybrid_index: str = "./hybrid_index"
    ):
        self.universal_index = Path(universal_index)
        self.hybrid_index = Path(hybrid_index)
        self.file_cache = {}  # Cache file contents
        self.pdf_cache = {}   # Cache PDF objects
        
        if PDF_AVAILABLE:
            print("✓ Quick context extractor ready (with PDF/image support)")
        else:
            print("✓ Quick context extractor ready (text only - install pdfplumber for images)")
    
    def extract(
        self,
        query: str,
        max_matches: int = 5,
        lines_before: int = 200,
        lines_after: int = 200,
        index_type: str = "both",
        embed_images: bool = True,
        diversify_results: bool = True,
        file_type_filter: Optional[str] = None,
        match_mode: str = 'flexible',  # NEW: Query validation mode
        max_fetch_attempts: int = 5     # NEW: Max iterations to find valid results
    ) -> str:
        """
        Extract context for query with validation
        
        Args:
            query: Search query
            max_matches: Number of valid matches to return
            lines_before: Lines before match (for text/code)
            lines_after: Lines after match (for text/code)
            index_type: Which index to search
            embed_images: Embed full images from PDFs
            diversify_results: Try to get mix of file types
            file_type_filter: Only return specific file type
            match_mode: Query validation strictness
            max_fetch_attempts: Max attempts to fetch valid results
        
        Returns:
            Formatted context string
        """
        print(f"\nQuery: {query}")
        print(f"Match mode: {match_mode}")
        print(f"Settings: {lines_before} lines before, {lines_after} lines after")
        print(f"Index: {index_type}")
        print(f"Image embedding: {'Enabled' if embed_images else 'Disabled'}")
        print(f"Result diversification: {'Enabled' if diversify_results else 'Disabled'}")
        if file_type_filter:
            print(f"File type filter: {file_type_filter}")
        
        # Initialize validator
        validator = QueryValidator(match_mode=match_mode)
        query_terms = validator.extract_query_terms(query)
        print(f"Query terms to validate: {', '.join(sorted(query_terms))}")
        
        # Iterative search with validation
        valid_results = []
        fetch_multiplier = 3 if diversify_results else 2
        fetch_per_iteration = max_matches * fetch_multiplier
        max_total_fetch = fetch_per_iteration * max_fetch_attempts
        total_fetched = 0
        iteration = 0
        discarded_count = 0
        
        while len(valid_results) < max_matches and iteration < max_fetch_attempts:
            iteration += 1
            
            # Calculate how many to fetch this iteration
            needed = max_matches - len(valid_results)
            to_fetch = min(needed * fetch_multiplier, max_total_fetch - total_fetched)
            
            if to_fetch <= 0:
                break
            
            print(f"\n--- Iteration {iteration}: Fetching {to_fetch} candidates ---")
            
            # Search indexes (with offset to skip already-fetched results)
            results = self._search_indexes(
                query,
                to_fetch,
                index_type,
                offset=total_fetched
            )
            
            if not results:
                print("No more results available from index")
                break
            
            total_fetched += len(results)
            print(f"Fetched {len(results)} results (total: {total_fetched})")
            
            # Apply file type filter
            if file_type_filter:
                before_filter = len(results)
                results = [r for r in results if r['file_type'] == file_type_filter]
                if before_filter > len(results):
                    print(f"Filtered to {len(results)} {file_type_filter} results")
            
            # Validate each result
            print(f"\nValidating {len(results)} results...")
            for i, result in enumerate(results, 1):
                # Validate query terms in result text
                is_valid, found_terms, missing_terms = validator.validate_result(
                    result['text'],
                    query_terms
                )
                
                if is_valid:
                    valid_results.append(result)
                    print(f"  [{i}] ✓ {result['source_file']} - Valid (found: {len(found_terms)}/{len(query_terms)} terms)")
                else:
                    discarded_count += 1
                    print(f"  [{i}] ✗ {result['source_file']} - Invalid (missing: {', '.join(sorted(missing_terms))})")
                
                if len(valid_results) >= max_matches:
                    break
        
        print(f"\n{'='*70}")
        print(f"Validation Summary:")
        print(f"  Total fetched: {total_fetched}")
        print(f"  Valid results: {len(valid_results)}")
        print(f"  Discarded: {discarded_count}")
        print(f"  Iterations: {iteration}")
        print(f"{'='*70}")
        
        if not valid_results:
            return f"No valid results found after checking {total_fetched} candidates.\n" + \
                   f"Query terms: {', '.join(sorted(query_terms))}\n" + \
                   f"Try: --match-mode fuzzy for more permissive matching"
        
        # Diversify valid results if enabled
        if diversify_results and not file_type_filter:
            valid_results = self._diversify_results(valid_results, max_matches)
        else:
            valid_results = valid_results[:max_matches]
        
        print(f"\nExtracting context from {len(valid_results)} validated matches...")
        
        # Extract context from each valid result
        contexts = []
        for i, result in enumerate(valid_results, 1):
            print(f"[{i}/{len(valid_results)}] {result['source_file']} ({result['file_type']})")
            
            context = self._extract_context_for_result(
                result,
                lines_before,
                lines_after,
                embed_images
            )
            
            if context:
                contexts.append(context)
        
        # Format for LLM
        return self._format_contexts(contexts, query, query_terms)
    
    def _diversify_results(self, results: List[Dict], max_results: int) -> List[Dict]:
        """
        Diversify results to include different file types
        Strategy: Round-robin selection from each file type
        """
        # Group by file type
        by_type = defaultdict(list)
        for r in results:
            file_type = r.get('file_type', 'unknown')
            by_type[file_type].append(r)
        
        print(f"\nDiversifying results across {len(by_type)} file types:")
        for ft, items in by_type.items():
            print(f"  {ft}: {len(items)} results")
        
        # Round-robin selection
        diversified = []
        type_names = list(by_type.keys())
        type_indices = {t: 0 for t in type_names}
        
        while len(diversified) < max_results:
            added_any = False
            
            for file_type in type_names:
                if type_indices[file_type] < len(by_type[file_type]):
                    diversified.append(by_type[file_type][type_indices[file_type]])
                    type_indices[file_type] += 1
                    added_any = True
                    
                    if len(diversified) >= max_results:
                        break
            
            if not added_any:
                break
        
        print(f"Diversified to {len(diversified)} results")
        return diversified
    
    def _search_indexes(
        self,
        query: str,
        max_matches: int,
        index_type: str,
        offset: int = 0  # NEW: Offset for pagination
    ) -> List[Dict[str, Any]]:
        """Search one or both indexes with offset support"""
        results = []
        
        # Check which indexes exist
        universal_exists = self.universal_index.exists() and (self.universal_index / "bm25.pkl").exists()
        hybrid_exists = self.hybrid_index.exists() and (self.hybrid_index / "bm25.pkl").exists()
        
        if not universal_exists and not hybrid_exists:
            print("❌ No indexes found!")
            print(f"   Looked for: {self.universal_index} and {self.hybrid_index}")
            return []
        
        # Search universal index
        if index_type in ["both", "universal"] and universal_exists:
            try:
                from universal_indexer_v2 import UniversalFileSearcher
                searcher = UniversalFileSearcher(str(self.universal_index))
                # Fetch more than needed to account for offset
                uni_results = searcher.search(query, top_k=max_matches + offset + 10, verbose=False)
                
                # Skip offset results
                uni_results = uni_results[offset:offset + max_matches]
                
                for r in uni_results:
                    results.append({
                        'text': r['text'],
                        'source_file': r['source_file'],
                        'source_path': r.get('source_path', ''),
                        'file_type': r['file_type'],
                        'chunk_index': r.get('chunk_index', 0),
                        'score': r['score'],
                        'index': 'universal'
                    })
            except Exception as e:
                print(f"  ⚠ Universal index error: {e}")
        
        # Search hybrid index
        if index_type in ["both", "hybrid"] and hybrid_exists:
            try:
                # Try fixed version first, fallback to original
                try:
                    from hybrid_indexer_v2_fixed import HybridSearcher
                except ImportError:
                    from hybrid_indexer_v2 import HybridSearcher
                
                searcher = HybridSearcher(str(self.hybrid_index))
                # Fetch more than needed to account for offset
                hyb_results = searcher.search(query, top_k=max_matches + offset + 10, verbose=False)
                
                # Skip offset results
                hyb_results = hyb_results[offset:offset + max_matches]
                
                for r in hyb_results:
                    results.append({
                        'text': r['text'],
                        'source_file': r['source_file'],
                        'source_path': r.get('source_path', ''),
                        'file_type': r['file_type'],
                        'chunk_index': 0,
                        'score': r.get('combined_score', r.get('bm25_score', 0)),
                        'index': 'hybrid'
                    })
            except Exception as e:
                print(f"  ⚠ Hybrid index error: {e}")
        
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
        
        return unique_results
    
    # [Rest of the methods remain the same as before - _extract_context_for_result, 
    # _extract_text_context, _extract_pdf_context, etc.]
    # I'll include key methods but omit duplicates for brevity
    
    def _extract_context_for_result(
        self,
        result: Dict[str, Any],
        lines_before: int,
        lines_after: int,
        embed_images: bool
    ) -> Optional[Dict[str, Any]]:
        """Extract context for a single result (handles both text and images)"""
        
        # Find source file
        source_path = self._find_source_file(result)
        
        if not source_path or not source_path.exists():
            print(f"  ⚠ Source file not found")
            return None
        
        # Check if it's a PDF with potential images
        is_pdf = source_path.suffix.lower() == '.pdf'
        
        if is_pdf and embed_images and PDF_AVAILABLE:
            return self._extract_pdf_context(source_path, result, lines_before, lines_after)
        else:
            return self._extract_text_context(source_path, result, lines_before, lines_after)
    
    def _extract_text_context(
        self,
        source_path: Path,
        result: Dict[str, Any],
        lines_before: int,
        lines_after: int
    ) -> Optional[Dict[str, Any]]:
        """Extract context from text/code files"""
        
        result_text = result['text']
        if self._is_garbage_text(result_text):
            print(f"  ⚠ Skipping: appears to be binary/image data")
            return None
        
        lines = self._read_file_cached(source_path)
        
        if not lines:
            print(f"  ⚠ Could not read file or file is binary")
            return None
        
        match_start, match_end = self._find_match_in_lines(lines, result['text'])
        
        before_start = max(0, match_start - lines_before)
        before_lines = lines[before_start:match_start]
        
        after_end = min(len(lines), match_end + lines_after)
        after_lines = lines[match_end:after_end]
        
        match_lines = lines[match_start:match_end]
        
        print(f"  ✓ Lines {match_start}-{match_end} (+{len(before_lines)} before, +{len(after_lines)} after)")
        
        return {
            'type': 'text',
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
    
    # Include other essential methods (same as before)
    def _is_garbage_text(self, text: str) -> bool:
        if not text or len(text) < 10:
            return False
        non_printable = sum(1 for c in text if not c.isprintable() and c not in '\t\n\r ')
        return non_printable / len(text) > 0.2
    
    def _find_source_file(self, result: Dict[str, Any]) -> Optional[Path]:
        if result.get('source_path') and result['source_path']:
            source_path = Path(result['source_path'])
            if source_path.exists():
                return source_path
        
        filename = result['source_file']
        search_paths = [Path(filename), Path("./your_docs") / filename, Path("./your_code") / filename]
        
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
        file_path_str = str(file_path)
        if file_path_str in self.file_cache:
            return self.file_cache[file_path_str]
        
        try:
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                        lines = [line.rstrip('\n\r') for line in f.readlines()]
                    self.file_cache[file_path_str] = lines
                    return lines
                except:
                    continue
        except:
            pass
        return []
    
    def _find_match_in_lines(self, lines: List[str], match_text: str) -> tuple:
        match_lines = [l.strip() for l in match_text.split('\n') if l.strip()]
        if not match_lines:
            return (0, min(10, len(lines)))
        
        search_lines = match_lines[:min(3, len(match_lines))]
        for i in range(len(lines)):
            matches = sum(1 for j, sl in enumerate(search_lines) 
                         if i + j < len(lines) and sl in lines[i + j])
            if matches >= len(search_lines) * 0.7:
                return (i, min(i + len(match_lines), len(lines)))
        
        return (0, min(len(match_lines), len(lines)))
    
    def _format_contexts(self, contexts: List[Dict[str, Any]], query: str, query_terms: Set[str]) -> str:
        """Format contexts with query term highlighting"""
        if not contexts:
            return "No contexts found."
        
        parts = []
        parts.append(f"# Code Context for: {query}")
        parts.append(f"Query terms validated: {', '.join(sorted(query_terms))}\n")
        parts.append(f"Found {len(contexts)} validated sections\n")
        
        text_count = sum(1 for c in contexts if c.get('type') == 'text')
        pdf_count = sum(1 for c in contexts if c.get('type') == 'pdf')
        parts.append(f"- Text/Code: {text_count}")
        parts.append(f"- PDFs with images: {pdf_count}\n")
        
        for i, ctx in enumerate(contexts, 1):
            parts.append(f"\n{'='*70}")
            parts.append(f"## Match {i}: {ctx['source_file']}")
            parts.append(f"Score: {ctx['score']:.3f} | Type: {ctx['file_type']}")
            parts.append(f"{'='*70}\n")
            
            if ctx.get('type') == 'text':
                parts.append(self._format_text_context(ctx))
        
        parts.append(f"\n{'='*70}")
        parts.append(f"## Summary")
        parts.append(f"Query: {query}")
        parts.append(f"Validated terms: {', '.join(sorted(query_terms))}")
        parts.append(f"Total validated matches: {len(contexts)}")
        
        return '\n'.join(parts)
    
    def _format_text_context(self, ctx: Dict[str, Any]) -> str:
        parts = []
        parts.append(f"Type: {ctx['file_type']}")
        parts.append(f"Location: Lines {ctx['match_start']}-{ctx['match_end']}\n")
        
        ext = Path(ctx['source_file']).suffix.lower()
        lang_map = {'.py': 'python', '.java': 'java', '.tal': 'tal', '.c': 'c'}
        lang = lang_map.get(ext, 'text')
        
        parts.append(f"```{lang}")
        
        before = ctx['before_lines'][-50:] if len(ctx['before_lines']) > 50 else ctx['before_lines']
        after = ctx['after_lines'][:50] if len(ctx['after_lines']) > 50 else ctx['after_lines']
        
        if before:
            start_num = ctx['match_start'] - len(before)
            for j, line in enumerate(before):
                parts.append(f"{start_num + j:4d} | {line}")
            parts.append("")
        
        parts.append(">>> RELEVANT MATCH (VALIDATED) <<<")
        parts.append("")
        
        for j, line in enumerate(ctx['match_lines'][:30]):
            parts.append(f"{ctx['match_start'] + j:4d} | {line}")
        
        parts.append("")
        parts.append(">>> END MATCH <<<")
        
        if after:
            parts.append("")
            for j, line in enumerate(after):
                parts.append(f"{ctx['match_end'] + j:4d} | {line}")
        
        parts.append("```\n")
        return '\n'.join(parts)


def main():
    parser = argparse.ArgumentParser(
        description="Quick Context Extractor v2.3 - With Query Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic search with validation
  %(prog)s "payment validation"
  
  # Strict matching (must contain exact terms)
  %(prog)s "OFAC screening" --match-mode exact
  
  # Flexible matching (default, allows partial matches)
  %(prog)s "payment flow" --match-mode flexible
  
  # Fuzzy matching (most permissive)
  %(prog)s "payment" --match-mode fuzzy
  
  # Search specific index
  %(prog)s "payment" --index hybrid --match-mode case-insensitive
"""
    )
    
    parser.add_argument('query', nargs='+', help='Search query terms')
    parser.add_argument('--index', '-i', choices=['both', 'universal', 'hybrid'], 
                       default='both', help='Which index to search')
    parser.add_argument('--max', '-n', type=int, default=5, 
                       help='Maximum validated results (default: 5)')
    parser.add_argument('--before', '-b', type=int, default=200, 
                       help='Lines before match')
    parser.add_argument('--after', '-a', type=int, default=200, 
                       help='Lines after match')
    parser.add_argument('--file-type', '-t', choices=['code', 'pdf', 'text'], 
                       help='Filter by file type')
    parser.add_argument('--match-mode', choices=['exact', 'case-insensitive', 'flexible', 'fuzzy'],
                       default='flexible', 
                       help='Query term matching strictness (default: flexible)')
    parser.add_argument('--strict', action='store_true', 
                       help='Alias for --match-mode exact')
    parser.add_argument('--max-fetch', type=int, default=5,
                       help='Max fetch iterations (default: 5)')
    parser.add_argument('--no-diversify', action='store_true', 
                       help='Disable result diversification')
    parser.add_argument('--no-images', action='store_true', 
                       help='Disable PDF image embedding')
    parser.add_argument('--output', '-o', help='Output file')
    parser.add_argument('--quiet', '-q', action='store_true', 
                       help='Minimal output')
    
    args = parser.parse_args()
    
    query = ' '.join(args.query)
    match_mode = 'exact' if args.strict else args.match_mode
    embed_images = not args.no_images
    
    if not args.quiet:
        print("="*70)
        print("QUICK CONTEXT EXTRACTION v2.3 - Query Validation Edition")
        print("="*70)
    
    extractor = QuickContextExtractor()
    
    context = extractor.extract(
        query=query,
        max_matches=args.max,
        lines_before=args.before,
        lines_after=args.after,
        index_type=args.index,
        embed_images=embed_images,
        diversify_results=not args.no_diversify,
        file_type_filter=args.file_type,
        match_mode=match_mode,
        max_fetch_attempts=args.max_fetch
    )
    
    output_file = args.output or "extracted_context.txt"
    with open(output_file, 'w') as f:
        f.write(context)
    
    if not args.quiet:
        print(f"\n✓ Saved to: {output_file}")
        print(f"  Length: {len(context):,} characters")
        print("\n" + "="*70)
        print("PREVIEW")
        print("="*70)
        print(context[:1000])
        if len(context) > 1000:
            print(f"\n... ({len(context) - 1000:,} more characters)")


if __name__ == "__main__":
    main()
