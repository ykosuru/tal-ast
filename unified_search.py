"""
Unified Search System v2.0 - Production Ready
Combines Universal Indexer (PDFs/docs) + Hybrid Indexer (code)
Features: Score normalization, deduplication, context compression

Ready to use, no modifications needed.
"""

import numpy as np
import hashlib
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass

from universal_indexer_v2 import UniversalFileSearcher
from hybrid_indexer_v2 import HybridSearcher


# ============================================================================
# Search Result Model
# ============================================================================

@dataclass
class SearchResult:
    """Standardized search result"""
    text: str
    source_file: str
    file_type: str
    chunk_index: int
    score: float
    source_indexer: str
    keywords: List[str]
    capabilities: List[str]
    language: str
    
    @property
    def id(self) -> str:
        """Unique ID for deduplication"""
        content = f"{self.source_file}:{self.chunk_index}:{self.text[:200]}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'text': self.text,
            'source_file': self.source_file,
            'file_type': self.file_type,
            'chunk_index': self.chunk_index,
            'score': self.score,
            'source_indexer': self.source_indexer,
            'keywords': self.keywords,
            'capabilities': self.capabilities,
            'language': self.language,
            'id': self.id
        }


# ============================================================================
# Score Normalization
# ============================================================================

class ScoreNormalizer:
    """Normalize scores from different indexers to [0, 1] range"""
    
    @staticmethod
    def normalize_bm25(scores: np.ndarray, k: float = 2.0) -> np.ndarray:
        """
        Normalize BM25 scores (0-∞) to [0, 1]
        Uses tanh to map unbounded scores to bounded range
        """
        if len(scores) == 0:
            return scores
        return np.tanh(scores / k)
    
    @staticmethod
    def normalize_minmax(scores: np.ndarray) -> np.ndarray:
        """Min-max normalization to [0, 1]"""
        if len(scores) == 0 or scores.max() == scores.min():
            return np.zeros_like(scores)
        return (scores - scores.min()) / (scores.max() - scores.min())


# ============================================================================
# Deduplication
# ============================================================================

class ResultDeduplicator:
    """Remove duplicate results using fuzzy matching"""
    
    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Compute Jaccard similarity"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def deduplicate(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        Remove duplicates, keeping highest scoring version
        
        Strategy:
        1. Sort by score (highest first)
        2. Remove exact duplicates (same ID)
        3. Remove fuzzy duplicates (similar text)
        """
        if not results:
            return []
        
        # Sort by score
        results = sorted(results, key=lambda r: r.score, reverse=True)
        
        seen_ids = set()
        seen_texts = []
        deduplicated = []
        
        for result in results:
            # Skip exact duplicates
            if result.id in seen_ids:
                continue
            
            # Check fuzzy duplicates
            is_duplicate = False
            for seen_text in seen_texts:
                similarity = self._text_similarity(result.text, seen_text)
                if similarity >= self.similarity_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                seen_ids.add(result.id)
                seen_texts.append(result.text)
                deduplicated.append(result)
        
        return deduplicated


# ============================================================================
# Context Compression
# ============================================================================

class ContextCompressor:
    """Compress search results for LLM consumption"""
    
    def __init__(self, max_tokens: int = 4000):
        self.max_tokens = max_tokens
        self.chars_per_token = 4  # Rough estimate
    
    def compress(
        self,
        results: List[SearchResult],
        query: str,
        focus_on_capabilities: bool = True
    ) -> str:
        """
        Compress results into compact context for LLM
        
        Strategy:
        1. Group by capability or file type
        2. Extract key snippets
        3. Remove redundancy
        4. Format for easy LLM parsing
        """
        if not results:
            return "No relevant results found."
        
        max_chars = self.max_tokens * self.chars_per_token
        
        # Group results
        if focus_on_capabilities:
            grouped = self._group_by_capability(results)
        else:
            grouped = self._group_by_file(results)
        
        # Build context
        context_parts = []
        context_parts.append(f"# Search Results for: {query}\n")
        
        chars_used = len(context_parts[0])
        
        for group_name, group_results in grouped.items():
            if chars_used >= max_chars:
                break
            
            group_text = self._format_group(
                group_name,
                group_results,
                remaining_chars=max_chars - chars_used
            )
            
            if group_text:
                context_parts.append(group_text)
                chars_used += len(group_text)
        
        # Add summary
        summary = self._create_summary(results)
        context_parts.append(f"\n## Summary\n{summary}")
        
        return "\n".join(context_parts)
    
    def _group_by_capability(
        self,
        results: List[SearchResult]
    ) -> Dict[str, List[SearchResult]]:
        """Group by primary capability"""
        grouped = defaultdict(list)
        
        for result in results:
            if result.capabilities:
                primary = result.capabilities[0]
            else:
                primary = result.file_type
            
            grouped[primary].append(result)
        
        # Sort groups by total score
        sorted_groups = sorted(
            grouped.items(),
            key=lambda x: sum(r.score for r in x[1]),
            reverse=True
        )
        
        return dict(sorted_groups)
    
    def _group_by_file(
        self,
        results: List[SearchResult]
    ) -> Dict[str, List[SearchResult]]:
        """Group by source file"""
        grouped = defaultdict(list)
        for result in results:
            grouped[result.source_file].append(result)
        return dict(grouped)
    
    def _format_group(
        self,
        group_name: str,
        results: List[SearchResult],
        remaining_chars: int
    ) -> str:
        """Format a group of results"""
        if remaining_chars < 200:
            return ""
        
        lines = [f"\n## {group_name}\n"]
        chars_used = len(lines[0])
        
        for result in results[:5]:  # Top 5 per group
            snippet = self._extract_snippet(result.text, max_len=300)
            
            result_text = (
                f"**{result.source_file}** "
                f"({result.file_type}, score: {result.score:.2f})\n"
                f"```{result.language or 'text'}\n"
                f"{snippet}\n"
                f"```\n"
            )
            
            if chars_used + len(result_text) > remaining_chars:
                break
            
            lines.append(result_text)
            chars_used += len(result_text)
        
        return "".join(lines) if len(lines) > 1 else ""
    
    def _extract_snippet(self, text: str, max_len: int = 300) -> str:
        """Extract relevant snippet"""
        if len(text) <= max_len:
            return text
        
        # Try to keep complete sentences
        sentences = re.split(r'[.!?]\s+', text)
        
        snippet = ""
        for sentence in sentences:
            if len(snippet) + len(sentence) > max_len:
                break
            snippet += sentence + ". "
        
        if not snippet:
            snippet = text[:max_len] + "..."
        
        return snippet.strip()
    
    def _create_summary(self, results: List[SearchResult]) -> str:
        """Create summary of results"""
        total = len(results)
        
        # Count by file type
        file_types = defaultdict(int)
        for r in results:
            file_types[r.file_type] += 1
        
        # Count by capability
        capabilities = defaultdict(int)
        for r in results:
            for cap in r.capabilities:
                capabilities[cap] += 1
        
        # Top capabilities
        top_caps = sorted(capabilities.items(), key=lambda x: x[1], reverse=True)[:5]
        
        summary = f"Found {total} relevant results:\n"
        summary += f"- File types: {dict(file_types)}\n"
        if top_caps:
            summary += f"- Top capabilities: {', '.join(f'{cap} ({count})' for cap, count in top_caps)}\n"
        
        return summary


# ============================================================================
# Unified Search
# ============================================================================

class UnifiedCodeSearch:
    """
    Unified search interface combining both indexers
    
    Usage:
        search = UnifiedCodeSearch(
            universal_index_path="./universal_index",
            hybrid_index_path="./hybrid_index"
        )
        
        results = search.search("payment drawdown", top_k=20)
        
        compressed = search.get_compressed_context(results, "payment drawdown")
    """
    
    def __init__(
        self,
        universal_index_path: Optional[str] = None,
        hybrid_index_path: Optional[str] = None,
        universal_weight: float = 0.6,
        hybrid_weight: float = 0.4
    ):
        self.universal_weight = universal_weight
        self.hybrid_weight = hybrid_weight
        
        # Load indexers
        self.universal_searcher = None
        self.hybrid_searcher = None
        
        if universal_index_path and Path(universal_index_path).exists():
            try:
                self.universal_searcher = UniversalFileSearcher(universal_index_path)
                print("✓ Universal indexer loaded")
            except Exception as e:
                print(f"⚠ Could not load universal indexer: {e}")
        
        if hybrid_index_path and Path(hybrid_index_path).exists():
            try:
                self.hybrid_searcher = HybridSearcher(hybrid_index_path)
                print("✓ Hybrid indexer loaded")
            except Exception as e:
                print(f"⚠ Could not load hybrid indexer: {e}")
        
        # Utilities
        self.normalizer = ScoreNormalizer()
        self.deduplicator = ResultDeduplicator()
        self.compressor = ContextCompressor()
    
    def search(
        self,
        query: str,
        top_k: int = 20,
        file_type_filter: Optional[str] = None,
        use_query_expansion: bool = True,
        verbose: bool = False
    ) -> List[SearchResult]:
        """
        Search across both indexers with unified scoring
        
        Args:
            query: Search query
            top_k: Number of results
            file_type_filter: Filter by file type
            use_query_expansion: Expand with synonyms
            verbose: Show details
        
        Returns:
            List of deduplicated SearchResult objects
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"Unified Search: {query}")
            print(f"{'='*70}")
        
        all_results = []
        
        # Search universal indexer
        if self.universal_searcher:
            if verbose:
                print("\n[1/2] Searching Universal Indexer...")
            
            try:
                universal_results = self.universal_searcher.search(
                    query,
                    top_k=top_k * 2,
                    file_type_filter=file_type_filter,
                    use_query_expansion=use_query_expansion,
                    verbose=False
                )
                
                # Normalize scores
                scores = np.array([r['score'] for r in universal_results])
                normalized = self.normalizer.normalize_bm25(scores)
                
                for i, result in enumerate(universal_results):
                    all_results.append(SearchResult(
                        text=result['text'],
                        source_file=result['source_file'],
                        file_type=result['file_type'],
                        chunk_index=result.get('chunk_index', 0),
                        score=float(normalized[i]) * self.universal_weight,
                        source_indexer='universal',
                        keywords=result.get('keywords', []),
                        capabilities=result.get('capabilities', []),
                        language=result.get('language', '')
                    ))
                
                if verbose:
                    print(f"  Found {len(universal_results)} results")
            
            except Exception as e:
                if verbose:
                    print(f"  Error: {e}")
        
        # Search hybrid indexer
        if self.hybrid_searcher:
            if verbose:
                print("\n[2/2] Searching Hybrid Indexer...")
            
            try:
                hybrid_results = self.hybrid_searcher.search(
                    query,
                    top_k=top_k * 2,
                    use_query_expansion=use_query_expansion,
                    verbose=False
                )
                
                # Normalize scores
                scores = np.array([r['combined_score'] for r in hybrid_results])
                normalized = self.normalizer.normalize_minmax(scores)
                
                for i, result in enumerate(hybrid_results):
                    all_results.append(SearchResult(
                        text=result['text'],
                        source_file=result['source_file'],
                        file_type=result['file_type'],
                        chunk_index=0,
                        score=float(normalized[i]) * self.hybrid_weight,
                        source_indexer='hybrid',
                        keywords=result.get('keywords', []),
                        capabilities=result.get('capabilities', []),
                        language=result.get('language', '')
                    ))
                
                if verbose:
                    print(f"  Found {len(hybrid_results)} results")
            
            except Exception as e:
                if verbose:
                    print(f"  Error: {e}")
        
        # Deduplicate
        if verbose:
            print(f"\n[3/3] Deduplicating {len(all_results)} results...")
        
        deduplicated = self.deduplicator.deduplicate(all_results)
        
        if verbose:
            print(f"  After deduplication: {len(deduplicated)} results")
        
        # Sort by score and return top_k
        deduplicated.sort(key=lambda r: r.score, reverse=True)
        final = deduplicated[:top_k]
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Final Results: {len(final)}")
            print(f"{'='*70}")
        
        return final
    
    def get_compressed_context(
        self,
        results: List[SearchResult],
        query: str,
        max_tokens: int = 4000,
        focus_on_capabilities: bool = True
    ) -> str:
        """
        Get compressed context for LLM consumption
        
        Args:
            results: Search results
            query: Original query
            max_tokens: Max tokens for context
            focus_on_capabilities: Group by capability vs file
        
        Returns:
            Compressed context string
        """
        self.compressor.max_tokens = max_tokens
        
        return self.compressor.compress(
            results,
            query,
            focus_on_capabilities=focus_on_capabilities
        )
    
    def search_and_compress(
        self,
        query: str,
        top_k: int = 20,
        max_tokens: int = 4000,
        verbose: bool = False
    ) -> Tuple[List[SearchResult], str]:
        """
        One-shot: search and compress
        
        Returns:
            (results, compressed_context)
        """
        results = self.search(query, top_k=top_k, verbose=verbose)
        compressed = self.get_compressed_context(results, query, max_tokens=max_tokens)
        
        return results, compressed


# ============================================================================
# Example Usage
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Unified Search System v2.0"
    )
    parser.add_argument(
        "--universal-index",
        default="./universal_index",
        help="Path to universal index"
    )
    parser.add_argument(
        "--hybrid-index",
        default="./hybrid_index",
        help="Path to hybrid index"
    )
    parser.add_argument(
        "--query",
        required=True,
        help="Search query"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of results"
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="Show compressed context"
    )
    parser.add_argument(
        "--verbose",
        action="store_true"
    )
    
    args = parser.parse_args()
    
    # Initialize
    search = UnifiedCodeSearch(
        universal_index_path=args.universal_index,
        hybrid_index_path=args.hybrid_index
    )
    
    # Search
    results, compressed = search.search_and_compress(
        query=args.query,
        top_k=args.top_k,
        verbose=args.verbose
    )
    
    # Display results
    print(f"\n{'='*70}")
    print(f"Results for: {args.query}")
    print(f"{'='*70}\n")
    
    for i, result in enumerate(results, 1):
        print(f"[{i}] {result.source_file} ({result.source_indexer})")
        print(f"    Score: {result.score:.3f}")
        print(f"    Type: {result.file_type}")
        if result.capabilities:
            print(f"    Capabilities: {', '.join(result.capabilities[:2])}")
        print(f"    {result.text[:150]}...")
        print()
    
    # Display compressed context
    if args.compress:
        print(f"\n{'='*70}")
        print("COMPRESSED CONTEXT (for LLM)")
        print(f"{'='*70}\n")
        print(compressed)


if __name__ == "__main__":
    main()
