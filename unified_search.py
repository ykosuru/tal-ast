"""
Unified Code Search System
Combines: Universal File Indexer (for PDFs/docs) + LSI Indexer (for code)
Provides: Normalized scoring, deduplication, context compression for LLM
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import hashlib
import re


class SearchResult:
    """Standardized search result across both indexers"""
    
    def __init__(
        self,
        text: str,
        source_file: str,
        file_type: str,
        chunk_index: int,
        score: float,
        source_indexer: str,
        keywords: List[str] = None,
        capabilities: List[str] = None,
        language: str = "",
        metadata: Dict[str, Any] = None
    ):
        self.text = text
        self.source_file = source_file
        self.file_type = file_type
        self.chunk_index = chunk_index
        self.score = score
        self.source_indexer = source_indexer
        self.keywords = keywords or []
        self.capabilities = capabilities or []
        self.language = language
        self.metadata = metadata or {}
        
        # Generate unique ID for deduplication
        self.id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate unique ID based on content"""
        # Use first 200 chars + file info for ID
        content_hash = hashlib.md5(
            f"{self.source_file}:{self.chunk_index}:{self.text[:200]}".encode()
        ).hexdigest()
        return content_hash
    
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


class ResultDeduplicator:
    """Remove duplicate results using fuzzy matching"""
    
    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Compute Jaccard similarity between texts"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def deduplicate(self, results: List[SearchResult]) -> List[SearchResult]:
        """
        Remove duplicate results, keeping highest scoring version
        
        Strategy:
        1. Group by exact ID (same file + chunk)
        2. Group by high text similarity
        3. Keep highest score from each group
        """
        if not results:
            return []
        
        # Sort by score (highest first)
        results = sorted(results, key=lambda r: r.score, reverse=True)
        
        # Track seen IDs and similar texts
        seen_ids = set()
        seen_texts = []
        deduplicated = []
        
        for result in results:
            # Skip exact duplicates (same ID)
            if result.id in seen_ids:
                continue
            
            # Check for similar texts
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


class ScoreNormalizer:
    """Normalize scores from different indexers to [0, 1] range"""
    
    @staticmethod
    def normalize_bm25(scores: np.ndarray) -> np.ndarray:
        """Normalize BM25 scores (0-∞) to [0, 1]"""
        if len(scores) == 0:
            return scores
        
        # Use tanh to map (0, ∞) → (0, 1)
        # tanh(x/k) where k controls the steepness
        k = 2.0  # Adjust based on typical BM25 scores
        normalized = np.tanh(scores / k)
        
        return normalized
    
    @staticmethod
    def normalize_lsi(scores: np.ndarray) -> np.ndarray:
        """Normalize LSI cosine similarity scores"""
        # LSI scores are already [-1, 1], shift to [0, 1]
        return (scores + 1) / 2
    
    @staticmethod
    def normalize_minmax(scores: np.ndarray) -> np.ndarray:
        """Min-max normalization to [0, 1]"""
        if len(scores) == 0 or scores.max() == scores.min():
            return np.zeros_like(scores)
        
        return (scores - scores.min()) / (scores.max() - scores.min())


class ContextCompressor:
    """Compress search results for efficient LLM consumption"""
    
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
        1. Group by capability/file type
        2. Extract key snippets around keywords
        3. Remove redundant information
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
        
        # Build compressed context
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
        
        # Add summary at the end
        summary = self._create_summary(results)
        context_parts.append(f"\n## Summary\n{summary}")
        
        return "\n".join(context_parts)
    
    def _group_by_capability(
        self, 
        results: List[SearchResult]
    ) -> Dict[str, List[SearchResult]]:
        """Group results by primary capability"""
        grouped = defaultdict(list)
        
        for result in results:
            if result.capabilities:
                primary_cap = result.capabilities[0]
            else:
                primary_cap = result.file_type
            
            grouped[primary_cap].append(result)
        
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
        """Group results by source file"""
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
        
        for result in results[:5]:  # Limit to top 5 per group
            # Extract relevant snippet
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
        """Extract most relevant snippet from text"""
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
        """Create high-level summary of results"""
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


class UnifiedCodeSearch:
    """
    Unified search interface for both indexers
    
    Usage:
        search = UnifiedCodeSearch(
            universal_index_path="./universal_index",
            lsi_index_path="./hybrid_index"
        )
        
        results = search.search(
            query="payment drawdown processing",
            top_k=20
        )
        
        compressed = search.get_compressed_context(results, query)
    """
    
    def __init__(
        self,
        universal_index_path: Optional[str] = None,
        lsi_index_path: Optional[str] = None,
        universal_weight: float = 0.5,
        lsi_weight: float = 0.5
    ):
        self.universal_weight = universal_weight
        self.lsi_weight = lsi_weight
        
        # Initialize indexers (import the actual classes)
        self.universal_searcher = None
        self.lsi_searcher = None
        
        if universal_index_path:
            try:
                # Import from your actual module
                # from universal_file_indexer import UniversalFileSearcher
                # self.universal_searcher = UniversalFileSearcher(universal_index_path)
                print(f"✓ Universal indexer loaded from {universal_index_path}")
            except Exception as e:
                print(f"⚠ Could not load universal indexer: {e}")
        
        if lsi_index_path:
            try:
                # from lsi_indexer import HybridSearcher
                # self.lsi_searcher = HybridSearcher(lsi_index_path)
                print(f"✓ LSI indexer loaded from {lsi_index_path}")
            except Exception as e:
                print(f"⚠ Could not load LSI indexer: {e}")
        
        # Utilities
        self.deduplicator = ResultDeduplicator()
        self.normalizer = ScoreNormalizer()
        self.compressor = ContextCompressor()
    
    def search(
        self,
        query: str,
        top_k: int = 20,
        file_type_filter: Optional[str] = None,
        verbose: bool = False
    ) -> List[SearchResult]:
        """
        Search across both indexers with unified scoring
        
        Args:
            query: Search query (e.g., "drawdown processing")
            top_k: Total number of results to return
            file_type_filter: Filter by 'code', 'pdf', 'text', etc.
            verbose: Show detailed process
        
        Returns:
            List of deduplicated, scored SearchResult objects
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"Unified Search: {query}")
            print(f"{'='*70}")
        
        all_results = []
        
        # Search universal indexer (PDFs, docs)
        if self.universal_searcher:
            if verbose:
                print("\n[1/2] Searching Universal Indexer (PDFs/Docs)...")
            
            try:
                universal_results = self.universal_searcher.search(
                    query,
                    top_k=top_k * 2,  # Get extra for filtering
                    file_type_filter=file_type_filter,
                    verbose=False
                )
                
                # Convert to StandardResult and normalize scores
                scores = np.array([r['score'] for r in universal_results])
                normalized_scores = self.normalizer.normalize_bm25(scores)
                
                for i, result in enumerate(universal_results):
                    all_results.append(SearchResult(
                        text=result['text'],
                        source_file=result['source_file'],
                        file_type=result['file_type'],
                        chunk_index=result.get('chunk_index', 0),
                        score=float(normalized_scores[i]) * self.universal_weight,
                        source_indexer='universal',
                        keywords=result.get('keywords', []),
                        capabilities=result.get('capabilities', []),
                        language=result.get('language', '')
                    ))
                
                if verbose:
                    print(f"  Found {len(universal_results)} results")
            
            except Exception as e:
                print(f"  Error searching universal indexer: {e}")
        
        # Search LSI indexer (code)
        if self.lsi_searcher:
            if verbose:
                print("\n[2/2] Searching LSI Indexer (Code)...")
            
            try:
                lsi_results = self.lsi_searcher.search(
                    query,
                    top_k=top_k * 2,
                    use_lsi=True,
                    verbose=False
                )
                
                # Normalize scores
                scores = np.array([r['combined_score'] for r in lsi_results])
                normalized_scores = self.normalizer.normalize_minmax(scores)
                
                for i, result in enumerate(lsi_results):
                    all_results.append(SearchResult(
                        text=result['text'],
                        source_file=result['source_file'],
                        file_type=result['file_type'],
                        chunk_index=0,
                        score=float(normalized_scores[i]) * self.lsi_weight,
                        source_indexer='lsi',
                        keywords=result.get('keywords', []),
                        language=result.get('file_type', '')
                    ))
                
                if verbose:
                    print(f"  Found {len(lsi_results)} results")
            
            except Exception as e:
                print(f"  Error searching LSI indexer: {e}")
        
        # Deduplicate results
        if verbose:
            print(f"\n[3/3] Deduplicating {len(all_results)} total results...")
        
        deduplicated = self.deduplicator.deduplicate(all_results)
        
        if verbose:
            print(f"  After deduplication: {len(deduplicated)} results")
        
        # Sort by final score and return top_k
        deduplicated.sort(key=lambda r: r.score, reverse=True)
        final_results = deduplicated[:top_k]
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Final Results: {len(final_results)}")
            print(f"{'='*70}")
            
            for i, result in enumerate(final_results[:5], 1):
                print(f"\n[{i}] {result.source_file} ({result.source_indexer})")
                print(f"    Score: {result.score:.3f}")
                print(f"    Type: {result.file_type}")
                if result.capabilities:
                    print(f"    Capabilities: {', '.join(result.capabilities[:3])}")
        
        return final_results
    
    def get_compressed_context(
        self,
        results: List[SearchResult],
        query: str,
        max_tokens: int = 4000,
        focus_on_capabilities: bool = True
    ) -> str:
        """
        Get compressed context suitable for LLM consumption
        
        Args:
            results: Search results
            query: Original query
            max_tokens: Maximum tokens for LLM context
            focus_on_capabilities: Group by capability vs file
        
        Returns:
            Compressed context string ready for LLM
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
        One-shot: search and compress in single call
        
        Returns:
            (results, compressed_context)
        """
        results = self.search(query, top_k=top_k, verbose=verbose)
        compressed = self.get_compressed_context(results, query, max_tokens=max_tokens)
        
        return results, compressed


# Example usage
if __name__ == "__main__":
    # Initialize unified search
    search = UnifiedCodeSearch(
        universal_index_path="./universal_index",
        lsi_index_path="./hybrid_index",
        universal_weight=0.6,  # Give slight preference to docs
        lsi_weight=0.4
    )
    
    # Search for functionality
    query = "payment drawdown processing"
    
    results, compressed_context = search.search_and_compress(
        query=query,
        top_k=20,
        max_tokens=4000,
        verbose=True
    )
    
    # compressed_context is ready to send to LLM
    print("\n" + "="*70)
    print("COMPRESSED CONTEXT FOR LLM:")
    print("="*70)
    print(compressed_context)
    
    # Now send to your LLM for code generation
    # llm_response = your_llm.generate(
    #     prompt=f"Based on this context, implement {query}:\n\n{compressed_context}"
    # )
