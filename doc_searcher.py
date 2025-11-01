"""
Enhanced Wire Processing Searcher
Fixes issue where "how do I implement X" returns different results than "X"
Key improvements:
1. Extracts core terms from queries (removes stopwords)
2. Focuses expansion on key terms only
3. Adds phrase matching bonus
4. Better score normalization
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import numpy as np

# Add this to wire_indexer_lite.py or use as separate module


class ImprovedQueryProcessor:
    """
    Pre-process queries to extract core terms and handle natural language
    Ensures "how do I implement X" returns same results as "X"
    """
    
    def __init__(self):
        # Comprehensive stopwords
        self.stopwords = {
            # Question words
            'how', 'what', 'when', 'where', 'why', 'who', 'which', 'whose',
            
            # Common verbs
            'do', 'does', 'did', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'will', 'would', 'should', 'could', 'can',
            'may', 'might', 'must', 'shall',
            
            # Implementation/action words (keep these for context but lower weight)
            'implement', 'create', 'build', 'make', 'develop', 'setup', 'configure',
            
            # Articles and prepositions
            'a', 'an', 'the', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'from', 'by', 'as', 'into', 'through', 'during', 'before', 'after',
            
            # Pronouns
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her',
            'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their',
            
            # Conjunctions
            'and', 'or', 'but', 'if', 'then', 'than', 'so', 'because',
            
            # Other common words
            'this', 'that', 'these', 'those', 'there', 'here',
        }
        
        # Words that suggest action/question but should be removed for search
        self.action_indicators = {
            'how', 'implement', 'create', 'build', 'setup', 'configure',
            'explain', 'describe', 'show', 'tell', 'need', 'want', 'help'
        }
    
    def extract_core_terms(self, query: str) -> Dict[str, Any]:
        """
        Extract core searchable terms from query
        
        Returns:
            {
                'core_terms': ['credit', 'party', 'determination'],
                'original_query': 'how do I implement credit party determination',
                'cleaned_query': 'credit party determination',
                'is_question': True,
                'action_type': 'implement'
            }
        """
        original = query.strip()
        query_lower = query.lower()
        
        # Detect if it's a question
        is_question = any(query_lower.startswith(q) for q in 
                         ['how', 'what', 'when', 'where', 'why', 'who'])
        
        # Detect action type
        action_type = None
        for action in self.action_indicators:
            if action in query_lower:
                action_type = action
                break
        
        # Tokenize
        # Keep hyphenated terms together
        tokens = re.findall(r'\b[\w-]+\b', query_lower)
        
        # Extract core terms (non-stopwords)
        core_terms = []
        for token in tokens:
            # Skip single letters and numbers
            if len(token) <= 1 or token.isdigit():
                continue
            
            # Skip stopwords
            if token in self.stopwords:
                continue
            
            core_terms.append(token)
        
        # Reconstruct cleaned query
        cleaned_query = ' '.join(core_terms)
        
        return {
            'core_terms': core_terms,
            'original_query': original,
            'cleaned_query': cleaned_query,
            'is_question': is_question,
            'action_type': action_type,
            'stopwords_removed': len(tokens) - len(core_terms)
        }
    
    def extract_phrases(self, query: str) -> List[str]:
        """Extract important multi-word phrases"""
        phrases = []
        
        # Extract quoted phrases
        quoted = re.findall(r'"([^"]+)"', query)
        phrases.extend(quoted)
        
        # Extract capitalized phrases (likely important)
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', query)
        phrases.extend(capitalized)
        
        # Extract technical patterns
        technical_patterns = [
            r'pacs\.\d+',
            r'ISO[\s-]?\d+',
            r'MT\d{3}',
            r'[A-Z]{3,5}(?:\s+[A-Z]{3,5})*'  # Acronym sequences
        ]
        
        for pattern in technical_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            phrases.extend(matches)
        
        return list(set(phrases))


class EnhancedWireProcessingSearcher:
    """
    Enhanced searcher with improved query processing
    Fixes inconsistent results between similar queries
    """
    
    def __init__(
        self, 
        index_path: str = "./wire_index",
        enable_query_expansion: bool = True,
        expansion_level: str = "medium"
    ):
        from wire_indexer_lite import (
            WireProcessingSearcher,
            FastKeywordExtractor,
            CapabilityMapper,
            QueryExpander,
            BusinessCapabilityTaxonomy
        )
        
        # Initialize base searcher
        self.base_searcher = WireProcessingSearcher(
            index_path=index_path,
            enable_query_expansion=False  # We'll handle expansion
        )
        
        # Copy attributes
        self.index_path = self.base_searcher.index_path
        self.stats = self.base_searcher.stats
        self.metadata_store = self.base_searcher.metadata_store
        self.document_store = self.base_searcher.document_store
        self.bm25 = self.base_searcher.bm25
        
        # Initialize processors
        self.query_processor = ImprovedQueryProcessor()
        self.keyword_extractor = FastKeywordExtractor()
        self.capability_mapper = CapabilityMapper()
        
        # Query expansion
        self.enable_query_expansion = enable_query_expansion
        if self.enable_query_expansion:
            self.query_expander = QueryExpander(
                taxonomy=BusinessCapabilityTaxonomy,
                expansion_level=expansion_level,
                max_expansions=5
            )
            print(f"✓ Enhanced searcher with query preprocessing (level: {expansion_level})")
        else:
            self.query_expander = None
    
    def search(
        self,
        query: str,
        top_k: int = 20,
        capability_filter: Optional[List[str]] = None,
        min_capability_score: float = 0.3,
        use_query_expansion: Optional[bool] = None,
        verbose: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Enhanced search with better query preprocessing
        """
        if verbose:
            print(f"\n{'='*80}")
            print(f"Original Query: {query}")
            print(f"{'='*80}")
        
        # Step 1: Extract core terms
        query_analysis = self.query_processor.extract_core_terms(query)
        
        if verbose:
            print(f"\nQuery Analysis:")
            print(f"  Core terms: {query_analysis['core_terms']}")
            print(f"  Cleaned query: {query_analysis['cleaned_query']}")
            print(f"  Is question: {query_analysis['is_question']}")
            print(f"  Stopwords removed: {query_analysis['stopwords_removed']}")
        
        # Use cleaned query for search
        search_query = query_analysis['cleaned_query']
        
        # Extract phrases
        phrases = self.query_processor.extract_phrases(query)
        if phrases and verbose:
            print(f"  Important phrases: {phrases}")
        
        # Step 2: Extract keywords and capabilities from CLEANED query
        query_keywords = self.keyword_extractor.extract(search_query)
        query_capabilities = self.capability_mapper.map_to_capabilities(
            query_keywords,
            search_query
        )
        
        if verbose and query_capabilities:
            print(f"  Detected capabilities: {[cap for cap, _ in query_capabilities[:3]]}")
        
        # Step 3: Query expansion (on cleaned query)
        expanded_info = None
        if use_query_expansion is None:
            use_query_expansion = self.enable_query_expansion
        
        if use_query_expansion and self.query_expander:
            if verbose:
                print(f"\nQuery Expansion:")
            
            expanded_info = self.query_expander.expand_query(
                search_query,  # Use cleaned query, not original
                query_capabilities
            )
            
            if verbose:
                print(f"  Expanded queries ({len(expanded_info['expanded_queries'])}):")
                for eq in expanded_info['expanded_queries'][:3]:
                    print(f"    - {eq}")
                
                if expanded_info['expanded_terms']:
                    print(f"  Expanded terms: {expanded_info['expanded_terms'][:5]}")
        
        # Step 4: Auto-select capabilities
        if not capability_filter and query_capabilities:
            capability_filter = [cap for cap, score in query_capabilities 
                               if score >= min_capability_score][:3]
            if verbose and capability_filter:
                print(f"  Auto-selected capabilities: {capability_filter}")
        
        # Step 5: Search with enhanced scoring
        results = self._enhanced_search(
            search_query=search_query,
            original_query=query,
            core_terms=query_analysis['core_terms'],
            phrases=phrases,
            query_capabilities=query_capabilities,
            capability_filter=capability_filter,
            expanded_info=expanded_info,
            top_k=top_k
        )
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"Found {len(results)} results")
            print(f"{'='*80}")
        
        return results
    
    def _enhanced_search(
        self,
        search_query: str,
        original_query: str,
        core_terms: List[str],
        phrases: List[str],
        query_capabilities: List[Tuple[str, float]],
        capability_filter: Optional[List[str]],
        expanded_info: Optional[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Enhanced search with better scoring
        """
        # Collect all query variants
        query_variants = [search_query]  # Start with cleaned query
        
        if expanded_info:
            query_variants.extend(expanded_info.get('expanded_queries', []))
            if expanded_info.get('expanded_terms'):
                for term in expanded_info['expanded_terms'][:5]:
                    query_variants.append(term)
        
        # Add core terms as individual queries
        for term in core_terms:
            if len(term) > 3:  # Only substantial terms
                query_variants.append(term)
        
        # Aggregate scores across all variants
        aggregated_scores = np.zeros(len(self.document_store))
        
        for variant_idx, variant in enumerate(query_variants):
            query_tokens = variant.lower().split()
            variant_scores = self.bm25.get_scores(query_tokens)
            
            # Weight scoring:
            # - Cleaned query: full weight (1.0)
            # - Core terms: high weight (0.8)
            # - Expanded queries: medium weight (0.5)
            # - Expanded terms: lower weight (0.3)
            
            if variant_idx == 0:
                weight = 1.0  # Cleaned query
            elif variant in core_terms:
                weight = 0.8  # Core terms
            elif expanded_info and variant in expanded_info.get('expanded_queries', []):
                weight = 0.5  # Expanded queries
            else:
                weight = 0.3  # Expanded terms
            
            aggregated_scores += variant_scores * weight
        
        # Add phrase matching bonus
        if phrases:
            phrase_bonuses = self._compute_phrase_bonuses(phrases)
            aggregated_scores += phrase_bonuses * 0.3  # 30% bonus for phrase matches
        
        # Get top candidates
        search_k = min(1000, top_k * 20) if capability_filter else top_k
        top_indices = np.argsort(aggregated_scores)[::-1][:search_k]
        
        # Format results
        formatted_results = []
        for idx in top_indices:
            if aggregated_scores[idx] < 0.01:  # Skip very low scores
                continue
            
            metadata = self.metadata_store[idx]
            
            # Filter by capability
            if capability_filter:
                if metadata['primary_capability'] not in capability_filter:
                    if not any(cap in capability_filter for cap in metadata['capabilities']):
                        continue
            
            # Compute capability overlap
            if query_capabilities:
                query_caps = set([cap for cap, _ in query_capabilities[:5]])
                doc_caps = set(metadata['capabilities'])
                overlap = len(query_caps & doc_caps)
                capability_overlap = overlap / len(query_caps) if query_caps else 0
            else:
                capability_overlap = 1.0
            
            # Normalize score
            max_score = aggregated_scores.max() if aggregated_scores.max() > 0 else 1.0
            normalized_score = aggregated_scores[idx] / max_score
            
            # Check for exact phrase matches (bonus)
            doc_text_lower = self.document_store[idx].lower()
            phrase_match_bonus = 0.0
            for phrase in phrases:
                if phrase.lower() in doc_text_lower:
                    phrase_match_bonus += 0.1
            
            formatted_results.append({
                "text": self.document_store[idx],
                "source_file": metadata['source_file'],
                "chunk_index": metadata['chunk_index'],
                "bm25_score": aggregated_scores[idx],
                "normalized_score": normalized_score,
                "capabilities": metadata['capabilities'],
                "keywords": metadata['keywords'],
                "capability_overlap": capability_overlap,
                "primary_capability": metadata['primary_capability'],
                "phrase_match_bonus": phrase_match_bonus
            })
        
        # Re-rank by combined score
        for result in formatted_results:
            result['combined_score'] = (
                result['normalized_score'] * 0.5 +
                result['capability_overlap'] * 0.3 +
                result['phrase_match_bonus'] * 0.2
            )
        
        formatted_results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return formatted_results[:top_k]
    
    def _compute_phrase_bonuses(self, phrases: List[str]) -> np.ndarray:
        """Compute bonus scores for documents containing exact phrases"""
        bonuses = np.zeros(len(self.document_store))
        
        for phrase in phrases:
            phrase_lower = phrase.lower()
            for idx, doc in enumerate(self.document_store):
                if phrase_lower in doc.lower():
                    # Count occurrences
                    count = doc.lower().count(phrase_lower)
                    bonuses[idx] += np.log1p(count)
        
        return bonuses
    
    def search_by_capability(
        self,
        capability: str,
        top_k: int = 20
    ) -> List[Dict[str, Any]]:
        """Search documents by specific capability"""
        return self.base_searcher.search_by_capability(capability, top_k)


# Convenience function for easy replacement
def create_enhanced_searcher(
    index_path: str = "./wire_index",
    enable_query_expansion: bool = True,
    expansion_level: str = "medium"
):
    """
    Create enhanced searcher with better query preprocessing
    
    Usage:
        searcher = create_enhanced_searcher()
        results = searcher.search("how do I implement credit party determination")
    """
    return EnhancedWireProcessingSearcher(
        index_path=index_path,
        enable_query_expansion=enable_query_expansion,
        expansion_level=expansion_level
    )


if __name__ == "__main__":
    # Test the improvement
    print("Testing Enhanced Query Processing\n")
    print("=" * 80)
    
    processor = ImprovedQueryProcessor()
    
    test_queries = [
        "how do I implement credit party determination",
        "credit party determination",
        "what is OFAC screening?",
        "OFAC screening",
        "explain ISO 20022 pacs.008 validation",
        "ISO 20022 pacs.008 validation"
    ]
    
    for query in test_queries:
        analysis = processor.extract_core_terms(query)
        print(f"\nOriginal: {query}")
        print(f"Cleaned:  {analysis['cleaned_query']}")
        print(f"Core terms: {analysis['core_terms']}")
        print("-" * 80)
