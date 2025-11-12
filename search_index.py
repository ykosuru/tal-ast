"""
Quick Context Extractor v2.5 - REQUIREMENTS-OPTIMIZED
Works with existing indexes
Handles both text/code AND images!

NEW IN v2.5 - REQUIREMENTS OPTIMIZATION:
- Synonym expansion for payment/banking/requirements terminology
- Multiple search strategies (exact, fuzzy, broad)
- Re-ranking by domain relevance (prioritize requirements docs)
- Configurable search modes: 'precise', 'balanced', 'comprehensive'
- Automatic query expansion for requirements keywords
- Better handling of technical abbreviations (ACH, OFAC, AML, etc.)
- Domain-specific scoring boost for requirements language
- Cascade search: if few results, automatically retry with relaxed settings

Previous features:
- Reduced PDF rendering resolution for smaller images
- JPEG compression with quality=85
- Two-stage deduplication
- Query term validation (now optional)
- Result diversification
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict
import pickle
import re
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


# DOMAIN-SPECIFIC SYNONYMS FOR BANKING/PAYMENTS/REQUIREMENTS
DEFAULT_PAYMENT_SYNONYMS = {
    # Payment terms
    'payment': ['transaction', 'transfer', 'remittance', 'settlement', 'disbursement', 'fund'],
    'wire': ['transfer', 'swift', 'fedwire', 'ach', 'eft', 'electronic-transfer'],
    'validate': ['verify', 'check', 'confirm', 'authenticate', 'review', 'validate'],
    'validation': ['verification', 'checking', 'confirmation', 'authentication', 'review'],
    'process': ['execute', 'handle', 'perform', 'run', 'complete', 'processing'],
    'processing': ['execution', 'handling', 'operation', 'performing'],
    'drawdown': ['withdrawal', 'disbursement', 'advance', 'draw'],
    'account': ['acct', 'customer', 'client', 'party'],
    'balance': ['funds', 'amount', 'total', 'value'],
    'credit': ['deposit', 'increase', 'addition', 'posting'],
    'debit': ['withdrawal', 'decrease', 'charge', 'posting'],
    
    # Requirements terms
    'requirement': ['spec', 'specification', 'criteria', 'constraint', 'rule', 'condition'],
    'must': ['shall', 'required', 'mandatory', 'needed'],
    'should': ['recommended', 'preferred', 'suggested', 'advised'],
    'function': ['capability', 'feature', 'operation', 'functionality', 'service'],
    'system': ['application', 'platform', 'software', 'solution', 'service'],
    'user': ['operator', 'customer', 'actor', 'personnel', 'party'],
    'data': ['information', 'record', 'field', 'value', 'element'],
    'report': ['output', 'document', 'statement', 'listing', 'file'],
    'error': ['exception', 'failure', 'issue', 'problem', 'fault', 'reject'],
    'message': ['notification', 'alert', 'warning', 'response', 'advice'],
    
    # Banking/Financial terms
    'ofac': ['sanctions', 'screening', 'compliance', 'watchlist'],
    'aml': ['anti-money-laundering', 'laundering', 'kyc', 'compliance'],
    'kyc': ['know-your-customer', 'customer-identification', 'cip'],
    'compliance': ['regulatory', 'regulation', 'rule', 'policy', 'governance'],
    'audit': ['review', 'examination', 'inspection', 'verification', 'check'],
    'approval': ['authorization', 'clearance', 'permission', 'consent'],
    'reject': ['decline', 'deny', 'refuse', 'block', 'rejection'],
    'pending': ['waiting', 'queued', 'hold', 'suspended', 'in-process'],
    'complete': ['finished', 'done', 'finalized', 'closed', 'completed'],
    
    # Technical terms
    'code': ['program', 'source', 'script', 'implementation', 'software'],
    'procedure': ['function', 'routine', 'subroutine', 'method', 'process'],
    'call': ['invoke', 'execute', 'run', 'trigger', 'invocation'],
    'database': ['db', 'datastore', 'repository', 'storage', 'data'],
    'table': ['relation', 'entity', 'dataset', 'file'],
    'query': ['search', 'lookup', 'select', 'retrieve', 'find'],
    'update': ['modify', 'change', 'edit', 'revise', 'alter'],
    'delete': ['remove', 'purge', 'drop', 'erase', 'eliminate'],
    'create': ['add', 'insert', 'generate', 'build', 'establish'],
    
    # Status/Flow terms
    'status': ['state', 'condition', 'stage', 'phase', 'standing'],
    'flow': ['process', 'workflow', 'sequence', 'procedure', 'path'],
    'step': ['stage', 'phase', 'action', 'task', 'activity'],
    'input': ['parameter', 'argument', 'value', 'data', 'request'],
    'output': ['result', 'return', 'response', 'outcome', 'reply'],
}


class QuickContextExtractor:
    """
    Extract context without requiring line numbers in index
    Works with existing indexes!
    NOW WITH REQUIREMENTS OPTIMIZATION!
    """
    
    def __init__(
        self,
        universal_index: str = "./universal_index",
        hybrid_index: str = "./hybrid_index",
        custom_synonyms: Optional[Dict[str, List[str]]] = None,
        use_synonyms: bool = True
    ):
        self.universal_index = Path(universal_index)
        self.hybrid_index = Path(hybrid_index)
        self.file_cache = {}  # Cache file contents
        self.pdf_cache = {}   # Cache PDF objects
        
        # Synonym configuration
        self.use_synonyms = use_synonyms
        self.synonyms = DEFAULT_PAYMENT_SYNONYMS.copy()
        
        # Merge custom synonyms
        if custom_synonyms:
            for key, values in custom_synonyms.items():
                if key in self.synonyms:
                    self.synonyms[key].extend(values)
                else:
                    self.synonyms[key] = values
        
        # Build reverse lookup (synonym -> main term)
        self.synonym_reverse = {}
        for main_term, syn_list in self.synonyms.items():
            for syn in syn_list:
                if syn not in self.synonym_reverse:
                    self.synonym_reverse[syn] = []
                self.synonym_reverse[syn].append(main_term)
        
        if PDF_AVAILABLE:
            print("✓ Quick context extractor ready (with PDF/image support)")
        else:
            print("✓ Quick context extractor ready (text only - install pdfplumber for images)")
        
        if self.use_synonyms:
            print(f"✓ Synonym expansion enabled ({len(self.synonyms)} term groups)")
    
    def extract(
        self,
        query: str,
        max_matches: int = 5,
        lines_before: int = 200,
        lines_after: int = 200,
        index_type: str = "both",  # "both", "universal", or "hybrid"
        embed_images: bool = True,  # Embed images from PDFs
        diversify_results: bool = True,  # Mix different file types
        file_type_filter: Optional[str] = None,  # Filter: 'code', 'pdf', 'text'
        validate_query_terms: bool = False,  # NEW DEFAULT: False for requirements
        min_query_terms: int = 1,
        dedup_similarity: float = 0.90,  # NEW DEFAULT: Less aggressive (was 0.85)
        search_mode: str = 'comprehensive',  # NEW: 'precise', 'balanced', 'comprehensive'
        use_synonym_expansion: Optional[bool] = None,  # NEW: Synonym expansion
        boost_requirements: bool = True  # NEW: Boost requirements documents
    ) -> str:
        """
        Extract context for query
        
        Args:
            query: Search query
            max_matches: Number of matches
            lines_before: Lines before match (for text/code)
            lines_after: Lines after match (for text/code)
            index_type: Which index to search
            embed_images: Embed full images from PDFs
            diversify_results: Try to get mix of file types
            file_type_filter: Only return specific file type
            validate_query_terms: Only keep results that contain query terms (FALSE by default now)
            min_query_terms: Minimum number of query terms that must appear in result
            dedup_similarity: Similarity threshold for deduplication (0.90 = 90% similar)
            search_mode: 'precise' (exact), 'balanced' (default), 'comprehensive' (max recall)
            use_synonym_expansion: Enable synonym expansion (default: True for comprehensive)
            boost_requirements: Boost documents with requirements language
        
        Returns:
            Formatted context string
        """
        # Auto-configure based on search mode
        if search_mode == 'precise':
            validate_query_terms = True
            expand_synonyms = False
            retrieval_multiplier = 3
        elif search_mode == 'balanced':
            validate_query_terms = True
            expand_synonyms = True
            retrieval_multiplier = 5
        else:  # comprehensive
            validate_query_terms = False  # Get everything
            expand_synonyms = True
            retrieval_multiplier = 8  # Get many candidates
        
        # Override if explicitly set
        if use_synonym_expansion is not None:
            expand_synonyms = use_synonym_expansion
        
        print(f"\n{'='*70}")
        print(f"SEARCH MODE: {search_mode.upper()}")
        print(f"{'='*70}")
        print(f"Query: {query}")
        print(f"Settings: {lines_before} lines before, {lines_after} lines after")
        print(f"Image embedding: {'Enabled' if embed_images else 'Disabled'}")
        print(f"Result diversification: {'Enabled' if diversify_results else 'Disabled'}")
        print(f"Query validation: {'Enabled' if validate_query_terms else 'Disabled'}")
        print(f"Synonym expansion: {'Enabled' if expand_synonyms else 'Disabled'}")
        print(f"Requirements boost: {'Enabled' if boost_requirements else 'Disabled'}")
        print(f"Deduplication threshold: {dedup_similarity:.2f}")
        
        # Store settings
        self.dedup_similarity = dedup_similarity
        
        # Expand query with synonyms if enabled
        if expand_synonyms and self.use_synonyms:
            expanded_query, original_terms, all_terms = self._expand_query_with_synonyms(query)
            search_query = expanded_query
            validation_terms = all_terms
        else:
            search_query = query
            validation_terms = None
        
        # Search indexes with expanded query
        results = self._search_indexes(
            search_query, 
            max_matches * retrieval_multiplier,
            index_type
        )
        
        if not results:
            return "No results found."
        
        print(f"Retrieved {len(results)} initial results")
        
        # Boost requirements documents if enabled
        if boost_requirements:
            results = self._boost_requirements_docs(results)
        
        # Validate query terms (only if enabled)
        if validate_query_terms:
            results = self._validate_query_terms(
                results, 
                query, 
                min_query_terms,
                expanded_terms=validation_terms
            )
            print(f"After validation: {len(results)} results contain query terms")
        
        # CASCADE FALLBACK: If too few results, retry with more lenient settings
        if len(results) < max_matches and search_mode != 'comprehensive':
            print(f"\n⚠ Only {len(results)} results found. Retrying with comprehensive mode...")
            return self.extract(
                query=query,
                max_matches=max_matches,
                lines_before=lines_before,
                lines_after=lines_after,
                index_type=index_type,
                embed_images=embed_images,
                diversify_results=diversify_results,
                file_type_filter=file_type_filter,
                validate_query_terms=False,  # Disable validation
                dedup_similarity=0.95,  # More lenient
                search_mode='comprehensive',
                use_synonym_expansion=True,
                boost_requirements=boost_requirements
            )
        
        if not results:
            return f"No results found for query: {query}"
        
        # Apply file type filter
        if file_type_filter:
            results = [r for r in results if r['file_type'] == file_type_filter]
            print(f"Filtered to {len(results)} {file_type_filter} results")
        
        # Diversify results if enabled
        if diversify_results and not file_type_filter:
            results = self._diversify_results(results, max_matches)
        else:
            results = results[:max_matches]
        
        print(f"\nFound {len(results)} final matches")
        
        # Extract context from each result
        contexts = []
        for i, result in enumerate(results, 1):
            print(f"[{i}/{len(results)}] {result['source_file']} ({result['file_type']}) - Score: {result['score']:.3f}")
            
            context = self._extract_context_for_result(
                result,
                lines_before,
                lines_after,
                embed_images
            )
            
            if context:
                contexts.append(context)
        
        # Format for LLM
        return self._format_contexts(contexts, query)
    
    def _expand_query_with_synonyms(
        self,
        query: str,
        max_synonyms_per_term: int = 3
    ) -> Tuple[str, List[str], List[str]]:
        """
        Expand query with synonyms
        
        Returns:
            (expanded_query, original_terms, all_expanded_terms)
        """
        stopwords = {
            'how', 'what', 'when', 'where', 'why', 'who', 'which',
            'do', 'does', 'did', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'will', 'would', 'should', 'could', 'can',
            'a', 'an', 'the', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'from', 'by', 'as', 'into', 'through', 'during', 'before', 'after',
            'and', 'or', 'but', 'if', 'then', 'than', 'so',
            'this', 'that', 'these', 'those'
        }
        
        query_lower = query.lower()
        query_words = re.findall(r'\b[\w-]+\b', query_lower)
        
        # Get meaningful terms (length >= 2 for technical terms like 'id', 'db')
        original_terms = [
            word for word in query_words 
            if word not in stopwords and len(word) >= 2
        ]
        
        if not original_terms:
            original_terms = [w for w in query_words if len(w) >= 2]
        
        # Expand with synonyms
        expanded_terms = set(original_terms)
        expansion_map = {}
        
        for term in original_terms:
            # Check if term has synonyms
            if term in self.synonyms:
                synonyms = self.synonyms[term][:max_synonyms_per_term]
                expanded_terms.update(synonyms)
                expansion_map[term] = synonyms
            
            # Check if term is itself a synonym
            elif term in self.synonym_reverse:
                main_terms = self.synonym_reverse[term]
                for main_term in main_terms:
                    expanded_terms.add(main_term)
                    # Also add other synonyms of the main term
                    other_syns = [s for s in self.synonyms.get(main_term, [])[:max_synonyms_per_term] if s != term]
                    expanded_terms.update(other_syns)
                    if other_syns:
                        expansion_map[term] = [main_term] + other_syns
        
        # Build expanded query
        expanded_query = ' '.join(expanded_terms)
        
        # Log expansion
        if expansion_map:
            print("\n📖 Synonym Expansion:")
            for term, syns in list(expansion_map.items())[:5]:  # Show first 5
                print(f"   '{term}' → {', '.join(syns[:3])}")
            if len(expansion_map) > 5:
                print(f"   ... and {len(expansion_map) - 5} more expansions")
            print(f"   Expanded from {len(original_terms)} to {len(expanded_terms)} terms\n")
        
        return expanded_query, original_terms, list(expanded_terms)
    
    def _boost_requirements_docs(self, results: List[Dict]) -> List[Dict]:
        """
        Boost documents that contain requirements language
        """
        requirements_keywords = {
            'shall', 'must', 'should', 'will', 'requirement', 'requirements',
            'specification', 'specifications', 'mandatory', 'optional',
            'functional', 'non-functional', 'constraint', 'criteria'
        }
        
        for result in results:
            text_lower = result.get('text', '').lower()
            
            # Count requirements keywords
            req_count = sum(1 for kw in requirements_keywords if kw in text_lower)
            
            # Boost score if requirements keywords found
            if req_count > 0:
                boost_factor = 1.0 + (req_count * 0.1)  # 10% boost per keyword
                result['original_score'] = result['score']
                result['score'] = result['score'] * boost_factor
                result['requirements_boost'] = boost_factor
        
        # Re-sort by boosted score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        boosted_count = sum(1 for r in results if r.get('requirements_boost', 1.0) > 1.0)
        if boosted_count > 0:
            print(f"✓ Boosted {boosted_count} documents with requirements language")
        
        return results
    
    def _validate_query_terms(
        self,
        results: List[Dict],
        query: str,
        min_terms: int = 1,
        expanded_terms: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Validate that results contain query terms (or their synonyms)
        """
        stopwords = {
            'how', 'what', 'when', 'where', 'why', 'who', 'which',
            'do', 'does', 'did', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'will', 'would', 'should', 'could', 'can',
            'a', 'an', 'the', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'from', 'by', 'as', 'into', 'through', 'during', 'before', 'after',
            'and', 'or', 'but', 'if', 'then', 'than', 'so',
            'this', 'that', 'these', 'those'
        }
        
        query_lower = query.lower()
        query_words = re.findall(r'\b[\w-]+\b', query_lower)
        
        # Use expanded terms if provided, otherwise extract from query
        if expanded_terms:
            query_terms = expanded_terms
        else:
            query_terms = [
                word for word in query_words 
                if word not in stopwords and len(word) > 2
            ]
        
        if not query_terms:
            print("⚠ No meaningful query terms - returning all results")
            return results
        
        print(f"Validating against {len(query_terms)} terms (showing first 10): {', '.join(query_terms[:10])}")
        
        # Get stemmed versions
        stemmed_terms = self._try_stem_terms(query_terms)
        all_terms = set(query_terms + stemmed_terms)
        
        validated_results = []
        
        for result in results:
            text = result.get('text', '').lower()
            
            # Count matches with partial matching for longer terms
            matches = 0
            matched_terms = []
            
            for term in all_terms:
                if len(term) >= 4:  # Longer terms: allow substring match
                    if any(term in word for word in text.split()):
                        matches += 1
                        matched_terms.append(term)
                else:  # Short terms: exact word match
                    if re.search(r'\b' + re.escape(term) + r'\b', text):
                        matches += 1
                        matched_terms.append(term)
            
            # Accept if at least one term matches (very lenient for requirements)
            if matches > 0:
                result['query_term_matches'] = matches
                result['matched_terms'] = matched_terms[:5]  # Store first 5
                validated_results.append(result)
        
        validated_results.sort(
            key=lambda x: (x.get('query_term_matches', 0), x.get('score', 0)), 
            reverse=True
        )
        
        return validated_results
    
    def _try_stem_terms(self, terms: List[str]) -> List[str]:
        """
        Try to get stemmed versions of terms using the indexer's stemmer
        """
        stemmed = []
        
        try:
            # Try to import and use the stemmer from universal indexer
            from doc_indexer import TextStemmer
            stemmer = TextStemmer(use_stemming=True)
            
            for term in terms:
                stemmed_term = stemmer.stem(term)
                if stemmed_term != term:
                    stemmed.append(stemmed_term)
        except:
            # If stemmer not available, just return empty
            pass
        
        return stemmed
    
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
        index_type: str
    ) -> List[Dict[str, Any]]:
        """Search one or both indexes"""
        from doc_indexer import UniversalFileSearcher
        try:
            from code_indexer import HybridSearcher
        except ImportError:
            try:
                from code_indexer import HybridSearcher
            except ImportError:
                HybridSearcher = None
        
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
                        'source_path': r.get('source_path', ''),
                        'file_type': r['file_type'],
                        'chunk_index': r.get('chunk_index', 0),
                        'score': r['score'],
                        'index': 'universal'
                    })
            except Exception as e:
                print(f"⚠ Universal index error: {e}")
        
        # Search hybrid index
        if index_type in ["both", "hybrid"] and self.hybrid_index.exists() and HybridSearcher:
            try:
                searcher = HybridSearcher(str(self.hybrid_index))
                hyb_results = searcher.search(query, top_k=max_matches, verbose=False)
                
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
                print(f"⚠ Hybrid index error: {e}")
        
        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # IMPROVED DEDUPLICATION
        # Strategy 1: Remove exact duplicates by file + chunk
        seen_chunks = set()
        unique_results = []
        
        for r in results:
            chunk_key = f"{r['source_file']}:{r['chunk_index']}"
            if chunk_key not in seen_chunks:
                seen_chunks.add(chunk_key)
                unique_results.append(r)
        
        # Strategy 2: Remove highly similar results from same file
        final_results = []
        file_texts = defaultdict(list)  # Track texts per file
        
        dedup_threshold = getattr(self, 'dedup_similarity', 0.90)
        
        for r in unique_results:
            source_file = r['source_file']
            text = r['text']
            
            # Check if this text is too similar to existing texts from same file
            is_duplicate = False
            for existing_text in file_texts[source_file]:
                similarity = self._text_similarity(text, existing_text)
                if similarity > dedup_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                file_texts[source_file].append(text)
                final_results.append(r)
        
        print(f"Deduplication: {len(results)} → {len(unique_results)} → {len(final_results)} results")
        
        return final_results
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity between two texts"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    # ... [REST OF THE METHODS FROM v2.4 - KEEPING THEM IDENTICAL] ...
    # (All the _extract_context_for_result, _extract_text_context, _extract_pdf_context, 
    #  _find_page_in_pdf, _is_garbage_text, _find_source_file, _is_binary_file,
    #  _clean_line, _read_file_cached, _find_match_in_lines, _similarity,
    #  _format_contexts, _format_text_context, _format_pdf_context methods)
    
    def _extract_context_for_result(self, result: Dict[str, Any], lines_before: int, lines_after: int, embed_images: bool) -> Optional[Dict[str, Any]]:
        source_path = self._find_source_file(result)
        if not source_path or not source_path.exists():
            print(f"  ⚠ Source file not found")
            return None
        is_pdf = source_path.suffix.lower() == '.pdf'
        if is_pdf and embed_images and PDF_AVAILABLE:
            return self._extract_pdf_context(source_path, result, lines_before, lines_after)
        else:
            return self._extract_text_context(source_path, result, lines_before, lines_after)
    
    def _extract_text_context(self, source_path: Path, result: Dict[str, Any], lines_before: int, lines_after: int) -> Optional[Dict[str, Any]]:
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
            'file_type': result['file_type'],
            'query_term_matches': result.get('query_term_matches', 0)
        }
    
    def _extract_pdf_context(self, source_path: Path, result: Dict[str, Any], lines_before: int, lines_after: int) -> Optional[Dict[str, Any]]:
        try:
            if str(source_path) in self.pdf_cache:
                pdf = self.pdf_cache[str(source_path)]
            else:
                pdf = pdfplumber.open(source_path)
                self.pdf_cache[str(source_path)] = pdf
            page_num = self._find_page_in_pdf(pdf, result['text'])
            if page_num is None:
                print(f"  ⚠ Could not find match in PDF")
                return self._extract_text_context(source_path, result, lines_before, lines_after)
            page = pdf.pages[page_num]
            page_text = page.extract_text() or ""
            text_lines = page_text.split('\n')
            images = []
            try:
                print(f"  Rendering full page {page_num + 1} as single image...")
                img = page.to_image(resolution=100)
                img_pil = img.original
                max_width = 800
                if img_pil.width > max_width:
                    ratio = max_width / img_pil.width
                    new_height = int(img_pil.height * ratio)
                    img_pil = img_pil.resize((max_width, new_height), Image.Resampling.LANCZOS)
                buffered = io.BytesIO()
                try:
                    if img_pil.mode in ('RGBA', 'LA'):
                        background = Image.new('RGB', img_pil.size, (255, 255, 255))
                        background.paste(img_pil, mask=img_pil.split()[-1] if img_pil.mode == 'RGBA' else None)
                        img_pil = background
                    img_pil.save(buffered, format="JPEG", quality=85, optimize=True)
                    img_format = "JPEG"
                except:
                    img_pil.save(buffered, format="PNG", optimize=True)
                    img_format = "PNG"
                img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                size_mb = len(img_base64) * 3 / 4 / (1024 * 1024)
                if size_mb > 3:
                    print(f"  ⚠ Warning: Image size is {size_mb:.1f}MB, may cause issues with LLM")
                images.append({
                    'index': 0,
                    'base64': img_base64,
                    'format': img_format,
                    'width': img_pil.width,
                    'height': img_pil.height,
                    'full_page': True,
                    'size_mb': round(size_mb, 2)
                })
                print(f"  ✓ Captured full page as single {img_format} image ({img_pil.width}x{img_pil.height}, {size_mb:.1f}MB)")
            except Exception as e:
                print(f"  ⚠ Could not render page: {e}")
            context_pages_before = []
            context_pages_after = []
            for i in range(max(0, page_num - 2), page_num):
                try:
                    ctx_page = pdf.pages[i]
                    ctx_text = ctx_page.extract_text() or ""
                    context_pages_before.append({'page_num': i, 'text': ctx_text[:500]})
                except:
                    pass
            for i in range(page_num + 1, min(len(pdf.pages), page_num + 3)):
                try:
                    ctx_page = pdf.pages[i]
                    ctx_text = ctx_page.extract_text() or ""
                    context_pages_after.append({'page_num': i, 'text': ctx_text[:500]})
                except:
                    pass
            print(f"  ✓ Page {page_num + 1} captured as single image")
            return {
                'type': 'pdf',
                'source_file': result['source_file'],
                'source_path': source_path,
                'page_num': page_num,
                'page_text': text_lines,
                'images': images,
                'context_pages_before': context_pages_before,
                'context_pages_after': context_pages_after,
                'score': result['score'],
                'file_type': result['file_type'],
                'query_term_matches': result.get('query_term_matches', 0)
            }
        except Exception as e:
            print(f"  ⚠ PDF extraction error: {e}")
            return self._extract_text_context(source_path, result, lines_before, lines_after)
    
    def _find_page_in_pdf(self, pdf, match_text: str) -> Optional[int]:
        match_words = set(match_text.lower().split()[:10])
        best_page = 0
        best_match_count = 0
        for page_num, page in enumerate(pdf.pages):
            page_text = (page.extract_text() or "").lower()
            page_words = set(page_text.split())
            match_count = len(match_words & page_words)
            if match_count > best_match_count:
                best_match_count = match_count
                best_page = page_num
        return best_page if best_match_count > 0 else None
    
    def _is_garbage_text(self, text: str) -> bool:
        if not text or len(text) < 10:
            return False
        non_printable = sum(1 for c in text if not c.isprintable() and c not in '\t\n\r ')
        if non_printable / len(text) > 0.2:
            return True
        garbage_indicators = [b'\x00'.decode('latin-1'), '\ufffd']
        for indicator in garbage_indicators:
            if indicator in text:
                return True
        non_ascii = sum(1 for c in text if ord(c) > 127)
        if non_ascii / len(text) > 0.5:
            return True
        return False
    
    def _find_source_file(self, result: Dict[str, Any]) -> Optional[Path]:
        if result.get('source_path') and result['source_path']:
            source_path = Path(result['source_path'])
            if source_path.exists():
                return source_path
        filename = result['source_file']
        search_paths = [
            Path(filename),
            Path("./your_docs") / filename,
            Path("./your_code") / filename,
            Path("./docs") / filename,
            Path("./code") / filename,
            Path("./src") / filename,
        ]
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
        elif result['index'] == 'hybrid':
            try:
                with open(self.hybrid_index / "metadata.pkl", 'rb') as f:
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
    
    def _is_binary_file(self, file_path: Path) -> bool:
        binary_extensions = {
            '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
            '.zip', '.tar', '.gz', '.exe', '.dll', '.so', '.dylib',
            '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.ico',
            '.mp3', '.mp4', '.avi', '.mov', '.wav',
            '.bin', '.dat', '.db'
        }
        if file_path.suffix.lower() in binary_extensions:
            return True
        try:
            with open(file_path, 'rb') as f:
                chunk = f.read(8192)
            if b'\x00' in chunk:
                return True
            text_chars = bytearray({7, 8, 9, 10, 12, 13, 27} | set(range(0x20, 0x100)) - {0x7f})
            non_text = chunk.translate(None, text_chars)
            if len(non_text) / max(1, len(chunk)) > 0.3:
                return True
        except:
            pass
        return False
    
    def _clean_line(self, line: str) -> str:
        cleaned = ''.join(char for char in line if char.isprintable() or char in '\t\n\r')
        return cleaned
    
    def _read_file_cached(self, file_path: Path) -> List[str]:
        file_path_str = str(file_path)
        if file_path_str in self.file_cache:
            return self.file_cache[file_path_str]
        if self._is_binary_file(file_path):
            print(f"  ⚠ Skipping binary file: {file_path.name}")
            return []
        try:
            for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                        lines = [self._clean_line(line.rstrip('\n\r')) for line in f.readlines()]
                    cleaned_lines = []
                    for line in lines:
                        non_ascii = sum(1 for c in line if ord(c) > 127)
                        if len(line) > 0 and non_ascii / len(line) < 0.5:
                            cleaned_lines.append(line)
                        elif len(line) == 0:
                            cleaned_lines.append(line)
                    self.file_cache[file_path_str] = cleaned_lines
                    return cleaned_lines
                except UnicodeDecodeError:
                    continue
        except Exception as e:
            print(f"  Error reading file: {e}")
        return []
    
    def _find_match_in_lines(self, lines: List[str], match_text: str) -> tuple:
        match_lines = [l.strip() for l in match_text.split('\n') if l.strip()]
        if not match_lines:
            return (0, min(10, len(lines)))
        search_lines = match_lines[:min(3, len(match_lines))]
        for i in range(len(lines)):
            matches = 0
            for j, search_line in enumerate(search_lines):
                if i + j >= len(lines):
                    break
                file_line = lines[i + j].strip()
                if (search_line in file_line or file_line in search_line or self._similarity(search_line, file_line) > 0.7):
                    matches += 1
            if matches >= len(search_lines) * 0.7:
                start_line = i
                end_line = min(i + len(match_lines), len(lines))
                return (start_line, end_line)
        estimated_lines = len(match_lines)
        return (0, min(estimated_lines, len(lines)))
    
    def _similarity(self, s1: str, s2: str) -> float:
        words1 = set(s1.lower().split())
        words2 = set(s2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union if union > 0 else 0.0
    
    def _format_contexts(self, contexts: List[Dict[str, Any]], query: str) -> str:
        if not contexts:
            return "No contexts found."
        parts = []
        parts.append(f"# Code Context for: {query}\n")
        parts.append(f"Found {len(contexts)} relevant sections\n")
        text_count = sum(1 for c in contexts if c.get('type') == 'text')
        pdf_count = sum(1 for c in contexts if c.get('type') == 'pdf')
        parts.append(f"- Text/Code: {text_count}")
        parts.append(f"- PDFs with images: {pdf_count}\n")
        for i, ctx in enumerate(contexts, 1):
            parts.append(f"\n{'='*70}")
            parts.append(f"## Match {i}: {ctx['source_file']}")
            parts.append(f"Score: {ctx['score']:.3f}")
            if ctx.get('query_term_matches', 0) > 0:
                parts.append(f"Query term matches: {ctx['query_term_matches']}")
            parts.append(f"{'='*70}\n")
            if ctx.get('type') == 'pdf':
                parts.append(self._format_pdf_context(ctx))
            else:
                parts.append(self._format_text_context(ctx))
        parts.append(f"\n{'='*70}")
        parts.append("## Summary")
        parts.append(f"Total matches: {len(contexts)}")
        total_lines = sum(len(c.get('before_lines', [])) + len(c.get('match_lines', [])) + len(c.get('after_lines', [])) for c in contexts if c.get('type') == 'text')
        total_images = sum(len(c.get('images', [])) for c in contexts if c.get('type') == 'pdf')
        parts.append(f"Total text lines: {total_lines}")
        parts.append(f"Total images: {total_images}")
        return '\n'.join(parts)
    
    def _format_text_context(self, ctx: Dict[str, Any]) -> str:
        parts = []
        parts.append(f"Type: Text/Code")
        parts.append(f"Location: Lines {ctx['match_start']}-{ctx['match_end']}\n")
        ext = Path(ctx['source_file']).suffix.lower()
        lang_map = {'.py': 'python', '.java': 'java', '.c': 'c', '.cpp': 'cpp', '.js': 'javascript', '.tal': 'tal', '.cbl': 'cobol', '.sql': 'sql', '.sh': 'bash', '.txt': 'text'}
        lang = lang_map.get(ext, 'text')
        parts.append(f"```{lang}")
        before = ctx['before_lines'][-50:] if len(ctx['before_lines']) > 50 else ctx['before_lines']
        after = ctx['after_lines'][:50] if len(ctx['after_lines']) > 50 else ctx['after_lines']
        if before:
            start_num = ctx['match_start'] - len(before)
            for j, line in enumerate(before):
                parts.append(f"{start_num + j:4d} | {line}")
            parts.append("")
        parts.append(">>> RELEVANT MATCH <<<")
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
    
    def _format_pdf_context(self, ctx: Dict[str, Any]) -> str:
        parts = []
        parts.append(f"Type: PDF")
        parts.append(f"Page: {ctx['page_num'] + 1}\n")
        if ctx['context_pages_before']:
            parts.append(f"### Context from Previous Pages\n")
            for page_ctx in ctx['context_pages_before']:
                parts.append(f"**Page {page_ctx['page_num'] + 1}:**")
                lines = page_ctx['text'].split('\n')[:5]
                for line in lines:
                    if line.strip():
                        parts.append(f"  {line}")
            parts.append("")
        if ctx['images']:
            parts.append(f"### Full Page {ctx['page_num'] + 1} Image\n")
            for img in ctx['images']:
                if img.get('full_page'):
                    parts.append(f"**Complete page rendered as image** ({img['width']:.0f}x{img['height']:.0f} pixels):\n")
                else:
                    parts.append(f"**Page Image** ({img['width']:.0f}x{img['height']:.0f} pixels):\n")
                parts.append(f"![Page Image](data:image/png;base64,{img['base64']})\n")
        if ctx['page_text'] and len(ctx['page_text']) < 100:
            parts.append(f"### Page {ctx['page_num'] + 1} Text Content\n")
            parts.append("```")
            for line in ctx['page_text'][:30]:
                if line.strip():
                    parts.append(line)
            parts.append("```\n")
        if ctx['context_pages_after']:
            parts.append(f"### Context from Following Pages\n")
            for page_ctx in ctx['context_pages_after']:
                parts.append(f"**Page {page_ctx['page_num'] + 1}:**")
                lines = page_ctx['text'].split('\n')[:5]
                for line in lines:
                    if line.strip():
                        parts.append(f"  {line}")
            parts.append("")
        return '\n'.join(parts)


# Quick usage function
def quick_extract(
    query: str,
    lines_before: int = 200,
    lines_after: int = 200,
    max_matches: int = 10,  # NEW DEFAULT: 10 for requirements
    embed_images: bool = True,
    diversify_results: bool = True,
    file_type_filter: Optional[str] = None,
    validate_query_terms: bool = False,  # NEW DEFAULT: False
    min_query_terms: int = 1,
    dedup_similarity: float = 0.90,  # NEW DEFAULT: 0.90
    search_mode: str = 'comprehensive',  # NEW: 'precise', 'balanced', 'comprehensive'
    custom_synonyms: Optional[Dict[str, List[str]]] = None
) -> str:
    """
    One-liner: extract context for query (OPTIMIZED FOR REQUIREMENTS)
    
    Args:
        query: Search query
        lines_before: Lines before match
        lines_after: Lines after match
        max_matches: Number of matches (default: 10 for requirements)
        embed_images: Embed full images from PDFs
        diversify_results: Get mix of file types
        file_type_filter: Only get specific type ('code', 'pdf', 'text')
        validate_query_terms: Only keep results containing query terms (default: False)
        min_query_terms: Minimum query terms that must appear
        dedup_similarity: Similarity threshold (0.90 = 90% similar)
        search_mode: 'precise' (exact), 'balanced', 'comprehensive' (max recall)
        custom_synonyms: Custom synonym dictionary
    
    Usage:
        # COMPREHENSIVE mode (best for requirements) - DEFAULT
        context = quick_extract("payment validation requirements")
        
        # Balanced mode (some validation)
        context = quick_extract("wire transfer rules", search_mode='balanced')
        
        # Precise mode (strict matching)
        context = quick_extract("exact function name", search_mode='precise')
        
        # Custom synonyms for your domain
        custom_syns = {
            'tal': ['transaction-language', 'tandem'],
            'cbl': ['cobol-language']
        }
        context = quick_extract("TAL procedure", custom_synonyms=custom_syns)
        
        # PDFs only
        context = quick_extract("requirements", file_type_filter='pdf', max_matches=15)
    """
    extractor = QuickContextExtractor(custom_synonyms=custom_synonyms)
    return extractor.extract(
        query,
        max_matches,
        lines_before,
        lines_after,
        embed_images=embed_images,
        diversify_results=diversify_results,
        file_type_filter=file_type_filter,
        validate_query_terms=validate_query_terms,
        min_query_terms=min_query_terms,
        dedup_similarity=dedup_similarity,
        search_mode=search_mode
    )


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        query = ' '.join(sys.argv[1:])
    else:
        query = "payment validation requirements"
    
    print("="*70)
    print("QUICK CONTEXT EXTRACTION v2.5")
    print("REQUIREMENTS-OPTIMIZED with Synonym Expansion!")
    print("="*70)
    
    # Auto-detect if looking for requirements/docs
    requirements_keywords = ['requirement', 'specification', 'document', 'policy', 'rule', 'must', 'shall']
    is_requirements_query = any(kw in query.lower() for kw in requirements_keywords)
    
    if is_requirements_query:
        print(f"\n💡 Requirements query detected - using COMPREHENSIVE mode")
        search_mode = 'comprehensive'
    else:
        search_mode = 'balanced'
    
    context = quick_extract(
        query,
        lines_before=200,
        lines_after=200,
        max_matches=10,
        search_mode=search_mode
    )
    
    print("\n" + "="*70)
    print("RESULT")
    print("="*70)
    print(context[:2000])
    if len(context) > 2000:
        print(f"\n... ({len(context) - 2000:,} more characters)")
    
    # Save to file
    with open("extracted_context.txt", 'w') as f:
        f.write(context)
    
    print(f"\n✓ Saved to: extracted_context.txt")
    print(f"  Length: {len(context):,} chars")
