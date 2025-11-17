"""
Quick Context Extractor v3.0 - NOW WITH UNIVERSAL SEARCHER INTEGRATION
Uses doc_indexer.py's UniversalFileSearcher for best search results!

MAJOR UPGRADE:
- Uses UniversalFileSearcher for multi-term phrase search
- 12+ ranking features from doc_indexer
- Better relevance with BM25 + re-ranking
- Still extracts full context (200 lines before/after)
- Still embeds PDF images
- Still creates HTML output
- Still validates query terms
- Still deduplicates results

Features:
- Text/Code: Extract N lines before/after (configurable)
- Images: Extract and embed full image as base64
- PDFs: Extract images from matched pages with compression
- Smart result mixing: Get both code and PDFs in results
- Best-in-class search ranking from UniversalFileSearcher
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import pickle
import re
import base64
import io

# Import the superior search engine!
try:
    from doc_indexer import UniversalFileSearcher
    UNIVERSAL_SEARCHER_AVAILABLE = True
except ImportError:
    UNIVERSAL_SEARCHER_AVAILABLE = False
    print("⚠ doc_indexer.py not found. Please ensure it's in the same directory.")

# Optional: PDF and image support
try:
    import pdfplumber
    from PIL import Image
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("⚠ PDF/Image support not available. Install: pip install pdfplumber Pillow")


class QuickContextExtractor:
    """
    Extract context using UniversalFileSearcher for best results!
    Then extract full context with images.
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
        
        # Initialize the superior searcher!
        if UNIVERSAL_SEARCHER_AVAILABLE and self.universal_index.exists():
            try:
                self.searcher = UniversalFileSearcher(str(self.universal_index))
                print("✓ Quick context extractor ready with UniversalFileSearcher (best ranking!)")
            except Exception as e:
                print(f"⚠ Could not load UniversalFileSearcher: {e}")
                self.searcher = None
        else:
            self.searcher = None
            print("⚠ UniversalFileSearcher not available - using fallback search")
        
        if PDF_AVAILABLE:
            print("✓ PDF/image support enabled")
        else:
            print("⚠ PDF support disabled - install pdfplumber for images")
    
    def extract(
        self,
        query: str,
        max_matches: int = 5,
        lines_before: int = 200,
        lines_after: int = 200,
        embed_images: bool = True,
        diversify_results: bool = True,
        file_type_filter: Optional[str] = None,
        validate_query_terms: bool = True,
        min_query_terms: int = 1,
        dedup_similarity: float = 0.85,
        use_query_expansion: bool = True,
        enable_multi_term: bool = True,
        show_explanations: bool = False
    ) -> str:
        """
        Extract context for query using UniversalFileSearcher
        
        Args:
            query: Search query
            max_matches: Number of matches
            lines_before: Lines before match (for text/code)
            lines_after: Lines after match (for text/code)
            embed_images: Embed full images from PDFs
            diversify_results: Try to get mix of file types
            file_type_filter: Only return specific file type ('code', 'pdf', 'text')
            validate_query_terms: Only keep results that contain query terms
            min_query_terms: Minimum query terms that must appear in result
            dedup_similarity: Similarity threshold for deduplication (0.0-1.0)
            use_query_expansion: Use domain synonym expansion (from doc_indexer)
            enable_multi_term: Enable multi-term phrase detection (from doc_indexer)
            show_explanations: Show ranking explanations from doc_indexer
        
        Returns:
            Formatted context string
        """
        
        print(f"\n{'='*70}")
        print(f"Query: {query}")
        print(f"{'='*70}")
        print(f"Settings:")
        print(f"  - Context: {lines_before} lines before, {lines_after} lines after")
        print(f"  - Image embedding: {'Enabled' if embed_images else 'Disabled'}")
        print(f"  - Result diversification: {'Enabled' if diversify_results else 'Disabled'}")
        print(f"  - Query term validation: {'Enabled' if validate_query_terms else 'Disabled'}")
        print(f"  - Multi-term search: {'Enabled' if enable_multi_term else 'Disabled'}")
        print(f"  - Deduplication threshold: {dedup_similarity:.2f}")
        
        # Search using UniversalFileSearcher (BETTER RANKING!)
        results = self._search_with_universal(
            query,
            max_matches * 3,  # Get more candidates for filtering
            file_type_filter,
            use_query_expansion,
            enable_multi_term,
            show_explanations
        )
        
        if not results:
            return "No results found."
        
        print(f"\nInitial results: {len(results)}")
        
        # Validate query terms (if enabled)
        if validate_query_terms:
            results = self._validate_query_terms(results, query, min_query_terms)
            print(f"After validation: {len(results)} results contain query terms")
        
        if not results:
            return f"No results found containing the query terms: {query}"
        
        # Deduplicate
        results = self._deduplicate_results(results, dedup_similarity)
        print(f"After deduplication: {len(results)} unique results")
        
        # Diversify results if enabled
        if diversify_results and not file_type_filter:
            results = self._diversify_results(results, max_matches)
        else:
            results = results[:max_matches]
        
        print(f"\nExtracting context from {len(results)} matches...")
        
        # Extract context from each result
        contexts = []
        for i, result in enumerate(results, 1):
            print(f"[{i}/{len(results)}] {result['source_file']} ({result['file_type']}) - Score: {result['score']:.2f}")
            
            if show_explanations and 'explanation' in result:
                print(f"  Ranking signals:")
                for exp in result['explanation'][:3]:
                    print(f"    • {exp}")
            
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
    
    def _search_with_universal(
        self,
        query: str,
        max_results: int,
        file_type_filter: Optional[str],
        use_query_expansion: bool,
        enable_multi_term: bool,
        show_explanations: bool
    ) -> List[Dict[str, Any]]:
        """
        Search using UniversalFileSearcher (BEST RANKING!)
        Falls back to legacy search if not available
        """
        
        if self.searcher:
            print("\n🔍 Searching with UniversalFileSearcher (multi-term + advanced ranking)...")
            
            try:
                # Use the superior search!
                search_results = self.searcher.search(
                    query=query,
                    top_k=max_results,
                    file_type_filter=file_type_filter,
                    use_query_expansion=use_query_expansion,
                    enable_multi_term=enable_multi_term,
                    explain=show_explanations,
                    verbose=False
                )
                
                # Convert to our format
                results = []
                for r in search_results:
                    results.append({
                        'text': r['text'],
                        'source_file': r['source_file'],
                        'source_path': '',  # Will be resolved later
                        'file_type': r['file_type'],
                        'chunk_index': r.get('chunk_index', 0),
                        'score': r['score'],
                        'bm25_score': r.get('bm25_score', 0),
                        'explanation': r.get('explanation', []),
                        'capabilities': r.get('capabilities', []),
                        'keywords': r.get('keywords', []),
                        'index': 'universal'
                    })
                
                print(f"✓ Found {len(results)} results using UniversalFileSearcher")
                return results
                
            except Exception as e:
                print(f"⚠ UniversalFileSearcher error: {e}")
                print("  Falling back to legacy search...")
                return self._search_legacy(query, max_results, file_type_filter)
        else:
            print("\n⚠ Using legacy search (UniversalFileSearcher not available)")
            return self._search_legacy(query, max_results, file_type_filter)
    
    def _search_legacy(
        self,
        query: str,
        max_results: int,
        file_type_filter: Optional[str]
    ) -> List[Dict[str, Any]]:
        """
        Fallback: Direct BM25 search (if UniversalFileSearcher unavailable)
        """
        results = []
        
        # Try loading index directly
        if self.universal_index.exists():
            try:
                with open(self.universal_index / "metadata.pkl", 'rb') as f:
                    metadata_store = pickle.load(f)
                with open(self.universal_index / "documents.pkl", 'rb') as f:
                    document_store = pickle.load(f)
                with open(self.universal_index / "bm25.pkl", 'rb') as f:
                    bm25 = pickle.load(f)
                
                # Simple BM25 search
                query_tokens = query.lower().split()
                scores = bm25.get_scores(query_tokens)
                
                # Get top results
                import numpy as np
                top_indices = np.argsort(scores)[::-1][:max_results]
                
                for idx in top_indices:
                    if scores[idx] > 0:
                        results.append({
                            'text': document_store[idx][:500],
                            'source_file': metadata_store[idx]['source_file'],
                            'source_path': '',
                            'file_type': metadata_store[idx]['file_type'],
                            'chunk_index': metadata_store[idx].get('chunk_index', 0),
                            'score': float(scores[idx]),
                            'bm25_score': float(scores[idx]),
                            'explanation': [],
                            'index': 'universal'
                        })
                
                # Apply file type filter
                if file_type_filter:
                    results = [r for r in results if r['file_type'] == file_type_filter]
                
                print(f"✓ Found {len(results)} results using legacy search")
                
            except Exception as e:
                print(f"⚠ Legacy search error: {e}")
        
        return results
    
    def _validate_query_terms(
        self,
        results: List[Dict],
        query: str,
        min_terms: int = 1
    ) -> List[Dict]:
        """
        Validate that results contain query terms
        """
        stopwords = {
            'how', 'what', 'when', 'where', 'why', 'who', 'which',
            'do', 'does', 'did', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'will', 'would', 'should', 'could', 'can',
            'a', 'an', 'the', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'from', 'by', 'as', 'into', 'through', 'during', 'before', 'after',
            'and', 'or', 'but', 'if', 'then', 'than', 'so', 'because',
            'this', 'that', 'these', 'those'
        }
        
        query_lower = query.lower()
        query_words = re.findall(r'\b[\w-]+\b', query_lower)
        query_terms = [w for w in query_words if w not in stopwords and len(w) > 2]
        
        if not query_terms:
            query_terms = query_words
        
        print(f"  Validating against terms: {', '.join(query_terms)}")
        
        validated_results = []
        
        for result in results:
            text = result.get('text', '').lower()
            matches = sum(1 for term in query_terms if term in text)
            
            if matches >= min_terms:
                result['query_term_matches'] = matches
                validated_results.append(result)
        
        # Sort by query term matches (higher is better)
        validated_results.sort(
            key=lambda x: (x.get('query_term_matches', 0), x.get('score', 0)),
            reverse=True
        )
        
        return validated_results
    
    def _deduplicate_results(
        self,
        results: List[Dict],
        threshold: float = 0.85
    ) -> List[Dict]:
        """
        Remove duplicate results (exact + similarity-based)
        """
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
        file_texts = defaultdict(list)
        
        for r in unique_results:
            source_file = r['source_file']
            text = r['text']
            
            # Check if similar to existing texts from same file
            is_duplicate = False
            for existing_text in file_texts[source_file]:
                if self._text_similarity(text, existing_text) > threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                file_texts[source_file].append(text)
                final_results.append(r)
        
        print(f"  Dedup: {len(results)} → {len(unique_results)} → {len(final_results)}")
        
        return final_results
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _diversify_results(self, results: List[Dict], max_results: int) -> List[Dict]:
        """
        Diversify results to include different file types
        """
        by_type = defaultdict(list)
        for r in results:
            file_type = r.get('file_type', 'unknown')
            by_type[file_type].append(r)
        
        print(f"  Diversifying across {len(by_type)} file types:")
        for ft, items in by_type.items():
            print(f"    {ft}: {len(items)} results")
        
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
        
        return diversified
    
    def _extract_context_for_result(
        self,
        result: Dict[str, Any],
        lines_before: int,
        lines_after: int,
        embed_images: bool
    ) -> Optional[Dict[str, Any]]:
        """Extract context from result"""
        source_path = self._find_source_file(result)
        
        if not source_path or not source_path.exists():
            print(f"  ⚠ Could not find source file")
            return None
        
        is_pdf = source_path.suffix.lower() == '.pdf'
        
        if is_pdf and PDF_AVAILABLE and embed_images:
            return self._extract_pdf_context(
                source_path,
                result,
                lines_before,
                lines_after
            )
        else:
            return self._extract_text_context(
                source_path,
                result,
                lines_before,
                lines_after
            )
    
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
            print(f"  ⚠ Skipping: appears to be binary data")
            return None
        
        lines = self._read_file_cached(source_path)
        
        if not lines:
            print(f"  ⚠ Could not read file")
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
            'query_term_matches': result.get('query_term_matches', 0),
            'explanation': result.get('explanation', [])
        }
    
    def _extract_pdf_context(
        self,
        source_path: Path,
        result: Dict[str, Any],
        lines_before: int,
        lines_after: int
    ) -> Optional[Dict[str, Any]]:
        """Extract context from PDF with full page image"""
        
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
                print(f"  Rendering full page {page_num + 1}...")
                
                img = page.to_image(resolution=100)
                img_pil = img.original
                
                # Resize if too large
                max_width = 800
                if img_pil.width > max_width:
                    ratio = max_width / img_pil.width
                    new_height = int(img_pil.height * ratio)
                    img_pil = img_pil.resize((max_width, new_height), Image.Resampling.LANCZOS)
                
                # Convert to base64
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
                    print(f"  ⚠ Warning: Image is {size_mb:.1f}MB")
                
                images.append({
                    'index': 0,
                    'base64': img_base64,
                    'format': img_format,
                    'width': img_pil.width,
                    'height': img_pil.height,
                    'full_page': True,
                    'size_mb': round(size_mb, 2)
                })
                
                print(f"  ✓ Page {page_num + 1} → {img_format} ({img_pil.width}x{img_pil.height}, {size_mb:.1f}MB)")
                
            except Exception as e:
                print(f"  ⚠ Could not render page: {e}")
            
            # Context pages
            context_pages_before = []
            context_pages_after = []
            
            for i in range(max(0, page_num - 2), page_num):
                try:
                    ctx_page = pdf.pages[i]
                    ctx_text = ctx_page.extract_text() or ""
                    context_pages_before.append({
                        'page_num': i,
                        'text': ctx_text[:500]
                    })
                except:
                    pass
            
            for i in range(page_num + 1, min(len(pdf.pages), page_num + 3)):
                try:
                    ctx_page = pdf.pages[i]
                    ctx_text = ctx_page.extract_text() or ""
                    context_pages_after.append({
                        'page_num': i,
                        'text': ctx_text[:500]
                    })
                except:
                    pass
            
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
                'query_term_matches': result.get('query_term_matches', 0),
                'explanation': result.get('explanation', [])
            }
        
        except Exception as e:
            print(f"  ⚠ PDF extraction error: {e}")
            return self._extract_text_context(source_path, result, lines_before, lines_after)
    
    def _find_page_in_pdf(self, pdf, match_text: str) -> Optional[int]:
        """Find which page contains the match"""
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
        """Check if text is binary garbage"""
        if not text or len(text) < 10:
            return False
        
        non_printable = sum(1 for c in text if not c.isprintable() and c not in '\t\n\r ')
        if non_printable / len(text) > 0.2:
            return True
        
        return False
    
    def _find_source_file(self, result: Dict[str, Any]) -> Optional[Path]:
        """Find source file path"""
        filename = result['source_file']
        
        # Common locations
        search_paths = [
            Path(filename),
            Path("./your_docs") / filename,
            Path("./your_code") / filename,
            Path("./docs") / filename,
            Path("./code") / filename,
            Path("./src") / filename,
        ]
        
        # Try metadata
        if self.universal_index.exists():
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
    
    def _is_binary_file(self, file_path: Path) -> bool:
        """Check if file is binary"""
        binary_extensions = {
            '.pdf', '.doc', '.docx', '.xls', '.xlsx',
            '.png', '.jpg', '.jpeg', '.gif',
            '.zip', '.tar', '.gz', '.exe'
        }
        
        if file_path.suffix.lower() in binary_extensions:
            return True
        
        try:
            with open(file_path, 'rb') as f:
                chunk = f.read(8192)
                if b'\x00' in chunk:
                    return True
        except:
            pass
        
        return False
    
    def _clean_line(self, line: str) -> str:
        """Remove non-printable characters"""
        return ''.join(c for c in line if c.isprintable() or c in '\t\n\r')
    
    def _read_file_cached(self, file_path: Path) -> List[str]:
        """Read file with caching"""
        file_path_str = str(file_path)
        
        if file_path_str in self.file_cache:
            return self.file_cache[file_path_str]
        
        if self._is_binary_file(file_path):
            return []
        
        try:
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                        lines = [self._clean_line(line.rstrip('\n\r')) for line in f.readlines()]
                    
                    self.file_cache[file_path_str] = lines
                    return lines
                except UnicodeDecodeError:
                    continue
        except:
            pass
        
        return []
    
    def _find_match_in_lines(self, lines: List[str], match_text: str) -> Tuple[int, int]:
        """Find where match appears in lines"""
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
                
                if (search_line in file_line or
                    file_line in search_line or
                    self._text_similarity(search_line, file_line) > 0.7):
                    matches += 1
            
            if matches >= len(search_lines) * 0.7:
                return (i, min(i + len(match_lines), len(lines)))
        
        return (0, min(len(match_lines), len(lines)))
    
    def _format_contexts(self, contexts: List[Dict[str, Any]], query: str) -> str:
        """Format contexts for LLM"""
        if not contexts:
            return "No contexts found."
        
        parts = []
        
        # Header
        parts.append(f"# Code Context for: {query}\n")
        parts.append(f"Found {len(contexts)} relevant sections\n")
        
        text_count = sum(1 for c in contexts if c.get('type') == 'text')
        pdf_count = sum(1 for c in contexts if c.get('type') == 'pdf')
        parts.append(f"— Text/Code: {text_count}")
        parts.append(f"— PDFs with images: {pdf_count}\n")
        
        # Each context
        for i, ctx in enumerate(contexts, 1):
            parts.append(f"\n{'='*70}")
            parts.append(f"## Match {i}: {ctx['source_file']}")
            parts.append(f"Score: {ctx['score']:.3f}")
            if ctx.get('query_term_matches', 0) > 0:
                parts.append(f"Query term matches: {ctx['query_term_matches']}")
            
            # Show ranking explanation if available
            if ctx.get('explanation'):
                parts.append(f"Ranking signals:")
                for exp in ctx['explanation'][:3]:
                    parts.append(f"  • {exp}")
            
            parts.append(f"{'='*70}\n")
            
            if ctx.get('type') == 'pdf':
                parts.append(self._format_pdf_context(ctx))
            else:
                parts.append(self._format_text_context(ctx))
        
        # Summary
        parts.append(f"\n{'='*70}")
        parts.append("## Summary")
        parts.append(f"Total matches: {len(contexts)}")
        
        total_lines = sum(
            len(c.get('before_lines', [])) + len(c.get('match_lines', [])) + len(c.get('after_lines', []))
            for c in contexts if c.get('type') == 'text'
        )
        total_images = sum(len(c.get('images', [])) for c in contexts if c.get('type') == 'pdf')
        
        parts.append(f"Total text lines: {total_lines}")
        parts.append(f"Total images: {total_images}\n")
        
        return '\n'.join(parts)
    
    def _format_text_context(self, ctx: Dict[str, Any]) -> str:
        """Format text/code context"""
        parts = []
        
        parts.append(f"Type: Text/Code")
        parts.append(f"Location: Lines {ctx['match_start']}-{ctx['match_end']}\n")
        
        ext = Path(ctx['source_file']).suffix.lower()
        lang_map = {
            '.py': 'python', '.java': 'java', '.c': 'c', '.cpp': 'cpp',
            '.js': 'javascript', '.ts': 'typescript', '.tal': 'tal',
            '.cbl': 'cobol', '.sql': 'sql', '.sh': 'bash', '.TXT': 'tal'
        }
        
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
        
        parts.append("```")
        
        return '\n'.join(parts)
    
    def _format_pdf_context(self, ctx: Dict[str, Any]) -> str:
        """Format PDF context with full page image"""
        parts = []
        
        parts.append(f"Type: PDF")
        parts.append(f"Page: {ctx['page_num'] + 1}\n")
        
        if ctx['context_pages_before']:
            parts.append(f"### Context from Previous Pages\n")
            for page_ctx in ctx['context_pages_before']:
                parts.append(f"**Page {page_ctx['page_num'] + 1}:**")
                for line in page_ctx['text'].split('\n')[:5]:
                    if line.strip():
                        parts.append(f"  {line}")
                parts.append("")
        
        if ctx['images']:
            parts.append(f"### Full Page {ctx['page_num'] + 1} Image\n")
            for img in ctx['images']:
                parts.append(f"**Full page** ({img['width']}x{img['height']} px, {img.get('size_mb', 0):.1f}MB):\n")
                parts.append(f"![Page Image](data:image/png;base64,{img['base64']})\n")
        
        if ctx['page_text'] and len(ctx['page_text']) < 100:
            parts.append(f"### Page {ctx['page_num'] + 1} Text\n")
            parts.append("```")
            for line in ctx['page_text'][:30]:
                if line.strip():
                    parts.append(line)
            parts.append("```\n")
        
        if ctx['context_pages_after']:
            parts.append(f"### Context from Following Pages\n")
            for page_ctx in ctx['context_pages_after']:
                parts.append(f"**Page {page_ctx['page_num'] + 1}:**")
                for line in page_ctx['text'].split('\n')[:5]:
                    if line.strip():
                        parts.append(f"  {line}")
                parts.append("")
        
        return '\n'.join(parts)


def create_html_content(query: str, context: str) -> str:
    """Create HTML visualization with embedded images"""
    if "data:image/png;base64," not in context:
        return None
    
    print(f"\n📄 Creating HTML view with embedded images...")
    
    html_content = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Context: """ + query + """</title>
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; max-width: 1400px; margin: 20px auto; padding: 20px; background: #f5f5f5; }
        .container { background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        h2 { color: #34495e; border-bottom: 2px solid #95a5a6; padding-bottom: 8px; margin-top: 30px; }
        h3 { color: #7f8c8d; margin-top: 20px; }
        pre { background: #2c3e50; color: #ecf0f1; padding: 15px; border-radius: 5px; overflow-x: auto; font-size: 13px; line-height: 1.5; }
        .image-container { margin: 20px 0; padding: 20px; background: #f8f9fa; border-radius: 8px; border: 2px solid #dee2e6; text-align: center; }
        .image-container img { max-width: 100%; width: auto; height: auto; max-height: 1200px; border: 1px solid #ccc; border-radius: 4px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15); display: block; margin: 15px auto; }
        .signal { color: #16a085; font-size: 0.9em; margin: 5px 0; }
    </style>
</head>
<body>
    <div class="container">
"""
    
    lines = context.split('\n')
    in_code_block = False
    
    for line in lines:
        if line.startswith('```'):
            if in_code_block:
                html_content += "</pre>\n"
                in_code_block = False
            else:
                html_content += "<pre>\n"
                in_code_block = True
            continue
        
        if '![Page Image](data:image/png;base64,' in line:
            match = re.search(r'!\[Page Image\]\(data:image/png;base64,([^)]+)\)', line)
            if match:
                base64_data = match.group(1)
                html_content += f'<div class="image-container">\n'
                html_content += f'<img src="data:image/png;base64,{base64_data}" alt="Page Image" />\n'
                html_content += '</div>\n'
            continue
        
        if line.startswith('# '):
            html_content += f"<h1>{line[2:]}</h1>\n"
        elif line.startswith('## '):
            html_content += f"<h2>{line[3:]}</h2>\n"
        elif line.startswith('### '):
            html_content += f"<h3>{line[4:]}</h3>\n"
        elif line.strip().startswith('•'):
            html_content += f'<div class="signal">{line}</div>\n'
        elif line.strip() == '='*70:
            html_content += '<hr>\n'
        elif line.strip() and not in_code_block:
            html_content += f"{line}<br>\n"
        elif in_code_block:
            html_content += f"{line}\n"
    
    if in_code_block:
        html_content += "</pre>\n"
    
    html_content += """
    </div>
</body>
</html>
"""
    
    return html_content


def quick_extract(
    query: str,
    lines_before: int = 200,
    lines_after: int = 200,
    max_matches: int = 5,
    embed_images: bool = True,
    diversify_results: bool = True,
    file_type_filter: Optional[str] = None,
    validate_query_terms: bool = True,
    min_query_terms: int = 1,
    dedup_similarity: float = 0.85,
    use_query_expansion: bool = True,
    enable_multi_term: bool = True,
    show_explanations: bool = False
) -> str:
    """
    One-liner: extract context using UniversalFileSearcher
    
    NOW WITH SUPERIOR RANKING FROM doc_indexer.py!
    
    Args:
        query: Search query
        lines_before: Lines before match
        lines_after: Lines after match
        max_matches: Number of matches
        embed_images: Embed PDF images
        diversify_results: Mix file types
        file_type_filter: Only specific type ('code', 'pdf', 'text')
        validate_query_terms: Ensure results contain query
        min_query_terms: Min terms that must match
        dedup_similarity: Dedup threshold (0.85 = 85%)
        use_query_expansion: Use domain synonyms (from doc_indexer)
        enable_multi_term: Enable phrase detection (from doc_indexer)
        show_explanations: Show ranking signals (from doc_indexer)
    
    Usage:
        # Best results with all features
        context = quick_extract("cutoff times")
        
        # With ranking explanations
        context = quick_extract("cutoff times", show_explanations=True)
        
        # Code only
        context = quick_extract("payment validation", file_type_filter='code')
        
        # PDFs only
        context = quick_extract("flow diagram", file_type_filter='pdf')
    """
    extractor = QuickContextExtractor()
    return extractor.extract(
        query,
        max_matches=max_matches,
        lines_before=lines_before,
        lines_after=lines_after,
        embed_images=embed_images,
        diversify_results=diversify_results,
        file_type_filter=file_type_filter,
        validate_query_terms=validate_query_terms,
        min_query_terms=min_query_terms,
        dedup_similarity=dedup_similarity,
        use_query_expansion=use_query_expansion,
        enable_multi_term=enable_multi_term,
        show_explanations=show_explanations
    )


if __name__ == "__main__":
    import argparse
    import sys
    
    if len(sys.argv) == 1:
        print("Usage: python3 search_index.py --search <search term>")
        sys.exit(0)
    
    parser = argparse.ArgumentParser(
        description='Context Extractor v3.0 - Powered by UniversalFileSearcher',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s -s "cutoff times"
    %(prog)s -s "payment validation" --explain
    %(prog)s -s "flow diagram" --type pdf
        """
    )
    
    parser.add_argument('--search', '-s', required=True, help='Search term')
    parser.add_argument('--max', type=int, default=5, help='Max matches')
    parser.add_argument('--type', choices=['code', 'pdf', 'text'], help='Filter file type')
    parser.add_argument('--explain', action='store_true', help='Show ranking explanations')
    parser.add_argument('--no-images', action='store_true', help='Disable PDF images')
    
    args = parser.parse_args()
    
    print("="*70)
    print("CONTEXT EXTRACTOR v3.0")
    print("Powered by UniversalFileSearcher!")
    print("="*70)
    
    context = quick_extract(
        args.search,
        max_matches=args.max,
        file_type_filter=args.type,
        embed_images=not args.no_images,
        show_explanations=args.explain,
        validate_query_terms=True,
        enable_multi_term=True,
        use_query_expansion=True
    )
    
    print("\n" + "="*70)
    print("RESULT")
    print("="*70)
    print(context[:2000])
    if len(context) > 2000:
        print(f"\n... ({len(context) - 2000:,} more characters)")
    
    # Save outputs
    with open("extracted_context.txt", 'w') as f:
        f.write(context)
    
    html_content = create_html_content(args.search, context)
    if html_content:
        with open("extracted_context.html", 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"\n✅ Saved HTML: extracted_context.html")
    
    print(f"\n✅ Saved: extracted_context.txt ({len(context):,} chars)")
