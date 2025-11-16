"""
Quick Context Extractor v2.4 - IMPROVED PDF IMAGE COMPRESSION
Works with existing indexes
Handles both text/code AND images!


- Reduced PDF rendering resolution (200 -> 100) for smaller images
- Added image resizing if width exceeds 800px
- JPEG compression with quality=85 for better size management
- Fallback to PNG if JPEG fails
- Size warnings for images >3MB
- Better memory management for large PDFs
- Two-stage deduplication: exact chunk + similarity-based
- Prevents duplicate code results from same file
- Configurable similarity threshold (default 0.85)
- Shows deduplication stats in output
- Query term validation: Only keep results that actually contain search terms
- Prevents PDF results from dominating with irrelevant content
- Validates against both original and stemmed query terms
- Configurable strictness for matching
- Result diversification (mix of PDFs and code)
- Better source path resolution using metadata
- Configurable file type filtering

Features:
- Text/Code: Extract N lines before/after (configurable)
- Images: Extract and embed full image as base64
- PDFs: Extract images from matched pages with compression
- Smart result mixing: Get both code and PDFs in results
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
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


class QuickContextExtractor:
    """
    Extract context without requiring line numbers in index
    Works with existing indexes!
    NOW WITH QUERY TERM VALIDATION!
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
        index_type: str = "both",  # "both", "universal", or "hybrid"
        embed_images: bool = True,  # Embed images from PDFs
        diversify_results: bool = True,  # Mix different file types
        file_type_filter: Optional[str] = None,  # Filter: 'code', 'pdf', 'text'
        validate_query_terms: bool = False,  # NEW: Validate results contain query terms
        min_query_terms: int = 1,  # NEW: Minimum query terms that must match
        dedup_similarity: float = 0.85  # NEW: Similarity threshold for deduplication (0.0-1.0)
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
            validate_query_terms: Only keep results that contain query terms
            min_query_terms: Minimum query terms that must appear in result
            dedup_similarity: Similarity threshold for deduplication (0.0-1.0)
        
        Returns:
            Formatted context string
        """
        
        print(f"\nQuery: {query}")
        print(f"Settings: {lines_before} lines before, {lines_after} lines after")
        print(f"Image embedding: {'Enabled' if embed_images else 'Disabled'}")
        print(f"Result diversification: {'Enabled' if diversify_results else 'Disabled'}")
        print(f"Query term validation: {'Enabled' if validate_query_terms else 'Disabled'}")
        print(f"Deduplication threshold: {dedup_similarity:.2f}")
        
        # Store dedup_similarity for use in _search_indexes
        self.dedup_similarity = dedup_similarity
        
        # Search indexes
        results = self._search_indexes(
            query,
            max_matches * 5 if diversify_results else max_matches * 3,  # Get more candidates for validation
            index_type
        )
        
        if not results:
            return "No results found."
        
        # NEW: Validate query terms
        if validate_query_terms:
            results = self._validate_query_terms(results, query, min_query_terms)
            print(f"After validation: {len(results)} results contain query terms")
        
        if not results:
            return f"No results found containing the query terms: {query}"
        
        # Apply file type filter
        if file_type_filter:
            results = [r for r in results if r['file_type'] == file_type_filter]
            print(f"Filtered to {len(results)} {file_type_filter} results")
        
        # Diversify results if enabled
        if diversify_results and not file_type_filter:
            results = self._diversify_results(results, max_matches)
        else:
            results = results[:max_matches]
        
        print(f"\nFound {len(results)} matches")
        
        # Extract context from each result
        contexts = []
        for i, result in enumerate(results, 1):
            print(f"[{i}/{len(results)}] {result['source_file']} ({result['file_type']})")
            
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
    
    def _validate_query_terms(
        self,
        results: List[Dict],
        query: str,
        min_terms: int = 1
    ) -> List[Dict]:
        """
        Validate that results contain query terms
        Checks both original query and stemmed version
        """
        # Extract query terms (remove common words)
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
        
        # Filter out stopwords and very short terms
        query_terms = [
            word for word in query_words
            if word not in stopwords and len(word) > 2
        ]
        
        if not query_terms:
            # If no meaningful terms, use all words
            query_terms = query_words
        
        print(f"Validating against terms: {', '.join(query_terms)}")
        
        # Also try to get stemmed versions if available
        stemmed_terms = self._try_stem_terms(query_terms)
        
        validated_results = []
        
        for result in results:
            text = result.get('text', '').lower()
            
            # Count how many query terms appear in the result
            matches = 0
            
            for term in query_terms:
                if term in text:
                    matches += 1
            
            # Also check stemmed versions
            for stemmed_term in stemmed_terms:
                if stemmed_term in text and stemmed_term not in query_terms:
                    matches += 1
            
            # Check if result meets minimum threshold
            if matches >= min_terms:
                result['query_term_matches'] = matches
                validated_results.append(result)
        
        # Sort by number of query term matches (higher is better)
        validated_results.sort(key=lambda x: (x.get('query_term_matches', 0), x.get('score', 0)), reverse=True)
        
        return validated_results

    def _try_stem_terms(self, terms: List[str]) -> List[str]:
        """
        Try to get stemmed versions of terms using the indexer's stemmer
        """
        stemmed = []
        
        try:
            # Try to import and use the stemmer from universal_indexer
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
        """
        Search one or both indexes
        """
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
        
        dedup_threshold = getattr(self, 'dedup_similarity', 0.85)
        
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
    
    def _extract_context_for_result(
        self,
        result: Dict[str, Any],
        lines_before: int,
        lines_after: int,
        embed_images
    ) -> Optional[Dict[str, Any]]:
        """Extract context from each result"""
        # Find source file
        source_path = self._find_source_file(result)
        
        if not source_path or not source_path.exists():
            print(f"  ⚠ Could not find source file")
            return None
        is_pdf = source_path.suffix.lower() == '.pdf'
        
        # Extract based on file type
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
        
        # Check if result text looks like garbage (binary data)
        result_text = result['text']
        if self._is_garbage_text(result_text):
            print(f"  ⚠ Skipping: appears to be binary/image data")
            return None
        
        # Read file (with caching)
        lines = self._read_file_cached(source_path)
        
        if not lines:
            print(f"  ⚠ Could not read file or file is binary")
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
    
    def _extract_pdf_context(
        self,
        source_path: Path,
        result: Dict[str, Any],
        lines_before: int,
        lines_after: int
    ) -> Optional[Dict[str, Any]]:
        """Extract context from PDF (renders full page as single image)"""
        
        try:
            # Open PDF
            if str(source_path) in self.pdf_cache:
                pdf = self.pdf_cache[str(source_path)]
            else:
                pdf = pdfplumber.open(source_path)
                self.pdf_cache[str(source_path)] = pdf
            
            # Find which page the match is on
            page_num = self._find_page_in_pdf(pdf, result['text'])
            
            if page_num is None:
                print(f"  ⚠ Could not find match in PDF")
                return self._extract_text_context(source_path, result, lines_before, lines_after)
            
            page = pdf.pages[page_num]
            
            # Extract text from page
            page_text = page.extract_text() or ""
            text_lines = page_text.split('\n')
            
            # Render FULL PAGE as single image
            images = []
            
            try:
                print(f"  Rendering full page {page_num + 1} as single image...")
                
                # Render entire page at lower resolution to reduce size
                img = page.to_image(resolution=100)  # Reduced from 200
                img_pil = img.original
                
                # Resize image if too large (optional additional compression)
                max_width = 800
                if img_pil.width > max_width:
                    ratio = max_width / img_pil.width
                    new_height = int(img_pil.height * ratio)
                    img_pil = img_pil.resize((max_width, new_height), Image.Resampling.LANCZOS)
                
                # Convert to base64 with better compression
                buffered = io.BytesIO()
                
                # Try JPEG first for better compression (if no transparency needed)
                try:
                    # Convert RGBA to RGB if necessary for JPEG
                    if img_pil.mode in ('RGBA', 'LA'):
                        background = Image.new('RGB', img_pil.size, (255, 255, 255))
                        background.paste(img_pil, mask=img_pil.split()[-1] if img_pil.mode == 'RGBA' else None)
                        img_pil = background
                    
                    img_pil.save(buffered, format="JPEG", quality=85, optimize=True)
                    img_format = "JPEG"
                except:
                    # Fallback to PNG if JPEG fails
                    img_pil.save(buffered, format="PNG", optimize=True)
                    img_format = "PNG"
                
                img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                
                # Check size and warn if too large
                size_mb = len(img_base64) * 3 / 4 / (1024 * 1024)  # Approximate size in MB
                if size_mb > 3:  # Warn if larger than 3MB
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
            
            # Get context pages (pages before and after)
            context_pages_before = []
            context_pages_after = []
            
            # Pages before (2 pages max)
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
            
            # Pages after (2 pages max)
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
        """Find which page contains the match text"""
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
        """Check if text looks like binary garbage"""
        if not text or len(text) < 10:
            return False
        
        non_printable = sum(1 for c in text if not c.isprintable() and c not in '\t\n\r ')
        if non_printable / len(text) > 0.2:
            return True
        
        garbage_indicators = [
            b'\x00'.decode('latin-1'),
            '\ufffd',
        ]
        
        for indicator in garbage_indicators:
            if indicator in text:
                return True
        
        non_ascii = sum(1 for c in text if ord(c) > 127)
        if non_ascii / len(text) > 0.5:
            return True
        
        return False
    
    def _find_source_file(self, result: Dict[str, Any]) -> Optional[Path]:
        """
        Find the source file path
        NOW WITH BETTER PATH RESOLUTION!
        """
        # Try source_path from metadata first (if available from fixed indexer)
        if result.get('source_path') and result['source_path']:
            source_path = Path(result['source_path'])
            if source_path.exists():
                print(f"  ✓ Using stored path: {source_path}")
                return source_path
        
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
        """Check if file is binary"""
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
                text_chars = bytearray({7, 8, 9, 10, 12, 13, 27} | set (range(0x20, 0x100)) - {0x7f})
                non_text = chunk.translate(None, text_chars)

                if len(non_text) /max (1, len(chunk)) > 0.3:
                    return True
        except:
            pass
        
        return False
    
    def _clean_line(self, line: str) -> str:
        """Remove non-printable characters from line"""
        cleaned = ''.join(char for char in line if char.isprintable() or char in '\t\n\r')
        return cleaned
    
    def _read_file_cached(self, file_path: Path) -> List[str]:
        """Read file with caching and binary detection"""
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
        """Find where match_text appears in lines"""
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
                    self._similarity(search_line, file_line) > 0.7):
                    matches += 1
            
            if matches >= len(search_lines) * 0.7:
                start_line = i
                end_line = min(i + len(match_lines), len(lines))
                return (start_line, end_line)
        
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
    
    def _format_contexts(self, contexts: List[Dict[str, Any]], query: str) -> str:
        """Format contexts for LLM (handles both text and images)"""
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
            '.js': 'javascript', '.ts': 'typescript', '.tal': 'tal', '.cbl': 'cobol',
            '.sql': 'sql', '.sh': 'bash', '.txt': 'text', '.TXT':'tal'
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


def create_html_content(query: str, context: str) -> str:
    """Also save as HTML if there are images"""
    if "data:image/png;base64," not in context:
        return None
    
    print(f"\n📄 Images detected - creating HTML view...")
    
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
        
        if '![Page Image](data:image/png;base64,' in line or '![Image](data:image/png;base64,' in line:
            match = re.search(r'!\[(?:Page )?Image\]\(data:image/png;base64,([^)]+)\)', line)
            if match:
                base64_data = match.group(1)
                html_content += f'<div class="image-container">\n'
                html_content += f'<img src="data:image/png;base64,{base64_data}" alt="Full Page Image" />\n'
                html_content += '</div>\n'
            continue
        
        if line.startswith('# '):
            html_content += f"<h1>{line[2:]}</h1>\n"
        elif line.startswith('## '):
            html_content += f"<h2>{line[3:]}</h2>\n"
        elif line.startswith('### '):
            html_content += f"<h3>{line[4:]}</h3>\n"
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


# Quick usage function
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
    dedup_similarity: float = 0.85
) -> str:
    """
    One-liner: extract context for query (text + images)
    
    Args:
        query: Search query
        lines_before: Lines before match (for text/code)
        lines_after: Lines after match (for text/code)
        max_matches: Number of matches
        embed_images: Embed full images from PDFs
        diversify_results: Get mix of file types
        file_type_filter: Only return specific file type ('code', 'pdf', 'text')
        validate_query_terms: Only keep results containing query terms
        min_query_terms: Minimum query terms that must appear in result
        dedup_similarity: Similarity threshold for deduplication (0.85 = 85% similar)
    
    Usage:
        # Mixed results with validation (default)
        context = quick_extract("payment validation", validate_query_terms=True)
        
        # Code only, validated
        context = quick_extract("payment validation", file_type_filter='code')
        
        # PDFs only
        context = quick_extract("flow diagram", file_type_filter='pdf')
        
        # Strict matching (require at least 2 query terms)
        context = quick_extract("payment wire transfer", min_query_terms=2)
        
        # Less strict deduplication (allow more similar results)
        context = quick_extract("payment", dedup_similarity=0.95)
    """
    extractor = QuickContextExtractor()
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
        dedup_similarity=dedup_similarity
    )


def quick_search(query: str):
    """
    # Check if we should look for specific file types
    image_keywords = ['diagram', 'chart', 'flow', 'graph', 'image', 'figure', 'illustration', 'picture']
    has_image_keyword = any(kw in query.lower() for kw in image_keywords)
    
    if has_image_keyword:
        print(f"\n🖼 Query contains image keywords - will include PDFs with images")
    
    context = quick_extract(
        query,
        lines_before=200,
        lines_after=200,
        max_matches=5,
        embed_images=True,
        diversify_results=True,
        validate_query_terms=True,  # ✅ Validate query terms
        min_query_terms=1  # ✅ At least 1 query term must match
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
    
    # Also save as HTML if there are images
    html_content = create_html_content(query, context)
    if html_content:
        with open("extracted_context.html", 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"\n✅ Saved HTML: extracted_context.html")
    
    print(f"\n✅ Saved to: extracted_context.txt")
    print(f"  Length: {len(context):,} chars")


if __name__ == "__main__":
    import argparse
    import sys
    
    if len(sys.argv) ==0:
        print("Usage: python3 search_index.py --search <search term>")
        sys.exit(0)
    
    parser = argparse.ArgumentParser(
        description='Index TAL code with Wire Processing Taxonomy',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
            %(prog)s -s "drawdown"
        """
    )
    
    parser.add_argument('--search', '-s',
                       help='search term')
    
    args = parser.parse_args()
    
    if not args.search:
        print("Please provide a search term within quotes")
        parser.print_help()
        sys.exit(1)
    else:
        query = args.search
        print(f"Search term: {query}")
    
    print("="*70)
    print("QUICK CONTEXT EXTRACTION v2.4")
    print("With Improved PDF Compression!")
    print("="*70)
    
    # Check if we should look for specific file types
    image_keywords = ['diagram', 'chart', 'flow', 'graph', 'image', 'figure', 'illustration']
    has_image_keyword = any(kw in query.lower() for kw in image_keywords)
    
    if has_image_keyword:
        print(f"\n🖼 Query contains image keywords - will include PDFs with images")
    
    context = quick_extract(
        query,
        lines_before=200,
        lines_after=200,
        max_matches=5,
        embed_images=True,
        diversify_results=True,
        validate_query_terms=True,  # ✅ Validate query terms
        min_query_terms=1  # ✅ At least 1 query term must match
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
    
    # Also save as HTML if there are images
    html_content = create_html_content(query, context)
    if html_content:
        with open("extracted_context.html", 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"\n✅ Saved HTML: extracted_context.html")
    
    print(f"\n✅ Saved to: extracted_context.txt")
    print(f"  Length: {len(context):,} chars")
