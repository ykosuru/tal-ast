"""
Quick Context Extractor v3.0 - FULLY COMPATIBLE
Works with any version of doc_indexer.py
Auto-detects available search parameters
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import pickle
import re
import base64
import io

# Import searcher
try:
    from doc_indexer import UniversalFileSearcher
    UNIVERSAL_SEARCHER_AVAILABLE = True
except ImportError:
    UNIVERSAL_SEARCHER_AVAILABLE = False
    print("⚠ doc_indexer.py not found")

# PDF support
try:
    import pdfplumber
    from PIL import Image
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("⚠ PDF support not available. Install: pip install pdfplumber Pillow")


class QuickContextExtractor:
    """Extract context using UniversalFileSearcher"""
    
    def __init__(
        self,
        universal_index: str = "./universal_index",
        hybrid_index: str = "./hybrid_index"
    ):
        self.universal_index = Path(universal_index)
        self.hybrid_index = Path(hybrid_index)
        self.file_cache = {}
        self.pdf_cache = {}
        
        if UNIVERSAL_SEARCHER_AVAILABLE and self.universal_index.exists():
            try:
                self.searcher = UniversalFileSearcher(str(self.universal_index))
                print("✓ UniversalFileSearcher loaded (best ranking!)")
            except Exception as e:
                print(f"⚠ Could not load UniversalFileSearcher: {e}")
                self.searcher = None
        else:
            self.searcher = None
            print("⚠ UniversalFileSearcher not available")
        
        if PDF_AVAILABLE:
            print("✓ PDF/image support enabled")
    
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
        show_explanations: bool = False
    ) -> str:
        """Extract context for query"""
        
        print(f"\n{'='*70}")
        print(f"Query: {query}")
        print(f"{'='*70}")
        print(f"Settings:")
        print(f"  - Context: {lines_before} lines before, {lines_after} lines after")
        print(f"  - Images: {'Enabled' if embed_images else 'Disabled'}")
        print(f"  - Diversification: {'Enabled' if diversify_results else 'Disabled'}")
        print(f"  - Validation: {'Enabled' if validate_query_terms else 'Disabled'}")
        print(f"  - Dedup threshold: {dedup_similarity:.2f}")
        
        # Search
        results = self._search_with_universal(
            query,
            max_matches * 3,
            file_type_filter,
            show_explanations
        )
        
        if not results:
            return "No results found."
        
        print(f"\nInitial results: {len(results)}")
        
        # Validate
        if validate_query_terms:
            results = self._validate_query_terms(results, query, min_query_terms)
            print(f"After validation: {len(results)} results")
        
        if not results:
            return f"No results containing: {query}"
        
        # Deduplicate
        results = self._deduplicate_results(results, dedup_similarity)
        print(f"After dedup: {len(results)} unique results")
        
        # Diversify
        if diversify_results and not file_type_filter:
            results = self._diversify_results(results, max_matches)
        else:
            results = results[:max_matches]
        
        print(f"\nExtracting context from {len(results)} matches...")
        
        # Extract contexts
        contexts = []
        for i, result in enumerate(results, 1):
            print(f"[{i}/{len(results)}] {result['source_file']} ({result['file_type']}) - Score: {result['score']:.2f}")
            
            if show_explanations and 'explanation' in result:
                print(f"  Ranking:")
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
        
        return self._format_contexts(contexts, query)
    
    def _search_with_universal(
        self,
        query: str,
        max_results: int,
        file_type_filter: Optional[str],
        show_explanations: bool
    ) -> List[Dict[str, Any]]:
        """Search using UniversalFileSearcher with compatible parameters"""
        
        if not self.searcher:
            print("\n⚠ No searcher available")
            return []
        
        print("\n🔍 Searching with UniversalFileSearcher...")
        
        try:
            # Build kwargs with ONLY the parameters we know exist
            search_kwargs = {
                'query': query,
                'top_k': max_results,
                'verbose': False
            }
            
            # Try to add optional parameters (graceful if they don't exist)
            try:
                # Test if these parameters exist by doing a dry run
                test_sig = self.searcher.search.__code__.co_varnames
                
                if 'file_type_filter' in test_sig and file_type_filter:
                    search_kwargs['file_type_filter'] = file_type_filter
                
                if 'use_query_expansion' in test_sig:
                    search_kwargs['use_query_expansion'] = True
                
                if 'explain' in test_sig:
                    search_kwargs['explain'] = show_explanations
                
                # NOTE: NOT using enable_multi_term - it's handled internally by the searcher
                
            except:
                pass  # If parameter detection fails, just use basic params
            
            # Execute search
            search_results = self.searcher.search(**search_kwargs)
            
            # Convert to our format
            results = []
            for r in search_results:
                results.append({
                    'text': r['text'],
                    'source_file': r['source_file'],
                    'source_path': '',
                    'file_type': r['file_type'],
                    'chunk_index': r.get('chunk_index', 0),
                    'score': r['score'],
                    'bm25_score': r.get('bm25_score', 0),
                    'explanation': r.get('explanation', []),
                    'capabilities': r.get('capabilities', []),
                    'keywords': r.get('keywords', []),
                    'index': 'universal'
                })
            
            print(f"✓ Found {len(results)} results")
            return results
            
        except Exception as e:
            print(f"⚠ Search error: {e}")
            return []
    
    def _validate_query_terms(
        self,
        results: List[Dict],
        query: str,
        min_terms: int = 1
    ) -> List[Dict]:
        """Validate results contain query terms"""
        stopwords = {
            'how', 'what', 'when', 'where', 'why', 'who', 'which',
            'do', 'does', 'did', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'will', 'would', 'should', 'could', 'can',
            'a', 'an', 'the', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'from', 'by', 'as', 'into', 'through', 'during', 'before', 'after',
            'and', 'or', 'but', 'if', 'then', 'than', 'so', 'because',
            'this', 'that', 'these', 'those'
        }
        
        query_words = re.findall(r'\b[\w-]+\b', query.lower())
        query_terms = [w for w in query_words if w not in stopwords and len(w) > 2]
        
        if not query_terms:
            query_terms = query_words
        
        print(f"  Validating against: {', '.join(query_terms)}")
        
        validated = []
        for result in results:
            text = result.get('text', '').lower()
            matches = sum(1 for term in query_terms if term in text)
            
            if matches >= min_terms:
                result['query_term_matches'] = matches
                validated.append(result)
        
        validated.sort(
            key=lambda x: (x.get('query_term_matches', 0), x.get('score', 0)),
            reverse=True
        )
        
        return validated
    
    def _deduplicate_results(
        self,
        results: List[Dict],
        threshold: float = 0.85
    ) -> List[Dict]:
        """Remove duplicates"""
        # Exact duplicates
        seen_chunks = set()
        unique = []
        
        for r in results:
            chunk_key = f"{r['source_file']}:{r['chunk_index']}"
            if chunk_key not in seen_chunks:
                seen_chunks.add(chunk_key)
                unique.append(r)
        
        # Similarity-based
        final = []
        file_texts = defaultdict(list)
        
        for r in unique:
            source_file = r['source_file']
            text = r['text']
            
            is_duplicate = False
            for existing_text in file_texts[source_file]:
                if self._text_similarity(text, existing_text) > threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                file_texts[source_file].append(text)
                final.append(r)
        
        print(f"  Dedup: {len(results)} → {len(unique)} → {len(final)}")
        return final
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Jaccard similarity"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _diversify_results(self, results: List[Dict], max_results: int) -> List[Dict]:
        """Diversify by file type"""
        by_type = defaultdict(list)
        for r in results:
            by_type[r.get('file_type', 'unknown')].append(r)
        
        print(f"  Diversifying across {len(by_type)} types:")
        for ft, items in by_type.items():
            print(f"    {ft}: {len(items)}")
        
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
        """Extract context"""
        source_path = self._find_source_file(result)
        
        if not source_path or not source_path.exists():
            print(f"  ⚠ File not found")
            return None
        
        is_pdf = source_path.suffix.lower() == '.pdf'
        
        if is_pdf and PDF_AVAILABLE and embed_images:
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
        """Extract text context"""
        
        if self._is_garbage_text(result['text']):
            print(f"  ⚠ Binary data")
            return None
        
        lines = self._read_file_cached(source_path)
        if not lines:
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
        """Extract PDF context with image"""
        
        try:
            if str(source_path) in self.pdf_cache:
                pdf = self.pdf_cache[str(source_path)]
            else:
                pdf = pdfplumber.open(source_path)
                self.pdf_cache[str(source_path)] = pdf
            
            page_num = self._find_page_in_pdf(pdf, result['text'])
            
            if page_num is None:
                return self._extract_text_context(source_path, result, lines_before, lines_after)
            
            page = pdf.pages[page_num]
            page_text = page.extract_text() or ""
            text_lines = page_text.split('\n')
            
            images = []
            
            try:
                print(f"  Rendering page {page_num + 1}...")
                
                img = page.to_image(resolution=100)
                img_pil = img.original
                
                # Resize if needed
                max_width = 800
                if img_pil.width > max_width:
                    ratio = max_width / img_pil.width
                    new_height = int(img_pil.height * ratio)
                    img_pil = img_pil.resize((max_width, new_height), Image.Resampling.LANCZOS)
                
                # Compress
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
                    print(f"  ⚠ Image: {size_mb:.1f}MB")
                
                images.append({
                    'index': 0,
                    'base64': img_base64,
                    'format': img_format,
                    'width': img_pil.width,
                    'height': img_pil.height,
                    'full_page': True,
                    'size_mb': round(size_mb, 2)
                })
                
                print(f"  ✓ Page {page_num + 1} → {img_format} ({size_mb:.1f}MB)")
                
            except Exception as e:
                print(f"  ⚠ Render error: {e}")
            
            # Context pages
            context_pages_before = []
            context_pages_after = []
            
            for i in range(max(0, page_num - 2), page_num):
                try:
                    ctx_text = pdf.pages[i].extract_text() or ""
                    context_pages_before.append({'page_num': i, 'text': ctx_text[:500]})
                except:
                    pass
            
            for i in range(page_num + 1, min(len(pdf.pages), page_num + 3)):
                try:
                    ctx_text = pdf.pages[i].extract_text() or ""
                    context_pages_after.append({'page_num': i, 'text': ctx_text[:500]})
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
            print(f"  ⚠ PDF error: {e}")
            return self._extract_text_context(source_path, result, lines_before, lines_after)
    
    def _find_page_in_pdf(self, pdf, match_text: str) -> Optional[int]:
        """Find page with match"""
        match_words = set(match_text.lower().split()[:10])
        
        best_page = 0
        best_count = 0
        
        for page_num, page in enumerate(pdf.pages):
            page_text = (page.extract_text() or "").lower()
            page_words = set(page_text.split())
            count = len(match_words & page_words)
            
            if count > best_count:
                best_count = count
                best_page = page_num
        
        return best_page if best_count > 0 else None
    
    def _is_garbage_text(self, text: str) -> bool:
        """Check for binary"""
        if not text or len(text) < 10:
            return False
        
        non_printable = sum(1 for c in text if not c.isprintable() and c not in '\t\n\r ')
        return non_printable / len(text) > 0.2
    
    def _find_source_file(self, result: Dict[str, Any]) -> Optional[Path]:
        """Find source file"""
        filename = result['source_file']
        
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
        """Check binary"""
        binary_exts = {'.pdf', '.doc', '.docx', '.png', '.jpg', '.zip', '.exe'}
        
        if file_path.suffix.lower() in binary_exts:
            return True
        
        try:
            with open(file_path, 'rb') as f:
                return b'\x00' in f.read(8192)
        except:
            pass
        
        return False
    
    def _clean_line(self, line: str) -> str:
        """Clean line"""
        return ''.join(c for c in line if c.isprintable() or c in '\t\n\r')
    
    def _read_file_cached(self, file_path: Path) -> List[str]:
        """Read file"""
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
        """Find match"""
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
        """Format for LLM"""
        if not contexts:
            return "No contexts found."
        
        parts = []
        
        parts.append(f"# Code Context for: {query}\n")
        parts.append(f"Found {len(contexts)} relevant sections\n")
        
        text_count = sum(1 for c in contexts if c.get('type') == 'text')
        pdf_count = sum(1 for c in contexts if c.get('type') == 'pdf')
        parts.append(f"— Text/Code: {text_count}")
        parts.append(f"— PDFs: {pdf_count}\n")
        
        for i, ctx in enumerate(contexts, 1):
            parts.append(f"\n{'='*70}")
            parts.append(f"## Match {i}: {ctx['source_file']}")
            parts.append(f"Score: {ctx['score']:.3f}")
            if ctx.get('query_term_matches', 0) > 0:
                parts.append(f"Query matches: {ctx['query_term_matches']}")
            
            if ctx.get('explanation'):
                parts.append(f"Ranking:")
                for exp in ctx['explanation'][:3]:
                    parts.append(f"  • {exp}")
            
            parts.append(f"{'='*70}\n")
            
            if ctx.get('type') == 'pdf':
                parts.append(self._format_pdf_context(ctx))
            else:
                parts.append(self._format_text_context(ctx))
        
        parts.append(f"\n{'='*70}")
        parts.append("## Summary")
        parts.append(f"Total matches: {len(contexts)}")
        
        total_lines = sum(
            len(c.get('before_lines', [])) + len(c.get('match_lines', [])) + len(c.get('after_lines', []))
            for c in contexts if c.get('type') == 'text'
        )
        total_images = sum(len(c.get('images', [])) for c in contexts if c.get('type') == 'pdf')
        
        parts.append(f"Total lines: {total_lines}")
        parts.append(f"Total images: {total_images}\n")
        
        return '\n'.join(parts)
    
    def _format_text_context(self, ctx: Dict[str, Any]) -> str:
        """Format text"""
        parts = []
        
        parts.append(f"Type: Text/Code")
        parts.append(f"Location: Lines {ctx['match_start']}-{ctx['match_end']}\n")
        
        ext = Path(ctx['source_file']).suffix.lower()
        lang_map = {
            '.py': 'python', '.java': 'java', '.c': 'c', '.cpp': 'cpp',
            '.js': 'javascript', '.tal': 'tal', '.cbl': 'cobol',
            '.sql': 'sql', '.sh': 'bash', '.TXT': 'tal'
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
        """Format PDF"""
        parts = []
        
        parts.append(f"Type: PDF")
        parts.append(f"Page: {ctx['page_num'] + 1}\n")
        
        if ctx['context_pages_before']:
            parts.append(f"### Previous Pages\n")
            for page_ctx in ctx['context_pages_before']:
                parts.append(f"**Page {page_ctx['page_num'] + 1}:**")
                for line in page_ctx['text'].split('\n')[:5]:
                    if line.strip():
                        parts.append(f"  {line}")
                parts.append("")
        
        if ctx['images']:
            parts.append(f"### Page {ctx['page_num'] + 1} Image\n")
            for img in ctx['images']:
                parts.append(f"**Full page** ({img['width']}x{img['height']} px):\n")
                parts.append(f"![Page Image](data:image/png;base64,{img['base64']})\n")
        
        if ctx['page_text'] and len(ctx['page_text']) < 100:
            parts.append(f"### Page Text\n")
            parts.append("```")
            for line in ctx['page_text'][:30]:
                if line.strip():
                    parts.append(line)
            parts.append("```\n")
        
        if ctx['context_pages_after']:
            parts.append(f"### Following Pages\n")
            for page_ctx in ctx['context_pages_after']:
                parts.append(f"**Page {page_ctx['page_num'] + 1}:**")
                for line in page_ctx['text'].split('\n')[:5]:
                    if line.strip():
                        parts.append(f"  {line}")
                parts.append("")
        
        return '\n'.join(parts)


def create_html_content(query: str, context: str) -> str:
    """Create HTML with images"""
    if "data:image/png;base64," not in context:
        return None
    
    print(f"\n📄 Creating HTML...")
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Context: {query}</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial; max-width: 1400px; margin: 20px auto; padding: 20px; background: #f5f5f5; }}
        .container {{ background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #95a5a6; padding-bottom: 8px; margin-top: 30px; }}
        h3 {{ color: #7f8c8d; margin-top: 20px; }}
        pre {{ background: #2c3e50; color: #ecf0f1; padding: 15px; border-radius: 5px; overflow-x: auto; font-size: 13px; line-height: 1.5; }}
        .image-container {{ margin: 20px 0; padding: 20px; background: #f8f9fa; border-radius: 8px; border: 2px solid #dee2e6; text-align: center; }}
        .image-container img {{ max-width: 100%; height: auto; max-height: 1200px; border: 1px solid #ccc; border-radius: 4px; box-shadow: 0 4px 8px rgba(0,0,0,0.15); margin: 15px auto; display: block; }}
    </style>
</head>
<body>
    <div class="container">
"""
    
    lines = context.split('\n')
    in_code = False
    
    for line in lines:
        if line.startswith('```'):
            html_content += "</pre>\n" if in_code else "<pre>\n"
            in_code = not in_code
            continue
        
        if '![Page Image](data:image/png;base64,' in line:
            match = re.search(r'!\[Page Image\]\(data:image/png;base64,([^)]+)\)', line)
            if match:
                html_content += f'<div class="image-container"><img src="data:image/png;base64,{match.group(1)}" /></div>\n'
            continue
        
        if line.startswith('# '):
            html_content += f"<h1>{line[2:]}</h1>\n"
        elif line.startswith('## '):
            html_content += f"<h2>{line[3:]}</h2>\n"
        elif line.startswith('### '):
            html_content += f"<h3>{line[4:]}</h3>\n"
        elif line.strip() == '='*70:
            html_content += '<hr>\n'
        elif line.strip() and not in_code:
            html_content += f"{line}<br>\n"
        elif in_code:
            html_content += f"{line}\n"
    
    if in_code:
        html_content += "</pre>\n"
    
    html_content += "</div></body></html>"
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
    show_explanations: bool = False
) -> str:
    """
    Extract context using UniversalFileSearcher
    
    Usage:
        context = quick_extract("cutoff times")
        context = quick_extract("payment validation", file_type_filter='code')
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
        show_explanations=show_explanations
    )


if __name__ == "__main__":
    import argparse
    import sys
    
    if len(sys.argv) == 1:
        print("Usage: python3 search_index.py --search <term>")
        sys.exit(0)
    
    parser = argparse.ArgumentParser(description='Context Extractor v3.0')
    parser.add_argument('--search', '-s', required=True, help='Search term')
    parser.add_argument('--max', type=int, default=5, help='Max matches')
    parser.add_argument('--type', choices=['code', 'pdf', 'text'], help='File type filter')
    parser.add_argument('--explain', action='store_true', help='Show explanations')
    parser.add_argument('--no-images', action='store_true', help='Disable PDF images')
    
    args = parser.parse_args()
    
    print("="*70)
    print("CONTEXT EXTRACTOR v3.0 - COMPATIBLE")
    print("="*70)
    
    context = quick_extract(
        args.search,
        max_matches=args.max,
        file_type_filter=args.type,
        embed_images=not args.no_images,
        show_explanations=args.explain,
        validate_query_terms=True
    )
    
    print("\n" + "="*70)
    print("RESULT")
    print("="*70)
    print(context[:2000])
    if len(context) > 2000:
        print(f"\n... ({len(context) - 2000:,} more chars)")
    
    with open("extracted_context.txt", 'w') as f:
        f.write(context)
    
    html_content = create_html_content(args.search, context)
    if html_content:
        with open("extracted_context.html", 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"\n✅ Saved HTML: extracted_context.html")
    
    print(f"\n✅ Saved: extracted_context.txt ({len(context):,} chars)")
