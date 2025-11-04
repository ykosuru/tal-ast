"""
Quick Context Extractor - Works with existing indexes
Handles both text/code AND images!

Features:
- Text/Code: Extract N lines before/after (configurable)
- Images: Extract and embed full image as base64
- PDFs: Extract images from matched pages

Usage:
    from quick_context_extractor import QuickContextExtractor
    
    extractor = QuickContextExtractor()
    context = extractor.extract(
        query="payment flow diagram",
        lines_before=200,
        lines_after=200,
        embed_images=True
    )
    print(context)
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
        embed_images: bool = True  # Embed images from PDFs
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
        
        Returns:
            Formatted context string
        """
        print(f"\nQuery: {query}")
        print(f"Settings: {lines_before} lines before, {lines_after} lines after")
        print(f"Image embedding: {'Enabled' if embed_images else 'Disabled'}")
        
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
                lines_after,
                embed_images
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
            # Extract images from PDF
            return self._extract_pdf_context(
                source_path,
                result,
                lines_before,
                lines_after
            )
        else:
            # Extract text/code context
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
            'file_type': result['file_type']
        }
    
    def _extract_pdf_context(
        self,
        source_path: Path,
        result: Dict[str, Any],
        lines_before: int,
        lines_after: int
    ) -> Optional[Dict[str, Any]]:
        """Extract context from PDF (text + images)"""
        
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
            
            # Extract images from page
            images = []
            if hasattr(page, 'images') and page.images:
                print(f"  Found {len(page.images)} images on page {page_num + 1}")
                
                for img_idx, img_info in enumerate(page.images):
                    try:
                        # Get image from page
                        img_bbox = (img_info['x0'], img_info['top'], img_info['x1'], img_info['bottom'])
                        img = page.within_bbox(img_bbox).to_image()
                        
                        # Convert to PIL Image
                        img_pil = img.original
                        
                        # Convert to base64
                        buffered = io.BytesIO()
                        img_pil.save(buffered, format="PNG")
                        img_base64 = base64.b64encode(buffered.getvalue()).decode()
                        
                        images.append({
                            'index': img_idx,
                            'base64': img_base64,
                            'width': img_info['width'],
                            'height': img_info['height']
                        })
                        
                        print(f"    ✓ Extracted image {img_idx + 1} ({img_info['width']:.0f}x{img_info['height']:.0f})")
                    except Exception as e:
                        print(f"    ⚠ Could not extract image {img_idx + 1}: {e}")
            
            # If no images found, try to extract as raster
            if not images:
                try:
                    # Render page as image
                    img = page.to_image(resolution=150)
                    img_pil = img.original
                    
                    # Convert to base64
                    buffered = io.BytesIO()
                    img_pil.save(buffered, format="PNG")
                    img_base64 = base64.b64encode(buffered.getvalue()).decode()
                    
                    images.append({
                        'index': 0,
                        'base64': img_base64,
                        'width': page.width,
                        'height': page.height,
                        'full_page': True
                    })
                    
                    print(f"  ✓ Rendered full page as image")
                except Exception as e:
                    print(f"  ⚠ Could not render page: {e}")
            
            # Get context pages (pages before and after)
            context_pages_before = []
            context_pages_after = []
            
            # Pages before (limited)
            for i in range(max(0, page_num - 2), page_num):
                try:
                    ctx_page = pdf.pages[i]
                    ctx_text = ctx_page.extract_text() or ""
                    context_pages_before.append({
                        'page_num': i,
                        'text': ctx_text
                    })
                except:
                    pass
            
            # Pages after (limited)
            for i in range(page_num + 1, min(len(pdf.pages), page_num + 3)):
                try:
                    ctx_page = pdf.pages[i]
                    ctx_text = ctx_page.extract_text() or ""
                    context_pages_after.append({
                        'page_num': i,
                        'text': ctx_text
                    })
                except:
                    pass
            
            print(f"  ✓ Page {page_num + 1} with {len(images)} image(s)")
            
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
                'file_type': result['file_type']
            }
            
        except Exception as e:
            print(f"  ⚠ PDF extraction error: {e}")
            # Fallback to text extraction
            return self._extract_text_context(source_path, result, lines_before, lines_after)
    
    def _find_page_in_pdf(self, pdf, match_text: str) -> Optional[int]:
        """Find which page contains the match text"""
        match_words = set(match_text.lower().split()[:10])  # First 10 words
        
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
        
        # Count non-printable characters
        non_printable = sum(1 for c in text if not c.isprintable() and c not in '\t\n\r ')
        
        # If more than 20% non-printable, likely garbage
        if non_printable / len(text) > 0.2:
            return True
        
        # Check for common binary indicators
        garbage_indicators = [
            b'\x00'.decode('latin-1'),  # Null bytes
            '\ufffd',  # Replacement character (encoding failed)
        ]
        
        for indicator in garbage_indicators:
            if indicator in text:
                return True
        
        # Check if mostly non-ASCII
        non_ascii = sum(1 for c in text if ord(c) > 127)
        if non_ascii / len(text) > 0.5:
            return True
        
        return False
    
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
    
    def _is_binary_file(self, file_path: Path) -> bool:
        """Check if file is binary"""
        # Check by extension first
        binary_extensions = {
            '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
            '.zip', '.tar', '.gz', '.exe', '.dll', '.so', '.dylib',
            '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.ico',
            '.mp3', '.mp4', '.avi', '.mov', '.wav',
            '.bin', '.dat', '.db'
        }
        
        if file_path.suffix.lower() in binary_extensions:
            return True
        
        # Check file content (read first 8KB)
        try:
            with open(file_path, 'rb') as f:
                chunk = f.read(8192)
            
            # Check for null bytes
            if b'\x00' in chunk:
                return True
            
            # Check if mostly printable
            text_chars = bytearray({7, 8, 9, 10, 12, 13, 27} | set(range(0x20, 0x100)) - {0x7f})
            non_text = chunk.translate(None, text_chars)
            
            # If more than 30% non-text, consider binary
            if len(non_text) / max(1, len(chunk)) > 0.3:
                return True
        except:
            pass
        
        return False
    
    def _clean_line(self, line: str) -> str:
        """Remove non-printable characters from line"""
        # Remove control characters but keep tabs and newlines
        cleaned = ''.join(char for char in line if char.isprintable() or char in '\t\n\r')
        return cleaned
    
    def _read_file_cached(self, file_path: Path) -> List[str]:
        """Read file with caching and binary detection"""
        file_path_str = str(file_path)
        
        if file_path_str in self.file_cache:
            return self.file_cache[file_path_str]
        
        # Check if binary
        if self._is_binary_file(file_path):
            print(f"  ⚠ Skipping binary file: {file_path.name}")
            return []
        
        try:
            # Try different encodings
            for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                        lines = [self._clean_line(line.rstrip('\n\r')) for line in f.readlines()]
                    
                    # Filter out lines with too many non-ASCII chars (likely garbage)
                    cleaned_lines = []
                    for line in lines:
                        # Count non-ASCII characters
                        non_ascii = sum(1 for c in line if ord(c) > 127)
                        # If more than 50% non-ASCII, skip
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
        """Format contexts for LLM (handles both text and images)"""
        if not contexts:
            return "No contexts found."
        
        parts = []
        
        # Header
        parts.append(f"# Code Context for: {query}\n")
        parts.append(f"Found {len(contexts)} relevant sections\n")
        
        text_count = sum(1 for c in contexts if c.get('type') == 'text')
        pdf_count = sum(1 for c in contexts if c.get('type') == 'pdf')
        parts.append(f"- Text/Code: {text_count}")
        parts.append(f"- PDFs with images: {pdf_count}\n")
        
        # Each context
        for i, ctx in enumerate(contexts, 1):
            parts.append(f"\n{'='*70}")
            parts.append(f"## Match {i}: {ctx['source_file']}")
            parts.append(f"Score: {ctx['score']:.3f}")
            parts.append(f"{'='*70}\n")
            
            if ctx.get('type') == 'pdf':
                # Format PDF with images
                parts.append(self._format_pdf_context(ctx))
            else:
                # Format text/code context
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
        parts.append(f"Total images: {total_images}")
        
        return '\n'.join(parts)
    
    def _format_text_context(self, ctx: Dict[str, Any]) -> str:
        """Format text/code context"""
        parts = []
        
        parts.append(f"Type: Text/Code")
        parts.append(f"Location: Lines {ctx['match_start']}-{ctx['match_end']}\n")
        
        # Determine language for syntax highlighting
        ext = Path(ctx['source_file']).suffix.lower()
        lang_map = {
            '.py': 'python', '.java': 'java', '.c': 'c', '.cpp': 'cpp',
            '.js': 'javascript', '.tal': 'tal', '.cbl': 'cobol',
            '.sql': 'sql', '.sh': 'bash', '.txt': 'text'
        }
        lang = lang_map.get(ext, 'text')
        
        parts.append(f"```{lang}")
        
        # Show limited context
        before = ctx['before_lines'][-50:] if len(ctx['before_lines']) > 50 else ctx['before_lines']
        after = ctx['after_lines'][:50] if len(ctx['after_lines']) > 50 else ctx['after_lines']
        
        # Lines before
        if before:
            start_num = ctx['match_start'] - len(before)
            for j, line in enumerate(before):
                parts.append(f"{start_num + j:4d} | {line}")
            parts.append("")
        
        # Match (highlighted)
        parts.append(">>> RELEVANT MATCH <<<")
        parts.append("")
        
        for j, line in enumerate(ctx['match_lines'][:30]):  # Limit match display
            parts.append(f"{ctx['match_start'] + j:4d} | {line}")
        
        parts.append("")
        parts.append(">>> END MATCH <<<")
        
        # Lines after
        if after:
            parts.append("")
            for j, line in enumerate(after):
                parts.append(f"{ctx['match_end'] + j:4d} | {line}")
        
        parts.append("```\n")
        
        return '\n'.join(parts)
    
    def _format_pdf_context(self, ctx: Dict[str, Any]) -> str:
        """Format PDF context with embedded images"""
        parts = []
        
        parts.append(f"Type: PDF")
        parts.append(f"Page: {ctx['page_num'] + 1}\n")
        
        # Show context from previous pages (if any)
        if ctx['context_pages_before']:
            parts.append(f"### Context from Previous Pages\n")
            for page_ctx in ctx['context_pages_before']:
                parts.append(f"**Page {page_ctx['page_num'] + 1}:**")
                # Show first few lines
                lines = page_ctx['text'].split('\n')[:10]
                for line in lines:
                    if line.strip():
                        parts.append(f"  {line}")
            parts.append("")
        
        # Show page text
        parts.append(f"### Page {ctx['page_num'] + 1} Content\n")
        
        if ctx['page_text']:
            parts.append("```")
            for line in ctx['page_text'][:50]:  # Limit lines
                parts.append(line)
            parts.append("```\n")
        
        # Embed images
        if ctx['images']:
            parts.append(f"### Images from Page {ctx['page_num'] + 1}\n")
            
            for img in ctx['images']:
                if img.get('full_page'):
                    parts.append(f"**Full Page Rendered as Image:**\n")
                else:
                    parts.append(f"**Image {img['index'] + 1}** ({img['width']:.0f}x{img['height']:.0f} pixels):\n")
                
                # Embed as markdown image (base64)
                parts.append(f"![Image](data:image/png;base64,{img['base64']})\n")
                
                # Also provide raw base64 for LLMs that need it
                parts.append("<details>")
                parts.append("<summary>Base64 Image Data (click to expand)</summary>\n")
                parts.append("```")
                parts.append(f"data:image/png;base64,{img['base64'][:100]}...")
                parts.append("```")
                parts.append("</details>\n")
        
        # Show context from next pages (if any)
        if ctx['context_pages_after']:
            parts.append(f"### Context from Following Pages\n")
            for page_ctx in ctx['context_pages_after']:
                parts.append(f"**Page {page_ctx['page_num'] + 1}:**")
                # Show first few lines
                lines = page_ctx['text'].split('\n')[:10]
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
    max_matches: int = 5,
    embed_images: bool = True
) -> str:
    """
    One-liner: extract context for query (text + images)
    
    Args:
        query: Search query
        lines_before: Lines before match (for text/code)
        lines_after: Lines after match (for text/code)
        max_matches: Number of matches
        embed_images: Embed full images from PDFs
    
    Usage:
        # Text/code files: get 200 lines before/after
        context = quick_extract("payment validation", lines_before=200, lines_after=200)
        
        # PDFs with images: embed full images
        context = quick_extract("payment flow diagram", embed_images=True)
        
        # Configurable context size
        context = quick_extract("error handling", lines_before=100, lines_after=100)
    """
    extractor = QuickContextExtractor()
    return extractor.extract(
        query,
        max_matches,
        lines_before,
        lines_after,
        embed_images=embed_images
    )


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        query = ' '.join(sys.argv[1:])
    else:
        query = "payment drawdown processing"
    
    print("="*70)
    print("QUICK CONTEXT EXTRACTION")
    print("With Image Embedding Support!")
    print("="*70)
    
    # Check if we should look for images
    image_keywords = ['diagram', 'chart', 'flow', 'graph', 'image', 'figure', 'illustration']
    has_image_keyword = any(kw in query.lower() for kw in image_keywords)
    
    if has_image_keyword:
        print(f"\n💡 Query contains image keywords - enabling image extraction")
    
    context = quick_extract(
        query,
        lines_before=200,
        lines_after=200,
        max_matches=5,
        embed_images=True  # Always try to embed images
    )
    
    print("\n" + "="*70)
    print("RESULT")
    print("="*70)
    print(context[:2000])  # Show first 2000 chars
    if len(context) > 2000:
        print(f"\n... ({len(context) - 2000:,} more characters)")
    
    # Save to file
    with open("extracted_context.txt", 'w') as f:
        f.write(context)
    
    # Also save as HTML if there are images
    if "data:image/png;base64," in context:
        print(f"\n📸 Images detected - creating HTML view...")
        
        # Convert to proper HTML with rendered images
        html_content = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Context: """ + query + """</title>
    <style>
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            max-width: 1400px;
            margin: 20px auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            color: #34495e;
            border-bottom: 2px solid #95a5a6;
            padding-bottom: 8px;
            margin-top: 30px;
        }
        h3 {
            color: #7f8c8d;
            margin-top: 20px;
        }
        pre {
            background: #2c3e50;
            color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
            font-size: 13px;
            line-height: 1.5;
        }
        .match-highlight {
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 15px 0;
            border-radius: 4px;
        }
        .image-container {
            margin: 20px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
            border: 2px solid #dee2e6;
        }
        .image-container img {
            max-width: 100%;
            height: auto;
            border: 1px solid #ccc;
            border-radius: 4px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            display: block;
            margin: 10px 0;
        }
        .image-info {
            color: #6c757d;
            font-size: 14px;
            margin-bottom: 10px;
        }
        .score-badge {
            background: #28a745;
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 14px;
            display: inline-block;
        }
        .file-badge {
            background: #17a2b8;
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 14px;
            display: inline-block;
            margin-left: 10px;
        }
        .separator {
            border-top: 2px dashed #dee2e6;
            margin: 40px 0;
        }
        code {
            background: #f8f9fa;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }
        .summary {
            background: #e7f3ff;
            padding: 20px;
            border-radius: 8px;
            margin-top: 30px;
            border-left: 4px solid #3498db;
        }
    </style>
</head>
<body>
    <div class="container">
"""
        
        # Parse the context and convert to HTML with proper image tags
        lines = context.split('\n')
        in_code_block = False
        in_match = False
        
        for line in lines:
            # Handle code blocks
            if line.startswith('```'):
                if in_code_block:
                    html_content += "</pre>\n"
                    in_code_block = False
                else:
                    html_content += "<pre>\n"
                    in_code_block = True
                continue
            
            # Handle markdown images - convert to proper HTML img tags
            if '![Image](data:image/png;base64,' in line:
                # Extract base64 data
                import re
                match = re.search(r'!\[Image\]\(data:image/png;base64,([^)]+)\)', line)
                if match:
                    base64_data = match.group(1)
                    html_content += f'<div class="image-container">\n'
                    html_content += f'<div class="image-info">📸 Embedded Image</div>\n'
                    html_content += f'<img src="data:image/png;base64,{base64_data}" alt="Extracted Image" />\n'
                    html_content += '</div>\n'
                continue
            
            # Handle headers
            if line.startswith('# '):
                html_content += f"<h1>{line[2:]}</h1>\n"
            elif line.startswith('## '):
                html_content += f"<h2>{line[3:]}</h2>\n"
            elif line.startswith('### '):
                html_content += f"<h3>{line[4:]}</h3>\n"
            
            # Handle match highlights
            elif '>>> RELEVANT MATCH <<<' in line:
                html_content += '<div class="match-highlight">\n'
                html_content += '<strong>>>> RELEVANT MATCH <<<</strong><br>\n'
                in_match = True
            elif '>>> END MATCH <<<' in line:
                html_content += '<strong>>>> END MATCH <<<</strong>\n'
                html_content += '</div>\n'
                in_match = False
            
            # Handle separators
            elif line.strip() == '='*70:
                html_content += '<div class="separator"></div>\n'
            
            # Handle score and file type badges
            elif line.startswith('Score:'):
                score = line.split(':')[1].strip()
                html_content += f'<span class="score-badge">Score: {score}</span>\n'
            elif line.startswith('Type:'):
                file_type = line.split(':')[1].strip()
                html_content += f'<span class="file-badge">Type: {file_type}</span>\n'
            
            # Skip base64 details sections
            elif '<details>' in line or '</details>' in line or '<summary>' in line or '</summary>' in line:
                continue
            
            # Regular text
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
        
        with open("extracted_context.html", 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"\n✅ Saved HTML with RENDERED images: extracted_context.html")
        print(f"  🌐 Open in browser to see actual images!")
        print(f"  💡 Try: open extracted_context.html  (Mac)")
        print(f"       or: start extracted_context.html  (Windows)")
        print(f"       or: xdg-open extracted_context.html  (Linux)")
        
        # Also save images as separate PNG files
        print(f"\n📸 Extracting images as separate files...")
        image_count = 0
        for line in context.split('\n'):
            if '![Image](data:image/png;base64,' in line:
                match = re.search(r'data:image/png;base64,([^)]+)', line)
                if match:
                    try:
                        base64_data = match.group(1)
                        img_data = base64.b64decode(base64_data)
                        
                        img_filename = f"extracted_image_{image_count + 1}.png"
                        with open(img_filename, 'wb') as f:
                            f.write(img_data)
                        
                        print(f"  ✓ Saved: {img_filename}")
                        image_count += 1
                    except Exception as e:
                        print(f"  ⚠ Could not save image {image_count + 1}: {e}")
        
        if image_count > 0:
            print(f"\n✅ Extracted {image_count} image(s) as separate PNG files")
        
        print(f"\n{'='*70}")
        print(f"📁 FILES CREATED:")
        print(f"{'='*70}")
        print(f"  1. extracted_context.txt  - Text format (for copying)")
        print(f"  2. extracted_context.html - HTML with images (OPEN THIS IN BROWSER!)")
        if image_count > 0:
            print(f"  3. extracted_image_*.png  - Individual image files ({image_count} images)")
        print(f"{'='*70}")
    
    print(f"\n✓ Saved to: extracted_context.txt")
    print(f"  Length: {len(context):,} chars")
    
    # Usage examples
    print("\n" + "="*70)
    print("USAGE EXAMPLES")
    print("="*70)
    print("""
# Extract text/code with 200 lines before/after
python quick_context_extractor.py "payment validation"

# Extract with custom line counts
context = quick_extract("error handling", lines_before=100, lines_after=100)

# Extract images from PDFs
context = quick_extract("payment flow diagram", embed_images=True)

# Adjust number of matches
context = quick_extract("OFAC screening", max_matches=10)

# Code only (skip PDFs)
extractor = QuickContextExtractor()
context = extractor.extract("validation", index_type="hybrid")
""")
