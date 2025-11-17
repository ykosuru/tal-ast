"""
Quick Context Extractor v3.5 - WITH LLM SUMMARIZATION
Works with any version of doc_indexer.py
Now with optional Claude AI summarization!
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import pickle
import re
import base64
import io
import os

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

# Anthropic API support
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("⚠ Anthropic SDK not available. Install: pip install anthropic")


class QuickContextExtractor:
    """Extract context using UniversalFileSearcher with optional LLM summarization"""
    
    def __init__(
        self,
        universal_index: str = "./universal_index",
        hybrid_index: str = "./hybrid_index",
        api_key: Optional[str] = None
    ):
        self.universal_index = Path(universal_index)
        self.hybrid_index = Path(hybrid_index)
        self.file_cache = {}
        self.pdf_cache = {}
        
        # Initialize searcher
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
        
        # Initialize Anthropic client (optional)
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.anthropic_client = None
        
        if ANTHROPIC_AVAILABLE and self.api_key:
            try:
                self.anthropic_client = anthropic.Anthropic(api_key=self.api_key)
                print("✓ Claude API initialized")
            except Exception as e:
                print(f"⚠ Could not initialize Claude API: {e}")
    
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
        show_explanations: bool = False,
        use_llm: bool = False,
        llm_instructions: Optional[str] = None,
        llm_model: str = "claude-sonnet-4-20250514"
    ) -> str:
        """
        Extract context for query with optional LLM summarization
        
        Args:
            query: Search query
            max_matches: Number of matches
            lines_before: Lines before match
            lines_after: Lines after match
            embed_images: Embed PDF images
            diversify_results: Mix file types
            file_type_filter: Filter by 'code', 'pdf', 'text'
            validate_query_terms: Validate results contain query terms
            min_query_terms: Minimum query terms required
            dedup_similarity: Dedup threshold
            show_explanations: Show ranking signals
            use_llm: Use Claude AI to summarize results
            llm_instructions: Custom instructions for LLM
            llm_model: Claude model to use
        
        Returns:
            Context string (or LLM summary if use_llm=True)
        """
        
        print(f"\n{'='*70}")
        print(f"Query: {query}")
        print(f"{'='*70}")
        print(f"Settings:")
        print(f"  - Context: {lines_before} lines before, {lines_after} lines after")
        print(f"  - Images: {'Enabled' if embed_images else 'Disabled'}")
        print(f"  - Diversification: {'Enabled' if diversify_results else 'Disabled'}")
        print(f"  - Validation: {'Enabled' if validate_query_terms else 'Disabled'}")
        print(f"  - LLM Summary: {'Enabled' if use_llm else 'Disabled'}")
        
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
        
        # Format context
        formatted_context = self._format_contexts(contexts, query)
        
        # Use LLM summarization if requested
        if use_llm:
            if not self.anthropic_client:
                print("\n⚠ LLM summarization requested but API not available")
                print("  Set ANTHROPIC_API_KEY environment variable or pass api_key")
                return formatted_context
            
            print("\n🤖 Generating LLM summary...")
            return self._summarize_with_llm(
                query,
                formatted_context,
                contexts,
                llm_instructions,
                llm_model
            )
        
        return formatted_context
    
    def _summarize_with_llm(
        self,
        query: str,
        formatted_context: str,
        contexts: List[Dict[str, Any]],
        custom_instructions: Optional[str],
        model: str
    ) -> str:
        """
        Use Claude AI to summarize the search results
        Handles both text and images properly
        """
        
        # Parse context to extract images and text
        parsed = self._parse_for_llm(formatted_context, contexts)
        
        print(f"  - Text blocks: {len(parsed['text_blocks'])}")
        print(f"  - Images: {len(parsed['images'])}")
        print(f"  - Sources: {len(parsed['sources'])}")
        
        # Build Claude message
        message_content = self._build_llm_message(query, parsed, custom_instructions)
        
        try:
            # Call Claude API
            response = self.anthropic_client.messages.create(
                model=model,
                max_tokens=4096,
                messages=[{
                    "role": "user",
                    "content": message_content
                }]
            )
            
            summary = response.content[0].text
            
            # Format output with metadata
            output_parts = []
            output_parts.append(f"# LLM Summary: {query}\n")
            output_parts.append(f"**Model:** {model}")
            output_parts.append(f"**Sources:** {len(parsed['sources'])}")
            output_parts.append(f"**Images:** {len(parsed['images'])}")
            output_parts.append(f"**Tokens:** {response.usage.input_tokens:,} in / {response.usage.output_tokens:,} out\n")
            output_parts.append(f"{'='*70}\n")
            output_parts.append(summary)
            output_parts.append(f"\n{'='*70}")
            output_parts.append(f"\n## Sources Referenced")
            for source in parsed['sources']:
                output_parts.append(f"- {source}")
            
            result = '\n'.join(output_parts)
            
            print(f"✓ Summary generated ({response.usage.output_tokens} tokens)")
            
            return result
            
        except Exception as e:
            print(f"✗ LLM API Error: {e}")
            print("  Returning original context instead")
            return formatted_context
    
    def _parse_for_llm(
        self,
        formatted_context: str,
        contexts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Parse formatted context for LLM consumption
        Extracts text blocks, images, and sources
        """
        parsed = {
            'text_blocks': [],
            'images': [],
            'sources': []
        }
        
        # Extract sources from contexts
        for ctx in contexts:
            source = ctx.get('source_file', '')
            if source and source not in parsed['sources']:
                parsed['sources'].append(source)
        
        # Extract images from contexts (more reliable than regex)
        for ctx in contexts:
            if ctx.get('type') == 'pdf' and ctx.get('images'):
                for img in ctx['images']:
                    parsed['images'].append({
                        'source': ctx['source_file'],
                        'page': ctx.get('page_num', 0) + 1,
                        'format': img.get('format', 'PNG'),
                        'data': img['base64'],
                        'width': img.get('width', 0),
                        'height': img.get('height', 0)
                    })
        
        # Extract text blocks (split by sections)
        sections = formatted_context.split('='*70)
        for section in sections:
            # Remove image markdown to get clean text
            section_clean = re.sub(
                r'!\[.*?\]\(data:image/[^;]+;base64,[A-Za-z0-9+/=]+\)',
                '[PDF Image]',
                section
            )
            
            # Keep substantial text blocks
            if len(section_clean.strip()) > 100:
                parsed['text_blocks'].append(section_clean.strip())
        
        return parsed
    
    def _build_llm_message(
        self,
        query: str,
        parsed: Dict[str, Any],
        custom_instructions: Optional[str]
    ) -> List[Dict]:
        """
        Build Claude API message with multimodal content
        """
        content_blocks = []
        
        # 1. Instructions
        if custom_instructions:
            instructions = custom_instructions
        else:
            instructions = f"""I searched for "{query}" in our codebase and documentation. 

Please provide a comprehensive analysis covering:

1. **Overview** - What did we find related to "{query}"?
2. **Key Technical Details** - Code snippets, configurations, or technical specifications
3. **Visual Information** - If there are diagrams/images, explain what they show
4. **Implementation Details** - How is this implemented? What are the key components?
5. **Sources** - Which documents/files contain the most relevant information?

Be thorough but concise. Focus on actionable insights."""
        
        content_blocks.append({
            "type": "text",
            "text": instructions
        })
        
        # 2. Add images with context
        if parsed['images']:
            content_blocks.append({
                "type": "text",
                "text": f"\n\n## PDF Images Found ({len(parsed['images'])} total)\n\n"
            })
            
            for i, img in enumerate(parsed['images'], 1):
                # Add image description
                content_blocks.append({
                    "type": "text",
                    "text": f"**Image {i}** from `{img['source']}` (Page {img['page']}):\n"
                })
                
                # Add the actual image
                media_type = f"image/{img['format'].lower()}"
                if media_type == "image/jpg":
                    media_type = "image/jpeg"
                
                content_blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": img['data']
                    }
                })
                
                content_blocks.append({
                    "type": "text",
                    "text": "\n"
                })
        
        # 3. Add text/code content
        if parsed['text_blocks']:
            content_blocks.append({
                "type": "text",
                "text": "\n## Code and Text Context\n\n"
            })
            
            # Combine and limit text size
            combined_text = "\n\n---\n\n".join(parsed['text_blocks'])
            
            # Truncate if too large (Claude has context limits)
            max_text_length = 80000  # Leave room for images
            if len(combined_text) > max_text_length:
                combined_text = combined_text[:max_text_length] + "\n\n[... content truncated for length ...]"
            
            content_blocks.append({
                "type": "text",
                "text": combined_text
            })
        
        # 4. Add sources footer
        if parsed['sources']:
            sources_text = "\n\n## Source Files\n\n" + "\n".join(
                f"- {source}" for source in parsed['sources']
            )
            content_blocks.append({
                "type": "text",
                "text": sources_text
            })
        
        return content_blocks
    
    # ... (keep all existing methods: _search_with_universal, _validate_query_terms, etc.)
    
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
            search_kwargs = {
                'query': query,
                'top_k': max_results,
                'verbose': False
            }
            
            try:
                test_sig = self.searcher.search.__code__.co_varnames
                
                if 'file_type_filter' in test_sig and file_type_filter:
                    search_kwargs['file_type_filter'] = file_type_filter
                
                if 'use_query_expansion' in test_sig:
                    search_kwargs['use_query_expansion'] = True
                
                if 'explain' in test_sig:
                    search_kwargs['explain'] = show_explanations
            except:
                pass
            
            search_results = self.searcher.search(**search_kwargs)
            
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
        seen_chunks = set()
        unique = []
        
        for r in results:
            chunk_key = f"{r['source_file']}:{r['chunk_index']}"
            if chunk_key not in seen_chunks:
                seen_chunks.add(chunk_key)
                unique.append(r)
        
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


def create_html_content(query: str, context: str, include_images: bool = True) -> Optional[str]:
    """
    Enhanced HTML creation with better styling and optional image embedding
    
    Args:
        query: Search query
        context: Context string
        include_images: Include embedded images (can be large)
    
    Returns:
        HTML string or None if no content
    """
    
    # Check if we have images
    has_images = "data:image/png;base64," in context or "data:image/jpeg;base64," in context
    
    if not has_images and not context.strip():
        return None
    
    print(f"\n📄 Creating HTML view...")
    
    # Enhanced CSS
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Search Results: {query}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px 40px;
        }}
        
        .header h1 {{
            font-size: 2em;
            margin-bottom: 10px;
        }}
        
        .header .meta {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        
        .content {{
            padding: 40px;
        }}
        
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #667eea;
            padding-bottom: 15px;
            margin: 30px 0 20px 0;
        }}
        
        h2 {{
            color: #34495e;
            border-bottom: 2px solid #95a5a6;
            padding-bottom: 10px;
            margin: 25px 0 15px 0;
            font-size: 1.5em;
        }}
        
        h3 {{
            color: #7f8c8d;
            margin: 20px 0 10px 0;
            font-size: 1.2em;
        }}
        
        pre {{
            background: #2c3e50;
            color: #ecf0f1;
            padding: 20px;
            border-radius: 8px;
            overflow-x: auto;
            font-size: 13px;
            line-height: 1.5;
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            border-left: 4px solid #667eea;
            margin: 15px 0;
        }}
        
        .image-container {{
            margin: 30px 0;
            padding: 25px;
            background: #f8f9fa;
            border-radius: 12px;
            border: 2px solid #dee2e6;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        
        .image-container img {{
            max-width: 100%;
            height: auto;
            max-height: 1200px;
            border: 1px solid #ccc;
            border-radius: 8px;
            box-shadow: 0 8px 16px rgba(0,0,0,0.15);
            margin: 15px auto;
            display: block;
            transition: transform 0.3s ease;
        }}
        
        .image-container img:hover {{
            transform: scale(1.02);
            cursor: zoom-in;
        }}
        
        .image-label {{
            color: #666;
            font-size: 0.9em;
            margin-top: 10px;
            font-style: italic;
        }}
        
        hr {{
            border: none;
            border-top: 2px solid #e0e0e0;
            margin: 30px 0;
        }}
        
        .match-header {{
            background: #f8f9fa;
            padding: 15px 20px;
            border-radius: 8px;
            margin: 20px 0;
            border-left: 4px solid #667eea;
        }}
        
        .score {{
            display: inline-block;
            background: #667eea;
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: bold;
        }}
        
        .ranking-signals {{
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            padding: 12px 16px;
            border-radius: 6px;
            margin: 10px 0;
            font-size: 0.9em;
        }}
        
        .ranking-signals ul {{
            margin: 5px 0 0 20px;
        }}
        
        code {{
            background: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 0.9em;
        }}
        
        @media print {{
            body {{ background: white; }}
            .container {{ box-shadow: none; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 Search Results</h1>
            <div class="meta">Query: <strong>{query}</strong></div>
        </div>
        <div class="content">
"""
    
    lines = context.split('\n')
    in_code = False
    in_match_header = False
    
    for line in lines:
        # Handle code blocks
        if line.startswith('```'):
            html_content += "</pre>\n" if in_code else "<pre>\n"
            in_code = not in_code
            continue
        
        # Handle images
        if '![Page Image](data:image/' in line and include_images:
            match = re.search(r'!\[.*?\]\(data:image/(png|jpeg);base64,([^)]+)\)', line)
            if match:
                img_type = match.group(1)
                img_data = match.group(2)
                html_content += f'<div class="image-container">\n'
                html_content += f'<img src="data:image/{img_type};base64,{img_data}" alt="PDF Page Image" />\n'
                html_content += f'<div class="image-label">PDF Page Image</div>\n'
                html_content += '</div>\n'
            continue
        
        # Handle headers
        if line.startswith('# '):
            html_content += f"<h1>{line[2:]}</h1>\n"
        elif line.startswith('## Match'):
            html_content += f'<div class="match-header">{line[3:]}'
            in_match_header = True
        elif line.startswith('## '):
            html_content += f"<h2>{line[3:]}</h2>\n"
        elif line.startswith('### '):
            html_content += f"<h3>{line[4:]}</h3>\n"
        
        # Handle score lines
        elif line.startswith('Score:') and in_match_header:
            score_value = line.split(':')[1].strip()
            html_content += f' <span class="score">Score: {score_value}</span></div>\n'
            in_match_header = False
        
        # Handle ranking signals
        elif line.strip().startswith('•'):
            if '•' in line:
                html_content += f'<div class="ranking-signals"><ul>\n'
                html_content += f'<li>{line.strip()[1:].strip()}</li>\n'
        
        # Handle separators
        elif line.strip() == '='*70:
            html_content += '<hr>\n'
        
        # Handle regular content
        elif line.strip() and not in_code:
            html_content += f"<p>{line}</p>\n"
        elif in_code:
            html_content += f"{line}\n"
    
    if in_code:
        html_content += "</pre>\n"
    
    html_content += """
        </div>
    </div>
    <script>
        // Add click-to-zoom for images
        document.querySelectorAll('.image-container img').forEach(img => {
            img.addEventListener('click', function() {
                if (this.style.maxWidth === 'none') {
                    this.style.maxWidth = '100%';
                    this.style.cursor = 'zoom-in';
                } else {
                    this.style.maxWidth = 'none';
                    this.style.cursor = 'zoom-out';
                }
            });
        });
    </script>
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
    show_explanations: bool = False,
    use_llm: bool = False,
    llm_instructions: Optional[str] = None,
    llm_model: str = "claude-sonnet-4-20250514",
    api_key: Optional[str] = None
) -> str:
    """
    Extract context using UniversalFileSearcher with optional LLM summarization
    
    Args:
        query: Search query
        lines_before: Lines before match
        lines_after: Lines after match
        max_matches: Number of matches
        embed_images: Embed PDF images
        diversify_results: Mix file types
        file_type_filter: Filter by 'code', 'pdf', 'text'
        validate_query_terms: Validate results contain query
        min_query_terms: Minimum query terms required
        dedup_similarity: Dedup threshold (0.85 = 85% similar)
        show_explanations: Show ranking signals
        use_llm: Use Claude AI to summarize results
        llm_instructions: Custom instructions for LLM
        llm_model: Claude model to use
        api_key: Anthropic API key (or use ANTHROPIC_API_KEY env var)
    
    Usage:
        # Basic search
        context = quick_extract("cutoff times")
        
        # With LLM summary
        summary = quick_extract("cutoff times", use_llm=True)
        
        # Custom LLM instructions
        summary = quick_extract(
            "payment validation",
            use_llm=True,
            llm_instructions="Explain the validation rules with code examples"
        )
        
        # PDFs only with LLM
        summary = quick_extract("architecture diagram", file_type_filter='pdf', use_llm=True)
    """
    extractor = QuickContextExtractor(api_key=api_key)
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
        show_explanations=show_explanations,
        use_llm=use_llm,
        llm_instructions=llm_instructions,
        llm_model=llm_model
    )


if __name__ == "__main__":
    import argparse
    import sys
    
    if len(sys.argv) == 1:
        print("Usage: python3 search_index.py --search <term>")
        sys.exit(0)
    
    parser = argparse.ArgumentParser(description='Context Extractor v3.5 - With LLM')
    parser.add_argument('--search', '-s', required=True, help='Search term')
    parser.add_argument('--max', type=int, default=5, help='Max matches')
    parser.add_argument('--type', choices=['code', 'pdf', 'text'], help='File type filter')
    parser.add_argument('--explain', action='store_true', help='Show explanations')
    parser.add_argument('--no-images', action='store_true', help='Disable PDF images')
    parser.add_argument('--llm', action='store_true', help='Use LLM to summarize')
    parser.add_argument('--instructions', '-i', help='Custom LLM instructions')
    parser.add_argument('--model', default='claude-sonnet-4-20250514', help='LLM model')
    
    args = parser.parse_args()
    
    print("="*70)
    print("CONTEXT EXTRACTOR v3.5 - WITH LLM SUMMARIZATION")
    print("="*70)
    
    context = quick_extract(
        args.search,
        max_matches=args.max,
        file_type_filter=args.type,
        embed_images=not args.no_images,
        show_explanations=args.explain,
        validate_query_terms=True,
        use_llm=args.llm,
        llm_instructions=args.instructions,
        llm_model=args.model
    )
    
    print("\n" + "="*70)
    print("RESULT")
    print("="*70)
    print(context[:2000])
    if len(context) > 2000:
        print(f"\n... ({len(context) - 2000:,} more chars)")
    
    # Save text output
    output_file = "llm_summary.md" if args.llm else "extracted_context.txt"
    with open(output_file, 'w') as f:
        f.write(context)
    print(f"\n✅ Saved: {output_file} ({len(context):,} chars)")
    
    # Save HTML (only if not using LLM)
    if not args.llm:
        html_content = create_html_content(args.search, context)
        if html_content:
            with open("extracted_context.html", 'w', encoding='utf-8') as f:
                f.write(html_content)
            print(f"✅ Saved HTML: extracted_context.html")
