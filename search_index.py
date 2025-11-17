"""
Quick Context Extractor v3.6 - WITH ASYNC LLM SUMMARIZATION
Enhanced with async API calls and HTML rendering
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import pickle
import re
import base64
import io
import os
import asyncio
from datetime import datetime

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
        self.async_anthropic_client = None
        
        if ANTHROPIC_AVAILABLE and self.api_key:
            try:
                self.anthropic_client = anthropic.Anthropic(api_key=self.api_key)
                self.async_anthropic_client = anthropic.AsyncAnthropic(api_key=self.api_key)
                print("✓ Claude API initialized (sync + async)")
            except Exception as e:
                print(f"⚠ Could not initialize Claude API: {e}")
    
    async def llm_wrapper(
        self,
        system_prompt: str,
        user_prompt: List[Dict[str, Any]],
        content_type: str = "multimodal",
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        """
        Async wrapper for Claude API chat completion
        
        Args:
            system_prompt: System prompt for Claude
            user_prompt: User message content (list of content blocks for multimodal)
            content_type: "multimodal" or "text"
            model: Claude model to use
            max_tokens: Maximum tokens to generate
        
        Returns:
            Dict with response text and metadata
        """
        
        if not self.async_anthropic_client:
            raise ValueError("Anthropic async client not initialized")
        
        try:
            # Build messages based on content type
            if content_type == "multimodal":
                # user_prompt is already a list of content blocks
                messages = [{
                    "role": "user",
                    "content": user_prompt
                }]
            else:
                # user_prompt should be a string
                if isinstance(user_prompt, list):
                    user_prompt = user_prompt[0].get("text", "")
                messages = [{
                    "role": "user",
                    "content": user_prompt
                }]
            
            # Call Claude API asynchronously
            response = await self.async_anthropic_client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=messages
            )
            
            # Extract response text
            response_text = response.content[0].text
            
            return {
                "success": True,
                "text": response_text,
                "model": model,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                },
                "stop_reason": response.stop_reason
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "text": ""
            }
    
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
            if not self.async_anthropic_client:
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
        Use Claude AI to summarize the search results (synchronous wrapper)
        Saves both markdown and HTML outputs
        """
        
        # Parse context to extract images and text
        parsed = self._parse_for_llm(formatted_context, contexts)
        
        print(f"  - Text blocks: {len(parsed['text_blocks'])}")
        print(f"  - Images: {len(parsed['images'])}")
        print(f"  - Sources: {len(parsed['sources'])}")
        
        # Build prompts
        system_prompt = self._build_system_prompt()
        user_content = self._build_user_content(query, parsed, custom_instructions)
        
        # Determine content type
        content_type = "multimodal" if parsed['images'] else "text"
        
        try:
            # Call async LLM wrapper using asyncio.run()
            print(f"  - Calling Claude API ({content_type} mode)...")
            response = asyncio.run(
                self.llm_wrapper(
                    system_prompt=system_prompt,
                    user_prompt=user_content,
                    content_type=content_type,
                    model=model,
                    max_tokens=4096
                )
            )
            
            if not response["success"]:
                print(f"✗ LLM API Error: {response.get('error', 'Unknown error')}")
                print("  Returning original context instead")
                return formatted_context
            
            summary = response["text"]
            usage = response["usage"]
            
            # Create metadata
            metadata = {
                "query": query,
                "model": model,
                "sources": parsed['sources'],
                "num_images": len(parsed['images']),
                "num_text_blocks": len(parsed['text_blocks']),
                "input_tokens": usage['input_tokens'],
                "output_tokens": usage['output_tokens'],
                "timestamp": datetime.now().isoformat()
            }
            
            # Format markdown output
            markdown_output = self._format_markdown_summary(summary, metadata)
            
            # Save markdown
            with open("llm_summary.md", 'w', encoding='utf-8') as f:
                f.write(markdown_output)
            print(f"✓ Saved: llm_summary.md")
            
            # Generate and save HTML
            html_output = self._format_html_summary(summary, metadata, parsed)
            with open("llm_summary.html", 'w', encoding='utf-8') as f:
                f.write(html_output)
            print(f"✓ Saved: llm_summary.html")
            
            print(f"✓ Summary generated ({usage['output_tokens']} tokens)")
            
            return markdown_output
            
        except Exception as e:
            print(f"✗ Error during LLM summarization: {e}")
            import traceback
            traceback.print_exc()
            print("  Returning original context instead")
            return formatted_context
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for Claude"""
        return """You are an expert technical analyst and documentation specialist. 
Your role is to analyze code, documentation, and technical diagrams to provide clear, actionable insights.

When analyzing search results:
- Focus on concrete technical details and implementation specifics
- Explain what the code/documentation shows, not just what it says
- Identify key patterns, dependencies, and relationships
- Highlight important configurations, constants, or edge cases
- When images are present, describe what they show and how they relate to the text
- Structure your response logically with clear sections
- Be thorough but concise - focus on what matters

Format your response in clean markdown with appropriate headers, code blocks, and lists."""
    
    def _build_user_content(
        self,
        query: str,
        parsed: Dict[str, Any],
        custom_instructions: Optional[str]
    ) -> List[Dict]:
        """Build user content blocks for Claude API"""
        
        content_blocks = []
        
        # 1. Instructions
        if custom_instructions:
            instructions = custom_instructions
        else:
            instructions = f"""I searched our codebase and documentation for: **"{query}"**

Please provide a comprehensive analysis with the following structure:

## 1. Executive Summary
Quick overview of what was found related to "{query}"

## 2. Key Findings
Main technical details, configurations, or implementations discovered

## 3. Code Analysis
If code snippets are present, explain:
- What the code does
- Key functions, classes, or procedures
- Important variables, constants, or configurations
- Dependencies or integrations

## 4. Visual Analysis
If diagrams/images are present, explain:
- What the diagram shows
- Key components and their relationships
- Flow of data or processes
- Important annotations or labels

## 5. Implementation Details
How is this implemented? What are the technical specifics?

## 6. Important Notes
Any edge cases, warnings, TODOs, or special considerations

Please be specific and reference actual details from the code and documentation."""
        
        content_blocks.append({
            "type": "text",
            "text": instructions
        })
        
        # 2. Add images with context
        if parsed['images']:
            content_blocks.append({
                "type": "text",
                "text": f"\n\n## PDF Images/Diagrams ({len(parsed['images'])} found)\n\n"
            })
            
            for i, img in enumerate(parsed['images'], 1):
                # Add image description
                content_blocks.append({
                    "type": "text",
                    "text": f"**Image {i}**: From `{img['source']}` (Page {img['page']})\n"
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
                "text": "\n## Code and Documentation Context\n\n"
            })
            
            # Combine and limit text size
            combined_text = "\n\n---\n\n".join(parsed['text_blocks'])
            
            # Truncate if too large
            max_text_length = 80000
            if len(combined_text) > max_text_length:
                combined_text = combined_text[:max_text_length] + "\n\n[... content truncated for length ...]"
            
            content_blocks.append({
                "type": "text",
                "text": combined_text
            })
        
        # 4. Add sources footer
        if parsed['sources']:
            sources_text = "\n\n## Source Files Referenced\n\n" + "\n".join(
                f"- `{source}`" for source in parsed['sources']
            )
            content_blocks.append({
                "type": "text",
                "text": sources_text
            })
        
        return content_blocks
    
    def _format_markdown_summary(
        self,
        summary: str,
        metadata: Dict[str, Any]
    ) -> str:
        """Format summary as markdown with metadata"""
        
        parts = []
        parts.append(f"# LLM Summary: {metadata['query']}\n")
        parts.append(f"**Generated:** {metadata['timestamp']}")
        parts.append(f"**Model:** {metadata['model']}")
        parts.append(f"**Sources:** {len(metadata['sources'])}")
        parts.append(f"**Images:** {metadata['num_images']}")
        parts.append(f"**Tokens:** {metadata['input_tokens']:,} in / {metadata['output_tokens']:,} out\n")
        parts.append("---\n")
        parts.append(summary)
        parts.append("\n\n---\n")
        parts.append("## Source Files\n")
        for source in metadata['sources']:
            parts.append(f"- {source}")
        
        return '\n'.join(parts)
    
    def _format_html_summary(
        self,
        summary: str,
        metadata: Dict[str, Any],
        parsed: Dict[str, Any]
    ) -> str:
        """
        Format summary as beautiful HTML for browser rendering
        Converts markdown to HTML and adds styling
        """
        
        # Convert markdown to HTML (basic conversion)
        html_content = self._markdown_to_html(summary)
        
        # Build complete HTML document
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Summary: {metadata['query']}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 30px 20px;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px 50px;
            position: relative;
        }}
        
        .header::before {{
            content: '🤖';
            position: absolute;
            top: 20px;
            right: 40px;
            font-size: 3em;
            opacity: 0.3;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 15px;
            font-weight: 700;
        }}
        
        .header .query {{
            font-size: 1.3em;
            opacity: 0.95;
            font-weight: 500;
            margin-bottom: 20px;
        }}
        
        .metadata {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-top: 20px;
            padding-top: 20px;
            border-top: 1px solid rgba(255, 255, 255, 0.3);
        }}
        
        .metadata-item {{
            display: flex;
            flex-direction: column;
        }}
        
        .metadata-label {{
            font-size: 0.85em;
            opacity: 0.8;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .metadata-value {{
            font-size: 1.1em;
            font-weight: 600;
            margin-top: 3px;
        }}
        
        .content {{
            padding: 50px;
        }}
        
        .summary {{
            font-size: 1.05em;
            line-height: 1.8;
        }}
        
        .summary h1 {{
            color: #2c3e50;
            font-size: 2em;
            margin: 40px 0 20px 0;
            padding-bottom: 15px;
            border-bottom: 3px solid #667eea;
        }}
        
        .summary h2 {{
            color: #34495e;
            font-size: 1.6em;
            margin: 35px 0 18px 0;
            padding-bottom: 10px;
            border-bottom: 2px solid #95a5a6;
        }}
        
        .summary h3 {{
            color: #7f8c8d;
            font-size: 1.3em;
            margin: 25px 0 12px 0;
        }}
        
        .summary p {{
            margin: 15px 0;
            text-align: justify;
        }}
        
        .summary ul, .summary ol {{
            margin: 15px 0 15px 30px;
        }}
        
        .summary li {{
            margin: 8px 0;
        }}
        
        .summary code {{
            background: #f4f4f4;
            padding: 2px 8px;
            border-radius: 4px;
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            font-size: 0.9em;
            color: #e83e8c;
        }}
        
        .summary pre {{
            background: #2c3e50;
            color: #ecf0f1;
            padding: 25px;
            border-radius: 8px;
            overflow-x: auto;
            font-size: 0.95em;
            line-height: 1.5;
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            border-left: 4px solid #667eea;
            margin: 20px 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        
        .summary pre code {{
            background: none;
            padding: 0;
            color: inherit;
        }}
        
        .summary strong {{
            color: #2c3e50;
            font-weight: 600;
        }}
        
        .summary blockquote {{
            border-left: 4px solid #667eea;
            padding-left: 20px;
            margin: 20px 0;
            color: #555;
            font-style: italic;
            background: #f8f9fa;
            padding: 15px 20px;
            border-radius: 4px;
        }}
        
        .sources {{
            margin-top: 50px;
            padding-top: 30px;
            border-top: 2px solid #e0e0e0;
        }}
        
        .sources h2 {{
            color: #34495e;
            margin-bottom: 20px;
        }}
        
        .source-list {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 10px;
        }}
        
        .source-item {{
            background: #f8f9fa;
            padding: 12px 16px;
            border-radius: 6px;
            border-left: 3px solid #667eea;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 0.9em;
            transition: all 0.2s ease;
        }}
        
        .source-item:hover {{
            background: #e9ecef;
            transform: translateX(5px);
        }}
        
        .footer {{
            background: #f8f9fa;
            padding: 20px 50px;
            text-align: center;
            color: #666;
            font-size: 0.9em;
            border-top: 1px solid #e0e0e0;
        }}
        
        @media print {{
            body {{
                background: white;
                padding: 0;
            }}
            .container {{
                box-shadow: none;
            }}
        }}
        
        @media (max-width: 768px) {{
            .header {{
                padding: 30px 25px;
            }}
            .content {{
                padding: 30px 25px;
            }}
            .header h1 {{
                font-size: 2em;
            }}
            .metadata {{
                grid-template-columns: 1fr 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>AI Analysis Summary</h1>
            <div class="query">Query: "{metadata['query']}"</div>
            <div class="metadata">
                <div class="metadata-item">
                    <span class="metadata-label">Model</span>
                    <span class="metadata-value">{metadata['model'].split('-')[1].title()}</span>
                </div>
                <div class="metadata-item">
                    <span class="metadata-label">Sources</span>
                    <span class="metadata-value">{len(metadata['sources'])} files</span>
                </div>
                <div class="metadata-item">
                    <span class="metadata-label">Images</span>
                    <span class="metadata-value">{metadata['num_images']} diagrams</span>
                </div>
                <div class="metadata-item">
                    <span class="metadata-label">Input Tokens</span>
                    <span class="metadata-value">{metadata['input_tokens']:,}</span>
                </div>
                <div class="metadata-item">
                    <span class="metadata-label">Output Tokens</span>
                    <span class="metadata-value">{metadata['output_tokens']:,}</span>
                </div>
                <div class="metadata-item">
                    <span class="metadata-label">Generated</span>
                    <span class="metadata-value">{datetime.fromisoformat(metadata['timestamp']).strftime('%H:%M')}</span>
                </div>
            </div>
        </div>
        
        <div class="content">
            <div class="summary">
                {html_content}
            </div>
            
            <div class="sources">
                <h2>📁 Source Files Analyzed</h2>
                <div class="source-list">
                    {''.join(f'<div class="source-item">{source}</div>' for source in metadata['sources'])}
                </div>
            </div>
        </div>
        
        <div class="footer">
            Generated by Claude AI • {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
        </div>
    </div>
</body>
</html>
"""
        
        return html
    
    def _markdown_to_html(self, markdown_text: str) -> str:
        """
        Convert markdown to HTML (basic conversion)
        Handles headers, code blocks, lists, bold, italic, code
        """
        
        html_parts = []
        lines = markdown_text.split('\n')
        in_code_block = False
        in_list = False
        code_language = ""
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Code blocks
            if line.startswith('```'):
                if not in_code_block:
                    # Starting code block
                    code_language = line[3:].strip()
                    html_parts.append('<pre><code>')
                    in_code_block = True
                else:
                    # Ending code block
                    html_parts.append('</code></pre>')
                    in_code_block = False
                i += 1
                continue
            
            if in_code_block:
                # Escape HTML in code
                escaped = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                html_parts.append(escaped)
                i += 1
                continue
            
            # Headers
            if line.startswith('### '):
                html_parts.append(f'<h3>{self._process_inline_markdown(line[4:])}</h3>')
            elif line.startswith('## '):
                html_parts.append(f'<h2>{self._process_inline_markdown(line[3:])}</h2>')
            elif line.startswith('# '):
                html_parts.append(f'<h1>{self._process_inline_markdown(line[2:])}</h1>')
            
            # Lists
            elif line.startswith('- ') or line.startswith('* '):
                if not in_list:
                    html_parts.append('<ul>')
                    in_list = True
                html_parts.append(f'<li>{self._process_inline_markdown(line[2:])}</li>')
            
            elif line.startswith(tuple(f'{i}. ' for i in range(10))):
                if not in_list:
                    html_parts.append('<ol>')
                    in_list = True
                content = line.split('. ', 1)[1] if '. ' in line else line
                html_parts.append(f'<li>{self._process_inline_markdown(content)}</li>')
            
            # Blockquotes
            elif line.startswith('> '):
                html_parts.append(f'<blockquote>{self._process_inline_markdown(line[2:])}</blockquote>')
            
            # Horizontal rule
            elif line.strip() in ['---', '***', '___']:
                html_parts.append('<hr>')
            
            # Regular paragraph
            elif line.strip():
                if in_list:
                    html_parts.append('</ul>')
                    in_list = False
                html_parts.append(f'<p>{self._process_inline_markdown(line)}</p>')
            
            # Empty line
            else:
                if in_list:
                    html_parts.append('</ul>')
                    in_list = False
            
            i += 1
        
        if in_list:
            html_parts.append('</ul>')
        
        return '\n'.join(html_parts)
    
    def _process_inline_markdown(self, text: str) -> str:
        """
        Process inline markdown: bold, italic, code, links
        """
        
        # Bold (**text** or __text__)
        text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
        text = re.sub(r'__(.+?)__', r'<strong>\1</strong>', text)
        
        # Italic (*text* or _text_)
        text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
        text = re.sub(r'_(.+?)_', r'<em>\1</em>', text)
        
        # Inline code (`code`)
        text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
        
        # Links [text](url)
        text = re.sub(r'\[(.+?)\]\((.+?)\)', r'<a href="\2">\1</a>', text)
        
        return text
    
    def _parse_for_llm(
        self,
        formatted_context: str,
        contexts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Parse formatted context for LLM consumption"""
        
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
        
        # Extract images from contexts
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
        
        # Extract text blocks
        sections = formatted_context.split('='*70)
        for section in sections:
            section_clean = re.sub(
                r'!\[.*?\]\(data:image/[^;]+;base64,[A-Za-z0-9+/=]+\)',
                '[PDF Image]',
                section
            )
            
            if len(section_clean.strip()) > 100:
                parsed['text_blocks'].append(section_clean.strip())
        
        return parsed
    
    # ... (keep all existing helper methods from previous version)
    # _search_with_universal, _validate_query_terms, _deduplicate_results, etc.
    
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
    def create_html_content(query: str, context: str, include_images: bool = True) -> Optional[str]:
    """
    Enhanced HTML creation for raw search context (non-LLM)
    """
    
    if not context.strip():
        return None
    
    print(f"\n📄 Creating HTML view...")
    
    # Enhanced CSS with beautiful styling
    html = f"""<!DOCTYPE html>
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
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 30px 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px 50px;
            position: relative;
        }}
        
        .header::before {{
            content: '🔍';
            position: absolute;
            top: 20px;
            right: 40px;
            font-size: 3em;
            opacity: 0.3;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 15px;
            font-weight: 700;
        }}
        
        .header .query {{
            font-size: 1.3em;
            opacity: 0.95;
            font-weight: 500;
        }}
        
        .content {{
            padding: 50px;
        }}
        
        h1 {{
            color: #2c3e50;
            font-size: 2em;
            margin: 30px 0 20px 0;
            padding-bottom: 15px;
            border-bottom: 3px solid #667eea;
        }}
        
        h2 {{
            color: #34495e;
            font-size: 1.6em;
            margin: 25px 0 15px 0;
            padding-bottom: 10px;
            border-bottom: 2px solid #95a5a6;
        }}
        
        h3 {{
            color: #7f8c8d;
            font-size: 1.3em;
            margin: 20px 0 10px 0;
        }}
        
        pre {{
            background: #2c3e50;
            color: #ecf0f1;
            padding: 25px;
            border-radius: 8px;
            overflow-x: auto;
            font-size: 0.95em;
            line-height: 1.5;
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            border-left: 4px solid #667eea;
            margin: 20px 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
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
            cursor: zoom-in;
        }}
        
        .image-container img:hover {{
            transform: scale(1.02);
        }}
        
        .image-label {{
            color: #666;
            font-size: 0.9em;
            margin-top: 10px;
            font-style: italic;
        }}
        
        .match-header {{
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 20px 25px;
            border-radius: 8px;
            margin: 30px 0 20px 0;
            border-left: 5px solid #667eea;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        
        .match-title {{
            font-size: 1.3em;
            color: #2c3e50;
            font-weight: 600;
            margin-bottom: 10px;
        }}
        
        .match-meta {{
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
            font-size: 0.9em;
            color: #666;
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
            margin: 15px 0;
            font-size: 0.9em;
        }}
        
        .ranking-signals strong {{
            color: #856404;
        }}
        
        .ranking-signals ul {{
            margin: 5px 0 0 20px;
            color: #856404;
        }}
        
        hr {{
            border: none;
            border-top: 2px solid #e0e0e0;
            margin: 30px 0;
        }}
        
        .footer {{
            background: #f8f9fa;
            padding: 20px 50px;
            text-align: center;
            color: #666;
            font-size: 0.9em;
            border-top: 1px solid #e0e0e0;
        }}
        
        code {{
            background: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 0.9em;
            color: #e83e8c;
        }}
        
        strong {{
            color: #2c3e50;
        }}
        
        @media print {{
            body {{
                background: white;
                padding: 0;
            }}
            .container {{
                box-shadow: none;
            }}
        }}
        
        @media (max-width: 768px) {{
            .header {{
                padding: 30px 25px;
            }}
            .content {{
                padding: 30px 25px;
            }}
            .header h1 {{
                font-size: 2em;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Search Results</h1>
            <div class="query">Query: "{query}"</div>
        </div>
        
        <div class="content">
"""
    
    lines = context.split('\n')
    in_code = False
    current_match = None
    
    for line in lines:
        # Handle code blocks
        if line.startswith('```'):
            html += "</pre>\n" if in_code else "<pre>\n"
            in_code = not in_code
            continue
        
        if in_code:
            # Escape HTML in code
            escaped = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            html += f"{escaped}\n"
            continue
        
        # Handle images
        if '![Page Image](data:image/' in line and include_images:
            match = re.search(r'!\[.*?\]\(data:image/(png|jpeg);base64,([^)]+)\)', line)
            if match:
                img_type = match.group(1)
                img_data = match.group(2)
                html += f'<div class="image-container">\n'
                html += f'<img src="data:image/{img_type};base64,{img_data}" alt="PDF Page Image" />\n'
                html += f'<div class="image-label">Click to zoom</div>\n'
                html += '</div>\n'
            continue
        
        # Handle match headers
        if line.startswith('## Match'):
            if current_match:
                html += '</div>\n'  # Close previous match
            html += '<div class="match-header">\n'
            html += f'<div class="match-title">{line[3:]}</div>\n'
            html += '<div class="match-meta">\n'
            current_match = True
            continue
        
        # Handle score in match header
        if line.startswith('Score:') and current_match:
            score_value = line.split(':')[1].strip()
            html += f'<span class="score">Score: {score_value}</span>\n'
            continue
        
        # Handle query matches
        if line.startswith('Query matches:'):
            html += f'<span>{line}</span>\n'
            continue
        
        # Handle ranking signals
        if line.startswith('Ranking:'):
            html += '</div>\n'  # Close match-meta
            html += '</div>\n'  # Close match-header
            html += '<div class="ranking-signals">\n'
            html += '<strong>Ranking Signals:</strong>\n<ul>\n'
            current_match = False
            continue
        
        if line.strip().startswith('•'):
            html += f'<li>{line.strip()[1:].strip()}</li>\n'
            continue
        
        # Close ranking signals at separator
        if line.strip() == '='*70:
            if not current_match:
                html += '</ul></div>\n'
            html += '<hr>\n'
            continue
        
        # Headers
        if line.startswith('# '):
            html += f"<h1>{line[2:]}</h1>\n"
        elif line.startswith('### '):
            html += f"<h3>{line[4:]}</h3>\n"
        elif line.strip().startswith('—'):
            html += f"<p>{line}</p>\n"
        elif line.strip() and not line.startswith('Type:') and not line.startswith('Location:'):
            html += f"<p>{line}</p>\n"
    
    if in_code:
        html += "</pre>\n"
    
    html += """
        </div>
        
        <div class="footer">
            Generated by Search Index • """ + datetime.now().strftime('%B %d, %Y at %I:%M %p') + """
        </div>
    </div>
    
    <script>
        // Add click-to-zoom for images
        document.querySelectorAll('.image-container img').forEach(img => {
            img.addEventListener('click', function() {
                if (this.style.maxWidth === 'none') {
                    this.style.maxWidth = '100%';
                    this.style.maxHeight = '1200px';
                    this.style.cursor = 'zoom-in';
                } else {
                    this.style.maxWidth = 'none';
                    this.style.maxHeight = 'none';
                    this.style.cursor = 'zoom-out';
                }
            });
        });
    </script>
</body>
</html>
"""
    
    return html


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
    
    When use_llm=True, saves both llm_summary.md and llm_summary.html
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
        print("Usage: python3 search_index.py --search <term> [--llm]")
        sys.exit(0)
    
    parser = argparse.ArgumentParser(description='Context Extractor v3.6 - Async LLM')
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
    print("CONTEXT EXTRACTOR v3.6 - ASYNC LLM + HTML")
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
    
    if args.llm:
        print("\n📂 Output files:")
        print("  - llm_summary.md (markdown)")
        print("  - llm_summary.html (browser-ready)")
        print("\n💡 Open llm_summary.html in your browser!")
    else:
        # Save text output
        output_file = "extracted_context.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(context)
        print(f"\n✅ Saved: {output_file} ({len(context):,} chars)")
        
        # Generate and save HTML
        html_content = create_html_content(args.search, context, include_images=not args.no_images)
        if html_content:
            with open("extracted_context.html", 'w', encoding='utf-8') as f:
                f.write(html_content)
            print(f"✅ Saved HTML: extracted_context.html")
            print("\n💡 Open extracted_context.html in your browser!")
        else:
            print("⚠ No HTML generated (no content)")
