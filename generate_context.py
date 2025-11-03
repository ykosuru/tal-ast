"""
Extended Context Extractor for LLM
Extracts 200 lines before/after each search match from original source files
Constructs optimized input context for LLM code generation

Usage:
    extractor = ExtendedContextExtractor(
        universal_index="./universal_index",
        hybrid_index="./hybrid_index"
    )
    
    context = extractor.extract_and_format(
        query="payment drawdown processing",
        lines_before=200,
        lines_after=200,
        max_matches=10
    )
    
    # Send context to LLM
    llm.generate(context)
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from unified_search_v2 import UnifiedCodeSearch, SearchResult
import re


@dataclass
class ExtendedContext:
    """Extended context with file location and surrounding lines"""
    source_file: str
    source_path: Path
    file_type: str
    language: str
    
    # Match location
    match_line_start: int
    match_line_end: int
    match_text: str
    
    # Extended context
    lines_before: List[str]
    lines_after: List[str]
    
    # Metadata
    score: float
    capabilities: List[str]
    keywords: List[str]
    
    def get_full_context(self) -> str:
        """Get full context as single string"""
        lines = []
        
        # Lines before
        start_line = self.match_line_start - len(self.lines_before)
        for i, line in enumerate(self.lines_before):
            lines.append(f"{start_line + i:4d} | {line}")
        
        # Match lines (highlighted)
        lines.append(f"{'='*70}")
        lines.append(f">>> MATCH (score: {self.score:.3f})")
        lines.append(f"{'='*70}")
        for i, line in enumerate(self.match_text.split('\n')[:20]):  # Limit match display
            lines.append(f"{self.match_line_start + i:4d} | {line}")
        lines.append(f"{'='*70}")
        
        # Lines after
        for i, line in enumerate(self.lines_after):
            lines.append(f"{self.match_line_end + i + 1:4d} | {line}")
        
        return '\n'.join(lines)
    
    def get_summary(self) -> str:
        """Get summary of this context"""
        return (
            f"{self.source_file} (lines {self.match_line_start}-{self.match_line_end})\n"
            f"  Type: {self.file_type}, Language: {self.language}\n"
            f"  Score: {self.score:.3f}\n"
            f"  Capabilities: {', '.join(self.capabilities[:2])}\n"
        )


class SourceFileReader:
    """Read source files and extract line-based context"""
    
    def __init__(self):
        self.file_cache = {}  # Cache file contents
    
    def read_file_lines(self, file_path: Path) -> List[str]:
        """Read file and return lines (with caching)"""
        file_path_str = str(file_path)
        
        if file_path_str in self.file_cache:
            return self.file_cache[file_path_str]
        
        try:
            # Try different encodings
            for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        lines = f.readlines()
                    
                    # Remove trailing newlines but keep content
                    lines = [line.rstrip('\n\r') for line in lines]
                    
                    self.file_cache[file_path_str] = lines
                    return lines
                except UnicodeDecodeError:
                    continue
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return []
        
        return []
    
    def find_match_location(
        self,
        lines: List[str],
        match_text: str,
        chunk_index: int = 0
    ) -> Tuple[int, int]:
        """
        Find where match_text appears in the file
        
        Returns:
            (start_line, end_line) - 0-indexed
        """
        # Clean match text
        match_lines = [line.strip() for line in match_text.split('\n') if line.strip()]
        
        if not match_lines:
            return (0, 0)
        
        # Search for first few lines of match
        search_lines = match_lines[:min(3, len(match_lines))]
        
        for i in range(len(lines) - len(search_lines) + 1):
            # Check if lines match (fuzzy)
            match_count = 0
            for j, search_line in enumerate(search_lines):
                file_line = lines[i + j].strip()
                # Fuzzy match: check if search line is substring or very similar
                if search_line in file_line or file_line in search_line:
                    match_count += 1
                elif self._line_similarity(search_line, file_line) > 0.8:
                    match_count += 1
            
            if match_count >= len(search_lines) * 0.7:  # 70% match
                start_line = i
                end_line = min(i + len(match_lines), len(lines))
                return (start_line, end_line)
        
        # If not found, estimate based on chunk_index
        lines_per_chunk = 100  # Rough estimate
        estimated_start = chunk_index * lines_per_chunk
        estimated_end = estimated_start + len(match_lines)
        
        return (estimated_start, min(estimated_end, len(lines)))
    
    def _line_similarity(self, line1: str, line2: str) -> float:
        """Compute similarity between two lines"""
        words1 = set(line1.lower().split())
        words2 = set(line2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def extract_context(
        self,
        file_path: Path,
        match_text: str,
        chunk_index: int,
        lines_before: int = 200,
        lines_after: int = 200
    ) -> Tuple[int, int, List[str], List[str]]:
        """
        Extract context around a match
        
        Returns:
            (match_start, match_end, lines_before_list, lines_after_list)
        """
        lines = self.read_file_lines(file_path)
        
        if not lines:
            return (0, 0, [], [])
        
        # Find match location
        match_start, match_end = self.find_match_location(lines, match_text, chunk_index)
        
        # Extract before context
        before_start = max(0, match_start - lines_before)
        before_lines = lines[before_start:match_start]
        
        # Extract after context
        after_end = min(len(lines), match_end + lines_after)
        after_lines = lines[match_end:after_end]
        
        return (match_start, match_end, before_lines, after_lines)


class ExtendedContextExtractor:
    """
    Extract extended context (200 lines before/after) for LLM consumption
    
    Workflow:
    1. Search both indexes
    2. Get top matches
    3. Go back to source files
    4. Extract 200 lines before/after each match
    5. Format for LLM
    """
    
    def __init__(
        self,
        universal_index: str = "./universal_index",
        hybrid_index: str = "./hybrid_index"
    ):
        # Initialize search
        self.search = UnifiedCodeSearch(
            universal_index_path=universal_index,
            hybrid_index_path=hybrid_index
        )
        
        # Initialize file reader
        self.file_reader = SourceFileReader()
        
        print("✓ Extended context extractor initialized")
    
    def extract_contexts(
        self,
        query: str,
        max_matches: int = 10,
        lines_before: int = 200,
        lines_after: int = 200,
        file_type_filter: Optional[str] = None,
        verbose: bool = False
    ) -> List[ExtendedContext]:
        """
        Extract extended contexts for all matches
        
        Args:
            query: Search query
            max_matches: Max number of matches to extract
            lines_before: Lines to extract before match
            lines_after: Lines to extract after match
            file_type_filter: Filter by file type ('code', 'pdf', etc.)
            verbose: Show progress
        
        Returns:
            List of ExtendedContext objects
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"Extracting Extended Context for: {query}")
            print(f"{'='*70}")
        
        # Step 1: Search indexes
        if verbose:
            print(f"\n[1/3] Searching indexes...")
        
        results = self.search.search(
            query=query,
            top_k=max_matches,
            file_type_filter=file_type_filter,
            verbose=False
        )
        
        if verbose:
            print(f"  Found {len(results)} matches")
        
        # Step 2: Extract contexts from source files
        if verbose:
            print(f"\n[2/3] Extracting contexts from source files...")
        
        contexts = []
        
        for i, result in enumerate(results, 1):
            if verbose:
                print(f"  [{i}/{len(results)}] {result.source_file}")
            
            # Find source file path
            source_path = self._find_source_file(result)
            
            if not source_path or not source_path.exists():
                if verbose:
                    print(f"    ⚠ Source file not found: {result.source_file}")
                continue
            
            # Extract context
            match_start, match_end, before_lines, after_lines = self.file_reader.extract_context(
                file_path=source_path,
                match_text=result.text,
                chunk_index=result.chunk_index,
                lines_before=lines_before,
                lines_after=lines_after
            )
            
            # Create ExtendedContext
            context = ExtendedContext(
                source_file=result.source_file,
                source_path=source_path,
                file_type=result.file_type,
                language=result.language,
                match_line_start=match_start,
                match_line_end=match_end,
                match_text=result.text,
                lines_before=before_lines,
                lines_after=after_lines,
                score=result.score,
                capabilities=result.capabilities,
                keywords=result.keywords
            )
            
            contexts.append(context)
            
            if verbose:
                print(f"    ✓ Lines {match_start}-{match_end} (+{len(before_lines)} before, +{len(after_lines)} after)")
        
        if verbose:
            print(f"\n[3/3] Extracted {len(contexts)} contexts")
        
        return contexts
    
    def _find_source_file(self, result: SearchResult) -> Optional[Path]:
        """Find the actual source file path"""
        # Try to get from metadata (stored during indexing)
        # First try the search system's metadata
        try:
            # Check universal indexer
            if result.source_indexer == 'universal':
                index_path = Path(self.search.universal_searcher.index_path)
            else:
                index_path = Path(self.search.hybrid_searcher.index_path)
            
            # Load metadata to find source path
            import pickle
            with open(index_path / "metadata.pkl", 'rb') as f:
                metadata_store = pickle.load(f)
            
            # Find matching metadata
            for metadata in metadata_store:
                if metadata.get('source_file') == result.source_file:
                    source_path = metadata.get('source_path')
                    if source_path:
                        return Path(source_path)
                    break
        except Exception as e:
            pass
        
        # Fallback: search common locations
        search_paths = [
            Path("./your_docs") / result.source_file,
            Path("./your_code") / result.source_file,
            Path(result.source_file),
        ]
        
        for path in search_paths:
            if path.exists():
                return path
        
        return None
    
    def format_for_llm(
        self,
        contexts: List[ExtendedContext],
        query: str,
        include_line_numbers: bool = True,
        group_by_file: bool = True
    ) -> str:
        """
        Format extracted contexts for LLM consumption
        
        Args:
            contexts: List of ExtendedContext objects
            query: Original query
            include_line_numbers: Include line numbers in output
            group_by_file: Group contexts from same file together
        
        Returns:
            Formatted string ready for LLM
        """
        if not contexts:
            return "No contexts found."
        
        output_parts = []
        
        # Header
        output_parts.append(f"# Code Context for Query: {query}\n")
        output_parts.append(f"Found {len(contexts)} relevant code sections\n")
        
        # Group by file if requested
        if group_by_file:
            file_groups = {}
            for ctx in contexts:
                if ctx.source_file not in file_groups:
                    file_groups[ctx.source_file] = []
                file_groups[ctx.source_file].append(ctx)
            
            # Process each file
            for file_name, file_contexts in file_groups.items():
                output_parts.append(f"\n{'='*70}")
                output_parts.append(f"## File: {file_name}")
                output_parts.append(f"{'='*70}\n")
                
                for ctx in file_contexts:
                    output_parts.append(self._format_single_context(ctx, include_line_numbers))
        else:
            # Output contexts in score order
            for i, ctx in enumerate(contexts, 1):
                output_parts.append(f"\n{'='*70}")
                output_parts.append(f"## Match {i}: {ctx.source_file}")
                output_parts.append(f"Score: {ctx.score:.3f}, Type: {ctx.file_type}")
                output_parts.append(f"{'='*70}\n")
                
                output_parts.append(self._format_single_context(ctx, include_line_numbers))
        
        # Summary
        output_parts.append(f"\n{'='*70}")
        output_parts.append("## Summary")
        output_parts.append(f"{'='*70}")
        output_parts.append(f"Total contexts: {len(contexts)}")
        output_parts.append(f"Unique files: {len(set(ctx.source_file for ctx in contexts))}")
        output_parts.append(f"Total lines: {sum(len(ctx.lines_before) + len(ctx.lines_after) for ctx in contexts)}")
        
        return '\n'.join(output_parts)
    
    def _format_single_context(
        self,
        ctx: ExtendedContext,
        include_line_numbers: bool
    ) -> str:
        """Format a single context"""
        parts = []
        
        # Context info
        parts.append(f"Location: Lines {ctx.match_line_start}-{ctx.match_line_end}")
        if ctx.capabilities:
            parts.append(f"Capabilities: {', '.join(ctx.capabilities[:3])}")
        parts.append("")
        
        # Code block
        parts.append(f"```{ctx.language or 'text'}")
        
        if include_line_numbers:
            # Lines before
            start_line = ctx.match_line_start - len(ctx.lines_before)
            for i, line in enumerate(ctx.lines_before[-50:]):  # Last 50 lines before
                parts.append(f"{start_line + len(ctx.lines_before) - 50 + i:4d} | {line}")
            
            # Match (highlighted)
            parts.append("")
            parts.append(">>> RELEVANT MATCH <<<")
            parts.append("")
            for i, line in enumerate(ctx.match_text.split('\n')[:30]):  # First 30 lines
                parts.append(f"{ctx.match_line_start + i:4d} | {line}")
            parts.append("")
            parts.append(">>> END MATCH <<<")
            parts.append("")
            
            # Lines after
            for i, line in enumerate(ctx.lines_after[:50]):  # First 50 lines after
                parts.append(f"{ctx.match_line_end + i + 1:4d} | {line}")
        else:
            # Without line numbers
            parts.append('\n'.join(ctx.lines_before[-50:]))
            parts.append("")
            parts.append(">>> RELEVANT MATCH <<<")
            parts.append(ctx.match_text[:1000])  # Limit match size
            parts.append(">>> END MATCH <<<")
            parts.append("")
            parts.append('\n'.join(ctx.lines_after[:50]))
        
        parts.append("```")
        parts.append("")
        
        return '\n'.join(parts)
    
    def extract_and_format(
        self,
        query: str,
        max_matches: int = 10,
        lines_before: int = 200,
        lines_after: int = 200,
        file_type_filter: Optional[str] = None,
        verbose: bool = True
    ) -> str:
        """
        One-shot: Extract contexts and format for LLM
        
        Returns:
            Formatted string ready to send to LLM
        """
        # Extract contexts
        contexts = self.extract_contexts(
            query=query,
            max_matches=max_matches,
            lines_before=lines_before,
            lines_after=lines_after,
            file_type_filter=file_type_filter,
            verbose=verbose
        )
        
        # Format for LLM
        formatted = self.format_for_llm(contexts, query)
        
        return formatted


# ============================================================================
# Example Usage
# ============================================================================

def example_basic_usage():
    """Example 1: Basic usage"""
    
    print("="*70)
    print("EXAMPLE 1: Basic Extended Context Extraction")
    print("="*70)
    
    # Initialize
    extractor = ExtendedContextExtractor(
        universal_index="./universal_index",
        hybrid_index="./hybrid_index"
    )
    
    # Extract and format
    context = extractor.extract_and_format(
        query="payment drawdown processing",
        max_matches=5,
        lines_before=200,
        lines_after=200,
        verbose=True
    )
    
    # Print result
    print("\n" + "="*70)
    print("FORMATTED CONTEXT (Ready for LLM)")
    print("="*70)
    print(context)
    
    # Save to file
    with open("llm_context.txt", 'w') as f:
        f.write(context)
    
    print("\n✓ Context saved to llm_context.txt")
    print(f"  Length: {len(context)} chars")


def example_send_to_llm():
    """Example 2: Extract and send to LLM"""
    
    print("="*70)
    print("EXAMPLE 2: Extract Context and Send to LLM")
    print("="*70)
    
    # Initialize
    extractor = ExtendedContextExtractor()
    
    # Extract context
    context = extractor.extract_and_format(
        query="OFAC screening validation",
        max_matches=3,
        lines_before=100,
        lines_after=100
    )
    
    # Build LLM prompt
    llm_prompt = f"""
Based on this codebase context:

{context}

Task: Implement a new function that performs OFAC screening validation with the following requirements:
1. Use existing patterns from the codebase
2. Handle both party and transaction screening
3. Include error handling
4. Add logging for audit trail
5. Follow the coding style shown in the examples

Generate the complete implementation in TAL.
"""
    
    print("\n" + "="*70)
    print("LLM PROMPT (Ready to Send)")
    print("="*70)
    print(llm_prompt[:500] + "...")
    
    # Send to your LLM
    # response = your_llm.generate(llm_prompt)
    # print(response)


def example_filter_by_file_type():
    """Example 3: Extract only from code files"""
    
    print("="*70)
    print("EXAMPLE 3: Extract from Code Files Only")
    print("="*70)
    
    extractor = ExtendedContextExtractor()
    
    # Extract only from code
    context = extractor.extract_and_format(
        query="wire transfer validation",
        max_matches=10,
        lines_before=150,
        lines_after=150,
        file_type_filter="code",  # Only code files
        verbose=True
    )
    
    print(f"\n✓ Context length: {len(context)} chars")


def example_custom_formatting():
    """Example 4: Custom formatting"""
    
    print("="*70)
    print("EXAMPLE 4: Custom Context Formatting")
    print("="*70)
    
    extractor = ExtendedContextExtractor()
    
    # Extract contexts
    contexts = extractor.extract_contexts(
        query="ACE repair codes",
        max_matches=5,
        lines_before=100,
        lines_after=100,
        verbose=True
    )
    
    # Custom formatting
    print("\n" + "="*70)
    print("CUSTOM FORMAT")
    print("="*70)
    
    for i, ctx in enumerate(contexts, 1):
        print(f"\n[{i}] {ctx.source_file}")
        print(f"    Lines: {ctx.match_line_start}-{ctx.match_line_end}")
        print(f"    Score: {ctx.score:.3f}")
        print(f"    Context: {len(ctx.lines_before)} lines before, {len(ctx.lines_after)} lines after")
        print(f"    Snippet: {ctx.match_text[:100]}...")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--example":
        example_num = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        
        examples = {
            1: example_basic_usage,
            2: example_send_to_llm,
            3: example_filter_by_file_type,
            4: example_custom_formatting
        }
        
        if example_num in examples:
            examples[example_num]()
        else:
            print(f"Example {example_num} not found! Available: 1-4")
    else:
        # Run basic example
        example_basic_usage()
