"""
LLM-Powered Document Reprocessor v1.0
Takes search results and uses LLM to extract query-relevant content

Architecture:
1. Index search finds candidate documents (BM25/LSI)
2. Retrieve full document content
3. LLM analyzes document and extracts relevant sections
4. Return refined, query-focused results

Usage:
    from llm_document_reprocessor import LLMReprocessor
    
    # Initialize
    reprocessor = LLMReprocessor(
        universal_index="./universal_index",
        hybrid_index="./hybrid_index"
    )
    
    # Wire up your LLM
    def my_llm_function(prompt, context):
        # Your LLM integration here
        return llm_response
    
    reprocessor.set_llm_function(my_llm_function)
    
    # Search with LLM refinement
    results = reprocessor.search_and_refine(
        "How does payment validation work?",
        max_results=3
    )
"""

import pickle
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from collections import defaultdict


class LLMReprocessor:
    """
    Uses LLM to reprocess search results and extract query-relevant content
    """
    
    def __init__(
        self,
        universal_index: str = "./universal_index",
        hybrid_index: str = "./hybrid_index"
    ):
        self.universal_index = Path(universal_index)
        self.hybrid_index = Path(hybrid_index)
        self.llm_function: Optional[Callable] = None
        
        # Load document store for full text retrieval
        self.full_documents = {}
        self._load_full_documents()
        
        print("✓ LLM Document Reprocessor initialized")
    
    def set_llm_function(self, llm_func: Callable[[str, str], str]):
        """
        Set the LLM function to use for document analysis
        
        Args:
            llm_func: Function that takes (prompt, context) and returns LLM response
        
        Example:
            def my_llm(prompt, context):
                return openai.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": context}
                    ]
                ).choices[0].message.content
            
            reprocessor.set_llm_function(my_llm)
        """
        self.llm_function = llm_func
        print("✓ LLM function registered")
    
    def _load_full_documents(self):
        """Load full document texts for reprocessing"""
        # Try to load from universal index
        if self.universal_index.exists():
            try:
                with open(self.universal_index / "metadata.pkl", 'rb') as f:
                    metadata = pickle.load(f)
                
                with open(self.universal_index / "documents.pkl", 'rb') as f:
                    documents = pickle.load(f)
                
                # Group chunks by source file to reconstruct full documents
                file_chunks = defaultdict(list)
                for i, meta in enumerate(metadata):
                    source_file = meta.get('source_file', '')
                    chunk_index = meta.get('chunk_index', 0)
                    file_chunks[source_file].append((chunk_index, documents[i]))
                
                # Reconstruct full documents
                for source_file, chunks in file_chunks.items():
                    # Sort by chunk index
                    chunks.sort(key=lambda x: x[0])
                    # Join chunks (remove duplicates from overlap)
                    full_text = ' '.join([chunk[1] for chunk in chunks])
                    self.full_documents[source_file] = full_text
                
                print(f"  Loaded {len(self.full_documents)} full documents from universal index")
            except Exception as e:
                print(f"  ⚠ Could not load from universal index: {e}")
        
        # Try to load from hybrid index
        if self.hybrid_index.exists():
            try:
                with open(self.hybrid_index / "metadata.pkl", 'rb') as f:
                    metadata = pickle.load(f)
                
                with open(self.hybrid_index / "documents.pkl", 'rb') as f:
                    documents = pickle.load(f)
                
                # Add any new documents
                file_chunks = defaultdict(list)
                for i, meta in enumerate(metadata):
                    source_file = meta.get('source_file', '')
                    if source_file not in self.full_documents:
                        file_chunks[source_file].append(documents[i])
                
                for source_file, chunks in file_chunks.items():
                    self.full_documents[source_file] = ' '.join(chunks)
                
                print(f"  Total full documents: {len(self.full_documents)}")
            except Exception as e:
                print(f"  ⚠ Could not load from hybrid index: {e}")
    
    def search_and_refine(
        self,
        query: str,
        max_results: int = 3,
        include_full_doc: bool = False,
        max_doc_length: int = 8000
    ) -> List[Dict[str, Any]]:
        """
        Search indexes and use LLM to refine results
        
        Args:
            query: User's search query
            max_results: Number of documents to process with LLM
            include_full_doc: Include full document in results (for reference)
            max_doc_length: Maximum document length to send to LLM (in words)
        
        Returns:
            List of refined results with LLM-extracted relevant content
        """
        if not self.llm_function:
            raise ValueError("LLM function not set. Call set_llm_function() first.")
        
        print(f"\n{'='*70}")
        print(f"LLM-Powered Search: {query}")
        print(f"{'='*70}")
        
        # Step 1: Initial search to find candidate documents
        print("\n[1/3] Searching indexes for candidate documents...")
        candidates = self._search_indexes(query, max_results * 2)
        
        if not candidates:
            return []
        
        print(f"  Found {len(candidates)} candidate documents")
        
        # Step 2: Get full document content
        print("\n[2/3] Retrieving full document content...")
        documents_to_process = []
        
        for candidate in candidates[:max_results]:
            source_file = candidate['source_file']
            if source_file in self.full_documents:
                full_text = self.full_documents[source_file]
                
                # Truncate if too long
                words = full_text.split()
                if len(words) > max_doc_length:
                    full_text = ' '.join(words[:max_doc_length]) + "\n...[truncated]"
                
                documents_to_process.append({
                    'source_file': source_file,
                    'file_type': candidate.get('file_type', 'unknown'),
                    'full_text': full_text,
                    'initial_score': candidate.get('score', 0)
                })
                print(f"  ✓ {source_file} ({len(words)} words)")
            else:
                print(f"  ⚠ {source_file} - full text not available")
        
        # Step 3: Use LLM to extract relevant content
        print("\n[3/3] Using LLM to extract query-relevant content...")
        refined_results = []
        
        for doc in documents_to_process:
            print(f"  Processing: {doc['source_file']}")
            
            # Create prompt for LLM
            prompt = self._create_extraction_prompt(query)
            context = self._format_document_for_llm(doc)
            
            try:
                # Call LLM
                llm_response = self.llm_function(prompt, context)
                
                # Parse LLM response
                extracted = self._parse_llm_response(llm_response)
                
                result = {
                    'source_file': doc['source_file'],
                    'file_type': doc['file_type'],
                    'initial_score': doc['initial_score'],
                    'llm_extracted_content': extracted.get('relevant_sections', []),
                    'llm_summary': extracted.get('summary', ''),
                    'llm_key_points': extracted.get('key_points', []),
                    'llm_confidence': extracted.get('confidence', 'medium'),
                    'query': query
                }
                
                if include_full_doc:
                    result['full_document'] = doc['full_text']
                
                refined_results.append(result)
                print(f"    ✓ Extracted {len(extracted.get('relevant_sections', []))} relevant sections")
                
            except Exception as e:
                print(f"    ⚠ LLM processing failed: {e}")
                # Fallback: include original content
                refined_results.append({
                    'source_file': doc['source_file'],
                    'file_type': doc['file_type'],
                    'initial_score': doc['initial_score'],
                    'error': str(e),
                    'fallback_text': doc['full_text'][:1000]
                })
        
        print(f"\n✓ Processed {len(refined_results)} documents")
        return refined_results
    
    def _create_extraction_prompt(self, query: str) -> str:
        """Create prompt for LLM to extract relevant content"""
        prompt = f"""You are an expert document analyzer. Your task is to analyze a document and extract ONLY the content that is directly relevant to answering the user's query.

USER QUERY: "{query}"

INSTRUCTIONS:
1. Read the document carefully
2. Identify ALL sections, paragraphs, or code snippets that directly relate to the query
3. Extract these relevant sections verbatim (copy exact text)
4. For each section, provide:
   - The extracted text (exact copy from document)
   - Why it's relevant (brief explanation)
   - Location hint (what part of doc: beginning/middle/end, or line numbers if code)

5. Provide a summary of how the document answers the query
6. List 3-5 key points that directly address the query
7. Rate your confidence: high/medium/low based on how well the document answers the query

OUTPUT FORMAT (JSON):
{{
  "relevant_sections": [
    {{
      "text": "exact text from document",
      "relevance": "why this section matters",
      "location": "where in document"
    }}
  ],
  "summary": "brief summary of how document answers query",
  "key_points": ["point 1", "point 2", "point 3"],
  "confidence": "high|medium|low"
}}

IMPORTANT:
- Extract EXACT text, don't paraphrase
- Include enough context to be understandable
- Focus on sections that DIRECTLY answer the query
- If document doesn't answer query well, say so (low confidence)
- For code: include function names, key logic, important comments
"""
        return prompt
    
    def _format_document_for_llm(self, doc: Dict[str, Any]) -> str:
        """Format document for LLM processing"""
        context = f"""DOCUMENT: {doc['source_file']}
TYPE: {doc['file_type']}

CONTENT:
{doc['full_text']}

---
END OF DOCUMENT
"""
        return context
    
    def _parse_llm_response(self, llm_response: str) -> Dict[str, Any]:
        """Parse LLM response (expects JSON format)"""
        import json
        
        # Try to extract JSON from response
        # Handle cases where LLM wraps JSON in markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', llm_response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON object
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = llm_response
        
        try:
            parsed = json.loads(json_str)
            return parsed
        except json.JSONDecodeError:
            # Fallback: return raw response
            return {
                'relevant_sections': [{'text': llm_response, 'relevance': 'LLM response', 'location': 'N/A'}],
                'summary': 'Could not parse LLM response',
                'key_points': [],
                'confidence': 'low'
            }
    
    def _search_indexes(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search both indexes for candidate documents"""
        from universal_indexer_v2 import UniversalFileSearcher
        try:
            from hybrid_indexer_v2_fixed import HybridSearcher
        except ImportError:
            try:
                from hybrid_indexer_v2 import HybridSearcher
            except ImportError:
                HybridSearcher = None
        
        results = []
        seen_files = set()
        
        # Search universal index
        if self.universal_index.exists():
            try:
                searcher = UniversalFileSearcher(str(self.universal_index))
                uni_results = searcher.search(query, top_k=max_results, verbose=False)
                
                for r in uni_results:
                    source_file = r['source_file']
                    if source_file not in seen_files:
                        results.append({
                            'source_file': source_file,
                            'file_type': r['file_type'],
                            'score': r['score'],
                            'index': 'universal'
                        })
                        seen_files.add(source_file)
            except Exception as e:
                print(f"  ⚠ Universal index error: {e}")
        
        # Search hybrid index
        if self.hybrid_index.exists() and HybridSearcher:
            try:
                searcher = HybridSearcher(str(self.hybrid_index))
                hyb_results = searcher.search(query, top_k=max_results, verbose=False)
                
                for r in hyb_results:
                    source_file = r['source_file']
                    if source_file not in seen_files:
                        results.append({
                            'source_file': source_file,
                            'file_type': r['file_type'],
                            'score': r.get('combined_score', r.get('bm25_score', 0)),
                            'index': 'hybrid'
                        })
                        seen_files.add(source_file)
            except Exception as e:
                print(f"  ⚠ Hybrid index error: {e}")
        
        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)
        return results
    
    def format_results_for_display(self, results: List[Dict[str, Any]]) -> str:
        """Format refined results for display"""
        if not results:
            return "No results found."
        
        output = []
        output.append("="*70)
        output.append("LLM-REFINED SEARCH RESULTS")
        output.append("="*70)
        
        for i, result in enumerate(results, 1):
            output.append(f"\n{'='*70}")
            output.append(f"RESULT {i}: {result['source_file']}")
            output.append(f"Type: {result['file_type']}")
            output.append(f"Confidence: {result.get('llm_confidence', 'N/A').upper()}")
            output.append(f"{'='*70}")
            
            # Summary
            if result.get('llm_summary'):
                output.append(f"\n📋 SUMMARY:")
                output.append(f"{result['llm_summary']}")
            
            # Key points
            if result.get('llm_key_points'):
                output.append(f"\n🔑 KEY POINTS:")
                for point in result['llm_key_points']:
                    output.append(f"  • {point}")
            
            # Relevant sections
            if result.get('llm_extracted_content'):
                output.append(f"\n📄 RELEVANT SECTIONS:")
                for j, section in enumerate(result['llm_extracted_content'], 1):
                    output.append(f"\n  [{j}] {section.get('location', 'Unknown location')}")
                    output.append(f"  Relevance: {section.get('relevance', 'N/A')}")
                    output.append(f"\n  Content:")
                    # Indent the content
                    content_lines = section.get('text', '').split('\n')
                    for line in content_lines[:20]:  # Limit to 20 lines per section
                        output.append(f"    {line}")
                    if len(content_lines) > 20:
                        output.append(f"    ... ({len(content_lines) - 20} more lines)")
            
            # Error handling
            if result.get('error'):
                output.append(f"\n⚠️  ERROR: {result['error']}")
                if result.get('fallback_text'):
                    output.append(f"\nFallback content (first 1000 chars):")
                    output.append(result['fallback_text'])
        
        output.append(f"\n{'='*70}")
        output.append(f"Total results: {len(results)}")
        output.append(f"{'='*70}")
        
        return '\n'.join(output)


# ============================================================================
# Stub LLM Function (for testing/demo)
# ============================================================================

def call_llm(prompt: str, context: str) -> str:
    """
    STUB: Replace this with your actual LLM integration
    
    Args:
        prompt: System prompt with instructions
        context: Document content to analyze
    
    Returns:
        LLM response (should be JSON formatted)
    
    INTEGRATION EXAMPLES:
    
    # OpenAI:
    def call_llm(prompt, context):
        import openai
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": context}
            ]
        )
        return response.choices[0].message.content
    
    # Anthropic Claude:
    def call_llm(prompt, context):
        import anthropic
        client = anthropic.Anthropic(api_key="...")
        message = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=4096,
            messages=[
                {"role": "user", "content": f"{prompt}\\n\\n{context}"}
            ]
        )
        return message.content[0].text
    
    # Local LLM (Ollama):
    def call_llm(prompt, context):
        import requests
        response = requests.post('http://localhost:11434/api/generate', 
            json={
                'model': 'llama2',
                'prompt': f"{prompt}\\n\\n{context}",
                'stream': False
            })
        return response.json()['response']
    """
    
    # STUB IMPLEMENTATION - Returns mock response
    print("  ⚠️  Using STUB LLM function - replace with real LLM!")
    
    mock_response = {
        "relevant_sections": [
            {
                "text": "This is a mock extracted section from the document.",
                "relevance": "This section is relevant because...",
                "location": "Beginning of document"
            }
        ],
        "summary": "Mock summary: The document discusses the topic but needs real LLM analysis.",
        "key_points": [
            "Mock point 1",
            "Mock point 2",
            "Mock point 3"
        ],
        "confidence": "low"
    }
    
    import json
    return json.dumps(mock_response, indent=2)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    import sys
    
    # Initialize reprocessor
    reprocessor = LLMReprocessor(
        universal_index="./universal_index",
        hybrid_index="./hybrid_index"
    )
    
    # Wire up LLM (use stub for now)
    reprocessor.set_llm_function(call_llm)
    
    # Get query from command line or use default
    if len(sys.argv) > 1:
        query = ' '.join(sys.argv[1:])
    else:
        query = "How does payment validation work?"
    
    print("\n" + "="*70)
    print("LLM-POWERED DOCUMENT SEARCH")
    print("="*70)
    print(f"\nQuery: {query}")
    print("\nNOTE: Using STUB LLM function.")
    print("To use real LLM, replace the call_llm function with your integration.")
    print("="*70)
    
    # Search and refine
    results = reprocessor.search_and_refine(
        query=query,
        max_results=3,
        include_full_doc=False
    )
    
    # Display results
    formatted = reprocessor.format_results_for_display(results)
    print("\n" + formatted)
    
    # Save to file
    with open("llm_refined_results.txt", 'w', encoding='utf-8') as f:
        f.write(formatted)
    
    print("\n✓ Results saved to: llm_refined_results.txt")
