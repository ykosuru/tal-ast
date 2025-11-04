#!/usr/bin/env python3
"""
QUICKSTART: LLM Document Reprocessor
Minimal working example - get started in 5 minutes
"""

from llm_document_reprocessor import LLMReprocessor

# =============================================================================
# STEP 1: Choose your LLM and replace this function
# =============================================================================

def call_llm(prompt: str, context: str) -> str:
    """
    Replace this with YOUR LLM integration
    
    This function receives:
    - prompt: Instructions for the LLM
    - context: The document content to analyze
    
    It should return:
    - JSON string with extracted content (see format below)
    """
    
    # -------------------------------------------------------------------------
    # OPTION A: OpenAI (recommended for best results)
    # -------------------------------------------------------------------------
    """
    import openai
    openai.api_key = "your-api-key-here"
    
    response = openai.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": context}
        ],
        temperature=0.1,
        max_tokens=4096
    )
    return response.choices[0].message.content
    """
    
    # -------------------------------------------------------------------------
    # OPTION B: Anthropic Claude
    # -------------------------------------------------------------------------
    """
    import anthropic
    client = anthropic.Anthropic(api_key="your-api-key-here")
    
    message = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=4096,
        temperature=0.1,
        messages=[{"role": "user", "content": f"{prompt}\n\n{context}"}]
    )
    return message.content[0].text
    """
    
    # -------------------------------------------------------------------------
    # OPTION C: Local Ollama (free!)
    # -------------------------------------------------------------------------
    """
    import requests
    
    response = requests.post(
        'http://localhost:11434/api/generate',
        json={
            'model': 'llama2',
            'prompt': f"{prompt}\n\n{context}",
            'stream': False
        }
    )
    return response.json()['response']
    """
    
    # -------------------------------------------------------------------------
    # CURRENT: Using stub for demo (replace with one of the above!)
    # -------------------------------------------------------------------------
    print("  ⚠️  Using STUB - Replace call_llm() with real LLM!")
    return '''
    {
        "relevant_sections": [
            {
                "text": "This is a demo response. Wire up your LLM to see real results!",
                "relevance": "Demo section",
                "location": "Replace call_llm() function"
            }
        ],
        "summary": "This is a stub response. Replace the call_llm() function above with your LLM integration.",
        "key_points": [
            "Replace call_llm() function",
            "Add your API key",
            "Choose OpenAI, Claude, or Ollama"
        ],
        "confidence": "low"
    }
    '''


# =============================================================================
# STEP 2: Initialize and run
# =============================================================================

def main():
    print("\n" + "="*70)
    print("LLM DOCUMENT REPROCESSOR - QUICKSTART")
    print("="*70)
    
    # Initialize reprocessor
    print("\n[1/4] Initializing...")
    reprocessor = LLMReprocessor(
        universal_index="./universal_index",  # Path to your universal index
        hybrid_index="./hybrid_index"         # Path to your hybrid index
    )
    
    # Wire up LLM
    print("\n[2/4] Wiring up LLM function...")
    reprocessor.set_llm_function(call_llm)
    
    # Run search with LLM refinement
    print("\n[3/4] Searching and refining results...")
    
    query = "How does payment validation work?"  # ← Change this to your query
    
    results = reprocessor.search_and_refine(
        query=query,
        max_results=3,          # Process top 3 documents
        include_full_doc=False  # Set True to include full document
    )
    
    # Display results
    print("\n[4/4] Displaying results...")
    print("\n" + "="*70)
    formatted = reprocessor.format_results_for_display(results)
    print(formatted)
    
    # Save to file
    with open("llm_results.txt", 'w', encoding='utf-8') as f:
        f.write(formatted)
    
    print("\n✓ Results saved to: llm_results.txt")
    print("="*70)
    
    return results


# =============================================================================
# STEP 3: Run it!
# =============================================================================

if __name__ == "__main__":
    """
    TO RUN:
    
    1. Replace call_llm() function above with your LLM integration
    2. Run: python quickstart_llm_reprocessor.py
    3. View results in llm_results.txt
    
    NEXT STEPS:
    - Adjust max_results (default: 3)
    - Change max_doc_length for longer documents
    - Customize the extraction prompt
    - Add cost tracking (see llm_integration_examples.py)
    - Enable caching for repeat queries
    """
    
    try:
        results = main()
        
        # Quick summary
        print("\n📊 QUICK SUMMARY:")
        print(f"   Documents processed: {len(results)}")
        
        high_conf = sum(1 for r in results if r.get('llm_confidence') == 'high')
        print(f"   High confidence: {high_conf}")
        
        total_sections = sum(
            len(r.get('llm_extracted_content', [])) 
            for r in results
        )
        print(f"   Relevant sections found: {total_sections}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTROUBLESHOOTING:")
        print("  1. Did you replace the call_llm() function?")
        print("  2. Are your indexes built (universal_index, hybrid_index)?")
        print("  3. Is your LLM API accessible?")
        print("\nSee LLM_REPROCESSOR_README.md for detailed help")
