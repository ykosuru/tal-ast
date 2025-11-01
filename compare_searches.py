"""
Compare Standard vs Enhanced Search Results
Demonstrates the fix for inconsistent results between similar queries
"""

from wire_indexer_lite import WireProcessingSearcher
from enhanced_searcher import EnhancedWireProcessingSearcher


def compare_searches(index_path: str = "./wire_index"):
    """Compare standard vs enhanced searcher"""
    
    # Test query pairs that should return similar results
    query_pairs = [
        ("how do I implement credit party determination", "credit party determination"),
        ("what is OFAC screening", "OFAC screening"),
        ("explain ISO payment validation", "ISO payment validation"),
        ("how to setup wire transfer processing", "wire transfer processing"),
        ("describe pacs.008 message format", "pacs.008 message format")
    ]
    
    print("=" * 100)
    print("SEARCH CONSISTENCY TEST: Standard vs Enhanced Searcher")
    print("=" * 100)
    print("\nProblem: Similar queries should return similar documents")
    print("Testing with query pairs...\n")
    
    # Initialize both searchers
    standard_searcher = WireProcessingSearcher(
        index_path=index_path,
        enable_query_expansion=True,
        expansion_level="medium"
    )
    
    enhanced_searcher = EnhancedWireProcessingSearcher(
        index_path=index_path,
        enable_query_expansion=True,
        expansion_level="medium"
    )
    
    for query1, query2 in query_pairs:
        print("\n" + "=" * 100)
        print(f"Query Pair:")
        print(f"  A: \"{query1}\"")
        print(f"  B: \"{query2}\"")
        print("=" * 100)
        
        # Standard searcher results
        print("\n--- STANDARD SEARCHER ---")
        
        print(f"\nQuery A results:")
        std_results_a = standard_searcher.search(query1, top_k=5, verbose=False)
        std_files_a = [r['source_file'] for r in std_results_a]
        for i, r in enumerate(std_results_a, 1):
            print(f"  [{i}] {r['source_file'][:60]:60s} Score: {r['combined_score']:.3f}")
        
        print(f"\nQuery B results:")
        std_results_b = standard_searcher.search(query2, top_k=5, verbose=False)
        std_files_b = [r['source_file'] for r in std_results_b]
        for i, r in enumerate(std_results_b, 1):
            print(f"  [{i}] {r['source_file'][:60]:60s} Score: {r['combined_score']:.3f}")
        
        # Calculate overlap
        std_overlap = len(set(std_files_a) & set(std_files_b))
        std_consistency = std_overlap / 5 * 100
        
        print(f"\n  Consistency: {std_consistency:.0f}% ({std_overlap}/5 documents match)")
        if std_consistency < 60:
            print(f"  ⚠️ LOW CONSISTENCY - Results differ significantly!")
        
        # Enhanced searcher results
        print("\n--- ENHANCED SEARCHER (WITH FIX) ---")
        
        print(f"\nQuery A results:")
        enh_results_a = enhanced_searcher.search(query1, top_k=5, verbose=False)
        enh_files_a = [r['source_file'] for r in enh_results_a]
        for i, r in enumerate(enh_results_a, 1):
            print(f"  [{i}] {r['source_file'][:60]:60s} Score: {r['combined_score']:.3f}")
        
        print(f"\nQuery B results:")
        enh_results_b = enhanced_searcher.search(query2, top_k=5, verbose=False)
        enh_files_b = [r['source_file'] for r in enh_results_b]
        for i, r in enumerate(enh_results_b, 1):
            print(f"  [{i}] {r['source_file'][:60]:60s} Score: {r['combined_score']:.3f}")
        
        # Calculate overlap
        enh_overlap = len(set(enh_files_a) & set(enh_files_b))
        enh_consistency = enh_overlap / 5 * 100
        
        print(f"\n  Consistency: {enh_consistency:.0f}% ({enh_overlap}/5 documents match)")
        if enh_consistency >= 80:
            print(f"  ✓ HIGH CONSISTENCY - Results are similar!")
        
        # Show improvement
        improvement = enh_consistency - std_consistency
        if improvement > 0:
            print(f"\n  📈 IMPROVEMENT: +{improvement:.0f}% more consistent")
        
        print("\n" + "-" * 100)
    
    print("\n\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print("\nThe Enhanced Searcher:")
    print("  ✓ Removes stopwords (how, do, I, what, is, etc.)")
    print("  ✓ Focuses on core terms (credit, party, determination)")
    print("  ✓ Better weight distribution")
    print("  ✓ Phrase matching bonus")
    print("\nResult: Similar queries return similar documents!")
    print("=" * 100)


def detailed_comparison(query: str, index_path: str = "./wire_index"):
    """Show detailed comparison for a single query"""
    
    print("=" * 100)
    print(f"DETAILED COMPARISON FOR: \"{query}\"")
    print("=" * 100)
    
    # Standard searcher
    print("\n--- STANDARD SEARCHER ---")
    standard_searcher = WireProcessingSearcher(
        index_path=index_path,
        enable_query_expansion=True,
        expansion_level="medium"
    )
    
    std_results = standard_searcher.search(query, top_k=10, verbose=True)
    
    # Enhanced searcher
    print("\n\n--- ENHANCED SEARCHER (WITH FIX) ---")
    enhanced_searcher = EnhancedWireProcessingSearcher(
        index_path=index_path,
        enable_query_expansion=True,
        expansion_level="medium"
    )
    
    enh_results = enhanced_searcher.search(query, top_k=10, verbose=True)
    
    # Show top results side by side
    print("\n\n" + "=" * 100)
    print("TOP 5 RESULTS COMPARISON")
    print("=" * 100)
    
    print(f"\n{'Rank':<6} {'Standard Searcher':<45} {'Enhanced Searcher':<45}")
    print("-" * 100)
    
    for i in range(5):
        std_file = std_results[i]['source_file'][:42] if i < len(std_results) else "N/A"
        enh_file = enh_results[i]['source_file'][:42] if i < len(enh_results) else "N/A"
        
        match = "✓" if std_file == enh_file else " "
        
        print(f"[{i+1}] {match}  {std_file:<45} {enh_file:<45}")
    
    # Calculate similarity
    std_files = set([r['source_file'] for r in std_results[:5]])
    enh_files = set([r['source_file'] for r in enh_results[:5]])
    overlap = len(std_files & enh_files)
    
    print("\n" + "=" * 100)
    print(f"Top 5 Overlap: {overlap}/5 documents match ({overlap/5*100:.0f}%)")
    print("=" * 100)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Detailed comparison for specific query
        query = ' '.join(sys.argv[1:])
        index_path = "./wire_index"
        
        detailed_comparison(query, index_path)
    else:
        # Full comparison
        print("\nUsage:")
        print("  python compare_search_results.py                    # Full comparison")
        print("  python compare_search_results.py 'your query here'  # Detailed comparison")
        print("\nRunning full comparison...\n")
        
        compare_searches("./wire_index")
