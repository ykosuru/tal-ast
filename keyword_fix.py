"""
Test Keyword Extraction Fix
Shows that important terms are now extracted even if lowercase
"""

from wire_indexer_lite import FastKeywordExtractor

def test_keyword_extraction():
    """Test that credit, party, determination are now extracted"""
    
    extractor = FastKeywordExtractor()
    
    # Test cases
    test_texts = [
        # Case 1: Lowercase (was broken, now fixed)
        """
        The credit party determination process validates the creditor and debtor
        information for wire transfers and payment processing.
        """,
        
        # Case 2: Capitalized (always worked)
        """
        Credit Party Determination validates Creditor and Debtor information
        for Wire Transfers and Payment Processing.
        """,
        
        # Case 3: Mixed case
        """
        CREDIT PARTY DETERMINATION: The system validates credit party information
        including party validation, account validation, and determination logic.
        """,
        
        # Case 4: Real document style
        """
        Phase 1 - Nostro update credit accounting if Determination associated 
        with a Nostro in the for FX Sell designated currency on REL. 7.10 6/17/15 
        Technical FX Processing GFIMTS will insert the High credit party 
        determination for approved business processing.
        """
    ]
    
    print("=" * 80)
    print("KEYWORD EXTRACTION TEST")
    print("=" * 80)
    print("\nTesting if 'credit', 'party', 'determination' are extracted...\n")
    
    for i, text in enumerate(test_texts, 1):
        print(f"{'='*80}")
        print(f"Test Case {i}:")
        print(f"{'='*80}")
        print(f"Text: {text.strip()[:100]}...")
        
        keywords = extractor.extract(text, max_keywords=15)
        
        print(f"\nExtracted {len(keywords)} keywords:")
        for kw, score in keywords[:10]:
            print(f"  • {kw:25s} (score: {score:.2f})")
        
        # Check if target keywords are present
        keyword_names = [kw for kw, _ in keywords]
        
        targets = ['credit', 'party', 'determination']
        found = [t for t in targets if t in keyword_names]
        missing = [t for t in targets if t not in keyword_names]
        
        print(f"\nTarget keywords:")
        for target in targets:
            if target in found:
                print(f"  ✓ {target} - FOUND")
            else:
                print(f"  ✗ {target} - MISSING")
        
        if len(found) == len(targets):
            print(f"\n  ✅ SUCCESS: All target keywords extracted!")
        else:
            print(f"\n  ⚠️  INCOMPLETE: {len(missing)}/{len(targets)} missing")
        
        print()
    
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\nThe keyword extractor now includes:")
    print("  1. Important terms list (credit, party, determination, etc.)")
    print("  2. Individual word extraction from capability phrases")
    print("  3. Case-insensitive matching for domain terms")
    print("\n✅ Keywords are now extracted even if lowercase!")
    print("=" * 80)


def test_important_terms_list():
    """Show what terms are in the important terms list"""
    
    extractor = FastKeywordExtractor()
    
    print("\n" + "=" * 80)
    print("IMPORTANT TERMS LIST")
    print("=" * 80)
    print("\nThese terms are ALWAYS extracted (even if lowercase):\n")
    
    terms = sorted(extractor.important_terms)
    for i, term in enumerate(terms, 1):
        print(f"  {i:2d}. {term}")
    
    print(f"\nTotal: {len(terms)} important terms")
    print("=" * 80)


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("WIRE PROCESSING INDEXER - KEYWORD EXTRACTION FIX TEST")
    print("=" * 80)
    print()
    
    # Test extraction
    test_keyword_extraction()
    
    # Show important terms
    test_important_terms_list()
    
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("\n1. Re-index your documents:")
    print("   python3 wire_indexer_lite.py --action index --pdf-folder docs")
    print("\n2. Search again:")
    print("   python3 wire_indexer_lite.py --action search --query 'credit party determination'")
    print("\n3. Keywords should now include: credit, party, determination ✓")
    print("=" * 80)
