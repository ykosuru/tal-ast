#!/usr/bin/env python3
"""
Test rules for 8026 against actual IFMLs.
8026 = BIC/IBAN Mismatch

Description: BIC country does not match IBAN country, or derivation issue

Includes:
- TP/TN/FP/FN metrics
- Precision/Recall/F1 calculation  
- Feature documentation

Usage:
    python test_8026_rules_v2.py --data-dir /path/to/ifml/data --limit 10000
    python test_8026_rules_v2.py --show-docs  # Show feature documentation
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

sys.path.insert(0, '/mnt/project')
from data_pipeline import IFMLDataPipeline

TARGET_CODE = '8026'

# =============================================================================
# FEATURE DOCUMENTATION
# =============================================================================
FEATURE_DOCS = """
================================================================================
FEATURE DOCUMENTATION FOR 8026 (BIC/IBAN Mismatch)
================================================================================

8026 fires when: BIC country does not match IBAN country, or derivation issue

FEATURE NAME                    | MEANING & HOW TO EXTRACT
--------------------------------|----------------------------------------------
bnf_has_bic                    | BNF party has a BIC/SWIFT code. Extract from <BIC> or <BICFI> tags.
bnf_has_iban                   | BNF party has an IBAN. Extract from <IBAN> tag.
bnf_bic_party_country_match    | BIC country matches BNF address country. Compare BIC[4:6] with party country.
bnf_bic_iban_match             | BIC country matches IBAN country. Compare BIC[4:6] with IBAN[0:2].
bnf_needs_iban                 | BNF is in IBAN country but no IBAN provided.
cdt_is_international           | CDT is in different country than sender.

--------------------------------------------------------------------------------
GENERAL EXTRACTION METHODS:
--------------------------------------------------------------------------------
1. Parse IFML XML/JSON structure
2. For each party (BNF, CDT, DBT, ORIG, INTM, SEND):
   - Extract BIC from <BIC> or <BICFI> tags
   - Extract IBAN from <IBAN> tag  
   - Extract account from <Acct>/<Id> tags
   - Extract country from <Ctry> or address fields

3. BIC Structure (8 or 11 chars): BANKCCLL[XXX]
   - BANK = Bank code (4 chars)
   - CC = Country code (2 chars) <-- Used for matching
   - LL = Location (2 chars)
   - XXX = Branch (optional, 3 chars)

4. IBAN Structure: CCkkBBBB...
   - CC = Country code (2 chars) <-- Used for matching
   - kk = Check digits (2 digits)
   - BBBB... = BBAN (country-specific)

5. IBAN Checksum Validation:
   - Move first 4 chars to end
   - Convert letters to numbers (A=10, B=11, ..., Z=35)
   - Result mod 97 should equal 1

================================================================================
"""


def check_rules(features: Dict) -> Tuple[bool, List[str]]:
    """
    Check if 8026 should fire based on extracted rules.
    Returns (should_fire, list_of_reasons)
    """
    reasons = []
    
    def get(feat, default=None):
        return features.get(feat, default)
    

    # Rule 1: Party needs IBAN
    for prefix in ['bnf_', 'cdt_', 'orig_', 'dbt_', 'intm_']:
        if get(f'{prefix}needs_iban', False):
            reasons.append(f"{prefix.upper().rstrip('_')}: needs_iban=True")
    
    # Rule 2: BIC/party country mismatch
    for prefix in ['bnf_', 'cdt_', 'dbt_']:
        if get(f'{prefix}has_bic', False) and get(f'{prefix}has_iban', False):
            if get(f'{prefix}bic_party_country_match') == False:
                reasons.append(f"{prefix.upper().rstrip('_')}: bic_party_country_match=False")
    
    # Rule 3: Has BIC but no IBAN
    for prefix in ['bnf_', 'cdt_', 'dbt_']:
        if get(f'{prefix}has_bic', False) and not get(f'{prefix}has_iban', False):
            reasons.append(f"{prefix.upper().rstrip('_')}: has_bic but no IBAN")
    
    # Rule 4: Has IBAN but no BIC
    for prefix in ['bnf_', 'cdt_', 'dbt_']:
        if get(f'{prefix}has_iban', False) and not get(f'{prefix}has_bic', False):
            reasons.append(f"{prefix.upper().rstrip('_')}: has IBAN but no BIC")
    
    # Rule 5: Has both BIC and IBAN
    for prefix in ['bnf_', 'cdt_', 'dbt_']:
        if get(f'{prefix}has_bic', False) and get(f'{prefix}has_iban', False):
            reasons.append(f"{prefix.upper().rstrip('_')}: has both BIC and IBAN")
    
    # Rule 6: CDT is international
    if get('cdt_is_international', False):
        reasons.append("CDT: is_international=True")
    
    # Rule 7: BNF missing both BIC and IBAN
    if not get('bnf_has_bic', False) and not get('bnf_has_iban', False):
        reasons.append("BNF: missing both BIC and IBAN")

    
    return len(reasons) > 0, reasons


def get_debug_features(features: Dict) -> Dict:
    """Extract key features for debugging failures."""
    return {
        'bnf_has_bic': features.get('bnf_has_bic'),
        'bnf_has_iban': features.get('bnf_has_iban'),
        'bnf_bic_party_country_match': features.get('bnf_bic_party_country_match'),
        'bnf_bic_iban_match': features.get('bnf_bic_iban_match'),
        'cdt_has_bic': features.get('cdt_has_bic'),
        'cdt_has_iban': features.get('cdt_has_iban'),
        'cdt_is_international': features.get('cdt_is_international'),
        'bnf_needs_iban': features.get('bnf_needs_iban'),
    }


def main():
    parser = argparse.ArgumentParser(description=f'Test {TARGET_CODE} rules against IFMLs')
    parser.add_argument('--data-dir', required=True, nargs='?', default=None, help='Directory with IFML JSON files')
    parser.add_argument('--limit', type=int, default=10000, help='Max records to process')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show all classifications')
    parser.add_argument('--show-failures', type=int, default=20, help='Number of FN to show')
    parser.add_argument('--show-fp', type=int, default=10, help='Number of FP to show')
    parser.add_argument('--show-docs', action='store_true', help='Show feature documentation')
    
    args = parser.parse_args()
    
    if args.show_docs:
        print(FEATURE_DOCS)
        return
    
    if not args.data_dir:
        parser.error("--data-dir is required unless using --show-docs")
    
    print(f"Loading IFML data for {TARGET_CODE} validation...")
    pipeline = IFMLDataPipeline()
    data_path = Path(args.data_dir)
    
    if data_path.is_file():
        pipeline.load_single_file(str(data_path))
    else:
        pipeline.load_directory(str(data_path), "*.json")
    
    print(f"Pipeline has {len(pipeline.records)} records")
    
    # =========================================================================
    # CLASSIFICATION: TP, TN, FP, FN
    # =========================================================================
    tp_list, tn_list, fp_list, fn_list = [], [], [], []
    
    for i, record in enumerate(pipeline.records):
        if i >= args.limit:
            break
        
        txn_id = record.transaction_id
        features = record.request_features
        actual_codes = record.error_codes_only
        composite_codes = record.composite_codes or []
        
        # Check if actually has target code
        has_actual = any(TARGET_CODE in str(c) for c in actual_codes + composite_codes)
        
        # Check if our rules predict it
        predicted, reasons = check_rules(features)
        
        if predicted and has_actual:
            tp_list.append((txn_id, reasons, composite_codes))
        elif not predicted and not has_actual:
            tn_list.append((txn_id,))
        elif predicted and not has_actual:
            fp_list.append((txn_id, reasons, composite_codes, get_debug_features(features)))
        else:  # not predicted and has_actual
            fn_list.append((txn_id, composite_codes, get_debug_features(features)))
        
        if args.verbose:
            status = "TP" if (predicted and has_actual) else "TN" if (not predicted and not has_actual) else "FP" if predicted else "FN"
            icon = "✅" if status in ["TP", "TN"] else "❌"
            print(f"{icon} {status} {txn_id}")
    
    # =========================================================================
    # CALCULATE METRICS
    # =========================================================================
    tp, tn, fp, fn = len(tp_list), len(tn_list), len(fp_list), len(fn_list)
    total = tp + tn + fp + fn
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / total if total > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # =========================================================================
    # PRINT REPORT
    # =========================================================================
    print("\n" + "="*70)
    print(f"TEST RESULTS: {TARGET_CODE} - BIC/IBAN Mismatch")
    print("="*70)
    
    print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│  CONFUSION MATRIX                                                    │
├─────────────────────────────────────────────────────────────────────┤
│                              Actual                                  │
│                      {TARGET_CODE}        Not {TARGET_CODE}                         │
│                  ┌──────────┬──────────┐                            │
│  Predicted       │          │          │                            │
│  {TARGET_CODE}          │ TP={tp:<5} │ FP={fp:<5} │  Predicted Pos: {tp+fp:<6}    │
│                  ├──────────┼──────────┤                            │
│  Not {TARGET_CODE}      │ FN={fn:<5} │ TN={tn:<5} │  Predicted Neg: {tn+fn:<6}    │
│                  └──────────┴──────────┘                            │
│                    Actual+    Actual-                               │
│                    {tp+fn:<6}     {tn+fp:<6}                                │
└─────────────────────────────────────────────────────────────────────┘

┌────────────────┬──────────┬─────────────────────────────────────────┐
│ Metric         │ Value    │ Meaning                                 │
├────────────────┼──────────┼─────────────────────────────────────────┤
│ Precision      │ {precision*100:>6.2f}%  │ TP/(TP+FP) - Prediction accuracy        │
│ Recall         │ {recall*100:>6.2f}%  │ TP/(TP+FN) - Catch rate (MOST IMPORTANT)│
│ F1 Score       │ {f1*100:>6.2f}%  │ Harmonic mean of Precision & Recall     │
│ Specificity    │ {specificity*100:>6.2f}%  │ TN/(TN+FP) - True negative rate         │
│ Accuracy       │ {accuracy*100:>6.2f}%  │ (TP+TN)/Total - Overall correctness     │
└────────────────┴──────────┴─────────────────────────────────────────┘
    """)
    
    # =========================================================================
    # FALSE NEGATIVES (Missed predictions - most important to fix)
    # =========================================================================
    if fn_list:
        print("\n" + "="*70)
        print(f"FALSE NEGATIVES ({fn} total, showing first {args.show_failures}):")
        print("="*70)
        print(f"These are MISSED predictions - {TARGET_CODE} occurred but rules didn't catch it.\n")
        
        # Pattern analysis
        fn_patterns = defaultdict(int)
        for item in fn_list:
            for k, v in item[2].items():
                if v is True:
                    fn_patterns[f"{k}=True"] += 1
                elif v is False:
                    fn_patterns[f"{k}=False"] += 1
        
        if fn_patterns:
            print("PATTERN ANALYSIS (common features in FN):")
            for pattern, count in sorted(fn_patterns.items(), key=lambda x: -x[1])[:10]:
                print(f"  {pattern}: {count} ({count/fn*100:.1f}%)")
            print()
        
        for i, (txn_id, codes, debug) in enumerate(fn_list[:args.show_failures], 1):
            print(f"{i}. TxnID: {txn_id}")
            print(f"   Actual codes: {codes}")
            print(f"   Features: {debug}\n")
    
    # =========================================================================
    # FALSE POSITIVES (Over-predictions)
    # =========================================================================
    if fp_list and args.show_fp > 0:
        print("\n" + "="*70)
        print(f"FALSE POSITIVES ({fp} total, showing first {args.show_fp}):")
        print("="*70)
        print(f"These are OVER-predictions - rules predicted {TARGET_CODE} but it didn't occur.\n")
        
        for i, (txn_id, reasons, codes, debug) in enumerate(fp_list[:args.show_fp], 1):
            print(f"{i}. TxnID: {txn_id}")
            print(f"   Predicted because: {reasons[0] if reasons else 'unknown'}")
            print(f"   Actual codes: {codes}\n")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    recall_icon = "✅" if recall >= 0.99 else ("⚠️" if recall >= 0.95 else "❌")
    precision_icon = "✅" if precision >= 0.50 else ("⚠️" if precision >= 0.20 else "❌")
    
    print(f"""
    Recall:    {recall*100:>6.1f}%  {recall_icon}  (Target: ≥99% - catch all errors)
    Precision: {precision*100:>6.1f}%  {precision_icon}  (Lower OK for error prediction)
    F1 Score:  {f1*100:>6.1f}%

    For ERROR PREDICTION, RECALL is most important.
    We want to catch ALL errors, even if it means some false positives.
    
    To see feature documentation: python {sys.argv[0]} --show-docs
    """)


if __name__ == "__main__":
    main()
