#!/usr/bin/env python3
"""
Test rules for 8894 against actual IFMLs.
8894 = IBAN Validation Failed

V4 CHANGES:
- Added BBAN validation (country-specific structure and check digits)
- New features: bban_structure_valid, bban_check_valid, iban_fully_valid
- Removed "needs IBAN but missing" (caused 859 FPs)

Usage:
    python test_8894_rules_v4.py --data-dir /path/to/ifml/data
    python test_8894_rules_v4.py --show-docs
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

sys.path.insert(0, '/mnt/project')
from data_pipeline import IFMLDataPipeline

TARGET_CODE = '8894'


def check_rules(features: Dict) -> Tuple[bool, List[str]]:
    """
    Check if 8894 should fire based on extracted rules.
    Returns (should_fire, list_of_reasons)
    
    8894 = IBAN Validation Failed
    Fires when IBAN has format/checksum/BBAN structure issues.
    """
    reasons = []
    
    def get(feat, default=None):
        return features.get(feat, default)

    # Check all party prefixes
    for prefix in ['bnf_', 'cdt_', 'dbt_', 'orig_', 'intm_', 'send_']:
        party = prefix.upper().rstrip('_')
        
        # Rule 1: Account contains invalid characters (STRONG signal)
        if get(f'{prefix}account_has_dirty_chars', False):
            reasons.append(f"{party}: account has dirty chars")
        
        # Rule 2: IBAN checksum is invalid (STRONG signal)
        if get(f'{prefix}has_iban', False) and get(f'{prefix}iban_checksum_valid') == False:
            reasons.append(f"{party}: IBAN checksum invalid")
        
        # Rule 3: IBAN format is invalid (STRONG signal)
        if get(f'{prefix}has_iban', False) and get(f'{prefix}iban_valid_format') == False:
            reasons.append(f"{party}: IBAN format invalid")
        
        # Rule 4: BBAN structure is invalid (NEW - for country-specific formats)
        if get(f'{prefix}has_iban', False) and get(f'{prefix}bban_structure_valid') == False:
            reasons.append(f"{party}: BBAN structure invalid")
        
        # Rule 5: BBAN internal check digit is invalid (NEW)
        if get(f'{prefix}has_iban', False) and get(f'{prefix}bban_check_valid') == False:
            reasons.append(f"{party}: BBAN check digit invalid")
        
        # Rule 6: Composite check - IBAN not fully valid
        # This catches any IBAN validation failure
        if get(f'{prefix}has_iban', False) and get(f'{prefix}iban_fully_valid') == False:
            # Only add if not already caught by specific rules above
            if not any(party in r for r in reasons):
                reasons.append(f"{party}: IBAN not fully valid")
    
    return len(reasons) > 0, reasons


def get_debug_features(features: Dict) -> Dict:
    """Extract key features for debugging failures."""
    debug = {}
    for prefix in ['bnf_', 'cdt_', 'dbt_']:
        debug[f'{prefix}has_iban'] = features.get(f'{prefix}has_iban')
        debug[f'{prefix}iban_valid_format'] = features.get(f'{prefix}iban_valid_format')
        debug[f'{prefix}iban_checksum_valid'] = features.get(f'{prefix}iban_checksum_valid')
        debug[f'{prefix}bban_structure_valid'] = features.get(f'{prefix}bban_structure_valid')
        debug[f'{prefix}bban_check_valid'] = features.get(f'{prefix}bban_check_valid')
        debug[f'{prefix}iban_fully_valid'] = features.get(f'{prefix}iban_fully_valid')
        debug[f'{prefix}account_has_dirty_chars'] = features.get(f'{prefix}account_has_dirty_chars')
        # Show the actual IBAN value for debugging
        acct = features.get(f'{prefix}account_value', '')
        if acct and len(acct) > 4:
            debug[f'{prefix}iban_preview'] = acct[:4] + '...' + acct[-4:] if len(acct) > 12 else acct
    return debug


def main():
    parser = argparse.ArgumentParser(description=f'Test {TARGET_CODE} rules V4 (with BBAN)')
    parser.add_argument('--data-dir', nargs='?', default=None, help='IFML JSON directory')
    parser.add_argument('--limit', type=int, default=100000, help='Max records')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show all')
    parser.add_argument('--show-failures', type=int, default=10, help='FN to show')
    parser.add_argument('--show-fp', type=int, default=10, help='FP to show')
    parser.add_argument('--show-docs', action='store_true', help='Show rule docs')
    
    args = parser.parse_args()
    
    if args.show_docs:
        print("""
================================================================================
8894 RULES V4 - IBAN Validation Failed (with BBAN validation)
================================================================================

RULES:
  1. account_has_dirty_chars = True
  2. has_iban=True AND iban_checksum_valid=False  (IBAN mod-97 check)
  3. has_iban=True AND iban_valid_format=False    (IBAN length/format)
  4. has_iban=True AND bban_structure_valid=False (NEW: country-specific format)
  5. has_iban=True AND bban_check_valid=False     (NEW: internal check digits)
  6. has_iban=True AND iban_fully_valid=False     (catches any validation failure)

NEW FEATURES (V4):
  - bban_structure_valid: Does BBAN match country-specific structure?
    Example: German BBAN must be 8 digits bank code + 10 digits account
  
  - bban_check_valid: Do internal check digits pass?
    Countries with internal checks: ES, IT, FR, BE, FI, NO, PL, PT, etc.
  
  - iban_fully_valid: Composite - all validation passes
    True only if: iban_valid_format AND iban_checksum_valid AND bban_*_valid

COUNTRIES WITH BBAN CHECK DIGITS:
  Spain (ES): 2 check digits in positions 9-10
  Italy (IT): CIN letter at position 1
  France (FR): RIB key at end (2 digits)
  Belgium (BE): mod 97 check
  Finland (FI): Luhn-style check
  Norway (NO): mod 11 check
  Poland (PL): check digit in bank code
  Portugal (PT): mod 97 check
================================================================================
        """)
        return
    
    if not args.data_dir:
        parser.error("--data-dir required (or use --show-docs)")
    
    print(f"Loading IFML data for {TARGET_CODE} validation (V4 with BBAN)...")
    pipeline = IFMLDataPipeline()
    data_path = Path(args.data_dir)
    
    if data_path.is_file():
        pipeline.load_single_file(str(data_path))
    else:
        pipeline.load_directory(str(data_path), "*.json")
    
    print(f"Pipeline has {len(pipeline.records)} records")
    
    # Classification
    tp_list, tn_list, fp_list, fn_list = [], [], [], []
    
    for i, record in enumerate(pipeline.records):
        if i >= args.limit:
            break
        
        txn_id = record.transaction_id
        features = record.request_features
        codes = record.error_codes_only + (record.composite_codes or [])
        
        has_actual = any(TARGET_CODE in str(c) for c in codes)
        predicted, reasons = check_rules(features)
        
        if predicted and has_actual:
            tp_list.append((txn_id, reasons, codes))
        elif not predicted and not has_actual:
            tn_list.append((txn_id,))
        elif predicted and not has_actual:
            fp_list.append((txn_id, reasons, codes, get_debug_features(features)))
        else:
            fn_list.append((txn_id, codes, get_debug_features(features)))
    
    # Metrics
    tp, tn, fp, fn = len(tp_list), len(tn_list), len(fp_list), len(fn_list)
    total = tp + tn + fp + fn
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / total if total > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Report
    print("\n" + "="*70)
    print(f"TEST RESULTS: {TARGET_CODE} - IBAN Validation Failed (V4 + BBAN)")
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
│ Recall         │ {recall*100:>6.2f}%  │ TP/(TP+FN) - Catch rate ⬅ IMPORTANT     │
│ F1 Score       │ {f1*100:>6.2f}%  │ Harmonic mean of Precision & Recall     │
│ Specificity    │ {specificity*100:>6.2f}%  │ TN/(TN+FP) - True negative rate         │
│ Accuracy       │ {accuracy*100:>6.2f}%  │ (TP+TN)/Total                           │
└────────────────┴──────────┴─────────────────────────────────────────┘
    """)
    
    # False Negatives
    if fn_list:
        print("\n" + "="*70)
        print(f"FALSE NEGATIVES ({fn} total, showing {min(fn, args.show_failures)}):")
        print("="*70)
        print("MISSED: 8894 occurred but rules didn't predict it.\n")
        
        # Analyze patterns in FN
        fn_patterns = defaultdict(int)
        for _, _, debug in fn_list:
            for k, v in debug.items():
                if v is True: 
                    fn_patterns[f"{k}=True"] += 1
                elif v is False: 
                    fn_patterns[f"{k}=False"] += 1
        
        if fn_patterns:
            print("Pattern Analysis (why we missed them):")
            for p, c in sorted(fn_patterns.items(), key=lambda x: -x[1])[:12]:
                print(f"  {p}: {c}/{fn} ({c/fn*100:.1f}%)")
            print()
        
        for i, (txn, codes, debug) in enumerate(fn_list[:args.show_failures], 1):
            print(f"{i}. {txn}")
            print(f"   Codes: {[c for c in codes if '8894' in str(c)]}")
            # Show IBAN-related debug features
            relevant = {k: v for k, v in debug.items() 
                       if v not in [None, ''] and 'iban' in k.lower() or 'bban' in k.lower()}
            print(f"   IBAN Features: {relevant}\n")
    
    # False Positives  
    if fp_list and args.show_fp > 0:
        print("\n" + "="*70)
        print(f"FALSE POSITIVES ({fp} total, showing {min(fp, args.show_fp)}):")
        print("="*70)
        print("OVER-PREDICTED: Rules said 8894 but it didn't occur.\n")
        
        for i, (txn, reasons, codes, debug) in enumerate(fp_list[:args.show_fp], 1):
            print(f"{i}. {txn}")
            print(f"   Trigger: {reasons}")
            print(f"   Actual codes: {codes[:5]}...\n")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    r_icon = "✅" if recall >= 0.95 else "⚠️" if recall >= 0.50 else "❌"
    p_icon = "✅" if precision >= 0.30 else "⚠️" if precision >= 0.10 else "ℹ️"
    
    print(f"""
    Recall:    {recall*100:>6.1f}%  {r_icon}  (Target: ≥95%)
    Precision: {precision*100:>6.1f}%  {p_icon}  (Acceptable: ≥10% for rare errors)
    F1 Score:  {f1*100:>6.1f}%
    
    V4 adds BBAN validation:
    - bban_structure_valid: Country-specific BBAN format check
    - bban_check_valid: Internal check digit validation (ES, IT, FR, BE, etc.)
    - iban_fully_valid: Composite of all IBAN/BBAN checks
    """)


if __name__ == "__main__":
    main()
