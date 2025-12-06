#!/usr/bin/env python3
"""
Test rules for 8894 - IBAN Validation Failed (FINAL VERSION)

Merged from:
- validate_8xxx_all.py (99.6% recall, 43.3% precision) - WINNING RULES
- test_8894_rules_v8.py - documentation and analysis structure

8894 = "IBAN Validation Failed"
ACE validates IBANs against external bank directories. This error fires when:
- IBAN checksum is invalid
- IBAN format is wrong for country
- BIC-IBAN consistency check fails
- Bank code not found in directory

Performance: 99.6% Recall, 43.3% Precision, F1=60%

Usage:
    python test_8894_rules.py --data-dir /path/to/ifml/data --limit 10000
    python test_8894_rules.py --show-docs
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

sys.path.insert(0, '/mnt/project')
from data_pipeline import IFMLDataPipeline

TARGET_CODE = '8894'

# =============================================================================
# DOCUMENTATION
# =============================================================================

FEATURE_DOCS = """
================================================================================
FEATURE DOCUMENTATION FOR 8894 (IBAN Validation Failed)
================================================================================

8894 fires when: ACE's IBAN validation fails (checksum, format, BIC consistency, 
or bank directory lookup)

RULES USED (from validate_8xxx_all.py - 99.6% recall, 43% precision):

┌─────┬────────────────────────────────┬─────────────────────────────────────────┐
│ #   │ Rule                           │ Why It Triggers 8894                    │
├─────┼────────────────────────────────┼─────────────────────────────────────────┤
│ 1   │ account_has_dirty_chars=True   │ Invalid chars in account trigger valid. │
│ 2   │ has_iban AND checksum=False    │ Direct IBAN checksum failure            │
│ 3   │ needs_iban=True                │ IBAN processing/validation triggered    │
│ 4   │ dbt_is_international=True      │ International payments trigger valid.   │
│ 5   │ has_bic AND has_iban           │ ACE validates BIC-IBAN consistency      │
└─────┴────────────────────────────────┴─────────────────────────────────────────┘

KEY INSIGHT:
Unlike 8004 (IBAN derivation), 8894 fires when IBAN VALIDATION fails.
This includes checksum errors, format errors, and BIC-IBAN mismatches.

WHY 43% PRECISION IS EXPECTED:
- ACE uses external bank directories we can't access
- Some validations depend on proprietary consistency checks
- We predict when validation will OCCUR, not always when it will FAIL

PREFIXES: bnf_ (beneficiary), cdt_ (creditor), dbt_ (debtor)

================================================================================
"""

PARTY_REFERENCE = """
================================================================================
PARTY REFERENCE - Understanding ACE Abbreviations
================================================================================

PARTIES (People/Companies):
┌──────────┬─────────────────┬────────────────────────────────────────────────┐
│ Abbrev   │ Full Name       │ Description                                    │
├──────────┼─────────────────┼────────────────────────────────────────────────┤
│ BNF      │ Beneficiary     │ Person/company RECEIVING the payment           │
│ CDT      │ Creditor        │ Same as Beneficiary (ISO 20022 term)           │
│ DBT      │ Debtor          │ Person/company SENDING/PAYING                  │
└──────────┴─────────────────┴────────────────────────────────────────────────┘

ACE ERROR CODE SUFFIXES:
┌──────────┬────────────────────────────────────────────────────────────────┐
│ Suffix   │ Meaning                                                        │
├──────────┼────────────────────────────────────────────────────────────────┤
│ _BNFBNK  │ Issue with Beneficiary's BANK                                  │
│ _BNPPTY  │ Issue with Beneficiary PARTY (the person/company)              │
│ _CDTPTY  │ Issue with Creditor PARTY                                      │
│ _DBTPTY  │ Issue with Debtor PARTY                                        │
└──────────┴────────────────────────────────────────────────────────────────┘

KEY DISTINCTION:
• IBAN belongs to the PARTY (it's their account number)
• BIC identifies the BANK (where the account is held)
• 8894 validates the IBAN itself, not just its presence

================================================================================
"""


# =============================================================================
# RULE IMPLEMENTATION (from validate_8xxx_all.py - WINNING VERSION)
# =============================================================================

def check_8894_rules(features: Dict) -> Tuple[bool, List[str]]:
    """
    8894 = IBAN Validation Failed
    
    Rules from validate_8xxx_all.py (99.6% recall, 43.3% precision)
    
    These rules predict when ACE will VALIDATE an IBAN, which may fail due to:
    - Invalid checksum
    - Wrong format for country
    - BIC-IBAN mismatch
    - Bank not in directory
    """
    reasons = []
    
    def get(feat, default=None):
        return features.get(feat, default)
    
    # -------------------------------------------------------------------------
    # Rule 1: Account has dirty/invalid characters
    # Invalid characters trigger validation errors
    # -------------------------------------------------------------------------
    for prefix in ['bnf_', 'cdt_', 'dbt_']:
        if get(f'{prefix}account_has_dirty_chars', False):
            party = prefix.rstrip('_').upper()
            reasons.append(f"{party}: account_has_dirty_chars=True")
    
    # -------------------------------------------------------------------------
    # Rule 2: IBAN checksum invalid
    # Direct checksum failure - most deterministic rule
    # -------------------------------------------------------------------------
    for prefix in ['bnf_', 'cdt_', 'dbt_']:
        has_iban = get(f'{prefix}has_iban', False)
        checksum_valid = get(f'{prefix}iban_checksum_valid')
        if has_iban and checksum_valid == False:
            party = prefix.rstrip('_').upper()
            reasons.append(f"{party}: iban_checksum_valid=False")
    
    # -------------------------------------------------------------------------
    # Rule 3: Party needs IBAN (IBAN processing triggered)
    # When party is in IBAN country, validation will occur
    # -------------------------------------------------------------------------
    for prefix in ['bnf_', 'cdt_', 'dbt_']:
        if get(f'{prefix}needs_iban', False):
            party = prefix.rstrip('_').upper()
            reasons.append(f"{party}: needs_iban=True")
    
    # -------------------------------------------------------------------------
    # Rule 4: International debtor transaction
    # Cross-border payments trigger additional validation
    # -------------------------------------------------------------------------
    if get('dbt_is_international', False):
        reasons.append("DBT: is_international=True")
    
    # -------------------------------------------------------------------------
    # Rule 5: Has both BIC and IBAN
    # ACE validates BIC-IBAN consistency when both present
    # -------------------------------------------------------------------------
    for prefix in ['bnf_', 'cdt_', 'dbt_']:
        has_bic = get(f'{prefix}has_bic', False)
        has_iban = get(f'{prefix}has_iban', False)
        if has_bic and has_iban:
            party = prefix.rstrip('_').upper()
            reasons.append(f"{party}: has both BIC and IBAN")
    
    return len(reasons) > 0, reasons


def get_debug_features(features: Dict) -> Dict:
    """Extract key features for debugging."""
    return {
        'bnf_has_iban': features.get('bnf_has_iban'),
        'bnf_iban_checksum_valid': features.get('bnf_iban_checksum_valid'),
        'bnf_needs_iban': features.get('bnf_needs_iban'),
        'bnf_has_bic': features.get('bnf_has_bic'),
        'bnf_account_has_dirty_chars': features.get('bnf_account_has_dirty_chars'),
        'cdt_has_iban': features.get('cdt_has_iban'),
        'cdt_needs_iban': features.get('cdt_needs_iban'),
        'dbt_is_international': features.get('dbt_is_international'),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description=f'Test {TARGET_CODE} rules')
    parser.add_argument('--data-dir', nargs='?', default=None, help='IFML JSON directory')
    parser.add_argument('--limit', type=int, default=10000, help='Max records')
    parser.add_argument('--show-fn', type=int, default=15, help='FN to show')
    parser.add_argument('--show-fp', type=int, default=10, help='FP to show')
    parser.add_argument('--show-tp', type=int, default=5, help='TP to show')
    parser.add_argument('--show-docs', action='store_true', help='Show feature docs')
    parser.add_argument('--show-party', action='store_true', help='Show party reference')
    
    args = parser.parse_args()
    
    if args.show_docs:
        print(FEATURE_DOCS)
        return
    
    if args.show_party:
        print(PARTY_REFERENCE)
        return
    
    if not args.data_dir:
        parser.error("--data-dir required (or use --show-docs / --show-party)")
    
    print(f"Loading IFML data for {TARGET_CODE} validation...")
    pipeline = IFMLDataPipeline()
    data_path = Path(args.data_dir)
    
    if data_path.is_file():
        pipeline.load_single_file(str(data_path))
    else:
        pipeline.load_directory(str(data_path), "*.json")
    
    print(f"Pipeline has {len(pipeline.records)} records")
    
    # Classification
    tp_list, tn_list, fp_list, fn_list = [], [], [], []
    trigger_counts = defaultdict(int)
    
    for i, record in enumerate(pipeline.records):
        if i >= args.limit:
            break
        
        txn_id = record.transaction_id
        features = record.request_features
        codes = record.error_codes_only + (record.composite_codes or [])
        
        has_actual = any(TARGET_CODE in str(c) for c in codes)
        predicted, reasons = check_8894_rules(features)
        
        # Track trigger frequency
        for r in reasons:
            trigger_type = r.split(':')[0] if ':' in r else r
            trigger_counts[trigger_type] += 1
        
        if predicted and has_actual:
            tp_list.append((txn_id, reasons, codes, get_debug_features(features)))
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
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Report
    print("\n" + "="*70)
    print(f"TEST RESULTS: {TARGET_CODE} - IBAN Validation Failed")
    print("="*70)
    
    print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│  CONFUSION MATRIX                                                   │
├─────────────────────────────────────────────────────────────────────┤
│                              Actual                                 │
│                      {TARGET_CODE}        Not {TARGET_CODE}                        │
│                  ┌──────────┬──────────┐                            │
│  Predicted       │          │          │                            │
│  {TARGET_CODE}          │ TP={tp:<5} │ FP={fp:<5} │  Predicted Pos: {tp+fp:<6}   │
│                  ├──────────┼──────────┤                            │
│  Not {TARGET_CODE}      │ FN={fn:<5} │ TN={tn:<5} │  Predicted Neg: {tn+fn:<6}   │
│                  └──────────┴──────────┘                            │
│                    Actual+    Actual-                               │
│                    {tp+fn:<6}     {tn+fp:<6}                               │
└─────────────────────────────────────────────────────────────────────┘

┌────────────────┬──────────┬─────────────────────────────────────────┐
│ Metric         │ Value    │ Meaning                                 │
├────────────────┼──────────┼─────────────────────────────────────────┤
│ Precision      │ {precision*100:>6.2f}%  │ When we predict, how often correct?     │
│ Recall         │ {recall*100:>6.2f}%  │ What % of actual {TARGET_CODE} do we catch?     │
│ F1 Score       │ {f1*100:>6.2f}%  │ Harmonic mean of Precision & Recall     │
│ Specificity    │ {specificity*100:>6.2f}%  │ True negative rate                      │
└────────────────┴──────────┴─────────────────────────────────────────┘
    """)
    
    # Trigger Analysis
    print("\nPREDICTION TRIGGERS (what rules fired):")
    for trigger, count in sorted(trigger_counts.items(), key=lambda x: -x[1])[:15]:
        pct = count / (tp + fp) * 100 if (tp + fp) > 0 else 0
        print(f"  {trigger}: {count:,} ({pct:.1f}% of predictions)")
    
    # True Positives
    if tp_list and args.show_tp > 0:
        print(f"\n{'='*70}")
        print(f"TRUE POSITIVES ({tp} total, showing {min(args.show_tp, tp)}):")
        print("="*70)
        print("These are correctly predicted 8894 errors.\n")
        
        for i, (txn, reasons, codes, debug) in enumerate(tp_list[:args.show_tp], 1):
            print(f"{i}. {txn}")
            print(f"   Triggers: {reasons[:2]}")
            print(f"   Codes: {[c for c in codes if TARGET_CODE in str(c)]}")
            print()
    
    # False Negatives
    if fn_list and args.show_fn > 0:
        print(f"\n{'='*70}")
        print(f"FALSE NEGATIVES ({fn} total, showing {min(args.show_fn, fn)}):")
        print("="*70)
        print(f"MISSED: {TARGET_CODE} occurred but rules didn't predict it.\n")
        
        # Pattern analysis
        fn_patterns = defaultdict(int)
        for _, _, debug in fn_list:
            for k, v in debug.items():
                if v is True:
                    fn_patterns[f"{k}=True"] += 1
                elif v is False:
                    fn_patterns[f"{k}=False"] += 1
        
        if fn_patterns:
            print("Pattern Analysis (features in FN cases):")
            for p, c in sorted(fn_patterns.items(), key=lambda x: -x[1])[:10]:
                print(f"  {p}: {c} ({c/fn*100:.1f}%)")
            print()
        
        for i, (txn, codes, debug) in enumerate(fn_list[:args.show_fn], 1):
            print(f"{i}. {txn}")
            print(f"   Codes: {[c for c in codes if TARGET_CODE in str(c)]}")
            print(f"   Features: bnf_has_iban={debug.get('bnf_has_iban')}, "
                  f"bnf_needs_iban={debug.get('bnf_needs_iban')}, "
                  f"dbt_is_international={debug.get('dbt_is_international')}")
            print()
    
    # False Positives
    if fp_list and args.show_fp > 0:
        print(f"\n{'='*70}")
        print(f"FALSE POSITIVES ({fp} total, showing {min(args.show_fp, fp)}):")
        print("="*70)
        print(f"OVER-PREDICTED: Rules said {TARGET_CODE} but it didn't occur.\n")
        
        # Analyze which rules trigger FPs
        fp_triggers = defaultdict(int)
        for _, reasons, _, _ in fp_list:
            for r in reasons:
                trigger = r.split(':')[0] if ':' in r else r
                fp_triggers[trigger] += 1
        
        if fp_triggers:
            print("FP Trigger Analysis (which rules cause false positives):")
            for trigger, count in sorted(fp_triggers.items(), key=lambda x: -x[1])[:10]:
                print(f"  {trigger}: {count} ({count/fp*100:.1f}%)")
            print()
        
        for i, (txn, reasons, codes, _) in enumerate(fp_list[:args.show_fp], 1):
            print(f"{i}. {txn}")
            print(f"   Trigger: {reasons[0] if reasons else '?'}")
            print(f"   Actual codes: {codes[:5]}")
            print()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    recall_icon = "✅" if recall >= 0.95 else "⚠️" if recall >= 0.5 else "❌"
    precision_icon = "✅" if precision >= 0.40 else "⚠️" if precision >= 0.20 else "ℹ️"
    
    print(f"""
    Recall:    {recall*100:>6.1f}%  {recall_icon}  (Target: ≥95%)
    Precision: {precision*100:>6.1f}%  {precision_icon}  (Expected ~43% - cannot predict ACE directory lookups)
    F1 Score:  {f1*100:>6.1f}%
    
    Rules (from validate_8xxx_all.py):
    1. account_has_dirty_chars=True (invalid characters)
    2. has_iban AND iban_checksum_valid=False (checksum failure)
    3. needs_iban=True (IBAN processing triggered)
    4. dbt_is_international=True (cross-border validation)
    5. has_bic AND has_iban (BIC-IBAN consistency check)
    
    NOTE: 43% precision is expected. We predict when IBAN VALIDATION occurs,
    but ACE's external bank directory lookups are unpredictable.
    
    --show-docs   Show feature documentation
    --show-party  Show party reference guide
    """)


if __name__ == "__main__":
    main()
