#!/usr/bin/env python3
"""
Test rules for 8894 - IBAN Validation Failed (V2 - STRICT RULES)

8894 = "IBAN Validation Failed"
ACE validates IBANs against external bank directories. This error fires when:
- IBAN checksum is invalid
- IBAN format is wrong for country
- BIC-IBAN consistency check fails

IMPORTANT: 8894 is RARE. Most "needs IBAN" cases result in 6021 (party validation),
not 8894 (IBAN validation). We need strict rules to avoid over-prediction.

V2 Changes:
- REMOVED: needs_iban=True alone (too broad - causes 95% of FPs)
- REMOVED: dbt_is_international alone (too broad)
- KEPT: iban_checksum_valid=False (direct signal)
- KEPT: account_has_dirty_chars=True (direct signal)
- GATED: BIC+IBAN only when checksum issues present

Usage:
    python test_8894_rules.py --data-dir /path/to/ifml/data --limit 10000
    python test_8894_rules.py --show-docs
    python test_8894_rules.py --loose  # For high recall version
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict

sys.path.insert(0, '/mnt/project')
from data_pipeline import IFMLDataPipeline

TARGET_CODE = '8894'

# =============================================================================
# DOCUMENTATION
# =============================================================================

FEATURE_DOCS = """
================================================================================
FEATURE DOCUMENTATION FOR 8894 (IBAN Validation Failed) - V2 STRICT
================================================================================

8894 fires when: ACE's IBAN validation fails (checksum, format, BIC consistency)

CRITICAL INSIGHT:
8894 is RARE. Most payments with needs_iban=True result in 6021 (party validation
failure), NOT 8894. The distinction:

┌────────────────────┬─────────────────────────────────────────────────────────┐
│ 6021               │ Party info invalid (bank doesn't exist, wrong format)  │
│ 8894               │ IBAN itself invalid (checksum, length, format)         │
└────────────────────┴─────────────────────────────────────────────────────────┘

V2 STRICT RULES (Precision-focused):

┌─────┬─────────────────────────────────┬────────────────────────────────────────┐
│ #   │ Rule                            │ Why It Triggers 8894                   │
├─────┼─────────────────────────────────┼────────────────────────────────────────┤
│ 1   │ has_iban + checksum=False       │ Direct IBAN checksum failure           │
│ 2   │ account_has_dirty_chars=True    │ Invalid chars cause validation error   │
│ 3   │ has_iban + iban_length_valid=F  │ IBAN wrong length for country          │
│ 4   │ has_iban + iban_country_valid=F │ IBAN country code invalid              │
└─────┴─────────────────────────────────┴────────────────────────────────────────┘

REMOVED RULES (caused 95%+ of false positives):
- needs_iban=True alone (too broad - most go to 6021)
- dbt_is_international=True alone (too broad)
- has_bic AND has_iban alone (too broad)

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

KEY DISTINCTION FOR 8894:
• 8894 = IBAN itself is malformed (checksum, length, format)
• 6021 = Party/Bank info invalid (bank not in directory, structural issues)

================================================================================
"""


# =============================================================================
# RULE IMPLEMENTATION - V2 STRICT (Precision-focused)
# =============================================================================

def check_8894_rules(features: Dict) -> Tuple[bool, List[str], Optional[str]]:
    """
    8894 = IBAN Validation Failed (V2 STRICT)
    
    Only predict when we have DIRECT evidence of IBAN validation failure:
    - Checksum invalid
    - Dirty characters in account
    - IBAN length/country invalid
    
    REMOVED broad rules that caused 95%+ false positives:
    - needs_iban=True alone
    - dbt_is_international alone
    - has_bic AND has_iban alone
    
    Returns:
        Tuple of (predicted: bool, reasons: List[str], predicted_code: Optional[str])
    """
    reasons = []
    triggering_parties: Set[str] = set()
    
    def get(feat, default=None):
        return features.get(feat, default)
    
    # -------------------------------------------------------------------------
    # Rule 1: IBAN checksum invalid (STRONGEST SIGNAL)
    # Direct evidence that IBAN validation will fail
    # -------------------------------------------------------------------------
    for prefix in ['bnf_', 'cdt_', 'dbt_']:
        has_iban = get(f'{prefix}has_iban', False)
        checksum_valid = get(f'{prefix}iban_checksum_valid')
        if has_iban and checksum_valid == False:
            party = prefix.rstrip('_').upper()
            reasons.append(f"{party}: has_iban + iban_checksum_valid=False (checksum failure)")
            triggering_parties.add(party)
    
    # -------------------------------------------------------------------------
    # Rule 2: Account has dirty/invalid characters
    # Invalid characters will cause IBAN validation to fail
    # -------------------------------------------------------------------------
    for prefix in ['bnf_', 'cdt_', 'dbt_']:
        if get(f'{prefix}account_has_dirty_chars', False):
            party = prefix.rstrip('_').upper()
            reasons.append(f"{party}: account_has_dirty_chars=True (invalid characters)")
            triggering_parties.add(party)
    
    # -------------------------------------------------------------------------
    # Rule 3: IBAN length invalid for country
    # If present, this is a direct validation failure signal
    # -------------------------------------------------------------------------
    for prefix in ['bnf_', 'cdt_', 'dbt_']:
        has_iban = get(f'{prefix}has_iban', False)
        length_valid = get(f'{prefix}iban_length_valid')
        if has_iban and length_valid == False:
            party = prefix.rstrip('_').upper()
            reasons.append(f"{party}: has_iban + iban_length_valid=False (wrong length)")
            triggering_parties.add(party)
    
    # -------------------------------------------------------------------------
    # Rule 4: IBAN country code invalid
    # If present, this is a direct validation failure signal
    # -------------------------------------------------------------------------
    for prefix in ['bnf_', 'cdt_', 'dbt_']:
        has_iban = get(f'{prefix}has_iban', False)
        country_valid = get(f'{prefix}iban_country_valid')
        if has_iban and country_valid == False:
            party = prefix.rstrip('_').upper()
            reasons.append(f"{party}: has_iban + iban_country_valid=False (invalid country)")
            triggering_parties.add(party)
    
    # -------------------------------------------------------------------------
    # Rule 5: IBAN format invalid (if feature exists)
    # -------------------------------------------------------------------------
    for prefix in ['bnf_', 'cdt_', 'dbt_']:
        has_iban = get(f'{prefix}has_iban', False)
        format_valid = get(f'{prefix}iban_format_valid')
        if has_iban and format_valid == False:
            party = prefix.rstrip('_').upper()
            reasons.append(f"{party}: has_iban + iban_format_valid=False (format error)")
            triggering_parties.add(party)
    
    predicted = len(reasons) > 0
    
    # Infer suffix for emission based on triggering party
    predicted_code = None
    if predicted:
        if 'BNF' in triggering_parties:
            suffix = '_BNPPTY'
        elif 'CDT' in triggering_parties:
            suffix = '_CDTPTY'
        elif 'DBT' in triggering_parties:
            suffix = '_DBTPTY'
        else:
            suffix = '_BNFBNK'
        predicted_code = f"{TARGET_CODE}{suffix}"
    
    return predicted, reasons, predicted_code


def check_8894_rules_loose(features: Dict) -> Tuple[bool, List[str], Optional[str]]:
    """
    8894 = IBAN Validation Failed (LOOSE VERSION - High Recall)
    
    Use this if you need maximum recall at the cost of precision.
    Includes the broad rules that cause many false positives.
    
    Returns:
        Tuple of (predicted: bool, reasons: List[str], predicted_code: Optional[str])
    """
    reasons = []
    triggering_parties: Set[str] = set()
    
    def get(feat, default=None):
        return features.get(feat, default)
    
    # All strict rules first
    for prefix in ['bnf_', 'cdt_', 'dbt_']:
        has_iban = get(f'{prefix}has_iban', False)
        checksum_valid = get(f'{prefix}iban_checksum_valid')
        if has_iban and checksum_valid == False:
            party = prefix.rstrip('_').upper()
            reasons.append(f"{party}: has_iban + iban_checksum_valid=False")
            triggering_parties.add(party)
    
    for prefix in ['bnf_', 'cdt_', 'dbt_']:
        if get(f'{prefix}account_has_dirty_chars', False):
            party = prefix.rstrip('_').upper()
            reasons.append(f"{party}: account_has_dirty_chars=True")
            triggering_parties.add(party)
    
    # LOOSE: needs_iban alone (HIGH FP RATE)
    for prefix in ['bnf_', 'cdt_', 'dbt_']:
        if get(f'{prefix}needs_iban', False):
            party = prefix.rstrip('_').upper()
            reasons.append(f"{party}: needs_iban=True (LOOSE)")
            triggering_parties.add(party)
    
    # LOOSE: dbt_is_international (HIGH FP RATE)
    if get('dbt_is_international', False):
        reasons.append("DBT: is_international=True (LOOSE)")
        triggering_parties.add('DBT')
    
    # LOOSE: BIC + IBAN (HIGH FP RATE)
    for prefix in ['bnf_', 'cdt_', 'dbt_']:
        if get(f'{prefix}has_bic', False) and get(f'{prefix}has_iban', False):
            party = prefix.rstrip('_').upper()
            reasons.append(f"{party}: has_bic + has_iban (LOOSE)")
            triggering_parties.add(party)
    
    predicted = len(reasons) > 0
    predicted_code = None
    if predicted:
        if 'BNF' in triggering_parties:
            suffix = '_BNPPTY'
        elif 'CDT' in triggering_parties:
            suffix = '_CDTPTY'
        elif 'DBT' in triggering_parties:
            suffix = '_DBTPTY'
        else:
            suffix = '_BNFBNK'
        predicted_code = f"{TARGET_CODE}{suffix}"
    
    return predicted, reasons, predicted_code


def get_debug_features(features: Dict) -> Dict:
    """Extract key features for debugging."""
    return {
        'bnf_has_iban': features.get('bnf_has_iban'),
        'bnf_iban_checksum_valid': features.get('bnf_iban_checksum_valid'),
        'bnf_iban_length_valid': features.get('bnf_iban_length_valid'),
        'bnf_iban_country_valid': features.get('bnf_iban_country_valid'),
        'bnf_needs_iban': features.get('bnf_needs_iban'),
        'bnf_has_bic': features.get('bnf_has_bic'),
        'bnf_account_has_dirty_chars': features.get('bnf_account_has_dirty_chars'),
        'cdt_has_iban': features.get('cdt_has_iban'),
        'cdt_iban_checksum_valid': features.get('cdt_iban_checksum_valid'),
        'cdt_needs_iban': features.get('cdt_needs_iban'),
        'dbt_is_international': features.get('dbt_is_international'),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description=f'Test {TARGET_CODE} rules (V2 STRICT)')
    parser.add_argument('--data-dir', nargs='?', default=None, help='IFML JSON directory')
    parser.add_argument('--limit', type=int, default=10000, help='Max records')
    parser.add_argument('--show-fn', type=int, default=15, help='FN to show')
    parser.add_argument('--show-fp', type=int, default=10, help='FP to show')
    parser.add_argument('--show-tp', type=int, default=5, help='TP to show')
    parser.add_argument('--show-docs', action='store_true', help='Show feature docs')
    parser.add_argument('--show-party', action='store_true', help='Show party reference')
    parser.add_argument('--loose', action='store_true', help='Use loose rules (high recall, low precision)')
    
    args = parser.parse_args()
    
    if args.show_docs:
        print(FEATURE_DOCS)
        return
    
    if args.show_party:
        print(PARTY_REFERENCE)
        return
    
    if not args.data_dir:
        parser.error("--data-dir required (or use --show-docs / --show-party)")
    
    # Select rule function
    rule_func = check_8894_rules_loose if args.loose else check_8894_rules
    rule_version = "LOOSE (high recall)" if args.loose else "STRICT (high precision)"
    
    print(f"Loading IFML data for {TARGET_CODE} validation...")
    print(f"Rule version: {rule_version}")
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
    suffix_accuracy = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for i, record in enumerate(pipeline.records):
        if i >= args.limit:
            break
        
        txn_id = record.transaction_id
        features = record.request_features
        codes = record.error_codes_only + (record.composite_codes or [])
        
        has_actual = any(TARGET_CODE in str(c) for c in codes)
        predicted, reasons, predicted_code = rule_func(features)
        
        # Track trigger frequency
        for r in reasons:
            trigger_type = r.split(':')[0] if ':' in r else r
            trigger_counts[trigger_type] += 1
        
        # Track suffix accuracy
        if predicted and predicted_code:
            actual_codes_8894 = [c for c in codes if TARGET_CODE in str(c)]
            suffix_accuracy[predicted_code]['total'] += 1
            if any(predicted_code in str(c) for c in actual_codes_8894):
                suffix_accuracy[predicted_code]['correct'] += 1
        
        if predicted and has_actual:
            tp_list.append((txn_id, reasons, codes, get_debug_features(features), predicted_code))
        elif not predicted and not has_actual:
            tn_list.append((txn_id,))
        elif predicted and not has_actual:
            fp_list.append((txn_id, reasons, codes, get_debug_features(features), predicted_code))
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
    print(f"TEST RESULTS: {TARGET_CODE} - IBAN Validation Failed ({rule_version})")
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
    if trigger_counts:
        print("\nPREDICTION TRIGGERS (what rules fired):")
        for trigger, count in sorted(trigger_counts.items(), key=lambda x: -x[1])[:15]:
            pct = count / (tp + fp) * 100 if (tp + fp) > 0 else 0
            print(f"  {trigger}: {count:,} ({pct:.1f}% of predictions)")
    else:
        print("\nNo predictions made (strict rules found no matches)")
    
    # Suffix Accuracy
    if any(s['total'] > 0 for s in suffix_accuracy.values()):
        print("\nSUFFIX PREDICTION ACCURACY:")
        for code, stats in sorted(suffix_accuracy.items(), key=lambda x: -x[1]['total']):
            if stats['total'] > 0:
                acc = stats['correct'] / stats['total'] * 100
                print(f"  {code}: {stats['correct']}/{stats['total']} ({acc:.1f}%)")
    
    # True Positives
    if tp_list and args.show_tp > 0:
        print(f"\n{'='*70}")
        print(f"TRUE POSITIVES ({tp} total, showing {min(args.show_tp, tp)}):")
        print("="*70)
        print("These are correctly predicted 8894 errors.\n")
        
        for i, (txn, reasons, codes, debug, pred_code) in enumerate(tp_list[:args.show_tp], 1):
            actual_8894 = [c for c in codes if TARGET_CODE in str(c)]
            print(f"{i}. {txn}")
            print(f"   Predicted: {pred_code}")
            print(f"   Actual: {actual_8894}")
            print(f"   Triggers: {reasons[:2]}")
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
                elif v is None:
                    fn_patterns[f"{k}=None"] += 1
        
        if fn_patterns:
            print("Pattern Analysis (features in FN cases):")
            for p, c in sorted(fn_patterns.items(), key=lambda x: -x[1])[:10]:
                pct = c / fn * 100 if fn > 0 else 0
                print(f"  {p}: {c} ({pct:.1f}%)")
            print()
        
        for i, (txn, codes, debug) in enumerate(fn_list[:args.show_fn], 1):
            actual_8894 = [c for c in codes if TARGET_CODE in str(c)]
            print(f"{i}. {txn}")
            print(f"   Codes: {actual_8894}")
            print(f"   bnf_has_iban={debug.get('bnf_has_iban')}, "
                  f"bnf_iban_checksum_valid={debug.get('bnf_iban_checksum_valid')}")
            print(f"   bnf_needs_iban={debug.get('bnf_needs_iban')}, "
                  f"bnf_account_has_dirty_chars={debug.get('bnf_account_has_dirty_chars')}")
            print()
    elif fn == 0:
        print(f"\n{'='*70}")
        print("FALSE NEGATIVES: None! All actual 8894 cases were caught.")
        print("="*70)
    
    # False Positives
    if fp_list and args.show_fp > 0:
        print(f"\n{'='*70}")
        print(f"FALSE POSITIVES ({fp} total, showing {min(args.show_fp, fp)}):")
        print("="*70)
        print(f"OVER-PREDICTED: Rules said {TARGET_CODE} but it didn't occur.\n")
        
        # Analyze which rules trigger FPs
        fp_triggers = defaultdict(int)
        for _, reasons, _, _, _ in fp_list:
            for r in reasons:
                trigger = r.split(':')[0] if ':' in r else r
                fp_triggers[trigger] += 1
        
        if fp_triggers:
            print("FP Trigger Analysis (which rules cause false positives):")
            for trigger, count in sorted(fp_triggers.items(), key=lambda x: -x[1])[:10]:
                pct = count / fp * 100 if fp > 0 else 0
                print(f"  {trigger}: {count} ({pct:.1f}%)")
            print()
        
        # What codes actually fired instead
        fp_actual_codes = defaultdict(int)
        for _, _, codes, _, _ in fp_list:
            for c in codes:
                code_base = str(c).split('_')[0]
                fp_actual_codes[code_base] += 1
        
        if fp_actual_codes:
            print("What codes fired instead of 8894:")
            for code, count in sorted(fp_actual_codes.items(), key=lambda x: -x[1])[:10]:
                print(f"  {code}: {count}")
            print()
        
        for i, (txn, reasons, codes, debug, pred_code) in enumerate(fp_list[:args.show_fp], 1):
            print(f"{i}. {txn}")
            print(f"   Predicted: {pred_code}")
            print(f"   Trigger: {reasons[0] if reasons else '?'}")
            print(f"   Actual codes: {codes[:5]}")
            print()
    elif fp == 0:
        print(f"\n{'='*70}")
        print("FALSE POSITIVES: None! No over-predictions.")
        print("="*70)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    recall_icon = "✅" if recall >= 0.95 else "⚠️" if recall >= 0.5 else "❌"
    precision_icon = "✅" if precision >= 0.40 else "⚠️" if precision >= 0.20 else "ℹ️"
    
    print(f"""
    Rule Version: {rule_version}
    
    Recall:    {recall*100:>6.1f}%  {recall_icon}  (Target: ≥95%)
    Precision: {precision*100:>6.1f}%  {precision_icon}  (Target: ≥40%)
    F1 Score:  {f1*100:>6.1f}%
    
    {"STRICT" if not args.loose else "LOOSE"} Rules:
    1. has_iban + iban_checksum_valid=False (checksum failure)
    2. account_has_dirty_chars=True (invalid characters)
    3. has_iban + iban_length_valid=False (wrong length)
    4. has_iban + iban_country_valid=False (invalid country)
    5. has_iban + iban_format_valid=False (format error)
    {"" if not args.loose else '''
    LOOSE ADDITIONS (high FP rate):
    6. needs_iban=True alone
    7. dbt_is_international=True alone
    8. has_bic + has_iban alone
    '''}
    NOTE: 8894 is rare. Most "needs IBAN" cases result in 6021, not 8894.
    Use --loose for high recall at cost of precision.
    
    --show-docs   Show feature documentation
    --show-party  Show party reference guide
    --loose       Use loose rules (high recall, low precision)
    """)


if __name__ == "__main__":
    main()
