#!/usr/bin/env python3
"""
Test rules for 8004 against actual IFMLs.
8004 = Missing IBAN

Description: IBAN is required but not provided for a party in an IBAN-country

Includes:
- TP/TN/FP/FN metrics with Precision/Recall/F1
- Feature documentation
- Clear PARTY vs BANK distinction

Usage:
    python test_8004_rules_v2.py --data-dir /path/to/ifml/data --limit 10000
    python test_8004_rules_v2.py --show-docs   # Feature documentation
    python test_8004_rules_v2.py --show-party  # Party reference guide
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

sys.path.insert(0, '/mnt/project')
from data_pipeline import IFMLDataPipeline

TARGET_CODE = '8004'

# =============================================================================
# PARTY REFERENCE (BNF vs BNP, Party vs Bank)
# =============================================================================
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
│ ORIG     │ Originator      │ Original initiator of the payment              │
└──────────┴─────────────────┴────────────────────────────────────────────────┘

BANKS/AGENTS (Financial Institutions):
┌──────────┬─────────────────┬────────────────────────────────────────────────┐
│ Abbrev   │ Full Name       │ Description                                    │
├──────────┼─────────────────┼────────────────────────────────────────────────┤
│ BNFBNK   │ Beneficiary Bank│ Bank where beneficiary has their account       │
│ CDTAGT   │ Creditor Agent  │ Same as Beneficiary Bank (ISO 20022 term)      │
│ DBTAGT   │ Debtor Agent    │ Bank where debtor has their account            │
│ INTM     │ Intermediary    │ Correspondent/intermediary bank in the chain   │
│ SEND     │ Sender          │ Sending financial institution                  │
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

IFML/ISO 20022 STRUCTURE:
┌─────────────────┬──────────────────┬───────────────────────────────────────┐
│ IFML Element    │ Contains         │ Example Fields                        │
├─────────────────┼──────────────────┼───────────────────────────────────────┤
│ <Cdtr>          │ Creditor PARTY   │ Name, Address, Country, ID            │
│ <CdtrAcct>      │ Creditor Account │ IBAN, Account Number (belongs to party)│
│ <CdtrAgt>       │ Creditor Agent   │ BIC (this is the BANK's identifier)   │
├─────────────────┼──────────────────┼───────────────────────────────────────┤
│ <Dbtr>          │ Debtor PARTY     │ Name, Address, Country, ID            │
│ <DbtrAcct>      │ Debtor Account   │ IBAN, Account Number                  │
│ <DbtrAgt>       │ Debtor Agent     │ BIC (debtor's bank)                   │
└─────────────────┴──────────────────┴───────────────────────────────────────┘

FEATURE PREFIX MEANINGS IN OUR CODE:
┌──────────┬────────────────────────────────────────────────────────────────┐
│ Prefix   │ What It Refers To                                              │
├──────────┼────────────────────────────────────────────────────────────────┤
│ bnf_     │ Beneficiary context (party info + their account + their bank)  │
│ cdt_     │ Creditor context (same as bnf_, ISO 20022 naming)              │
│ dbt_     │ Debtor context (party info + their account + their bank)       │
│ orig_    │ Originator context                                             │
│ intm_    │ Intermediary bank context                                      │
│ send_    │ Sender institution context                                     │
└──────────┴────────────────────────────────────────────────────────────────┘

IMPORTANT CLARIFICATIONS:
─────────────────────────────────────────────────────────────────────────────
• bnf_has_iban    = Beneficiary's ACCOUNT has IBAN (from <CdtrAcct>)
• bnf_has_bic     = Beneficiary's BANK has BIC (from <CdtrAgt>)
• bnf_has_account = Beneficiary has account number (from <CdtrAcct>)
• bnf_has_name    = Beneficiary PARTY has name (from <Cdtr>)
• bnf_country     = Beneficiary PARTY's country (from <Cdtr> address)

• The IBAN belongs to the PARTY (it's their account number)
• The BIC identifies the BANK (where the account is held)
─────────────────────────────────────────────────────────────────────────────
"""

# =============================================================================
# FEATURE DOCUMENTATION
# =============================================================================
FEATURE_DOCS = """
================================================================================
FEATURE DOCUMENTATION FOR 8004 (Missing IBAN)
================================================================================

8004 fires when: IBAN is required but not provided for a party in an IBAN-country

KEY FEATURES:
┌────────────────────────────────┬────────────────────────────────────────────┐
│ Feature                        │ Meaning                                    │
├────────────────────────────────┼────────────────────────────────────────────┤
│ bnf_present                    │ Beneficiary party exists in message        │
│ bnf_needs_iban                 │ Beneficiary is in IBAN-country             │
│ bnf_has_iban                   │ Beneficiary account has IBAN               │
│ bnf_has_account                │ Beneficiary has a bank account number      │
│ cdt_present                    │ Creditor party exists in message           │
│ cdt_needs_iban                 │ Creditor (=Beneficiary) needs IBAN         │
│ cdt_has_iban                   │ Creditor account has IBAN                  │
│ send_has_bic                   │ Sender has BIC                             │
│ send_bic_valid_format          │ Sender's BIC format is valid               │
└────────────────────────────────┴────────────────────────────────────────────┘

REMEMBER:
• bnf_has_iban = Beneficiary's ACCOUNT has IBAN (from <CdtrAcct>)
• bnf_has_bic  = Beneficiary's BANK has BIC (from <CdtrAgt>)
• The IBAN belongs to the PARTY (their account number)
• The BIC identifies the BANK (where account is held)

================================================================================
"""


def check_8004_rf_rules(features: Dict) -> Tuple[bool, List[str]]:
    """
    Check if 8004 should fire based on RF-extracted rules.
    Returns (should_fire, reasons)
    
    ORIGINAL LOGIC - RESTORED
    """
    reasons = []
    
    # Helper to get feature with fallback
    def get(feat, default=None):
        if feat in features:
            return features[feat]
        return default
    
    matches = 0
    
    # -------------------------------------------------------------------------
    # Rule 1: Original 8004 logic - needs_iban but doesn't have it
    # -------------------------------------------------------------------------
    for prefix in ['bnf_', 'cdt_', 'dbt_', 'orig_', 'intm_']:
        party_present = get(f'{prefix}present', False)
        needs_iban = get(f'{prefix}needs_iban', False)
        has_iban = get(f'{prefix}has_iban', False)
        
        if party_present and needs_iban and not has_iban:
            party = prefix.rstrip('_').upper()
            reasons.append(f"{party}: needs_iban=True, has_iban=False")
            matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 2: NEW - has_account but no IBAN (ACE tries to derive and fails)
    # -------------------------------------------------------------------------
    for prefix in ['bnf_', 'cdt_', 'dbt_', 'orig_']:
        party_present = get(f'{prefix}present', False)
        has_account = get(f'{prefix}has_account', False)
        has_iban = get(f'{prefix}has_iban', False)
        
        # If party has account but no IBAN, ACE may try to derive IBAN
        if party_present and has_account and not has_iban:
            party = prefix.rstrip('_').upper()
            reasons.append(f"{party}: has_account=True, has_iban=False (IBAN derivation may fail)")
            matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 3: RF pattern - BIC validation failures correlate with 8004
    # -------------------------------------------------------------------------
    for prefix in ['send_', 'bnf_', 'cdt_', 'intm_']:
        has_bic = get(f'{prefix}has_bic', False)
        bic_valid_format = get(f'{prefix}bic_valid_format', True)
        bic_valid_country = get(f'{prefix}bic_valid_country', True)
        
        if has_bic and not bic_valid_format:
            party = prefix.rstrip('_').upper()
            reasons.append(f"{party}: has_bic but bic_valid_format=False")
            matches += 1
        
        if has_bic and not bic_valid_country:
            party = prefix.rstrip('_').upper()
            reasons.append(f"{party}: has_bic but bic_valid_country=False")
            matches += 1
    
    # Fire if we found any matching pattern
    should_fire = matches > 0
    
    return should_fire, reasons


def get_debug_features(features: Dict) -> Dict:
    """Extract key features for debugging failures."""
    return {
        'bnf_present': features.get('bnf_present'),
        'bnf_needs_iban': features.get('bnf_needs_iban'),
        'bnf_has_iban': features.get('bnf_has_iban'),
        'bnf_has_account': features.get('bnf_has_account'),
        'cdt_present': features.get('cdt_present'),
        'cdt_needs_iban': features.get('cdt_needs_iban'),
        'cdt_has_iban': features.get('cdt_has_iban'),
        'send_has_bic': features.get('send_has_bic'),
        'send_bic_valid_format': features.get('send_bic_valid_format'),
    }


def main():
    parser = argparse.ArgumentParser(description=f'Test {TARGET_CODE} rules')
    parser.add_argument('--data-dir', nargs='?', default=None, help='IFML JSON directory')
    parser.add_argument('--limit', type=int, default=10000, help='Max records')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show all')
    parser.add_argument('--show-failures', type=int, default=20, help='FN to show')
    parser.add_argument('--show-fp', type=int, default=10, help='FP to show')
    parser.add_argument('--show-docs', action='store_true', help='Feature docs')
    parser.add_argument('--show-party', action='store_true', help='Party reference')
    
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
    
    for i, record in enumerate(pipeline.records):
        if i >= args.limit:
            break
        
        txn_id = record.transaction_id
        features = record.request_features
        codes = record.error_codes_only + (record.composite_codes or [])
        
        has_actual = any(TARGET_CODE in str(c) for c in codes)
        predicted, reasons = check_8004_rf_rules(features)
        
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
    print(f"TEST RESULTS: {TARGET_CODE} - Missing IBAN")
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
│ Recall         │ {recall*100:>6.2f}%  │ TP/(TP+FN) - Catch rate ⬅ MOST IMPORTANT │
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
        print(f"MISSED: {TARGET_CODE} occurred but rules didn't predict it.\n")
        
        fn_patterns = defaultdict(int)
        for _, _, debug in fn_list:
            for k, v in debug.items():
                if v is True: fn_patterns[f"{k}=True"] += 1
                elif v is False: fn_patterns[f"{k}=False"] += 1
        
        if fn_patterns:
            print("Pattern Analysis:")
            for p, c in sorted(fn_patterns.items(), key=lambda x: -x[1])[:10]:
                print(f"  {p}: {c} ({c/fn*100:.1f}%)")
            print()
        
        for i, (txn, codes, debug) in enumerate(fn_list[:args.show_failures], 1):
            print(f"{i}. {txn}")
            print(f"   Codes: {codes}")
            print(f"   Features: {debug}\n")
    
    # False Positives
    if fp_list and args.show_fp > 0:
        print("\n" + "="*70)
        print(f"FALSE POSITIVES ({fp} total, showing {min(fp, args.show_fp)}):")
        print("="*70)
        print(f"OVER-PREDICTED: Rules said {TARGET_CODE} but it didn't occur.\n")
        
        fp_patterns = defaultdict(int)
        for _, reasons, _, _ in fp_list:
            for r in reasons:
                trigger = r.split(':')[0] if ':' in r else r
                fp_patterns[trigger] += 1
        
        if fp_patterns:
            print("FP Trigger Analysis:")
            for p, c in sorted(fp_patterns.items(), key=lambda x: -x[1])[:10]:
                print(f"  {p}: {c} ({c/fp*100:.1f}%)")
            print()
        
        for i, (txn, reasons, codes, _) in enumerate(fp_list[:args.show_fp], 1):
            print(f"{i}. {txn}")
            print(f"   Trigger: {reasons[0] if reasons else '?'}")
            print(f"   Actual: {codes}\n")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    r_icon = "✅" if recall >= 0.99 else "⚠️" if recall >= 0.95 else "❌"
    p_icon = "✅" if precision >= 0.50 else "⚠️" if precision >= 0.20 else "ℹ️"
    print(f"""
    Recall:    {recall*100:>6.1f}%  {r_icon}  (Target: ≥99%)
    Precision: {precision*100:>6.1f}%  {p_icon}  (Lower OK for prediction)
    F1 Score:  {f1*100:>6.1f}%
    
    Rules (ORIGINAL LOGIC RESTORED):
    1. party_present AND needs_iban AND NOT has_iban
    2. party_present AND has_account AND NOT has_iban (IBAN derivation may fail)
    3. has_bic AND NOT bic_valid_format
    4. has_bic AND NOT bic_valid_country
    
    --show-docs   Show feature documentation
    --show-party  Show party reference (BNF vs BNP, Party vs Bank)
    """)


if __name__ == "__main__":
    main()
