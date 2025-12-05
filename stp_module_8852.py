#!/usr/bin/env python3
"""
Test rules for 8852 against actual IFMLs.
8852 = Incorrect Length

Description: Field length incorrect (IBAN, account number, BIC, or other fields)

Includes:
- TP/TN/FP/FN metrics with Precision/Recall/F1
- Feature documentation
- Clear PARTY vs BANK distinction

Usage:
    python test_8852_rules_v2.py --data-dir /path/to/ifml/data --limit 10000
    python test_8852_rules_v2.py --show-docs   # Feature documentation
    python test_8852_rules_v2.py --show-party  # Party reference guide
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

sys.path.insert(0, '/mnt/project')
from data_pipeline import IFMLDataPipeline

TARGET_CODE = '8852'

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
FEATURE DOCUMENTATION FOR 8852 (Incorrect Length)
================================================================================

8852 fires when: Field length incorrect (IBAN, account number, BIC, or other fields)

KEY FEATURES:
┌────────────────────────────────┬────────────────────────────────────────────┐
│ Feature                        │ Meaning                                    │
├────────────────────────────────┼────────────────────────────────────────────┤
│ iban_length                    │ Actual length of IBAN string               │
│ iban_country                   │ IBAN country code (first 2 chars)          │
│ has_length_violation           │ Field exceeds maximum allowed length       │
│ orig_present                   │ Originator party is present (triggers va... │
│ account_length                 │ Length of account number field             │
│ bic_length                     │ Length of BIC (should be 8 or 11)          │
│ IBAN_LENGTHS                   │ Expected IBAN length varies by country: ... │
└────────────────────────────────┴────────────────────────────────────────────┘

REMEMBER:
• bnf_has_iban = Beneficiary's ACCOUNT has IBAN (from <CdtrAcct>)
• bnf_has_bic  = Beneficiary's BANK has BIC (from <CdtrAgt>)
• The IBAN belongs to the PARTY (their account number)
• The BIC identifies the BANK (where account is held)

================================================================================
"""


def check_rules(features: Dict) -> Tuple[bool, List[str]]:
    """
    Check if 8852 should fire based on extracted rules.
    Returns (should_fire, list_of_reasons)
    """
    reasons = []
    
    def get(feat, default=None):
        return features.get(feat, default)

    # IBAN length check by country
    IBAN_LENGTHS = {
        'AL': 28, 'AD': 24, 'AT': 20, 'AZ': 28, 'BH': 22, 'BY': 28, 'BE': 16,
        'BA': 20, 'BR': 29, 'BG': 22, 'HR': 21, 'CY': 28, 'CZ': 24, 'DK': 18,
        'EE': 20, 'FO': 18, 'FI': 18, 'FR': 27, 'GE': 22, 'DE': 22, 'GI': 23,
        'GR': 27, 'GL': 18, 'HU': 28, 'IS': 26, 'IE': 22, 'IL': 23, 'IT': 27,
        'JO': 30, 'KZ': 20, 'XK': 20, 'KW': 30, 'LV': 21, 'LB': 28, 'LI': 21,
        'LT': 20, 'LU': 20, 'MT': 31, 'MR': 27, 'MU': 30, 'MC': 27, 'MD': 24,
        'ME': 22, 'NL': 18, 'NO': 15, 'PK': 24, 'PS': 29, 'PL': 28, 'PT': 25,
        'QA': 29, 'RO': 24, 'SM': 27, 'SA': 24, 'RS': 22, 'SK': 24, 'SI': 19,
        'ES': 24, 'SE': 24, 'CH': 21, 'TN': 24, 'TR': 26, 'UA': 29, 'AE': 23,
        'GB': 22, 'VA': 22, 'VG': 24
    }
    
    # Rule 1: IBAN length mismatch for country
    for prefix in ['bnf_', 'cdt_', 'dbt_', 'orig_']:
        if get(f'{prefix}has_iban', False):
            country = get(f'{prefix}iban_country', '')
            length = get(f'{prefix}iban_length', 0) or 0
            expected = IBAN_LENGTHS.get(country)
            if expected and length != expected:
                party = prefix.upper().rstrip('_')
                reasons.append(f"{party}: IBAN length {length} ≠ expected {expected} for {country}")
    
    # Rule 2: Explicit length violation flag
    for prefix in ['bnf_', 'cdt_', 'dbt_', 'orig_']:
        if get(f'{prefix}has_length_violation', False):
            party = prefix.upper().rstrip('_')
            reasons.append(f"{party}: has_length_violation=True")
    
    # Rule 3: Originator present (triggers field validation including length checks)
    if get('orig_present', False):
        reasons.append("ORIG: present (triggers validation)")
    
    # Rule 4: Account length present (account being validated)
    for prefix in ['bnf_', 'cdt_', 'dbt_']:
        acct_len = get(f'{prefix}account_length', 0) or 0
        if acct_len > 0:
            party = prefix.upper().rstrip('_')
            reasons.append(f"{party}: account_length={acct_len}")
    
    # Rule 5: BIC length validation (should be 8 or 11)
    for prefix in ['bnf_', 'cdt_', 'dbt_', 'send_', 'intm_']:
        bic_len = get(f'{prefix}bic_length', 0) or 0
        if bic_len > 0 and bic_len not in [8, 11]:
            party = prefix.upper().rstrip('_')
            reasons.append(f"{party}: BIC length {bic_len} (should be 8 or 11)")

    
    return len(reasons) > 0, reasons


def get_debug_features(features: Dict) -> Dict:
    """Extract key features for debugging failures."""
    return {
        'bnf_has_iban': features.get('bnf_has_iban'),
        'bnf_iban_length': features.get('bnf_iban_length'),
        'bnf_iban_country': features.get('bnf_iban_country'),
        'cdt_has_iban': features.get('cdt_has_iban'),
        'cdt_iban_length': features.get('cdt_iban_length'),
        'orig_present': features.get('orig_present'),
        'bnf_account_length': features.get('bnf_account_length'),
        'cdt_account_length': features.get('cdt_account_length'),
        'bnf_bic_length': features.get('bnf_bic_length'),
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
    print(f"TEST RESULTS: {TARGET_CODE} - Incorrect Length")
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
            for p, c in sorted(fn_patterns.items(), key=lambda x: -x[1])[:8]:
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
        
        for i, (txn, reasons, codes, _) in enumerate(fp_list[:args.show_fp], 1):
            print(f"{i}. {txn}")
            print(f"   Reason: {reasons[0] if reasons else '?'}")
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
    
    --show-docs   Show feature documentation
    --show-party  Show party reference (BNF vs BNP, Party vs Bank)
    """)


if __name__ == "__main__":
    main()
