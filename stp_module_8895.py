#!/usr/bin/env python3
"""
Test rules for 8895 against actual IFMLs.
8895 = NCH Validation Failed

Description: National Clearing House code validation failed (ABA, Sort Code, BLZ, BSB)

Includes:
- TP/TN/FP/FN metrics with Precision/Recall/F1
- Feature documentation
- Clear PARTY vs BANK distinction

Usage:
    python test_8895_rules_v2.py --data-dir /path/to/ifml/data --limit 10000
    python test_8895_rules_v2.py --show-docs   # Feature documentation
    python test_8895_rules_v2.py --show-party  # Party reference guide
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

sys.path.insert(0, '/mnt/project')
from data_pipeline import IFMLDataPipeline

TARGET_CODE = '8895'

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
FEATURE DOCUMENTATION FOR 8895 (NCH Validation Failed)
================================================================================

8895 fires when: National Clearing House code validation failed (ABA, Sort Code, BLZ, BSB)

KEY FEATURES:
┌────────────────────────────────┬────────────────────────────────────────────┐
│ Feature                        │ Meaning                                    │
├────────────────────────────────┼────────────────────────────────────────────┤
│ has_invalid_iban               │ Message contains an invalid IBAN           │
│ bic_iban_mismatch_count        │ Count of parties where BIC country ≠ IBA... │
│ nch_valid                      │ NCH routing code is valid (False = valid... │
│ has_nch                        │ Party has NCH routing code                 │
│ total_nch_sources              │ Total NCH codes found in message           │
│ has_nch_issue                  │ NCH validation issue detected              │
│ invalid_nch_format             │ NCH code has invalid format                │
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
    Check if 8895 should fire based on extracted rules.
    Returns (should_fire, list_of_reasons)
    """
    reasons = []
    
    def get(feat, default=None):
        return features.get(feat, default)

    # 8895 = NCH (National Clearing House) validation failed
    # NCH codes: ABA (US), Sort Code (UK), BLZ (DE), BSB (AU), etc.
    
    # Rule 1: Has invalid IBAN (strong signal)
    if get('has_invalid_iban', False):
        reasons.append("has_invalid_iban=True")
    
    # Rule 2: BIC/IBAN country mismatch (triggers NCH fallback)
    if (get('bic_iban_mismatch_count', 0) or 0) > 0:
        reasons.append(f"bic_iban_mismatch={get('bic_iban_mismatch_count')}")
    
    # Rule 3: NCH validation failed explicitly
    for prefix in ['bnf_', 'cdt_', 'dbt_', 'orig_']:
        if get(f'{prefix}nch_valid') == False:
            party = prefix.upper().rstrip('_')
            reasons.append(f"{party}: NCH validation failed")
    
    # Rule 4: Has NCH code but invalid format
    for prefix in ['bnf_', 'cdt_', 'dbt_']:
        if get(f'{prefix}has_nch', False) and get(f'{prefix}nch_valid') == False:
            party = prefix.upper().rstrip('_')
            reasons.append(f"{party}: has NCH but invalid")
    
    # Rule 5: Multiple NCH sources (ambiguous routing)
    if (get('total_nch_sources', 0) or 0) > 1:
        reasons.append(f"multiple NCH sources={get('total_nch_sources')}")
    
    # Rule 6: Domestic with NCH issues
    if get('dbt_is_domestic', False) and get('has_nch_issue', False):
        reasons.append("DBT: domestic with NCH issue")
    
    # Rule 7: NCH code present but party also has IBAN mismatch
    for prefix in ['bnf_', 'cdt_', 'dbt_']:
        has_nch = (get(f'{prefix}nch_sources', 0) or 0) > 0
        has_mismatch = get(f'{prefix}bic_iban_match') == False
        if has_nch and has_mismatch:
            party = prefix.upper().rstrip('_')
            reasons.append(f"{party}: NCH + BIC/IBAN mismatch")
    
    # Rule 8: Invalid NCH format detected
    if get('invalid_nch_format', False):
        reasons.append("invalid_nch_format=True")

    
    return len(reasons) > 0, reasons


def get_debug_features(features: Dict) -> Dict:
    """Extract key features for debugging failures."""
    return {
        'has_invalid_iban': features.get('has_invalid_iban'),
        'bic_iban_mismatch_count': features.get('bic_iban_mismatch_count'),
        'total_nch_sources': features.get('total_nch_sources'),
        'bnf_nch_valid': features.get('bnf_nch_valid'),
        'cdt_nch_valid': features.get('cdt_nch_valid'),
        'dbt_nch_valid': features.get('dbt_nch_valid'),
        'dbt_is_domestic': features.get('dbt_is_domestic'),
        'cdt_is_domestic': features.get('cdt_is_domestic'),
        'bnf_nch_sources': features.get('bnf_nch_sources'),
        'bnf_bic_iban_match': features.get('bnf_bic_iban_match'),
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
    print(f"TEST RESULTS: {TARGET_CODE} - NCH Validation Failed")
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
