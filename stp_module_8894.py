#!/usr/bin/env python3
"""
Test rules for 8894 - IBAN Validation Failed (V8 - BIC Exclusion + Focused Rules)

V8 FIXES:
- EXCLUDE BICs from "malformed IBAN" detection (caused 1M+ FP!)
- Focus on specific pattern: IBAN country but plain account number provided
- Target Costa Rica (CR) and similar patterns seen in FN analysis

Key insight from V7:
- BICs (8 or 11 chars, 4 letters + 2 letter country) were being flagged as malformed IBANs
- "IBAN country but no IBAN" was too broad
- Real pattern: CR (Costa Rica) expects IBAN but gets plain account like "256074974"

Usage:
    python test_8894_rules_v8.py --data-dir ./prd_emts
"""

import argparse
import json
import re
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set

TARGET_CODE = '8894'

# Countries that use IBAN
IBAN_COUNTRIES = {
    'AD', 'AE', 'AL', 'AT', 'AZ', 'BA', 'BE', 'BG', 'BH', 'BR', 'BY', 'CH',
    'CR', 'CY', 'CZ', 'DE', 'DK', 'DO', 'EE', 'ES', 'FI', 'FO', 'FR', 'GB',
    'GE', 'GI', 'GL', 'GR', 'GT', 'HR', 'HU', 'IE', 'IL', 'IQ', 'IS', 'IT',
    'JO', 'KW', 'KZ', 'LB', 'LI', 'LT', 'LU', 'LV', 'MC', 'MD', 'ME', 'MK',
    'MR', 'MT', 'MU', 'NL', 'NO', 'PK', 'PL', 'PS', 'PT', 'QA', 'RO', 'RS',
    'SA', 'SC', 'SE', 'SI', 'SK', 'SM', 'TL', 'TN', 'TR', 'UA', 'VA', 'VG', 'XK'
}

# IBAN lengths by country
IBAN_LENGTHS = {
    'CR': 22, 'DE': 22, 'FR': 27, 'ES': 24, 'IT': 27, 'GB': 22, 'NL': 18,
    'BE': 16, 'AT': 20, 'CH': 21, 'PL': 28, 'PT': 25, 'SE': 24, 'NO': 15,
    'DK': 18, 'FI': 18, 'IE': 22, 'LU': 20, 'GR': 27, 'CZ': 24, 'HU': 28,
    'RO': 24, 'BG': 22, 'HR': 21, 'SI': 19, 'SK': 24, 'LT': 20, 'LV': 21,
    'EE': 20, 'CY': 28, 'MT': 31, 'TR': 26, 'SA': 24, 'AE': 23, 'IL': 23,
    'JO': 30, 'KW': 30, 'BH': 22, 'QA': 29, 'DO': 28, 'GT': 28, 'BR': 29,
    'MU': 30, 'PK': 24, 'PS': 29, 'VG': 24, 'TN': 24, 'MD': 24, 'MK': 19,
    'AL': 28, 'AZ': 28, 'BA': 20, 'BY': 28, 'GE': 22, 'GI': 23, 'GL': 18,
    'FO': 18, 'IS': 26, 'IQ': 23, 'LB': 28, 'LI': 21, 'MC': 27, 'ME': 22,
    'MR': 27, 'RS': 22, 'SC': 31, 'SM': 27, 'TL': 23, 'UA': 29, 'VA': 22, 'XK': 20,
}


def looks_like_bic(s: str) -> bool:
    """
    Check if string looks like a BIC/SWIFT code.
    BIC format: 4 letters (bank) + 2 letters (country) + 2 alphanum (location) + optional 3 alphanum (branch)
    Length: 8 or 11 characters
    """
    if not s or not isinstance(s, str):
        return False
    
    cleaned = s.upper().strip()
    
    # Must be 8 or 11 characters
    if len(cleaned) not in (8, 11):
        return False
    
    # First 6 must be letters (bank code + country)
    if not cleaned[:6].isalpha():
        return False
    
    # Rest must be alphanumeric
    if not cleaned[6:].isalnum():
        return False
    
    return True


def looks_like_iban(s: str) -> bool:
    """Check if string looks like a valid IBAN."""
    if not s or not isinstance(s, str):
        return False
    
    cleaned = s.upper().replace(' ', '').replace('-', '')
    
    if len(cleaned) < 15 or len(cleaned) > 34:
        return False
    
    # Must start with 2 letters (country) + 2 digits (check)
    if not re.match(r'^[A-Z]{2}[0-9]{2}[A-Z0-9]+$', cleaned):
        return False
    
    # Country must be valid IBAN country
    country = cleaned[:2]
    if country not in IBAN_COUNTRIES:
        return False
    
    # Check expected length for country
    expected_len = IBAN_LENGTHS.get(country)
    if expected_len and len(cleaned) != expected_len:
        return False
    
    return True


def is_plain_account_number(s: str) -> bool:
    """Check if string looks like a plain account number (not IBAN)."""
    if not s or not isinstance(s, str):
        return False
    
    cleaned = s.strip()
    
    # Pure digits, reasonable length for account
    if cleaned.isdigit() and 5 <= len(cleaned) <= 20:
        return True
    
    # Alphanumeric but doesn't start with 2 letters + 2 digits (not IBAN pattern)
    if cleaned.isalnum() and len(cleaned) >= 5:
        if not re.match(r'^[A-Z]{2}[0-9]{2}', cleaned.upper()):
            return True
    
    return False


def extract_beneficiary_info(obj, path="") -> dict:
    """
    Extract beneficiary-specific information.
    Returns dict with country, account_value, account_type, has_iban
    """
    result = {
        'country': None,
        'account_value': None,
        'account_type': None,
        'has_valid_iban': False,
        'has_plain_account': False,
    }
    
    if not isinstance(obj, dict):
        return result
    
    # Look for beneficiary party sections
    for key in obj.keys():
        key_upper = key.upper()
        if 'BENEFICIARY' in key_upper or 'BNPPTY' in key_upper or 'CREDIT' in key_upper:
            party_data = obj[key]
            if isinstance(party_data, list):
                party_data = party_data[0] if party_data else {}
            if isinstance(party_data, dict):
                # Extract recursively from this party
                info = extract_party_details(party_data)
                if info['country']:
                    result['country'] = info['country']
                if info['account_value']:
                    result['account_value'] = info['account_value']
                    result['account_type'] = info['account_type']
                    result['has_valid_iban'] = looks_like_iban(info['account_value'])
                    result['has_plain_account'] = is_plain_account_number(info['account_value'])
    
    # Recurse into nested structures
    for value in obj.values():
        if isinstance(value, dict):
            nested = extract_beneficiary_info(value)
            if nested['country'] and not result['country']:
                result['country'] = nested['country']
            if nested['account_value'] and not result['account_value']:
                result['account_value'] = nested['account_value']
                result['account_type'] = nested['account_type']
                result['has_valid_iban'] = nested['has_valid_iban']
                result['has_plain_account'] = nested['has_plain_account']
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    nested = extract_beneficiary_info(item)
                    if nested['country'] and not result['country']:
                        result['country'] = nested['country']
                    if nested['account_value'] and not result['account_value']:
                        result['account_value'] = nested['account_value']
                        result['account_type'] = nested['account_type']
                        result['has_valid_iban'] = nested['has_valid_iban']
                        result['has_plain_account'] = nested['has_plain_account']
    
    return result


def extract_party_details(party: dict) -> dict:
    """Extract country and account from a party dict."""
    result = {'country': None, 'account_value': None, 'account_type': None}
    
    if not isinstance(party, dict):
        return result
    
    # Get country
    for key in ['Country', 'MailingCountry', 'ResidenceCountry']:
        if key in party:
            val = party[key]
            if isinstance(val, str) and len(val) == 2:
                result['country'] = val.upper()
                break
    
    # Get account/ID
    for id_key in ['ID', 'Id']:
        if id_key in party:
            id_field = party[id_key]
            if isinstance(id_field, dict):
                id_type = (id_field.get('Type') or id_field.get('@Type') or '').upper()
                id_text = str(id_field.get('text') or id_field.get('#text') or '')
                
                # Skip BICs
                if id_type in ('BIC', 'S', 'SWIFT') or looks_like_bic(id_text):
                    continue
                
                if id_text:
                    result['account_value'] = id_text
                    result['account_type'] = id_type
            elif isinstance(id_field, str) and not looks_like_bic(id_field):
                result['account_value'] = id_field
    
    # Check AcctIDInfo
    for acct_key in ['AcctIDInfo', 'AcctIDInf']:
        if acct_key in party:
            acct = party[acct_key]
            if isinstance(acct, dict):
                acct_id = acct.get('ID', {})
                if isinstance(acct_id, dict):
                    acct_type = (acct_id.get('Type') or acct_id.get('@Type') or '').upper()
                    acct_text = str(acct_id.get('text') or acct_id.get('#text') or '')
                    if acct_text and not looks_like_bic(acct_text):
                        result['account_value'] = acct_text
                        result['account_type'] = acct_type
    
    # Recurse into nested party info
    for nested_key in ['BasicPartyInfo', 'BasicPartyInf', 'BasicPartyBankInfo', 'BasicPartyBankInf']:
        if nested_key in party:
            nested = extract_party_details(party[nested_key])
            if nested['country'] and not result['country']:
                result['country'] = nested['country']
            if nested['account_value'] and not result['account_value']:
                result['account_value'] = nested['account_value']
                result['account_type'] = nested['account_type']
    
    return result


def extract_all_countries(obj) -> Set[str]:
    """Extract all country codes from message."""
    countries = set()
    
    if isinstance(obj, dict):
        for key in ['Country', 'MailingCountry', 'ResidenceCountry']:
            if key in obj:
                val = obj[key]
                if isinstance(val, str) and len(val) == 2:
                    countries.add(val.upper())
        
        # Extract from BIC (positions 5-6)
        for id_key in ['ID', 'Id']:
            if id_key in obj:
                id_field = obj[id_key]
                if isinstance(id_field, dict):
                    id_text = str(id_field.get('text') or id_field.get('#text') or '')
                    if looks_like_bic(id_text) and len(id_text) >= 6:
                        countries.add(id_text[4:6].upper())
        
        for value in obj.values():
            countries.update(extract_all_countries(value))
    
    elif isinstance(obj, list):
        for item in obj:
            countries.update(extract_all_countries(item))
    
    return countries


def extract_error_codes(obj) -> List[str]:
    """Extract error codes from response."""
    codes = []
    
    if isinstance(obj, dict):
        if 'MsgStatus' in obj:
            status_list = obj['MsgStatus']
            if isinstance(status_list, dict):
                status_list = [status_list]
            for status in status_list:
                if isinstance(status, dict):
                    code = status.get('Code')
                    info = status.get('InformationalData', '')
                    if code:
                        for prefix in ['BNFBNK', 'BNPPTY', 'CDTPTY', 'DBTPTY', 'INTBNK']:
                            if info.startswith(prefix):
                                codes.append(f"{code}_{prefix}")
                                break
                        codes.append(str(code))
        
        if 'AuditTrail' in obj:
            codes.extend(extract_error_codes(obj['AuditTrail']))
        
        for value in obj.values():
            codes.extend(extract_error_codes(value))
    elif isinstance(obj, list):
        for item in obj:
            codes.extend(extract_error_codes(item))
    
    return codes


def check_rules_v8(request_data: dict) -> Tuple[bool, List[str]]:
    """
    V8 Rules - Focused on IBAN country + plain account pattern.
    
    Primary rule:
    - Beneficiary is in IBAN country (like CR, DE, GB)
    - But account provided is a plain number, not a valid IBAN
    """
    reasons = []
    
    # Extract beneficiary info
    bnf_info = extract_beneficiary_info(request_data)
    
    # Extract all countries for context
    all_countries = extract_all_countries(request_data)
    
    # Rule 1: Beneficiary in IBAN country but account is NOT valid IBAN
    if bnf_info['country'] and bnf_info['country'] in IBAN_COUNTRIES:
        if bnf_info['account_value']:
            if not bnf_info['has_valid_iban']:
                # Has account but it's not a valid IBAN
                if bnf_info['has_plain_account']:
                    reasons.append(f"IBAN country {bnf_info['country']} but plain account: {bnf_info['account_value'][:15]}...")
                elif bnf_info['account_type'] == 'IBAN':
                    # Declared as IBAN but doesn't validate
                    reasons.append(f"Declared IBAN invalid for {bnf_info['country']}: {bnf_info['account_value'][:15]}...")
    
    # Rule 2: Check for Costa Rica specifically (high signal from FN analysis)
    if 'CR' in all_countries:
        if bnf_info['account_value']:
            if not looks_like_iban(bnf_info['account_value']):
                if not looks_like_bic(bnf_info['account_value']):
                    # CR in message, has account, but not IBAN format
                    if f"IBAN country CR" not in str(reasons):
                        reasons.append(f"CR (Costa Rica) detected but no valid IBAN")
    
    return len(reasons) > 0, reasons


def process_json_file(filepath: Path) -> List[Tuple[str, dict, dict, List[str]]]:
    """Process JSON file."""
    results = []
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return results
    
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict):
                request = value.get('Request', {})
                response = value.get('Response', {})
                error_codes = extract_error_codes(response)
                results.append((key, request, response, error_codes))
    
    return results


def main():
    parser = argparse.ArgumentParser(description=f'Test {TARGET_CODE} rules V8')
    parser.add_argument('--data-dir', required=True, help='IFML JSON directory')
    parser.add_argument('--limit', type=int, default=0, help='Max transactions')
    parser.add_argument('--show-fn', type=int, default=15, help='FN to show')
    parser.add_argument('--show-fp', type=int, default=10, help='FP to show')
    parser.add_argument('--show-tp', type=int, default=10, help='TP to show')
    
    args = parser.parse_args()
    
    print(f"Scanning {args.data_dir}...")
    data_path = Path(args.data_dir)
    json_files = list(data_path.glob("*.json"))
    print(f"Found {len(json_files)} files")
    
    tp_list, tn_list, fp_list, fn_list = [], [], [], []
    processed = 0
    trigger_counts = defaultdict(int)
    
    for json_file in json_files:
        transactions = process_json_file(json_file)
        
        for txn_id, request, response, error_codes in transactions:
            if args.limit > 0 and processed >= args.limit:
                break
            
            processed += 1
            
            has_actual = any(TARGET_CODE in str(c) for c in error_codes)
            predicted, reasons = check_rules_v8(request)
            
            for r in reasons:
                trigger_type = r.split(':')[0] if ':' in r else r.split(' but')[0]
                trigger_counts[trigger_type] += 1
            
            if predicted and has_actual:
                tp_list.append((txn_id, reasons, error_codes, request))
            elif not predicted and not has_actual:
                tn_list.append((txn_id,))
            elif predicted and not has_actual:
                fp_list.append((txn_id, reasons, error_codes[:5]))
            else:
                fn_list.append((txn_id, request, error_codes))
        
        if args.limit > 0 and processed >= args.limit:
            break
    
    # Metrics
    tp, tn, fp, fn = len(tp_list), len(tn_list), len(fp_list), len(fn_list)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"\n{'='*70}")
    print(f"TEST RESULTS: {TARGET_CODE} - IBAN Validation Failed (V8)")
    print("="*70)
    
    print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│  CONFUSION MATRIX                                                   │
├─────────────────────────────────────────────────────────────────────┤
│                      {TARGET_CODE}        Not {TARGET_CODE}                        │
│                  ┌──────────┬──────────┐                            │
│  Predicted {TARGET_CODE}  │ TP={tp:<5} │ FP={fp:<5} │  Pred Pos: {tp+fp:<7}   │
│                  ├──────────┼──────────┤                            │
│  Not Predicted   │ FN={fn:<5} │ TN={tn:<5} │  Pred Neg: {tn+fn:<7}   │
│                  └──────────┴──────────┘                            │
│                   Actual+     Actual-                               │
│                   {tp+fn:<7}    {tn+fp:<7}                              │
└─────────────────────────────────────────────────────────────────────┘

│ Precision      │ {precision*100:>6.2f}%  │ When we predict, how often right?       │
│ Recall         │ {recall*100:>6.2f}%  │ What % of actual 8894 do we catch?      │
│ F1 Score       │ {f1*100:>6.2f}%  │                                         │
│ Specificity    │ {specificity*100:>6.2f}%  │ True negative rate                      │
    """)
    
    print("\nPREDICTION TRIGGERS:")
    for trigger, count in sorted(trigger_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"  {trigger}: {count:,}")
    
    # True Positives
    if tp_list and args.show_tp > 0:
        print(f"\n{'='*70}")
        print(f"TRUE POSITIVES ({tp} total, showing {min(args.show_tp, tp)}):")
        print("="*70)
        
        for i, (txn, reasons, codes, req) in enumerate(tp_list[:args.show_tp], 1):
            bnf = extract_beneficiary_info(req)
            print(f"{i}. {txn}")
            print(f"   Trigger: {reasons}")
            print(f"   Beneficiary country: {bnf['country']}")
            print(f"   Account: {bnf['account_value'][:20] if bnf['account_value'] else 'None'}...")
            print()
    
    # False Negatives
    if fn_list and args.show_fn > 0:
        print(f"\n{'='*70}")
        print(f"FALSE NEGATIVES ({fn} total, showing {min(args.show_fn, fn)}):")
        print("="*70)
        
        for i, (txn, request, codes) in enumerate(fn_list[:args.show_fn], 1):
            bnf = extract_beneficiary_info(request)
            countries = extract_all_countries(request)
            
            print(f"{i}. {txn}")
            print(f"   Countries: {countries}")
            print(f"   Beneficiary country: {bnf['country']}")
            print(f"   Account: {bnf['account_value'][:25] if bnf['account_value'] else 'None'}...")
            print(f"   Is valid IBAN: {bnf['has_valid_iban']}")
            print(f"   Codes: {[c for c in codes if TARGET_CODE in str(c)]}")
            print()
    
    # False Positives
    if fp_list and args.show_fp > 0:
        print(f"\n{'='*70}")
        print(f"FALSE POSITIVES ({fp} total, showing {min(args.show_fp, fp)}):")
        print("="*70)
        
        for i, (txn, reasons, codes) in enumerate(fp_list[:args.show_fp], 1):
            print(f"{i}. {txn}")
            print(f"   Trigger: {reasons}")
            print(f"   Actual: {codes}")
            print()
    
    # Summary
    print("="*70)
    print("SUMMARY")
    print("="*70)
    
    recall_icon = "✅" if recall >= 0.95 else "⚠️" if recall >= 0.5 else "❌"
    precision_icon = "✅" if precision >= 0.5 else "⚠️" if precision >= 0.1 else "ℹ️"
    
    print(f"""
    Recall:    {recall*100:>6.1f}%  {recall_icon}
    Precision: {precision*100:>6.1f}%  {precision_icon}
    
    V8 Strategy:
    - Exclude BICs from malformed IBAN detection
    - Focus: IBAN country (like CR) but plain account number provided
    - Target specific high-signal patterns
    """)


if __name__ == "__main__":
    main()
