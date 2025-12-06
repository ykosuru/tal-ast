#!/usr/bin/env python3
"""
Test rules for 8894 - IBAN Validation Failed (V7 - Missing/Unextractable IBAN)

V7 INSIGHT:
Looking at false negatives, most 8894 cases have:
- NO extractable IBAN (parser can't find it)
- OR IBAN in unexpected format/location

New hypothesis: 8894 fires when IBAN is EXPECTED but not properly provided.

V7 RULES:
1. Has BeneficiaryParty but NO extractable IBAN → risky
2. International payment to IBAN country but no IBAN found
3. Has IBAN-like field but it fails basic pattern matching
4. Account field exists but doesn't look like valid IBAN

Usage:
    python test_8894_rules_v7.py --data-dir ./prd_emts
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

# Countries that do NOT use IBAN (should not expect IBAN)
NON_IBAN_COUNTRIES = {
    'US', 'CA', 'AU', 'NZ', 'JP', 'CN', 'IN', 'SG', 'HK', 'TW', 'KR', 'TH',
    'MY', 'ID', 'PH', 'VN', 'MX', 'AR', 'CL', 'CO', 'PE', 'ZA', 'NG', 'KE',
    'EG', 'MA'  # Note: Some of these may have adopted IBAN
}


def looks_like_iban(s: str) -> bool:
    """Check if string looks like a valid IBAN pattern."""
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
    return True


def looks_like_malformed_iban(s: str) -> bool:
    """Check if string looks like an IBAN attempt but is malformed."""
    if not s or not isinstance(s, str):
        return False
    
    cleaned = s.upper().replace(' ', '').replace('-', '')
    
    # Too short or too long
    if len(cleaned) < 10 or len(cleaned) > 40:
        return False
    
    # Starts with 2 letters but something is wrong
    if re.match(r'^[A-Z]{2}', cleaned):
        # Has letters at start but:
        # - Wrong length
        # - Invalid check digits (not 2 digits after country)
        # - Contains invalid characters
        # - Country not in IBAN list
        
        country = cleaned[:2]
        
        # Country code that doesn't use IBAN
        if country in NON_IBAN_COUNTRIES:
            return True  # Trying to use IBAN format for non-IBAN country
        
        # Position 3-4 should be digits
        if len(cleaned) >= 4 and not cleaned[2:4].isdigit():
            return True  # Malformed check digits
        
        # Contains lowercase (original had lowercase)
        if s != s.upper() and re.match(r'^[a-zA-Z]{2}', s):
            return True  # Lowercase IBAN attempt
        
        # Has special characters in middle
        if re.search(r'[^A-Z0-9]', cleaned):
            return True
    
    return False


def extract_all_potential_ibans(obj, path="") -> List[dict]:
    """
    Extract all potential IBAN-like values from JSON, tracking where they came from.
    Returns list of {value, path, is_valid_iban, is_malformed}
    """
    results = []
    
    if isinstance(obj, dict):
        # Check ID fields
        for id_key in ['ID', 'Id']:
            if id_key in obj:
                id_field = obj[id_key]
                if isinstance(id_field, dict):
                    id_type = (id_field.get('Type') or id_field.get('@Type') or '').upper()
                    id_text = str(id_field.get('text') or id_field.get('#text') or '')
                    
                    if id_text:
                        results.append({
                            'value': id_text,
                            'path': f"{path}.{id_key}",
                            'declared_type': id_type,
                            'is_valid_iban': looks_like_iban(id_text),
                            'is_malformed': looks_like_malformed_iban(id_text),
                        })
                elif isinstance(id_field, str) and id_field:
                    results.append({
                        'value': id_field,
                        'path': f"{path}.{id_key}",
                        'declared_type': '',
                        'is_valid_iban': looks_like_iban(id_field),
                        'is_malformed': looks_like_malformed_iban(id_field),
                    })
        
        # Check AcctIDInfo / AcctIDInf
        for acct_key in ['AcctIDInfo', 'AcctIDInf']:
            if acct_key in obj:
                acct = obj[acct_key]
                if isinstance(acct, dict):
                    acct_id = acct.get('ID', {})
                    if isinstance(acct_id, dict):
                        acct_type = (acct_id.get('Type') or acct_id.get('@Type') or '').upper()
                        acct_text = str(acct_id.get('text') or acct_id.get('#text') or '')
                        if acct_text:
                            results.append({
                                'value': acct_text,
                                'path': f"{path}.{acct_key}.ID",
                                'declared_type': acct_type,
                                'is_valid_iban': looks_like_iban(acct_text),
                                'is_malformed': looks_like_malformed_iban(acct_text),
                            })
        
        # Recurse
        for key, value in obj.items():
            results.extend(extract_all_potential_ibans(value, f"{path}.{key}"))
    
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            results.extend(extract_all_potential_ibans(item, f"{path}[{i}]"))
    
    return results


def extract_countries(obj) -> Set[str]:
    """Extract all country codes from the message."""
    countries = set()
    
    if isinstance(obj, dict):
        for key in ['Country', 'MailingCountry', 'ResidenceCountry']:
            if key in obj:
                val = obj[key]
                if isinstance(val, str) and len(val) == 2:
                    countries.add(val.upper())
        
        # Check BIC for country (positions 5-6)
        for id_key in ['ID', 'Id']:
            if id_key in obj:
                id_field = obj[id_key]
                if isinstance(id_field, dict):
                    id_type = (id_field.get('Type') or id_field.get('@Type') or '').upper()
                    id_text = str(id_field.get('text') or id_field.get('#text') or '')
                    if id_type in ('BIC', 'S', 'SWIFT') and len(id_text) >= 6:
                        countries.add(id_text[4:6].upper())
        
        for value in obj.values():
            countries.update(extract_countries(value))
    
    elif isinstance(obj, list):
        for item in obj:
            countries.update(extract_countries(item))
    
    return countries


def has_beneficiary_party(obj) -> bool:
    """Check if message has BeneficiaryParty or similar."""
    if isinstance(obj, dict):
        for key in obj.keys():
            if 'Beneficiary' in key or 'BNPPTY' in key or 'Credit' in key:
                return True
        for value in obj.values():
            if has_beneficiary_party(value):
                return True
    elif isinstance(obj, list):
        for item in obj:
            if has_beneficiary_party(item):
                return True
    return False


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


def check_rules_v7(request_data: dict) -> Tuple[bool, List[str]]:
    """
    V7 Rules: Predict 8894 when IBAN is expected but not properly extractable.
    
    Rules:
    1. Beneficiary in IBAN country but no valid IBAN found
    2. ID field declared as IBAN but doesn't match pattern
    3. Malformed IBAN-like string detected
    4. Has account field but it's not a valid IBAN for IBAN-required country
    """
    reasons = []
    
    # Extract all potential IBANs
    potential_ibans = extract_all_potential_ibans(request_data)
    
    # Extract countries mentioned
    countries = extract_countries(request_data)
    
    # Check if beneficiary exists
    has_beneficiary = has_beneficiary_party(request_data)
    
    # Count valid vs invalid
    valid_ibans = [p for p in potential_ibans if p['is_valid_iban']]
    declared_ibans = [p for p in potential_ibans if p['declared_type'] == 'IBAN']
    malformed = [p for p in potential_ibans if p['is_malformed']]
    
    # Rule 1: Field declared as IBAN but value is not valid IBAN
    for p in declared_ibans:
        if not p['is_valid_iban']:
            reasons.append(f"Declared IBAN invalid: {p['value'][:15]}...")
    
    # Rule 2: Malformed IBAN-like strings
    for p in malformed:
        # Skip if we already caught it as declared IBAN
        if p not in declared_ibans:
            reasons.append(f"Malformed IBAN-like: {p['value'][:15]}...")
    
    # Rule 3: Beneficiary in IBAN country but no valid IBAN
    iban_countries_in_msg = countries & IBAN_COUNTRIES
    if iban_countries_in_msg and has_beneficiary and not valid_ibans:
        reasons.append(f"IBAN country {iban_countries_in_msg} but no valid IBAN")
    
    # Rule 4: Has potential account values that fail IBAN validation
    for p in potential_ibans:
        # If it looks like it could be an IBAN attempt but fails
        val = p['value'].upper().replace(' ', '').replace('-', '')
        if len(val) >= 15 and len(val) <= 34:
            if val[:2].isalpha() and not p['is_valid_iban']:
                # Starts with 2 letters, right length, but not valid IBAN
                if p not in declared_ibans and p not in malformed:
                    country = val[:2]
                    if country in IBAN_COUNTRIES:
                        reasons.append(f"Invalid IBAN format: {val[:15]}...")
    
    return len(reasons) > 0, reasons


def process_json_file(filepath: Path) -> List[Tuple[str, dict, dict, List[str]]]:
    """Process JSON file, return (txn_id, request, response, error_codes)."""
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
    parser = argparse.ArgumentParser(description=f'Test {TARGET_CODE} rules V7')
    parser.add_argument('--data-dir', required=True, help='IFML JSON directory')
    parser.add_argument('--limit', type=int, default=0, help='Max transactions')
    parser.add_argument('--show-fn', type=int, default=15, help='FN to show')
    parser.add_argument('--show-fp', type=int, default=10, help='FP to show')
    parser.add_argument('--show-tp', type=int, default=5, help='TP to show')
    
    args = parser.parse_args()
    
    print(f"Scanning {args.data_dir} for JSON files...")
    data_path = Path(args.data_dir)
    json_files = list(data_path.glob("*.json"))
    print(f"Found {len(json_files)} files")
    
    # Classification
    tp_list, tn_list, fp_list, fn_list = [], [], [], []
    processed = 0
    
    # Track triggers
    trigger_counts = defaultdict(int)
    
    for json_file in json_files:
        transactions = process_json_file(json_file)
        
        for txn_id, request, response, error_codes in transactions:
            if args.limit > 0 and processed >= args.limit:
                break
            
            processed += 1
            
            has_actual = any(TARGET_CODE in str(c) for c in error_codes)
            predicted, reasons = check_rules_v7(request)
            
            for r in reasons:
                trigger_type = r.split(':')[0]
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
    
    # Report
    print(f"\n{'='*70}")
    print(f"TEST RESULTS: {TARGET_CODE} - IBAN Validation Failed (V7)")
    print("="*70)
    
    print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│  CONFUSION MATRIX                                                   │
├─────────────────────────────────────────────────────────────────────┤
│                              Actual                                 │
│                      {TARGET_CODE}        Not {TARGET_CODE}                        │
│                  ┌──────────┬──────────┐                            │
│  Predicted       │          │          │                            │
│  {TARGET_CODE}          │ TP={tp:<5} │ FP={fp:<5} │  Pred Pos: {tp+fp:<7}   │
│                  ├──────────┼──────────┤                            │
│  Not {TARGET_CODE}      │ FN={fn:<5} │ TN={tn:<5} │  Pred Neg: {tn+fn:<7}   │
│                  └──────────┴──────────┘                            │
│                   Actual+     Actual-                               │
│                   {tp+fn:<7}    {tn+fp:<7}                              │
└─────────────────────────────────────────────────────────────────────┘

┌────────────────┬──────────┬─────────────────────────────────────────┐
│ Metric         │ Value    │ Meaning                                 │
├────────────────┼──────────┼─────────────────────────────────────────┤
│ Precision      │ {precision*100:>6.2f}%  │ When we predict, how often right?       │
│ Recall         │ {recall*100:>6.2f}%  │ What % of actual 8894 do we catch?      │
│ F1 Score       │ {f1*100:>6.2f}%  │ Harmonic mean                           │
│ Specificity    │ {specificity*100:>6.2f}%  │ True negative rate                      │
└────────────────┴──────────┴─────────────────────────────────────────┘
    """)
    
    # Triggers
    print("PREDICTION TRIGGERS:")
    for trigger, count in sorted(trigger_counts.items(), key=lambda x: -x[1]):
        print(f"  {trigger}: {count:,}")
    
    # True Positives (what worked)
    if tp_list and args.show_tp > 0:
        print(f"\n{'='*70}")
        print(f"TRUE POSITIVES ({tp} total, showing {min(args.show_tp, tp)}):")
        print("="*70)
        print("SUCCESS: We correctly predicted 8894\n")
        
        for i, (txn, reasons, codes, _) in enumerate(tp_list[:args.show_tp], 1):
            print(f"{i}. {txn}")
            print(f"   Trigger: {reasons}")
            print(f"   Codes: {[c for c in codes if TARGET_CODE in str(c)]}")
            print()
    
    # False Negatives
    if fn_list:
        print(f"\n{'='*70}")
        print(f"FALSE NEGATIVES ({fn} total, showing {min(args.show_fn, fn)}):")
        print("="*70)
        print("MISSED: 8894 occurred but rules didn't predict\n")
        
        for i, (txn, request, codes) in enumerate(fn_list[:args.show_fn], 1):
            print(f"{i}. {txn}")
            
            # Extract what we found
            potential = extract_all_potential_ibans(request)
            countries = extract_countries(request)
            has_bnf = has_beneficiary_party(request)
            
            print(f"   Has beneficiary: {has_bnf}")
            print(f"   Countries: {countries}")
            print(f"   Potential IBANs found: {len(potential)}")
            
            for p in potential[:3]:
                valid = "✓" if p['is_valid_iban'] else "✗"
                print(f"     {valid} {p['declared_type'] or 'unknown'}: {p['value'][:25]}...")
            
            relevant = [c for c in codes if TARGET_CODE in str(c)]
            print(f"   Actual codes: {relevant}")
            print()
    
    # False Positives
    if fp_list:
        print(f"\n{'='*70}")
        print(f"FALSE POSITIVES ({fp} total, showing {min(args.show_fp, fp)}):")
        print("="*70)
        print("OVER-PREDICTED: Rules said 8894 but didn't occur\n")
        
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
    Recall:    {recall*100:>6.1f}%  {recall_icon}  (Target: ≥95%)
    Precision: {precision*100:>6.1f}%  {precision_icon}  (Acceptable: ≥10%)
    F1 Score:  {f1*100:>6.1f}%
    
    V7 Strategy: Predict 8894 when IBAN is expected but not properly provided
    - Declared IBAN but invalid format
    - Malformed IBAN-like strings
    - IBAN country in message but no valid IBAN found
    """)


if __name__ == "__main__":
    main()
