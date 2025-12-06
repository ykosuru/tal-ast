#!/usr/bin/env python3
"""
Test rules for 8894 - IBAN Validation Failed (V6 with Bank Directory Lookup)

V6 CHANGES:
- Integrates bank directory lookup (built from production data)
- Unknown bank codes trigger 8894 prediction
- Combined with existing BBAN validation rules

Usage:
    # First, build the bank directory (one-time)
    python build_bank_directory.py --data-dir ./prd_emts --output bank_directory.json
    
    # Then test with the directory
    python test_8894_rules_v6.py --data-dir ./prd_emts --directory bank_directory.json
"""

import argparse
import json
import re
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set

# Bank code positions by country (same as build_bank_directory.py)
BANK_CODE_POSITIONS = {
    'DE': (0, 8), 'FR': (0, 5), 'ES': (0, 4), 'IT': (1, 6), 'NL': (0, 4),
    'BE': (0, 3), 'AT': (0, 5), 'CH': (0, 5), 'LU': (0, 3), 'GB': (0, 4),
    'IE': (0, 4), 'FI': (0, 3), 'SE': (0, 3), 'NO': (0, 4), 'DK': (0, 4),
    'PL': (0, 8), 'CZ': (0, 4), 'SK': (0, 4), 'HU': (0, 3), 'RO': (0, 4),
    'BG': (0, 4), 'HR': (0, 7), 'SI': (0, 5), 'EE': (0, 2), 'LV': (0, 4),
    'LT': (0, 5), 'PT': (0, 4), 'GR': (0, 3), 'CY': (0, 3), 'MT': (0, 4),
    'SA': (0, 2), 'AE': (0, 3), 'TR': (0, 5), 'IL': (0, 3), 'QA': (0, 4),
    'KW': (0, 4), 'BH': (0, 4),
}

TARGET_CODE = '8894'


class BankDirectory:
    """Lookup for known-valid bank codes built from production data."""
    
    def __init__(self, directory_file: str = None):
        self.valid_codes: Dict[str, Set[str]] = {}
        self.potentially_invalid: Dict[str, Set[str]] = {}
        self.metadata = {}
        
        if directory_file:
            self.load(directory_file)
    
    def load(self, filepath: str):
        """Load bank directory from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.metadata = data.get('metadata', {})
        self.valid_codes = {
            country: set(codes)
            for country, codes in data.get('valid_bank_codes', {}).items()
        }
        self.potentially_invalid = {
            country: set(codes)
            for country, codes in data.get('potentially_invalid', {}).items()
        }
        
        total = sum(len(v) for v in self.valid_codes.values())
        print(f"Loaded bank directory: {total:,} valid codes from {len(self.valid_codes)} countries")
    
    def is_known_valid(self, country: str, bank_code: str) -> bool:
        """Check if bank code is in our known-valid set."""
        if not country or not bank_code:
            return True  # Can't validate, assume OK
        
        country = country.upper()
        
        if country not in self.valid_codes:
            return True  # Unknown country, can't validate
        
        return bank_code in self.valid_codes[country]
    
    def is_potentially_invalid(self, country: str, bank_code: str) -> bool:
        """Check if bank code was ONLY seen with 8894 errors."""
        if not country or not bank_code:
            return False
        
        country = country.upper()
        return bank_code in self.potentially_invalid.get(country, set())


def looks_like_iban(s: str) -> bool:
    """Check if string looks like an IBAN."""
    if not s or not isinstance(s, str):
        return False
    cleaned = s.upper().replace(' ', '').replace('-', '')
    if len(cleaned) < 15 or len(cleaned) > 34:
        return False
    return bool(re.match(r'^[A-Z]{2}[0-9]{2}[A-Z0-9]+$', cleaned))


def extract_bank_code(iban: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract country and bank code from IBAN."""
    if not iban:
        return None, None
    
    cleaned = iban.upper().replace(' ', '').replace('-', '')
    if len(cleaned) < 5:
        return None, None
    
    country = cleaned[:2]
    bban = cleaned[4:]
    
    positions = BANK_CODE_POSITIONS.get(country)
    if not positions:
        if len(bban) >= 4:
            return country, bban[:4]
        return country, None
    
    start, end = positions
    if len(bban) >= end:
        return country, bban[start:end]
    
    return country, None


def extract_ibans_from_obj(obj) -> List[str]:
    """Recursively extract all IBAN values from a JSON object."""
    ibans = []
    
    if isinstance(obj, dict):
        # Check ID fields
        if 'ID' in obj:
            id_field = obj['ID']
            if isinstance(id_field, dict):
                id_type = (id_field.get('Type') or id_field.get('@Type') or '').upper()
                id_text = id_field.get('text') or id_field.get('#text') or ''
                if id_type == 'IBAN' or looks_like_iban(str(id_text)):
                    ibans.append(str(id_text))
            elif isinstance(id_field, str) and looks_like_iban(id_field):
                ibans.append(id_field)
        
        # Check AcctIDInfo / AcctIDInf
        for acct_key in ['AcctIDInfo', 'AcctIDInf']:
            if acct_key in obj:
                acct = obj[acct_key]
                if isinstance(acct, dict):
                    acct_id = acct.get('ID', {})
                    if isinstance(acct_id, dict):
                        acct_type = (acct_id.get('Type') or acct_id.get('@Type') or '').upper()
                        acct_text = acct_id.get('text') or acct_id.get('#text') or ''
                        if acct_type == 'IBAN' or looks_like_iban(str(acct_text)):
                            ibans.append(str(acct_text))
        
        for value in obj.values():
            ibans.extend(extract_ibans_from_obj(value))
    
    elif isinstance(obj, list):
        for item in obj:
            ibans.extend(extract_ibans_from_obj(item))
    
    return ibans


def extract_error_codes(obj) -> List[str]:
    """Extract error codes from response object."""
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
                        # Include party hint if available
                        party = ''
                        for prefix in ['BNFBNK', 'BNPPTY', 'CDTPTY', 'DBTPTY', 'INTBNK']:
                            if info.startswith(prefix):
                                party = prefix
                                break
                        if party:
                            codes.append(f"{code}_{party}")
                        codes.append(str(code))
        
        if 'AuditTrail' in obj:
            codes.extend(extract_error_codes(obj['AuditTrail']))
        
        for value in obj.values():
            codes.extend(extract_error_codes(value))
    
    elif isinstance(obj, list):
        for item in obj:
            codes.extend(extract_error_codes(item))
    
    return codes


def check_iban_validation(iban: str) -> Dict[str, bool]:
    """
    Perform IBAN validation checks (format, checksum).
    Returns dict of validation results.
    """
    result = {
        'has_iban': False,
        'format_valid': False,
        'checksum_valid': False,
    }
    
    if not iban or not looks_like_iban(iban):
        return result
    
    result['has_iban'] = True
    cleaned = iban.upper().replace(' ', '').replace('-', '')
    
    # Check format (length for country)
    country = cleaned[:2]
    IBAN_LENGTHS = {
        'DE': 22, 'FR': 27, 'ES': 24, 'IT': 27, 'NL': 18, 'BE': 16, 'AT': 20,
        'CH': 21, 'GB': 22, 'IE': 22, 'PL': 28, 'PT': 25, 'SE': 24, 'NO': 15,
        'DK': 18, 'FI': 18, 'CZ': 24, 'SK': 24, 'HU': 28, 'RO': 24, 'BG': 22,
        'HR': 21, 'SI': 19, 'EE': 20, 'LV': 21, 'LT': 20, 'GR': 27, 'CY': 28,
        'MT': 31, 'LU': 20, 'SA': 24, 'AE': 23, 'TR': 26,
    }
    expected_len = IBAN_LENGTHS.get(country)
    if expected_len:
        result['format_valid'] = len(cleaned) == expected_len
    else:
        result['format_valid'] = 15 <= len(cleaned) <= 34
    
    # Check mod-97 checksum
    try:
        rearranged = cleaned[4:] + cleaned[:4]
        numeric = ''
        for char in rearranged:
            if char.isdigit():
                numeric += char
            elif char.isalpha():
                numeric += str(ord(char) - ord('A') + 10)
        result['checksum_valid'] = int(numeric) % 97 == 1
    except:
        result['checksum_valid'] = False
    
    return result


def check_rules_v6(ibans: List[str], bank_directory: BankDirectory) -> Tuple[bool, List[str]]:
    """
    Check if 8894 should fire based on V6 rules.
    
    Rules:
    1. IBAN checksum invalid (mod-97)
    2. IBAN format invalid (wrong length)
    3. Bank code not in known-valid directory
    4. Bank code only seen with 8894 errors (potentially invalid)
    """
    reasons = []
    
    for iban in ibans:
        if not iban:
            continue
        
        # Basic IBAN validation
        validation = check_iban_validation(iban)
        
        if validation['has_iban']:
            if not validation['checksum_valid']:
                reasons.append(f"IBAN checksum invalid: {iban[:10]}...")
            
            if not validation['format_valid']:
                reasons.append(f"IBAN format invalid: {iban[:10]}...")
        
        # Bank directory lookup
        country, bank_code = extract_bank_code(iban)
        
        if country and bank_code:
            if not bank_directory.is_known_valid(country, bank_code):
                reasons.append(f"Unknown bank code: {country}:{bank_code}")
            
            if bank_directory.is_potentially_invalid(country, bank_code):
                reasons.append(f"Potentially invalid bank code: {country}:{bank_code}")
    
    return len(reasons) > 0, reasons


def process_json_file(filepath: Path) -> List[Tuple[str, List[str], List[str]]]:
    """Process a JSON file, return list of (txn_id, ibans, error_codes)."""
    results = []
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        return results
    
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict):
                request = value.get('Request', {})
                response = value.get('Response', {})
                
                ibans = extract_ibans_from_obj(request)
                error_codes = extract_error_codes(response)
                
                results.append((key, ibans, error_codes))
    
    return results


def main():
    parser = argparse.ArgumentParser(description=f'Test {TARGET_CODE} rules V6 with bank directory')
    parser.add_argument('--data-dir', required=True, help='IFML JSON directory')
    parser.add_argument('--directory', required=True, help='Bank directory JSON file')
    parser.add_argument('--limit', type=int, default=0, help='Max transactions (0=all)')
    parser.add_argument('--show-fn', type=int, default=10, help='False negatives to show')
    parser.add_argument('--show-fp', type=int, default=10, help='False positives to show')
    
    args = parser.parse_args()
    
    # Load bank directory
    print(f"Loading bank directory from {args.directory}...")
    bank_dir = BankDirectory(args.directory)
    
    # Process data files
    print(f"\nScanning {args.data_dir} for JSON files...")
    data_path = Path(args.data_dir)
    json_files = list(data_path.glob("*.json"))
    print(f"Found {len(json_files)} files")
    
    # Classification
    tp_list, tn_list, fp_list, fn_list = [], [], [], []
    processed = 0
    
    # Track reasons for analysis
    reason_counts = defaultdict(int)
    
    for json_file in json_files:
        transactions = process_json_file(json_file)
        
        for txn_id, ibans, error_codes in transactions:
            if args.limit > 0 and processed >= args.limit:
                break
            
            processed += 1
            
            # Check actual result
            has_actual = any(TARGET_CODE in str(c) for c in error_codes)
            
            # Check predicted result
            predicted, reasons = check_rules_v6(ibans, bank_dir)
            
            # Track reasons
            for reason in reasons:
                reason_type = reason.split(':')[0]
                reason_counts[reason_type] += 1
            
            # Classify
            if predicted and has_actual:
                tp_list.append((txn_id, reasons, error_codes))
            elif not predicted and not has_actual:
                tn_list.append((txn_id,))
            elif predicted and not has_actual:
                fp_list.append((txn_id, reasons, error_codes[:5]))
            else:
                fn_list.append((txn_id, ibans[:3], error_codes[:5]))
        
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
    print(f"TEST RESULTS: {TARGET_CODE} - IBAN Validation Failed (V6 + Bank Directory)")
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
│                    {tp+fn:<6}    {tn+fp:<6}                                 │
└─────────────────────────────────────────────────────────────────────┘

┌────────────────┬──────────┬─────────────────────────────────────────┐
│ Metric         │ Value    │ Meaning                                 │
├────────────────┼──────────┼─────────────────────────────────────────┤
│ Precision      │ {precision*100:>6.2f}%  │ When we predict 8894, how often right?  │
│ Recall         │ {recall*100:>6.2f}%  │ What % of actual 8894 do we catch?      │
│ F1 Score       │ {f1*100:>6.2f}%  │ Harmonic mean                           │
│ Specificity    │ {specificity*100:>6.2f}%  │ True negative rate                      │
└────────────────┴──────────┴─────────────────────────────────────────┘
    """)
    
    # Reason breakdown
    print("PREDICTION TRIGGERS:")
    for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count:,}")
    
    # False negatives
    if fn_list:
        print(f"\n{'='*70}")
        print(f"FALSE NEGATIVES ({fn} total, showing {min(args.show_fn, fn)}):")
        print("="*70)
        print("MISSED: 8894 occurred but rules didn't predict it.\n")
        
        for i, (txn, ibans, codes) in enumerate(fn_list[:args.show_fn], 1):
            print(f"{i}. {txn}")
            print(f"   IBANs: {ibans}")
            
            # Show bank codes
            for iban in ibans:
                country, bank_code = extract_bank_code(iban)
                if country and bank_code:
                    known = "KNOWN" if bank_dir.is_known_valid(country, bank_code) else "UNKNOWN"
                    print(f"   Bank code: {country}:{bank_code} ({known})")
            
            relevant_codes = [c for c in codes if TARGET_CODE in str(c)]
            print(f"   Codes: {relevant_codes}")
            print()
    
    # False positives
    if fp_list:
        print(f"\n{'='*70}")
        print(f"FALSE POSITIVES ({fp} total, showing {min(args.show_fp, fp)}):")
        print("="*70)
        print("OVER-PREDICTED: Rules said 8894 but it didn't occur.\n")
        
        for i, (txn, reasons, codes) in enumerate(fp_list[:args.show_fp], 1):
            print(f"{i}. {txn}")
            print(f"   Trigger: {reasons}")
            print(f"   Actual codes: {codes}")
            print()
    
    # Summary
    print("="*70)
    print("SUMMARY")
    print("="*70)
    
    recall_icon = "✅" if recall >= 0.95 else "⚠️" if recall >= 0.5 else "❌"
    precision_icon = "✅" if precision >= 0.5 else "⚠️" if precision >= 0.1 else "ℹ️"
    
    print(f"""
    Recall:    {recall*100:>6.1f}%  {recall_icon}  (Target: ≥95%)
    Precision: {precision*100:>6.1f}%  {precision_icon}  (Acceptable: ≥10% for rare errors)
    F1 Score:  {f1*100:>6.1f}%
    
    V6 Strategy:
    - Bank directory lookup: Flag unknown bank codes
    - IBAN validation: Flag invalid checksum/format
    - Combined approach for comprehensive coverage
    
    Bank Directory Stats:
    - Valid bank codes: {sum(len(v) for v in bank_dir.valid_codes.values()):,}
    - Countries covered: {len(bank_dir.valid_codes)}
    """)


if __name__ == "__main__":
    main()
