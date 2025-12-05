#!/usr/bin/env python3
"""
Test extracted RandomForest rules for 8852 against actual IFMLs.
8852 = "Incorrect length of attribute"

Usage:
    python test_8852_rules.py --data-dir /path/to/ifml/data --limit 1000
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add project path for imports
sys.path.insert(0, '/mnt/project')

from data_pipeline import IFMLDataPipeline

# IBAN lengths by country (from ifml_parser.py)
IBAN_LENGTHS = {
    'AL': 28, 'AD': 24, 'AT': 20, 'AZ': 28, 'BH': 22, 'BY': 28, 'BE': 16, 'BA': 20,
    'BR': 29, 'BG': 22, 'CR': 22, 'HR': 21, 'CY': 28, 'CZ': 24, 'DK': 18, 'DO': 28,
    'TL': 23, 'EE': 20, 'FO': 18, 'FI': 18, 'FR': 27, 'GE': 22, 'DE': 22, 'GI': 23,
    'GR': 27, 'GL': 18, 'GT': 28, 'HU': 28, 'IS': 26, 'IQ': 23, 'IE': 22, 'IL': 23,
    'IT': 27, 'JO': 30, 'KZ': 20, 'XK': 20, 'KW': 30, 'LV': 21, 'LB': 28, 'LI': 21,
    'LT': 20, 'LU': 20, 'MK': 19, 'MT': 31, 'MR': 27, 'MU': 30, 'MC': 27, 'MD': 24,
    'ME': 22, 'NL': 18, 'NO': 15, 'PK': 24, 'PS': 29, 'PL': 28, 'PT': 25, 'QA': 29,
    'RO': 24, 'SM': 27, 'SA': 24, 'RS': 22, 'SC': 31, 'SK': 24, 'SI': 19, 'ES': 24,
    'SE': 24, 'CH': 21, 'TN': 24, 'TR': 26, 'AE': 23, 'GB': 22, 'VA': 22, 'VG': 24,
    'UA': 29,
}

# Field length limits
FIELD_LIMITS = {
    'name': 140,           # Party name max
    'address_line': 35,    # Per address line
    'account': 34,         # Account/IBAN max
    'bic': 11,             # BIC max (8 or 11)
    'reference': 35,       # Reference field
}


def check_iban_length(iban_value: str, party: str) -> Tuple[bool, str]:
    """
    Check if IBAN length matches expected length for its country.
    Returns (is_wrong_length, reason)
    """
    if not iban_value:
        return False, ""
    
    # Clean IBAN
    iban = iban_value.upper().replace(' ', '').replace('-', '')
    
    if len(iban) < 2:
        return False, ""
    
    country = iban[:2]
    expected_len = IBAN_LENGTHS.get(country)
    actual_len = len(iban)
    
    if expected_len:
        if actual_len != expected_len:
            return True, f"{party}: IBAN length {actual_len} != expected {expected_len} for {country}"
    else:
        # Unknown country - check general IBAN limits (15-34)
        if actual_len < 15 or actual_len > 34:
            return True, f"{party}: IBAN length {actual_len} outside valid range (15-34)"
    
    return False, ""


def check_8852_rf_rules(features: Dict) -> Tuple[bool, List[str]]:
    """
    Check if 8852 should fire based on RF-extracted rules.
    8852 = "Incorrect length of attribute"
    
    Now includes IBAN length validation by country.
    
    Returns (should_fire, reasons)
    """
    reasons = []
    
    def get(feat, default=None):
        if feat in features:
            return features[feat]
        return default
    
    matches = 0
    
    # -------------------------------------------------------------------------
    # Rule 1: IBAN length mismatch by country (NEW - direct check)
    # -------------------------------------------------------------------------
    for prefix in ['orig_', 'cdt_', 'dbt_', 'bnf_', 'intm_']:
        has_iban = get(f'{prefix}has_iban', False)
        iban_country = get(f'{prefix}iban_country')
        account_length = get(f'{prefix}account_length', 0)
        
        if has_iban and iban_country and account_length:
            expected_len = IBAN_LENGTHS.get(iban_country)
            if expected_len and account_length != expected_len:
                party = prefix.rstrip('_').upper()
                reasons.append(f"{party}: IBAN length {account_length} != {expected_len} for {iban_country}")
                matches += 1
            elif not expected_len and (account_length < 15 or account_length > 34):
                # Unknown country but outside valid range
                party = prefix.rstrip('_').upper()
                reasons.append(f"{party}: IBAN length {account_length} outside range (15-34)")
                matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 2: has_iban but iban_valid_format=False (may indicate length issue)
    # -------------------------------------------------------------------------
    for prefix in ['orig_', 'cdt_', 'dbt_', 'bnf_', 'intm_']:
        has_iban = get(f'{prefix}has_iban', False)
        iban_valid_format = get(f'{prefix}iban_valid_format', True)
        
        if has_iban and not iban_valid_format:
            party = prefix.rstrip('_').upper()
            reasons.append(f"{party}: has_iban but iban_valid_format=False (possible length issue)")
            matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 3: Account length > 34 (max IBAN length)
    # -------------------------------------------------------------------------
    for prefix in ['orig_', 'cdt_', 'dbt_', 'bnf_', 'intm_']:
        account_length = get(f'{prefix}account_length', 0)
        has_iban = get(f'{prefix}has_iban', False)
        
        # Only flag if not already an IBAN (avoid double counting)
        if account_length and account_length > 34 and not has_iban:
            party = prefix.rstrip('_').upper()
            reasons.append(f"{party}: account_length={account_length} > 34")
            matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 4: BIC length issues (should be 8 or 11)
    # -------------------------------------------------------------------------
    for prefix in ['orig_', 'cdt_', 'dbt_', 'bnf_', 'intm_', 'send_']:
        bic_length = get(f'{prefix}bic_length', 0)
        if bic_length and bic_length not in [0, 8, 11]:
            party = prefix.rstrip('_').upper()
            reasons.append(f"{party}: bic_length={bic_length} (not 8 or 11)")
            matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 5: Account needs length fix (from parser)
    # -------------------------------------------------------------------------
    for prefix in ['orig_', 'cdt_', 'dbt_', 'bnf_', 'intm_']:
        needs_fix = get(f'{prefix}account_needs_length_fix', False)
        if needs_fix:
            party = prefix.rstrip('_').upper()
            reasons.append(f"{party}: account_needs_length_fix=True")
            matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 6: RF pattern - originator with name (100% ratio in RF)
    # This is a correlation, not direct cause - lower priority
    # -------------------------------------------------------------------------
    orig_present = get('orig_present', False)
    orig_has_name = get('orig_has_name', False)
    bnf_is_international = get('bnf_is_international', False)
    
    # Only use this pattern if we didn't find direct length issues
    if matches == 0 and orig_present and orig_has_name and bnf_is_international:
        reasons.append("RF pattern: orig with name + international (correlation)")
        matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 7: Compound ID parts (may cause length issues when concatenated)
    # -------------------------------------------------------------------------
    for prefix in ['orig_', 'cdt_', 'dbt_', 'bnf_', 'intm_']:
        compound = get(f'{prefix}id_compound_parts', False)
        if compound:
            party = prefix.rstrip('_').upper()
            reasons.append(f"{party}: id_compound_parts=True")
            matches += 1
    
    should_fire = matches > 0
    return should_fire, reasons


def main():
    parser = argparse.ArgumentParser(description='Test RF 8852 rules against IFMLs')
    parser.add_argument('--data-dir', required=True, help='Directory with IFML JSON files')
    parser.add_argument('--limit', type=int, default=5000, help='Max records to process')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show details')
    parser.add_argument('--show-failures', type=int, default=20, help='Number of failures to show')
    
    args = parser.parse_args()
    
    print("Loading IFML data using IFMLDataPipeline...")
    pipeline = IFMLDataPipeline()
    
    data_path = Path(args.data_dir)
    
    if data_path.is_file():
        loaded = pipeline.load_single_file(str(data_path))
    else:
        loaded = pipeline.load_directory(str(data_path), "*.json")
    
    print(f"Loaded {loaded} total records")
    print(f"Pipeline has {len(pipeline.records)} records")
    
    # Track results
    successes = []
    failures = []
    total_8852 = 0
    processed = 0
    
    for record in pipeline.records:
        if processed >= args.limit:
            break
        
        actual_codes = record.error_codes_only
        composite_codes = record.composite_codes or []
        
        has_8852 = any('8852' in str(c) for c in actual_codes + composite_codes)
        
        if not has_8852:
            continue
        
        total_8852 += 1
        processed += 1
        
        txn_id = record.transaction_id
        features = record.request_features
        
        should_fire, reasons = check_8852_rf_rules(features)
        
        if should_fire:
            successes.append((txn_id, reasons))
            if args.verbose:
                print(f"✅ {txn_id}: {reasons[0]}")
        else:
            debug_info = {
                # IBAN length info
                'bnf_iban_country': features.get('bnf_iban_country'),
                'bnf_account_length': features.get('bnf_account_length'),
                'bnf_has_iban': features.get('bnf_has_iban'),
                'bnf_iban_valid_format': features.get('bnf_iban_valid_format'),
                'cdt_iban_country': features.get('cdt_iban_country'),
                'cdt_account_length': features.get('cdt_account_length'),
                'cdt_has_iban': features.get('cdt_has_iban'),
                'cdt_iban_valid_format': features.get('cdt_iban_valid_format'),
                # BIC length
                'bnf_bic_length': features.get('bnf_bic_length'),
                'send_bic_length': features.get('send_bic_length'),
                # RF patterns
                'orig_present': features.get('orig_present'),
                'orig_has_name': features.get('orig_has_name'),
                'bnf_is_international': features.get('bnf_is_international'),
            }
            failures.append((txn_id, composite_codes, debug_info))
            if args.verbose:
                print(f"❌ {txn_id}")
    
    # =========================================================================
    # REPORT
    # =========================================================================
    print("\n" + "="*70)
    print("TEST RESULTS: 8852 Rule Validation")
    print("="*70)
    
    print(f"\nTotal IFMLs with 8852: {total_8852}")
    print(f"Successes (rule matched): {len(successes)}")
    print(f"Failures (rule missed): {len(failures)}")
    
    if total_8852 > 0:
        success_rate = len(successes) / total_8852 * 100
        print(f"\n🎯 Success Rate: {success_rate:.1f}%")
    
    # Show success patterns
    if successes:
        print(f"\n{'='*70}")
        print(f"SUCCESS PATTERNS:")
        print("="*70)
        
        reason_counts = {}
        for txn_id, reasons in successes:
            for reason in reasons:
                key = reason.split(':')[0] if ':' in reason else reason[:50]
                reason_counts[key] = reason_counts.get(key, 0) + 1
        
        for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
            pct = count / len(successes) * 100
            print(f"  {reason}: {count} ({pct:.1f}%)")
    
    # Show failures
    if failures:
        print(f"\n{'='*70}")
        print(f"FAILURES ({len(failures)} total, showing first {args.show_failures}):")
        print("="*70)
        
        for i, (txn_id, codes, debug_info) in enumerate(failures[:args.show_failures]):
            print(f"\n{i+1}. TxnID: {txn_id}")
            print(f"   Actual codes: {[c for c in codes if '8852' in str(c)]}")
            print(f"   Features: {debug_info}")
        
        if len(failures) > args.show_failures:
            print(f"\n... and {len(failures) - args.show_failures} more failures")
        
        # Analyze failure patterns
        print(f"\n{'='*70}")
        print("FAILURE PATTERN ANALYSIS:")
        print("="*70)
        
        failure_patterns = {
            'bnf_has_iban_true': 0,
            'bnf_iban_valid_format_false': 0,
            'cdt_has_iban_true': 0,
            'cdt_iban_valid_format_false': 0,
            'orig_present_true': 0,
            'orig_has_name_true': 0,
            'bnf_is_international_true': 0,
            'has_account_length': 0,
        }
        
        for txn_id, codes, debug in failures:
            if debug.get('bnf_has_iban'):
                failure_patterns['bnf_has_iban_true'] += 1
            if debug.get('bnf_has_iban') and not debug.get('bnf_iban_valid_format'):
                failure_patterns['bnf_iban_valid_format_false'] += 1
            if debug.get('cdt_has_iban'):
                failure_patterns['cdt_has_iban_true'] += 1
            if debug.get('cdt_has_iban') and not debug.get('cdt_iban_valid_format'):
                failure_patterns['cdt_iban_valid_format_false'] += 1
            if debug.get('orig_present'):
                failure_patterns['orig_present_true'] += 1
            if debug.get('orig_has_name'):
                failure_patterns['orig_has_name_true'] += 1
            if debug.get('bnf_is_international'):
                failure_patterns['bnf_is_international_true'] += 1
            if debug.get('bnf_account_length') or debug.get('cdt_account_length'):
                failure_patterns['has_account_length'] += 1
        
        for pattern, count in sorted(failure_patterns.items(), key=lambda x: -x[1]):
            if count > 0:
                pct = count / len(failures) * 100
                print(f"  {pattern}: {count} ({pct:.1f}%)")
    
    # List all failure TxnIDs
    if failures:
        print(f"\n{'='*70}")
        print("ALL FAILURE TxnIDs:")
        print("="*70)
        for txn_id, _, _ in failures:
            print(f"  {txn_id}")
    
    print("\n" + "="*70)
    print("END OF REPORT")
    print("="*70)


if __name__ == '__main__':
    main()
