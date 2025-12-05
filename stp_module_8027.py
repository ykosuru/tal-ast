#!/usr/bin/env python3
"""
Test extracted RandomForest rules for 8027 against actual IFMLs.
8027 = IBAN/account validation issue

Usage:
    python test_8027_rules.py --data-dir /path/to/ifml/data --limit 1000
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add project path for imports
sys.path.insert(0, '/mnt/project')

from data_pipeline import IFMLDataPipeline


def check_8027_rf_rules(features: Dict) -> Tuple[bool, List[str]]:
    """
    Check if 8027 should fire based on RF-extracted rules.
    8027 = IBAN/account validation issue
    
    RF Patterns (from screenshots):
    For 8027_BNPPTY:
    - needs_iban_count = False (86%)
    - iban_checksum_valid_count = True (67%)
    
    Suggested CODE_TRIGGERS:
    - require_true: ['iban_checksum_valid_count']
    - require_false: ['needs_iban_count']
    
    Returns (should_fire, reasons)
    """
    reasons = []
    
    def get(feat, default=None):
        if feat in features:
            return features[feat]
        return default
    
    matches = 0
    
    # -------------------------------------------------------------------------
    # Rule 1: iban_checksum_valid_count = True (67% - from aggregated patterns)
    # -------------------------------------------------------------------------
    iban_checksum_valid_count = get('iban_checksum_valid_count', 0)
    if iban_checksum_valid_count and iban_checksum_valid_count > 0:
        reasons.append(f"iban_checksum_valid_count={iban_checksum_valid_count}")
        matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 2: iban_format_valid_count > 0 (from Path 1)
    # -------------------------------------------------------------------------
    iban_format_valid_count = get('iban_format_valid_count', 0)
    if iban_format_valid_count and iban_format_valid_count > 0:
        reasons.append(f"iban_format_valid_count={iban_format_valid_count}")
        matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 3: bnf_nch_sources = True (from Path 1, 3 - high confidence)
    # -------------------------------------------------------------------------
    bnf_nch_sources = get('bnf_nch_sources', False)
    if bnf_nch_sources:
        reasons.append("BNF: nch_sources=True")
        matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 4: cdt_account_has_dirty_chars = True (from Path 1)
    # -------------------------------------------------------------------------
    cdt_account_has_dirty_chars = get('cdt_account_has_dirty_chars', False)
    if cdt_account_has_dirty_chars:
        reasons.append("CDT: account_has_dirty_chars=True")
        matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 5: bnf_has_iban = True (from Path 2 - high confidence)
    # -------------------------------------------------------------------------
    bnf_has_iban = get('bnf_has_iban', False)
    if bnf_has_iban:
        reasons.append("BNF: has_iban=True")
        matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 6: dbt_is_bic_derivable = True (from Path 2)
    # -------------------------------------------------------------------------
    dbt_is_bic_derivable = get('dbt_is_bic_derivable', False)
    if dbt_is_bic_derivable:
        reasons.append("DBT: is_bic_derivable=True")
        matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 7: bnf_account_length <= 4 (from Path 3 - short account)
    # -------------------------------------------------------------------------
    bnf_account_length = get('bnf_account_length', 0)
    if bnf_account_length and bnf_account_length <= 4 and bnf_account_length > 0:
        reasons.append(f"BNF: account_length={bnf_account_length} <= 4 (short)")
        matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 8: orig_nch_validation_applicable = True (from Path 3)
    # -------------------------------------------------------------------------
    orig_nch_validation_applicable = get('orig_nch_validation_applicable', False)
    if orig_nch_validation_applicable:
        reasons.append("ORIG: nch_validation_applicable=True")
        matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 9: cdt_fedaba_checksum_valid = True (from Path 1)
    # -------------------------------------------------------------------------
    cdt_fedaba_checksum_valid = get('cdt_fedaba_checksum_valid', False)
    if cdt_fedaba_checksum_valid:
        reasons.append("CDT: fedaba_checksum_valid=True")
        matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 10: orig_iban_valid_format = True (from Path 1)
    # -------------------------------------------------------------------------
    orig_iban_valid_format = get('orig_iban_valid_format', False)
    if orig_iban_valid_format:
        reasons.append("ORIG: iban_valid_format=True")
        matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 11: Any party has IBAN (general trigger)
    # -------------------------------------------------------------------------
    for prefix in ['bnf_', 'cdt_', 'dbt_', 'orig_', 'intm_']:
        has_iban = get(f'{prefix}has_iban', False)
        if has_iban:
            party = prefix.rstrip('_').upper()
            reasons.append(f"{party}: has_iban=True")
            matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 12: avg_address_lines > 0 (from Path 3)
    # -------------------------------------------------------------------------
    avg_address_lines = get('avg_address_lines', 0)
    if avg_address_lines and avg_address_lines > 0:
        reasons.append(f"avg_address_lines={avg_address_lines}")
        matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 13: BNF missing critical account info (no IBAN, no NCH, no account)
    # Pattern from failures: all account-related features are False/0/None
    # -------------------------------------------------------------------------
    bnf_has_account = get('bnf_has_account', False)
    bnf_nch_sources = get('bnf_nch_sources', 0)
    bnf_account_length = get('bnf_account_length', 0)
    
    if not bnf_has_iban and not bnf_has_account and (not bnf_nch_sources or bnf_nch_sources == 0) and (not bnf_account_length or bnf_account_length == 0):
        reasons.append("BNF: missing all account info (no IBAN, no account, no NCH)")
        matches += 1
    
    should_fire = matches > 0
    return should_fire, reasons


def main():
    parser = argparse.ArgumentParser(description='Test RF 8027 rules against IFMLs')
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
    
    # Count 8027 codes in responses
    total_8027_in_responses = 0
    for record in pipeline.records:
        actual_codes = record.error_codes_only
        composite_codes = record.composite_codes or []
        if any('8027' in str(c) for c in actual_codes + composite_codes):
            total_8027_in_responses += 1
    
    print(f"\n📊 Found {total_8027_in_responses} IFMLs with 8027 in responses")
    
    # Track results
    successes = []
    failures = []
    total_8027 = 0
    processed = 0
    
    for record in pipeline.records:
        if processed >= args.limit:
            break
        
        actual_codes = record.error_codes_only
        composite_codes = record.composite_codes or []
        
        has_8027 = any('8027' in str(c) for c in actual_codes + composite_codes)
        
        if not has_8027:
            continue
        
        total_8027 += 1
        processed += 1
        
        txn_id = record.transaction_id
        features = record.request_features
        
        should_fire, reasons = check_8027_rf_rules(features)
        
        if should_fire:
            successes.append((txn_id, reasons))
            if args.verbose:
                print(f"✅ {txn_id}: {reasons[0]}")
        else:
            debug_info = {
                'bnf_has_iban': features.get('bnf_has_iban'),
                'bnf_nch_sources': features.get('bnf_nch_sources'),
                'bnf_account_length': features.get('bnf_account_length'),
                'cdt_account_has_dirty_chars': features.get('cdt_account_has_dirty_chars'),
                'dbt_is_bic_derivable': features.get('dbt_is_bic_derivable'),
                'iban_checksum_valid_count': features.get('iban_checksum_valid_count'),
                'iban_format_valid_count': features.get('iban_format_valid_count'),
                'orig_nch_validation_applicable': features.get('orig_nch_validation_applicable'),
                'avg_address_lines': features.get('avg_address_lines'),
            }
            failures.append((txn_id, composite_codes, debug_info))
            if args.verbose:
                print(f"❌ {txn_id}")
    
    # =========================================================================
    # REPORT
    # =========================================================================
    print("\n" + "="*70)
    print("TEST RESULTS: 8027 Rule Validation")
    print("="*70)
    
    print(f"\nTotal IFMLs with 8027: {total_8027}")
    print(f"Successes (rule matched): {len(successes)}")
    print(f"Failures (rule missed): {len(failures)}")
    
    if total_8027 > 0:
        success_rate = len(successes) / total_8027 * 100
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
            print(f"   Actual codes: {[c for c in codes if '8027' in str(c)]}")
            print(f"   Features: {debug_info}")
        
        if len(failures) > args.show_failures:
            print(f"\n... and {len(failures) - args.show_failures} more failures")
        
        # Analyze failure patterns
        print(f"\n{'='*70}")
        print("FAILURE PATTERN ANALYSIS:")
        print("="*70)
        
        failure_patterns = {
            'bnf_has_iban_true': 0,
            'bnf_nch_sources_true': 0,
            'cdt_account_has_dirty_chars_true': 0,
            'dbt_is_bic_derivable_true': 0,
            'iban_checksum_valid_count_gt_0': 0,
        }
        
        for txn_id, codes, debug in failures:
            if debug.get('bnf_has_iban'):
                failure_patterns['bnf_has_iban_true'] += 1
            if debug.get('bnf_nch_sources'):
                failure_patterns['bnf_nch_sources_true'] += 1
            if debug.get('cdt_account_has_dirty_chars'):
                failure_patterns['cdt_account_has_dirty_chars_true'] += 1
            if debug.get('dbt_is_bic_derivable'):
                failure_patterns['dbt_is_bic_derivable_true'] += 1
            if debug.get('iban_checksum_valid_count') and debug.get('iban_checksum_valid_count') > 0:
                failure_patterns['iban_checksum_valid_count_gt_0'] += 1
        
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
