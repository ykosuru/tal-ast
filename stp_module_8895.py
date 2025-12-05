#!/usr/bin/env python3
"""
Test extracted RandomForest rules for 8895 against actual IFMLs.
8895 = NCH validation / IBAN validation issue

Usage:
    python test_8895_rules.py --data-dir /path/to/ifml/data --limit 1000
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add project path for imports
sys.path.insert(0, '/mnt/project')

from data_pipeline import IFMLDataPipeline


def check_8895_rf_rules(features: Dict) -> Tuple[bool, List[str]]:
    """
    Check if 8895 should fire based on RF-extracted rules.
    8895 = NCH validation / IBAN validation issue
    
    RF Patterns (from screenshots):
    For 8895 (general):
    - has_invalid_iban = True (high confidence)
    - orig_nch_validation_applicable = True (high confidence)
    - bic_iban_mismatch_count > 0
    
    For 8895_CDTPTY:
    - total_nch_sources > 0, dbt_is_domestic = True
    - bnf_nch_validation_applicable = True (high confidence)
    - bnf_has_bic = True, bnf_is_international = False
    
    Returns (should_fire, reasons)
    """
    reasons = []
    
    def get(feat, default=None):
        if feat in features:
            return features[feat]
        return default
    
    matches = 0
    
    # -------------------------------------------------------------------------
    # Rule 1: has_invalid_iban = True (high confidence)
    # -------------------------------------------------------------------------
    has_invalid_iban = get('has_invalid_iban', False)
    if has_invalid_iban:
        reasons.append("has_invalid_iban=True")
        matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 2: nch_validation_applicable for any party (high confidence)
    # -------------------------------------------------------------------------
    for prefix in ['bnf_', 'cdt_', 'orig_', 'dbt_']:
        nch_val_applicable = get(f'{prefix}nch_validation_applicable', False)
        if nch_val_applicable:
            party = prefix.rstrip('_').upper()
            reasons.append(f"{party}: nch_validation_applicable=True")
            matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 3: bic_iban_mismatch_count > 0 (Path 3)
    # -------------------------------------------------------------------------
    bic_iban_mismatch_count = get('bic_iban_mismatch_count', 0) or 0
    if bic_iban_mismatch_count > 0:
        reasons.append(f"bic_iban_mismatch_count={bic_iban_mismatch_count}")
        matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 4: total_nch_sources > 0 (Path 1 CDTPTY)
    # -------------------------------------------------------------------------
    total_nch_sources = get('total_nch_sources', 0) or 0
    if total_nch_sources > 0:
        reasons.append(f"total_nch_sources={total_nch_sources}")
        matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 5: dbt_is_domestic = True (Path 1 CDTPTY)
    # -------------------------------------------------------------------------
    dbt_is_domestic = get('dbt_is_domestic', False)
    if dbt_is_domestic:
        reasons.append("DBT: is_domestic=True")
        matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 6: cdt_is_domestic = True (Path 2 general)
    # -------------------------------------------------------------------------
    cdt_is_domestic = get('cdt_is_domestic', False)
    if cdt_is_domestic:
        reasons.append("CDT: is_domestic=True")
        matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 7: bnf_has_bic + bnf_is_international=False (Path 2 CDTPTY)
    # -------------------------------------------------------------------------
    bnf_has_bic = get('bnf_has_bic', False)
    bnf_is_international = get('bnf_is_international', False)
    if bnf_has_bic and not bnf_is_international:
        reasons.append("BNF: has_bic + is_international=False (domestic)")
        matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 8: orig_bic_valid_format = True (Path 3 CDTPTY)
    # -------------------------------------------------------------------------
    orig_bic_valid_format = get('orig_bic_valid_format', False)
    if orig_bic_valid_format:
        reasons.append("ORIG: bic_valid_format=True")
        matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 9: bnf_id_has_slash = True (Path 3 CDTPTY)
    # -------------------------------------------------------------------------
    bnf_id_has_slash = get('bnf_id_has_slash', False)
    if bnf_id_has_slash:
        reasons.append("BNF: id_has_slash=True")
        matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 10: intm_is_iban_derivable = True (Path 1 general)
    # -------------------------------------------------------------------------
    intm_is_iban_derivable = get('intm_is_iban_derivable', False)
    if intm_is_iban_derivable:
        reasons.append("INTM: is_iban_derivable=True")
        matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 11: dbt_account_length > 0 (Path 1 general)
    # -------------------------------------------------------------------------
    dbt_account_length = get('dbt_account_length', 0) or 0
    if dbt_account_length > 0:
        reasons.append(f"DBT: account_length={dbt_account_length}")
        matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 12: orig_has_iban = True (Path 2 general)
    # -------------------------------------------------------------------------
    orig_has_iban = get('orig_has_iban', False)
    if orig_has_iban:
        reasons.append("ORIG: has_iban=True")
        matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 13: total_address_lines > 0 (Path 3 general)
    # -------------------------------------------------------------------------
    total_address_lines = get('total_address_lines', 0) or 0
    if total_address_lines > 0:
        reasons.append(f"total_address_lines={total_address_lines}")
        matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 14: dbt_account_numeric > 0 (Path 2 CDTPTY)
    # -------------------------------------------------------------------------
    dbt_account_numeric = get('dbt_account_numeric', False)
    if dbt_account_numeric:
        reasons.append("DBT: account_numeric=True")
        matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 15: bnf_iban_checksum_valid = True (from CDTPTY header)
    # -------------------------------------------------------------------------
    bnf_iban_checksum_valid = get('bnf_iban_checksum_valid', False)
    if bnf_iban_checksum_valid:
        reasons.append("BNF: iban_checksum_valid=True")
        matches += 1
    
    should_fire = matches > 0
    return should_fire, reasons


def main():
    parser = argparse.ArgumentParser(description='Test RF 8895 rules against IFMLs')
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
    
    # Count 8895 codes in responses
    total_8895_in_responses = 0
    for record in pipeline.records:
        actual_codes = record.error_codes_only
        composite_codes = record.composite_codes or []
        if any('8895' in str(c) for c in actual_codes + composite_codes):
            total_8895_in_responses += 1
    
    print(f"\n📊 Found {total_8895_in_responses} IFMLs with 8895 in responses")
    
    # Track results
    successes = []
    failures = []
    total_8895 = 0
    processed = 0
    
    for record in pipeline.records:
        if processed >= args.limit:
            break
        
        actual_codes = record.error_codes_only
        composite_codes = record.composite_codes or []
        
        has_8895 = any('8895' in str(c) for c in actual_codes + composite_codes)
        
        if not has_8895:
            continue
        
        total_8895 += 1
        processed += 1
        
        txn_id = record.transaction_id
        features = record.request_features
        
        should_fire, reasons = check_8895_rf_rules(features)
        
        if should_fire:
            successes.append((txn_id, reasons))
            if args.verbose:
                print(f"✅ {txn_id}: {reasons[0]}")
        else:
            debug_info = {
                'has_invalid_iban': features.get('has_invalid_iban'),
                'bnf_nch_validation_applicable': features.get('bnf_nch_validation_applicable'),
                'cdt_nch_validation_applicable': features.get('cdt_nch_validation_applicable'),
                'orig_nch_validation_applicable': features.get('orig_nch_validation_applicable'),
                'bic_iban_mismatch_count': features.get('bic_iban_mismatch_count'),
                'total_nch_sources': features.get('total_nch_sources'),
                'dbt_is_domestic': features.get('dbt_is_domestic'),
                'cdt_is_domestic': features.get('cdt_is_domestic'),
                'bnf_has_bic': features.get('bnf_has_bic'),
                'total_address_lines': features.get('total_address_lines'),
            }
            failures.append((txn_id, composite_codes, debug_info))
            if args.verbose:
                print(f"❌ {txn_id}")
    
    # =========================================================================
    # REPORT
    # =========================================================================
    print("\n" + "="*70)
    print("TEST RESULTS: 8895 Rule Validation")
    print("="*70)
    
    print(f"\nTotal IFMLs with 8895: {total_8895}")
    print(f"Successes (rule matched): {len(successes)}")
    print(f"Failures (rule missed): {len(failures)}")
    
    if total_8895 > 0:
        success_rate = len(successes) / total_8895 * 100
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
            print(f"   Actual codes: {[c for c in codes if '8895' in str(c)]}")
            print(f"   Features: {debug_info}")
        
        if len(failures) > args.show_failures:
            print(f"\n... and {len(failures) - args.show_failures} more failures")
        
        # Analyze failure patterns
        print(f"\n{'='*70}")
        print("FAILURE PATTERN ANALYSIS:")
        print("="*70)
        
        failure_patterns = {
            'has_invalid_iban_true': 0,
            'nch_validation_applicable_true': 0,
            'bic_iban_mismatch_count_gt_0': 0,
            'total_nch_sources_gt_0': 0,
            'dbt_is_domestic_true': 0,
            'cdt_is_domestic_true': 0,
        }
        
        for txn_id, codes, debug in failures:
            if debug.get('has_invalid_iban'):
                failure_patterns['has_invalid_iban_true'] += 1
            if debug.get('bnf_nch_validation_applicable') or debug.get('cdt_nch_validation_applicable') or debug.get('orig_nch_validation_applicable'):
                failure_patterns['nch_validation_applicable_true'] += 1
            if debug.get('bic_iban_mismatch_count') and debug.get('bic_iban_mismatch_count') > 0:
                failure_patterns['bic_iban_mismatch_count_gt_0'] += 1
            if debug.get('total_nch_sources') and debug.get('total_nch_sources') > 0:
                failure_patterns['total_nch_sources_gt_0'] += 1
            if debug.get('dbt_is_domestic'):
                failure_patterns['dbt_is_domestic_true'] += 1
            if debug.get('cdt_is_domestic'):
                failure_patterns['cdt_is_domestic_true'] += 1
        
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
