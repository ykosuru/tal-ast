#!/usr/bin/env python3
"""
Test extracted RandomForest rules for 8029 against actual IFMLs.
8029 = Account/ID validation issue (compound ID, slash in ID, etc.)

Usage:
    python test_8029_rules.py --data-dir /path/to/ifml/data --limit 1000
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add project path for imports
sys.path.insert(0, '/mnt/project')

from data_pipeline import IFMLDataPipeline


def check_8029_rf_rules(features: Dict) -> Tuple[bool, List[str]]:
    """
    Check if 8029 should fire based on RF-extracted rules.
    8029 = Account/ID validation issue (compound ID, slash in ID, etc.)
    
    RF Patterns (from screenshots):
    For 8029_BNFBNK:
    - primary_amount = False (100%)
    - bnf_id_has_slash = True (high confidence)
    - bnf_id_has_nch_pattern = True (high confidence)
    
    For 8029_BNPPTY:
    - orig_has_iban = True, total_nch_sources > 0
    - dbt_is_international = True
    
    Returns (should_fire, reasons)
    """
    reasons = []
    
    def get(feat, default=None):
        if feat in features:
            return features[feat]
        return default
    
    matches = 0
    
    # -------------------------------------------------------------------------
    # Rule 1: primary_amount = False (100% for BNFBNK)
    # -------------------------------------------------------------------------
    primary_amount = get('primary_amount', True)
    if primary_amount == False:
        reasons.append("primary_amount=False")
        matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 2: bnf_id_has_slash = True (high confidence, Path 2 & 3)
    # -------------------------------------------------------------------------
    bnf_id_has_slash = get('bnf_id_has_slash', False)
    if bnf_id_has_slash:
        reasons.append("BNF: id_has_slash=True")
        matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 3: bnf_id_has_nch_pattern = True (high confidence, Path 1)
    # -------------------------------------------------------------------------
    bnf_id_has_nch_pattern = get('bnf_id_has_nch_pattern', False)
    if bnf_id_has_nch_pattern:
        reasons.append("BNF: id_has_nch_pattern=True")
        matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 4: has_multiple_nch_sources = True (Path 2)
    # -------------------------------------------------------------------------
    has_multiple_nch_sources = get('has_multiple_nch_sources', False)
    if has_multiple_nch_sources:
        reasons.append("has_multiple_nch_sources=True")
        matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 5: cdt_is_international = True (Path 3)
    # -------------------------------------------------------------------------
    cdt_is_international = get('cdt_is_international', False)
    if cdt_is_international:
        reasons.append("CDT: is_international=True")
        matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 6: cdt_is_chips_aba = True (Path 3)
    # -------------------------------------------------------------------------
    cdt_is_chips_aba = get('cdt_is_chips_aba', False)
    if cdt_is_chips_aba:
        reasons.append("CDT: is_chips_aba=True")
        matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 7: orig_has_iban with total_nch_sources > 0 (Path 1 BNPPTY)
    # -------------------------------------------------------------------------
    orig_has_iban = get('orig_has_iban', False)
    total_nch_sources = get('total_nch_sources', 0) or 0
    if orig_has_iban and total_nch_sources > 0:
        reasons.append(f"ORIG: has_iban + total_nch_sources={total_nch_sources}")
        matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 8: dbt_is_international = True (Path 3 BNPPTY)
    # -------------------------------------------------------------------------
    dbt_is_international = get('dbt_is_international', False)
    if dbt_is_international:
        reasons.append("DBT: is_international=True")
        matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 9: orig_bic_valid_country = True (Path 3)
    # -------------------------------------------------------------------------
    orig_bic_valid_country = get('orig_bic_valid_country', False)
    if orig_bic_valid_country:
        reasons.append("ORIG: bic_valid_country=True")
        matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 10: cdt_account_length > 0 (Path 1)
    # -------------------------------------------------------------------------
    cdt_account_length = get('cdt_account_length', 0) or 0
    if cdt_account_length > 0:
        reasons.append(f"CDT: account_length={cdt_account_length}")
        matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 11: intm_has_name = True (Path 1)
    # -------------------------------------------------------------------------
    intm_has_name = get('intm_has_name', False)
    if intm_has_name:
        reasons.append("INTM: has_name=True")
        matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 12: send_address_lines > 1 (Path 2 BNPPTY)
    # -------------------------------------------------------------------------
    send_address_lines = get('send_address_lines', 0) or 0
    if send_address_lines > 1:
        reasons.append(f"SEND: address_lines={send_address_lines}")
        matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 13: bic_count > 2 (Path 2 BNPPTY)
    # -------------------------------------------------------------------------
    bic_count = get('bic_count', 0) or 0
    if bic_count > 2:
        reasons.append(f"bic_count={bic_count} > 2")
        matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 14: orig_bic_length > 3 (Path 2)
    # -------------------------------------------------------------------------
    orig_bic_length = get('orig_bic_length', 0) or 0
    if orig_bic_length > 3:
        reasons.append(f"ORIG: bic_length={orig_bic_length}")
        matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 15: dbt_address_lines > 0 (Path 1 BNPPTY)
    # -------------------------------------------------------------------------
    dbt_address_lines = get('dbt_address_lines', 0) or 0
    if dbt_address_lines > 0:
        reasons.append(f"DBT: address_lines={dbt_address_lines}")
        matches += 1
    
    should_fire = matches > 0
    return should_fire, reasons


def main():
    parser = argparse.ArgumentParser(description='Test RF 8029 rules against IFMLs')
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
    
    # Count 8029 codes in responses
    total_8029_in_responses = 0
    for record in pipeline.records:
        actual_codes = record.error_codes_only
        composite_codes = record.composite_codes or []
        if any('8029' in str(c) for c in actual_codes + composite_codes):
            total_8029_in_responses += 1
    
    print(f"\n📊 Found {total_8029_in_responses} IFMLs with 8029 in responses")
    
    # Track results
    successes = []
    failures = []
    total_8029 = 0
    processed = 0
    
    for record in pipeline.records:
        if processed >= args.limit:
            break
        
        actual_codes = record.error_codes_only
        composite_codes = record.composite_codes or []
        
        has_8029 = any('8029' in str(c) for c in actual_codes + composite_codes)
        
        if not has_8029:
            continue
        
        total_8029 += 1
        processed += 1
        
        txn_id = record.transaction_id
        features = record.request_features
        
        should_fire, reasons = check_8029_rf_rules(features)
        
        if should_fire:
            successes.append((txn_id, reasons))
            if args.verbose:
                print(f"✅ {txn_id}: {reasons[0]}")
        else:
            debug_info = {
                'primary_amount': features.get('primary_amount'),
                'bnf_id_has_slash': features.get('bnf_id_has_slash'),
                'bnf_id_has_nch_pattern': features.get('bnf_id_has_nch_pattern'),
                'has_multiple_nch_sources': features.get('has_multiple_nch_sources'),
                'cdt_is_international': features.get('cdt_is_international'),
                'dbt_is_international': features.get('dbt_is_international'),
                'orig_has_iban': features.get('orig_has_iban'),
                'total_nch_sources': features.get('total_nch_sources'),
                'cdt_account_length': features.get('cdt_account_length'),
                'dbt_address_lines': features.get('dbt_address_lines'),
            }
            failures.append((txn_id, composite_codes, debug_info))
            if args.verbose:
                print(f"❌ {txn_id}")
    
    # =========================================================================
    # REPORT
    # =========================================================================
    print("\n" + "="*70)
    print("TEST RESULTS: 8029 Rule Validation")
    print("="*70)
    
    print(f"\nTotal IFMLs with 8029: {total_8029}")
    print(f"Successes (rule matched): {len(successes)}")
    print(f"Failures (rule missed): {len(failures)}")
    
    if total_8029 > 0:
        success_rate = len(successes) / total_8029 * 100
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
            print(f"   Actual codes: {[c for c in codes if '8029' in str(c)]}")
            print(f"   Features: {debug_info}")
        
        if len(failures) > args.show_failures:
            print(f"\n... and {len(failures) - args.show_failures} more failures")
        
        # Analyze failure patterns
        print(f"\n{'='*70}")
        print("FAILURE PATTERN ANALYSIS:")
        print("="*70)
        
        failure_patterns = {
            'primary_amount_false': 0,
            'bnf_id_has_slash_true': 0,
            'bnf_id_has_nch_pattern_true': 0,
            'cdt_is_international_true': 0,
            'dbt_is_international_true': 0,
            'orig_has_iban_true': 0,
        }
        
        for txn_id, codes, debug in failures:
            if debug.get('primary_amount') == False:
                failure_patterns['primary_amount_false'] += 1
            if debug.get('bnf_id_has_slash'):
                failure_patterns['bnf_id_has_slash_true'] += 1
            if debug.get('bnf_id_has_nch_pattern'):
                failure_patterns['bnf_id_has_nch_pattern_true'] += 1
            if debug.get('cdt_is_international'):
                failure_patterns['cdt_is_international_true'] += 1
            if debug.get('dbt_is_international'):
                failure_patterns['dbt_is_international_true'] += 1
            if debug.get('orig_has_iban'):
                failure_patterns['orig_has_iban_true'] += 1
        
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
