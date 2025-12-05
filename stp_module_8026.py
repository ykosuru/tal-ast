#!/usr/bin/env python3
"""
Test extracted RandomForest rules for 8026 against actual IFMLs.
8026 = BIC/IBAN country mismatch or IBAN derivation issue

Usage:
    python test_8026_rules.py --data-dir /path/to/ifml/data --limit 1000
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add project path for imports
sys.path.insert(0, '/mnt/project')

from data_pipeline import IFMLDataPipeline


def check_8026_rf_rules(features: Dict) -> Tuple[bool, List[str]]:
    """
    Check if 8026 should fire based on RF-extracted rules.
    8026 fires when there's a BIC/IBAN country mismatch or IBAN derivation issue.
    
    RF Patterns:
    - needs_iban_count = True (75%)
    - bnf_bic_party_country_match = False (75%)
    
    Returns (should_fire, reasons)
    """
    reasons = []
    
    def get(feat, default=None):
        if feat in features:
            return features[feat]
        return default
    
    matches = 0
    
    # -------------------------------------------------------------------------
    # Rule 1: BIC/IBAN country mismatch (direct from RF)
    # bnf_bic_party_country_match = False
    # -------------------------------------------------------------------------
    for prefix in ['bnf_', 'cdt_', 'orig_', 'dbt_', 'intm_']:
        bic_party_country_match = get(f'{prefix}bic_party_country_match')
        has_bic = get(f'{prefix}has_bic', False)
        has_iban = get(f'{prefix}has_iban', False)
        
        # If has both BIC and IBAN but countries don't match
        if has_bic and has_iban and bic_party_country_match == False:
            party = prefix.rstrip('_').upper()
            reasons.append(f"{party}: bic_party_country_match=False (BIC/IBAN country mismatch)")
            matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 2: BIC/IBAN country mismatch using bic_iban_country_match
    # -------------------------------------------------------------------------
    for prefix in ['bnf_', 'cdt_', 'orig_', 'dbt_', 'intm_']:
        bic_iban_match = get(f'{prefix}bic_iban_match')
        has_bic = get(f'{prefix}has_bic', False)
        has_iban = get(f'{prefix}has_iban', False)
        
        if has_bic and has_iban and bic_iban_match == False:
            party = prefix.rstrip('_').upper()
            reasons.append(f"{party}: bic_iban_match=False (BIC/IBAN country mismatch)")
            matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 3: needs_iban but IBAN derivation may fail
    # From RF: needs_iban_count = True correlates with 8026
    # -------------------------------------------------------------------------
    needs_iban_count = get('needs_iban_count', 0)
    if needs_iban_count and needs_iban_count > 0:
        # Check if any party needs IBAN but has BIC country issue
        for prefix in ['bnf_', 'cdt_', 'orig_', 'dbt_', 'intm_']:
            needs_iban = get(f'{prefix}needs_iban', False)
            has_iban = get(f'{prefix}has_iban', False)
            has_bic = get(f'{prefix}has_bic', False)
            bic_valid_country = get(f'{prefix}bic_valid_country', True)
            
            # Needs IBAN, doesn't have it, but has BIC with invalid country
            if needs_iban and not has_iban and has_bic and not bic_valid_country:
                party = prefix.rstrip('_').upper()
                reasons.append(f"{party}: needs_iban, has BIC but bic_valid_country=False")
                matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 4: From RF Path 1 - bnf_has_bic + bnf_has_iban + orig conditions
    # -------------------------------------------------------------------------
    bnf_has_bic = get('bnf_has_bic', False)
    bnf_has_iban = get('bnf_has_iban', False)
    orig_bic_valid_country = get('orig_bic_valid_country', True)
    orig_address_lines = get('orig_address_lines', 0)
    
    if bnf_has_bic and bnf_has_iban and orig_address_lines and orig_address_lines > 0:
        # This pattern from RF - check for country validation issues
        if not orig_bic_valid_country:
            reasons.append("RF Path: bnf has BIC+IBAN, orig has address but invalid BIC country")
            matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 5: From RF Path 3 - cdt_is_international with account issues
    # -------------------------------------------------------------------------
    cdt_is_international = get('cdt_is_international', False)
    has_instructed_amount = get('has_instructed_amount', False)
    account_count = get('account_count', 0)
    
    if cdt_is_international and has_instructed_amount:
        # International credit party - may need IBAN derivation
        cdt_needs_iban = get('cdt_needs_iban', False)
        cdt_has_iban = get('cdt_has_iban', False)
        
        if cdt_needs_iban and not cdt_has_iban:
            reasons.append("CDT: is_international, needs_iban but has_iban=False")
            matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 6: dbt_account_has_dirty_chars (from RF Path 2)
    # Dirty chars in account may cause IBAN derivation failure
    # -------------------------------------------------------------------------
    for prefix in ['bnf_', 'cdt_', 'orig_', 'dbt_', 'intm_']:
        has_dirty_chars = get(f'{prefix}account_has_dirty_chars', False)
        if has_dirty_chars:
            party = prefix.rstrip('_').upper()
            reasons.append(f"{party}: account_has_dirty_chars=True")
            matches += 1
    
    should_fire = matches > 0
    return should_fire, reasons


def main():
    parser = argparse.ArgumentParser(description='Test RF 8026 rules against IFMLs')
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
    
    # Count 8026 codes in responses
    total_8026_in_responses = 0
    for record in pipeline.records:
        actual_codes = record.error_codes_only
        composite_codes = record.composite_codes or []
        if any('8026' in str(c) for c in actual_codes + composite_codes):
            total_8026_in_responses += 1
    
    print(f"\n📊 Found {total_8026_in_responses} IFMLs with 8026 in responses")
    
    # Track results
    successes = []
    failures = []
    total_8026 = 0
    processed = 0
    
    for record in pipeline.records:
        if processed >= args.limit:
            break
        
        actual_codes = record.error_codes_only
        composite_codes = record.composite_codes or []
        
        has_8026 = any('8026' in str(c) for c in actual_codes + composite_codes)
        
        if not has_8026:
            continue
        
        total_8026 += 1
        processed += 1
        
        txn_id = record.transaction_id
        features = record.request_features
        
        should_fire, reasons = check_8026_rf_rules(features)
        
        if should_fire:
            successes.append((txn_id, reasons))
            if args.verbose:
                print(f"✅ {txn_id}: {reasons[0]}")
        else:
            debug_info = {
                'bnf_has_bic': features.get('bnf_has_bic'),
                'bnf_has_iban': features.get('bnf_has_iban'),
                'bnf_bic_party_country_match': features.get('bnf_bic_party_country_match'),
                'bnf_bic_iban_match': features.get('bnf_bic_iban_match'),
                'cdt_has_bic': features.get('cdt_has_bic'),
                'cdt_has_iban': features.get('cdt_has_iban'),
                'cdt_is_international': features.get('cdt_is_international'),
                'needs_iban_count': features.get('needs_iban_count'),
                'bnf_needs_iban': features.get('bnf_needs_iban'),
                'cdt_needs_iban': features.get('cdt_needs_iban'),
            }
            failures.append((txn_id, composite_codes, debug_info))
            if args.verbose:
                print(f"❌ {txn_id}")
    
    # =========================================================================
    # REPORT
    # =========================================================================
    print("\n" + "="*70)
    print("TEST RESULTS: 8026 Rule Validation")
    print("="*70)
    
    print(f"\nTotal IFMLs with 8026: {total_8026}")
    print(f"Successes (rule matched): {len(successes)}")
    print(f"Failures (rule missed): {len(failures)}")
    
    if total_8026 > 0:
        success_rate = len(successes) / total_8026 * 100
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
            print(f"   Actual codes: {[c for c in codes if '8026' in str(c)]}")
            print(f"   Features: {debug_info}")
        
        if len(failures) > args.show_failures:
            print(f"\n... and {len(failures) - args.show_failures} more failures")
        
        # Analyze failure patterns
        print(f"\n{'='*70}")
        print("FAILURE PATTERN ANALYSIS:")
        print("="*70)
        
        failure_patterns = {
            'bnf_has_bic_true': 0,
            'bnf_has_iban_true': 0,
            'cdt_has_bic_true': 0,
            'cdt_has_iban_true': 0,
            'cdt_is_international_true': 0,
            'needs_iban_count_gt_0': 0,
        }
        
        for txn_id, codes, debug in failures:
            if debug.get('bnf_has_bic'):
                failure_patterns['bnf_has_bic_true'] += 1
            if debug.get('bnf_has_iban'):
                failure_patterns['bnf_has_iban_true'] += 1
            if debug.get('cdt_has_bic'):
                failure_patterns['cdt_has_bic_true'] += 1
            if debug.get('cdt_has_iban'):
                failure_patterns['cdt_has_iban_true'] += 1
            if debug.get('cdt_is_international'):
                failure_patterns['cdt_is_international_true'] += 1
            if debug.get('needs_iban_count') and debug.get('needs_iban_count') > 0:
                failure_patterns['needs_iban_count_gt_0'] += 1
        
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
