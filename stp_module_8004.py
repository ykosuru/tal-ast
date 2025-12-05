#!/usr/bin/env python3
"""
Test extracted RandomForest rules for 8004 against actual IFMLs.
Uses the project's IFMLDataPipeline to properly load and parse data.

Usage:
    python test_8004_rules_v2.py --data-dir /path/to/ifml/data --limit 1000
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add project path for imports
sys.path.insert(0, '/mnt/project')

from data_pipeline import IFMLDataPipeline


def check_8004_rf_rules(features: Dict) -> Tuple[bool, List[str]]:
    """
    Check if 8004 should fire based on RF-extracted rules.
    Returns (should_fire, reasons)
    """
    reasons = []
    
    # Helper to get feature with fallback
    def get(feat, default=None):
        if feat in features:
            return features[feat]
        return default
    
    matches = 0
    
    # -------------------------------------------------------------------------
    # Rule 1: Original 8004 logic - needs_iban but doesn't have it
    # -------------------------------------------------------------------------
    for prefix in ['bnf_', 'cdt_', 'dbt_', 'orig_', 'intm_']:
        party_present = get(f'{prefix}present', False)
        needs_iban = get(f'{prefix}needs_iban', False)
        has_iban = get(f'{prefix}has_iban', False)
        
        if party_present and needs_iban and not has_iban:
            party = prefix.rstrip('_').upper()
            reasons.append(f"{party}: needs_iban=True, has_iban=False")
            matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 2: NEW - has_account but no IBAN (ACE tries to derive and fails)
    # This catches the 149 failures where needs_iban=False but 8004 still fires
    # -------------------------------------------------------------------------
    for prefix in ['bnf_', 'cdt_', 'dbt_', 'orig_', 'intm_']:
        party_present = get(f'{prefix}present', False)
        has_account = get(f'{prefix}has_account', False)
        has_iban = get(f'{prefix}has_iban', False)
        
        # If party has account but no IBAN, ACE may try to derive IBAN
        if party_present and has_account and not has_iban:
            party = prefix.rstrip('_').upper()
            reasons.append(f"{party}: has_account=True, has_iban=False (IBAN derivation may fail)")
            matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 3: RF pattern - BIC validation failures correlate with 8004
    # -------------------------------------------------------------------------
    for prefix in ['send_', 'bnf_', 'cdt_', 'intm_']:
        has_bic = get(f'{prefix}has_bic', False)
        bic_valid_format = get(f'{prefix}bic_valid_format', True)
        bic_valid_country = get(f'{prefix}bic_valid_country', True)
        
        if has_bic and not bic_valid_format:
            party = prefix.rstrip('_').upper()
            reasons.append(f"{party}: has_bic but bic_valid_format=False")
            matches += 1
        
        if has_bic and not bic_valid_country:
            party = prefix.rstrip('_').upper()
            reasons.append(f"{party}: has_bic but bic_valid_country=False")
            matches += 1
    
    # Fire if we found any matching pattern
    should_fire = matches > 0
    
    return should_fire, reasons


def main():
    parser = argparse.ArgumentParser(description='Test RF 8004 rules against IFMLs')
    parser.add_argument('--data-dir', required=True, help='Directory with IFML JSON files')
    parser.add_argument('--limit', type=int, default=5000, help='Max records to process')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show details')
    parser.add_argument('--show-failures', type=int, default=20, help='Number of failures to show')
    
    args = parser.parse_args()
    
    print("Loading IFML data using IFMLDataPipeline...")
    pipeline = IFMLDataPipeline()
    
    data_path = Path(args.data_dir)
    
    # Load data
    if data_path.is_file():
        loaded = pipeline.load_single_file(str(data_path))
    else:
        loaded = pipeline.load_directory(str(data_path), "*.json")
    
    print(f"Loaded {loaded} total records")
    print(f"Pipeline has {len(pipeline.records)} records")
    
    # Track results
    successes = []
    failures = []
    total_8004 = 0
    processed = 0
    
    for record in pipeline.records:
        if processed >= args.limit:
            break
        
        # Check if 8004 is in actual codes
        actual_codes = record.error_codes_only
        composite_codes = record.composite_codes or []
        
        has_8004 = any('8004' in str(c) for c in actual_codes + composite_codes)
        
        if not has_8004:
            continue
        
        total_8004 += 1
        processed += 1
        
        txn_id = record.transaction_id
        features = record.request_features
        
        # Check our rules
        should_fire, reasons = check_8004_rf_rules(features)
        
        if should_fire:
            successes.append((txn_id, reasons))
            if args.verbose:
                print(f"✅ {txn_id}: {reasons[0]}")
        else:
            # Collect debug info
            debug_info = {
                'bnf_present': features.get('bnf_present'),
                'bnf_needs_iban': features.get('bnf_needs_iban'),
                'bnf_has_iban': features.get('bnf_has_iban'),
                'bnf_has_account': features.get('bnf_has_account'),
                'cdt_present': features.get('cdt_present'),
                'cdt_needs_iban': features.get('cdt_needs_iban'),
                'cdt_has_iban': features.get('cdt_has_iban'),
                'send_has_bic': features.get('send_has_bic'),
                'send_bic_valid_format': features.get('send_bic_valid_format'),
            }
            failures.append((txn_id, composite_codes, debug_info))
            if args.verbose:
                print(f"❌ {txn_id}")
    
    # =========================================================================
    # REPORT
    # =========================================================================
    print("\n" + "="*70)
    print("TEST RESULTS: 8004 Rule Validation")
    print("="*70)
    
    print(f"\nTotal IFMLs with 8004: {total_8004}")
    print(f"Successes (rule matched): {len(successes)}")
    print(f"Failures (rule missed): {len(failures)}")
    
    if total_8004 > 0:
        success_rate = len(successes) / total_8004 * 100
        print(f"\n🎯 Success Rate: {success_rate:.1f}%")
    
    # Show success patterns
    if successes:
        print(f"\n{'='*70}")
        print(f"SUCCESS PATTERNS:")
        print("="*70)
        
        reason_counts = {}
        for txn_id, reasons in successes:
            for reason in reasons:
                key = reason.split(':')[0] if ':' in reason else reason
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
            print(f"   Actual codes: {[c for c in codes if '8004' in str(c)]}")
            print(f"   Features: {debug_info}")
        
        if len(failures) > args.show_failures:
            print(f"\n... and {len(failures) - args.show_failures} more failures")
        
        # Analyze failure patterns
        print(f"\n{'='*70}")
        print("FAILURE PATTERN ANALYSIS:")
        print("="*70)
        
        # Count feature combinations in failures
        failure_patterns = {
            'bnf_needs_iban_false': 0,
            'bnf_has_iban_true': 0,
            'bnf_not_present': 0,
            'cdt_needs_iban_false': 0,
            'no_party_needs_iban': 0,
        }
        
        for txn_id, codes, debug in failures:
            if not debug.get('bnf_present'):
                failure_patterns['bnf_not_present'] += 1
            if not debug.get('bnf_needs_iban'):
                failure_patterns['bnf_needs_iban_false'] += 1
            if debug.get('bnf_has_iban'):
                failure_patterns['bnf_has_iban_true'] += 1
            if not debug.get('cdt_needs_iban'):
                failure_patterns['cdt_needs_iban_false'] += 1
            
            # Check if any party needs iban
            any_needs = debug.get('bnf_needs_iban') or debug.get('cdt_needs_iban')
            if not any_needs:
                failure_patterns['no_party_needs_iban'] += 1
        
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
