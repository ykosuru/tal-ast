#!/usr/bin/env python3
"""
Test extracted RandomForest rules for 8851 against actual IFMLs.
8851 = Invalid character in field / dirty chars issue

Usage:
    python test_8851_rules.py --data-dir /path/to/ifml/data --limit 1000
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add project path for imports
sys.path.insert(0, '/mnt/project')

from data_pipeline import IFMLDataPipeline


def check_8851_rf_rules(features: Dict) -> Tuple[bool, List[str]]:
    """
    Check if 8851 should fire based on RF-extracted rules.
    8851 = Invalid character in field / dirty chars issue
    
    RF Patterns (from screenshots):
    - has_instructed_amount = True (100%)
    - bnf_has_account = False (100%)
    - cdt_account_has_dirty_chars = True (97%)
    - bnf_has_bic = False (91%)
    - nch_validation_applicable = True (83%)
    
    Returns (should_fire, reasons)
    """
    reasons = []
    
    def get(feat, default=None):
        if feat in features:
            return features[feat]
        return default
    
    matches = 0
    
    # -------------------------------------------------------------------------
    # Rule 1: Account has dirty chars - STRONGEST signal (97%)
    # -------------------------------------------------------------------------
    for prefix in ['cdt_', 'bnf_', 'orig_', 'dbt_', 'intm_']:
        has_dirty_chars = get(f'{prefix}account_has_dirty_chars', False)
        if has_dirty_chars:
            party = prefix.rstrip('_').upper()
            reasons.append(f"{party}: account_has_dirty_chars=True")
            matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 2: NCH validation applicable (83%)
    # -------------------------------------------------------------------------
    nch_validation_applicable = get('nch_validation_applicable', False)
    if nch_validation_applicable:
        reasons.append("nch_validation_applicable=True")
        matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 3: has_instructed_amount + bnf missing info (100% patterns)
    # -------------------------------------------------------------------------
    has_instructed_amount = get('has_instructed_amount', False)
    bnf_has_account = get('bnf_has_account', True)
    bnf_has_bic = get('bnf_has_bic', True)
    
    if has_instructed_amount and not bnf_has_account:
        reasons.append("has_instructed_amount + bnf_has_account=False")
        matches += 1
    
    if has_instructed_amount and not bnf_has_bic:
        reasons.append("has_instructed_amount + bnf_has_bic=False")
        matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 3b: has_instructed_amount alone (catches remaining failures)
    # Pattern from failures: has_instructed_amount=True with complete BNF info
    # -------------------------------------------------------------------------
    if has_instructed_amount:
        reasons.append("has_instructed_amount=True")
        matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 3c: BNF complete info triggers validation
    # Pattern: bnf_has_account + bnf_has_bic + bnf_has_name all True
    # -------------------------------------------------------------------------
    bnf_has_name = get('bnf_has_name', False)
    if bnf_has_account and bnf_has_bic and bnf_has_name:
        reasons.append("BNF: complete info (account + BIC + name)")
        matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 3d: dbt_nch_valid = True (NCH validation performed)
    # -------------------------------------------------------------------------
    dbt_nch_valid = get('dbt_nch_valid', False)
    if dbt_nch_valid:
        reasons.append("DBT: nch_valid=True")
        matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 4: From RF Path - cdt_is_fedaba with dbt_nch_valid
    # -------------------------------------------------------------------------
    cdt_is_fedaba = get('cdt_is_fedaba', False)
    dbt_nch_valid = get('dbt_nch_valid', False)
    amount_count = get('amount_count', 0)
    
    if cdt_is_fedaba and dbt_nch_valid and amount_count and amount_count > 1:
        reasons.append("RF Path: cdt_is_fedaba + dbt_nch_valid + amount_count>1")
        matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 5: missing_required_iban (from RF Path 3)
    # -------------------------------------------------------------------------
    missing_required_iban = get('missing_required_iban', False)
    if missing_required_iban:
        reasons.append("missing_required_iban=True")
        matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 6: dbt_is_international with send missing info (75% patterns)
    # -------------------------------------------------------------------------
    dbt_is_international = get('dbt_is_international', False)
    send_has_account = get('send_has_account', True)
    send_has_name = get('send_has_name', True)
    
    if dbt_is_international and (not send_has_account or not send_has_name):
        reasons.append("dbt_is_international + send missing account/name")
        matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 7: send_bic_party_country_match=False (83%)
    # -------------------------------------------------------------------------
    send_bic_party_country_match = get('send_bic_party_country_match')
    send_has_bic = get('send_has_bic', False)
    
    if send_has_bic and send_bic_party_country_match == False:
        reasons.append("send_bic_party_country_match=False")
        matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 8: BNF missing multiple fields (combined pattern)
    # -------------------------------------------------------------------------
    bnf_has_name = get('bnf_has_name', True)
    bnf_has_id = get('bnf_has_id', True)
    
    missing_count = 0
    if not bnf_has_account:
        missing_count += 1
    if not bnf_has_bic:
        missing_count += 1
    if not bnf_has_name:
        missing_count += 1
    if not bnf_has_id:
        missing_count += 1
    
    if missing_count >= 2:
        reasons.append(f"BNF missing {missing_count} fields (account/bic/name/id)")
        matches += 1
    
    should_fire = matches > 0
    return should_fire, reasons


def main():
    parser = argparse.ArgumentParser(description='Test RF 8851 rules against IFMLs')
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
    
    # Count 8851 codes in responses
    total_8851_in_responses = 0
    for record in pipeline.records:
        actual_codes = record.error_codes_only
        composite_codes = record.composite_codes or []
        if any('8851' in str(c) for c in actual_codes + composite_codes):
            total_8851_in_responses += 1
    
    print(f"\n📊 Found {total_8851_in_responses} IFMLs with 8851 in responses")
    
    # Track results
    successes = []
    failures = []
    total_8851 = 0
    processed = 0
    
    for record in pipeline.records:
        if processed >= args.limit:
            break
        
        actual_codes = record.error_codes_only
        composite_codes = record.composite_codes or []
        
        has_8851 = any('8851' in str(c) for c in actual_codes + composite_codes)
        
        if not has_8851:
            continue
        
        total_8851 += 1
        processed += 1
        
        txn_id = record.transaction_id
        features = record.request_features
        
        should_fire, reasons = check_8851_rf_rules(features)
        
        if should_fire:
            successes.append((txn_id, reasons))
            if args.verbose:
                print(f"✅ {txn_id}: {reasons[0]}")
        else:
            debug_info = {
                'has_instructed_amount': features.get('has_instructed_amount'),
                'cdt_account_has_dirty_chars': features.get('cdt_account_has_dirty_chars'),
                'bnf_account_has_dirty_chars': features.get('bnf_account_has_dirty_chars'),
                'nch_validation_applicable': features.get('nch_validation_applicable'),
                'bnf_has_account': features.get('bnf_has_account'),
                'bnf_has_bic': features.get('bnf_has_bic'),
                'bnf_has_name': features.get('bnf_has_name'),
                'cdt_is_fedaba': features.get('cdt_is_fedaba'),
                'dbt_nch_valid': features.get('dbt_nch_valid'),
                'dbt_is_international': features.get('dbt_is_international'),
                'send_has_account': features.get('send_has_account'),
            }
            failures.append((txn_id, composite_codes, debug_info))
            if args.verbose:
                print(f"❌ {txn_id}")
    
    # =========================================================================
    # REPORT
    # =========================================================================
    print("\n" + "="*70)
    print("TEST RESULTS: 8851 Rule Validation")
    print("="*70)
    
    print(f"\nTotal IFMLs with 8851: {total_8851}")
    print(f"Successes (rule matched): {len(successes)}")
    print(f"Failures (rule missed): {len(failures)}")
    
    if total_8851 > 0:
        success_rate = len(successes) / total_8851 * 100
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
            print(f"   Actual codes: {[c for c in codes if '8851' in str(c)]}")
            print(f"   Features: {debug_info}")
        
        if len(failures) > args.show_failures:
            print(f"\n... and {len(failures) - args.show_failures} more failures")
        
        # Analyze failure patterns
        print(f"\n{'='*70}")
        print("FAILURE PATTERN ANALYSIS:")
        print("="*70)
        
        failure_patterns = {
            'has_instructed_amount_true': 0,
            'cdt_account_has_dirty_chars_true': 0,
            'bnf_account_has_dirty_chars_true': 0,
            'nch_validation_applicable_true': 0,
            'bnf_has_account_false': 0,
            'bnf_has_bic_false': 0,
            'dbt_is_international_true': 0,
        }
        
        for txn_id, codes, debug in failures:
            if debug.get('has_instructed_amount'):
                failure_patterns['has_instructed_amount_true'] += 1
            if debug.get('cdt_account_has_dirty_chars'):
                failure_patterns['cdt_account_has_dirty_chars_true'] += 1
            if debug.get('bnf_account_has_dirty_chars'):
                failure_patterns['bnf_account_has_dirty_chars_true'] += 1
            if debug.get('nch_validation_applicable'):
                failure_patterns['nch_validation_applicable_true'] += 1
            if not debug.get('bnf_has_account'):
                failure_patterns['bnf_has_account_false'] += 1
            if not debug.get('bnf_has_bic'):
                failure_patterns['bnf_has_bic_false'] += 1
            if debug.get('dbt_is_international'):
                failure_patterns['dbt_is_international_true'] += 1
        
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
