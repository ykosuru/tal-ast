#!/usr/bin/env python3
"""
Test extracted RandomForest rules for 8022 against actual IFMLs.
8022 = Account/routing number validation issue

Usage:
    python test_8022_rules.py --data-dir /path/to/ifml/data --limit 1000
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add project path for imports
sys.path.insert(0, '/mnt/project')

from data_pipeline import IFMLDataPipeline


def check_8022_rf_rules(features: Dict) -> Tuple[bool, List[str]]:
    """
    Check if 8022 should fire based on RF-extracted rules.
    8022 = Account/routing number validation issue
    
    RF Patterns (from screenshot):
    For 8022_BNFBNK:
    - No consistent patterns
    - Path 1: bnf_has_iban = True
    - Path 2: bnf_needs_iban = True, bnf_id_has_bic_and_nch = True
    - Path 3: send_bic_valid_country = False, intm_bic_valid_format = True
    
    Returns (should_fire, reasons)
    """
    reasons = []
    
    def get(feat, default=None):
        if feat in features:
            return features[feat]
        return default
    
    matches = 0
    
    # -------------------------------------------------------------------------
    # Rule 1: bnf_has_iban (from RF Path 1)
    # -------------------------------------------------------------------------
    bnf_has_iban = get('bnf_has_iban', False)
    if bnf_has_iban:
        reasons.append("BNF: has_iban=True")
        matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 2: bnf_needs_iban (from RF Path 2)
    # -------------------------------------------------------------------------
    bnf_needs_iban = get('bnf_needs_iban', False)
    if bnf_needs_iban:
        reasons.append("BNF: needs_iban=True")
        matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 3: bnf_id_has_bic_and_nch (from RF Paths)
    # -------------------------------------------------------------------------
    bnf_id_has_bic_and_nch = get('bnf_id_has_bic_and_nch', False)
    if bnf_id_has_bic_and_nch:
        reasons.append("BNF: id_has_bic_and_nch=True")
        matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 4: send_bic_valid_country = False (from RF Path 3)
    # -------------------------------------------------------------------------
    send_bic_valid_country = get('send_bic_valid_country')
    send_has_bic = get('send_has_bic', False)
    
    if send_has_bic and send_bic_valid_country == False:
        reasons.append("SEND: bic_valid_country=False")
        matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 5: intm_bic_valid_format (from RF Path 3)
    # -------------------------------------------------------------------------
    intm_bic_valid_format = get('intm_bic_valid_format', False)
    intm_has_bic = get('intm_has_bic', False)
    
    if intm_has_bic and intm_bic_valid_format:
        reasons.append("INTM: has BIC with valid format")
        matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 6: total_address_lines > 0 (from RF Path 1)
    # -------------------------------------------------------------------------
    total_address_lines = get('total_address_lines', 0)
    if total_address_lines and total_address_lines > 0:
        reasons.append(f"total_address_lines={total_address_lines}")
        matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 7: Account needs length fix
    # -------------------------------------------------------------------------
    for prefix in ['bnf_', 'cdt_', 'dbt_', 'orig_', 'intm_']:
        needs_fix = get(f'{prefix}account_needs_length_fix', False)
        if needs_fix:
            party = prefix.rstrip('_').upper()
            reasons.append(f"{party}: account_needs_length_fix=True")
            matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 8: Has BIC but no valid country
    # -------------------------------------------------------------------------
    for prefix in ['bnf_', 'cdt_', 'dbt_', 'orig_', 'intm_']:
        has_bic = get(f'{prefix}has_bic', False)
        bic_valid_country = get(f'{prefix}bic_valid_country')
        
        if has_bic and bic_valid_country == False:
            party = prefix.rstrip('_').upper()
            reasons.append(f"{party}: has_bic but bic_valid_country=False")
            matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 9: BNF has account (general trigger for account validation)
    # -------------------------------------------------------------------------
    bnf_has_account = get('bnf_has_account', False)
    if bnf_has_account:
        reasons.append("BNF: has_account=True")
        matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 10: Intermediary with address lines > 3 (from RF Path 2)
    # -------------------------------------------------------------------------
    intm_address_lines = get('intm_address_lines', 0)
    if intm_address_lines and intm_address_lines > 3:
        reasons.append(f"INTM: address_lines={intm_address_lines} > 3")
        matches += 1
    
    should_fire = matches > 0
    return should_fire, reasons


def main():
    parser = argparse.ArgumentParser(description='Test RF 8022 rules against IFMLs')
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
    
    # Count 8022 codes in responses
    total_8022_in_responses = 0
    for record in pipeline.records:
        actual_codes = record.error_codes_only
        composite_codes = record.composite_codes or []
        if any('8022' in str(c) for c in actual_codes + composite_codes):
            total_8022_in_responses += 1
    
    print(f"\n📊 Found {total_8022_in_responses} IFMLs with 8022 in responses")
    
    # Track results
    successes = []
    failures = []
    total_8022 = 0
    processed = 0
    
    for record in pipeline.records:
        if processed >= args.limit:
            break
        
        actual_codes = record.error_codes_only
        composite_codes = record.composite_codes or []
        
        has_8022 = any('8022' in str(c) for c in actual_codes + composite_codes)
        
        if not has_8022:
            continue
        
        total_8022 += 1
        processed += 1
        
        txn_id = record.transaction_id
        features = record.request_features
        
        should_fire, reasons = check_8022_rf_rules(features)
        
        if should_fire:
            successes.append((txn_id, reasons))
            if args.verbose:
                print(f"✅ {txn_id}: {reasons[0]}")
        else:
            debug_info = {
                'bnf_has_iban': features.get('bnf_has_iban'),
                'bnf_needs_iban': features.get('bnf_needs_iban'),
                'bnf_has_account': features.get('bnf_has_account'),
                'bnf_has_bic': features.get('bnf_has_bic'),
                'bnf_id_has_bic_and_nch': features.get('bnf_id_has_bic_and_nch'),
                'send_bic_valid_country': features.get('send_bic_valid_country'),
                'intm_has_bic': features.get('intm_has_bic'),
                'intm_bic_valid_format': features.get('intm_bic_valid_format'),
                'total_address_lines': features.get('total_address_lines'),
            }
            failures.append((txn_id, composite_codes, debug_info))
            if args.verbose:
                print(f"❌ {txn_id}")
    
    # =========================================================================
    # REPORT
    # =========================================================================
    print("\n" + "="*70)
    print("TEST RESULTS: 8022 Rule Validation")
    print("="*70)
    
    print(f"\nTotal IFMLs with 8022: {total_8022}")
    print(f"Successes (rule matched): {len(successes)}")
    print(f"Failures (rule missed): {len(failures)}")
    
    if total_8022 > 0:
        success_rate = len(successes) / total_8022 * 100
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
            print(f"   Actual codes: {[c for c in codes if '8022' in str(c)]}")
            print(f"   Features: {debug_info}")
        
        if len(failures) > args.show_failures:
            print(f"\n... and {len(failures) - args.show_failures} more failures")
        
        # Analyze failure patterns
        print(f"\n{'='*70}")
        print("FAILURE PATTERN ANALYSIS:")
        print("="*70)
        
        failure_patterns = {
            'bnf_has_iban_true': 0,
            'bnf_needs_iban_true': 0,
            'bnf_has_account_true': 0,
            'bnf_has_bic_true': 0,
            'intm_has_bic_true': 0,
        }
        
        for txn_id, codes, debug in failures:
            if debug.get('bnf_has_iban'):
                failure_patterns['bnf_has_iban_true'] += 1
            if debug.get('bnf_needs_iban'):
                failure_patterns['bnf_needs_iban_true'] += 1
            if debug.get('bnf_has_account'):
                failure_patterns['bnf_has_account_true'] += 1
            if debug.get('bnf_has_bic'):
                failure_patterns['bnf_has_bic_true'] += 1
            if debug.get('intm_has_bic'):
                failure_patterns['intm_has_bic_true'] += 1
        
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
  
