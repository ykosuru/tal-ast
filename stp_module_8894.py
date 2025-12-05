#!/usr/bin/env python3
"""
Test extracted RandomForest rules for 8894 against actual IFMLs.
8894 = IBAN validation issue / checksum failure

Usage:
    python test_8894_rules.py --data-dir /path/to/ifml/data --limit 1000
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add project path for imports
sys.path.insert(0, '/mnt/project')

from data_pipeline import IFMLDataPipeline


def check_8894_rf_rules(features: Dict) -> Tuple[bool, List[str]]:
    """
    Check if 8894 should fire based on RF-extracted rules.
    8894 = IBAN validation issue / checksum failure
    
    RF Patterns (from screenshots):
    For 8894_BNPPTY:
    - dbt_account_has_dirty_chars = True (100%)
    - dbt_is_international = True (100%)
    - needs_iban_count = False (80%)
    - iban_checksum_valid_count = False (79%)
    - cdt_account_has_dirty_chars = True (70%)
    
    Returns (should_fire, reasons)
    """
    reasons = []
    
    def get(feat, default=None):
        if feat in features:
            return features[feat]
        return default
    
    matches = 0
    
    # -------------------------------------------------------------------------
    # Rule 1: Account has dirty chars (strongest signal - 100% for dbt, 70% for cdt)
    # -------------------------------------------------------------------------
    for prefix in ['dbt_', 'cdt_', 'bnf_', 'orig_', 'intm_']:
        has_dirty_chars = get(f'{prefix}account_has_dirty_chars', False)
        if has_dirty_chars:
            party = prefix.rstrip('_').upper()
            reasons.append(f"{party}: account_has_dirty_chars=True")
            matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 2: IBAN checksum invalid (79% ratio)
    # -------------------------------------------------------------------------
    for prefix in ['bnf_', 'cdt_', 'dbt_', 'orig_', 'intm_']:
        has_iban = get(f'{prefix}has_iban', False)
        iban_checksum_valid = get(f'{prefix}iban_checksum_valid', True)
        
        if has_iban and not iban_checksum_valid:
            party = prefix.rstrip('_').upper()
            reasons.append(f"{party}: has_iban but checksum invalid")
            matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 3: IBAN format invalid
    # -------------------------------------------------------------------------
    for prefix in ['bnf_', 'cdt_', 'dbt_', 'orig_', 'intm_']:
        has_iban = get(f'{prefix}has_iban', False)
        iban_valid_format = get(f'{prefix}iban_valid_format', True)
        
        if has_iban and not iban_valid_format:
            party = prefix.rstrip('_').upper()
            reasons.append(f"{party}: has_iban but format invalid")
            matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 4: International + dirty chars combo (100% pattern)
    # -------------------------------------------------------------------------
    dbt_is_international = get('dbt_is_international', False)
    dbt_has_dirty = get('dbt_account_has_dirty_chars', False)
    
    if dbt_is_international and dbt_has_dirty:
        reasons.append("DBT: is_international + account_has_dirty_chars")
        matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 5: bnf_has_iban_no_bic pattern (from RF Path 1)
    # -------------------------------------------------------------------------
    bnf_has_iban = get('bnf_has_iban', False)
    bnf_has_bic = get('bnf_has_bic', False)
    
    if bnf_has_iban and not bnf_has_bic:
        reasons.append("BNF: has_iban but no BIC")
        matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 6: Cross-border with BIC/IBAN match issues (from RF Path 3)
    # -------------------------------------------------------------------------
    is_cross_border = get('is_cross_border', False)
    bnf_bic_iban_match = get('bnf_bic_iban_match')
    
    if is_cross_border and bnf_bic_iban_match == True:
        # RF Path 3 shows this pattern
        cdt_account_numeric = get('cdt_account_numeric', True)
        if cdt_account_numeric == False or cdt_account_numeric == 0:
            reasons.append("Cross-border + bnf_bic_iban_match + cdt non-numeric account")
            matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 7: bnf_id_has_nch_pattern (from multiple RF paths)
    # -------------------------------------------------------------------------
    bnf_id_has_nch_pattern = get('bnf_id_has_nch_pattern', False)
    if bnf_id_has_nch_pattern:
        reasons.append("BNF: id_has_nch_pattern=True")
        matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 8: bnf_id_has_bic_pattern (from RF Path 1 for BNPPTY)
    # -------------------------------------------------------------------------
    bnf_id_has_bic_pattern = get('bnf_id_has_bic_pattern', False)
    if bnf_id_has_bic_pattern:
        reasons.append("BNF: id_has_bic_pattern=True")
        matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 9: cdt_is_iban_derivable with domestic (from RF Path 2)
    # -------------------------------------------------------------------------
    is_any_domestic = get('is_any_domestic', False)
    cdt_is_iban_derivable = get('cdt_is_iban_derivable', False)
    
    if is_any_domestic and cdt_is_iban_derivable:
        reasons.append("Domestic + cdt_is_iban_derivable")
        matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 10: bnf_has_duplicate_info (from RF Path 3 for general 8894)
    # -------------------------------------------------------------------------
    bnf_has_duplicate_info = get('bnf_has_duplicate_info', False)
    if bnf_has_duplicate_info:
        reasons.append("BNF: has_duplicate_info=True")
        matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 11: needs_iban with issues
    # -------------------------------------------------------------------------
    for prefix in ['bnf_', 'cdt_', 'dbt_', 'orig_']:
        needs_iban = get(f'{prefix}needs_iban', False)
        has_iban = get(f'{prefix}has_iban', False)
        
        if needs_iban and has_iban:
            # Has IBAN when needed - check if it's valid
            iban_valid = get(f'{prefix}iban_valid_format', True)
            checksum_valid = get(f'{prefix}iban_checksum_valid', True)
            
            if not iban_valid or not checksum_valid:
                party = prefix.rstrip('_').upper()
                reasons.append(f"{party}: needs_iban, has_iban but validation failed")
                matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 12: needs_iban triggers IBAN processing - ACE may find issues
    # Pattern from failures: bnf_needs_iban=True even when our validation passes
    # -------------------------------------------------------------------------
    for prefix in ['bnf_', 'cdt_', 'dbt_', 'orig_']:
        needs_iban = get(f'{prefix}needs_iban', False)
        if needs_iban:
            party = prefix.rstrip('_').upper()
            reasons.append(f"{party}: needs_iban=True (IBAN processing triggered)")
            matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 13: dbt_is_international - triggers additional validation
    # -------------------------------------------------------------------------
    dbt_is_international = get('dbt_is_international', False)
    if dbt_is_international:
        reasons.append("DBT: is_international=True")
        matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 14: BNF missing both IBAN and BIC - missing critical identifiers
    # -------------------------------------------------------------------------
    bnf_has_iban = get('bnf_has_iban', False)
    bnf_has_bic = get('bnf_has_bic', False)
    bnf_present = get('bnf_present', False)
    
    if bnf_present and not bnf_has_iban and not bnf_has_bic:
        reasons.append("BNF: missing both IBAN and BIC")
        matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 15: BNF has both IBAN and BIC - ACE validates the pair
    # -------------------------------------------------------------------------
    if bnf_has_iban and bnf_has_bic:
        reasons.append("BNF: has both IBAN and BIC (ACE validates pair)")
        matches += 1
    
    should_fire = matches > 0
    return should_fire, reasons


def main():
    parser = argparse.ArgumentParser(description='Test RF 8894 rules against IFMLs')
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
    
    # Count 8894 codes in responses
    total_8894_in_responses = 0
    for record in pipeline.records:
        actual_codes = record.error_codes_only
        composite_codes = record.composite_codes or []
        if any('8894' in str(c) for c in actual_codes + composite_codes):
            total_8894_in_responses += 1
    
    print(f"\n📊 Found {total_8894_in_responses} IFMLs with 8894 in responses")
    
    # Track results
    successes = []
    failures = []
    total_8894 = 0
    processed = 0
    
    for record in pipeline.records:
        if processed >= args.limit:
            break
        
        actual_codes = record.error_codes_only
        composite_codes = record.composite_codes or []
        
        has_8894 = any('8894' in str(c) for c in actual_codes + composite_codes)
        
        if not has_8894:
            continue
        
        total_8894 += 1
        processed += 1
        
        txn_id = record.transaction_id
        features = record.request_features
        
        should_fire, reasons = check_8894_rf_rules(features)
        
        if should_fire:
            successes.append((txn_id, reasons))
            if args.verbose:
                print(f"✅ {txn_id}: {reasons[0]}")
        else:
            debug_info = {
                'dbt_account_has_dirty_chars': features.get('dbt_account_has_dirty_chars'),
                'cdt_account_has_dirty_chars': features.get('cdt_account_has_dirty_chars'),
                'bnf_account_has_dirty_chars': features.get('bnf_account_has_dirty_chars'),
                'dbt_is_international': features.get('dbt_is_international'),
                'bnf_has_iban': features.get('bnf_has_iban'),
                'bnf_iban_valid_format': features.get('bnf_iban_valid_format'),
                'bnf_iban_checksum_valid': features.get('bnf_iban_checksum_valid'),
                'bnf_has_bic': features.get('bnf_has_bic'),
                'is_cross_border': features.get('is_cross_border'),
                'bnf_id_has_nch_pattern': features.get('bnf_id_has_nch_pattern'),
                'bnf_needs_iban': features.get('bnf_needs_iban'),
            }
            failures.append((txn_id, composite_codes, debug_info))
            if args.verbose:
                print(f"❌ {txn_id}")
    
    # =========================================================================
    # REPORT
    # =========================================================================
    print("\n" + "="*70)
    print("TEST RESULTS: 8894 Rule Validation")
    print("="*70)
    
    print(f"\nTotal IFMLs with 8894: {total_8894}")
    print(f"Successes (rule matched): {len(successes)}")
    print(f"Failures (rule missed): {len(failures)}")
    
    if total_8894 > 0:
        success_rate = len(successes) / total_8894 * 100
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
            print(f"   Actual codes: {[c for c in codes if '8894' in str(c)]}")
            print(f"   Features: {debug_info}")
        
        if len(failures) > args.show_failures:
            print(f"\n... and {len(failures) - args.show_failures} more failures")
        
        # Analyze failure patterns
        print(f"\n{'='*70}")
        print("FAILURE PATTERN ANALYSIS:")
        print("="*70)
        
        failure_patterns = {
            'dbt_account_has_dirty_chars_true': 0,
            'cdt_account_has_dirty_chars_true': 0,
            'bnf_account_has_dirty_chars_true': 0,
            'dbt_is_international_true': 0,
            'bnf_has_iban_true': 0,
            'bnf_iban_valid_format_false': 0,
            'bnf_iban_checksum_valid_false': 0,
            'is_cross_border_true': 0,
        }
        
        for txn_id, codes, debug in failures:
            if debug.get('dbt_account_has_dirty_chars'):
                failure_patterns['dbt_account_has_dirty_chars_true'] += 1
            if debug.get('cdt_account_has_dirty_chars'):
                failure_patterns['cdt_account_has_dirty_chars_true'] += 1
            if debug.get('bnf_account_has_dirty_chars'):
                failure_patterns['bnf_account_has_dirty_chars_true'] += 1
            if debug.get('dbt_is_international'):
                failure_patterns['dbt_is_international_true'] += 1
            if debug.get('bnf_has_iban'):
                failure_patterns['bnf_has_iban_true'] += 1
            if debug.get('bnf_has_iban') and not debug.get('bnf_iban_valid_format'):
                failure_patterns['bnf_iban_valid_format_false'] += 1
            if debug.get('bnf_has_iban') and not debug.get('bnf_iban_checksum_valid'):
                failure_patterns['bnf_iban_checksum_valid_false'] += 1
            if debug.get('is_cross_border'):
                failure_patterns['is_cross_border_true'] += 1
        
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
