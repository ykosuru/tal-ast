#!/usr/bin/env python3
"""
Test extracted RandomForest rules for 8896 against actual IFMLs.
8896 = IBAN/BIC validation issue

Usage:
    python test_8896_rules.py --data-dir /path/to/ifml/data --limit 1000
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add project path for imports
sys.path.insert(0, '/mnt/project')

from data_pipeline import IFMLDataPipeline


def check_8896_rf_rules(features: Dict) -> Tuple[bool, List[str]]:
    """
    Check if 8896 should fire based on RF-extracted rules.
    8896 = IBAN/BIC validation issue
    
    RF Patterns (from screenshot):
    For 8896_BNPPTY:
    Path 1: bnf_has_id=False, bnf_is_international=True, bnf_iban_valid_format=True
    Path 2: bnf_bic_valid_format=False, missing_required_iban=True, is_any_domestic=True
    Path 3: missing_bnf_bic=True, bnf_iban_valid_format=True
    
    Returns (should_fire, reasons)
    """
    reasons = []
    
    def get(feat, default=None):
        if feat in features:
            return features[feat]
        return default
    
    matches = 0
    
    # -------------------------------------------------------------------------
    # Rule 1: bnf_iban_valid_format = True (high confidence, Path 1 & 3)
    # -------------------------------------------------------------------------
    bnf_iban_valid_format = get('bnf_iban_valid_format', False)
    if bnf_iban_valid_format:
        reasons.append("BNF: iban_valid_format=True")
        matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 2: bnf_is_international = True (Path 1)
    # -------------------------------------------------------------------------
    bnf_is_international = get('bnf_is_international', False)
    if bnf_is_international:
        reasons.append("BNF: is_international=True")
        matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 3: bnf_has_id = False (Path 1)
    # -------------------------------------------------------------------------
    bnf_has_id = get('bnf_has_id', True)
    if bnf_has_id == False:
        reasons.append("BNF: has_id=False")
        matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 4: missing_bnf_bic = True (Path 3)
    # -------------------------------------------------------------------------
    missing_bnf_bic = get('missing_bnf_bic', False)
    if missing_bnf_bic:
        reasons.append("missing_bnf_bic=True")
        matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 5: bnf_bic_valid_format = False (Path 2)
    # -------------------------------------------------------------------------
    bnf_bic_valid_format = get('bnf_bic_valid_format')
    bnf_has_bic = get('bnf_has_bic', False)
    if bnf_has_bic and bnf_bic_valid_format == False:
        reasons.append("BNF: has_bic but bic_valid_format=False")
        matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 6: missing_required_iban = True (Path 2)
    # -------------------------------------------------------------------------
    missing_required_iban = get('missing_required_iban', False)
    if missing_required_iban:
        reasons.append("missing_required_iban=True")
        matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 7: is_any_domestic = True (Path 2)
    # -------------------------------------------------------------------------
    is_any_domestic = get('is_any_domestic', False)
    if is_any_domestic:
        reasons.append("is_any_domestic=True")
        matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 8: cdt_is_domestic = True (Path 1)
    # -------------------------------------------------------------------------
    cdt_is_domestic = get('cdt_is_domestic', False)
    if cdt_is_domestic:
        reasons.append("CDT: is_domestic=True")
        matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 9: primary_amount = False (Path 1)
    # -------------------------------------------------------------------------
    primary_amount = get('primary_amount', True)
    if primary_amount == False:
        reasons.append("primary_amount=False")
        matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 10: orig_present = False (Path 3)
    # -------------------------------------------------------------------------
    orig_present = get('orig_present', True)
    if orig_present == False:
        reasons.append("orig_present=False")
        matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 11: intm_has_bban_in_iban_country > 3 (Path 2)
    # -------------------------------------------------------------------------
    intm_has_bban_in_iban_country = get('intm_has_bban_in_iban_country', 0) or 0
    if intm_has_bban_in_iban_country > 3:
        reasons.append(f"INTM: has_bban_in_iban_country={intm_has_bban_in_iban_country}")
        matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 12: BNF has IBAN but no BIC (common 8896 pattern)
    # -------------------------------------------------------------------------
    bnf_has_iban = get('bnf_has_iban', False)
    bnf_has_bic = get('bnf_has_bic', False)
    if bnf_has_iban and not bnf_has_bic:
        reasons.append("BNF: has_iban but no BIC")
        matches += 1
    
    # -------------------------------------------------------------------------
    # Rule 13: BNF needs IBAN (IBAN derivation triggered)
    # -------------------------------------------------------------------------
    bnf_needs_iban = get('bnf_needs_iban', False)
    if bnf_needs_iban:
        reasons.append("BNF: needs_iban=True")
        matches += 1
    
    should_fire = matches > 0
    return should_fire, reasons


def main():
    parser = argparse.ArgumentParser(description='Test RF 8896 rules against IFMLs')
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
    
    # Count 8896 codes in responses
    total_8896_in_responses = 0
    for record in pipeline.records:
        actual_codes = record.error_codes_only
        composite_codes = record.composite_codes or []
        if any('8896' in str(c) for c in actual_codes + composite_codes):
            total_8896_in_responses += 1
    
    print(f"\n📊 Found {total_8896_in_responses} IFMLs with 8896 in responses")
    
    # Track results
    successes = []
    failures = []
    total_8896 = 0
    processed = 0
    
    for record in pipeline.records:
        if processed >= args.limit:
            break
        
        actual_codes = record.error_codes_only
        composite_codes = record.composite_codes or []
        
        has_8896 = any('8896' in str(c) for c in actual_codes + composite_codes)
        
        if not has_8896:
            continue
        
        total_8896 += 1
        processed += 1
        
        txn_id = record.transaction_id
        features = record.request_features
        
        should_fire, reasons = check_8896_rf_rules(features)
        
        if should_fire:
            successes.append((txn_id, reasons))
            if args.verbose:
                print(f"✅ {txn_id}: {reasons[0]}")
        else:
            debug_info = {
                'bnf_iban_valid_format': features.get('bnf_iban_valid_format'),
                'bnf_is_international': features.get('bnf_is_international'),
                'bnf_has_id': features.get('bnf_has_id'),
                'bnf_has_bic': features.get('bnf_has_bic'),
                'bnf_bic_valid_format': features.get('bnf_bic_valid_format'),
                'bnf_has_iban': features.get('bnf_has_iban'),
                'bnf_needs_iban': features.get('bnf_needs_iban'),
                'missing_bnf_bic': features.get('missing_bnf_bic'),
                'missing_required_iban': features.get('missing_required_iban'),
                'is_any_domestic': features.get('is_any_domestic'),
                'cdt_is_domestic': features.get('cdt_is_domestic'),
            }
            failures.append((txn_id, composite_codes, debug_info))
            if args.verbose:
                print(f"❌ {txn_id}")
    
    # =========================================================================
    # REPORT
    # =========================================================================
    print("\n" + "="*70)
    print("TEST RESULTS: 8896 Rule Validation")
    print("="*70)
    
    print(f"\nTotal IFMLs with 8896: {total_8896}")
    print(f"Successes (rule matched): {len(successes)}")
    print(f"Failures (rule missed): {len(failures)}")
    
    if total_8896 > 0:
        success_rate = len(successes) / total_8896 * 100
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
            print(f"   Actual codes: {[c for c in codes if '8896' in str(c)]}")
            print(f"   Features: {debug_info}")
        
        if len(failures) > args.show_failures:
            print(f"\n... and {len(failures) - args.show_failures} more failures")
        
        # Analyze failure patterns
        print(f"\n{'='*70}")
        print("FAILURE PATTERN ANALYSIS:")
        print("="*70)
        
        failure_patterns = {
            'bnf_iban_valid_format_true': 0,
            'bnf_is_international_true': 0,
            'bnf_has_id_false': 0,
            'bnf_has_bic_true': 0,
            'bnf_has_iban_true': 0,
            'missing_bnf_bic_true': 0,
            'is_any_domestic_true': 0,
        }
        
        for txn_id, codes, debug in failures:
            if debug.get('bnf_iban_valid_format'):
                failure_patterns['bnf_iban_valid_format_true'] += 1
            if debug.get('bnf_is_international'):
                failure_patterns['bnf_is_international_true'] += 1
            if debug.get('bnf_has_id') == False:
                failure_patterns['bnf_has_id_false'] += 1
            if debug.get('bnf_has_bic'):
                failure_patterns['bnf_has_bic_true'] += 1
            if debug.get('bnf_has_iban'):
                failure_patterns['bnf_has_iban_true'] += 1
            if debug.get('missing_bnf_bic'):
                failure_patterns['missing_bnf_bic_true'] += 1
            if debug.get('is_any_domestic'):
                failure_patterns['is_any_domestic_true'] += 1
        
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
