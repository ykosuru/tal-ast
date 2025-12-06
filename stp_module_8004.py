#!/usr/bin/env python3
"""
Test 8004 Rules - V10 (Cross-Border Payments)

8004 = IBAN Cannot Be Derived

Key insight from V8/V9:
- V8 (IBAN only): FN for non-IBAN countries (MX, AU, TH)
- V9 (non-IBAN only): FN for IBAN countries (BE, GB, ES)
- 8004 fires for BOTH IBAN and non-IBAN countries!

The real pattern: 8004 fires for CROSS-BORDER (international) payments
- US→US domestic = no 8004 (gets 6021 instead)
- US→anywhere else = 8004 possible
- Anywhere→anywhere else = 8004 possible

Rules:
1. Cross-border payment (originator country != beneficiary country)
   AND cdt_has_account=True AND cdt_has_iban=False
2. missing_required_iban=True
"""

import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

sys.path.insert(0, '/mnt/project')
from data_pipeline import IFMLDataPipeline

TARGET_CODE = '8004'


def get_country(features: Dict, prefix: str) -> str:
    """Get country for a party from various sources."""
    country = (
        features.get(f'{prefix}_residence_country') or
        features.get(f'{prefix}_address_country') or
        features.get(f'{prefix}_country') or
        features.get(f'{prefix}_bic_country') or
        features.get(f'{prefix}_iban_country') or
        ''
    )
    return country.upper() if country else ''


def check_8004_rules(features: Dict) -> Tuple[bool, List[str]]:
    """
    Check if 8004 should be predicted.
    
    V10: Check for cross-border (international) payments
    """
    reasons = []
    
    def get(feat, default=None):
        return features.get(feat, default)
    
    # Get countries for originator (dbt) and beneficiary (cdt)
    dbt_country = get_country(features, 'dbt')
    cdt_country = get_country(features, 'cdt')
    
    # Also check orig_ prefix for originator
    if not dbt_country:
        dbt_country = get_country(features, 'orig')
    
    # Rule 1: Cross-border payment with account but no IBAN
    cdt_has_iban = get('cdt_has_iban', False)
    cdt_has_account = get('cdt_has_account', False)
    
    # Cross-border = countries are different (and both known)
    is_cross_border = (dbt_country and cdt_country and dbt_country != cdt_country)
    
    if is_cross_border and cdt_has_account and not cdt_has_iban:
        reasons.append(f"Cross-border: {dbt_country}→{cdt_country} + has_account + no IBAN")
    
    # Rule 2: System flag for missing required IBAN
    if get('missing_required_iban', False):
        reasons.append("missing_required_iban=True")
    
    return len(reasons) > 0, reasons


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Test 8004 rules (V10 - Cross-Border)')
    parser.add_argument('--data-dir', required=True, help='Directory with IFML JSON files')
    parser.add_argument('--sample-size', type=int, default=10, help='Number of samples to show')
    
    args = parser.parse_args()
    
    print("V10: Predict 8004 for CROSS-BORDER payments (originator != beneficiary country)")
    
    # Load data
    print("\nLoading IFML data...")
    pipeline = IFMLDataPipeline()
    
    data_path = Path(args.data_dir)
    if data_path.is_file():
        pipeline.load_single_file(str(data_path))
    else:
        json_files = sorted(data_path.glob("*.json"))
        for i, f in enumerate(json_files[:10]):
            count = pipeline.load_single_file(str(f))
            print(f"  [{i+1:2d}] {f.name}: {count} payment(s)")
        print("-" * 60)
    
    print(f"Pipeline has {len(pipeline.records)} records")
    
    # Test rules
    tp, fp, tn, fn = 0, 0, 0, 0
    fp_list = []
    fn_list = []
    tp_list = []
    
    prediction_triggers = defaultdict(int)
    
    for record in pipeline.records:
        features = record.request_features
        codes = record.error_codes_only + (record.composite_codes or [])
        
        # Check if 8004 actually fired
        actual_8004 = any(TARGET_CODE in str(c) for c in codes)
        
        # Predict using our rules
        predicted, reasons = check_8004_rules(features)
        
        # Track what triggered predictions
        for r in reasons:
            if "Cross-border" in r:
                prediction_triggers["Cross-border + has_account + no IBAN"] += 1
            else:
                prediction_triggers[r] += 1
        
        # Confusion matrix
        if predicted and actual_8004:
            tp += 1
            tp_list.append((record.transaction_id, reasons, codes, features))
        elif predicted and not actual_8004:
            fp += 1
            fp_list.append((record.transaction_id, reasons, codes, features))
        elif not predicted and actual_8004:
            fn += 1
            fn_list.append((record.transaction_id, reasons, codes, features))
        else:
            tn += 1
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Print results
    print("\n" + "=" * 70)
    print(f"TEST RESULTS: 8004 - IBAN Cannot Be Derived (V10 - Cross-Border)")
    print("=" * 70)
    
    print(f"\nCONFUSION MATRIX:")
    print(f"                 Actual")
    print(f"                 8004        Not 8004")
    print(f"Predicted")
    print(f"  8004      |  TP={tp:<6}  |  FP={fp:<6}  |")
    print(f"  Not 8004  |  FN={fn:<6}  |  TN={tn:<6}  |")
    
    print(f"\nMETRICS:")
    print(f"  Precision:  {precision*100:.2f}%  (TP / (TP + FP))")
    print(f"  Recall:     {recall*100:.2f}%  (TP / (TP + FN))")
    print(f"  F1 Score:   {f1*100:.2f}%")
    
    print(f"\nRULES (V10 - CROSS-BORDER):")
    print(f"  1. dbt_country != cdt_country AND cdt_has_account=True AND cdt_has_iban=False")
    print(f"  2. missing_required_iban=True")
    print(f"\n  KEY INSIGHT: 8004 fires for cross-border payments regardless of IBAN status")
    
    print(f"\nPREDICTION TRIGGERS:")
    for trigger, count in sorted(prediction_triggers.items(), key=lambda x: -x[1]):
        print(f"  {trigger}: {count}")
    
    # Show FP analysis
    print("\n" + "=" * 70)
    print(f"FALSE POSITIVES ({len(fp_list)} total, showing {min(args.sample_size, len(fp_list))}):")
    print("=" * 70)
    
    # Analyze what codes ACTUALLY fired
    fp_actual_codes = defaultdict(int)
    for _, _, codes, _ in fp_list:
        for c in codes:
            code_base = str(c).split('_')[0]
            fp_actual_codes[code_base] += 1
    
    print(f"\nWhat codes ACTUALLY fired (instead of 8004):")
    for code, count in sorted(fp_actual_codes.items(), key=lambda x: -x[1])[:10]:
        print(f"  {code}: {count}")
    
    # Analyze FP by route
    fp_routes = defaultdict(int)
    for _, _, _, features in fp_list[:200]:
        dbt = get_country(features, 'dbt') or get_country(features, 'orig')
        cdt = get_country(features, 'cdt')
        if dbt and cdt:
            fp_routes[f"{dbt}→{cdt}"] += 1
    
    print(f"\nFP by route (originator→beneficiary):")
    for route, count in sorted(fp_routes.items(), key=lambda x: -x[1])[:10]:
        print(f"  {route}: {count}")
    
    print(f"\nSample FPs:")
    for i, (txn_id, reasons, codes, features) in enumerate(fp_list[:args.sample_size]):
        dbt = get_country(features, 'dbt') or get_country(features, 'orig')
        cdt = get_country(features, 'cdt')
        print(f"  {i+1}. {txn_id}")
        print(f"     Route: {dbt}→{cdt}")
        print(f"     Actual codes: {codes[:5]}")
    
    # Show FN analysis
    print("\n" + "=" * 70)
    print(f"FALSE NEGATIVES ({len(fn_list)} total, showing {min(args.sample_size, len(fn_list))}):")
    print("=" * 70)
    
    # Analyze patterns in FN
    fn_patterns = defaultdict(int)
    fn_routes = defaultdict(int)
    for _, _, codes, features in fn_list[:100]:
        dbt = get_country(features, 'dbt') or get_country(features, 'orig')
        cdt = get_country(features, 'cdt')
        
        if dbt and cdt:
            fn_routes[f"{dbt}→{cdt}"] += 1
            if dbt == cdt:
                fn_patterns['same_country (domestic)'] += 1
            else:
                fn_patterns['different_country (cross-border)'] += 1
        else:
            if not dbt:
                fn_patterns['dbt_country=None'] += 1
            if not cdt:
                fn_patterns['cdt_country=None'] += 1
            
        if features.get('cdt_has_iban') == False:
            fn_patterns['cdt_has_iban=False'] += 1
        if features.get('cdt_has_account') == True:
            fn_patterns['cdt_has_account=True'] += 1
        if features.get('cdt_has_account') == False:
            fn_patterns['cdt_has_account=False'] += 1
    
    print(f"\nPattern Analysis (features in FN cases):")
    for pattern, count in sorted(fn_patterns.items(), key=lambda x: -x[1])[:10]:
        pct = count / min(100, len(fn_list)) * 100
        print(f"  {pattern}: {count} ({pct:.1f}%)")
    
    print(f"\nFN by route:")
    for route, count in sorted(fn_routes.items(), key=lambda x: -x[1])[:10]:
        print(f"  {route}: {count}")
    
    print(f"\nSample FNs:")
    for i, (txn_id, reasons, codes, features) in enumerate(fn_list[:args.sample_size]):
        dbt = get_country(features, 'dbt') or get_country(features, 'orig')
        cdt = get_country(features, 'cdt')
        print(f"  {i+1}. {txn_id}")
        print(f"     Codes: {[c for c in codes if TARGET_CODE in str(c)]}")
        print(f"     Route: {dbt}→{cdt}")
        print(f"     cdt_has_account={features.get('cdt_has_account')}")
        print(f"     cdt_has_iban={features.get('cdt_has_iban')}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    recall_status = "✓" if recall >= 0.95 else "✗"
    precision_status = "✓" if precision >= 0.30 else "⚠" if precision >= 0.10 else "✗"
    
    print(f"\n  Recall:     {recall*100:.1f}% {recall_status}")
    print(f"  Precision:  {precision*100:.1f}% {precision_status}")
    print(f"  F1 Score:   {f1*100:.1f}%")
    
    print(f"\nV10 Changes:")
    print(f"  - Check for cross-border (originator country != beneficiary country)")
    print(f"  - 8004 fires for international payments regardless of IBAN/non-IBAN")
    print(f"  - US→US domestic should NOT trigger (gets 6021 instead)")


if __name__ == "__main__":
    main()
