#!/usr/bin/env python3
"""
Test 8004 Rules - V9 (NON-IBAN Countries)

8004 = IBAN Cannot Be Derived

Key insight from V8:
- FN: 8004 fires for MX, AU, TH, PH, IN, CA, ZA, JP (NON-IBAN countries!)
- FP: 8004 does NOT fire for BE, GB, CH, ES (IBAN countries get 6021 instead)

The logic is INVERTED:
- For IBAN countries: other validation (6021) runs first
- For NON-IBAN countries: ACE tries bank lookup, fails → 8004

Rules:
1. CDT country NOT IN IBAN_COUNTRIES AND cdt_has_account=True AND cdt_has_iban=False
2. missing_required_iban=True
"""

import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

sys.path.insert(0, '/mnt/project')
from data_pipeline import IFMLDataPipeline
from ifml_parser import IBAN_LENGTHS

TARGET_CODE = '8004'

# IBAN countries from the parser (73 countries)
IBAN_COUNTRIES = set(IBAN_LENGTHS.keys())


def get_cdt_country(features: Dict) -> str:
    """Get CDT country from various sources."""
    return (
        features.get('cdt_residence_country') or
        features.get('cdt_address_country') or
        features.get('cdt_country') or
        features.get('cdt_bic_country') or
        features.get('cdt_iban_country') or
        ''
    )


def check_8004_rules(features: Dict) -> Tuple[bool, List[str]]:
    """
    Check if 8004 should be predicted.
    
    V9: Check if CDT country is NOT in IBAN_COUNTRIES (inverted logic!)
    """
    reasons = []
    
    def get(feat, default=None):
        return features.get(feat, default)
    
    # Get CDT country from various sources
    cdt_country = get_cdt_country(features).upper() if get_cdt_country(features) else ''
    
    # Rule 1: CDT is NOT in IBAN country, has account, but no IBAN
    cdt_has_iban = get('cdt_has_iban', False)
    cdt_has_account = get('cdt_has_account', False)
    
    # Check if country does NOT use IBAN (and country is known)
    is_non_iban_country = cdt_country and cdt_country not in IBAN_COUNTRIES
    
    if is_non_iban_country and cdt_has_account and not cdt_has_iban:
        reasons.append(f"CDT: country={cdt_country} (non-IBAN) + has_account + no IBAN")
    
    # Rule 2: System flag for missing required IBAN
    if get('missing_required_iban', False):
        reasons.append("missing_required_iban=True")
    
    return len(reasons) > 0, reasons


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Test 8004 rules (V9 - Non-IBAN)')
    parser.add_argument('--data-dir', required=True, help='Directory with IFML JSON files')
    parser.add_argument('--sample-size', type=int, default=10, help='Number of samples to show')
    
    args = parser.parse_args()
    
    print(f"IBAN_COUNTRIES has {len(IBAN_COUNTRIES)} countries")
    print(f"V9: Predict 8004 for countries NOT in IBAN_COUNTRIES")
    
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
            if "non-IBAN" in r:
                prediction_triggers["CDT: non-IBAN country + has_account + no IBAN"] += 1
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
    print(f"TEST RESULTS: 8004 - IBAN Cannot Be Derived (V9 - Non-IBAN)")
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
    
    print(f"\nRULES (V9 - NON-IBAN COUNTRIES):")
    print(f"  1. cdt_country NOT IN IBAN_COUNTRIES AND cdt_has_account=True AND cdt_has_iban=False")
    print(f"  2. missing_required_iban=True")
    print(f"\n  KEY INSIGHT: 8004 fires for NON-IBAN countries (MX, AU, TH, etc.)")
    print(f"               IBAN countries get 6021 instead")
    
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
    
    # Analyze FP countries
    fp_countries = defaultdict(int)
    for _, _, _, features in fp_list[:200]:
        country = get_cdt_country(features).upper() if get_cdt_country(features) else ''
        if country:
            fp_countries[country] += 1
    
    print(f"\nFP by CDT country:")
    for country, count in sorted(fp_countries.items(), key=lambda x: -x[1])[:10]:
        in_iban = "IBAN" if country in IBAN_COUNTRIES else "non-IBAN"
        print(f"  {country}: {count} ({in_iban})")
    
    print(f"\nSample FPs:")
    for i, (txn_id, reasons, codes, features) in enumerate(fp_list[:args.sample_size]):
        cdt_country = get_cdt_country(features).upper() if get_cdt_country(features) else ''
        in_iban = "IBAN" if cdt_country in IBAN_COUNTRIES else "non-IBAN"
        print(f"  {i+1}. {txn_id}")
        print(f"     CDT country: {cdt_country} ({in_iban})")
        print(f"     Actual codes: {codes[:5]}")
    
    # Show FN analysis
    print("\n" + "=" * 70)
    print(f"FALSE NEGATIVES ({len(fn_list)} total, showing {min(args.sample_size, len(fn_list))}):")
    print("=" * 70)
    
    # Analyze patterns in FN
    fn_patterns = defaultdict(int)
    fn_countries = defaultdict(int)
    for _, _, codes, features in fn_list[:100]:
        cdt_country = get_cdt_country(features).upper() if get_cdt_country(features) else ''
        
        if cdt_country:
            fn_countries[cdt_country] += 1
            in_iban = cdt_country in IBAN_COUNTRIES
            fn_patterns[f'in_IBAN_COUNTRIES={in_iban}'] += 1
        else:
            fn_patterns['cdt_country=None'] += 1
            
        if features.get('cdt_has_iban') == False:
            fn_patterns['cdt_has_iban=False'] += 1
        if features.get('cdt_has_account') == True:
            fn_patterns['cdt_has_account=True'] += 1
        if features.get('cdt_has_account') == False:
            fn_patterns['cdt_has_account=False'] += 1
    
    print(f"\nPattern Analysis (CDT features in FN cases):")
    for pattern, count in sorted(fn_patterns.items(), key=lambda x: -x[1])[:10]:
        pct = count / min(100, len(fn_list)) * 100
        print(f"  {pattern}: {count} ({pct:.1f}%)")
    
    print(f"\nFN by CDT country:")
    for country, count in sorted(fn_countries.items(), key=lambda x: -x[1])[:10]:
        in_iban = "IBAN" if country in IBAN_COUNTRIES else "non-IBAN"
        print(f"  {country}: {count} ({in_iban})")
    
    print(f"\nSample FNs:")
    for i, (txn_id, reasons, codes, features) in enumerate(fn_list[:args.sample_size]):
        cdt_country = get_cdt_country(features).upper() if get_cdt_country(features) else ''
        in_iban = "IBAN" if cdt_country in IBAN_COUNTRIES else "non-IBAN"
        print(f"  {i+1}. {txn_id}")
        print(f"     Codes: {[c for c in codes if TARGET_CODE in str(c)]}")
        print(f"     cdt_country={cdt_country} ({in_iban})")
        print(f"     cdt_has_account={features.get('cdt_has_account')}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    recall_status = "✓" if recall >= 0.95 else "✗"
    precision_status = "✓" if precision >= 0.30 else "⚠" if precision >= 0.10 else "✗"
    
    print(f"\n  Recall:     {recall*100:.1f}% {recall_status}")
    print(f"  Precision:  {precision*100:.1f}% {precision_status}")
    print(f"  F1 Score:   {f1*100:.1f}%")
    
    print(f"\nV9 Changes:")
    print(f"  - INVERTED logic from V8")
    print(f"  - Predict 8004 for NON-IBAN countries (MX, AU, TH, IN, etc.)")
    print(f"  - Don't predict for IBAN countries (they get 6021 instead)")


if __name__ == "__main__":
    main()
