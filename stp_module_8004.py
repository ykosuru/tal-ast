#!/usr/bin/env python3
"""
Test 8004 Rules - V13 (IBAN-based filtering, no hardcoding)

Logic:
- IBAN countries have standardized bank directories → ACE validates via directory → 6021
- US has strong domestic bank validation → 6021
- Non-IBAN countries (TH, PH, AU, IN, JP, NZ, ZA, MX, CA, etc.) → no directory → 8004

Rule:
- If beneficiary is in IBAN country OR US → don't predict 8004
- If beneficiary is in non-IBAN country (and not US) → predict 8004
"""

import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Set

sys.path.insert(0, '/mnt/project')
from data_pipeline import IFMLDataPipeline
from ifml_parser import IBAN_LENGTHS

TARGET_CODE = '8004'

# Get IBAN countries from parser (no hardcoding!)
IBAN_COUNTRIES = set(IBAN_LENGTHS.keys())

# US has strong bank directory validation (ABA/routing)
# So exclude IBAN countries + US
EXCLUDE_BENEFICIARY_COUNTRIES = IBAN_COUNTRIES | {'US'}

US_STATE_CODES = {
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
    'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
    'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
    'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
    'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY',
    'DC', 'PR', 'VI', 'GU', 'AS', 'MP'
}


def get_country(features: Dict, prefix: str) -> str:
    country = (
        features.get(f'{prefix}_residence_country') or
        features.get(f'{prefix}_address_country') or
        features.get(f'{prefix}_country') or
        features.get(f'{prefix}_bic_country') or
        features.get(f'{prefix}_iban_country') or
        ''
    )
    return country.upper() if country else ''


def normalize_originator_country(code: str) -> str:
    if code and len(code) == 2 and code in US_STATE_CODES:
        return 'US'
    return code


def check_8004_rules(features: Dict) -> Tuple[bool, List[str]]:
    """
    Check if 8004 should be predicted.
    
    V13: Only predict 8004 for non-IBAN, non-US beneficiary countries
    """
    reasons = []
    
    def get(feat, default=None):
        return features.get(feat, default)
    
    # Get countries
    dbt_country_raw = get_country(features, 'dbt')
    if not dbt_country_raw:
        dbt_country_raw = get_country(features, 'orig')
    cdt_country = get_country(features, 'cdt')
    
    dbt_country = normalize_originator_country(dbt_country_raw)
    
    # FILTER: Only predict 8004 for non-IBAN, non-US beneficiaries
    # IBAN countries and US have bank directory validation → get 6021 instead
    if cdt_country in EXCLUDE_BENEFICIARY_COUNTRIES:
        return False, []
    
    # Rule 1: Cross-border or unknown originator with international beneficiary
    cdt_has_iban = get('cdt_has_iban', False)
    cdt_has_account = get('cdt_has_account', False)
    
    if dbt_country and cdt_country:
        is_cross_border = (dbt_country != cdt_country)
    elif cdt_country and cdt_country != 'US':
        is_cross_border = True
    else:
        is_cross_border = False
    
    if is_cross_border and cdt_has_account and not cdt_has_iban:
        if dbt_country:
            reasons.append(f"Cross-border: {dbt_country}→{cdt_country} (non-IBAN) + has_account + no IBAN")
        else:
            reasons.append(f"International beneficiary: →{cdt_country} (non-IBAN) + has_account + no IBAN")
    
    # Rule 2: System flag
    if get('missing_required_iban', False):
        reasons.append("missing_required_iban=True")
    
    return len(reasons) > 0, reasons


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Test 8004 rules (V13 - IBAN-based)')
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--sample-size', type=int, default=10)
    
    args = parser.parse_args()
    
    print("V13: IBAN-based filtering (no hardcoding)")
    print(f"  IBAN countries from IBAN_LENGTHS: {len(IBAN_COUNTRIES)}")
    print(f"  Excluded: IBAN countries + US = {len(EXCLUDE_BENEFICIARY_COUNTRIES)} countries")
    print(f"\n  Logic:")
    print(f"    - IBAN countries have bank directories → get 6021")
    print(f"    - US has ABA/routing validation → get 6021")
    print(f"    - Non-IBAN countries (TH, PH, AU, JP, MX, CA, etc.) → get 8004")
    
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
    
    print(f"Pipeline has {len(pipeline.records)} records")
    
    tp, fp, tn, fn = 0, 0, 0, 0
    fp_list = []
    fn_list = []
    
    filtered_out_iban = 0
    filtered_out_us = 0
    
    for record in pipeline.records:
        features = record.request_features
        codes = record.error_codes_only + (record.composite_codes or [])
        
        actual_8004 = any(TARGET_CODE in str(c) for c in codes)
        predicted, reasons = check_8004_rules(features)
        
        # Track filtering
        cdt_country = get_country(features, 'cdt')
        if cdt_country == 'US':
            filtered_out_us += 1
        elif cdt_country in IBAN_COUNTRIES:
            filtered_out_iban += 1
        
        if predicted and actual_8004:
            tp += 1
        elif predicted and not actual_8004:
            fp += 1
            fp_list.append((record.transaction_id, reasons, codes, features))
        elif not predicted and actual_8004:
            fn += 1
            fn_list.append((record.transaction_id, reasons, codes, features))
        else:
            tn += 1
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\n" + "=" * 70)
    print(f"TEST RESULTS: 8004 - IBAN Cannot Be Derived (V13 - IBAN-based)")
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
    
    print(f"\nFILTERING STATS:")
    print(f"  Filtered (IBAN countries): {filtered_out_iban}")
    print(f"  Filtered (US): {filtered_out_us}")
    print(f"  Total filtered: {filtered_out_iban + filtered_out_us}")
    
    # FP analysis
    print("\n" + "=" * 70)
    print(f"FALSE POSITIVES ({len(fp_list)} total):")
    print("=" * 70)
    
    fp_actual_codes = defaultdict(int)
    for _, _, codes, _ in fp_list:
        for c in codes:
            code_base = str(c).split('_')[0]
            fp_actual_codes[code_base] += 1
    
    print(f"\nWhat codes fired instead of 8004:")
    for code, count in sorted(fp_actual_codes.items(), key=lambda x: -x[1])[:10]:
        print(f"  {code}: {count}")
    
    fp_countries = defaultdict(int)
    for _, _, _, features in fp_list[:200]:
        cdt = get_country(features, 'cdt')
        in_iban = "IBAN" if cdt in IBAN_COUNTRIES else "non-IBAN"
        fp_countries[f"{cdt} ({in_iban})"] += 1
    
    print(f"\nFP by beneficiary country:")
    for country, count in sorted(fp_countries.items(), key=lambda x: -x[1])[:10]:
        print(f"  {country}: {count}")
    
    print(f"\nSample FPs:")
    for i, (txn_id, reasons, codes, features) in enumerate(fp_list[:args.sample_size]):
        dbt = normalize_originator_country(get_country(features, 'dbt') or get_country(features, 'orig'))
        cdt = get_country(features, 'cdt')
        in_iban = "IBAN" if cdt in IBAN_COUNTRIES else "non-IBAN"
        print(f"  {i+1}. {txn_id}")
        print(f"     Route: {dbt}→{cdt} ({in_iban})")
        print(f"     Actual codes: {codes[:3]}")
    
    # FN analysis
    print("\n" + "=" * 70)
    print(f"FALSE NEGATIVES ({len(fn_list)} total):")
    print("=" * 70)
    
    fn_countries = defaultdict(int)
    fn_reasons = defaultdict(int)
    for _, _, codes, features in fn_list[:100]:
        cdt = get_country(features, 'cdt')
        in_iban = "IBAN" if cdt in IBAN_COUNTRIES else "non-IBAN"
        fn_countries[f"{cdt} ({in_iban})"] += 1
        
        if cdt in IBAN_COUNTRIES:
            fn_reasons['Filtered: IBAN country'] += 1
        elif cdt == 'US':
            fn_reasons['Filtered: US'] += 1
        else:
            fn_reasons['Not filtered but missed'] += 1
    
    print(f"\nFN by reason:")
    for reason, count in sorted(fn_reasons.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}")
    
    print(f"\nFN by beneficiary country:")
    for country, count in sorted(fn_countries.items(), key=lambda x: -x[1])[:10]:
        print(f"  {country}: {count}")
    
    print(f"\nSample FNs:")
    for i, (txn_id, reasons, codes, features) in enumerate(fn_list[:args.sample_size]):
        dbt = normalize_originator_country(get_country(features, 'dbt') or get_country(features, 'orig'))
        cdt = get_country(features, 'cdt')
        in_iban = "IBAN" if cdt in IBAN_COUNTRIES else "non-IBAN"
        print(f"  {i+1}. {txn_id}")
        print(f"     Route: {dbt}→{cdt} ({in_iban})")
        print(f"     Codes: {[c for c in codes if TARGET_CODE in str(c)]}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    recall_status = "✓" if recall >= 0.95 else "⚠" if recall >= 0.90 else "✗"
    precision_status = "✓" if precision >= 0.50 else "⚠" if precision >= 0.30 else "✗"
    
    print(f"\n  Recall:     {recall*100:.1f}% {recall_status}")
    print(f"  Precision:  {precision*100:.1f}% {precision_status}")
    print(f"  F1 Score:   {f1*100:.1f}%")
    
    print(f"\nV13 Logic:")
    print(f"  - Uses IBAN_LENGTHS from ifml_parser.py (no hardcoding)")
    print(f"  - Excludes: IBAN countries ({len(IBAN_COUNTRIES)}) + US")
    print(f"  - Only predicts 8004 for non-IBAN beneficiaries (TH, PH, AU, JP, MX, CA, etc.)")


if __name__ == "__main__":
    main()
