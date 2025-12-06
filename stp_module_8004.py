#!/usr/bin/env python3
"""
Test 8004 Rules - V14 (IBAN-based + strong signals in IBAN/US + weak non-IBAN targeting)
Logic:
- IBAN countries/US: Predict 8004 only on strong signals (missing_required_iban or has_account + no IBAN)
- Non-IBAN: Target "weak validation" countries (e.g., TH/PH/AU/JP/NZ/ZA/PG/ST/NC); require cross-border + missing_required_iban
- This addresses: IBAN leaks (boost recall), CA/MX/IN FPs (boost precision)
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
# So base exclude: IBAN countries + US
EXCLUDE_BENEFICIARY_COUNTRIES = IBAN_COUNTRIES | {'US'}
# For non-IBAN: Subset to "weak" (high expected 8004, low alt validation like 6021)
# Tuned from FP analysis: Exclude CA/MX/IN/SV/GA (partial codes); keep TH/PH/AU/JP/NZ/ZA/PG/ST/NC
WEAK_NON_IBAN_COUNTRIES = {'TH', 'PH', 'AU', 'JP', 'NZ', 'ZA', 'PG', 'ST', 'NC', 'ID', 'CO'}  # Expand as needed
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
   
    V14:
    - IBAN/US: Allow on strong_signal (missing_required_iban or has_account + no IBAN)
    - Non-IBAN: Only weak countries; require cross-border + missing_required_iban + has_account + no IBAN
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
   
    # Early exit if no beneficiary country
    if not cdt_country:
        return False, ["Missing cdt_country"]
   
    # Strong signal check (universal)
    cdt_has_iban = get('cdt_has_iban', False)
    cdt_has_account = get('cdt_has_account', False)
    missing_required_iban = get('missing_required_iban', False)
    strong_signal = missing_required_iban or (cdt_has_account and not cdt_has_iban)
   
    # Rule 0: Strong signal alone triggers (even in IBAN/US)
    if strong_signal:
        reasons.append("Strong signal: missing_required_iban or (has_account + no IBAN)")
        if missing_required_iban:
            reasons.append("  - missing_required_iban=True")
        if cdt_has_account and not cdt_has_iban:
            reasons.append("  - has_account=True + no IBAN")
        return True, reasons
   
    # No strong signal: Filter out non-targets
    if cdt_country in EXCLUDE_BENEFICIARY_COUNTRIES:
        return False, [f"Filtered: IBAN/US without strong signal ({cdt_country})"]
    if cdt_country not in WEAK_NON_IBAN_COUNTRIES:
        return False, [f"Filtered: Non-weak non-IBAN ({cdt_country})"]
   
    # Now: Cross-border check for weak non-IBAN (requires missing_required_iban too for precision)
    if dbt_country and cdt_country:
        is_cross_border = (dbt_country != cdt_country)
    elif cdt_country and cdt_country != 'US':
        is_cross_border = True
    else:
        is_cross_border = False
   
    if is_cross_border and cdt_has_account and not cdt_has_iban and missing_required_iban:
        if dbt_country:
            reasons.append(f"Cross-border: {dbt_country}→{cdt_country} (weak non-IBAN) + has_account + no IBAN + missing_required_iban")
        else:
            reasons.append(f"International beneficiary: →{cdt_country} (weak non-IBAN) + has_account + no IBAN + missing_required_iban")
        return True, reasons
    else:
        miss_reasons = []
        if not is_cross_border:
            miss_reasons.append("Domestic/not cross-border")
        if not (cdt_has_account and not cdt_has_iban):
            miss_reasons.append("Missing has_account + no IBAN")
        if not missing_required_iban:
            miss_reasons.append("Missing missing_required_iban")
        return False, [f"Missed weak non-IBAN: {', '.join(miss_reasons)}"]
   
    return False, reasons  # Fallback
def main():
    import argparse
    parser = argparse.ArgumentParser(description='Test 8004 rules (V14 - Strong signals + weak non-IBAN)')
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--sample-size', type=int, default=10)
   
    args = parser.parse_args()
   
    print("V14: IBAN/US strong signals + weak non-IBAN targeting")
    print(f" IBAN countries from IBAN_LENGTHS: {len(IBAN_COUNTRIES)}")
    print(f" Base exclude: IBAN + US = {len(EXCLUDE_BENEFICIARY_COUNTRIES)}")
    print(f" Weak non-IBAN targets: {len(WEAK_NON_IBAN_COUNTRIES)} ({sorted(WEAK_NON_IBAN_COUNTRIES)})")
    print(f"\n Logic:")
    print(f" - Strong signal (any country): missing_required_iban or (has_account + no IBAN) → predict")
    print(f" - Weak non-IBAN: + cross-border + missing_required_iban → predict")
    print(f" - Others: No predict (filters IBAN/US leaks, CA/MX/IN FPs)")
   
    print("\nLoading IFML data...")
    pipeline = IFMLDataPipeline()
   
    data_path = Path(args.data_dir)
    if data_path.is_file():
        pipeline.load_single_file(str(data_path))
    else:
        json_files = sorted(data_path.glob("*.json"))
        for i, f in enumerate(json_files[:10]):
            count = pipeline.load_single_file(str(f))
            print(f" [{i+1:2d}] {f.name}: {count} payment(s)")
   
    print(f"Pipeline has {len(pipeline.records)} records")
   
    tp, fp, tn, fn = 0, 0, 0, 0
    fp_list = []
    fn_list = []
    fn_reasons = defaultdict(int)
   
    filtered_out_iban = 0
    filtered_out_us = 0
    filtered_out_non_weak = 0
   
    total_records = len(pipeline.records)
   
    for record in pipeline.records:
        features = record.request_features
        codes = record.error_codes_only + (record.composite_codes or [])
       
        actual_8004 = any(TARGET_CODE in str(c) for c in codes)
        predicted, reasons = check_8004_rules(features)
       
        # Track filtering (post-strong-signal)
        cdt_country = get_country(features, 'cdt')
        if cdt_country == 'US':
            filtered_out_us += 1
        elif cdt_country in IBAN_COUNTRIES:
            filtered_out_iban += 1
        elif cdt_country not in WEAK_NON_IBAN_COUNTRIES and cdt_country not in EXCLUDE_BENEFICIARY_COUNTRIES:
            filtered_out_non_weak += 1
       
        # FN subtype logging (for non-filtered misses)
        if not predicted and actual_8004:
            if cdt_country in EXCLUDE_BENEFICIARY_COUNTRIES:
                fn_reasons['Filtered: IBAN/US'] += 1
            elif cdt_country not in WEAK_NON_IBAN_COUNTRIES:
                fn_reasons['Filtered: Non-weak non-IBAN'] += 1
            else:
                # Parse reasons for subtype
                if 'Domestic' in ' '.join(reasons):
                    fn_reasons['Missed: Domestic non-IBAN'] += 1
                elif 'Missing has_account' in ' '.join(reasons):
                    fn_reasons['Missed: No account/IBAN signal'] += 1
                elif 'Missing missing_required_iban' in ' '.join(reasons):
                    fn_reasons['Missed: No required IBAN flag'] += 1
                else:
                    fn_reasons['Missed: Other'] += 1
       
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
    accuracy = (tp + tn) / total_records if total_records > 0 else 0
   
    print("\n" + "=" * 70)
    print(f"TEST RESULTS: 8004 - IBAN Cannot Be Derived (V14 - Strong + Weak)")
    print("=" * 70)
   
    print(f"\nCONFUSION MATRIX:")
    print(f" Actual")
    print(f" 8004 Not 8004")
    print(f"Predicted")
    print(f" 8004 | TP={tp:<6} | FP={fp:<6} |")
    print(f" Not 8004 | FN={fn:<6} | TN={tn:<6} |")
   
    print(f"\nMETRICS:")
    print(f" Precision: {precision*100:.2f}% (TP / (TP + FP))")
    print(f" Recall: {recall*100:.2f}% (TP / (TP + FN))")
    print(f" F1 Score: {f1*100:.2f}%")
    print(f" Accuracy: {accuracy*100:.2f}% ((TP + TN) / Total)")
   
    print(f"\nFILTERING STATS:")
    print(f" Filtered (IBAN countries): {filtered_out_iban}")
    print(f" Filtered (US): {filtered_out_us}")
    print(f" Filtered (Non-weak non-IBAN): {filtered_out_non_weak}")
    print(f" Total filtered: {filtered_out_iban + filtered_out_us + filtered_out_non_weak}")
   
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
        print(f" {code}: {count}")
   
    fp_countries = defaultdict(int)
    for _, _, _, features in fp_list[:200]:
        cdt = get_country(features, 'cdt')
        in_iban = "IBAN" if cdt in IBAN_COUNTRIES else "non-IBAN"
        weak_flag = "weak" if cdt in WEAK_NON_IBAN_COUNTRIES else "non-weak"
        fp_countries[f"{cdt} ({in_iban}-{weak_flag})"] += 1
   
    print(f"\nFP by beneficiary country:")
    for country, count in sorted(fp_countries.items(), key=lambda x: -x[1])[:10]:
        print(f" {country}: {count}")
   
    print(f"\nSample FPs:")
    for i, (txn_id, reasons, codes, features) in enumerate(fp_list[:args.sample_size]):
        dbt = normalize_originator_country(get_country(features, 'dbt') or get_country(features, 'orig'))
        cdt = get_country(features, 'cdt')
        in_iban = "IBAN" if cdt in IBAN_COUNTRIES else "non-IBAN"
        weak_flag = "weak" if cdt in WEAK_NON_IBAN_COUNTRIES else "non-weak"
        print(f" {i+1}. {txn_id}")
        print(f" Route: {dbt}→{cdt} ({in_iban}-{weak_flag})")
        print(f" Predicted reasons: {reasons}")
        print(f" Actual codes: {codes[:3]}")
   
    # FN analysis
    print("\n" + "=" * 70)
    print(f"FALSE NEGATIVES ({len(fn_list)} total):")
    print("=" * 70)
   
    print(f"\nFN by reason:")
    for reason, count in sorted(fn_reasons.items(), key=lambda x: -x[1]):
        print(f" {reason}: {count}")
   
    fn_countries = defaultdict(int)
    for _, _, codes, features in fn_list[:100]:
        cdt = get_country(features, 'cdt')
        in_iban = "IBAN" if cdt in IBAN_COUNTRIES else "non-IBAN"
        weak_flag = "weak" if cdt in WEAK_NON_IBAN_COUNTRIES else "non-weak"
        fn_countries[f"{cdt} ({in_iban}-{weak_flag})"] += 1
   
    print(f"\nFN by beneficiary country:")
    for country, count in sorted(fn_countries.items(), key=lambda x: -x[1])[:10]:
        print(f" {country}: {count}")
   
    print(f"\nSample FNs:")
    for i, (txn_id, reasons, codes, features) in enumerate(fn_list[:args.sample_size]):
        dbt = normalize_originator_country(get_country(features, 'dbt') or get_country(features, 'orig'))
        cdt = get_country(features, 'cdt')
        in_iban = "IBAN" if cdt in IBAN_COUNTRIES else "non-IBAN"
        weak_flag = "weak" if cdt in WEAK_NON_IBAN_COUNTRIES else "non-weak"
        print(f" {i+1}. {txn_id}")
        print(f" Route: {dbt}→{cdt} ({in_iban}-{weak_flag})")
        print(f" Missed reasons: {reasons}")
        print(f" Codes: {[c for c in codes if TARGET_CODE in str(c)]}")
   
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
   
    recall_status = "✓" if recall >= 0.95 else "⚠" if recall >= 0.90 else "✗"
    precision_status = "✓" if precision >= 0.50 else "⚠" if precision >= 0.30 else "✗"
   
    print(f"\n Recall: {recall*100:.1f}% {recall_status}")
    print(f" Precision: {precision*100:.1f}% {precision_status}")
    print(f" F1 Score: {f1*100:.1f}%")
    print(f" Accuracy: {accuracy*100:.1f}%")
   
    print(f"\nV14 Logic:")
    print(f" - Strong signals allow IBAN/US predictions (catches leaks)")
    print(f" - Weak non-IBAN only + stricter rules (reduces CA/MX FPs)")
    print(f" - Tune WEAK_NON_IBAN_COUNTRIES based on ongoing FP analysis")

if __name__ == "__main__":
    main()
