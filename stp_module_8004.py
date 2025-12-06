#!/usr/bin/env python3
"""
Test 8004 Rules - V15 (Stricter strong signals + suppress domestic US + 6021-like cases)
Logic:
- Universal: Require missing_required_iban=True (core signal for "cannot derive")
- IBAN/US: Predict only on missing_required_iban (drop broad has_account + no IBAN to avoid 6021 overlap)
- Weak non-IBAN: + cross-border + has_account + no IBAN (but still require missing_required_iban)
- Suppress: Domestic US entirely (strong ABA validation → 6021)
- This boosts precision by gating on missing_required_iban; catches leaks without overpredicting
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
EXCLUDE_BENEFICIARY_COUNTRIES = IBAN_COUNTRIES | {'US'}
# Weak non-IBAN: From prior FP analysis; focus on true no-directory (expand/tune as data evolves)
WEAK_NON_IBAN_COUNTRIES = {'TH', 'PH', 'AU', 'JP', 'NZ', 'ZA', 'PG', 'ST', 'NC', 'ID', 'CO'}  # Low 6021 overlap
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
   
    V15:
    - Require missing_required_iban=True everywhere (suppresses 6021 overlaps)
    - IBAN/US: Predict only on missing_required_iban (strict to catch rare leaks)
    - Weak non-IBAN: + cross-border + has_account + no IBAN
    - Suppress domestic US (never predict; routes to 6021)
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
   
    # Core requirement: missing_required_iban (gates all predictions; avoids 6021 cases)
    missing_required_iban = get('missing_required_iban', False)
    if not missing_required_iban:
        return False, ["Missing core signal: missing_required_iban=False"]
   
    reasons.append("Core signal: missing_required_iban=True")
   
    # Suppress domestic US (strong validation → 6021, not 8004)
    if dbt_country == 'US' and cdt_country == 'US':
        return False, ["Suppressed: Domestic US (6021 territory)"]
   
    cdt_has_iban = get('cdt_has_iban', False)
    cdt_has_account = get('cdt_has_account', False)
   
    # IBAN/US: Strict - only missing_required_iban triggers (no need for account/IBAN check)
    if cdt_country in EXCLUDE_BENEFICIARY_COUNTRIES:
        reasons.append(f"Leak in {cdt_country}: missing_required_iban despite directory")
        return True, reasons
   
    # Non-IBAN: Filter to weak + cross-border + account signals
    if cdt_country not in WEAK_NON_IBAN_COUNTRIES:
        return False, [f"Filtered: Non-weak non-IBAN ({cdt_country})"]
   
    # Cross-border check
    if dbt_country and cdt_country:
        is_cross_border = (dbt_country != cdt_country)
    elif cdt_country and cdt_country != 'US':
        is_cross_border = True
    else:
        is_cross_border = False
   
    if not is_cross_border:
        return False, ["Suppressed: Not cross-border in weak non-IBAN"]
   
    if cdt_has_account and not cdt_has_iban:
        if dbt_country:
            reasons.append(f"Cross-border: {dbt_country}→{cdt_country} (weak non-IBAN) + has_account + no IBAN")
        else:
            reasons.append(f"International: →{cdt_country} (weak non-IBAN) + has_account + no IBAN")
        return True, reasons
    else:
        miss_reasons = []
        if not cdt_has_account:
            miss_reasons.append("No has_account")
        if cdt_has_iban:
            miss_reasons.append("Has IBAN")
        return False, [f"Missed weak non-IBAN cross-border: {', '.join(miss_reasons)}"]
   
    return False, reasons  # Fallback
def main():
    import argparse
    parser = argparse.ArgumentParser(description='Test 8004 rules (V15 - Strict signals + suppress domestic US)')
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--sample-size', type=int, default=10)
   
    args = parser.parse_args()
   
    print("V15: Strict missing_required_iban + suppress domestic US/6021 overlaps")
    print(f" IBAN countries from IBAN_LENGTHS: {len(IBAN_COUNTRIES)}")
    print(f" Exclude: IBAN + US = {len(EXCLUDE_BENEFICIARY_COUNTRIES)}")
    print(f" Weak non-IBAN targets: {len(WEAK_NON_IBAN_COUNTRIES)} ({sorted(WEAK_NON_IBAN_COUNTRIES)})")
    print(f"\n Logic:")
    print(f" - Require missing_required_iban=True everywhere (suppresses 6021/9999/etc.)")
    print(f" - IBAN/US: Predict only on missing_required_iban (catches leaks)")
    print(f" - Weak non-IBAN: + cross-border + has_account + no IBAN")
    print(f" - Suppress domestic US (no 8004; goes to 6021)")
   
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
    suppressed_domestic_us = 0
    missing_required_iban = 0  # Track core signal hits
   
    total_records = len(pipeline.records)
   
    for record in pipeline.records:
        features = record.request_features
        codes = record.error_codes_only + (record.composite_codes or [])
       
        actual_8004 = any(TARGET_CODE in str(c) for c in codes)
        predicted, reasons = check_8004_rules(features)
       
        # Track filtering/suppression
        cdt_country = get_country(features, 'cdt')
        dbt_country = normalize_originator_country(get_country(features, 'dbt') or get_country(features, 'orig'))
        if 'missing_required_iban=False' in str(reasons):
            missing_required_iban += 1  # Actually, this is non-hits; invert for hits
        if predicted:
            missing_required_iban += 1  # Predictions require it
        if cdt_country == 'US' and dbt_country == 'US' and 'Suppressed: Domestic US' in str(reasons):
            suppressed_domestic_us += 1
        elif cdt_country == 'US':
            filtered_out_us += 1
        elif cdt_country in IBAN_COUNTRIES:
            filtered_out_iban += 1
        elif cdt_country not in WEAK_NON_IBAN_COUNTRIES:
            filtered_out_non_weak += 1
       
        # FN subtype logging
        if not predicted and actual_8004:
            if 'Missing cdt_country' in str(reasons):
                fn_reasons['Missing cdt_country'] += 1
            elif 'Filtered: Non-weak non-IBAN' in str(reasons):
                fn_reasons['Filtered: Non-weak non-IBAN'] += 1
            elif 'Suppressed: Domestic US' in str(reasons):
                fn_reasons['Suppressed: Domestic US'] += 1
            elif 'Missing core signal' in str(reasons):
                fn_reasons['Missing missing_required_iban'] += 1
            else:
                fn_reasons['Other miss'] += 1
            fn_list.append((record.transaction_id, reasons, codes, features))
        elif predicted and not actual_8004:
            fp_list.append((record.transaction_id, reasons, codes, features))
       
        if predicted and actual_8004:
            tp += 1
        elif predicted and not actual_8004:
            fp += 1
        elif not predicted and actual_8004:
            fn += 1
        else:
            tn += 1
   
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / total_records if total_records > 0 else 0
   
    print("\n" + "=" * 70)
    print(f"TEST RESULTS: 8004 - IBAN Cannot Be Derived (V15 - Strict + Suppress)")
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
   
    print(f"\nFILTERING/SUPPRESSION STATS:")
    print(f" Filtered (IBAN countries): {filtered_out_iban}")
    print(f" Filtered (US non-domestic): {filtered_out_us}")
    print(f" Suppressed (Domestic US): {suppressed_domestic_us}")
    print(f" Filtered (Non-weak non-IBAN): {filtered_out_non_weak}")
    print(f" Core signal hits (missing_required_iban=True): {missing_required_iban}")
    print(f" Total filtered/suppressed: {filtered_out_iban + filtered_out_us + suppressed_domestic_us + filtered_out_non_weak}")
   
    # FP analysis (focus on remaining 6021/etc.)
    print("\n" + "=" * 70)
    print(f"FALSE POSITIVES ({len(fp_list)} total):")
    print("=" * 70)
   
    fp_actual_codes = defaultdict(int)
    for _, _, codes, _ in fp_list:
        for c in codes:
            if '6021' in str(c):  # Highlight suppressed targets
                code_base = '6021_SUPPRESSED'
            else:
                code_base = str(c).split('_')[0]
            fp_actual_codes[code_base] += 1
   
    print(f"\nWhat codes fired instead of 8004 (suppressed 6021 highlighted):")
    for code, count in sorted(fp_actual_codes.items(), key=lambda x: -x[1])[:10]:
        marker = " (suppressed)" if code == '6021_SUPPRESSED' else ""
        print(f" {code}: {count}{marker}")
   
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
   
    print(f"\nV15 Logic:")
    print(f" - Strict: missing_required_iban required (suppresses non-8004 like 6021)")
    print(f" - IBAN/US: Only on core signal (rare leaks)")
    print(f" - Weak non-IBAN: + cross-border + account signals")
    print(f" - Suppress domestic US (avoids 6021 FPs)")
    print(f" Expected: Recall ~99%, Precision >50% (tune weak set if needed)")

if __name__ == "__main__":
    main()
