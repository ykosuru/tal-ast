#!/usr/bin/env python3
"""
Test 8004 Rules - V17 (Focus: 'IBAN cannot be derived' per ACE docs)
Logic:
- Core: Predict if cdt_has_iban=False (direct match to "cannot be derived")
- Enhancers: + missing_required_iban (if present) or cross-border (for non-IBAN)
- IBAN/US: Allow on core (!has_iban) to catch derivation failures
- Non-IBAN: Core + cross-border to avoid domestic local-format success
- Suppress: Domestic US with has_iban=True (ABA succeeds)

IBAN Countries sourced from official SWIFT IBAN Registry Release 100 (Oct 2025):
https://www.swift.com/sites/default/files/files/iban-registry-100.pdf
"""
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Set

sys.path.insert(0, '/mnt/project')
from data_pipeline import IFMLDataPipeline

TARGET_CODE = '8004'

# Official SWIFT IBAN Registry Release 100 (Oct 2025)
# Source: https://www.swift.com/sites/default/files/files/iban-registry-100.pdf
IBAN_LENGTHS: Dict[str, int] = {
    # Primary IBAN countries (89 total per registry)
    'AD': 24,  # Andorra
    'AE': 23,  # United Arab Emirates
    'AL': 28,  # Albania
    'AT': 20,  # Austria
    'AZ': 28,  # Azerbaijan
    'BA': 20,  # Bosnia and Herzegovina
    'BE': 16,  # Belgium
    'BG': 22,  # Bulgaria
    'BH': 22,  # Bahrain
    'BI': 27,  # Burundi
    'BR': 29,  # Brazil
    'BY': 28,  # Belarus
    'CH': 21,  # Switzerland
    'CR': 22,  # Costa Rica
    'CY': 28,  # Cyprus
    'CZ': 24,  # Czech Republic (Czechia)
    'DE': 22,  # Germany
    'DJ': 27,  # Djibouti
    'DK': 18,  # Denmark
    'DO': 28,  # Dominican Republic
    'EE': 20,  # Estonia
    'EG': 29,  # Egypt
    'ES': 24,  # Spain
    'FI': 18,  # Finland
    'FK': 18,  # Falkland Islands (Malvinas)
    'FO': 18,  # Faroe Islands
    'FR': 27,  # France
    'GB': 22,  # United Kingdom
    'GE': 22,  # Georgia
    'GI': 23,  # Gibraltar
    'GL': 18,  # Greenland
    'GR': 27,  # Greece
    'GT': 28,  # Guatemala
    'HN': 28,  # Honduras
    'HR': 21,  # Croatia
    'HU': 28,  # Hungary
    'IE': 22,  # Ireland
    'IL': 23,  # Israel
    'IQ': 23,  # Iraq
    'IS': 26,  # Iceland
    'IT': 27,  # Italy
    'JO': 30,  # Jordan
    'KW': 30,  # Kuwait
    'KZ': 20,  # Kazakhstan
    'LB': 28,  # Lebanon
    'LC': 32,  # Saint Lucia
    'LI': 21,  # Liechtenstein
    'LT': 20,  # Lithuania
    'LU': 20,  # Luxembourg
    'LV': 21,  # Latvia
    'LY': 25,  # Libya
    'MC': 27,  # Monaco
    'MD': 24,  # Moldova
    'ME': 22,  # Montenegro
    'MK': 19,  # North Macedonia
    'MN': 20,  # Mongolia
    'MR': 27,  # Mauritania
    'MT': 31,  # Malta
    'MU': 30,  # Mauritius
    'NI': 28,  # Nicaragua
    'NL': 18,  # Netherlands
    'NO': 15,  # Norway
    'OM': 23,  # Oman
    'PK': 24,  # Pakistan
    'PL': 28,  # Poland
    'PS': 29,  # Palestine, State of
    'PT': 25,  # Portugal
    'QA': 29,  # Qatar
    'RO': 24,  # Romania
    'RS': 22,  # Serbia
    'RU': 33,  # Russian Federation
    'SA': 24,  # Saudi Arabia
    'SC': 31,  # Seychelles
    'SD': 18,  # Sudan
    'SE': 24,  # Sweden
    'SI': 19,  # Slovenia
    'SK': 24,  # Slovakia
    'SM': 27,  # San Marino
    'SO': 23,  # Somalia
    'ST': 25,  # Sao Tome and Principe
    'SV': 28,  # El Salvador
    'TL': 23,  # Timor-Leste
    'TN': 24,  # Tunisia
    'TR': 26,  # Turkey
    'UA': 29,  # Ukraine
    'VA': 22,  # Holy See (Vatican)
    'VG': 24,  # Virgin Islands (British)
    'XK': 20,  # Kosovo
    'YE': 30,  # Yemen
    
    # Territories sharing parent country IBAN format
    # France territories (FR format, 27 chars)
    'GF': 27,  # French Guiana
    'GP': 27,  # Guadeloupe
    'MQ': 27,  # Martinique
    'RE': 27,  # Réunion
    'PF': 27,  # French Polynesia
    'TF': 27,  # French Southern Territories
    'YT': 27,  # Mayotte
    'NC': 27,  # New Caledonia
    'BL': 27,  # Saint Barthélemy
    'MF': 27,  # Saint Martin (French part)
    'PM': 27,  # Saint Pierre and Miquelon
    'WF': 27,  # Wallis and Futuna
    
    # UK territories (GB format, 22 chars)
    'IM': 22,  # Isle of Man
    'JE': 22,  # Jersey
    'GG': 22,  # Guernsey
    
    # Finland territory
    'AX': 18,  # Åland Islands (uses FI format)
}

# Extract IBAN countries set
IBAN_COUNTRIES: Set[str] = set(IBAN_LENGTHS.keys())

# US has strong bank directory validation (ABA/routing)
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
    """Get country code from features."""
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
    """Normalize US state codes to 'US'."""
    if code and len(code) == 2 and code in US_STATE_CODES:
        return 'US'
    return code


def check_8004_rules(features: Dict) -> Tuple[bool, List[str]]:
    """
    Check if 8004 should be predicted.
    
    V17 (per ACE docs: 'IBAN cannot be derived'):
    - Core signal: cdt_has_iban=False
    - IBAN/US: Core alone (derivation failed despite directory)
    - Non-IBAN: Core + cross-border (local formats may succeed domestically)
    - Enhancer: + missing_required_iban (if True, strengthens)
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
    
    # Core signal: IBAN cannot be derived
    cdt_has_iban = get('cdt_has_iban', False)
    if cdt_has_iban:
        return False, ["Has IBAN: Derivation succeeded"]
    
    reasons.append("Core: cdt_has_iban=False ('IBAN cannot be derived')")
    
    missing_required_iban = get('missing_required_iban', False)
    if missing_required_iban:
        reasons.append("  - + missing_required_iban=True")
    
    cdt_has_account = get('cdt_has_account', False)
    if not cdt_has_account:
        return False, ["No account details: No derivation attempted"]
    
    # Domestic US suppress (ABA/routing derives alt, not IBAN)
    if dbt_country == 'US' and cdt_country == 'US':
        return False, ["Suppressed: Domestic US (uses ABA, not IBAN)"]
    
    # Cross-border check (for non-IBAN: required for prediction)
    if dbt_country and cdt_country:
        is_cross_border = (dbt_country != cdt_country)
    elif cdt_country and cdt_country != 'US':
        is_cross_border = True
    else:
        is_cross_border = False
    
    if cdt_country in EXCLUDE_BENEFICIARY_COUNTRIES:
        # IBAN/US: Core suffices (failed derivation)
        reasons.append(f"  - In {cdt_country}: Derivation failed despite directory")
        return True, reasons
    else:
        # Non-IBAN: Require cross-border (IBAN often needed for intl routing)
        if is_cross_border:
            if dbt_country:
                reasons.append(f"  - Cross-border: {dbt_country}→{cdt_country} (non-IBAN)")
            else:
                reasons.append(f"  - International: →{cdt_country} (non-IBAN)")
            return True, reasons
        else:
            return False, ["Domestic non-IBAN: Local format may suffice (no IBAN req)"]
    
    return False, reasons  # Fallback


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Test 8004 rules (V17 - "IBAN cannot be derived" per ACE docs)')
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--sample-size', type=int, default=10)
    
    args = parser.parse_args()
    
    print("V17: Aligned to ACE docs - '8004': 'IBAN cannot be derived'")
    print(f"  IBAN countries: {len(IBAN_COUNTRIES)} (SWIFT Registry Release 100, Oct 2025)")
    print(f"  Exclude base: IBAN + US = {len(EXCLUDE_BENEFICIARY_COUNTRIES)}")
    print(f"\n  Logic:")
    print(f"  - Core: cdt_has_iban=False + has_account=True")
    print(f"  - IBAN/US: Core alone (derivation fail)")
    print(f"  - Non-IBAN: Core + cross-border")
    print(f"  - Suppress: Domestic US or has_iban=True")
    
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
    fn_reasons = defaultdict(int)
    
    # Correlation tracking
    actual_8004_count = 0
    no_iban_in_actual = 0
    missing_req_in_actual = 0
    has_account_in_actual = 0
    
    filtered_out_iban = 0
    filtered_out_us = 0
    suppressed_domestic_us = 0
    
    total_records = len(pipeline.records)
    
    for record in pipeline.records:
        features = record.request_features
        codes = record.error_codes_only + (record.composite_codes or [])
        
        actual_8004 = any(TARGET_CODE in str(c) for c in codes)
        if actual_8004:
            actual_8004_count += 1
            no_iban = not features.get('cdt_has_iban', False)
            has_acc = features.get('cdt_has_account', False)
            miss_req = features.get('missing_required_iban', False)
            if no_iban:
                no_iban_in_actual += 1
            if has_acc:
                has_account_in_actual += 1
            if miss_req:
                missing_req_in_actual += 1
        
        predicted, reasons = check_8004_rules(features)
        
        # Track
        cdt_country = get_country(features, 'cdt')
        dbt_country = normalize_originator_country(get_country(features, 'dbt') or get_country(features, 'orig'))
        if cdt_country == 'US' and dbt_country == 'US' and 'Suppressed: Domestic US' in str(reasons):
            suppressed_domestic_us += 1
        elif cdt_country == 'US':
            filtered_out_us += 1
        elif cdt_country in IBAN_COUNTRIES and not predicted:
            filtered_out_iban += 1
        
        if not predicted and actual_8004:
            if 'Missing cdt_country' in str(reasons):
                fn_reasons['Missing cdt_country'] += 1
            elif 'Has IBAN' in str(reasons):
                fn_reasons['Has IBAN (unexpected)'] += 1
            elif 'No account details' in str(reasons):
                fn_reasons['No has_account'] += 1
            elif 'Suppressed: Domestic US' in str(reasons):
                fn_reasons['Suppressed: Domestic US'] += 1
            elif 'Domestic non-IBAN' in str(reasons):
                fn_reasons['Domestic non-IBAN'] += 1
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
    print(f"TEST RESULTS: 8004 - 'IBAN cannot be derived' (V17 - ACE docs aligned)")
    print("=" * 70)
    
    print(f"\nCONFUSION MATRIX:")
    print(f"                      Actual")
    print(f"                   8004    Not 8004")
    print(f"Predicted")
    print(f"     8004      | TP={tp:<6} | FP={fp:<6} |")
    print(f"  Not 8004     | FN={fn:<6} | TN={tn:<6} |")
    
    print(f"\nMETRICS:")
    print(f"  Precision: {precision*100:.2f}% (TP / (TP + FP))")
    print(f"  Recall:    {recall*100:.2f}% (TP / (TP + FN))")
    print(f"  F1 Score:  {f1*100:.2f}%")
    print(f"  Accuracy:  {accuracy*100:.2f}% ((TP + TN) / Total)")
    
    if actual_8004_count > 0:
        print(f"\nCORE SIGNAL CORRELATION IN ACTUAL 8004 ({actual_8004_count} cases):")
        print(f"  cdt_has_iban=False:        {no_iban_in_actual} ({no_iban_in_actual / actual_8004_count * 100:.1f}%)")
        print(f"  cdt_has_account=True:      {has_account_in_actual} ({has_account_in_actual / actual_8004_count * 100:.1f}%)")
        print(f"  missing_required_iban=True: {missing_req_in_actual} ({missing_req_in_actual / actual_8004_count * 100:.1f}%)")
    
    print(f"\nFILTERING/SUPPRESSION STATS:")
    print(f"  Filtered (IBAN w/o core):        {filtered_out_iban}")
    print(f"  Filtered (US non-domestic w/o):  {filtered_out_us}")
    print(f"  Suppressed (Domestic US):        {suppressed_domestic_us}")
    
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
        print(f"       Route: {dbt}→{cdt} ({in_iban})")
        print(f"       Predicted reasons: {reasons}")
        print(f"       Actual codes: {codes[:3]}")
    
    # FN analysis
    print("\n" + "=" * 70)
    print(f"FALSE NEGATIVES ({len(fn_list)} total):")
    print("=" * 70)
    
    print(f"\nFN by reason:")
    for reason, count in sorted(fn_reasons.items(), key=lambda x: -x[1]):
        pct = count / len(fn_list) * 100 if fn_list else 0
        print(f"  {reason}: {count} ({pct:.1f}%)")
    
    fn_countries = defaultdict(int)
    for _, _, codes, features in fn_list[:100]:
        cdt = get_country(features, 'cdt')
        in_iban = "IBAN" if cdt in IBAN_COUNTRIES else "non-IBAN"
        fn_countries[f"{cdt} ({in_iban})"] += 1
    
    print(f"\nFN by beneficiary country:")
    for country, count in sorted(fn_countries.items(), key=lambda x: -x[1])[:10]:
        print(f"  {country}: {count}")
    
    print(f"\nSample FNs:")
    for i, (txn_id, reasons, codes, features) in enumerate(fn_list[:args.sample_size]):
        dbt = normalize_originator_country(get_country(features, 'dbt') or get_country(features, 'orig'))
        cdt = get_country(features, 'cdt')
        in_iban = "IBAN" if cdt in IBAN_COUNTRIES else "non-IBAN"
        print(f"  {i+1}. {txn_id}")
        print(f"       Route: {dbt}→{cdt} ({in_iban})")
        print(f"       Missed reasons: {reasons}")
        print(f"       Codes: {[c for c in codes if TARGET_CODE in str(c)]}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    recall_status = "✓" if recall >= 0.95 else "⚠" if recall >= 0.90 else "✗"
    precision_status = "✓" if precision >= 0.50 else "⚠" if precision >= 0.30 else "✗"
    
    print(f"\n  Recall:    {recall*100:.1f}% {recall_status}")
    print(f"  Precision: {precision*100:.1f}% {precision_status}")
    print(f"  F1 Score:  {f1*100:.1f}%")
    print(f"  Accuracy:  {accuracy*100:.1f}%")
    
    if actual_8004_count > 0:
        print(f"\nV17 Updates (ACE-aligned):")
        print(f"  - Core on !cdt_has_iban (covers ~{no_iban_in_actual / actual_8004_count * 100:.0f}% of actual 8004)")
        print(f"  - Allows IBAN/US on core (catches derivation fails)")
        print(f"  - Non-IBAN requires cross-border (intl IBAN need)")
        print(f"  - If low recall: Check data for cdt_has_iban in actual 8004")


if __name__ == "__main__":
    main()
