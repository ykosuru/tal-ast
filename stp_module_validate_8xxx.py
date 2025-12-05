#!/usr/bin/env python3
"""
Unified 8XXX Validator - Tests ALL 8XXX rule checkers against IFMLs

For each IFML:
1. Extract features from request
2. Run all 8XXX rule checkers
3. Compare predicted codes with actual 8XXX codes in response
4. Report precision, recall, accuracy

Usage:
    python validate_8xxx_all.py --data-dir /path/to/ifml/data --limit 1000
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict

# Add project path for imports
sys.path.insert(0, '/mnt/project')

from data_pipeline import IFMLDataPipeline


# =============================================================================
# IMPORT ALL 8XXX RULE CHECKERS
# =============================================================================

def check_8004_rules(features: Dict) -> Tuple[bool, List[str]]:
    """8004 = Missing IBAN when required"""
    reasons = []
    
    def get(feat, default=None):
        return features.get(feat, default)
    
    # Rule 1: needs_iban but no IBAN
    for prefix in ['bnf_', 'cdt_', 'dbt_', 'orig_']:
        needs_iban = get(f'{prefix}needs_iban', False)
        has_iban = get(f'{prefix}has_iban', False)
        if needs_iban and not has_iban:
            party = prefix.rstrip('_').upper()
            reasons.append(f"{party}: needs_iban but no IBAN")
    
    # Rule 2: has_account but no IBAN (IBAN derivation needed)
    for prefix in ['bnf_', 'cdt_', 'dbt_']:
        has_account = get(f'{prefix}has_account', False)
        has_iban = get(f'{prefix}has_iban', False)
        if has_account and not has_iban:
            party = prefix.rstrip('_').upper()
            reasons.append(f"{party}: has_account but no IBAN")
    
    # Rule 3: missing_required_iban
    if get('missing_required_iban', False):
        reasons.append("missing_required_iban=True")
    
    return len(reasons) > 0, reasons


def check_8022_rules(features: Dict) -> Tuple[bool, List[str]]:
    """8022 = Account/routing number validation"""
    reasons = []
    
    def get(feat, default=None):
        return features.get(feat, default)
    
    # Rule 1: bnf_has_iban
    if get('bnf_has_iban', False):
        reasons.append("BNF: has_iban=True")
    
    # Rule 2: bnf_needs_iban
    if get('bnf_needs_iban', False):
        reasons.append("BNF: needs_iban=True")
    
    # Rule 3: bnf_id_has_bic_and_nch
    if get('bnf_id_has_bic_and_nch', False):
        reasons.append("BNF: id_has_bic_and_nch=True")
    
    # Rule 4: bic_valid_country=False
    for prefix in ['bnf_', 'cdt_', 'send_', 'intm_']:
        has_bic = get(f'{prefix}has_bic', False)
        bic_valid_country = get(f'{prefix}bic_valid_country')
        if has_bic and bic_valid_country == False:
            party = prefix.rstrip('_').upper()
            reasons.append(f"{party}: bic_valid_country=False")
    
    # Rule 5: account_needs_length_fix
    for prefix in ['bnf_', 'cdt_', 'dbt_']:
        if get(f'{prefix}account_needs_length_fix', False):
            party = prefix.rstrip('_').upper()
            reasons.append(f"{party}: account_needs_length_fix=True")
    
    # Rule 6: bnf_has_account
    if get('bnf_has_account', False):
        reasons.append("BNF: has_account=True")
    
    return len(reasons) > 0, reasons


def check_8026_rules(features: Dict) -> Tuple[bool, List[str]]:
    """8026 = BIC/IBAN mismatch or derivation issue"""
    reasons = []
    
    def get(feat, default=None):
        return features.get(feat, default)
    
    # Rule 1: needs_iban (main trigger - IBAN derivation)
    for prefix in ['bnf_', 'cdt_', 'dbt_', 'orig_']:
        if get(f'{prefix}needs_iban', False):
            party = prefix.rstrip('_').upper()
            reasons.append(f"{party}: needs_iban=True")
    
    # Rule 2: bic_party_country_match=False
    for prefix in ['bnf_', 'cdt_', 'dbt_']:
        if get(f'{prefix}bic_party_country_match') == False:
            party = prefix.rstrip('_').upper()
            reasons.append(f"{party}: bic_party_country_match=False")
    
    # Rule 3: bic_iban_match=False
    for prefix in ['bnf_', 'cdt_', 'dbt_']:
        has_bic = get(f'{prefix}has_bic', False)
        has_iban = get(f'{prefix}has_iban', False)
        bic_iban_match = get(f'{prefix}bic_iban_country_match')
        if has_bic and has_iban and bic_iban_match == False:
            party = prefix.rstrip('_').upper()
            reasons.append(f"{party}: bic_iban_country_match=False")
    
    # Rule 4: has both BIC and IBAN (ACE validates the pair)
    for prefix in ['bnf_', 'cdt_', 'dbt_']:
        has_bic = get(f'{prefix}has_bic', False)
        has_iban = get(f'{prefix}has_iban', False)
        if has_bic and has_iban:
            party = prefix.rstrip('_').upper()
            reasons.append(f"{party}: has both BIC and IBAN")
    
    return len(reasons) > 0, reasons


def check_8027_rules(features: Dict) -> Tuple[bool, List[str]]:
    """8027 = IBAN/account validation"""
    reasons = []
    
    def get(feat, default=None):
        return features.get(feat, default)
    
    # Rule 1: iban_checksum_valid_count > 0
    iban_checksum_valid_count = get('iban_checksum_valid_count', 0) or 0
    if iban_checksum_valid_count > 0:
        reasons.append(f"iban_checksum_valid_count={iban_checksum_valid_count}")
    
    # Rule 2: any party has_iban
    for prefix in ['bnf_', 'cdt_', 'dbt_', 'orig_']:
        if get(f'{prefix}has_iban', False):
            party = prefix.rstrip('_').upper()
            reasons.append(f"{party}: has_iban=True")
    
    # Rule 3: bnf_nch_sources > 0
    bnf_nch_sources = get('bnf_nch_sources', 0) or 0
    if bnf_nch_sources > 0:
        reasons.append(f"bnf_nch_sources={bnf_nch_sources}")
    
    # Rule 4: BNF missing all account info
    bnf_has_iban = get('bnf_has_iban', False)
    bnf_account_length = get('bnf_account_length', 0) or 0
    if (bnf_has_iban == False or bnf_has_iban is None) and bnf_account_length == 0:
        reasons.append("BNF: no IBAN and account_length=0")
    
    return len(reasons) > 0, reasons


def check_8029_rules(features: Dict) -> Tuple[bool, List[str]]:
    """8029 = Account/ID validation issue"""
    reasons = []
    
    def get(feat, default=None):
        return features.get(feat, default)
    
    # Rule 1: primary_amount = False
    if get('primary_amount') == False:
        reasons.append("primary_amount=False")
    
    # Rule 2: bnf_id_has_slash
    if get('bnf_id_has_slash', False):
        reasons.append("BNF: id_has_slash=True")
    
    # Rule 3: bnf_id_has_nch_pattern
    if get('bnf_id_has_nch_pattern', False):
        reasons.append("BNF: id_has_nch_pattern=True")
    
    # Rule 4: has_multiple_nch_sources
    if get('has_multiple_nch_sources', False):
        reasons.append("has_multiple_nch_sources=True")
    
    # Rule 5: cdt_is_international
    if get('cdt_is_international', False):
        reasons.append("CDT: is_international=True")
    
    # Rule 6: dbt_is_international
    if get('dbt_is_international', False):
        reasons.append("DBT: is_international=True")
    
    return len(reasons) > 0, reasons


def check_8851_rules(features: Dict) -> Tuple[bool, List[str]]:
    """8851 = Invalid character"""
    reasons = []
    
    def get(feat, default=None):
        return features.get(feat, default)
    
    # Rule 1: account_has_dirty_chars
    for prefix in ['bnf_', 'cdt_', 'dbt_']:
        if get(f'{prefix}account_has_dirty_chars', False):
            party = prefix.rstrip('_').upper()
            reasons.append(f"{party}: account_has_dirty_chars=True")
    
    # Rule 2: nch_validation_applicable
    if get('nch_validation_applicable', False):
        reasons.append("nch_validation_applicable=True")
    
    # Rule 3: missing_required_iban
    if get('missing_required_iban', False):
        reasons.append("missing_required_iban=True")
    
    return len(reasons) > 0, reasons


def check_8852_rules(features: Dict) -> Tuple[bool, List[str]]:
    """8852 = Incorrect length of attribute"""
    reasons = []
    
    def get(feat, default=None):
        return features.get(feat, default)
    
    # IBAN length check
    IBAN_LENGTHS = {
        'AL': 28, 'AD': 24, 'AT': 20, 'AZ': 28, 'BH': 22, 'BY': 28, 'BE': 16,
        'BA': 20, 'BR': 29, 'BG': 22, 'CR': 22, 'HR': 21, 'CY': 28, 'CZ': 24,
        'DK': 18, 'DO': 28, 'TL': 23, 'EE': 20, 'FO': 18, 'FI': 18, 'FR': 27,
        'GE': 22, 'DE': 22, 'GI': 23, 'GR': 27, 'GL': 18, 'GT': 28, 'HU': 28,
        'IS': 26, 'IQ': 23, 'IE': 22, 'IL': 23, 'IT': 27, 'JO': 30, 'KZ': 20,
        'XK': 20, 'KW': 30, 'LV': 21, 'LB': 28, 'LI': 21, 'LT': 20, 'LU': 20,
        'MK': 19, 'MT': 31, 'MR': 27, 'MU': 30, 'MC': 27, 'MD': 24, 'ME': 22,
        'NL': 18, 'NO': 15, 'PK': 24, 'PS': 29, 'PL': 28, 'PT': 25, 'QA': 29,
        'RO': 24, 'SM': 27, 'SA': 24, 'RS': 22, 'SC': 31, 'SK': 24, 'SI': 19,
        'ES': 24, 'SE': 24, 'CH': 21, 'TN': 24, 'TR': 26, 'UA': 29, 'AE': 23,
        'GB': 22, 'VA': 22, 'VG': 24
    }
    
    for prefix in ['bnf_', 'cdt_', 'dbt_', 'orig_']:
        has_iban = get(f'{prefix}has_iban', False)
        if has_iban:
            iban_country = get(f'{prefix}iban_country', '')
            iban_length = get(f'{prefix}iban_length', 0) or 0
            expected = IBAN_LENGTHS.get(iban_country)
            if expected and iban_length != expected:
                party = prefix.rstrip('_').upper()
                reasons.append(f"{party}: IBAN length {iban_length} != expected {expected} for {iban_country}")
    
    # Field length violations
    for prefix in ['bnf_', 'cdt_', 'dbt_', 'orig_']:
        if get(f'{prefix}has_length_violation', False):
            party = prefix.rstrip('_').upper()
            reasons.append(f"{party}: has_length_violation=True")
    
    return len(reasons) > 0, reasons


def check_8894_rules(features: Dict) -> Tuple[bool, List[str]]:
    """8894 = IBAN validation issue"""
    reasons = []
    
    def get(feat, default=None):
        return features.get(feat, default)
    
    # Rule 1: account_has_dirty_chars
    for prefix in ['bnf_', 'cdt_', 'dbt_']:
        if get(f'{prefix}account_has_dirty_chars', False):
            party = prefix.rstrip('_').upper()
            reasons.append(f"{party}: account_has_dirty_chars=True")
    
    # Rule 2: IBAN checksum invalid
    for prefix in ['bnf_', 'cdt_', 'dbt_']:
        has_iban = get(f'{prefix}has_iban', False)
        checksum_valid = get(f'{prefix}iban_checksum_valid')
        if has_iban and checksum_valid == False:
            party = prefix.rstrip('_').upper()
            reasons.append(f"{party}: iban_checksum_valid=False")
    
    # Rule 3: needs_iban (IBAN processing triggered)
    for prefix in ['bnf_', 'cdt_', 'dbt_']:
        if get(f'{prefix}needs_iban', False):
            party = prefix.rstrip('_').upper()
            reasons.append(f"{party}: needs_iban=True")
    
    # Rule 4: dbt_is_international
    if get('dbt_is_international', False):
        reasons.append("DBT: is_international=True")
    
    # Rule 5: has both BIC and IBAN (ACE validates)
    for prefix in ['bnf_', 'cdt_', 'dbt_']:
        has_bic = get(f'{prefix}has_bic', False)
        has_iban = get(f'{prefix}has_iban', False)
        if has_bic and has_iban:
            party = prefix.rstrip('_').upper()
            reasons.append(f"{party}: has both BIC and IBAN")
    
    return len(reasons) > 0, reasons


def check_8895_rules(features: Dict) -> Tuple[bool, List[str]]:
    """8895 = NCH validation / IBAN validation issue"""
    reasons = []
    
    def get(feat, default=None):
        return features.get(feat, default)
    
    # Rule 1: has_invalid_iban
    if get('has_invalid_iban', False):
        reasons.append("has_invalid_iban=True")
    
    # Rule 2: nch_validation_applicable
    for prefix in ['bnf_', 'cdt_', 'orig_', 'dbt_']:
        if get(f'{prefix}nch_validation_applicable', False):
            party = prefix.rstrip('_').upper()
            reasons.append(f"{party}: nch_validation_applicable=True")
    
    # Rule 3: bic_iban_mismatch_count > 0
    bic_iban_mismatch_count = get('bic_iban_mismatch_count', 0) or 0
    if bic_iban_mismatch_count > 0:
        reasons.append(f"bic_iban_mismatch_count={bic_iban_mismatch_count}")
    
    # Rule 4: total_nch_sources > 0
    total_nch_sources = get('total_nch_sources', 0) or 0
    if total_nch_sources > 0:
        reasons.append(f"total_nch_sources={total_nch_sources}")
    
    # Rule 5: dbt_is_domestic
    if get('dbt_is_domestic', False):
        reasons.append("DBT: is_domestic=True")
    
    return len(reasons) > 0, reasons


def check_8896_rules(features: Dict) -> Tuple[bool, List[str]]:
    """8896 = IBAN/BIC validation issue"""
    reasons = []
    
    def get(feat, default=None):
        return features.get(feat, default)
    
    # Rule 1: bnf_iban_valid_format
    if get('bnf_iban_valid_format', False):
        reasons.append("BNF: iban_valid_format=True")
    
    # Rule 2: bnf_is_international
    if get('bnf_is_international', False):
        reasons.append("BNF: is_international=True")
    
    # Rule 3: missing_bnf_bic
    if get('missing_bnf_bic', False):
        reasons.append("missing_bnf_bic=True")
    
    # Rule 4: missing_required_iban
    if get('missing_required_iban', False):
        reasons.append("missing_required_iban=True")
    
    # Rule 5: is_any_domestic
    if get('is_any_domestic', False):
        reasons.append("is_any_domestic=True")
    
    # Rule 6: BNF has IBAN but no BIC
    if get('bnf_has_iban', False) and not get('bnf_has_bic', False):
        reasons.append("BNF: has_iban but no BIC")
    
    # Rule 7: bnf_needs_iban
    if get('bnf_needs_iban', False):
        reasons.append("BNF: needs_iban=True")
    
    return len(reasons) > 0, reasons


# =============================================================================
# RULE CHECKER REGISTRY
# =============================================================================

RULE_CHECKERS = {
    '8004': check_8004_rules,
    '8022': check_8022_rules,
    '8026': check_8026_rules,
    '8027': check_8027_rules,
    '8029': check_8029_rules,
    '8851': check_8851_rules,
    '8852': check_8852_rules,
    '8894': check_8894_rules,
    '8895': check_8895_rules,
    '8896': check_8896_rules,
}


def extract_8xxx_codes(codes: List) -> Set[str]:
    """Extract just the 8XXX base codes from a list of codes."""
    result = set()
    for code in codes:
        code_str = str(code)
        # Handle composite codes like "8004_BNPPTY"
        if '_' in code_str:
            base = code_str.split('_')[0]
        else:
            base = code_str
        
        # Only include 8XXX codes
        if base.startswith('8') and len(base) == 4 and base.isdigit():
            result.add(base)
    
    return result


def predict_8xxx_codes(features: Dict) -> Dict[str, Tuple[bool, List[str]]]:
    """Run all rule checkers and return predictions."""
    predictions = {}
    for code, checker in RULE_CHECKERS.items():
        should_fire, reasons = checker(features)
        predictions[code] = (should_fire, reasons)
    return predictions


def main():
    parser = argparse.ArgumentParser(description='Unified 8XXX Validator')
    parser.add_argument('--data-dir', required=True, help='Directory with IFML JSON files')
    parser.add_argument('--limit', type=int, default=10000, help='Max records to process')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show per-IFML details')
    parser.add_argument('--show-mismatches', type=int, default=20, help='Number of mismatches to show')
    
    args = parser.parse_args()
    
    print("="*80)
    print("UNIFIED 8XXX VALIDATOR")
    print("="*80)
    print(f"\nRule checkers available: {list(RULE_CHECKERS.keys())}")
    
    print("\nLoading IFML data...")
    pipeline = IFMLDataPipeline()
    
    data_path = Path(args.data_dir)
    if data_path.is_file():
        loaded = pipeline.load_single_file(str(data_path))
    else:
        loaded = pipeline.load_directory(str(data_path), "*.json")
    
    print(f"Loaded {loaded} records, processing up to {args.limit}")
    
    # Stats tracking
    total_processed = 0
    
    # Per-code metrics
    per_code_stats = defaultdict(lambda: {
        'tp': 0,  # True Positive: predicted AND actual
        'fp': 0,  # False Positive: predicted but NOT actual
        'fn': 0,  # False Negative: NOT predicted but actual
        'tn': 0,  # True Negative: NOT predicted and NOT actual
    })
    
    # Overall metrics
    total_actual_8xxx = 0
    total_predicted_8xxx = 0
    total_correct_predictions = 0
    
    # Track mismatches for debugging
    false_positives = []  # Predicted but not actual
    false_negatives = []  # Actual but not predicted
    
    for record in pipeline.records:
        if total_processed >= args.limit:
            break
        
        total_processed += 1
        
        txn_id = record.transaction_id
        features = record.request_features
        
        # Get actual 8XXX codes from response
        actual_codes = record.error_codes_only or []
        composite_codes = record.composite_codes or []
        actual_8xxx = extract_8xxx_codes(actual_codes + composite_codes)
        
        # Get predicted codes
        predictions = predict_8xxx_codes(features)
        predicted_8xxx = {code for code, (should_fire, _) in predictions.items() if should_fire}
        
        # Update overall stats
        total_actual_8xxx += len(actual_8xxx)
        total_predicted_8xxx += len(predicted_8xxx)
        
        # Compare per code
        all_codes = set(RULE_CHECKERS.keys())
        
        for code in all_codes:
            predicted = code in predicted_8xxx
            actual = code in actual_8xxx
            
            if predicted and actual:
                per_code_stats[code]['tp'] += 1
                total_correct_predictions += 1
            elif predicted and not actual:
                per_code_stats[code]['fp'] += 1
                if len(false_positives) < 100:
                    false_positives.append((txn_id, code, predictions[code][1], actual_8xxx))
            elif not predicted and actual:
                per_code_stats[code]['fn'] += 1
                if len(false_negatives) < 100:
                    false_negatives.append((txn_id, code, actual_8xxx))
            else:
                per_code_stats[code]['tn'] += 1
        
        if args.verbose:
            if actual_8xxx or predicted_8xxx:
                match = "✅" if actual_8xxx == predicted_8xxx else "❌"
                print(f"{match} {txn_id}: actual={actual_8xxx}, predicted={predicted_8xxx}")
    
    # ==========================================================================
    # REPORT
    # ==========================================================================
    
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    print(f"\nTotal IFMLs processed: {total_processed}")
    print(f"Total actual 8XXX codes found: {total_actual_8xxx}")
    print(f"Total predicted 8XXX codes: {total_predicted_8xxx}")
    
    # Per-code metrics
    print("\n" + "-"*80)
    print("PER-CODE METRICS:")
    print("-"*80)
    print(f"{'Code':<8} {'TP':>6} {'FP':>6} {'FN':>6} {'TN':>6} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-"*80)
    
    total_tp = total_fp = total_fn = total_tn = 0
    
    for code in sorted(RULE_CHECKERS.keys()):
        stats = per_code_stats[code]
        tp = stats['tp']
        fp = stats['fp']
        fn = stats['fn']
        tn = stats['tn']
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Highlight codes with actual occurrences
        marker = "⚠️" if (tp + fn) == 0 else ""
        
        print(f"{code:<8} {tp:>6} {fp:>6} {fn:>6} {tn:>6} {precision:>10.1%} {recall:>10.1%} {f1:>10.3f} {marker}")
    
    print("-"*80)
    
    # Overall metrics
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    
    print(f"{'TOTAL':<8} {total_tp:>6} {total_fp:>6} {total_fn:>6} {total_tn:>6} {overall_precision:>10.1%} {overall_recall:>10.1%} {overall_f1:>10.3f}")
    
    print("\n" + "="*80)
    print("KEY METRICS:")
    print("="*80)
    print(f"  Overall Precision: {overall_precision:.1%} (of predicted, how many were correct)")
    print(f"  Overall Recall:    {overall_recall:.1%} (of actual, how many did we predict)")
    print(f"  Overall F1 Score:  {overall_f1:.3f}")
    
    # Show false negatives (missed predictions)
    if false_negatives:
        print("\n" + "="*80)
        print(f"FALSE NEGATIVES (actual but not predicted) - showing first {args.show_mismatches}:")
        print("="*80)
        
        fn_by_code = defaultdict(list)
        for txn_id, code, actual_8xxx in false_negatives:
            fn_by_code[code].append((txn_id, actual_8xxx))
        
        for code in sorted(fn_by_code.keys()):
            print(f"\n{code}: ({len(fn_by_code[code])} missed)")
            for txn_id, actual in fn_by_code[code][:5]:
                print(f"  - {txn_id}: actual={actual}")
    
    # Show false positives (over-predictions)
    if false_positives:
        print("\n" + "="*80)
        print(f"FALSE POSITIVES (predicted but not actual) - showing first {args.show_mismatches}:")
        print("="*80)
        
        fp_by_code = defaultdict(list)
        for txn_id, code, reasons, actual_8xxx in false_positives:
            fp_by_code[code].append((txn_id, reasons, actual_8xxx))
        
        for code in sorted(fp_by_code.keys()):
            print(f"\n{code}: ({len(fp_by_code[code])} over-predicted)")
            for txn_id, reasons, actual in fp_by_code[code][:5]:
                print(f"  - {txn_id}: predicted because {reasons[0] if reasons else 'unknown'}, actual={actual}")
    
    print("\n" + "="*80)
    print("LEGEND:")
    print("="*80)
    print("  TP = True Positive (correctly predicted)")
    print("  FP = False Positive (predicted but didn't occur - over-prediction)")
    print("  FN = False Negative (occurred but not predicted - missed)")
    print("  TN = True Negative (correctly not predicted)")
    print("  ⚠️ = No actual occurrences of this code in dataset")
    print("\n  Precision = TP/(TP+FP) - How accurate are our predictions?")
    print("  Recall = TP/(TP+FN) - How many actual errors do we catch?")
    print("  F1 = Harmonic mean of Precision and Recall")
    
    print("\n" + "="*80)
    print("END OF REPORT")
    print("="*80)


if __name__ == '__main__':
    main()
