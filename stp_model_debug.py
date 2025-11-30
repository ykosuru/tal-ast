#!/usr/bin/env python3
"""
ACE Pelican Payment Debug Tool
==============================

Analyzes a payment message and shows:
1. All extracted features (non-empty values)
2. Features relevant to each potential error code
3. Precondition check results
4. Prediction likelihood for 8XXX/9XXX codes

Usage:
    python debug_payment.py payment.json
    python debug_payment.py payment.json --model-dir ./models_8x
    python debug_payment.py payment.json --code 9018
    python debug_payment.py payment.json --verbose
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

# Import parser
from ifml_parser import IFMLParser


# ============================================================================
# CODE DEFINITIONS AND TRIGGERS
# ============================================================================

CODE_DESCRIPTIONS = {
    # 8XXX Validation Errors
    '8001': 'Invalid BIC',
    '8004': 'IBAN cannot be derived',
    '8005': 'Invalid BIC4',
    '8006': 'Invalid country code',
    '8022': 'IBAN inconsistent with BIC',
    '8023': 'IBAN inconsistency in message',
    '8024': 'BBAN inconsistency in message',
    '8025': 'Domestic Account inconsistency',
    '8026': 'NCH inconsistency in message',
    '8027': 'ISO Country Code inconsistency',
    '8028': 'BIC4 inconsistency',
    '8029': 'Account Number inconsistency',
    '8030': 'IBAN derivation not supported',
    '8033': 'CLABE inconsistency',
    '8851': 'Incorrect field size',
    '8852': 'Incorrect length of attribute',
    '8892': 'Invalid Account number',
    '8894': 'Invalid IBAN',
    '8895': 'Invalid NCH code',
    '8896': 'Invalid Domestic Account Number',
    '8897': 'Invalid BBAN',
    '8898': 'IBAN Check Digit failed',
    
    # 9XXX Repair Codes
    '9002': 'Account number cleaned',
    '9004': 'IBAN derived/repaired',
    '9005': 'BIC derived from NCH',
    '9006': 'IBAN cleaned/formatted',
    '9007': 'Account replaced by IBAN',
    '9008': 'BIC derived from IBAN',
    '9015': 'Name cleaned',
    '9017': 'Multiple party information present',
    '9018': 'Duplicate party information removed',
    '9019': 'Party identifier cleaned',
    '9021': 'NCH/FEDABA formatted',
    '9024': 'Push Up performed',
    '9025': 'Account length fixed',
    '9477': 'BIC enriched 8->11 chars',
    '9479': 'Party enriched from BIC directory',
    '9480': 'Credit party enriched',
    '9490': 'Fee Code Updated',
    '9999': 'Field Repair (generic)',
}

# Features that trigger each code
CODE_TRIGGERS = {
    '8001': {
        'require_true': ['has_bic'],
        'require_false': ['bic_valid_format', 'bic_valid_country'],
        'relevant_features': ['has_bic', 'bic_valid_format', 'bic_valid_country', 'bic_length', 'bic_country'],
    },
    '8004': {
        'require_true': ['needs_iban'],
        'require_false': ['has_iban'],
        'relevant_features': ['needs_iban', 'has_iban', 'has_account', 'country', 'is_international'],
    },
    '8022': {
        'require_true': ['has_iban', 'has_bic'],
        'require_false': ['bic_iban_match'],
        'relevant_features': ['has_iban', 'has_bic', 'bic_iban_match', 'bic_country', 'iban_country'],
    },
    '8026': {
        'require_true': ['has_nch'],
        'require_false': [],
        'relevant_features': ['has_nch', 'nch_valid', 'nch_sources', 'id_has_bic_and_nch', 'id_is_compound', 'fedaba_checksum_valid'],
    },
    '8894': {
        'require_true': ['has_iban'],
        'require_false': ['iban_valid_format', 'iban_checksum_valid'],
        'relevant_features': ['has_iban', 'iban_valid_format', 'iban_checksum_valid', 'iban_country'],
    },
    '8895': {
        'require_true': ['has_nch'],
        'require_false': ['nch_valid', 'fedaba_checksum_valid'],
        'relevant_features': ['has_nch', 'nch_valid', 'fedaba_checksum_valid', 'nch_validation_applicable', 'is_fedaba'],
    },
    '9002': {
        'require_true': ['has_account', 'account_has_dirty_chars'],
        'require_false': [],
        'relevant_features': ['has_account', 'account_has_dirty_chars', 'account_has_spaces', 'account_has_special_chars'],
    },
    '9004': {
        'require_true': ['needs_iban'],
        'require_false': ['has_iban'],
        'relevant_features': ['needs_iban', 'has_iban', 'is_iban_derivable', 'has_bban_in_iban_country'],
    },
    '9005': {
        'require_true': ['has_nch'],
        'require_false': ['has_bic'],
        'relevant_features': ['has_nch', 'has_bic', 'has_nch_no_bic', 'is_bic_derivable'],
    },
    '9008': {
        'require_true': ['has_iban'],
        'require_false': ['has_bic'],
        'relevant_features': ['has_iban', 'has_bic', 'has_iban_no_bic', 'is_bic_derivable'],
    },
    '9018': {
        'require_true': ['intm_has_multiple', 'intm_has_redundant_info'],
        'require_false': [],
        'relevant_features': ['intm_count', 'intm_has_multiple', 'intm_entries_share_adr_bank_id', 
                              'intm_entries_share_country', 'intm_entries_share_bic_prefix', 
                              'intm_has_redundant_info', 'intm_has_multiple_types', 'has_intermediary'],
        'global': True,
    },
    '9019': {
        'require_true': ['has_id'],
        'require_false': [],
        'relevant_features': ['has_id', 'id_has_slash', 'id_is_compound', 'account_has_dirty_chars'],
    },
    '9024': {
        'require_true': ['has_intermediary'],
        'require_false': [],
        'relevant_features': ['has_intermediary', 'intm_count', 'intm_has_redundant_info', 
                              'intm_has_multiple', 'intm_has_multiple_types'],
        'global': True,
    },
    '9490': {
        'require_true': [],
        'require_false': [],
        'relevant_features': [],
        'directory_dependent': True,
        'note': 'Requires fee directory lookup - not predictable from message',
    },
    '9999': {
        'require_true': [],
        'require_false': [],
        'relevant_features': [],
        'unpredictable': True,
        'note': 'Generic field repair - manual intervention, not predictable',
    },
}

PARTY_PREFIXES = ['orig', 'send', 'dbt', 'cdt', 'intm', 'bnf', 'ordi', 'acwi']
PARTY_NAMES = {
    'orig': 'OriginatingParty',
    'send': 'SendingBank',
    'dbt': 'DebitParty',
    'cdt': 'CreditParty',
    'intm': 'IntermediaryBank',
    'bnf': 'BeneficiaryBank',
    'ordi': 'OrderingInstitution',
    'acwi': 'AccountWithInstitution',
}


# ============================================================================
# FEATURE ANALYSIS
# ============================================================================

def extract_features(payment_json: dict) -> Tuple[dict, Any]:
    """Extract features from payment JSON."""
    parser = IFMLParser()
    features_obj = parser.parse(payment_json)
    feature_dict = parser.to_dict(features_obj)
    return feature_dict, features_obj


def get_non_empty_features(features: dict) -> dict:
    """Get only features with non-empty/non-default values."""
    non_empty = {}
    for k, v in features.items():
        # Keep True values, non-zero numbers, non-empty strings/lists
        if v is True:
            non_empty[k] = v
        elif v is not None and v is not False and v != 0 and v != '' and v != []:
            non_empty[k] = v
    return non_empty


def group_features_by_party(features: dict) -> Dict[str, dict]:
    """Group features by party prefix."""
    grouped = defaultdict(dict)
    global_features = {}
    
    for k, v in features.items():
        matched = False
        for prefix in PARTY_PREFIXES:
            if k.startswith(f'{prefix}_'):
                suffix = k[len(prefix)+1:]
                grouped[prefix][suffix] = v
                matched = True
                break
        if not matched:
            global_features[k] = v
    
    grouped['_global'] = global_features
    return dict(grouped)


def check_preconditions(features: dict, code: str, party_prefix: str = None) -> Tuple[bool, List[str]]:
    """
    Check if preconditions are met for a code.
    Returns (can_fire, reasons).
    """
    if code not in CODE_TRIGGERS:
        return True, ['No preconditions defined']
    
    triggers = CODE_TRIGGERS[code]
    reasons = []
    can_fire = True
    
    # Check require_true conditions
    for feat in triggers.get('require_true', []):
        if triggers.get('global'):
            # Global feature - check directly
            val = features.get(feat)
            feat_name = feat
        else:
            # Party-specific feature
            if party_prefix:
                feat_name = f'{party_prefix}_{feat}'
                val = features.get(feat_name)
            else:
                # Check all parties
                val = None
                feat_name = feat
                for prefix in PARTY_PREFIXES:
                    full_feat = f'{prefix}_{feat}'
                    if features.get(full_feat):
                        val = True
                        feat_name = full_feat
                        break
        
        if val:
            reasons.append(f'✅ {feat_name} = {val}')
        else:
            reasons.append(f'❌ {feat_name} = {val} (required: True)')
            can_fire = False
    
    # Check require_false conditions
    for feat in triggers.get('require_false', []):
        if triggers.get('global'):
            val = features.get(feat)
            feat_name = feat
        else:
            if party_prefix:
                feat_name = f'{party_prefix}_{feat}'
                val = features.get(feat_name)
            else:
                val = None
                feat_name = feat
                for prefix in PARTY_PREFIXES:
                    full_feat = f'{prefix}_{feat}'
                    v = features.get(full_feat)
                    if v is False:
                        val = False
                        feat_name = full_feat
                        break
        
        if val is False or val is None:
            reasons.append(f'✅ {feat_name} = {val} (validation failed as expected)')
        else:
            reasons.append(f'❌ {feat_name} = {val} (required: False/validation must fail)')
            can_fire = False
    
    # Add notes for special codes
    if triggers.get('directory_dependent'):
        reasons.append('⚠️  Directory-dependent: Cannot fully predict from message alone')
    if triggers.get('unpredictable'):
        reasons.append('⚠️  Unpredictable: Generic repair code')
    if triggers.get('note'):
        reasons.append(f'ℹ️  {triggers["note"]}')
    
    return can_fire, reasons


def get_relevant_features_for_code(features: dict, code: str) -> Dict[str, Any]:
    """Get features relevant to a specific code."""
    if code not in CODE_TRIGGERS:
        return {}
    
    triggers = CODE_TRIGGERS[code]
    relevant = {}
    
    for feat_base in triggers.get('relevant_features', []):
        if triggers.get('global'):
            # Global feature
            if feat_base in features:
                relevant[feat_base] = features[feat_base]
        else:
            # Party-specific - get for all parties
            for prefix in PARTY_PREFIXES:
                feat_name = f'{prefix}_{feat_base}'
                if feat_name in features:
                    val = features[feat_name]
                    if val is not None and val != False and val != 0:
                        relevant[feat_name] = val
    
    return relevant


# ============================================================================
# PREDICTION (without model)
# ============================================================================

def predict_codes_rule_based(features: dict) -> List[Tuple[str, str, float, List[str]]]:
    """
    Rule-based prediction of likely codes.
    Returns list of (code, party, confidence, reasons).
    """
    predictions = []
    
    for code, triggers in CODE_TRIGGERS.items():
        if triggers.get('unpredictable') or triggers.get('directory_dependent'):
            continue
        
        if triggers.get('global'):
            # Check global preconditions
            can_fire, reasons = check_preconditions(features, code)
            if can_fire:
                confidence = 0.7 if len(triggers.get('require_true', [])) > 1 else 0.5
                predictions.append((code, 'GLOBAL', confidence, reasons))
        else:
            # Check per-party
            for prefix in PARTY_PREFIXES:
                if not features.get(f'{prefix}_present'):
                    continue
                can_fire, reasons = check_preconditions(features, code, prefix)
                if can_fire:
                    confidence = 0.6
                    party_name = PARTY_NAMES.get(prefix, prefix.upper())
                    predictions.append((code, party_name, confidence, reasons))
    
    # Sort by confidence
    predictions.sort(key=lambda x: -x[2])
    return predictions


# ============================================================================
# MODEL-BASED PREDICTION
# ============================================================================

def predict_with_model(features: dict, model_dir: str) -> List[Tuple[str, float]]:
    """
    Use trained model for prediction.
    Returns list of (code, probability).
    """
    try:
        from predictor import ACEPredictor
        
        predictor = ACEPredictor(model_dir)
        
        # We need the original payment JSON for the predictor
        # Since we only have features, we'll use a workaround
        # by directly using the model's predict method
        
        import pandas as pd
        import numpy as np
        
        X_raw = pd.DataFrame([features])
        X_transformed = predictor.feature_engineer.transform(X_raw)
        X = X_transformed.values
        
        probas = predictor.model.predict_proba(X)[0]
        
        results = []
        for i, prob in enumerate(probas):
            if i < len(predictor.model.class_names):
                code = predictor.model.class_names[i]
                if not code.startswith('__') and prob > 0.3:
                    results.append((code, float(prob)))
        
        results.sort(key=lambda x: -x[1])
        return results
        
    except Exception as e:
        return [('ERROR', f'Could not load model: {e}')]


# ============================================================================
# OUTPUT FORMATTING
# ============================================================================

def print_section(title: str, char: str = '='):
    """Print a section header."""
    print(f"\n{char * 60}")
    print(title)
    print(char * 60)


def print_features_table(features: dict, title: str = "Features"):
    """Print features in a formatted table."""
    if not features:
        print("  (none)")
        return
    
    max_key_len = max(len(k) for k in features.keys())
    for k, v in sorted(features.items()):
        print(f"  {k:<{max_key_len}} : {v}")


def analyze_payment(payment_json: dict, model_dir: str = None, 
                    target_code: str = None, verbose: bool = False):
    """Main analysis function."""
    
    print_section("ACE PELICAN PAYMENT ANALYZER")
    
    # Extract features
    features, features_obj = extract_features(payment_json)
    non_empty = get_non_empty_features(features)
    
    # Basic info
    print(f"\nTransaction ID: {features.get('transaction_id', 'N/A')}")
    print(f"Total features extracted: {len(features)}")
    print(f"Non-empty features: {len(non_empty)}")
    
    # Show parties present
    print("\nParties present:")
    for prefix in PARTY_PREFIXES:
        if features.get(f'{prefix}_present'):
            party_name = PARTY_NAMES.get(prefix, prefix)
            print(f"  ✓ {party_name}")
    
    # Global features
    print_section("GLOBAL FEATURES", '-')
    global_features = {k: v for k, v in non_empty.items() 
                       if not any(k.startswith(f'{p}_') for p in PARTY_PREFIXES)}
    print_features_table(global_features)
    
    # Intermediary redundancy features (key for 9018/9024)
    print_section("INTERMEDIARY REDUNDANCY ANALYSIS (9018/9024)", '-')
    intm_features = {k: v for k, v in features.items() if k.startswith('intm_') and 'present' not in k}
    print_features_table({k: v for k, v in intm_features.items() if v})
    
    if features.get('intm_has_redundant_info'):
        print("\n  ⚠️  REDUNDANCY DETECTED - 9018/9024 likely")
    elif features.get('intm_has_multiple'):
        print("\n  ℹ️  Multiple intermediaries but no redundancy detected")
    else:
        print("\n  ℹ️  Single or no intermediary - 9018 unlikely")
    
    # Compound ID analysis (key for 8026)
    print_section("COMPOUND ID ANALYSIS (8026)", '-')
    compound_found = False
    for prefix in PARTY_PREFIXES:
        if features.get(f'{prefix}_id_has_bic_and_nch'):
            party_name = PARTY_NAMES.get(prefix, prefix)
            print(f"  ⚠️  {party_name}: ID contains both BIC and NCH patterns")
            print(f"      id_has_slash: {features.get(f'{prefix}_id_has_slash')}")
            print(f"      id_compound_parts: {features.get(f'{prefix}_id_compound_parts')}")
            compound_found = True
    
    if not compound_found:
        print("  ℹ️  No compound BIC+NCH IDs detected")
    
    # Per-party features (verbose mode)
    if verbose:
        print_section("PER-PARTY FEATURES", '-')
        grouped = group_features_by_party(non_empty)
        for prefix in PARTY_PREFIXES:
            if prefix in grouped and grouped[prefix]:
                party_name = PARTY_NAMES.get(prefix, prefix)
                print(f"\n  [{party_name}]")
                for k, v in sorted(grouped[prefix].items()):
                    print(f"    {k}: {v}")
    
    # Analyze specific code if requested
    if target_code:
        print_section(f"ANALYSIS FOR CODE {target_code}", '-')
        desc = CODE_DESCRIPTIONS.get(target_code, 'Unknown code')
        print(f"Description: {desc}")
        
        # Get relevant features
        relevant = get_relevant_features_for_code(features, target_code)
        print("\nRelevant features:")
        print_features_table(relevant)
        
        # Check preconditions
        print("\nPrecondition check:")
        if CODE_TRIGGERS.get(target_code, {}).get('global'):
            can_fire, reasons = check_preconditions(features, target_code)
            for r in reasons:
                print(f"  {r}")
            print(f"\n  → Can fire: {'YES' if can_fire else 'NO'}")
        else:
            for prefix in PARTY_PREFIXES:
                if features.get(f'{prefix}_present'):
                    can_fire, reasons = check_preconditions(features, target_code, prefix)
                    if can_fire or verbose:
                        party_name = PARTY_NAMES.get(prefix, prefix)
                        print(f"\n  [{party_name}]")
                        for r in reasons:
                            print(f"    {r}")
                        print(f"    → Can fire: {'YES' if can_fire else 'NO'}")
    
    # Rule-based predictions
    print_section("RULE-BASED PREDICTIONS")
    predictions = predict_codes_rule_based(features)
    
    if predictions:
        print(f"\n{'Code':<12} {'Party':<20} {'Conf':<6} Description")
        print("-" * 70)
        for code, party, conf, reasons in predictions[:15]:
            desc = CODE_DESCRIPTIONS.get(code, '')[:30]
            print(f"{code:<12} {party:<20} {conf:.0%}    {desc}")
        
        if verbose and predictions:
            print("\nDetailed reasons for top predictions:")
            for code, party, conf, reasons in predictions[:5]:
                print(f"\n  {code} ({party}):")
                for r in reasons:
                    print(f"    {r}")
    else:
        print("  No codes predicted based on rules")
    
    # Model-based predictions
    if model_dir:
        print_section("MODEL-BASED PREDICTIONS")
        model_preds = predict_with_model(features, model_dir)
        
        if model_preds and model_preds[0][0] != 'ERROR':
            print(f"\n{'Code':<15} {'Probability':<12} Description")
            print("-" * 60)
            for code, prob in model_preds[:10]:
                desc = CODE_DESCRIPTIONS.get(code.split('_')[0], '')[:30]
                print(f"{code:<15} {prob:.1%}         {desc}")
        else:
            print(f"  {model_preds[0][1] if model_preds else 'No predictions'}")
    
    print_section("END OF ANALYSIS")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='ACE Pelican Payment Debug Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python debug_payment.py payment.json
  python debug_payment.py payment.json --verbose
  python debug_payment.py payment.json --code 9018
  python debug_payment.py payment.json --model-dir ./models_8x
        """
    )
    
    parser.add_argument('payment_file', help='Path to payment JSON file')
    parser.add_argument('--model-dir', '-m', help='Model directory for ML predictions')
    parser.add_argument('--code', '-c', help='Specific code to analyze')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show all features')
    
    args = parser.parse_args()
    
    # Load payment
    payment_path = Path(args.payment_file)
    if not payment_path.exists():
        print(f"Error: File not found: {payment_path}")
        sys.exit(1)
    
    with open(payment_path) as f:
        payment_data = json.load(f)
    
    # Handle different wrapper formats
    # If it's {txn_id: {Request: {...}}} format, extract the Request
    if len(payment_data) == 1:
        key = list(payment_data.keys())[0]
        if isinstance(payment_data[key], dict):
            if 'Request' in payment_data[key]:
                payment_data = payment_data[key]['Request']
            elif key not in ['IFML', 'File', 'Message', 'BasicPayment']:
                # It's a transaction ID wrapper
                if 'IFML' not in payment_data[key]:
                    payment_data = payment_data[key]
    
    # Run analysis
    analyze_payment(
        payment_data,
        model_dir=args.model_dir,
        target_code=args.code,
        verbose=args.verbose
    )


if __name__ == '__main__':
    main()
