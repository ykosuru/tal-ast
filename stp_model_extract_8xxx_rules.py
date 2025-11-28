#!/usr/bin/env python3
"""
Explain 8XXX predictions - show which input fields triggered each code.

Usage:
    python explain_predictions.py --model-dir ./models_8x --data-dir ./raw_data --output explanations.md --limit 100
"""

import argparse
import json
import pickle
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import numpy as np


# Feature to human-readable explanation mapping
FEATURE_EXPLANATIONS = {
    # IBAN related
    'needs_iban': 'Party is in IBAN-required country',
    'has_iban': 'IBAN is provided',
    'iban_valid_format': 'IBAN format is valid',
    'iban_checksum_valid': 'IBAN checksum passes',
    'missing_required_iban': 'Required IBAN is missing',
    
    # BIC related
    'has_bic': 'BIC is provided',
    'bic_valid_format': 'BIC format is valid (8 or 11 chars)',
    'bic_valid_country': 'BIC country code is valid',
    'bic_iban_match': 'BIC and IBAN countries match',
    'bic_party_country_match': 'BIC matches party country',
    
    # NCH/Routing related
    'has_nch': 'NCH/routing number is provided',
    'nch_valid': 'NCH format is valid',
    'fedaba_checksum_valid': 'FEDABA checksum passes',
    'nch_validation_applicable': 'NCH validation applies to this payment',
    'has_adr_bank_id': 'Bank ID in address block',
    
    # Account related
    'has_account': 'Account number is provided',
    'account_numeric': 'Account is numeric only',
    'is_iban': 'Account is IBAN format',
    'is_clabe': 'Account is CLABE format (Mexico)',
    'is_fedaba': 'Account is FEDABA format',
    
    # Party presence
    'present': 'Party is present in message',
    'has_id': 'Party has ID',
    'has_name': 'Party has name',
    
    # Payment type
    'is_domestic': 'Domestic payment',
    'is_international': 'International payment',
    'is_cross_border': 'Cross-border payment',
}

# Code to relevant features mapping - based on ACE Pelican definitions
# Key: (features_that_must_be_true, features_that_must_be_false)
CODE_TRIGGERS = {
    '8001': {  # Invalid BIC
        'require_true': ['has_bic'],
        'require_false': ['bic_valid_format', 'bic_valid_country'],
        'description': 'BIC present but format/country invalid'
    },
    '8004': {  # IBAN cannot be derived
        'require_true': ['needs_iban'],
        'require_false': ['has_iban'],
        'description': 'IBAN required but missing and cannot be derived'
    },
    '8022': {  # IBAN inconsistent with Account With Institution BIC
        'require_true': ['has_iban', 'has_bic'],
        'require_false': ['bic_iban_match'],
        'description': 'IBAN and BIC countries do not match'
    },
    '8026': {  # NCH inconsistency found in message
        'require_true': ['has_nch'],
        'require_false': [],  # Inconsistency between multiple NCH values
        'check_field': 'nch_sources',  # Multiple sources with different values
        'description': 'Multiple NCH values that are inconsistent'
    },
    '8030': {  # IBAN derivation not supported for the country
        'require_true': ['needs_iban'],
        'require_false': ['has_iban'],
        'description': 'Country does not support IBAN derivation'
    },
    '8852': {  # Incorrect length of attribute
        'require_true': ['has_account'],
        'require_false': [],
        'description': 'Field length is incorrect'
    },
    '8892': {  # Invalid Account number
        'require_true': ['has_account'],
        'require_false': ['account_numeric'],  # or other validation
        'description': 'Account number format is invalid'
    },
    '8894': {  # Invalid IBAN
        'require_true': ['has_iban'],
        'require_false': ['iban_valid_format', 'iban_checksum_valid'],
        'description': 'IBAN present but format or checksum invalid'
    },
    '8895': {  # Invalid NCH code
        'require_true': ['has_nch'],
        'require_false': ['nch_valid', 'fedaba_checksum_valid'],
        'description': 'NCH/routing number present but invalid'
    },
    '8896': {  # Invalid Domestic Account Number
        'require_true': ['has_account', 'is_domestic'],
        'require_false': [],
        'description': 'Domestic account number format invalid'
    },
    '8898': {  # IBAN Check Digit calculation/validation failed
        'require_true': ['has_iban'],
        'require_false': ['iban_checksum_valid'],
        'description': 'IBAN checksum validation failed'
    },
}

# Legacy mapping for backward compatibility
CODE_RELEVANT_FEATURES = {
    '8004': ['needs_iban', 'has_iban', 'missing_required_iban', 'is_international'],
    '8022': ['has_iban', 'has_bic', 'bic_iban_match', 'iban_valid_format'],
    '8026': ['has_nch', 'nch_valid', 'has_adr_bank_id', 'nch_sources'],
    '8001': ['has_bic', 'bic_valid_format', 'bic_valid_country'],
    '8852': ['has_account', 'account_length'],
    '8894': ['has_iban', 'iban_valid_format', 'iban_checksum_valid'],
    '8895': ['has_nch', 'nch_valid', 'fedaba_checksum_valid', 'nch_validation_applicable', 'is_domestic'],
    '8896': ['has_nch', 'is_domestic', 'nch_validation_applicable'],
    '8898': ['has_iban', 'iban_checksum_valid'],
}

# Party prefixes
PARTY_PREFIXES = ['orig', 'send', 'dbt', 'cdt', 'intm', 'bnf', 'ordi', 'acwi']

PARTY_NAMES = {
    'orig': 'Originator',
    'send': 'Sending Bank', 
    'dbt': 'Debit Party',
    'cdt': 'Credit Party',
    'intm': 'Intermediary Bank',
    'bnf': 'Beneficiary Bank',
    'ordi': 'Originator (alt)',
    'acwi': 'Credit Party (alt)',
}


def get_feature_explanation(feature_name: str) -> str:
    """Get human-readable explanation for a feature."""
    # Strip prefix
    for prefix in PARTY_PREFIXES:
        if feature_name.startswith(f'{prefix}_'):
            base_name = feature_name[len(prefix)+1:]
            party_name = PARTY_NAMES.get(prefix, prefix)
            base_explanation = FEATURE_EXPLANATIONS.get(base_name, base_name)
            return f"{party_name}: {base_explanation}"
    
    return FEATURE_EXPLANATIONS.get(feature_name, feature_name)


def explain_prediction(features: Dict, predicted_codes: List[str], 
                       feature_importance: Optional[Dict] = None) -> Dict[str, List[dict]]:
    """
    Explain why each code was predicted using ACE Pelican trigger logic.
    
    Returns dict mapping code -> dict with:
        - triggers: list of triggering conditions
        - warning: optional warning if prediction may be false positive
        - description: code description from ACE
    """
    explanations = {}
    
    for code in predicted_codes:
        base_code = code.split('_')[0]
        party_suffix = code.split('_')[1] if '_' in code else None
        
        triggers = []
        warning = None
        
        # Get trigger definition for this code
        trigger_def = CODE_TRIGGERS.get(base_code, {})
        require_true = trigger_def.get('require_true', [])
        require_false = trigger_def.get('require_false', [])
        description = trigger_def.get('description', '')
        
        # Determine which party to check
        if party_suffix:
            party_prefix = _suffix_to_prefix(party_suffix)
            prefixes_to_check = [party_prefix] if party_prefix else PARTY_PREFIXES
        else:
            prefixes_to_check = PARTY_PREFIXES
        
        # Track if trigger conditions are met
        true_conditions_met = []
        true_conditions_failed = []
        false_conditions_met = []  # Should be False, and is False
        false_conditions_failed = []  # Should be False, but is True
        
        for prefix in prefixes_to_check:
            # Check require_true features (must be True for valid trigger)
            for feat_base in require_true:
                feat_name = f'{prefix}_{feat_base}'
                if feat_name in features:
                    value = features[feat_name]
                    if isinstance(value, bool):
                        if value:
                            true_conditions_met.append({
                                'feature': feat_name,
                                'value': value,
                                'explanation': get_feature_explanation(feat_name),
                                'trigger_type': 'present'
                            })
                        else:
                            true_conditions_failed.append(feat_name)
            
            # Check require_false features (must be False for valid trigger)
            for feat_base in require_false:
                feat_name = f'{prefix}_{feat_base}'
                if feat_name in features:
                    value = features[feat_name]
                    if isinstance(value, bool):
                        if not value:
                            false_conditions_met.append({
                                'feature': feat_name,
                                'value': value,
                                'explanation': get_feature_explanation(feat_name),
                                'trigger_type': 'invalid/missing'
                            })
                        else:
                            false_conditions_failed.append(feat_name)
        
        # Build trigger list - show conditions that match expected pattern
        triggers = true_conditions_met + false_conditions_met
        
        # Detect potential false positive
        if require_true and not true_conditions_met:
            # Required condition not present
            warning = f"⚠️ POSSIBLE FALSE POSITIVE: {base_code} requires {require_true} but none found"
        elif true_conditions_failed and require_true:
            # e.g., 8022 requires has_iban=True but has_iban=False
            warning = f"⚠️ POSSIBLE FALSE POSITIVE: Expected {true_conditions_failed} = True but got False"
        
        # Sort by importance if available
        if feature_importance:
            triggers.sort(key=lambda x: -feature_importance.get(x['feature'], 0))
        
        explanations[code] = {
            'triggers': triggers,
            'warning': warning,
            'description': description
        }
    
    return explanations


def _suffix_to_prefix(suffix: str) -> Optional[str]:
    """Convert party suffix to feature prefix."""
    mapping = {
        'ORGPTY': 'orig',
        'SNDBNK': 'send',
        'DBTPTY': 'dbt',
        'CDTPTY': 'cdt',
        'INTBNK': 'intm',
        'BNFBNK': 'bnf',
        'BNPPTY': 'bnf',  # Beneficiary party often maps to bnf
    }
    return mapping.get(suffix)


def format_explanation(txn_id: str, features: Dict, predicted_codes: List[str], 
                       actual_codes: List[str], explanations: Dict) -> List[str]:
    """Format explanation as markdown."""
    lines = []
    lines.append(f"### Transaction: {txn_id}")
    lines.append("")
    
    # Show prediction vs actual
    pred_str = ', '.join(predicted_codes) if predicted_codes else 'None'
    actual_str = ', '.join(actual_codes) if actual_codes else 'None'
    
    match = set(predicted_codes) == set(actual_codes)
    status = "✓ MATCH" if match else "✗ MISMATCH"
    
    lines.append(f"**Predicted:** {pred_str}")
    lines.append(f"**Actual:** {actual_str}")
    lines.append(f"**Status:** {status}")
    lines.append("")
    
    # Show explanations for each predicted code
    for code in predicted_codes:
        lines.append(f"#### Why {code}?")
        lines.append("")
        
        code_explanation = explanations.get(code, {})
        
        # Handle both old format (list) and new format (dict)
        if isinstance(code_explanation, list):
            triggers = code_explanation
            warning = None
            description = None
        else:
            triggers = code_explanation.get('triggers', [])
            warning = code_explanation.get('warning')
            description = code_explanation.get('description')
        
        # Show description
        if description:
            lines.append(f"**ACE Definition:** {description}")
            lines.append("")
        
        # Show warning if present
        if warning:
            lines.append(f"**{warning}**")
            lines.append("")
        
        if triggers:
            lines.append("| Field | Value | Explanation |")
            lines.append("|-------|-------|-------------|")
            for t in triggers[:5]:  # Top 5 triggers
                feat = t['feature']
                val = t['value']
                expl = t['explanation']
                lines.append(f"| `{feat}` | {val} | {expl} |")
            lines.append("")
        else:
            lines.append("*No specific triggers identified (may be pattern-based)*")
            lines.append("")
    
    # Show key input values
    lines.append("#### Key Input Fields")
    lines.append("")
    lines.append("```")
    
    key_fields = [
        'primary_currency', 'is_cross_border',
        'bnf_has_iban', 'bnf_needs_iban', 'bnf_has_bic', 'bnf_has_nch',
        'bnf_bic_iban_match', 'bnf_iban_valid_format', 'bnf_iban_checksum_valid',
        'cdt_has_iban', 'cdt_needs_iban', 'cdt_has_account',
        'intm_present', 'intm_has_bic',
    ]
    
    for field in key_fields:
        if field in features:
            lines.append(f"{field}: {features[field]}")
    
    lines.append("```")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    return lines


def load_feature_importance(model_dir: str) -> Dict[str, float]:
    """Load feature importance from model."""
    importance = {}
    
    try:
        info_path = Path(model_dir) / 'training_info.json'
        if info_path.exists():
            with open(info_path) as f:
                info = json.load(f)
                if 'feature_importance' in info:
                    for item in info['feature_importance']:
                        importance[item['feature']] = item['importance']
    except Exception as e:
        print(f"Warning: Could not load feature importance: {e}")
    
    return importance


def load_model_components(model_dir: str):
    """Load model, feature engineer, and class names."""
    import pandas as pd
    
    model_path = Path(model_dir) / 'model.pkl'
    fe_path = Path(model_dir) / 'feature_engineer.pkl'
    info_path = Path(model_dir) / 'training_info.json'
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    # Handle model as dict
    if isinstance(model_data, dict):
        model = model_data.get('model')
        print(f"Extracted model from dict, type: {type(model).__name__}")
    else:
        model = model_data
        print(f"Model type: {type(model).__name__}")
    
    with open(fe_path, 'rb') as f:
        feature_engineer = pickle.load(f)
    
    # If feature_engineer is a dict, try to reconstruct from module
    if isinstance(feature_engineer, dict):
        try:
            from feature_engineering import IFMLFeatureEngineer
            fe_obj = IFMLFeatureEngineer()
            # Restore fitted state from dict
            if 'label_encoders' in feature_engineer:
                fe_obj.label_encoders = feature_engineer['label_encoders']
            if 'frequency_maps' in feature_engineer:
                fe_obj.frequency_maps = feature_engineer['frequency_maps']
            if 'feature_columns' in feature_engineer:
                fe_obj.feature_columns = feature_engineer['feature_columns']
            fe_obj.fitted = True
            feature_engineer = fe_obj
            print("Reconstructed feature engineer from dict")
        except ImportError:
            print("Warning: Could not import IFMLFeatureEngineer, using dict mode")
    
    with open(info_path) as f:
        info = json.load(f)
    
    class_names = info.get('class_names', [])
    threshold = info.get('threshold', 0.5)
    feature_columns = info.get('feature_columns', [])
    
    return model, feature_engineer, class_names, threshold, feature_columns


def transform_features(features: Dict, feature_engineer, feature_columns: List[str]):
    """Transform features dict to model input, handling dict or object feature_engineer."""
    import pandas as pd
    
    # If feature_engineer is an object with transform method, use it
    if hasattr(feature_engineer, 'transform'):
        df = pd.DataFrame([features])
        return feature_engineer.transform(df)
    
    # If feature_engineer is a dict, manually build feature vector
    if isinstance(feature_engineer, dict):
        # Get label encoders if present
        label_encoders = feature_engineer.get('label_encoders', {})
        frequency_maps = feature_engineer.get('frequency_maps', {})
        columns = feature_engineer.get('feature_columns', feature_columns)
        
        if not columns:
            raise ValueError("No feature_columns found in feature_engineer or training_info")
        
        # Build feature vector
        row = {}
        for col in columns:
            if col in features:
                val = features[col]
                
                # Handle encoded columns
                if col in label_encoders:
                    le = label_encoders[col]
                    if hasattr(le, 'transform'):
                        try:
                            val = le.transform([str(val)])[0]
                        except:
                            val = 0  # Unknown value
                    elif isinstance(le, dict):
                        val = le.get(str(val), 0)
                
                # Handle frequency encoded columns
                elif col in frequency_maps:
                    freq_map = frequency_maps[col]
                    val = freq_map.get(str(val), 0.0)
                
                # Handle booleans
                elif isinstance(val, bool):
                    val = 1 if val else 0
                
                row[col] = val
            else:
                # Default values
                if '_encoded' in col or '_freq' in col:
                    row[col] = 0
                else:
                    row[col] = 0
        
        df = pd.DataFrame([row])
        # Ensure column order matches training
        for col in columns:
            if col not in df.columns:
                df[col] = 0
        df = df[columns]
        
        return df.values
    
    raise ValueError(f"Unknown feature_engineer type: {type(feature_engineer)}")


def predict_from_features(features: Dict, model, feature_engineer, 
                          class_names: List[str], threshold: float = 0.5,
                          series: str = '8', feature_columns: List[str] = None) -> List[str]:
    """Run prediction directly from features dict."""
    import pandas as pd
    
    # Transform features
    X = transform_features(features, feature_engineer, feature_columns or [])
    
    # Get probabilities
    if hasattr(model, 'predict_proba'):
        probas = model.predict_proba(X)
        
        # Handle different probability formats
        if isinstance(probas, list):
            # MultiOutput: list of arrays
            predicted = []
            for idx, proba in enumerate(probas):
                if idx < len(class_names):
                    code = class_names[idx]
                    if code.startswith(series):
                        if len(proba.shape) > 1 and proba.shape[1] >= 2:
                            prob = proba[0, 1]
                        else:
                            prob = proba[0]
                        if prob >= threshold:
                            predicted.append(code)
            return predicted
        else:
            # Single output
            predicted = []
            for idx, prob in enumerate(probas[0]):
                if idx < len(class_names):
                    code = class_names[idx]
                    if code.startswith(series) and prob >= threshold:
                        predicted.append(code)
            return predicted
    else:
        # No probabilities, use predict
        preds = model.predict(X)
        predicted = []
        for idx, pred in enumerate(preds[0]):
            if pred == 1 and idx < len(class_names):
                code = class_names[idx]
                if code.startswith(series):
                    predicted.append(code)
        return predicted


def run_explanations(model_dir: str, data_dir: str, output_file: str, 
                     limit: int = 100, series: str = '8'):
    """Run predictions and generate explanations."""
    from data_pipeline import IFMLDataPipeline
    
    # Load model components
    print(f"Loading model from {model_dir}...")
    model, feature_engineer, class_names, threshold, feature_columns = load_model_components(model_dir)
    
    # Load feature importance
    importance = load_feature_importance(model_dir)
    
    # Load data
    print(f"Loading data from {data_dir}...")
    pipeline = IFMLDataPipeline()
    pipeline.load_combined_files(data_dir, '*.json')
    
    print(f"Loaded {len(pipeline.records)} records")
    
    lines = [
        f"# {series}XXX Prediction Explanations",
        "",
        "## Overview",
        "",
        f"This document shows what input fields triggered each {series}XXX prediction.",
        "",
        "---",
        ""
    ]
    
    # Process records
    count = 0
    for rec in pipeline.records:
        if count >= limit:
            break
        
        features = rec.request_features
        # Use composite_codes if available, otherwise error_codes_only
        if hasattr(rec, 'composite_codes') and rec.composite_codes:
            actual = [c for c in rec.composite_codes if c.startswith(series)]
        else:
            actual = [c for c in rec.error_codes_only if c.startswith(series)]
        
        if not actual:
            continue  # Skip records without target codes
        
        # Run prediction
        predicted = predict_from_features(features, model, feature_engineer, 
                                          class_names, threshold, series, feature_columns)
        
        # Get explanations
        explanations = explain_prediction(features, predicted, importance)
        
        # Format
        txn_id = rec.transaction_id if hasattr(rec, 'transaction_id') else f"record_{count}"
        formatted = format_explanation(txn_id, features, predicted, actual, explanations)
        lines.extend(formatted)
        
        count += 1
        if count % 20 == 0:
            print(f"Processed {count} records...")
    
    # Write output
    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"Explanations written to {output_file}")


def explain_single_payment(model_dir: str, payment_json: str, series: str = '8'):
    """Explain prediction for a single payment."""
    from ifml_parser import IFMLParser
    
    # Load model components
    model, feature_engineer, class_names, threshold, feature_columns = load_model_components(model_dir)
    importance = load_feature_importance(model_dir)
    
    # Parse payment
    with open(payment_json) as f:
        data = json.load(f)
    
    parser = IFMLParser()
    features_obj = parser.parse(data)
    features = parser.to_dict(features_obj)
    
    # Predict
    predicted = predict_from_features(features, model, feature_engineer,
                                       class_names, threshold, series, feature_columns)
    
    # Explain
    explanations = explain_prediction(features, predicted, importance)
    
    print("\n" + "="*60)
    print("PREDICTION EXPLANATION")
    print("="*60)
    print(f"\nPredicted codes: {predicted}")
    print("")
    
    for code in predicted:
        print(f"\n--- {code} ---")
        triggers = explanations.get(code, [])
        if triggers:
            print(f"{'Feature':<40} {'Value':<10} {'Reason'}")
            print("-" * 70)
            for t in triggers:
                print(f"{t['feature']:<40} {str(t['value']):<10} {t['explanation']}")
        else:
            print("No specific triggers identified")
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description='Explain 8XXX predictions')
    parser.add_argument('--model-dir', required=True, help='Model directory')
    parser.add_argument('--data-dir', default=None, help='Raw data directory')
    parser.add_argument('--payment', default=None, help='Single payment JSON file')
    parser.add_argument('--output', default='explanations.md', help='Output file')
    parser.add_argument('--limit', type=int, default=100, help='Max records to process')
    parser.add_argument('--series', default='8', help='Code series (8 or 9)')
    
    args = parser.parse_args()
    
    if args.payment:
        explain_single_payment(args.model_dir, args.payment, args.series)
    elif args.data_dir:
        run_explanations(args.model_dir, args.data_dir, args.output, args.limit, args.series)
    else:
        print("Error: Either --data-dir or --payment required")


if __name__ == '__main__':
    main()
