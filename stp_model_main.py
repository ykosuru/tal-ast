#!/usr/bin/env python3
"""
ACE Pelican IFML Error Code Prediction System
=============================================

Main entry point for training models and making predictions on
ACE Pelican IFML payment messages.

Usage:
    # Train a model
    python main.py train --data-dir /path/to/data --output-dir /path/to/models
    
    # Predict on new data
    python main.py predict --model-dir /path/to/models --input /path/to/request.json
    
    # Run demo with sample data
    python main.py demo
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd

from ifml_parser import IFMLParser, IFMLResponseParser
from feature_engineering import IFMLFeatureEngineer, ErrorCodeLabelEncoder
from data_pipeline import IFMLDataPipeline, PaymentRecord, create_synthetic_training_data
from model_training import (
    ACEErrorCodeModel, ModelConfig, RuleExtractor,
    train_with_cross_validation
)
from predictor import ACEPredictor, PredictionResult


def _extract_simple_rules(tree, feature_names, max_rules=5, feature_engineer=None):
    """
    Extract human-readable rules from a decision tree.
    Returns the top rules that lead to positive predictions.
    Decodes encoded features back to original values when possible.
    """
    from sklearn.tree import _tree
    
    # Build reverse lookup for encoded features
    reverse_encoders = {}
    bool_features = set()
    freq_features = set()
    
    if feature_engineer:
        # Get label encoder reverse mappings
        for col, le in feature_engineer.label_encoders.items():
            encoded_name = f"{col}_encoded"
            reverse_encoders[encoded_name] = {i: c for i, c in enumerate(le.classes_)}
        
        # Track boolean and frequency features
        bool_features = set(feature_engineer.BOOL_COLS)
        freq_features = {f"{col}_freq" for col in feature_engineer.FREQUENCY_COLS}
    
    tree_ = tree.tree_
    rules = []
    
    def decode_condition(feature, threshold, direction):
        """Convert encoded condition to human-readable form."""
        # Handle boolean features
        if feature in bool_features:
            if direction == '<=':
                return f"{feature} = False"
            else:
                return f"{feature} = True"
        
        # Handle label-encoded features
        if feature in reverse_encoders:
            encoder = reverse_encoders[feature]
            base_name = feature.replace('_encoded', '')
            
            if direction == '<=':
                # Values at or below threshold
                matches = [v for k, v in encoder.items() if k <= threshold and not v.startswith('__')]
                if len(matches) == 1:
                    return f"{base_name} = '{matches[0]}'"
                elif len(matches) <= 3:
                    return f"{base_name} IN {matches}"
                else:
                    # Show what's excluded
                    excluded = [v for k, v in encoder.items() if k > threshold and not v.startswith('__')]
                    if len(excluded) <= 2:
                        return f"{base_name} NOT IN {excluded}"
            else:
                # Values above threshold
                matches = [v for k, v in encoder.items() if k > threshold and not v.startswith('__')]
                if len(matches) == 1:
                    return f"{base_name} = '{matches[0]}'"
                elif len(matches) <= 3:
                    return f"{base_name} IN {matches}"
        
        # Handle frequency-encoded features
        if feature in freq_features:
            base_name = feature.replace('_freq', '')
            if direction == '<=':
                if threshold < 0.1:
                    return f"{base_name} is rare"
                else:
                    return f"{base_name} frequency <= {threshold:.0%}"
            else:
                if threshold < 0.1:
                    return f"{base_name} is common"
                else:
                    return f"{base_name} frequency > {threshold:.0%}"
        
        # Handle numeric features
        display_name = feature.replace('_', ' ')
        if direction == '<=':
            if threshold == 0.5 and threshold == int(threshold):
                return f"{display_name} = 0"
            return f"{display_name} <= {threshold:.2f}"
        else:
            if threshold == 0.5 and threshold == int(threshold):
                return f"{display_name} > 0"
            return f"{display_name} > {threshold:.2f}"
    
    def recurse(node, path):
        if tree_.feature[node] == _tree.TREE_UNDEFINED:
            # Leaf node
            value = tree_.value[node][0]
            if len(value) > 1:
                total = value.sum()
                positive = value[1]
                confidence = positive / total if total > 0 else 0
                # Only include rules that predict positive class with some confidence
                if confidence > 0.3 and positive >= 2:
                    rules.append({
                        'condition': ' AND '.join(path) if path else 'Always',
                        'confidence': confidence,
                        'support': int(positive)
                    })
        else:
            feature = feature_names[tree_.feature[node]]
            threshold = tree_.threshold[node]
            
            # Left branch (<=)
            left_cond = decode_condition(feature, threshold, '<=')
            recurse(tree_.children_left[node], path + [left_cond])
            
            # Right branch (>)
            right_cond = decode_condition(feature, threshold, '>')
            recurse(tree_.children_right[node], path + [right_cond])
    
    recurse(0, [])
    
    # Sort by confidence and support, return top rules
    rules.sort(key=lambda x: (-x['confidence'], -x['support']))
    return rules[:max_rules]


def train_model(data_dir: str = None, data_file: str = None,
                output_dir: str = './models', 
                config: ModelConfig = None,
                filter_severity: List[str] = None,
                use_composite_codes: bool = False,
                min_code_samples: int = 5) -> Dict[str, Any]:
    """
    Train error code prediction model from IFML data.
    
    Args:
        data_dir: Directory containing IFML request/response files
        data_file: Single JSON file with multiple payments (alternative to data_dir)
        output_dir: Directory to save trained models
        config: Model configuration (uses defaults if None)
        filter_severity: Filter codes by severity (e.g., ['E'] for errors only)
        use_composite_codes: If True, use code+party labels (e.g., '8004_BNPPTY')
        min_code_samples: Minimum samples for a code to have its own class
    
    Returns:
        Training results dictionary
    """
    config = config or ModelConfig()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("ACE Error Code Model Training")
    print("=" * 60)
    
    # Load data
    print("\n[1/5] Loading data...")
    pipeline = IFMLDataPipeline()
    
    if data_file:
        # Single file with multiple payments
        n_loaded = pipeline.load_single_file(data_file)
    elif data_dir:
        data_path = Path(data_dir)
        if (data_path / 'requests').exists() and (data_path / 'responses').exists():
            # Paired directories
            n_loaded = pipeline.load_paired_files(
                str(data_path / 'requests'),
                str(data_path / 'responses')
            )
        else:
            # Combined files in directory
            n_loaded = pipeline.load_combined_files(str(data_path))
    else:
        print("ERROR: Must provide either --data-dir or --data-file")
        return {'error': 'No data source specified'}
    
    if n_loaded == 0:
        print("ERROR: No data loaded. Check data directory structure.")
        return {'error': 'No data loaded'}
    
    print(f"   Loaded {n_loaded} records")
    print(pipeline.describe())
    
    # Create dataset
    print("\n[2/5] Creating dataset...")
    X_raw, X_transformed, y_multilabel = pipeline.create_dataset(
        filter_severity=filter_severity,
        min_code_samples=min_code_samples,
        use_composite_codes=use_composite_codes
    )
    
    X = X_transformed.values.astype(np.float32)
    y = y_multilabel.values.astype(np.float32)
    feature_names = X_transformed.columns.tolist()
    class_names = y_multilabel.columns.tolist()
    
    print(f"   Features: {X.shape[1]}")
    print(f"   Classes: {y.shape[1]}")
    print(f"   Samples: {X.shape[0]}")
    
    # Train model
    print("\n[3/5] Training model with cross-validation...")
    model, cv_results = train_with_cross_validation(
        X, y, config, feature_names, class_names
    )
    
    # Evaluate
    print("\n[4/5] Evaluating model...")
    metrics = model.evaluate(X, y)
    
    print(f"\n   Model Performance:")
    print(f"   - Accuracy: {metrics.get('accuracy', 0):.4f}")
    print(f"   - Macro F1: {metrics.get('macro_f1', 0):.4f}")
    print(f"   - Micro F1: {metrics.get('micro_f1', 0):.4f}")
    
    # Per-class breakdown
    report = metrics.get('classification_report', {})
    if report:
        print("\n   Per-class performance:")
        print(f"   {'Code':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
        print(f"   {'-'*55}")
        for cls, vals in sorted(report.items(), key=lambda x: -x[1].get('support', 0)):
            if isinstance(vals, dict) and 'f1-score' in vals:
                print(f"   {cls:<15} {vals['precision']:>10.3f} {vals['recall']:>10.3f} {vals['f1-score']:>10.3f} {vals['support']:>10}")
    
    # Feature importance
    print("\n   Top 10 Important Features:")
    importance = model.get_feature_importance(10)
    for _, row in importance.iterrows():
        print(f"   - {row['feature']}: {row['importance']:.4f}")
    
    # Decision rules for each code
    print("\n" + "=" * 60)
    print("DECISION RULES BY CODE")
    print("=" * 60)
    if model.rule_extractor:
        for code in class_names:
            if code.startswith('__'):  # Skip __RARE__, __NO_ERROR__
                continue
            print(f"\n>>> {code}")
            
            # Get top features for this code
            top_feats = model.rule_extractor.get_top_features(code, top_n=3)
            if top_feats:
                print(f"    Key features: {', '.join([f[0] for f in top_feats])}")
            
            # Get simplified rules
            if code in model.rule_extractor.trees:
                tree = model.rule_extractor.trees[code]
                rules = _extract_simple_rules(tree, feature_names, max_rules=3, feature_engineer=pipeline.feature_engineer)
                for rule in rules:
                    print(f"    IF {rule['condition']} THEN {code} (confidence: {rule['confidence']:.0%})")
    else:
        print("   Rule extractor not available.")
    print("=" * 60)
    
    # Save models
    print("\n[5/5] Saving models...")
    model.save(str(output_path / 'model.pkl'))
    pipeline.save_encoders(str(output_path))
    
    # Extract rules for saving
    code_rules = {}
    if model.rule_extractor:
        for code in class_names:
            if code.startswith('__'):
                continue
            if code in model.rule_extractor.trees:
                tree = model.rule_extractor.trees[code]
                rules = _extract_simple_rules(tree, feature_names, max_rules=5, feature_engineer=pipeline.feature_engineer)
                top_feats = model.rule_extractor.get_top_features(code, top_n=5)
                code_rules[code] = {
                    'key_features': [f[0] for f in top_feats],
                    'rules': [{'condition': r['condition'], 'confidence': r['confidence']} for r in rules]
                }
    
    # Save training info
    training_info = {
        'n_samples': int(X.shape[0]),
        'n_features': int(X.shape[1]),
        'n_classes': int(y.shape[1]),
        'class_names': class_names,
        'feature_names': feature_names,
        'cv_mean_f1': float(cv_results['mean_f1']),
        'cv_std_f1': float(cv_results['std_f1']),
        'config': {
            'model_type': config.model_type,
            'task_type': config.task_type,
            'n_estimators': config.n_estimators,
            'max_depth': config.max_depth,
        },
        'decision_rules': code_rules
    }
    
    with open(output_path / 'training_info.json', 'w') as f:
        json.dump(training_info, f, indent=2)
    
    print(f"\n   Models saved to: {output_path}")
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    
    return {
        'metrics': metrics,
        'cv_results': cv_results,
        'training_info': training_info
    }


def predict(model_dir: str, input_path: str = None, 
            input_json: dict = None) -> PredictionResult:
    """
    Predict error codes for an IFML request.
    
    Args:
        model_dir: Directory containing trained models
        input_path: Path to IFML JSON file
        input_json: IFML as dict (alternative to input_path)
    
    Returns:
        PredictionResult
    """
    predictor = ACEPredictor(model_dir)
    
    if input_path:
        with open(input_path, 'r') as f:
            input_json = json.load(f)
    
    if not input_json:
        raise ValueError("Must provide either input_path or input_json")
    
    return predictor.predict(input_json)


def analyze(model_dir: str, input_path: str) -> Dict[str, Any]:
    """
    Comprehensive analysis of an IFML request.
    
    Args:
        model_dir: Directory containing trained models
        input_path: Path to IFML JSON file
    
    Returns:
        Analysis dictionary
    """
    predictor = ACEPredictor(model_dir)
    
    with open(input_path, 'r') as f:
        input_json = json.load(f)
    
    return predictor.analyze_request(input_json)


def run_demo():
    """Run demonstration with sample data."""
    print("=" * 60)
    print("ACE Error Code Prediction - Demo")
    print("=" * 60)
    
    # Sample IFML request (from user's uploaded file)
    sample_request = {
        "2025092900000192": {
            "Request": {
                "IFML": {
                    "File": {
                        "Message": {
                            "BasicPayment": {
                                "TransactionID": "2025092900000XXX",
                                "SourceCode": "SWF",
                                "IncomingMsgType": "103",
                                "IncomingFormat": "SWF",
                                "BankOperationCode": "CRED",
                                "MonetaryAmount": [
                                    {"@Type": "Amount", "@Currency": "USD", "Amount": "12000.00"},
                                    {"@Type": "BaseAmt", "Currency": "USD", "Amount": "12000.00"},
                                    {"@Type": "RelatedAmt", "SubType": "INSTRUCTED AMOUNT", 
                                     "Currency": "USD", "Amount": "11000.00"}
                                ],
                                "PartyInfo": {
                                    "OriginatingPartyInfo": {
                                        "BasicPartyInfo": {
                                            "ID": "102518",
                                            "AddressInfo": [
                                                {"@Seq": "1", "#text": "URS CORPORATION"},
                                                {"@Seq": "2", "#text": "600 MONTGOMERY ST"},
                                                {"@Seq": "3", "#text": "SAN FRANCISCO"},
                                                {"@Seq": "4", "#text": "CA UNITED STATES"}
                                            ],
                                            "Country": "US"
                                        }
                                    },
                                    "SendingBankInfo": {
                                        "BasicIDInfo": {
                                            "ID": {"@Type": "BIC", "#text": "USBKUS6F"},
                                            "Country": "US"
                                        }
                                    },
                                    "CreditPartyInfo": {
                                        "AccountPartyInfo": {
                                            "AcctIDInfo": {
                                                "ID": {"@Type": "BBAN", "#text": "01000021"}
                                            }
                                        },
                                        "BasicPartyInfo": {
                                            "ID": "41310123456",
                                            "AddressInfo": [
                                                {"@Seq": "1", "#text": "NINE UNITED LIMITED"},
                                                {"@Seq": "2", "#text": "15F, No.100"},
                                                {"@Seq": "3", "#text": "THE EAST YANAN ROAD"},
                                                {"@Seq": "4", "#text": "SHANGHAI"}
                                            ],
                                            "MailingCountry": "JP"
                                        }
                                    },
                                    "BeneficiaryBankInfo": {
                                        "BasicPartyBankInfo": {
                                            "AdrBankID": "121",
                                            "Type": "MS2"
                                        },
                                        "AddressInfo": [
                                            {"@Seq": "1", "#text": "SECURITIES CO., LTD"},
                                            {"@Seq": "2", "#text": "3-11"},
                                            {"@Seq": "3", "#text": "CHUO, JP-103-0026"}
                                        ],
                                        "Country": "JP"
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    # Sample responses showing different error codes
    sample_responses = [
        {
            "Response": {
                "IFML": {
                    "File": {
                        "Message": {
                            "BasicPayment": {"TransactionUID": "2025092900000193"},
                            "AuditTrail": {
                                "MsgStatus": [
                                    {"Code": "701", "Severity": "W", 
                                     "InformationalData": "CDTPTY Narrative Present"},
                                    {"Code": "602", "Severity": "I", 
                                     "InformationalData": "CDTPTY ISO Country Code extracted from Field"},
                                    {"Code": "6007", "Severity": "I", 
                                     "InformationalData": "CDTPTY NCH code extracted from field"}
                                ]
                            }
                        }
                    }
                }
            }
        },
        {
            "Response": {
                "IFML": {
                    "File": {
                        "Message": {
                            "BasicPayment": {"TransactionUID": "2025092900000282"},
                            "AuditTrail": {
                                "MsgStatus": [
                                    {"Code": "602", "Severity": "I", 
                                     "InformationalData": "BNFBNK ISO Country Code extracted from Field"},
                                    {"Code": "6007", "Severity": "I", 
                                     "InformationalData": "BNFBNK NCH code extracted from field"},
                                    {"Code": "8004", "Severity": "E", 
                                     "InformationalData": "BNPPTY IBAN cannot be derived"},
                                    {"Code": "7452", "Severity": "W", 
                                     "InformationalData": "CDTPTY No Fields for Push Down available"}
                                ]
                            }
                        }
                    }
                }
            }
        }
    ]
    
    # 1. Parse IFML
    print("\n[1] Parsing IFML Request")
    print("-" * 40)
    parser = IFMLParser()
    features = parser.parse(sample_request)
    feature_dict = parser.to_dict(features)
    
    print(f"Transaction: {feature_dict.get('transaction_id')}")
    print(f"Source: {feature_dict.get('source_code')} / {feature_dict.get('incoming_msg_type')}")
    print(f"Amount: {feature_dict.get('primary_currency')} {feature_dict.get('primary_amount')}")
    print(f"Cross-border: {feature_dict.get('is_cross_border')}")
    print(f"Credit Party Account Type: {feature_dict.get('cdt_account_type')}")
    print(f"Beneficiary Country: {feature_dict.get('beneficiary_country')}")
    
    # 2. Parse Responses
    print("\n[2] Parsing IFML Responses")
    print("-" * 40)
    resp_parser = IFMLResponseParser()
    
    for i, resp in enumerate(sample_responses):
        txn_uid, codes = resp_parser.parse(resp)
        print(f"\nResponse {i+1} (TXN: {txn_uid}):")
        for code in codes:
            severity_emoji = {'E': '❌', 'W': '⚠️', 'I': 'ℹ️'}.get(code['severity'], '•')
            print(f"  {severity_emoji} {code['code']} [{code['severity']}]: {code['info'][:50]}...")
    
    # 3. Create Training Data Pipeline
    print("\n[3] Creating Training Pipeline")
    print("-" * 40)
    
    pipeline = IFMLDataPipeline()
    n_loaded = pipeline.load_sample_data(sample_request, sample_responses)
    print(f"Loaded {n_loaded} sample records")
    
    # 4. Generate Synthetic Data for Demo
    print("\n[4] Generating Synthetic Training Data")
    print("-" * 40)
    
    features_df, code_lists = create_synthetic_training_data(n_samples=500)
    
    # Add synthetic records to pipeline
    for i, (_, row) in enumerate(features_df.iterrows()):
        severity_map = {
            c: 'E' if c.startswith('8') else 'W' if c.startswith('7') else 'I' 
            for c in code_lists[i]
        }
        record = PaymentRecord(
            transaction_id=f"SYNTH{i:06d}",
            request_features=row.to_dict(),
            response_codes=[
                {'code': c, 'severity': severity_map[c], 'info': f'Synthetic {c}'} 
                for c in code_lists[i]
            ],
            error_codes_only=code_lists[i],
            severity_map=severity_map
        )
        pipeline.records.append(record)
    
    print(f"Total records: {len(pipeline.records)}")
    print("\nCode distribution (top 10):")
    stats = pipeline.get_code_statistics()
    for _, row in stats.head(10).iterrows():
        print(f"  {row['code']}: {row['total']} ({row['pct']:.1f}%)")
    
    # 5. Train Model
    print("\n[5] Training Model")
    print("-" * 40)
    
    X_raw, X_transformed, y_multilabel = pipeline.create_dataset(
        filter_severity=['E', 'W'],  # Focus on errors and warnings
        min_code_samples=15
    )
    
    X = X_transformed.values.astype(np.float32)
    y = y_multilabel.values.astype(np.float32)
    feature_names = X_transformed.columns.tolist()
    class_names = y_multilabel.columns.tolist()
    
    print(f"Dataset shape: {X.shape[0]} samples x {X.shape[1]} features")
    print(f"Target classes: {len(class_names)}")
    
    config = ModelConfig(
        model_type='random_forest',
        task_type='multilabel',
        n_estimators=100,
        max_depth=6,
        cv_folds=3
    )
    
    model, cv_results = train_with_cross_validation(
        X, y, config, feature_names, class_names
    )
    
    # 6. Feature Importance
    print("\n[6] Feature Importance")
    print("-" * 40)
    
    importance = model.get_feature_importance(15)
    for _, row in importance.iterrows():
        bar = '█' * int(row['importance'] * 50)
        print(f"  {row['feature'][:25]:<25} {bar} {row['importance']:.3f}")
    
    # 7. Rule Extraction
    print("\n[7] Extracted Rules for Key Codes")
    print("-" * 40)
    
    if model.rule_extractor:
        for code in ['8004', '7452', '602'][:2]:
            if code in model.rule_extractor.rules:
                print(f"\nRules for {code}:")
                rules = model.rule_extractor.get_rules(code)
                # Print first few lines
                for line in rules.split('\n')[:10]:
                    print(f"  {line}")
                print("  ...")
    
    # 8. Sample Predictions
    print("\n[8] Sample Predictions")
    print("-" * 40)
    
    # Predict on original sample
    X_sample = model.scaler.transform(X[:3])
    predictions = model.predict(X[:3])
    probabilities = model.predict_proba(X[:3])
    
    for i in range(3):
        print(f"\nSample {i+1}:")
        pred_codes = [class_names[j] for j, p in enumerate(predictions[i]) if p == 1]
        print(f"  Predicted codes: {pred_codes if pred_codes else 'None'}")
        
        # Top probabilities
        top_probs = sorted(
            [(class_names[j], probabilities[i][j]) for j in range(len(class_names))],
            key=lambda x: x[1],
            reverse=True
        )[:5]
        print(f"  Top probabilities:")
        for code, prob in top_probs:
            print(f"    {code}: {prob:.3f}")
    
    # 9. Explanation
    print("\n[9] Prediction Explanation")
    print("-" * 40)
    
    explanation = model.explain_prediction(X, sample_idx=0)
    print(f"Prediction: {explanation['prediction']}")
    print(f"Top contributing features:")
    for feat in explanation.get('top_features', [])[:5]:
        print(f"  - {feat['feature']}: value={feat['value']:.2f}")
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Load your production IFML data")
    print("2. Train: python main.py train --data-dir /your/data --output-dir ./models")
    print("3. Predict: python main.py predict --model-dir ./models --input request.json")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='ACE Pelican IFML Error Code Prediction System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run demo with sample data
  python main.py demo
  
  # Train model from single file (your format)
  python main.py train --data-file payments.json --output-dir ./models
  
  # Train model from directory
  python main.py train --data-dir ./data --output-dir ./models
  
  # Predict
  python main.py predict --model-dir ./models --input request.json
  
  # Analyze request
  python main.py analyze --model-dir ./models --input request.json
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run demonstration')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train model')
    train_parser.add_argument('--data-dir', help='Directory with data files')
    train_parser.add_argument('--data-file', help='Single JSON file with all payments')
    train_parser.add_argument('--output-dir', default='./models', help='Output directory')
    train_parser.add_argument('--model-type', default='random_forest',
                             choices=['lightgbm', 'xgboost', 'random_forest', 'gradient_boost', 'decision_tree'])
    train_parser.add_argument('--severity', nargs='+', default=None,
                             help='Filter codes by severity (E, W, I, R). Default: all')
    train_parser.add_argument('--composite', action='store_true',
                             help='Use composite labels (code+party, e.g., 8004_BNPPTY)')
    train_parser.add_argument('--min-samples', type=int, default=5,
                             help='Minimum samples for a code to have its own class (default: 5)')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict error codes')
    predict_parser.add_argument('--model-dir', required=True, help='Model directory')
    predict_parser.add_argument('--input', required=True, help='Input IFML JSON file')
    predict_parser.add_argument('--threshold', type=float, default=0.5,
                               help='Prediction threshold')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze request')
    analyze_parser.add_argument('--model-dir', required=True, help='Model directory')
    analyze_parser.add_argument('--input', required=True, help='Input IFML JSON file')
    
    args = parser.parse_args()
    
    if args.command == 'demo':
        run_demo()
    
    elif args.command == 'train':
        if not args.data_dir and not args.data_file:
            print("ERROR: Must provide either --data-dir or --data-file")
            sys.exit(1)
        
        config = ModelConfig(
            model_type=args.model_type,
            task_type='multilabel'
        )
        results = train_model(
            data_dir=args.data_dir,
            data_file=args.data_file,
            output_dir=args.output_dir,
            config=config,
            filter_severity=args.severity,
            use_composite_codes=args.composite,
            min_code_samples=args.min_samples
        )
        print(json.dumps(results.get('training_info', {}), indent=2))
    
    elif args.command == 'predict':
        result = predict(args.model_dir, args.input)
        print(json.dumps({
            'transaction_id': result.transaction_id,
            'predicted_codes': result.predicted_codes,
            'confidence': result.confidence,
            'probabilities': {k: round(v, 4) for k, v in result.probabilities.items()},
            'warnings': result.warnings
        }, indent=2))
    
    elif args.command == 'analyze':
        analysis = analyze(args.model_dir, args.input)
        print(json.dumps(analysis, indent=2, default=str))
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
