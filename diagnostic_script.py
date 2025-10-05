#!/usr/bin/env python3
"""
Diagnostic script to identify why predictions are poor
"""

import json
import pickle
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def diagnose_model(model_dir: str, test_file: str = None):
    """Diagnose issues with the trained model"""
    
    print("="*70)
    print("MODEL DIAGNOSTICS")
    print("="*70)
    
    # Load processor
    with open(Path(model_dir) / 'processor.pkl', 'rb') as f:
        processor = pickle.load(f)
    
    # 1. Check vocabulary distribution
    print("\n1. VOCABULARY ANALYSIS:")
    print(f"   Total repair types: {len(processor.repair_vocabulary)}")
    
    # Check if vocabulary is properly built
    if len(processor.repair_vocabulary) == 0:
        print("   ❌ ERROR: Empty vocabulary! Model cannot make predictions.")
        return
    
    print("\n   Sample repair IDs in vocabulary:")
    for repair_id in list(processor.repair_vocabulary.keys())[:10]:
        print(f"   - {repair_id}")
    
    # 2. Check feature patterns learned
    print("\n2. PATTERN ANALYSIS:")
    if hasattr(processor, 'repair_patterns'):
        total_patterns = sum(len(patterns) for patterns in processor.repair_patterns.values())
        print(f"   Total patterns learned: {total_patterns}")
        
        if total_patterns == 0:
            print("   ❌ ERROR: No patterns learned! Training may have failed.")
        else:
            # Show top patterns
            print("\n   Top repair patterns:")
            for repair_id in list(processor.repair_patterns.keys())[:5]:
                patterns = processor.repair_patterns[repair_id]
                if patterns:
                    top_pattern = max(patterns.items(), key=lambda x: x[1])
                    print(f"   - Repair {repair_id}: {top_pattern[0]} ({top_pattern[1]} times)")
    
    # 3. Check structural detector patterns
    print("\n3. STRUCTURAL PATTERNS:")
    if hasattr(processor, 'structural_detector'):
        detector = processor.structural_detector
        
        # Source/clearing patterns
        if hasattr(detector, 'source_clearing_patterns'):
            num_patterns = len(detector.source_clearing_patterns)
            print(f"   Source/clearing patterns: {num_patterns}")
            if num_patterns == 0:
                print("   ⚠️ WARNING: No source/clearing patterns learned")
        
        # Database lookup patterns
        if hasattr(detector, 'database_lookup_patterns'):
            for lookup_type, examples in detector.database_lookup_patterns.items():
                print(f"   - {lookup_type}: {len(examples)} examples")
    
    # 4. Feature extraction test
    if test_file:
        print("\n4. FEATURE EXTRACTION TEST:")
        with open(test_file, 'r') as f:
            test_data = json.load(f)
        
        # Extract transaction
        if isinstance(test_data, dict) and len(test_data) == 1:
            txn_id = list(test_data.keys())[0]
            txn_data = test_data[txn_id]
        else:
            txn_data = test_data
            txn_id = "test"
        
        # Try to extract features
        try:
            if hasattr(processor, 'structural_detector'):
                analysis = processor.structural_detector.analyze_transaction(txn_id, txn_data)
                features = processor._extract_features(txn_data, analysis)
                
                print(f"   Feature vector shape: {features.shape}")
                print(f"   Non-zero features: {np.sum(features != 0)}/{len(features)}")
                print(f"   Feature statistics:")
                print(f"   - Mean: {np.mean(features):.4f}")
                print(f"   - Std:  {np.std(features):.4f}")
                print(f"   - Max:  {np.max(features):.4f}")
                
                # Check for feature problems
                if np.sum(features != 0) < 5:
                    print("   ❌ ERROR: Very few active features! Feature extraction may be failing.")
                
                # Show which features are active
                print("\n   Active feature indices:")
                active_indices = np.where(features != 0)[0]
                print(f"   {active_indices[:20]}...")  # Show first 20
                
            else:
                print("   ❌ ERROR: No structural detector found")
        except Exception as e:
            print(f"   ❌ ERROR extracting features: {e}")
    
    # 5. Check if models exist
    print("\n5. MODEL FILES:")
    model_files = ['rf_model.pkl', 'nn_model.pt', 'model_info.json']
    for file in model_files:
        path = Path(model_dir) / file
        if path.exists():
            print(f"   ✓ {file}: {path.stat().st_size / 1024:.1f} KB")
        else:
            print(f"   ❌ {file}: Missing!")
    
    # 6. Recommendations
    print("\n6. RECOMMENDATIONS:")
    print("   If predictions are poor, consider:")
    print("   1. Check if training completed successfully")
    print("   2. Verify training data has both 'before' and 'after' states")
    print("   3. Ensure sufficient training examples (>1000 transactions)")
    print("   4. Check if repair IDs in test match training vocabulary")
    print("   5. Verify feature extraction is working correctly")

def analyze_training_data(data_dir: str):
    """Analyze training data quality"""
    print("\n" + "="*70)
    print("TRAINING DATA ANALYSIS")
    print("="*70)
    
    json_files = list(Path(data_dir).glob('**/*.json'))
    print(f"\nFound {len(json_files)} JSON files")
    
    repair_counts = Counter()
    entity_counts = Counter()
    has_before_after = 0
    total_transactions = 0
    
    for json_file in json_files[:10]:  # Sample first 10 files
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Extract transactions
            if isinstance(data, dict):
                transactions = data
            elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                transactions = {}
                for item in data:
                    if len(item) == 1:
                        key = list(item.keys())[0]
                        transactions[key] = item[key]
            else:
                continue
            
            for txn_id, txn_data in transactions.items():
                if not isinstance(txn_data, dict):
                    continue
                
                total_transactions += 1
                
                # Count repairs
                for repair in txn_data.get('ace', []):
                    if isinstance(repair, dict):
                        repair_id = repair.get('id', 'unknown')
                    else:
                        repair_id = str(repair)
                    repair_counts[repair_id] += 1
                
                # Check for before/after
                entities_with_changes = 0
                for entity in ['cdtrAgt', 'dbtrAgt', 'instgAgt', 'instdAgt']:
                    if entity in txn_data and isinstance(txn_data[entity], dict):
                        entity_data = txn_data[entity]
                        if 'before' in entity_data or 'after' in entity_data:
                            entities_with_changes += 1
                            entity_counts[entity] += 1
                
                if entities_with_changes > 0:
                    has_before_after += 1
                    
        except Exception as e:
            print(f"Error reading {json_file}: {e}")
    
    print(f"\nAnalyzed {total_transactions} transactions from sample")
    print(f"Transactions with before/after: {has_before_after} ({100*has_before_after/max(1,total_transactions):.1f}%)")
    
    print(f"\nTop 10 repair IDs:")
    for repair_id, count in repair_counts.most_common(10):
        print(f"  {repair_id}: {count} occurrences")
    
    print(f"\nEntity change distribution:")
    for entity, count in entity_counts.most_common():
        print(f"  {entity}: {count} changes")
    
    if has_before_after == 0:
        print("\n❌ CRITICAL: No before/after states found! Check data format.")
    elif has_before_after < total_transactions * 0.5:
        print("\n⚠️ WARNING: Less than 50% of transactions have before/after states")

def check_feature_importance(model_dir: str):
    """Check which features the Random Forest considers important"""
    print("\n" + "="*70)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*70)
    
    try:
        # Load RF model
        with open(Path(model_dir) / 'rf_model.pkl', 'rb') as f:
            rf_model = pickle.load(f)
        
        # Get feature importances for each repair
        print("\nChecking feature importance...")
        
        # For MultiOutputClassifier, each estimator has its own feature importance
        if hasattr(rf_model, 'estimators_'):
            avg_importance = np.zeros(rf_model.estimators_[0].feature_importances_.shape)
            
            for estimator in rf_model.estimators_:
                if hasattr(estimator, 'feature_importances_'):
                    avg_importance += estimator.feature_importances_
            
            avg_importance /= len(rf_model.estimators_)
            
            # Show top features
            top_features = np.argsort(avg_importance)[-10:][::-1]
            print("\nTop 10 most important feature indices:")
            for idx in top_features:
                print(f"  Feature {idx}: {avg_importance[idx]:.4f}")
            
            # Check if features are being used
            if np.max(avg_importance) < 0.01:
                print("\n❌ WARNING: Very low feature importance! Model may not be learning patterns.")
        else:
            print("❌ Model structure not recognized")
            
    except Exception as e:
        print(f"Error analyzing feature importance: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Diagnose model issues')
    parser.add_argument('--model_dir', default='./models', help='Model directory')
    parser.add_argument('--test_file', help='Test JSON file for feature extraction')
    parser.add_argument('--data_dir', help='Training data directory to analyze')
    
    args = parser.parse_args()
    
    # Run diagnostics
    diagnose_model(args.model_dir, args.test_file)
    
    if args.data_dir:
        analyze_training_data(args.data_dir)
    
    if args.model_dir:
        check_feature_importance(args.model_dir)
    
    print("\n" + "="*70)
    print("DIAGNOSTIC COMPLETE")
    print("="*70)
