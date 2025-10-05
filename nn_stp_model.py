"""
ace_repair_predictor.py
=======================
Production-ready ACE Payment Repair Predictor
Learns from diffs to predict which repairs should be applied

Author: Production Version 4.0
"""

import json
import argparse
import os
import pickle
import logging
import re
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
from pathlib import Path
import glob

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Model configuration"""
    hidden_dim: int = 256
    dropout: float = 0.3
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 50
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    model_dir: str = "./models"
    ml_threshold: float = 0.5
    use_class_weights: bool = True
    max_class_weight: float = 10.0
    
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: str):
        with open(path, 'r') as f:
            return cls(**json.load(f))


# ============================================================================
# DIFF ANALYZER
# ============================================================================

class DiffAnalyzer:
    """Analyzes diffs to learn repair patterns"""
    
    def __init__(self):
        self.repair_patterns = defaultdict(lambda: {
            'fields_added': Counter(),
            'fields_dropped': Counter(),
            'fields_transformed': Counter(),
            'typical_entities': Counter(),
            'total_diffs': 0
        })
        self.lookup_tables = {
            'bic_to_name': {},
            'bic_to_address': {},
            'country_extractions': Counter()
        }
    
    def analyze_transaction(self, txn_id: str, txn_data: Dict) -> Dict:
        """Analyze diffs in a transaction to learn patterns"""
        
        # Get repair IDs
        repair_ids = [r['id'] for r in txn_data.get('ace', [])]
        
        # Initialize diff features
        diff_features = {
            'num_additions': 0,
            'num_drops': 0,
            'num_transformations': 0,
            'affected_entities': set(),
            'affected_fields': set(),
            'has_bic_changes': False,
            'has_name_changes': False,
            'has_address_changes': False,
            'has_country_changes': False,
            'has_clearing_changes': False
        }
        
        # Process each entity that might have diffs
        entities_with_diffs = ['cdtrAgt', 'dbtrAgt', 'instgAgt', 'instdAgt',
                               'cdtr', 'dbtr', 'cdtrAcct', 'dbtrAcct', 
                               'intrmyAgt1', 'intrmyAgt2', 'rmtInf']
        
        for entity_key in entities_with_diffs:
            if entity_key not in txn_data:
                continue
            
            entity_data = txn_data[entity_key]
            
            # Check if this entity has diffs
            if not isinstance(entity_data, dict):
                continue
                
            if 'diffs' in entity_data and isinstance(entity_data['diffs'], list):
                diff_features['affected_entities'].add(entity_key.lower())
                
                # Also learn from before/after if present
                self._learn_from_before_after(entity_data, repair_ids, entity_key)
                
                # Process each diff
                for diff in entity_data['diffs']:
                    # Extract diff info
                    action = diff.get('msg', '').lower()
                    field_path = diff.get('key', '')
                    value = diff.get('val', '')
                    desc = diff.get('desc', '')
                    
                    # Update counts
                    if action == 'added':
                        diff_features['num_additions'] += 1
                    elif action == 'dropped':
                        diff_features['num_drops'] += 1
                    elif action in ['transformed', 'edited']:
                        diff_features['num_transformations'] += 1
                    
                    # Track affected fields
                    field_lower = field_path.lower()
                    diff_features['affected_fields'].add(field_lower)
                    
                    # Detect specific change types
                    if 'bic' in field_lower:
                        diff_features['has_bic_changes'] = True
                    if 'nm' in field_lower or 'name' in field_lower:
                        diff_features['has_name_changes'] = True
                    if 'adr' in field_lower or 'pstl' in field_lower:
                        diff_features['has_address_changes'] = True
                    if 'ctry' in field_lower or 'ctryofres' in field_lower:
                        diff_features['has_country_changes'] = True
                        # Learn country extractions
                        if value and len(value) == 2:
                            self.lookup_tables['country_extractions'][value] += 1
                    if 'clrsys' in field_lower or 'mmbid' in field_lower:
                        diff_features['has_clearing_changes'] = True
                    
                    # Learn patterns for each repair
                    for repair_id in repair_ids:
                        self.repair_patterns[repair_id]['total_diffs'] += 1
                        self.repair_patterns[repair_id]['typical_entities'][entity_key.lower()] += 1
                        
                        if action == 'added':
                            self.repair_patterns[repair_id]['fields_added'][field_lower] += 1
                        elif action == 'dropped':
                            self.repair_patterns[repair_id]['fields_dropped'][field_lower] += 1
                        elif action in ['transformed', 'edited']:
                            self.repair_patterns[repair_id]['fields_transformed'][field_lower] += 1
        
        return diff_features
    
    def _learn_from_before_after(self, entity_data: Dict, repair_ids: List[str], entity_key: str):
        """Learn lookup patterns from before/after states"""
        before = entity_data.get('before', {})
        after = entity_data.get('after', {})
        
        # Learn BIC to name/address mappings
        bic = self._find_value_in_dict(before, 'bicfi') or self._find_value_in_dict(before, 'bic')
        if not bic:
            bic = self._find_value_in_dict(after, 'bicfi') or self._find_value_in_dict(after, 'bic')
        
        if bic:
            # Look for name in after state
            name = self._find_value_in_dict(after, 'nm') or self._find_value_in_dict(after, 'name')
            if name and name != self._find_value_in_dict(before, 'nm'):
                self.lookup_tables['bic_to_name'][bic] = name
            
            # Look for address in after state
            addr = self._find_value_in_dict(after, 'adrline')
            if addr:
                self.lookup_tables['bic_to_address'][bic] = addr
    
    def _find_value_in_dict(self, d: Dict, key_pattern: str) -> Optional[Any]:
        """Recursively find value by key pattern in nested dict"""
        if not isinstance(d, dict):
            return None
        
        key_pattern = key_pattern.lower()
        for k, v in d.items():
            if key_pattern in str(k).lower():
                return v
            if isinstance(v, dict):
                result = self._find_value_in_dict(v, key_pattern)
                if result is not None:
                    return result
        return None
    
    def print_summary(self):
        """Print analysis summary"""
        logger.info("\n" + "="*70)
        logger.info("DIFF ANALYSIS SUMMARY")
        logger.info("="*70)
        
        # Repair patterns
        logger.info(f"\nAnalyzed {len(self.repair_patterns)} unique repairs:")
        for repair_id, pattern in list(self.repair_patterns.items())[:5]:
            logger.info(f"\n{repair_id} ({pattern['total_diffs']} diffs):")
            if pattern['typical_entities']:
                logger.info(f"  Entities: {pattern['typical_entities'].most_common(3)}")
            if pattern['fields_added']:
                logger.info(f"  Typically adds: {pattern['fields_added'].most_common(3)}")
            if pattern['fields_transformed']:
                logger.info(f"  Transforms: {pattern['fields_transformed'].most_common(3)}")
        
        # Lookup tables
        logger.info(f"\nLearned Lookups:")
        logger.info(f"  BIC to Name mappings: {len(self.lookup_tables['bic_to_name'])}")
        logger.info(f"  BIC to Address mappings: {len(self.lookup_tables['bic_to_address'])}")
        logger.info(f"  Country codes seen: {len(self.lookup_tables['country_extractions'])}")


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

class FeatureExtractor:
    """Extracts features from payment structure and diffs"""
    
    def __init__(self):
        self.feature_names = []
        self._build_feature_names()
    
    def _build_feature_names(self):
        """Define feature names"""
        # Structural features
        self.feature_names.extend([
            'has_bic', 'has_iban', 'has_clearing_id', 'has_bank_name',
            'has_address', 'has_country', 'has_cdtrAgt', 'has_dbtrAgt',
            'has_instgAgt', 'has_instdAgt', 'has_cdtr', 'has_dbtr',
            'has_rmtInf', 'missing_bic_has_clearing', 'missing_name_has_bic',
            'missing_country_has_bic', 'entity_count', 'field_count'
        ])
        
        # Diff-based features
        self.feature_names.extend([
            'diff_num_additions', 'diff_num_drops', 'diff_num_transformations',
            'diff_affected_entities', 'diff_affected_fields',
            'diff_has_bic_changes', 'diff_has_name_changes', 
            'diff_has_address_changes', 'diff_has_country_changes',
            'diff_has_clearing_changes', 'diff_affects_cdtragt',
            'diff_affects_dbtragt', 'diff_affects_instgagt'
        ])
    
    def extract_features(self, payment: Dict, diff_features: Optional[Dict] = None) -> np.ndarray:
        """Extract all features from payment and diffs"""
        features = np.zeros(len(self.feature_names))
        idx = 0
        
        # Normalize payment
        payment = self._normalize_keys(payment)
        
        # Structural features
        features[idx] = float(self._has_field(payment, 'bic')); idx += 1
        features[idx] = float(self._has_field(payment, 'iban')); idx += 1
        features[idx] = float(self._has_field(payment, 'mmbid')); idx += 1
        features[idx] = float(self._has_field(payment, 'nm')); idx += 1
        features[idx] = float(self._has_field(payment, 'adrline')); idx += 1
        features[idx] = float(self._has_field(payment, 'ctryofres')); idx += 1
        features[idx] = float('cdtragt' in payment); idx += 1
        features[idx] = float('dbtragt' in payment); idx += 1
        features[idx] = float('instgagt' in payment); idx += 1
        features[idx] = float('instdagt' in payment); idx += 1
        features[idx] = float('cdtr' in payment); idx += 1
        features[idx] = float('dbtr' in payment); idx += 1
        features[idx] = float('rmtinf' in payment); idx += 1
        
        has_bic = self._has_field(payment, 'bic')
        has_clearing = self._has_field(payment, 'mmbid')
        has_name = self._has_field(payment, 'nm')
        has_country = self._has_field(payment, 'ctryofres')
        
        features[idx] = float(not has_bic and has_clearing); idx += 1
        features[idx] = float(not has_name and has_bic); idx += 1
        features[idx] = float(not has_country and has_bic); idx += 1
        
        entity_count = sum(1 for k in payment.keys() if isinstance(payment[k], dict))
        features[idx] = min(entity_count / 10, 1.0); idx += 1
        
        field_count = self._count_fields(payment)
        features[idx] = min(field_count / 50, 1.0); idx += 1
        
        # Diff-based features
        if diff_features:
            features[idx] = min(diff_features.get('num_additions', 0) / 10, 1.0); idx += 1
            features[idx] = min(diff_features.get('num_drops', 0) / 10, 1.0); idx += 1
            features[idx] = min(diff_features.get('num_transformations', 0) / 10, 1.0); idx += 1
            
            affected_entities = diff_features.get('affected_entities', set())
            features[idx] = min(len(affected_entities) / 5, 1.0); idx += 1
            
            affected_fields = diff_features.get('affected_fields', set())
            features[idx] = min(len(affected_fields) / 20, 1.0); idx += 1
            
            features[idx] = float(diff_features.get('has_bic_changes', False)); idx += 1
            features[idx] = float(diff_features.get('has_name_changes', False)); idx += 1
            features[idx] = float(diff_features.get('has_address_changes', False)); idx += 1
            features[idx] = float(diff_features.get('has_country_changes', False)); idx += 1
            features[idx] = float(diff_features.get('has_clearing_changes', False)); idx += 1
            features[idx] = float('cdtragt' in affected_entities); idx += 1
            features[idx] = float('dbtragt' in affected_entities); idx += 1
            features[idx] = float('instgagt' in affected_entities); idx += 1
        else:
            # Fill with zeros if no diff features
            for _ in range(13):
                features[idx] = 0.0; idx += 1
        
        return features
    
    def _normalize_keys(self, obj):
        """Normalize all keys to lowercase"""
        if isinstance(obj, dict):
            return {k.lower() if isinstance(k, str) else k: self._normalize_keys(v)
                   for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._normalize_keys(item) for item in obj]
        return obj
    
    def _has_field(self, payment: Dict, field_name: str) -> bool:
        """Check if field exists anywhere in payment"""
        field_name = field_name.lower()
        
        def search(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if field_name in str(k).lower():
                        return True
                    if search(v):
                        return True
            elif isinstance(obj, list):
                for item in obj:
                    if search(item):
                        return True
            return False
        
        return search(payment)
    
    def _count_fields(self, obj, depth=0) -> int:
        """Count total fields in nested structure"""
        if depth > 10:  # Prevent infinite recursion
            return 0
        
        count = 0
        if isinstance(obj, dict):
            count += len(obj)
            for v in obj.values():
                count += self._count_fields(v, depth + 1)
        elif isinstance(obj, list):
            for item in obj:
                count += self._count_fields(item, depth + 1)
        return count


# ============================================================================
# DATA PROCESSOR
# ============================================================================

class DataProcessor:
    """Process training data with diff analysis"""
    
    def __init__(self):
        self.diff_analyzer = DiffAnalyzer()
        self.feature_extractor = FeatureExtractor()
        self.repair_vocabulary = {}
        self.idx_to_repair = {}
        self.diff_features_cache = {}
    
    def load_and_process(self, path: str) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """Load and process training data from file"""
        logger.info(f"Loading data from {path}")
        
        # Read the file
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Handle array wrapper format
        all_transactions = {}
        
        if isinstance(data, list):
            # File is an array of transaction objects
            for item in data:
                if isinstance(item, dict):
                    all_transactions.update(item)
        elif isinstance(data, dict):
            # File is a direct object
            all_transactions = data
        else:
            raise ValueError(f"Unexpected data format in {path}")
        
        logger.info(f"Loaded {len(all_transactions)} transactions")
        
        # Process transactions
        processed_count = 0
        skipped_count = 0
        
        # First pass: Analyze all diffs
        logger.info("Analyzing diffs...")
        for txn_id, txn_data in all_transactions.items():
            if not isinstance(txn_data, dict):
                continue
            
            # Must have repairs to be valid training data
            if 'ace' not in txn_data or not txn_data['ace']:
                skipped_count += 1
                continue
            
            # Analyze diffs
            diff_features = self.diff_analyzer.analyze_transaction(txn_id, txn_data)
            self.diff_features_cache[txn_id] = diff_features
            processed_count += 1
            
            if processed_count % 1000 == 0:
                logger.info(f"  Processed {processed_count} transactions...")
        
        # Print diff analysis
        self.diff_analyzer.print_summary()
        
        # Build repair vocabulary
        self._build_repair_vocabulary(all_transactions)
        
        # Second pass: Extract features
        logger.info("Extracting features...")
        all_features = []
        all_labels = []
        all_payments = []
        
        for txn_id, txn_data in all_transactions.items():
            if txn_id not in self.diff_features_cache:
                continue
            
            try:
                # Get repairs
                repairs = [r['id'] for r in txn_data.get('ace', [])]
                
                # Extract before state for features
                payment = self._extract_before_state(txn_data)
                
                # Get cached diff features
                diff_features = self.diff_features_cache[txn_id]
                
                # Extract combined features
                features = self.feature_extractor.extract_features(payment, diff_features)
                
                # Create labels
                labels = self._repairs_to_labels(repairs)
                
                all_features.append(features)
                all_labels.append(labels)
                all_payments.append(txn_data)
                
            except Exception as e:
                logger.warning(f"Error processing {txn_id}: {e}")
                continue
        
        features_array = np.array(all_features)
        labels_array = np.array(all_labels)
        
        logger.info(f"\nProcessing complete:")
        logger.info(f"  Total transactions: {len(all_transactions)}")
        logger.info(f"  Trainable transactions: {len(features_array)}")
        logger.info(f"  Skipped (no repairs): {skipped_count}")
        logger.info(f"  Feature dimensions: {features_array.shape}")
        logger.info(f"  Label dimensions: {labels_array.shape}")
        
        return features_array, labels_array, all_payments
    
    def _extract_before_state(self, txn_data: Dict) -> Dict:
        """Extract 'before' state from transaction"""
        payment = {}
        
        # Copy metadata
        for key in ['source', 'clearing', 'flags', 'parties']:
            if key in txn_data:
                payment[key] = txn_data[key]
        
        # Extract before states from entities
        entities = ['cdtrAgt', 'dbtrAgt', 'instgAgt', 'instdAgt',
                   'cdtr', 'dbtr', 'cdtrAcct', 'dbtrAcct', 'rmtInf']
        
        for entity in entities:
            if entity in txn_data:
                entity_data = txn_data[entity]
                if isinstance(entity_data, dict) and 'before' in entity_data:
                    payment[entity] = entity_data['before']
                elif not isinstance(entity_data, dict) or 'after' not in entity_data:
                    # If no before/after structure, use as is
                    payment[entity] = entity_data
        
        return payment
    
    def _build_repair_vocabulary(self, data: Dict):
        """Build repair vocabulary from all transactions"""
        all_repairs = set()
        
        for txn_data in data.values():
            if not isinstance(txn_data, dict):
                continue
            for repair in txn_data.get('ace', []):
                all_repairs.add(repair['id'])
        
        sorted_repairs = sorted(all_repairs)
        self.repair_vocabulary = {repair_id: idx for idx, repair_id in enumerate(sorted_repairs)}
        self.idx_to_repair = {idx: repair_id for repair_id, idx in self.repair_vocabulary.items()}
        
        logger.info(f"Built vocabulary with {len(self.repair_vocabulary)} unique repairs")
        logger.info(f"Repairs: {sorted_repairs}")
    
    def _repairs_to_labels(self, repairs: List[str]) -> np.ndarray:
        """Convert repair IDs to multi-hot encoding"""
        labels = np.zeros(len(self.repair_vocabulary))
        for repair_id in repairs:
            if repair_id in self.repair_vocabulary:
                labels[self.repair_vocabulary[repair_id]] = 1.0
        return labels


# ============================================================================
# NEURAL NETWORK
# ============================================================================

class RepairPredictor(nn.Module):
    """Neural network for repair prediction"""
    
    def __init__(self, num_features: int, num_repairs: int, 
                 hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, num_repairs),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)


class RepairDataset(Dataset):
    """PyTorch dataset for repairs"""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ============================================================================
# MODEL TRAINER
# ============================================================================

class ModelTrainer:
    """Handles model training and evaluation"""
    
    def __init__(self, config: Config):
        self.config = config
        self.processor = DataProcessor()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.rf_model = None
    
    def train(self, data_path: str) -> Dict:
        """Train the model"""
        logger.info("="*70)
        logger.info("TRAINING ACE REPAIR PREDICTOR")
        logger.info("="*70)
        
        # Load and process data
        features, labels, _ = self.processor.load_and_process(data_path)
        
        # Split data
        n = len(features)
        indices = np.random.permutation(n)
        
        train_size = int(n * self.config.train_split)
        val_size = int(n * self.config.val_split)
        
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]
        test_idx = indices[train_size + val_size:]
        
        X_train, y_train = features[train_idx], labels[train_idx]
        X_val, y_val = features[val_idx], labels[val_idx]
        X_test, y_test = features[test_idx], labels[test_idx]
        
        logger.info(f"\nData split:")
        logger.info(f"  Train: {len(X_train)}")
        logger.info(f"  Validation: {len(X_val)}")
        logger.info(f"  Test: {len(X_test)}")
        
        # Train Random Forest
        logger.info("\nTraining Random Forest...")
        self.rf_model = MultiOutputClassifier(
            RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
        )
        self.rf_model.fit(X_train, y_train)
        logger.info("Random Forest training complete")
        
        # Train Neural Network
        logger.info("\nTraining Neural Network...")
        self.model = RepairPredictor(
            num_features=features.shape[1],
            num_repairs=labels.shape[1],
            hidden_dim=self.config.hidden_dim,
            dropout=self.config.dropout
        ).to(self.device)
        
        self._train_nn(X_train, y_train, X_val, y_val)
        
        # Evaluate on test set
        logger.info("\nEvaluating on test set...")
        metrics = self._evaluate(X_test, y_test)
        
        # Save models
        self._save_models()
        
        return metrics
    
    def _train_nn(self, X_train, y_train, X_val, y_val):
        """Train neural network"""
        train_dataset = RepairDataset(X_train, y_train)
        val_dataset = RepairDataset(X_val, y_val)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.batch_size
        )
        
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.config.learning_rate
        )
        criterion = nn.BCELoss()
        
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(1, self.config.num_epochs + 1):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for features, labels in train_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for features, labels in val_loader:
                    features = features.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(features)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            # Logging
            if epoch % 5 == 0:
                logger.info(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss - 0.001:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 
                          os.path.join(self.config.model_dir, 'best_model.pt'))
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        # Load best model
        self.model.load_state_dict(
            torch.load(os.path.join(self.config.model_dir, 'best_model.pt'))
        )
    
    def _evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        self.model.eval()
        
        dataset = RepairDataset(X_test, y_test)
        loader = DataLoader(dataset, batch_size=self.config.batch_size)
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for features, labels in loader:
                features = features.to(self.device)
                outputs = self.model(features)
                all_preds.append(outputs.cpu().numpy())
                all_labels.append(labels.numpy())
        
        preds = np.vstack(all_preds)
        labels = np.vstack(all_labels)
        preds_binary = (preds > 0.5).astype(int)
        
        # Calculate metrics
        exact_match = np.mean(np.all(preds_binary == labels, axis=1))
        hamming = np.mean(preds_binary == labels)
        
        # Per-repair accuracy
        per_repair_acc = []
        for i in range(labels.shape[1]):
            acc = np.mean(preds_binary[:, i] == labels[:, i])
            per_repair_acc.append(acc)
            repair_id = self.processor.idx_to_repair.get(i, f"repair_{i}")
            logger.info(f"  {repair_id}: {acc:.2%}")
        
        metrics = {
            'exact_match_accuracy': exact_match,
            'hamming_accuracy': hamming,
            'mean_repair_accuracy': np.mean(per_repair_acc)
        }
        
        logger.info(f"\nOverall Metrics:")
        logger.info(f"  Exact Match: {exact_match:.2%}")
        logger.info(f"  Hamming Accuracy: {hamming:.2%}")
        logger.info(f"  Mean Repair Accuracy: {np.mean(per_repair_acc):.2%}")
        
        return metrics
    
    def _save_models(self):
        """Save trained models"""
        os.makedirs(self.config.model_dir, exist_ok=True)
        
        # Save config
        self.config.save(os.path.join(self.config.model_dir, 'config.json'))
        
        # Save processor
        with open(os.path.join(self.config.model_dir, 'processor.pkl'), 'wb') as f:
            pickle.dump(self.processor, f)
        
        # Save Random Forest
        with open(os.path.join(self.config.model_dir, 'rf_model.pkl'), 'wb') as f:
            pickle.dump(self.rf_model, f)
        
        # Save Neural Network
        torch.save(self.model.state_dict(), 
                  os.path.join(self.config.model_dir, 'neural_model.pt'))
        
        logger.info(f"\nModels saved to {self.config.model_dir}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='ACE Payment Repair Predictor'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--input', required=True, help='Training data file')
    train_parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    train_parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict repairs')
    predict_parser.add_argument('--input', required=True, help='Input file')
    predict_parser.add_argument('--model', default='./models', help='Model directory')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        config = Config()
        if args.epochs:
            config.num_epochs = args.epochs
        if args.batch_size:
            config.batch_size = args.batch_size
        
        trainer = ModelTrainer(config)
        trainer.train(args.input)
    
    elif args.command == 'predict':
        # Prediction logic would go here
        logger.info("Prediction not yet implemented in this version")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
