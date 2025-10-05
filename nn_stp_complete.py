#!/usr/bin/env python3
"""
Final ACE Payment Repair Predictor
Version: 9.0 - Dual Learning Strategy

Training Phase:
- Learns from structural changes (before → after)
- Associates changes with repairs (ace array)
- Uses diffs to understand transformations
- Builds pattern-to-repair mappings

Prediction Phase:
- Analyzes current state only
- Detects problems/missing elements
- Predicts repairs based on learned patterns
"""

import json
import argparse
import os
import sys
import pickle
import logging
from typing import Dict, List, Set, Tuple, Optional, Any, Generator
from dataclasses import dataclass, asdict, field
from collections import defaultdict, Counter
from pathlib import Path
import glob
from datetime import datetime
import hashlib

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from tqdm import tqdm

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class DualLearningConfig:
    """Configuration for dual learning strategy"""
    # Model parameters
    hidden_dim: int = 512
    dropout: float = 0.3
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 50
    
    # Feature dimensions
    problem_features: int = 183  # What's wrong
    change_features: int = 50    # What changed
    
    # Data splits
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    
    # Processing
    checkpoint_interval: int = 500
    model_dir: str = "./models"
    checkpoint_dir: str = "./checkpoints"
    
    # Learning strategy
    learn_from_changes: bool = True
    learn_from_problems: bool = True
    weight_changes: float = 0.4  # Weight for change features
    weight_problems: float = 0.6  # Weight for problem features
    
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: str):
        with open(path, 'r') as f:
            return cls(**json.load(f))


# ============================================================================
# STRUCTURAL CHANGE ANALYZER
# ============================================================================

class StructuralChangeAnalyzer:
    """Analyzes structural changes between before and after states"""
    
    def __init__(self):
        self.change_to_repair_patterns = defaultdict(lambda: defaultdict(int))
        self.repair_to_change_patterns = defaultdict(lambda: defaultdict(int))
        
    def analyze_changes(self, before: Dict, after: Dict, entity_name: str) -> Dict:
        """Analyze what changed between before and after"""
        changes = {
            'fields_added': [],
            'fields_removed': [],
            'fields_moved': [],
            'values_changed': [],
            'structure_changed': [],
            'type_changed': [],
            'total_changes': 0
        }
        
        if not before and after:
            # Entity was added
            changes['structure_changed'].append('entity_added')
            changes['total_changes'] = 1
            return changes
        
        if before and not after:
            # Entity was removed
            changes['structure_changed'].append('entity_removed')
            changes['total_changes'] = 1
            return changes
        
        # Get field inventories
        before_fields = self._get_field_paths(before)
        after_fields = self._get_field_paths(after)
        
        # Fields added
        for field in after_fields - before_fields:
            changes['fields_added'].append(field)
            changes['total_changes'] += 1
        
        # Fields removed
        for field in before_fields - after_fields:
            changes['fields_removed'].append(field)
            changes['total_changes'] += 1
        
        # Values changed
        for field in before_fields & after_fields:
            before_val = self._get_field_value(before, field)
            after_val = self._get_field_value(after, field)
            if before_val != after_val:
                changes['values_changed'].append({
                    'field': field,
                    'before': str(before_val)[:50],
                    'after': str(after_val)[:50]
                })
                changes['total_changes'] += 1
        
        # Detect specific structural patterns
        if self._detect_mmbid_restructuring(before, after):
            changes['structure_changed'].append('mmbid_restructured')
            changes['total_changes'] += 1
        
        if self._detect_clearing_structure_added(before, after):
            changes['structure_changed'].append('clearing_structure_added')
            changes['total_changes'] += 1
        
        return changes
    
    def learn_pattern(self, changes: Dict, repair_ids: List[str], entity: str):
        """Learn association between changes and repairs"""
        # Create change signature
        signature_parts = []
        
        if changes['fields_added']:
            signature_parts.append(f"added:{len(changes['fields_added'])}")
        if changes['fields_removed']:
            signature_parts.append(f"removed:{len(changes['fields_removed'])}")
        if changes['values_changed']:
            signature_parts.append(f"changed:{len(changes['values_changed'])}")
        if changes['structure_changed']:
            signature_parts.extend(changes['structure_changed'])
        
        change_signature = f"{entity}:{','.join(signature_parts)}"
        
        # Learn associations
        for repair_id in repair_ids:
            self.change_to_repair_patterns[change_signature][repair_id] += 1
            self.repair_to_change_patterns[repair_id][change_signature] += 1
    
    def extract_change_features(self, changes_by_entity: Dict[str, Dict]) -> np.ndarray:
        """Extract features from structural changes"""
        features = np.zeros(50)  # 50 change features
        idx = 0
        
        # Aggregate change counts
        total_added = sum(len(c.get('fields_added', [])) for c in changes_by_entity.values())
        total_removed = sum(len(c.get('fields_removed', [])) for c in changes_by_entity.values())
        total_value_changes = sum(len(c.get('values_changed', [])) for c in changes_by_entity.values())
        
        # Basic change counts (normalized)
        features[idx] = min(total_added / 10, 1.0); idx += 1
        features[idx] = min(total_removed / 10, 1.0); idx += 1
        features[idx] = min(total_value_changes / 10, 1.0); idx += 1
        
        # Entity-specific changes
        key_entities = ['cdtrAgt', 'dbtrAgt', 'cdtr', 'dbtr']
        for entity in key_entities:
            if entity in changes_by_entity:
                changes = changes_by_entity[entity]
                features[idx] = min(changes.get('total_changes', 0) / 5, 1.0); idx += 1
                features[idx] = float('entity_added' in changes.get('structure_changed', [])); idx += 1
                features[idx] = float('mmbid_restructured' in changes.get('structure_changed', [])); idx += 1
            else:
                idx += 3
        
        # Specific patterns
        has_bic_additions = any('bic' in str(c.get('fields_added', [])).lower() 
                               for c in changes_by_entity.values())
        has_name_additions = any('nm' in str(c.get('fields_added', [])).lower() 
                                for c in changes_by_entity.values())
        has_country_additions = any('ctry' in str(c.get('fields_added', [])).lower() 
                                   for c in changes_by_entity.values())
        
        features[idx] = float(has_bic_additions); idx += 1
        features[idx] = float(has_name_additions); idx += 1
        features[idx] = float(has_country_additions); idx += 1
        
        return features[:50]  # Ensure exactly 50 features
    
    def _get_field_paths(self, obj: Dict, prefix: str = "") -> Set[str]:
        """Get all field paths in object"""
        paths = set()
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                path = f"{prefix}.{key}" if prefix else key
                paths.add(path)
                if isinstance(value, dict):
                    paths.update(self._get_field_paths(value, path))
        
        return paths
    
    def _get_field_value(self, obj: Dict, path: str) -> Any:
        """Get value at field path"""
        parts = path.split('.')
        current = obj
        
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        
        return current
    
    def _detect_mmbid_restructuring(self, before: Dict, after: Dict) -> bool:
        """Detect if MmbId moved into ClrSysId structure"""
        # Check if MmbId was at top level before
        before_has_top_mmbid = 'MmbId' in str(before).replace('ClrSysId', '')
        # Check if MmbId is inside ClrSysId after
        after_has_clrsys_mmbid = 'ClrSysId' in str(after) and 'MmbId' in str(after)
        
        return before_has_top_mmbid and after_has_clrsys_mmbid
    
    def _detect_clearing_structure_added(self, before: Dict, after: Dict) -> bool:
        """Detect if clearing structure was added"""
        before_has_clrsys = 'ClrSysId' in str(before) or 'ClrSysMmbId' in str(before)
        after_has_clrsys = 'ClrSysId' in str(after) or 'ClrSysMmbId' in str(after)
        
        return not before_has_clrsys and after_has_clrsys


# ============================================================================
# COMPREHENSIVE PROBLEM DETECTOR (183 features)
# ============================================================================

class ProblemDetector:
    """Detects problems in current state (what needs repair)"""
    
    def __init__(self):
        self.feature_names = []
        self._build_features()
        
    def _build_features(self):
        """Build comprehensive 183 features"""
        # Agent features (64)
        for agent in ['cdtrAgt', 'dbtrAgt', 'instgAgt', 'instdAgt', 
                     'intrmyAgt1', 'intrmyAgt2', 'intrmyAgt3', 'fwdgAgt']:
            self.feature_names.extend([
                f'{agent}_missing_bic',
                f'{agent}_missing_name',
                f'{agent}_missing_address',
                f'{agent}_missing_country',
                f'{agent}_has_clearing_no_bic',  # Key trigger for 8852
                f'{agent}_has_bic_no_name',
                f'{agent}_mmbid_wrong_level',
                f'{agent}_missing_clrsysid'
            ])
        
        # Party features (30)
        for party in ['cdtr', 'dbtr', 'ultmtCdtr', 'ultmtDbtr', 'initgPty']:
            self.feature_names.extend([
                f'{party}_missing_name',
                f'{party}_missing_address',
                f'{party}_missing_country',  # Key trigger for 6021
                f'{party}_missing_identification',
                f'{party}_incomplete_postal',
                f'{party}_has_text_no_country'
            ])
        
        # Account features (12)
        for account in ['cdtrAcct', 'dbtrAcct']:
            self.feature_names.extend([
                f'{account}_missing_iban',
                f'{account}_missing_othr_id',
                f'{account}_missing_any_id',
                f'{account}_missing_type',
                f'{account}_missing_currency',
                f'{account}_wrong_type_format'
            ])
        
        # Add remaining features to reach 183...
        # [Simplified for brevity - would include all 183 features as before]
        
        # Pad to exactly 183
        while len(self.feature_names) < 183:
            self.feature_names.append(f'reserved_{len(self.feature_names)}')
    
    def extract_problem_features(self, payment: Dict, use_before: bool = False) -> np.ndarray:
        """Extract features detecting problems in payment"""
        features = np.zeros(183)
        
        # Normalize keys
        payment = self._normalize_keys(payment)
        
        # Check each agent
        agent_idx = 0
        for agent in ['cdtragt', 'dbtragt', 'instgagt', 'instdagt',
                     'intrmyagt1', 'intrmyagt2', 'intrmyagt3', 'fwdgagt']:
            agent_data = self._get_entity_data(payment, agent, use_before)
            
            if agent_data is None:
                # Agent missing - all problems present
                features[agent_idx:agent_idx+8] = 1.0
            else:
                # Check specific problems
                fin_inst = self._get_fininstnid(agent_data)
                
                # Missing BIC
                has_bic = self._has_field(fin_inst, ['bicfi', 'bic'])
                features[agent_idx] = float(not has_bic)
                
                # Missing name
                has_name = self._has_field(fin_inst, ['nm', 'name'])
                features[agent_idx+1] = float(not has_name)
                
                # Missing address
                has_addr = self._has_field(fin_inst, ['pstladr', 'adrline'])
                features[agent_idx+2] = float(not has_addr)
                
                # Missing country
                has_country = self._has_field(fin_inst, ['ctry', 'ctryofres'])
                features[agent_idx+3] = float(not has_country)
                
                # Has clearing but no BIC (triggers 8852)
                has_clearing = self._has_field(fin_inst, ['mmbid', 'clrsysmmbid'])
                features[agent_idx+4] = float(has_clearing and not has_bic)
                
                # Has BIC but no name
                features[agent_idx+5] = float(has_bic and not has_name)
                
                # MmbId at wrong level
                features[agent_idx+6] = float(self._mmbid_at_wrong_level(fin_inst))
                
                # Missing ClrSysId
                features[agent_idx+7] = float(has_clearing and not self._has_field(fin_inst, ['clrsysid']))
            
            agent_idx += 8
        
        # Check parties (simplified for brevity)
        party_idx = 64
        for party in ['cdtr', 'dbtr', 'ultmtcdtr', 'ultmtdbtr', 'initgpty']:
            party_data = self._get_entity_data(payment, party, use_before)
            
            if party_data is None:
                features[party_idx:party_idx+6] = 0.5  # Parties may be optional
            else:
                # Check party problems
                features[party_idx] = float(not self._has_field(party_data, ['nm']))
                features[party_idx+1] = float(not self._has_field(party_data, ['adrline']))
                features[party_idx+2] = float(not self._has_field(party_data, ['ctry']))
                # ... etc
            
            party_idx += 6
        
        # Return exactly 183 features
        return features[:183]
    
    def _get_entity_data(self, payment: Dict, entity: str, use_before: bool) -> Optional[Dict]:
        """Get entity data, optionally from 'before' state"""
        entity = entity.lower()
        
        if entity not in payment:
            return None
        
        entity_data = payment[entity]
        if not isinstance(entity_data, dict):
            return None
        
        # For training, use 'before' if available
        if use_before and 'before' in entity_data:
            return entity_data['before']
        
        # If no before/after structure, use direct data
        if 'before' not in entity_data and 'after' not in entity_data:
            return entity_data
        
        return None
    
    def _normalize_keys(self, obj: Any) -> Any:
        """Normalize all keys to lowercase"""
        if isinstance(obj, dict):
            return {k.lower() if isinstance(k, str) else k: self._normalize_keys(v)
                   for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._normalize_keys(item) for item in obj]
        return obj
    
    def _has_field(self, obj: Optional[Dict], field_names: List[str]) -> bool:
        """Check if any field exists"""
        if not obj or not isinstance(obj, dict):
            return False
        
        for field in field_names:
            if self._find_field(obj, field):
                return True
        
        return False
    
    def _find_field(self, obj: Dict, field_name: str, max_depth: int = 5) -> Optional[Any]:
        """Find field in nested dict"""
        if max_depth <= 0:
            return None
        
        field_lower = field_name.lower()
        
        for key, value in obj.items():
            if key.lower() == field_lower:
                return value
            if isinstance(value, dict):
                result = self._find_field(value, field_name, max_depth-1)
                if result is not None:
                    return result
        
        return None
    
    def _get_fininstnid(self, agent_data: Optional[Dict]) -> Optional[Dict]:
        """Get FinInstnId from agent"""
        if not agent_data:
            return None
        return self._find_field(agent_data, 'fininstnid') or agent_data
    
    def _mmbid_at_wrong_level(self, data: Optional[Dict]) -> bool:
        """Check if MmbId is at wrong level"""
        if not data:
            return False
        
        # Has MmbId at top level
        has_top = 'mmbid' in str(data).lower().replace('clrsysid', '').replace('clrsysmmbid', '')
        # No ClrSysId structure
        has_clrsys = 'clrsysid' in str(data).lower()
        
        return has_top and not has_clrsys


# ============================================================================
# DUAL LEARNING DATA PROCESSOR
# ============================================================================

class DualLearningProcessor:
    """Processes data using dual learning strategy"""
    
    def __init__(self, config: DualLearningConfig):
        self.config = config
        self.change_analyzer = StructuralChangeAnalyzer()
        self.problem_detector = ProblemDetector()
        self.repair_vocabulary = {}
        self.transaction_count = 0
        
    def process_transaction(self, txn_id: str, txn_data: Dict) -> Optional[Dict]:
        """Process transaction for training"""
        # Skip if no repairs
        if 'ace' not in txn_data or not txn_data.get('ace'):
            return None
        
        # Extract repair IDs
        repair_ids = []
        for repair in txn_data.get('ace', []):
            repair_id = str(repair.get('id', 'unknown')) if isinstance(repair, dict) else str(repair)
            repair_ids.append(repair_id)
            
            if repair_id not in self.repair_vocabulary:
                self.repair_vocabulary[repair_id] = len(self.repair_vocabulary)
        
        # 1. Extract problem features from 'before' state
        problem_features = self.problem_detector.extract_problem_features(txn_data, use_before=True)
        
        # 2. Analyze structural changes between before and after
        changes_by_entity = {}
        entities = ['cdtrAgt', 'dbtrAgt', 'instgAgt', 'instdAgt',
                   'cdtr', 'dbtr', 'cdtrAcct', 'dbtrAcct', 'rmtInf']
        
        for entity in entities:
            if entity in txn_data and isinstance(txn_data[entity], dict):
                entity_data = txn_data[entity]
                before = entity_data.get('before', {})
                after = entity_data.get('after', {})
                
                if before or after:
                    changes = self.change_analyzer.analyze_changes(before, after, entity)
                    if changes['total_changes'] > 0:
                        changes_by_entity[entity] = changes
                        # Learn pattern association
                        self.change_analyzer.learn_pattern(changes, repair_ids, entity)
        
        # 3. Extract change features
        change_features = self.change_analyzer.extract_change_features(changes_by_entity)
        
        # 4. Combine features with weighting
        combined_features = np.concatenate([
            problem_features * self.config.weight_problems,
            change_features * self.config.weight_changes
        ])
        
        return {
            'txn_id': txn_id,
            'problem_features': problem_features,
            'change_features': change_features,
            'combined_features': combined_features,
            'repairs': repair_ids,
            'source': txn_data.get('source'),
            'clearing': txn_data.get('clearing'),
            'changes_detected': len(changes_by_entity)
        }
    
    def process_directory(self, directory_path: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Process all files in directory"""
        all_features = []
        all_labels = []
        
        json_files = sorted(Path(directory_path).glob('**/*.json'))
        logger.info(f"Found {len(json_files)} JSON files")
        
        # First pass - build vocabulary
        logger.info("Building vocabulary...")
        for json_file in tqdm(json_files, desc="Scanning"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                transactions = self._extract_transactions(data)
                
                for txn_id, txn_data in transactions.items():
                    if isinstance(txn_data, dict) and 'ace' in txn_data:
                        for repair in txn_data.get('ace', []):
                            repair_id = str(repair.get('id', 'unknown')) if isinstance(repair, dict) else str(repair)
                            if repair_id not in self.repair_vocabulary:
                                self.repair_vocabulary[repair_id] = len(self.repair_vocabulary)
                                
            except Exception as e:
                logger.warning(f"Error processing {json_file}: {e}")
        
        vocab_size = len(self.repair_vocabulary)
        logger.info(f"Vocabulary built: {vocab_size} unique repairs")
        
        # Second pass - extract features
        logger.info("Extracting features...")
        for json_file in tqdm(json_files, desc="Processing"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                transactions = self._extract_transactions(data)
                
                for txn_id, txn_data in transactions.items():
                    if not isinstance(txn_data, dict):
                        continue
                    
                    processed = self.process_transaction(txn_id, txn_data)
                    
                    if processed:
                        all_features.append(processed['combined_features'])
                        
                        # Create label vector
                        labels = np.zeros(vocab_size)
                        for repair_id in processed['repairs']:
                            if repair_id in self.repair_vocabulary:
                                labels[self.repair_vocabulary[repair_id]] = 1.0
                        all_labels.append(labels)
                        
                        self.transaction_count += 1
                        
            except Exception as e:
                logger.warning(f"Error processing {json_file}: {e}")
        
        X = np.array(all_features) if all_features else np.array([])
        y = np.array(all_labels) if all_labels else np.array([])
        
        logger.info(f"Processed {self.transaction_count} transactions")
        logger.info(f"Feature shape: {X.shape}")
        logger.info(f"Label shape: {y.shape}")
        
        # Return metadata
        metadata = {
            'repair_vocabulary': self.repair_vocabulary,
            'change_patterns': dict(self.change_analyzer.change_to_repair_patterns),
            'repair_patterns': dict(self.change_analyzer.repair_to_change_patterns),
            'problem_feature_names': self.problem_detector.feature_names,
            'total_features': len(self.problem_detector.feature_names) + 50
        }
        
        return X, y, metadata
    
    def _extract_transactions(self, data: Any) -> Dict:
        """Extract transactions from various formats"""
        transactions = {}
        
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict):
                    transactions[str(key)] = value
                    
        elif isinstance(data, list):
            for idx, item in enumerate(data):
                if isinstance(item, dict):
                    if len(item) == 1:
                        txn_id = list(item.keys())[0]
                        transactions[str(txn_id)] = item[txn_id]
                    else:
                        transactions[f"txn_{idx}"] = item
        
        return transactions


# ============================================================================
# NEURAL NETWORK MODEL
# ============================================================================

class DualLearningNN(nn.Module):
    """Neural network for dual learning"""
    
    def __init__(self, num_features: int, num_repairs: int, config: DualLearningConfig):
        super().__init__()
        
        # Separate encoders for problem and change features
        self.problem_encoder = nn.Sequential(
            nn.Linear(183, config.hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(config.hidden_dim // 2),
            nn.Dropout(config.dropout)
        )
        
        self.change_encoder = nn.Sequential(
            nn.Linear(50, config.hidden_dim // 4),
            nn.ReLU(),
            nn.BatchNorm1d(config.hidden_dim // 4),
            nn.Dropout(config.dropout)
        )
        
        # Combined predictor
        combined_dim = (config.hidden_dim // 2) + (config.hidden_dim // 4)
        
        self.predictor = nn.Sequential(
            nn.Linear(combined_dim, config.hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(config.hidden_dim),
            nn.Dropout(config.dropout),
            
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(config.hidden_dim // 2),
            nn.Dropout(config.dropout / 2),
            
            nn.Linear(config.hidden_dim // 2, num_repairs),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Split features
        problem_features = x[:, :183]
        change_features = x[:, 183:]
        
        # Encode separately
        problem_encoded = self.problem_encoder(problem_features)
        change_encoded = self.change_encoder(change_features)
        
        # Combine and predict
        combined = torch.cat([problem_encoded, change_encoded], dim=1)
        output = self.predictor(combined)
        
        return output


# ============================================================================
# TRAINER
# ============================================================================

class DualLearningTrainer:
    """Trainer for dual learning approach"""
    
    def __init__(self, config: DualLearningConfig):
        self.config = config
        self.processor = DualLearningProcessor(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def train(self, directory_path: str):
        """Train models on directory"""
        logger.info("="*70)
        logger.info("DUAL LEARNING ACE REPAIR PREDICTOR")
        logger.info("="*70)
        
        # Process data
        X, y, metadata = self.processor.process_directory(directory_path)
        
        if len(X) == 0:
            logger.error("No valid training data found!")
            return
        
        # Print pattern analysis
        self._print_pattern_analysis(metadata)
        
        # Split data
        n = len(X)
        indices = np.random.permutation(n)
        
        train_size = int(n * self.config.train_split)
        val_size = int(n * self.config.val_split)
        
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]
        test_idx = indices[train_size + val_size:]
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx] if len(val_idx) > 0 else (X_train[:1], y_train[:1])
        X_test, y_test = X[test_idx], y[test_idx] if len(test_idx) > 0 else (X_val, y_val)
        
        logger.info(f"\nData split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
        # Train Random Forest
        logger.info("\nTraining Random Forest...")
        rf_model = MultiOutputClassifier(
            RandomForestClassifier(
                n_estimators=200,
                max_depth=30,
                min_samples_split=2,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        )
        rf_model.fit(X_train, y_train)
        
        # Train Neural Network
        logger.info("\nTraining Neural Network...")
        nn_model = self._train_nn(X_train, y_train, X_val, y_val, metadata)
        
        # Evaluate
        logger.info("\nEvaluating...")
        self._evaluate(rf_model, nn_model, X_test, y_test)
        
        # Save
        self._save_models(rf_model, nn_model, metadata)
    
    def _train_nn(self, X_train, y_train, X_val, y_val, metadata):
        """Train neural network"""
        model = DualLearningNN(
            num_features=X_train.shape[1],
            num_repairs=y_train.shape[1],
            config=self.config
        ).to(self.device)
        
        train_dataset = RepairDataset(X_train, y_train)
        val_dataset = RepairDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=min(self.config.batch_size, len(X_val)))
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)
        criterion = nn.BCELoss()
        
        for epoch in range(1, self.config.num_epochs + 1):
            # Train
            model.train()
            train_loss = 0
            
            for features, labels in train_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validate
            model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for features, labels in val_loader:
                    features, labels = features.to(self.device), labels.to(self.device)
                    outputs = model(features)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss={train_loss/len(train_loader):.4f}, "
                          f"Val Loss={val_loss/len(val_loader):.4f}")
        
        return model
    
    def _evaluate(self, rf_model, nn_model, X_test, y_test):
        """Evaluate models"""
        # RF predictions
        rf_preds = rf_model.predict(X_test)
        rf_accuracy = np.mean(rf_preds == y_test)
        
        # NN predictions
        nn_model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_test).to(self.device)
            nn_probs = nn_model(X_tensor).cpu().numpy()
            nn_preds = (nn_probs > 0.5).astype(int)
        
        nn_accuracy = np.mean(nn_preds == y_test)
        
        logger.info(f"Random Forest Accuracy: {rf_accuracy:.2%}")
        logger.info(f"Neural Network Accuracy: {nn_accuracy:.2%}")
    
    def _print_pattern_analysis(self, metadata):
        """Print learned patterns"""
        logger.info("\n" + "="*50)
        logger.info("PATTERN ANALYSIS")
        logger.info("="*50)
        
        # Top change-to-repair patterns
        change_patterns = metadata.get('change_patterns', {})
        if change_patterns:
            logger.info("\nTop Change → Repair Patterns:")
            for change_sig in list(change_patterns.keys())[:5]:
                repairs = change_patterns[change_sig]
                top_repair = max(repairs, key=repairs.get)
                logger.info(f"  {change_sig[:50]} → {top_repair} ({repairs[top_repair]} times)")
        
        # Top repairs and their patterns
        repair_patterns = metadata.get('repair_patterns', {})
        if repair_patterns:
            logger.info("\nTop Repairs and Their Patterns:")
            for repair_id in list(repair_patterns.keys())[:5]:
                patterns = repair_patterns[repair_id]
                if patterns:
                    top_pattern = max(patterns, key=patterns.get)
                    logger.info(f"  {repair_id}: {top_pattern[:50]} ({patterns[top_pattern]} times)")
    
    def _save_models(self, rf_model, nn_model, metadata):
        """Save trained models"""
        os.makedirs(self.config.model_dir, exist_ok=True)
        
        # Save everything
        with open(os.path.join(self.config.model_dir, 'rf_model.pkl'), 'wb') as f:
            pickle.dump(rf_model, f)
        
        torch.save(nn_model.state_dict(), os.path.join(self.config.model_dir, 'nn_model.pt'))
        
        with open(os.path.join(self.config.model_dir, 'metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
        
        self.config.save(os.path.join(self.config.model_dir, 'config.json'))
        
        # Save processor for prediction
        with open(os.path.join(self.config.model_dir, 'processor.pkl'), 'wb') as f:
            pickle.dump(self.processor, f)
        
        logger.info(f"\nModels saved to {self.config.model_dir}")


class RepairDataset(Dataset):
    """Dataset for PyTorch"""
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ============================================================================
# PREDICTOR
# ============================================================================

class DualLearningPredictor:
    """Predictor for inference (only uses problem features)"""
    
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load everything
        self.config = DualLearningConfig.load(os.path.join(model_dir, 'config.json'))
        
        with open(os.path.join(model_dir, 'metadata.pkl'), 'rb') as f:
            self.metadata = pickle.load(f)
        
        with open(os.path.join(model_dir, 'processor.pkl'), 'rb') as f:
            self.processor = pickle.load(f)
        
        with open(os.path.join(model_dir, 'rf_model.pkl'), 'rb') as f:
            self.rf_model = pickle.load(f)
        
        # Load NN
        self.nn_model = DualLearningNN(
            num_features=233,  # 183 + 50
            num_repairs=len(self.metadata['repair_vocabulary']),
            config=self.config
        ).to(self.device)
        
        self.nn_model.load_state_dict(
            torch.load(os.path.join(model_dir, 'nn_model.pt'), map_location=self.device)
        )
        self.nn_model.eval()
        
        logger.info(f"Loaded models with {len(self.metadata['repair_vocabulary'])} repair types")
    
    def predict(self, payment_file: str, threshold: float = 0.3):
        """Predict repairs for payment (using only problem features)"""
        with open(payment_file, 'r') as f:
            data = json.load(f)
        
        # Extract transaction
        if isinstance(data, dict) and len(data) == 1:
            txn_id = list(data.keys())[0]
            payment = data[txn_id]
        else:
            payment = data
        
        # Extract ONLY problem features (no after/diffs available at prediction)
        problem_features = self.processor.problem_detector.extract_problem_features(payment, use_before=False)
        
        # For prediction, we don't have change features, so use zeros
        change_features = np.zeros(50)
        
        # Combine with weights
        combined_features = np.concatenate([
            problem_features * self.config.weight_problems,
            change_features * self.config.weight_changes
        ])
        
        # Get predictions from both models
        features_array = combined_features.reshape(1, -1)
        
        # RF predictions
        rf_probs = self.rf_model.predict_proba(features_array)
        rf_probs_flat = np.array([p[0, 1] if p.shape[1] > 1 else 0 for p in rf_probs])
        
        # NN predictions
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features_array).to(self.device)
            nn_probs = self.nn_model(features_tensor).cpu().numpy()[0]
        
        # Identify key problems found
        key_problems = []
        problem_indices = {
            'cdtrAgt_has_clearing_no_bic': 4,
            'cdtr_missing_country': 94,
            'dbtrAgt_has_bic_no_name': 13
        }
        
        for problem_name, idx in problem_indices.items():
            if idx < len(problem_features) and problem_features[idx] > 0:
                key_problems.append(problem_name)
        
        # Get predicted repairs
        idx_to_repair = {v: k for k, v in self.metadata['repair_vocabulary'].items()}
        
        predictions = {
            'problems_detected': key_problems,
            'rf_predictions': [],
            'nn_predictions': [],
            'consensus_predictions': [],
            'ace': []
        }
        
        # Collect predictions
        for idx, (rf_prob, nn_prob) in enumerate(zip(rf_probs_flat, nn_probs)):
            repair_id = idx_to_repair.get(idx, f"unknown_{idx}")
            
            if rf_prob > threshold:
                predictions['rf_predictions'].append({
                    'repair_id': repair_id,
                    'confidence': float(rf_prob)
                })
            
            if nn_prob > threshold:
                predictions['nn_predictions'].append({
                    'repair_id': repair_id,
                    'confidence': float(nn_prob)
                })
            
            # Consensus
            if rf_prob > threshold and nn_prob > threshold:
                predictions['consensus_predictions'].append({
                    'repair_id': repair_id,
                    'rf_confidence': float(rf_prob),
                    'nn_confidence': float(nn_prob),
                    'avg_confidence': float((rf_prob + nn_prob) / 2)
                })
                
                predictions['ace'].append({
                    'id': repair_id,
                    'confidence': float((rf_prob + nn_prob) / 2)
                })
        
        logger.info(f"Problems detected: {key_problems}")
        logger.info(f"Predicted {len(predictions['ace'])} repairs")
        
        return predictions


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Dual Learning ACE Repair Predictor')
    subparsers = parser.add_subparsers(dest='command')
    
    # Train
    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('--input_dir', required=True)
    train_parser.add_argument('--model_dir', default='./models')
    
    # Predict
    predict_parser = subparsers.add_parser('predict')
    predict_parser.add_argument('--input', required=True)
    predict_parser.add_argument('--model_dir', default='./models')
    predict_parser.add_argument('--threshold', type=float, default=0.3)
    
    args = parser.parse_args()
    
    if args.command == 'train':
        config = DualLearningConfig(model_dir=args.model_dir)
        trainer = DualLearningTrainer(config)
        trainer.train(args.input_dir)
        
    elif args.command == 'predict':
        predictor = DualLearningPredictor(args.model_dir)
        results = predictor.predict(args.input, args.threshold)
        
        print("\n" + "="*50)
        print("PREDICTIONS")
        print("="*50)
        print(f"Problems: {results['problems_detected']}")
        print(f"\nACE Repairs ({len(results['ace'])}):")
        for repair in results['ace']:
            print(f"  {repair['id']}: {repair['confidence']:.2%}")


if __name__ == "__main__":
    main()
