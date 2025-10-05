#!/usr/bin/env python3
"""
Enhanced ACE Payment Repair Predictor with Incremental Directory Processing
Version: 7.0 - Optimized for large-scale directory processing

Key Enhancements:
- Incremental file processing to handle large directories
- Enhanced structural change detection with pattern learning
- Source/Clearing combination analysis
- Improved before/after state comparison
- Database lookup pattern recognition
- Memory-efficient processing
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
# ENHANCED LOGGING CONFIGURATION
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create detailed analysis logger
analysis_logger = logging.getLogger('analysis')
analysis_logger.setLevel(logging.DEBUG)
analysis_handler = logging.FileHandler('detailed_analysis.log', mode='w')
analysis_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
analysis_logger.addHandler(analysis_handler)
analysis_logger.propagate = False

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class EnhancedConfig:
    """Enhanced configuration with incremental processing settings"""
    # Model parameters
    hidden_dim: int = 256
    dropout: float = 0.3
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 50
    
    # Data splits
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    
    # Processing parameters
    max_files_in_memory: int = 100  # Process files in batches
    checkpoint_interval: int = 500  # Save checkpoints every N transactions
    enable_incremental: bool = True  # Enable incremental processing
    
    # Paths
    model_dir: str = "./models"
    checkpoint_dir: str = "./checkpoints"
    
    # Analysis parameters
    analyze_source_clearing: bool = True  # Analyze source/clearing patterns
    track_database_lookups: bool = True   # Track database lookup patterns
    detect_value_transformations: bool = True  # Detect value transformation patterns
    
    # Thresholds
    ml_threshold: float = 0.5
    use_class_weights: bool = True
    max_class_weight: float = 10.0
    verbose_logging: bool = True
    
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: str):
        with open(path, 'r') as f:
            return cls(**json.load(f))


# ============================================================================
# ENHANCED STRUCTURAL CHANGE DETECTOR
# ============================================================================

@dataclass
class EnhancedStructuralChange:
    """Enhanced structural change representation"""
    entity: str
    change_type: str  # added, removed, moved, restructured, value_changed, lookup_enriched
    field_path: str
    old_path: Optional[str] = None
    old_value: Optional[Any] = None
    new_value: Optional[Any] = None
    transformation_type: Optional[str] = None  # lookup, formatting, enrichment
    confidence: float = 1.0
    details: Optional[Dict] = None


class EnhancedStructuralDetector:
    """Enhanced detector with pattern learning and database lookup detection"""
    
    def __init__(self, config: EnhancedConfig):
        self.config = config
        self.change_patterns = defaultdict(Counter)
        self.source_clearing_patterns = defaultdict(set)
        self.database_lookup_patterns = defaultdict(list)
        self.value_transformation_patterns = defaultdict(list)
        self.entity_change_sequences = defaultdict(list)
        
    def analyze_transaction(self, txn_id: str, txn_data: Dict) -> Dict:
        """Comprehensive transaction analysis"""
        analysis = {
            'structural_changes': [],
            'database_lookups': [],
            'value_transformations': [],
            'source_clearing_pattern': None,
            'affected_entities': set(),
            'change_complexity': 0
        }
        
        # Analyze source/clearing combination
        if self.config.analyze_source_clearing:
            source = txn_data.get('source', 'unknown')
            clearing = txn_data.get('clearing', 'unknown')
            pattern = f"{source}:{clearing}"
            analysis['source_clearing_pattern'] = pattern
            
            # Learn repair associations with source/clearing
            for repair in txn_data.get('ace', []):
                repair_id = self._get_repair_id(repair)
                self.source_clearing_patterns[pattern].add(repair_id)
        
        # Analyze each entity for changes
        entities = ['cdtrAgt', 'dbtrAgt', 'instgAgt', 'instdAgt',
                   'cdtr', 'dbtr', 'cdtrAcct', 'dbtrAcct', 'rmtInf',
                   'intrmyAgt1', 'intrmyAgt2']
        
        for entity in entities:
            if entity not in txn_data:
                continue
                
            entity_data = txn_data[entity]
            if not isinstance(entity_data, dict):
                continue
            
            before = entity_data.get('before', {})
            after = entity_data.get('after', {})
            
            if not (before or after):
                continue
            
            analysis['affected_entities'].add(entity)
            
            # Detect structural changes
            changes = self._detect_changes(before, after, entity)
            analysis['structural_changes'].extend(changes)
            
            # Detect database lookups
            if self.config.track_database_lookups:
                lookups = self._detect_database_lookups(before, after, entity)
                analysis['database_lookups'].extend(lookups)
            
            # Detect value transformations
            if self.config.detect_value_transformations:
                transformations = self._detect_value_transformations(before, after, entity)
                analysis['value_transformations'].extend(transformations)
        
        # Calculate complexity score
        analysis['change_complexity'] = self._calculate_complexity(analysis)
        
        return analysis
    
    def _detect_changes(self, before: Dict, after: Dict, entity: str) -> List[EnhancedStructuralChange]:
        """Detect all structural changes between states"""
        changes = []
        
        before_inventory = self._get_field_inventory(before)
        after_inventory = self._get_field_inventory(after)
        
        # Find additions
        for field_path in after_inventory.keys() - before_inventory.keys():
            changes.append(EnhancedStructuralChange(
                entity=entity,
                change_type='added',
                field_path=field_path,
                new_value=after_inventory[field_path]['value']
            ))
        
        # Find removals
        for field_path in before_inventory.keys() - after_inventory.keys():
            changes.append(EnhancedStructuralChange(
                entity=entity,
                change_type='removed',
                field_path=field_path,
                old_value=before_inventory[field_path]['value']
            ))
        
        # Find movements (same value, different location)
        value_locations_before = defaultdict(set)
        value_locations_after = defaultdict(set)
        
        for path, info in before_inventory.items():
            val_hash = self._hash_value(info['value'])
            if val_hash:
                value_locations_before[val_hash].add(path)
        
        for path, info in after_inventory.items():
            val_hash = self._hash_value(info['value'])
            if val_hash:
                value_locations_after[val_hash].add(path)
        
        for val_hash in value_locations_before.keys() & value_locations_after.keys():
            before_paths = value_locations_before[val_hash]
            after_paths = value_locations_after[val_hash]
            
            if before_paths != after_paths:
                for old_path in before_paths - after_paths:
                    for new_path in after_paths - before_paths:
                        changes.append(EnhancedStructuralChange(
                            entity=entity,
                            change_type='moved',
                            field_path=new_path,
                            old_path=old_path,
                            new_value=after_inventory[new_path]['value']
                        ))
        
        return changes
    
    def _detect_database_lookups(self, before: Dict, after: Dict, entity: str) -> List[Dict]:
        """Detect patterns indicating database lookups"""
        lookups = []
        
        # Pattern 1: BIC added when clearing number exists
        before_clearing = self._find_field(before, ['mmbId', 'clrSysMmbId'])
        after_bic = self._find_field(after, ['bicFi', 'bic'])
        before_bic = self._find_field(before, ['bicFi', 'bic'])
        
        if before_clearing and after_bic and not before_bic:
            lookups.append({
                'type': 'bic_lookup',
                'input': before_clearing,
                'output': after_bic,
                'entity': entity
            })
            self.database_lookup_patterns['bic_lookup'].append((before_clearing, after_bic))
        
        # Pattern 2: Name added when BIC exists
        before_bic = self._find_field(before, ['bicFi', 'bic'])
        after_name = self._find_field(after, ['nm', 'name'])
        before_name = self._find_field(before, ['nm', 'name'])
        
        if before_bic and after_name and not before_name:
            lookups.append({
                'type': 'name_lookup',
                'input': before_bic,
                'output': after_name,
                'entity': entity
            })
            self.database_lookup_patterns['name_lookup'].append((before_bic, after_name))
        
        # Pattern 3: Address enrichment
        before_has_partial = self._find_field(before, ['townName', 'ctry'])
        after_has_full = self._find_field(after, ['adrLine'])
        
        if before_has_partial and after_has_full:
            lookups.append({
                'type': 'address_enrichment',
                'entity': entity
            })
        
        return lookups
    
    def _detect_value_transformations(self, before: Dict, after: Dict, entity: str) -> List[Dict]:
        """Detect value transformation patterns"""
        transformations = []
        
        before_fields = self._flatten_dict(before)
        after_fields = self._flatten_dict(after)
        
        for field_name in before_fields.keys() & after_fields.keys():
            before_val = before_fields[field_name]
            after_val = after_fields[field_name]
            
            if before_val != after_val:
                transformation = self._classify_transformation(before_val, after_val)
                if transformation:
                    transformations.append({
                        'field': field_name,
                        'type': transformation,
                        'before': str(before_val)[:100],
                        'after': str(after_val)[:100],
                        'entity': entity
                    })
                    self.value_transformation_patterns[transformation].append((field_name, entity))
        
        return transformations
    
    def _classify_transformation(self, before_val: Any, after_val: Any) -> Optional[str]:
        """Classify the type of value transformation"""
        before_str = str(before_val).lower()
        after_str = str(after_val).lower()
        
        # Case normalization
        if before_str == after_str:
            return 'case_normalization'
        
        # Format standardization
        if before_str.replace(' ', '').replace(',', '') == after_str.replace(' ', '').replace(',', ''):
            return 'format_standardization'
        
        # Address formatting
        if any(term in before_str for term in ['street', 'ave', 'road']) or \
           any(term in after_str for term in ['street', 'ave', 'road']):
            return 'address_formatting'
        
        # Country code extraction
        if len(after_str) == 2 and after_str.isupper() and after_str in before_str:
            return 'country_extraction'
        
        # General enrichment
        if len(after_str) > len(before_str) * 1.5:
            return 'data_enrichment'
        
        # General transformation
        return 'value_transformation'
    
    def _get_field_inventory(self, obj: Any, path: str = "") -> Dict[str, Dict]:
        """Get inventory of all fields with paths"""
        inventory = {}
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                current_path = f"{path}.{key}" if path else key
                inventory[current_path] = {
                    'value': value,
                    'type': type(value).__name__
                }
                if isinstance(value, (dict, list)):
                    nested = self._get_field_inventory(value, current_path)
                    inventory.update(nested)
                    
        elif isinstance(obj, list):
            for idx, item in enumerate(obj):
                current_path = f"{path}[{idx}]"
                inventory[current_path] = {
                    'value': item,
                    'type': type(item).__name__
                }
                if isinstance(item, (dict, list)):
                    nested = self._get_field_inventory(item, current_path)
                    inventory.update(nested)
        
        return inventory
    
    def _find_field(self, obj: Dict, field_names: List[str]) -> Optional[Any]:
        """Find field by multiple possible names"""
        def search(o, depth=0):
            if depth > 10 or not isinstance(o, dict):
                return None
            
            for field_name in field_names:
                for key, value in o.items():
                    if field_name.lower() in key.lower():
                        return value
                    
            for value in o.values():
                if isinstance(value, dict):
                    result = search(value, depth + 1)
                    if result is not None:
                        return result
            return None
        
        return search(obj)
    
    def _flatten_dict(self, obj: Dict, parent_key: str = '') -> Dict:
        """Flatten nested dictionary"""
        items = []
        for k, v in obj.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def _hash_value(self, value: Any) -> Optional[str]:
        """Create hash of value for comparison"""
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            if str(value).strip():
                return hashlib.md5(str(value).encode()).hexdigest()[:8]
        return None
    
    def _calculate_complexity(self, analysis: Dict) -> float:
        """Calculate change complexity score"""
        score = 0.0
        score += len(analysis['structural_changes']) * 1.0
        score += len(analysis['database_lookups']) * 2.0
        score += len(analysis['value_transformations']) * 0.5
        score += len(analysis['affected_entities']) * 1.5
        return score
    
    def _get_repair_id(self, repair: Any) -> str:
        """Extract repair ID from repair data"""
        if isinstance(repair, dict):
            return str(repair.get('id', 'unknown'))
        return str(repair)


# ============================================================================
# INCREMENTAL DATA PROCESSOR
# ============================================================================

class IncrementalDataProcessor:
    """Process large datasets incrementally"""
    
    def __init__(self, config: EnhancedConfig):
        self.config = config
        self.structural_detector = EnhancedStructuralDetector(config)
        self.repair_patterns = defaultdict(lambda: defaultdict(int))
        self.repair_vocabulary = {}
        self.feature_cache = {}
        self.transaction_count = 0
        self.checkpoint_count = 0
        
    def process_directory_incrementally(self, directory_path: str, 
                                       callback_fn: Optional[callable] = None) -> Generator:
        """
        Process directory incrementally, yielding results file by file
        
        Args:
            directory_path: Path to directory containing JSON files
            callback_fn: Optional callback for each processed transaction
            
        Yields:
            Processed transaction data
        """
        json_files = sorted(Path(directory_path).glob('**/*.json'))
        
        logger.info(f"Found {len(json_files)} JSON files in {directory_path}")
        
        for file_idx, json_file in enumerate(tqdm(json_files, desc="Processing files")):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Handle different JSON structures
                transactions = self._extract_transactions(data)
                
                for txn_id, txn_data in transactions.items():
                    if not isinstance(txn_data, dict):
                        continue
                    
                    # Process transaction
                    processed = self._process_transaction(txn_id, txn_data)
                    
                    if processed:
                        self.transaction_count += 1
                        
                        # Checkpoint if needed
                        if self.config.checkpoint_interval > 0 and \
                           self.transaction_count % self.config.checkpoint_interval == 0:
                            self._save_checkpoint()
                        
                        # Callback for monitoring
                        if callback_fn:
                            callback_fn(self.transaction_count, processed)
                        
                        yield processed
                
                # Clear memory periodically
                if file_idx % self.config.max_files_in_memory == 0:
                    self._clear_old_cache()
                    
            except Exception as e:
                logger.warning(f"Error processing {json_file}: {e}")
                continue
        
        # Final checkpoint
        self._save_checkpoint()
        logger.info(f"Completed processing {self.transaction_count} transactions")
    
    def _extract_transactions(self, data: Any) -> Dict:
        """Extract transactions from various JSON formats"""
        transactions = {}
        
        if isinstance(data, dict):
            # Direct transaction dictionary
            for key, value in data.items():
                if isinstance(value, dict):
                    transactions[str(key)] = value
                    
        elif isinstance(data, list):
            # List of transactions
            for idx, item in enumerate(data):
                if isinstance(item, dict):
                    # Single transaction wrapped in list
                    if len(item) == 1:
                        txn_id = list(item.keys())[0]
                        transactions[str(txn_id)] = item[txn_id]
                    else:
                        # Direct transaction object
                        transactions[f"txn_{idx}"] = item
        
        return transactions
    
    def _process_transaction(self, txn_id: str, txn_data: Dict) -> Optional[Dict]:
        """Process a single transaction"""
        try:
            # Skip if no repairs
            if 'ace' not in txn_data or not txn_data.get('ace'):
                return None
            
            # Analyze transaction
            analysis = self.structural_detector.analyze_transaction(txn_id, txn_data)
            
            # Extract features
            features = self._extract_features(txn_data, analysis)
            
            # Get repair labels
            repairs = []
            for repair in txn_data.get('ace', []):
                repair_id = str(repair.get('id', 'unknown')) if isinstance(repair, dict) else str(repair)
                repairs.append(repair_id)
                
                # Update vocabulary
                if repair_id not in self.repair_vocabulary:
                    self.repair_vocabulary[repair_id] = len(self.repair_vocabulary)
                
                # Learn patterns
                self._learn_repair_patterns(repair_id, analysis)
            
            return {
                'txn_id': txn_id,
                'features': features,
                'repairs': repairs,
                'analysis': analysis,
                'source': txn_data.get('source'),
                'clearing': txn_data.get('clearing')
            }
            
        except Exception as e:
            logger.debug(f"Error processing transaction {txn_id}: {e}")
            return None
    
    def _extract_features(self, txn_data: Dict, analysis: Dict) -> np.ndarray:
        """Extract features from transaction and analysis"""
        features = []
        
        # Source/Clearing features
        source = txn_data.get('source', 'unknown')
        clearing = txn_data.get('clearing', 'unknown')
        features.extend([
            float(source == 'SWF'),
            float(source == 'ACH'),
            float(clearing == 'FED'),
            float(clearing == 'CHIPS')
        ])
        
        # Structural change features
        change_types = Counter(c.change_type for c in analysis['structural_changes'])
        features.extend([
            change_types.get('added', 0) / 10,
            change_types.get('removed', 0) / 10,
            change_types.get('moved', 0) / 5,
            change_types.get('restructured', 0) / 5
        ])
        
        # Database lookup features
        lookup_types = Counter(l['type'] for l in analysis['database_lookups'])
        features.extend([
            float(lookup_types.get('bic_lookup', 0) > 0),
            float(lookup_types.get('name_lookup', 0) > 0),
            float(lookup_types.get('address_enrichment', 0) > 0)
        ])
        
        # Entity involvement features
        entities = ['cdtrAgt', 'dbtrAgt', 'instgAgt', 'instdAgt']
        for entity in entities:
            features.append(float(entity in analysis['affected_entities']))
        
        # Complexity feature
        features.append(min(analysis['change_complexity'] / 50, 1.0))
        
        # Flags features
        flags = txn_data.get('flags', {})
        important_flags = [
            'ace_repairs', 'stp_failed', 'repair_queue',
            'ignored_party', 'credit_party_identified_by_name'
        ]
        for flag in important_flags:
            features.append(float(flags.get(flag, False)))
        
        return np.array(features)
    
    def _learn_repair_patterns(self, repair_id: str, analysis: Dict):
        """Learn patterns associated with repairs"""
        # Learn structural patterns
        for change in analysis['structural_changes']:
            pattern_key = f"{change.change_type}:{change.entity}"
            self.repair_patterns[repair_id][pattern_key] += 1
        
        # Learn lookup patterns
        for lookup in analysis['database_lookups']:
            pattern_key = f"lookup:{lookup['type']}"
            self.repair_patterns[repair_id][pattern_key] += 1
        
        # Learn source/clearing patterns
        if analysis['source_clearing_pattern']:
            self.repair_patterns[repair_id][analysis['source_clearing_pattern']] += 1
    
    def _clear_old_cache(self):
        """Clear old cached data to free memory"""
        if len(self.feature_cache) > 10000:
            # Keep only recent entries
            recent_keys = sorted(self.feature_cache.keys())[-5000:]
            self.feature_cache = {k: self.feature_cache[k] for k in recent_keys}
            logger.debug(f"Cleared cache, kept {len(self.feature_cache)} recent entries")
    
    def _save_checkpoint(self):
        """Save processing checkpoint"""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        self.checkpoint_count += 1
        
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f"checkpoint_{self.checkpoint_count}.pkl"
        )
        
        checkpoint_data = {
            'transaction_count': self.transaction_count,
            'repair_vocabulary': self.repair_vocabulary,
            'repair_patterns': dict(self.repair_patterns),
            'source_clearing_patterns': dict(self.structural_detector.source_clearing_patterns),
            'database_lookup_patterns': dict(self.structural_detector.database_lookup_patterns)
        }
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        logger.info(f"Saved checkpoint {self.checkpoint_count} ({self.transaction_count} transactions)")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load from checkpoint"""
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        self.transaction_count = checkpoint_data['transaction_count']
        self.repair_vocabulary = checkpoint_data['repair_vocabulary']
        self.repair_patterns = defaultdict(lambda: defaultdict(int), checkpoint_data['repair_patterns'])
        self.structural_detector.source_clearing_patterns = defaultdict(set, checkpoint_data['source_clearing_patterns'])
        self.structural_detector.database_lookup_patterns = defaultdict(list, checkpoint_data['database_lookup_patterns'])
        
        logger.info(f"Loaded checkpoint with {self.transaction_count} transactions")


# ============================================================================
# ENHANCED TRAINING PIPELINE
# ============================================================================

class EnhancedTrainer:
    """Enhanced trainer with incremental processing support"""
    
    def __init__(self, config: EnhancedConfig):
        self.config = config
        self.processor = IncrementalDataProcessor(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
    
    def train_on_directory(self, directory_path: str):
        """Train model on directory of JSON files"""
        logger.info("="*70)
        logger.info("ENHANCED ACE REPAIR PREDICTOR TRAINING")
        logger.info("="*70)
        logger.info(f"Processing directory: {directory_path}")
        
        # Collect processed data
        all_features = []
        all_labels = []
        
        def progress_callback(count, processed):
            if count % 100 == 0:
                logger.info(f"Processed {count} transactions...")
        
        # Process incrementally
        for processed_data in self.processor.process_directory_incrementally(
            directory_path, 
            callback_fn=progress_callback
        ):
            all_features.append(processed_data['features'])
            
            # Convert repairs to labels
            labels = np.zeros(len(self.processor.repair_vocabulary))
            for repair_id in processed_data['repairs']:
                if repair_id in self.processor.repair_vocabulary:
                    labels[self.processor.repair_vocabulary[repair_id]] = 1.0
            all_labels.append(labels)
        
        # Convert to arrays
        X = np.array(all_features)
        y = np.array(all_labels)
        
        logger.info(f"\nData shape: {X.shape}")
        logger.info(f"Unique repairs: {len(self.processor.repair_vocabulary)}")
        
        # Print analysis summary
        self._print_analysis_summary()
        
        # Train model
        self._train_model(X, y)
    
    def _print_analysis_summary(self):
        """Print summary of learned patterns"""
        logger.info("\n" + "="*70)
        logger.info("PATTERN ANALYSIS SUMMARY")
        logger.info("="*70)
        
        # Source/Clearing patterns
        logger.info("\nSource/Clearing Patterns:")
        patterns = self.processor.structural_detector.source_clearing_patterns
        for pattern, repairs in list(patterns.items())[:10]:
            logger.info(f"  {pattern}: {len(repairs)} repair types")
        
        # Database lookup patterns
        logger.info("\nDatabase Lookup Patterns:")
        for lookup_type, examples in self.processor.structural_detector.database_lookup_patterns.items():
            logger.info(f"  {lookup_type}: {len(examples)} examples")
        
        # Top repair patterns
        logger.info("\nTop Repair Patterns:")
        for repair_id, patterns in list(self.processor.repair_patterns.items())[:5]:
            logger.info(f"  Repair {repair_id}:")
            top_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:3]
            for pattern, count in top_patterns:
                logger.info(f"    {pattern}: {count} occurrences")
    
    def _train_model(self, X: np.ndarray, y: np.ndarray):
        """Train the model"""
        # Split data
        n = len(X)
        indices = np.random.permutation(n)
        
        train_size = int(n * self.config.train_split)
        val_size = int(n * self.config.val_split)
        
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]
        test_idx = indices[train_size + val_size:]
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        logger.info(f"\nData split:")
        logger.info(f"  Train: {len(X_train)}")
        logger.info(f"  Validation: {len(X_val)}")
        logger.info(f"  Test: {len(X_test)}")
        
        # Train Random Forest
        logger.info("\nTraining Random Forest...")
        rf_model = MultiOutputClassifier(
            RandomForestClassifier(
                n_estimators=200,
                max_depth=25,
                min_samples_split=3,
                min_samples_leaf=1,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        )
        rf_model.fit(X_train, y_train)
        
        # Evaluate
        rf_preds = rf_model.predict(X_test)
        exact_match = np.mean(np.all(rf_preds == y_test, axis=1))
        hamming = np.mean(rf_preds == y_test)
        
        logger.info(f"\nModel Performance:")
        logger.info(f"  Exact Match: {exact_match:.2%}")
        logger.info(f"  Hamming Accuracy: {hamming:.2%}")
        
        # Save model
        self._save_model(rf_model)
    
    def _save_model(self, model):
        """Save trained model and processor"""
        os.makedirs(self.config.model_dir, exist_ok=True)
        
        # Save config
        self.config.save(os.path.join(self.config.model_dir, 'config.json'))
        
        # Save processor
        with open(os.path.join(self.config.model_dir, 'processor.pkl'), 'wb') as f:
            pickle.dump(self.processor, f)
        
        # Save model
        with open(os.path.join(self.config.model_dir, 'model.pkl'), 'wb') as f:
            pickle.dump(model, f)
        
        logger.info(f"\nModel saved to {self.config.model_dir}")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Enhanced ACE Payment Repair Predictor'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--input_dir', required=True, help='Directory containing JSON files')
    train_parser.add_argument('--model_dir', default='./models')
    train_parser.add_argument('--checkpoint_dir', default='./checkpoints')
    train_parser.add_argument('--batch_size', type=int, default=100)
    train_parser.add_argument('--incremental', action='store_true', help='Use incremental processing')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze patterns in data')
    analyze_parser.add_argument('--input_dir', required=True)
    analyze_parser.add_argument('--output', help='Output analysis file')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        config = EnhancedConfig(
            model_dir=args.model_dir,
            checkpoint_dir=args.checkpoint_dir,
            max_files_in_memory=args.batch_size,
            enable_incremental=args.incremental
        )
        
        trainer = EnhancedTrainer(config)
        trainer.train_on_directory(args.input_dir)
        
        logger.info("\nTraining complete!")
        logger.info("Check detailed_analysis.log for comprehensive analysis")
        
    elif args.command == 'analyze':
        config = EnhancedConfig()
        processor = IncrementalDataProcessor(config)
        
        analysis_results = []
        for processed in processor.process_directory_incrementally(args.input_dir):
            analysis_results.append(processed['analysis'])
        
        # Save analysis
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(analysis_results, f, indent=2, default=str)
            logger.info(f"Analysis saved to {args.output}")
        
        # Print summary
        processor._print_analysis_summary()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
