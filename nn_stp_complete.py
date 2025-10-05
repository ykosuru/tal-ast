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

def create_counter_dict():
    """Factory function for creating defaultdict(int)"""
    return defaultdict(int)

class IncrementalDataProcessor:
    """Process large datasets incrementally"""
    
    def __init__(self, config: EnhancedConfig):
        self.config = config
        self.structural_detector = EnhancedStructuralDetector(config)
        self.repair_patterns = defaultdict(create_counter_dict)
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
    
    def print_analysis_summary(self):
        """Print summary of learned patterns"""
        logger.info("\n" + "="*70)
        logger.info("PATTERN ANALYSIS SUMMARY")
        logger.info("="*70)
        
        logger.info(f"Total transactions processed: {self.transaction_count}")
        logger.info(f"Unique repairs found: {len(self.repair_vocabulary)}")
        
        # Source/Clearing patterns
        if self.structural_detector.source_clearing_patterns:
            logger.info("\nSource/Clearing Patterns:")
            for pattern, repairs in list(self.structural_detector.source_clearing_patterns.items())[:10]:
                logger.info(f"  {pattern}: {len(repairs)} repair types - {list(repairs)[:5]}")
        
        # Database lookup patterns
        if self.structural_detector.database_lookup_patterns:
            logger.info("\nDatabase Lookup Patterns:")
            for lookup_type, examples in self.structural_detector.database_lookup_patterns.items():
                logger.info(f"  {lookup_type}: {len(examples)} examples")
        
        # Top repair patterns
        if self.repair_patterns:
            logger.info("\nTop Repair Patterns:")
            sorted_repairs = sorted(self.repair_patterns.items(), 
                                   key=lambda x: sum(x[1].values()), 
                                   reverse=True)
            for repair_id, patterns in sorted_repairs[:5]:
                logger.info(f"  Repair {repair_id}:")
                top_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:3]
                for pattern, count in top_patterns:
                    logger.info(f"    {pattern}: {count} occurrences")


# ============================================================================
# NEURAL NETWORK MODEL
# ============================================================================

class RepairNN(nn.Module):
    """Neural network for repair prediction"""
    
    def __init__(self, num_features: int, num_repairs: int, 
                 hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        
        self.feature_encoder = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.prediction_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.Dropout(dropout / 2),
            
            nn.Linear(hidden_dim // 4, num_repairs),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.feature_encoder(x)
        output = self.prediction_network(encoded)
        return output


class RepairDataset(Dataset):
    """PyTorch dataset for repairs"""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class EnhancedTrainer:
    """Enhanced trainer with incremental processing support"""
    
    def __init__(self, config: EnhancedConfig):
        self.config = config
        self.processor = IncrementalDataProcessor(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
    
    def build_vocabulary_from_directory(self, directory_path: str) -> Dict[str, int]:
        """Build complete vocabulary from directory (memory efficient)"""
        logger.info("Building vocabulary from directory...")
        vocabulary = {}
        transaction_count = 0
        
        json_files = sorted(Path(directory_path).glob('**/*.json'))
        
        for json_file in tqdm(json_files, desc="Scanning for repairs"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                transactions = self.processor._extract_transactions(data)
                
                for txn_id, txn_data in transactions.items():
                    if not isinstance(txn_data, dict):
                        continue
                    
                    # Extract repairs for vocabulary
                    for repair in txn_data.get('ace', []):
                        if isinstance(repair, dict):
                            repair_id = str(repair.get('id', 'unknown'))
                        else:
                            repair_id = str(repair)
                        
                        if repair_id not in vocabulary:
                            vocabulary[repair_id] = len(vocabulary)
                    
                    transaction_count += 1
                    
            except Exception as e:
                logger.debug(f"Error reading {json_file}: {e}")
                continue
        
        logger.info(f"Built vocabulary with {len(vocabulary)} unique repairs from {transaction_count} transactions")
        return vocabulary
    
    def train_on_directory(self, directory_path: str):
        """Train model on directory of JSON files"""
        logger.info("="*70)
        logger.info("ENHANCED ACE REPAIR PREDICTOR TRAINING")
        logger.info("="*70)
        logger.info(f"Processing directory: {directory_path}")
        
        # Option 1: Memory-efficient - build vocabulary first
        vocabulary = self.build_vocabulary_from_directory(directory_path)
        self.processor.repair_vocabulary = vocabulary
        
        # Now process with fixed vocabulary
        logger.info("Processing transactions with fixed vocabulary...")
        all_features = []
        all_labels = []
        vocab_size = len(vocabulary)
        
        def progress_callback(count, processed):
            if count % 100 == 0:
                logger.info(f"Processed {count} transactions...")
        
        # Process incrementally with fixed vocabulary
        for processed_data in self.processor.process_directory_incrementally(
            directory_path, 
            callback_fn=progress_callback
        ):
            all_features.append(processed_data['features'])
            
            # Create fixed-size label vector
            labels = np.zeros(vocab_size)
            for repair_id in processed_data['repairs']:
                if repair_id in vocabulary:
                    labels[vocabulary[repair_id]] = 1.0
            all_labels.append(labels)
        
        # Convert to arrays - now all labels have same size
        X = np.array(all_features)
        y = np.array(all_labels)
        
        logger.info(f"\nData shape: {X.shape}")
        logger.info(f"Label shape: {y.shape}")
        logger.info(f"Unique repairs: {vocab_size}")
        
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
        """Train both Random Forest and Neural Network models"""
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
        
        # Train Neural Network
        logger.info("\nTraining Neural Network...")
        nn_model = self._train_neural_network(X_train, y_train, X_val, y_val)
        
        # Evaluate both models
        logger.info("\nEvaluating models on test set...")
        
        # RF evaluation
        rf_preds = rf_model.predict(X_test)
        rf_exact_match = np.mean(np.all(rf_preds == y_test, axis=1))
        rf_hamming = np.mean(rf_preds == y_test)
        
        # NN evaluation
        nn_model.eval()
        test_dataset = RepairDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size)
        
        all_nn_preds = []
        with torch.no_grad():
            for features, _ in test_loader:
                features = features.to(self.device)
                outputs = nn_model(features)
                all_nn_preds.append(outputs.cpu().numpy())
        
        nn_preds = np.vstack(all_nn_preds)
        nn_preds_binary = (nn_preds > 0.5).astype(int)
        nn_exact_match = np.mean(np.all(nn_preds_binary == y_test, axis=1))
        nn_hamming = np.mean(nn_preds_binary == y_test)
        
        logger.info(f"\nModel Performance:")
        logger.info(f"  Random Forest:")
        logger.info(f"    Exact Match: {rf_exact_match:.2%}")
        logger.info(f"    Hamming Accuracy: {rf_hamming:.2%}")
        logger.info(f"  Neural Network:")
        logger.info(f"    Exact Match: {nn_exact_match:.2%}")
        logger.info(f"    Hamming Accuracy: {nn_hamming:.2%}")
        
        # Save both models
        self._save_models(rf_model, nn_model)
    
    def _train_neural_network(self, X_train, y_train, X_val, y_val):
        """Train neural network model"""
        num_features = X_train.shape[1]
        num_repairs = y_train.shape[1]
        
        model = RepairNN(
            num_features=num_features,
            num_repairs=num_repairs,
            hidden_dim=self.config.hidden_dim,
            dropout=self.config.dropout
        ).to(self.device)
        
        train_dataset = RepairDataset(X_train, y_train)
        val_dataset = RepairDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        criterion = nn.BCELoss()
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(1, self.config.num_epochs + 1):
            # Training
            model.train()
            train_loss = 0.0
            
            for features, labels in train_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for features, labels in val_loader:
                    features = features.to(self.device)
                    labels = labels.to(self.device)
                    outputs = model(features)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            scheduler.step(val_loss)
            
            if epoch % 10 == 0 or epoch == 1:
                logger.info(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss - 0.001:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 15:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        return model
    
    def _save_models(self, rf_model, nn_model):
        """Save both trained models and processor"""
        os.makedirs(self.config.model_dir, exist_ok=True)
        
        # Save config
        self.config.save(os.path.join(self.config.model_dir, 'config.json'))
        
        # Save processor
        with open(os.path.join(self.config.model_dir, 'processor.pkl'), 'wb') as f:
            pickle.dump(self.processor, f)
        
        # Save RF model
        with open(os.path.join(self.config.model_dir, 'rf_model.pkl'), 'wb') as f:
            pickle.dump(rf_model, f)
        
        # Save NN model
        torch.save(nn_model.state_dict(), os.path.join(self.config.model_dir, 'nn_model.pt'))
        
        # Save model info
        model_info = {
            'num_features': nn_model.feature_encoder[0].in_features,
            'num_repairs': nn_model.prediction_network[-2].out_features,
            'hidden_dim': self.config.hidden_dim,
            'dropout': self.config.dropout
        }
        with open(os.path.join(self.config.model_dir, 'model_info.json'), 'w') as f:
            json.dump(model_info, f, indent=2)
        
        logger.info(f"\nModels saved to {self.config.model_dir}")


# ============================================================================
# ENHANCED PREDICTOR WITH DUAL MODEL SUPPORT
# ============================================================================

class EnhancedPredictor:
    """Predictor that provides individual model predictions with confidence scores"""
    
    def __init__(self, model_dir: str = './models'):
        self.model_dir = model_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Loading models from {model_dir}")
        
        # Load config
        self.config = EnhancedConfig.load(os.path.join(model_dir, 'config.json'))
        
        # Load processor
        with open(os.path.join(model_dir, 'processor.pkl'), 'rb') as f:
            self.processor = pickle.load(f)
        
        # Load Random Forest
        with open(os.path.join(model_dir, 'rf_model.pkl'), 'rb') as f:
            self.rf_model = pickle.load(f)
        
        # Load Neural Network
        with open(os.path.join(model_dir, 'model_info.json'), 'r') as f:
            model_info = json.load(f)
        
        self.nn_model = RepairNN(
            num_features=model_info['num_features'],
            num_repairs=model_info['num_repairs'],
            hidden_dim=model_info['hidden_dim'],
            dropout=model_info['dropout']
        ).to(self.device)
        
        self.nn_model.load_state_dict(
            torch.load(os.path.join(model_dir, 'nn_model.pt'), 
                      map_location=self.device)
        )
        self.nn_model.eval()
        
        # Create reverse mapping for repair IDs
        self.idx_to_repair = {idx: repair_id for repair_id, idx in self.processor.repair_vocabulary.items()}
        
        logger.info(f"Models loaded successfully")
        logger.info(f"Vocabulary contains {len(self.processor.repair_vocabulary)} repair types")
    
    def predict(self, payment_file: str, threshold: float = 0.5) -> Dict:
        """
        Predict ACE repairs with individual model confidence scores
        
        Returns predictions from both Neural Network and Random Forest models
        with confidence levels for each predicted repair ID
        """
        
        # Load payment
        with open(payment_file, 'r') as f:
            data = json.load(f)
        
        # Extract transaction
        if isinstance(data, dict) and len(data) == 1:
            txn_id = list(data.keys())[0]
            txn_data = data[txn_id]
            logger.info(f"Processing transaction: {txn_id}")
        else:
            txn_data = data
            txn_id = "unknown"
        
        # Analyze transaction structure
        analysis = self.processor.structural_detector.analyze_transaction(txn_id, txn_data)
        
        # Extract features
        features = self.processor._extract_features(txn_data, analysis)
        
        # Get Neural Network predictions
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        with torch.no_grad():
            nn_probs = self.nn_model(features_tensor)[0].cpu().numpy()
        
        # Get Random Forest predictions
        rf_probs_raw = self.rf_model.predict_proba(features.reshape(1, -1))
        rf_probs = np.array([
            p[0, 1] if p.shape[1] > 1 else p[0, 0] 
            for p in rf_probs_raw
        ])
        
        # Prepare results
        predictions = {
            'transaction_id': txn_id,
            'source': txn_data.get('source', 'unknown'),
            'clearing': txn_data.get('clearing', 'unknown'),
            'structural_analysis': {
                'affected_entities': list(analysis['affected_entities']),
                'change_complexity': analysis['change_complexity'],
                'database_lookups_detected': len(analysis['database_lookups']),
                'structural_changes_detected': len(analysis['structural_changes'])
            },
            'neural_network_predictions': [],
            'random_forest_predictions': [],
            'consensus_predictions': [],
            'ace': []  # Final ACE repair array
        }
        
        # Process Neural Network predictions
        nn_predictions = []
        for idx, prob in enumerate(nn_probs):
            if prob > threshold:
                repair_id = self.idx_to_repair.get(idx, f"unknown_{idx}")
                nn_predictions.append({
                    'repair_id': repair_id,
                    'confidence': float(prob),
                    'confidence_level': self._get_confidence_level(prob)
                })
        
        # Process Random Forest predictions
        rf_predictions = []
        for idx, prob in enumerate(rf_probs):
            if prob > threshold:
                repair_id = self.idx_to_repair.get(idx, f"unknown_{idx}")
                rf_predictions.append({
                    'repair_id': repair_id,
                    'confidence': float(prob),
                    'confidence_level': self._get_confidence_level(prob)
                })
        
        # Find consensus predictions (predicted by both models)
        nn_repair_ids = {p['repair_id'] for p in nn_predictions}
        rf_repair_ids = {p['repair_id'] for p in rf_predictions}
        consensus_ids = nn_repair_ids & rf_repair_ids
        
        consensus_predictions = []
        for repair_id in consensus_ids:
            nn_conf = next(p['confidence'] for p in nn_predictions if p['repair_id'] == repair_id)
            rf_conf = next(p['confidence'] for p in rf_predictions if p['repair_id'] == repair_id)
            avg_conf = (nn_conf + rf_conf) / 2
            
            consensus_predictions.append({
                'repair_id': repair_id,
                'nn_confidence': float(nn_conf),
                'rf_confidence': float(rf_conf),
                'average_confidence': float(avg_conf),
                'confidence_level': self._get_confidence_level(avg_conf)
            })
        
        # Sort predictions by confidence
        nn_predictions.sort(key=lambda x: x['confidence'], reverse=True)
        rf_predictions.sort(key=lambda x: x['confidence'], reverse=True)
        consensus_predictions.sort(key=lambda x: x['average_confidence'], reverse=True)
        
        predictions['neural_network_predictions'] = nn_predictions
        predictions['random_forest_predictions'] = rf_predictions
        predictions['consensus_predictions'] = consensus_predictions
        
        # Create ACE array with repair details (using consensus or highest confidence)
        ace_repairs = []
        used_repairs = set()
        
        # First add consensus predictions
        for pred in consensus_predictions:
            repair_id = pred['repair_id']
            if repair_id not in used_repairs:
                ace_repairs.append({
                    'id': repair_id,
                    'code': 'I',  # Default code, would need mapping from training data
                    'field': self._get_repair_field(repair_id),
                    'text': self._get_repair_text(repair_id),
                    'confidence': pred['average_confidence'],
                    'predicted_by': 'both_models'
                })
                used_repairs.add(repair_id)
        
        # Add high-confidence single-model predictions
        for pred in nn_predictions:
            if pred['repair_id'] not in used_repairs and pred['confidence'] > 0.7:
                ace_repairs.append({
                    'id': pred['repair_id'],
                    'code': 'I',
                    'field': self._get_repair_field(pred['repair_id']),
                    'text': self._get_repair_text(pred['repair_id']),
                    'confidence': pred['confidence'],
                    'predicted_by': 'neural_network'
                })
                used_repairs.add(pred['repair_id'])
        
        for pred in rf_predictions:
            if pred['repair_id'] not in used_repairs and pred['confidence'] > 0.7:
                ace_repairs.append({
                    'id': pred['repair_id'],
                    'code': 'I',
                    'field': self._get_repair_field(pred['repair_id']),
                    'text': self._get_repair_text(pred['repair_id']),
                    'confidence': pred['confidence'],
                    'predicted_by': 'random_forest'
                })
                used_repairs.add(pred['repair_id'])
        
        predictions['ace'] = ace_repairs
        
        # Summary statistics
        predictions['summary'] = {
            'total_nn_predictions': len(nn_predictions),
            'total_rf_predictions': len(rf_predictions),
            'consensus_predictions': len(consensus_predictions),
            'total_unique_predictions': len(nn_repair_ids | rf_repair_ids),
            'high_confidence_predictions': len([p for p in ace_repairs if p['confidence'] > 0.8]),
            'threshold_used': threshold
        }
        
        return predictions
    
    def _get_confidence_level(self, confidence: float) -> str:
        """Categorize confidence level"""
        if confidence > 0.9:
            return 'very_high'
        elif confidence > 0.8:
            return 'high'
        elif confidence > 0.6:
            return 'medium'
        else:
            return 'low'
    
    def _get_repair_field(self, repair_id: str) -> str:
        """Get field associated with repair (would need mapping from training data)"""
        # This would ideally come from learned patterns
        field_mappings = {
            '6021': 'CDTPTY',
            '8852': 'DBTRAGT',
            # Add more mappings based on training data
        }
        return field_mappings.get(repair_id, 'UNKNOWN')
    
    def _get_repair_text(self, repair_id: str) -> str:
        """Get repair description text"""
        # This would ideally come from learned patterns
        text_mappings = {
            '6021': 'ISO Country Code extracted from Field',
            '8852': 'Bank Identifier Code Added',
            # Add more mappings based on training data
        }
        return text_mappings.get(repair_id, f'Repair {repair_id} applied')


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
    train_parser.add_argument('--epochs', type=int, default=50)
    train_parser.add_argument('--incremental', action='store_true', help='Use incremental processing')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict repairs for a payment')
    predict_parser.add_argument('--input', required=True, help='Input payment JSON file')
    predict_parser.add_argument('--model_dir', default='./models')
    predict_parser.add_argument('--threshold', type=float, default=0.5)
    predict_parser.add_argument('--output', help='Output file for predictions')
    
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
            num_epochs=args.epochs,
            enable_incremental=args.incremental
        )
        
        trainer = EnhancedTrainer(config)
        trainer.train_on_directory(args.input_dir)
        
        logger.info("\nTraining complete!")
        logger.info("Check detailed_analysis.log for comprehensive analysis")
        
    elif args.command == 'predict':
        predictor = EnhancedPredictor(args.model_dir)
        results = predictor.predict(args.input, args.threshold)
        
        logger.info(f"\n{'='*70}")
        logger.info(f"PREDICTION RESULTS")
        logger.info(f"{'='*70}")
        logger.info(f"Transaction: {results['transaction_id']}")
        logger.info(f"Source: {results['source']}, Clearing: {results['clearing']}")
        
        logger.info(f"\nNeural Network Predictions ({len(results['neural_network_predictions'])}):")
        for pred in results['neural_network_predictions'][:5]:
            logger.info(f"  {pred['repair_id']}: {pred['confidence']:.2%} ({pred['confidence_level']})")
        
        logger.info(f"\nRandom Forest Predictions ({len(results['random_forest_predictions'])}):")
        for pred in results['random_forest_predictions'][:5]:
            logger.info(f"  {pred['repair_id']}: {pred['confidence']:.2%} ({pred['confidence_level']})")
        
        logger.info(f"\nConsensus Predictions ({len(results['consensus_predictions'])}):")
        for pred in results['consensus_predictions']:
            logger.info(f"  {pred['repair_id']}: NN={pred['nn_confidence']:.2%}, RF={pred['rf_confidence']:.2%}")
        
        logger.info(f"\nFinal ACE Repairs ({len(results['ace'])}):")
        for repair in results['ace']:
            logger.info(f"  ID: {repair['id']}, Confidence: {repair['confidence']:.2%}, Predicted by: {repair['predicted_by']}")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"\nResults saved to {args.output}")
    
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
        processor.print_analysis_summary()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
