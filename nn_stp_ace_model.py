"""
ace_repair_predictor.py
=======================
Production-ready ACE Payment Repair Predictor
Learns from diffs and ACE repairs to predict which repairs should be applied

Author: Production Version 4.1
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
# ACE REPAIR LEARNER (Enhanced)
# ============================================================================

def _repair_pattern_factory():
    """Factory function for repair pattern defaultdict"""
    return {
        'fields_added': Counter(),
        'fields_dropped': Counter(),
        'fields_transformed': Counter(),
        'typical_entities': Counter(),
        'total_diffs': 0,
        'descriptions': [],
        'codes': Counter(),
        'fields': Counter(),
        'co_occurring_repairs': Counter(),  # Track which repairs occur together
        'entity_field_patterns': defaultdict(Counter)  # Detailed entity.field patterns
    }


class AceRepairLearner:
    """Enhanced learner that deeply analyzes ACE repairs and their patterns"""
    
    def __init__(self):
        self.repair_patterns = defaultdict(_repair_pattern_factory)
        self.repair_taxonomy = {}  # Complete repair info
        self.repair_cooccurrence = defaultdict(Counter)  # Which repairs occur together
        self.repair_to_diffs_mapping = defaultdict(list)  # Map repairs to typical diffs
        self.entity_repair_associations = defaultdict(Counter)  # Which entities need which repairs
        self.lookup_tables = {
            'bic_to_name': {},
            'bic_to_address': {},
            'country_extractions': Counter(),
            'clearing_to_bic': {}  # Add clearing to BIC mapping
        }
        
    def learn_from_transaction(self, txn_id: str, txn_data: Dict) -> Dict:
        """Learn from both ACE repairs and diffs in a transaction"""
        
        # Extract all repairs from ace[]
        repairs_in_txn = []
        repair_details = {}
        
        for repair in txn_data.get('ace', []):
            repair_id = repair.get('id', 'unknown')
            repair_code = repair.get('code', 'I')
            repair_field = repair.get('field', '')
            repair_text = repair.get('text', '')
            
            repairs_in_txn.append(repair_id)
            repair_details[repair_id] = {
                'code': repair_code,
                'field': repair_field,
                'text': repair_text
            }
            
            # Update taxonomy
            if repair_id not in self.repair_taxonomy:
                self.repair_taxonomy[repair_id] = {
                    'code': repair_code,
                    'field': repair_field,
                    'text': repair_text,
                    'count': 0,
                    'typical_scenarios': []
                }
            
            self.repair_taxonomy[repair_id]['count'] += 1
            
            # Update patterns
            self.repair_patterns[repair_id]['codes'][repair_code] += 1
            self.repair_patterns[repair_id]['fields'][repair_field] += 1
            if repair_text not in self.repair_patterns[repair_id]['descriptions']:
                self.repair_patterns[repair_id]['descriptions'].append(repair_text)
        
        # Learn repair co-occurrence
        for i, repair1 in enumerate(repairs_in_txn):
            for repair2 in repairs_in_txn[i+1:]:
                self.repair_cooccurrence[repair1][repair2] += 1
                self.repair_cooccurrence[repair2][repair1] += 1
        
        # Process diffs and associate with repairs
        diff_features = self._analyze_diffs_with_repairs(txn_data, repairs_in_txn)
        
        # Learn lookup patterns from before/after states
        self._learn_lookup_patterns(txn_data, repairs_in_txn)
        
        # Learn repair-specific patterns based on code types
        self._learn_repair_code_patterns(repair_details, txn_data)
        
        return diff_features
    
    def _analyze_diffs_with_repairs(self, txn_data: Dict, repairs: List[str]) -> Dict:
        """Analyze diffs and associate them with specific repairs"""
        
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
            'has_clearing_changes': False,
            'specific_repairs_indicated': set()  # Which repairs are likely needed
        }
        
        entities_with_diffs = ['cdtrAgt', 'dbtrAgt', 'instgAgt', 'instdAgt',
                               'cdtr', 'dbtr', 'cdtrAcct', 'dbtrAcct', 
                               'intrmyAgt1', 'intrmyAgt2', 'rmtInf']
        
        for entity_key in entities_with_diffs:
            if entity_key not in txn_data:
                continue
            
            entity_data = txn_data[entity_key]
            if not isinstance(entity_data, dict) or 'diffs' not in entity_data:
                continue
            
            diff_features['affected_entities'].add(entity_key.lower())
            
            # Track which repairs are associated with this entity
            for repair_id in repairs:
                self.entity_repair_associations[entity_key.lower()][repair_id] += 1
            
            # Process each diff
            for diff in entity_data.get('diffs', []):
                action = diff.get('msg', '').lower()
                field_path = diff.get('key', '')
                value = diff.get('val', '')
                desc = diff.get('desc', '')
                
                # Update counts
                if action == 'added':
                    diff_features['num_additions'] += 1
                    # Check for specific repair indicators
                    if 'ctry' in field_path.lower() and value and len(value) == 2:
                        diff_features['specific_repairs_indicated'].add('6021')  # Country extraction
                    if 'bicfi' in field_path.lower():
                        diff_features['specific_repairs_indicated'].add('6035')  # BIC lookup
                    if 'nm' in field_path.lower():
                        diff_features['specific_repairs_indicated'].add('6036')  # Name lookup
                        
                elif action == 'dropped':
                    diff_features['num_drops'] += 1
                elif action in ['transformed', 'edited']:
                    diff_features['num_transformations'] += 1
                
                # Track affected fields
                field_lower = field_path.lower()
                diff_features['affected_fields'].add(field_lower)
                
                # Map this diff pattern to repairs
                for repair_id in repairs:
                    self.repair_to_diffs_mapping[repair_id].append({
                        'entity': entity_key.lower(),
                        'field': field_lower,
                        'action': action,
                        'has_value': bool(value)
                    })
                    
                    # Track entity.field patterns for each repair
                    pattern_key = f"{entity_key.lower()}.{field_lower}"
                    self.repair_patterns[repair_id]['entity_field_patterns'][pattern_key][action] += 1
                
                # Detect specific change types
                if 'bic' in field_lower:
                    diff_features['has_bic_changes'] = True
                if 'nm' in field_lower or 'name' in field_lower:
                    diff_features['has_name_changes'] = True
                if 'adr' in field_lower or 'pstl' in field_lower:
                    diff_features['has_address_changes'] = True
                if 'ctry' in field_lower or 'ctryofres' in field_lower:
                    diff_features['has_country_changes'] = True
                    if value and len(value) == 2:
                        self.lookup_tables['country_extractions'][value] += 1
                if 'clrsys' in field_lower or 'mmbid' in field_lower:
                    diff_features['has_clearing_changes'] = True
                
                # Update repair patterns
                for repair_id in repairs:
                    self.repair_patterns[repair_id]['total_diffs'] += 1
                    self.repair_patterns[repair_id]['typical_entities'][entity_key.lower()] += 1
                    
                    if action == 'added':
                        self.repair_patterns[repair_id]['fields_added'][field_lower] += 1
                    elif action == 'dropped':
                        self.repair_patterns[repair_id]['fields_dropped'][field_lower] += 1
                    elif action in ['transformed', 'edited']:
                        self.repair_patterns[repair_id]['fields_transformed'][field_lower] += 1
        
        return diff_features
    
    def _learn_lookup_patterns(self, txn_data: Dict, repair_ids: List[str]):
        """Learn lookup patterns from before/after states"""
        
        entities = ['cdtrAgt', 'dbtrAgt', 'instgAgt', 'instdAgt', 'intrmyAgt1', 'intrmyAgt2']
        
        for entity_key in entities:
            if entity_key not in txn_data:
                continue
                
            entity_data = txn_data[entity_key]
            if not isinstance(entity_data, dict):
                continue
            
            before = entity_data.get('before', {})
            after = entity_data.get('after', {})
            
            # Learn BIC lookups from clearing
            clearing = self._find_value_in_dict(before, 'mmbid') or self._find_value_in_dict(before, 'clrsysmmbid')
            bic_after = self._find_value_in_dict(after, 'bicfi') or self._find_value_in_dict(after, 'bic')
            
            if clearing and bic_after and not self._find_value_in_dict(before, 'bicfi'):
                self.lookup_tables['clearing_to_bic'][clearing] = bic_after
                # This indicates repair 6035 (BIC lookup from clearing)
                if '6035' in repair_ids:
                    self.repair_taxonomy['6035']['typical_scenarios'].append('BIC from clearing')
            
            # Learn name lookups from BIC
            bic = bic_after or self._find_value_in_dict(before, 'bicfi')
            if bic:
                name_after = self._find_value_in_dict(after, 'nm')
                name_before = self._find_value_in_dict(before, 'nm')
                
                if name_after and name_after != name_before:
                    self.lookup_tables['bic_to_name'][bic] = name_after
                    # This indicates repair 6036 (Name lookup from BIC)
                    if '6036' in repair_ids:
                        self.repair_taxonomy['6036']['typical_scenarios'].append('Name from BIC')
                
                # Learn address lookups
                addr_after = self._find_value_in_dict(after, 'adrline')
                if addr_after:
                    self.lookup_tables['bic_to_address'][bic] = addr_after
    
    def _learn_repair_code_patterns(self, repair_details: Dict, txn_data: Dict):
        """Learn patterns specific to repair codes (I/W/E/R)"""
        
        for repair_id, details in repair_details.items():
            code = details['code']
            field = details['field']
            
            # Learn patterns based on repair code type
            if code == 'I':  # Information added
                # These repairs typically add missing information
                self.repair_patterns[repair_id]['typical_action'] = 'add'
                
            elif code == 'W':  # Warning
                # These repairs indicate potential issues but don't change data
                self.repair_patterns[repair_id]['typical_action'] = 'warn'
                
            elif code == 'E':  # Error that was fixed
                # These repairs fix errors in the data
                self.repair_patterns[repair_id]['typical_action'] = 'fix'
                
            elif code == 'R':  # Rejected/removed
                # These repairs remove invalid data
                self.repair_patterns[repair_id]['typical_action'] = 'remove'
    
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
    
    def get_repair_signature(self, repair_id: str) -> Dict:
        """Get the learned signature for a repair"""
        if repair_id not in self.repair_patterns:
            return {}
        
        pattern = self.repair_patterns[repair_id]
        
        # Get most common patterns
        signature = {
            'typical_entities': pattern['typical_entities'].most_common(3),
            'typical_adds': pattern['fields_added'].most_common(3),
            'typical_transforms': pattern['fields_transformed'].most_common(3),
            'co_occurring': self.repair_cooccurrence[repair_id].most_common(3),
            'entity_field_patterns': dict(pattern['entity_field_patterns'])
        }
        
        return signature
    
    def print_summary(self):
        """Print comprehensive learning summary"""
        logger.info("\n" + "="*70)
        logger.info("ACE REPAIR LEARNING SUMMARY")
        logger.info("="*70)
        
        # Repair taxonomy
        logger.info(f"\nLearned ACE Repair Taxonomy ({len(self.repair_taxonomy)} repairs):")
        sorted_repairs = sorted(self.repair_taxonomy.items(), key=lambda x: x[1]['count'], reverse=True)
        for repair_id, info in sorted_repairs[:10]:
            logger.info(f"  {repair_id}: [{info['code']}] {info['field']} - {info['text'][:50]}...")
            logger.info(f"    Count: {info['count']}, Scenarios: {', '.join(set(info['typical_scenarios'][:3]))}")
        
        # Co-occurrence patterns
        logger.info(f"\nRepair Co-occurrence Patterns:")
        for repair_id in list(self.repair_cooccurrence.keys())[:5]:
            co_occurring = self.repair_cooccurrence[repair_id].most_common(3)
            if co_occurring:
                logger.info(f"  {repair_id} often occurs with: {co_occurring}")
        
        # Entity associations
        logger.info(f"\nEntity-Repair Associations:")
        for entity, repairs in list(self.entity_repair_associations.items())[:5]:
            top_repairs = repairs.most_common(3)
            logger.info(f"  {entity}: {top_repairs}")
        
        # Lookup tables
        logger.info(f"\nLearned Lookup Tables:")
        logger.info(f"  BIC to Name mappings: {len(self.lookup_tables['bic_to_name'])}")
        logger.info(f"  BIC to Address mappings: {len(self.lookup_tables['bic_to_address'])}")
        logger.info(f"  Clearing to BIC mappings: {len(self.lookup_tables['clearing_to_bic'])}")
        logger.info(f"  Country codes seen: {len(self.lookup_tables['country_extractions'])}")


# ============================================================================
# FEATURE EXTRACTION (Enhanced)
# ============================================================================

class FeatureExtractor:
    """Extracts features from payment structure and diffs"""
    
    def __init__(self, ace_learner: Optional[AceRepairLearner] = None):
        self.ace_learner = ace_learner
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
        
        # ACE-specific features (new)
        self.feature_names.extend([
            'needs_country_extraction',  # Missing country but has other location info
            'needs_bic_lookup',  # Has clearing but no BIC
            'needs_name_lookup',  # Has BIC but no name
            'needs_address_enrichment',  # Has partial address info
            'multiple_agents_incomplete',  # Multiple agents with missing info
            'clearing_system_present',  # Has clearing system info
            'partial_bank_info'  # Has some but not all bank info
        ])
    
    def extract_features(self, payment: Dict, diff_features: Optional[Dict] = None) -> np.ndarray:
        """Extract all features from payment and diffs"""
        features = np.zeros(len(self.feature_names))
        idx = 0
        
        # Normalize payment
        payment = self._normalize_keys(payment)
        
        # Basic structural features
        has_bic = self._has_field(payment, 'bic')
        has_iban = self._has_field(payment, 'iban')
        has_clearing = self._has_field(payment, 'mmbid')
        has_name = self._has_field(payment, 'nm')
        has_address = self._has_field(payment, 'adrline')
        has_country = self._has_field(payment, 'ctryofres') or self._has_field(payment, 'ctry')
        
        features[idx] = float(has_bic); idx += 1
        features[idx] = float(has_iban); idx += 1
        features[idx] = float(has_clearing); idx += 1
        features[idx] = float(has_name); idx += 1
        features[idx] = float(has_address); idx += 1
        features[idx] = float(has_country); idx += 1
        features[idx] = float('cdtragt' in payment); idx += 1
        features[idx] = float('dbtragt' in payment); idx += 1
        features[idx] = float('instgagt' in payment); idx += 1
        features[idx] = float('instdagt' in payment); idx += 1
        features[idx] = float('cdtr' in payment); idx += 1
        features[idx] = float('dbtr' in payment); idx += 1
        features[idx] = float('rmtinf' in payment); idx += 1
        
        # Derived features
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
        
        # ACE-specific features
        # Check if needs country extraction
        has_location_info = has_address or self._has_field(payment, 'townname')
        features[idx] = float(not has_country and has_location_info); idx += 1
        
        # Check if needs BIC lookup
        features[idx] = float(has_clearing and not has_bic); idx += 1
        
        # Check if needs name lookup
        features[idx] = float(has_bic and not has_name); idx += 1
        
        # Check if needs address enrichment
        has_partial_address = self._has_field(payment, 'townname') or self._has_field(payment, 'pstlcd')
        features[idx] = float(has_partial_address and not has_address); idx += 1
        
        # Check multiple agents incomplete
        agent_entities = ['cdtragt', 'dbtragt', 'instgagt', 'instdagt']
        incomplete_agents = 0
        for agent in agent_entities:
            if agent in payment:
                agent_data = payment[agent]
                if isinstance(agent_data, dict):
                    agent_has_bic = self._has_field(agent_data, 'bic')
                    agent_has_name = self._has_field(agent_data, 'nm')
                    if not (agent_has_bic and agent_has_name):
                        incomplete_agents += 1
        features[idx] = min(incomplete_agents / 3, 1.0); idx += 1
        
        # Has clearing system info
        features[idx] = float(has_clearing or self._has_field(payment, 'clrsys')); idx += 1
        
        # Has partial bank info
        has_some_bank_info = has_bic or has_name or has_clearing
        has_complete_bank_info = has_bic and has_name
        features[idx] = float(has_some_bank_info and not has_complete_bank_info); idx += 1
        
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
# DATA PROCESSOR (Enhanced)
# ============================================================================

class DataProcessor:
    """Process training data with enhanced ACE learning"""
    
    def __init__(self):
        self.ace_learner = AceRepairLearner()
        self.feature_extractor = FeatureExtractor(self.ace_learner)
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
        
        # First pass: Learn from ACE repairs and diffs
        logger.info("Learning from ACE repairs and diffs...")
        for txn_id, txn_data in all_transactions.items():
            if not isinstance(txn_data, dict):
                continue
            
            # Must have repairs to be valid training data
            if 'ace' not in txn_data or not txn_data['ace']:
                skipped_count += 1
                continue
            
            # Learn from transaction
            diff_features = self.ace_learner.learn_from_transaction(txn_id, txn_data)
            self.diff_features_cache[txn_id] = diff_features
            processed_count += 1
            
            if processed_count % 1000 == 0:
                logger.info(f"  Processed {processed_count} transactions...")
        
        # Print learning summary
        self.ace_learner.print_summary()
        
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
        logger.info(f"  Unique repairs: {len(self.repair_vocabulary)}")
        
        # Calculate repair frequency
        repair_freq = np.sum(labels_array, axis=0)
        logger.info(f"\nRepair frequency distribution:")
        for i, freq in enumerate(repair_freq):
            if freq > 0:
                repair_id = self.idx_to_repair[i]
                logger.info(f"  {repair_id}: {int(freq)} occurrences ({freq/len(labels_array)*100:.1f}%)")
        
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
                   'cdtr', 'dbtr', 'cdtrAcct', 'dbtrAcct', 'rmtInf',
                   'intrmyAgt1', 'intrmyAgt2']
        
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
        if len(sorted_repairs) <= 20:  # Only show if reasonable number
            logger.info(f"Repairs: {sorted_repairs}")
    
    def _repairs_to_labels(self, repairs: List[str]) -> np.ndarray:
        """Convert repair IDs to multi-hot encoding"""
        labels = np.zeros(len(self.repair_vocabulary))
        for repair_id in repairs:
            if repair_id in self.repair_vocabulary:
                labels[self.repair_vocabulary[repair_id]] = 1.0
        return labels


# ============================================================================
# NEURAL NETWORK (Enhanced with attention)
# ============================================================================

class RepairNN(nn.Module):
    """Enhanced neural network for repair prediction with attention mechanism"""
    
    def __init__(self, num_features: int, num_repairs: int, 
                 hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        
        # Feature processing layers
        self.feature_encoder = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Attention layer for feature importance
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softmax(dim=1)
        )
        
        # Main prediction network
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
        # Encode features
        encoded = self.feature_encoder(x)
        
        # Apply attention (simplified for batch processing)
        # In a more complex model, you could use multi-head attention
        
        # Make predictions
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


# ============================================================================
# MODEL TRAINER (Enhanced)
# ============================================================================

class ModelTrainer:
    """Handles model training and evaluation"""
    
    def __init__(self, config: Config):
        self.config = config
        self.processor = DataProcessor()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.rf_model = None
        logger.info(f"Using device: {self.device}")
    
    def train(self, data_path: str) -> Dict:
        """Train the model"""
        logger.info("="*70)
        logger.info("TRAINING ACE REPAIR PREDICTOR")
        logger.info("="*70)
        
        # Load and process data
        features, labels, _ = self.processor.load_and_process(data_path)
        
        # Calculate class weights for imbalanced repairs
        if self.config.use_class_weights:
            class_weights = self._calculate_class_weights(labels)
        else:
            class_weights = None
        
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
                n_estimators=150,  # Increased for better performance
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        )
        self.rf_model.fit(X_train, y_train)
        logger.info("Random Forest training complete")
        
        # Train Neural Network
        logger.info("\nTraining Neural Network...")
        self.model = RepairNN(
            num_features=features.shape[1],
            num_repairs=labels.shape[1],
            hidden_dim=self.config.hidden_dim,
            dropout=self.config.dropout
        ).to(self.device)
        
        self._train_nn(X_train, y_train, X_val, y_val, class_weights)
        
        # Evaluate on test set
        logger.info("\nEvaluating on test set...")
        metrics = self._evaluate(X_test, y_test)
        
        # Save models
        self._save_models()
        
        return metrics
    
    def _calculate_class_weights(self, labels: np.ndarray) -> torch.Tensor:
        """Calculate class weights for imbalanced data"""
        # Calculate frequency of each repair
        repair_freq = np.sum(labels, axis=0) + 1  # Add 1 to avoid division by zero
        
        # Calculate weights (inverse frequency)
        weights = len(labels) / (len(self.processor.repair_vocabulary) * repair_freq)
        
        # Cap weights to prevent extreme values
        weights = np.minimum(weights, self.config.max_class_weight)
        
        logger.info(f"Class weights calculated (min: {weights.min():.2f}, max: {weights.max():.2f})")
        
        return torch.FloatTensor(weights).to(self.device)
    
    def _train_nn(self, X_train, y_train, X_val, y_val, class_weights=None):
        """Train neural network with improvements"""
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
        
        # Use AdamW with weight decay for better generalization
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.config.learning_rate,
            weight_decay=0.01
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Loss function with optional class weights
        if class_weights is not None:
            criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
        else:
            criterion = nn.BCELoss()
        
        best_val_loss = float('inf')
        patience = 15
        patience_counter = 0
        
        for epoch in range(1, self.config.num_epochs + 1):
            # Training
            self.model.train()
            train_loss = 0.0
            train_acc = 0.0
            
            for features, labels in train_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                
                # Calculate accuracy
                preds = (outputs > 0.5).float()
                train_acc += (preds == labels).float().mean().item()
            
            train_loss /= len(train_loader)
            train_acc /= len(train_loader)
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            val_acc = 0.0
            
            with torch.no_grad():
                for features, labels in val_loader:
                    features = features.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(features)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    preds = (outputs > 0.5).float()
                    val_acc += (preds == labels).float().mean().item()
            
            val_loss /= len(val_loader)
            val_acc /= len(val_loader)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Logging
            if epoch % 5 == 0 or epoch == 1:
                logger.info(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                          f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
            
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
        
        # Also get Random Forest predictions for comparison
        rf_preds = self.rf_model.predict(X_test)
        
        # Binary predictions
        nn_preds_binary = (preds > 0.5).astype(int)
        
        # Calculate metrics
        exact_match_nn = np.mean(np.all(nn_preds_binary == labels, axis=1))
        exact_match_rf = np.mean(np.all(rf_preds == labels, axis=1))
        
        hamming_nn = np.mean(nn_preds_binary == labels)
        hamming_rf = np.mean(rf_preds == labels)
        
        logger.info(f"\nModel Performance Comparison:")
        logger.info(f"  Neural Network:")
        logger.info(f"    Exact Match: {exact_match_nn:.2%}")
        logger.info(f"    Hamming Accuracy: {hamming_nn:.2%}")
        logger.info(f"  Random Forest:")
        logger.info(f"    Exact Match: {exact_match_rf:.2%}")
        logger.info(f"    Hamming Accuracy: {hamming_rf:.2%}")
        
        # Per-repair accuracy
        logger.info(f"\nPer-repair Accuracy (Neural Network):")
        per_repair_acc = []
        for i in range(labels.shape[1]):
            if np.sum(labels[:, i]) > 0:  # Only evaluate repairs that appear in test set
                acc = np.mean(nn_preds_binary[:, i] == labels[:, i])
                per_repair_acc.append(acc)
                repair_id = self.processor.idx_to_repair.get(i, f"repair_{i}")
                support = int(np.sum(labels[:, i]))
                logger.info(f"  {repair_id}: {acc:.2%} (support: {support})")
        
        metrics = {
            'exact_match_accuracy': exact_match_nn,
            'hamming_accuracy': hamming_nn,
            'mean_repair_accuracy': np.mean(per_repair_acc) if per_repair_acc else 0,
            'rf_exact_match': exact_match_rf,
            'rf_hamming': hamming_rf
        }
        
        logger.info(f"\nOverall Metrics:")
        logger.info(f"  Best Exact Match: {max(exact_match_nn, exact_match_rf):.2%}")
        logger.info(f"  Best Hamming: {max(hamming_nn, hamming_rf):.2%}")
        logger.info(f"  Mean Repair Accuracy: {metrics['mean_repair_accuracy']:.2%}")
        
        return metrics
    
    def _save_models(self):
        """Save trained models"""
        os.makedirs(self.config.model_dir, exist_ok=True)
        
        # Save config
        self.config.save(os.path.join(self.config.model_dir, 'config.json'))
        
        # Save processor (includes ACE learner)
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
# PREDICTOR (Enhanced)
# ============================================================================

class RepairPredictor:
    """Make predictions on new payments with enhanced ACE output"""
    
    def __init__(self, model_dir: str = './models'):
        self.model_dir = model_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load saved components
        logger.info(f"Loading models from {model_dir}")
        
        # Load config
        self.config = Config.load(os.path.join(model_dir, 'config.json'))
        
        # Load processor (includes ACE learner)
        with open(os.path.join(model_dir, 'processor.pkl'), 'rb') as f:
            self.processor = pickle.load(f)
        
        # Load Random Forest
        with open(os.path.join(model_dir, 'rf_model.pkl'), 'rb') as f:
            self.rf_model = pickle.load(f)
        
        # Load Neural Network
        self.nn_model = RepairNN(
            num_features=len(self.processor.feature_extractor.feature_names),
            num_repairs=len(self.processor.repair_vocabulary),
            hidden_dim=self.config.hidden_dim,
            dropout=self.config.dropout
        ).to(self.device)
        
        self.nn_model.load_state_dict(
            torch.load(os.path.join(model_dir, 'neural_model.pt'), 
                      map_location=self.device)
        )
        self.nn_model.eval()
        
        logger.info("Models loaded successfully")
    
    def predict(self, payment_file: str, threshold: float = 0.5, 
                use_ensemble: bool = True) -> Dict:
        """Predict repairs for a payment with enhanced ACE output"""
        
        # Load payment
        with open(payment_file, 'r') as f:
            data = json.load(f)
        
        # Extract payment (handle wrapper format)
        if isinstance(data, dict) and len(data) == 1:
            txn_id = list(data.keys())[0]
            payment = data[txn_id]
            logger.info(f"Processing transaction: {txn_id}")
        else:
            payment = data
        
        # Extract features (no diff features for prediction)
        features = self.processor.feature_extractor.extract_features(payment, None)
        
        # Get predictions from Neural Network
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        with torch.no_grad():
            nn_probs = self.nn_model(features_tensor)[0].cpu().numpy()
        
        # Get predictions from Random Forest
        rf_probs = self.rf_model.predict_proba(features.reshape(1, -1))
        rf_probs = np.array([p[0, 1] if p.shape[1] > 1 else p[0, 0] for p in rf_probs])
        
        # Ensemble or single model
        if use_ensemble:
            # Weighted ensemble (NN gets slightly more weight due to better performance)
            ensemble_probs = (nn_probs * 0.6 + rf_probs * 0.4)
        else:
            ensemble_probs = nn_probs
        
        # Apply co-occurrence boost
        boosted_probs = self._apply_cooccurrence_boost(ensemble_probs)
        
        # Get predicted repairs
        predicted_repairs = []
        ace_predictions = []  # Full ACE array format
        
        for idx, prob in enumerate(boosted_probs):
            if prob > threshold:
                repair_id = self.processor.idx_to_repair[idx]
                
                # Get repair information from learned taxonomy
                repair_info = self._get_repair_info(repair_id)
                
                predicted_repairs.append({
                    'repair_id': repair_id,
                    'confidence': float(prob),
                    'nn_prob': float(nn_probs[idx]),
                    'rf_prob': float(rf_probs[idx]),
                    'boosted': prob != ensemble_probs[idx]  # Was this boosted?
                })
                
                # Build ACE format entry
                ace_predictions.append({
                    'id': repair_id,
                    'code': repair_info['code'],
                    'field': repair_info['field'],
                    'text': repair_info['text']
                })
        
        # Sort by confidence
        predicted_repairs.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Sort ACE predictions by repair ID for consistency
        ace_predictions.sort(key=lambda x: x['id'])
        
        # Analyze what needs to be repaired based on features
        repair_analysis = self._analyze_repair_needs(payment, features, predicted_repairs)
        
        # Get repair signatures for predicted repairs
        repair_signatures = {}
        for repair in predicted_repairs:
            repair_id = repair['repair_id']
            if hasattr(self.processor.ace_learner, 'get_repair_signature'):
                repair_signatures[repair_id] = self.processor.ace_learner.get_repair_signature(repair_id)
        
        return {
            'predicted_repairs': predicted_repairs,
            'repair_ids': [r['repair_id'] for r in predicted_repairs],
            'ace': ace_predictions,  # Full ACE array with messages
            'repair_analysis': repair_analysis,
            'feature_analysis': self._get_feature_analysis(features),
            'repair_signatures': repair_signatures,  # Learned patterns for each repair
            'confidence_summary': {
                'high_confidence': [r for r in predicted_repairs if r['confidence'] > 0.8],
                'medium_confidence': [r for r in predicted_repairs if 0.5 < r['confidence'] <= 0.8],
                'total_repairs': len(predicted_repairs)
            }
        }
    
    def _apply_cooccurrence_boost(self, probs: np.ndarray) -> np.ndarray:
        """Boost probabilities based on co-occurrence patterns"""
        boosted_probs = probs.copy()
        
        # Check if we have co-occurrence data
        if not hasattr(self.processor.ace_learner, 'repair_cooccurrence'):
            return boosted_probs
        
        # For each repair with high probability, boost co-occurring repairs
        high_prob_repairs = np.where(probs > 0.7)[0]
        
        for idx in high_prob_repairs:
            repair_id = self.processor.idx_to_repair[idx]
            
            # Get co-occurring repairs
            if repair_id in self.processor.ace_learner.repair_cooccurrence:
                for co_repair, count in self.processor.ace_learner.repair_cooccurrence[repair_id].items():
                    if co_repair in self.processor.repair_vocabulary:
                        co_idx = self.processor.repair_vocabulary[co_repair]
                        # Small boost based on co-occurrence strength
                        boost = min(0.1, count / 100)  # Max 10% boost
                        boosted_probs[co_idx] = min(1.0, boosted_probs[co_idx] + boost)
        
        return boosted_probs
    
    def _get_repair_info(self, repair_id: str) -> Dict:
        """Get repair information from learned taxonomy"""
        # Try to get from learned taxonomy
        if hasattr(self.processor.ace_learner, 'repair_taxonomy'):
            if repair_id in self.processor.ace_learner.repair_taxonomy:
                return self.processor.ace_learner.repair_taxonomy[repair_id]
        
        # Fallback to default based on common patterns
        default_repairs = {
            '6021': {
                'code': 'I',
                'field': 'CDTPTY',
                'text': 'ISO Country Code extracted from Field'
            },
            '6035': {
                'code': 'I',
                'field': 'BICFI',
                'text': 'BIC looked up from Clearing System ID'
            },
            '6036': {
                'code': 'I',
                'field': 'NM',
                'text': 'Bank Name looked up from BIC'
            },
            '6037': {
                'code': 'I',
                'field': 'PSTLADR',
                'text': 'Address looked up from BIC'
            }
        }
        
        if repair_id in default_repairs:
            return default_repairs[repair_id]
        
        # Generic fallback
        return {
            'code': 'I',
            'field': 'UNKNOWN',
            'text': f'Repair {repair_id} applied'
        }
    
    def _analyze_repair_needs(self, payment: Dict, features: np.ndarray, 
                              predicted_repairs: List[Dict]) -> Dict:
        """Comprehensive analysis of why repairs are needed"""
        
        analysis = {
            'missing_fields': [],
            'incomplete_entities': [],
            'suggested_lookups': [],
            'repair_reasoning': {}
        }
        
        feature_names = self.processor.feature_extractor.feature_names
        
        # Check for missing critical fields
        critical_fields = {
            'has_bic': 'BIC code',
            'has_country': 'Country code',
            'has_bank_name': 'Bank name',
            'has_address': 'Address'
        }
        
        for feat_name, field_desc in critical_fields.items():
            if feat_name in feature_names:
                idx = feature_names.index(feat_name)
                if features[idx] < 0.5:
                    analysis['missing_fields'].append(field_desc)
        
        # Check for incomplete entities
        entities = ['cdtrAgt', 'dbtrAgt', 'instgAgt', 'instdAgt']
        for entity in entities:
            entity_feature = f'has_{entity.lower()}'
            if entity_feature in feature_names:
                idx = feature_names.index(entity_feature)
                if features[idx] > 0.5:
                    # Entity exists, check if complete
                    if self._is_entity_incomplete(payment, entity):
                        analysis['incomplete_entities'].append(entity)
        
        # Suggest lookups based on available data
        if 'missing_bic_has_clearing' in feature_names:
            idx = feature_names.index('missing_bic_has_clearing')
            if features[idx] > 0.5:
                analysis['suggested_lookups'].append('BIC from Clearing System ID')
        
        if 'missing_name_has_bic' in feature_names:
            idx = feature_names.index('missing_name_has_bic')
            if features[idx] > 0.5:
                analysis['suggested_lookups'].append('Bank Name from BIC')
        
        # Explain each predicted repair
        for repair in predicted_repairs:
            repair_id = repair['repair_id']
            repair_info = self._get_repair_info(repair_id)
            
            reasoning = {
                'action': repair_info['code'],
                'field': repair_info['field'],
                'description': repair_info['text'],
                'confidence': repair['confidence']
            }
            
            # Add specific reasoning based on repair type
            if '6021' in repair_id:
                reasoning['reason'] = 'Missing country code but location information available'
            elif '6035' in repair_id:
                reasoning['reason'] = 'BIC can be determined from clearing system information'
            elif '6036' in repair_id:
                reasoning['reason'] = 'Bank name can be looked up from BIC'
            elif '6037' in repair_id:
                reasoning['reason'] = 'Address can be enriched from BIC'
            
            analysis['repair_reasoning'][repair_id] = reasoning
        
        return analysis
    
    def _is_entity_incomplete(self, payment: Dict, entity: str) -> bool:
        """Check if an entity is incomplete"""
        if entity.lower() not in payment:
            return False
        
        entity_data = payment[entity.lower()]
        if not isinstance(entity_data, dict):
            return False
        
        # Check for key fields
        has_bic = self._has_field_in_dict(entity_data, 'bic')
        has_name = self._has_field_in_dict(entity_data, 'nm')
        
        return not (has_bic and has_name)
    
    def _has_field_in_dict(self, d: Dict, field: str) -> bool:
        """Check if field exists in dictionary"""
        if not isinstance(d, dict):
            return False
        
        field = field.lower()
        for k, v in d.items():
            if field in str(k).lower():
                return True
            if isinstance(v, dict) and self._has_field_in_dict(v, field):
                return True
        return False
    
    def _get_feature_analysis(self, features: np.ndarray) -> Dict:
        """Get human-readable feature analysis"""
        feature_names = self.processor.feature_extractor.feature_names
        
        analysis = {
            'entities_present': [],
            'fields_present': [],
            'fields_missing': [],
            'ace_indicators': []
        }
        
        # Check which entities are present
        entity_features = {
            'has_cdtrAgt': 'Creditor Agent',
            'has_dbtrAgt': 'Debtor Agent', 
            'has_instgAgt': 'Instructing Agent',
            'has_instdAgt': 'Instructed Agent',
            'has_cdtr': 'Creditor',
            'has_dbtr': 'Debtor'
        }
        
        for feat_name, entity_name in entity_features.items():
            if feat_name in feature_names:
                idx = feature_names.index(feat_name)
                if features[idx] > 0.5:
                    analysis['entities_present'].append(entity_name)
        
        # Check which fields are present/missing
        field_features = {
            'has_bic': 'BIC',
            'has_iban': 'IBAN',
            'has_clearing_id': 'Clearing ID',
            'has_bank_name': 'Bank Name',
            'has_address': 'Address',
            'has_country': 'Country Code'
        }
        
        for feat_name, field_name in field_features.items():
            if feat_name in feature_names:
                idx = feature_names.index(feat_name)
                if features[idx] > 0.5:
                    analysis['fields_present'].append(field_name)
                else:
                    analysis['fields_missing'].append(field_name)
        
        # Check ACE-specific indicators
        ace_features = {
            'needs_country_extraction': 'Needs country extraction',
            'needs_bic_lookup': 'Needs BIC lookup',
            'needs_name_lookup': 'Needs name lookup',
            'needs_address_enrichment': 'Needs address enrichment'
        }
        
        for feat_name, description in ace_features.items():
            if feat_name in feature_names:
                idx = feature_names.index(feat_name)
                if features[idx] > 0.5:
                    analysis['ace_indicators'].append(description)
        
        return analysis


# ============================================================================
# CLI (Enhanced)
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='ACE Payment Repair Predictor - Enhanced with Deep ACE Learning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  Train a model:
    python ace_repair_predictor.py train --input training_data.json
  
  Predict repairs:
    python ace_repair_predictor.py predict --input payment.json
  
  Predict with custom threshold:
    python ace_repair_predictor.py predict --input payment.json --threshold 0.3
        '''
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--input', required=True, help='Training data file')
    train_parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    train_parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    train_parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    train_parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    train_parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    train_parser.add_argument('--model_dir', default='./models', help='Model directory')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict repairs')
    predict_parser.add_argument('--input', required=True, help='Input payment file')
    predict_parser.add_argument('--model', default='./models', help='Model directory')
    predict_parser.add_argument('--threshold', type=float, default=0.5, 
                               help='Confidence threshold for predictions')
    predict_parser.add_argument('--output', help='Output file for results (optional)')
    predict_parser.add_argument('--no-ensemble', action='store_true', 
                               help='Use only neural network (no ensemble)')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        config = Config()
        if args.epochs:
            config.num_epochs = args.epochs
        if args.batch_size:
            config.batch_size = args.batch_size
        if args.hidden_dim:
            config.hidden_dim = args.hidden_dim
        if args.dropout:
            config.dropout = args.dropout
        if args.lr:
            config.learning_rate = args.lr
        if args.model_dir:
            config.model_dir = args.model_dir
        
        trainer = ModelTrainer(config)
        metrics = trainer.train(args.input)
        
        logger.info("\n" + "="*70)
        logger.info("TRAINING COMPLETE")
        logger.info("="*70)
        logger.info(f"Final metrics: {metrics}")
    
    elif args.command == 'predict':
        predictor = RepairPredictor(args.model)
        results = predictor.predict(
            args.input, 
            args.threshold,
            use_ensemble=not args.no_ensemble
        )
        
        # Display results
        logger.info("\n" + "="*70)
        logger.info("ACE REPAIR PREDICTIONS")
        logger.info("="*70)
        
        # Summary
        summary = results['confidence_summary']
        logger.info(f"\nPredicted {summary['total_repairs']} repairs:")
        logger.info(f"  High confidence (>80%): {len(summary['high_confidence'])}")
        logger.info(f"  Medium confidence (50-80%): {len(summary['medium_confidence'])}")
        
        # Detailed predictions
        if results['predicted_repairs']:
            logger.info(f"\nDetailed Predictions:")
            for repair in results['predicted_repairs']:
                logger.info(f"\n  Repair ID: {repair['repair_id']}")
                logger.info(f"    Confidence: {repair['confidence']:.2%}")
                if repair.get('boosted'):
                    logger.info(f"    (Boosted by co-occurrence patterns)")
                logger.info(f"    NN Probability: {repair['nn_prob']:.2%}")
                logger.info(f"    RF Probability: {repair['rf_prob']:.2%}")
                
                # Show reasoning if available
                repair_id = repair['repair_id']
                if repair_id in results['repair_analysis']['repair_reasoning']:
                    reasoning = results['repair_analysis']['repair_reasoning'][repair_id]
                    logger.info(f"    Action: [{reasoning['action']}] {reasoning['field']}")
                    logger.info(f"    Description: {reasoning['description']}")
                    if 'reason' in reasoning:
                        logger.info(f"    Reason: {reasoning['reason']}")
        else:
            logger.info("\nNo repairs predicted (all probabilities below threshold)")
        
        # ACE array output
        logger.info(f"\nACE Array Output ({len(results['ace'])} repairs):")
        for ace_entry in results['ace']:
            logger.info(f"  [{ace_entry['code']}] {ace_entry['id']}: {ace_entry['field']} - {ace_entry['text'][:60]}...")
        
        # Payment analysis
        analysis = results['feature_analysis']
        logger.info(f"\nPayment Analysis:")
        logger.info(f"  Entities present: {', '.join(analysis['entities_present']) or 'None'}")
        logger.info(f"  Fields present: {', '.join(analysis['fields_present']) or 'None'}")
        logger.info(f"  Fields missing: {', '.join(analysis['fields_missing']) or 'None'}")
        
        if analysis['ace_indicators']:
            logger.info(f"  ACE indicators: {', '.join(analysis['ace_indicators'])}")
        
        # Repair analysis
        repair_analysis = results['repair_analysis']
        if repair_analysis['suggested_lookups']:
            logger.info(f"\nSuggested lookups:")
            for lookup in repair_analysis['suggested_lookups']:
                logger.info(f"  - {lookup}")
        
        if repair_analysis['incomplete_entities']:
            logger.info(f"\nIncomplete entities: {', '.join(repair_analysis['incomplete_entities'])}")
        
        # Save results if output file specified
        if args.output:
            # Ensure repair_signatures are serializable
            if 'repair_signatures' in results:
                for repair_id, signature in results['repair_signatures'].items():
                    # Convert defaultdict to regular dict for JSON serialization
                    if 'entity_field_patterns' in signature:
                        signature['entity_field_patterns'] = dict(signature['entity_field_patterns'])
            
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"\nResults saved to {args.output}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
