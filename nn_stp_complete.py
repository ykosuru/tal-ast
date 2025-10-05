#!/usr/bin/env python3
"""
ace_repair_predictor_complete.py
=================================
Complete ACE Payment Repair Predictor with Structural Change Detection
Learns from diffs, ACE repairs, and ALL structural changes to predict repairs

Version: 5.0 - Production Ready
Features:
- Learns from explicit diffs[] array
- Learns from ace[] repairs  
- Detects ALL structural changes between before/after states
- Comprehensive logging of detected patterns
- Predicts repairs for new payments
"""

import json
import argparse
import os
import pickle
import logging
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
from pathlib import Path
import glob
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

# Main logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Structural changes logger (separate file)
structural_logger = logging.getLogger('structural_changes')
structural_logger.setLevel(logging.DEBUG)

# File handler for structural changes
fh = logging.FileHandler('structural_changes.log', mode='w')
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
structural_logger.addHandler(fh)

# Console handler for important structural changes
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
structural_logger.addHandler(ch)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def safe_get_repair_id(repair_data: Any) -> str:
    """Safely extract repair ID from potentially complex data structure"""
    if isinstance(repair_data, str):
        return repair_data
    elif isinstance(repair_data, (int, float)):
        return str(repair_data)
    elif isinstance(repair_data, dict):
        # Try common patterns for nested IDs
        for key in ['id', 'value', 'repair_id', 'ID']:
            if key in repair_data:
                value = repair_data[key]
                if isinstance(value, (str, int, float)):
                    return str(value)
                elif isinstance(value, dict):
                    # Recursively try to extract
                    return safe_get_repair_id(value)
        # If no valid ID found, create hashable representation
        return str(hash(frozenset(repair_data.items())))
    else:
        return str(repair_data)


def ensure_hashable(value: Any) -> str:
    """Ensure a value is hashable by converting to string"""
    if isinstance(value, (str, int, float)):
        return str(value)
    elif isinstance(value, dict):
        return str(sorted(value.items()))
    elif isinstance(value, (list, tuple)):
        return str(value)
    else:
        return str(value)


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
# STRUCTURAL CHANGE DETECTION
# ============================================================================

@dataclass
class StructuralChange:
    """Represents a structural change in the payment data"""
    entity: str
    change_type: str  # 'added', 'removed', 'moved', 'restructured', 'type_changed', 'value_changed'
    field_path: str
    old_path: Optional[str] = None
    old_value: Optional[Any] = None
    new_value: Optional[Any] = None
    details: Optional[Dict] = None


class StructuralChangeDetector:
    """Detects ALL structural changes between before and after states"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.change_patterns = defaultdict(Counter)
        self.field_movements = defaultdict(list)
        self.structural_patterns = defaultdict(set)
        self.all_detected_changes = []
        self.transaction_count = 0
        
    def detect_all_changes(self, before: Dict, after: Dict, entity_name: str, 
                          txn_id: str = None) -> List[StructuralChange]:
        """Detect all structural changes between before and after states"""
        
        self.transaction_count += 1
        changes = []
        
        if self.verbose:
            structural_logger.info(f"\n{'='*70}")
            structural_logger.info(f"Analyzing {entity_name} (Transaction: {txn_id or 'unknown'})")
        
        # Get complete field inventories
        before_fields = self._get_field_inventory(before)
        after_fields = self._get_field_inventory(after)
        
        # 1. Find added fields
        added_fields = after_fields.keys() - before_fields.keys()
        for field_path in added_fields:
            value = after_fields[field_path]['value']
            changes.append(StructuralChange(
                entity=entity_name,
                change_type='added',
                field_path=field_path,
                new_value=value
            ))
            
            # Log special additions
            if 'othr' in field_path.lower():
                structural_logger.info(f"  ⚠️ 'othr' structure added: {field_path}")
            if 'clrsysid.mmbid' in field_path.lower():
                structural_logger.info(f"  ⚠️ MmbId added inside ClrSysId: {field_path}")
        
        # 2. Find removed fields
        removed_fields = before_fields.keys() - after_fields.keys()
        for field_path in removed_fields:
            value = before_fields[field_path]['value']
            changes.append(StructuralChange(
                entity=entity_name,
                change_type='removed',
                field_path=field_path,
                old_value=value
            ))
        
        # 3. Find moved fields
        moved_changes = self._detect_field_movements(before_fields, after_fields, entity_name)
        for change in moved_changes:
            changes.append(change)
            if 'mmbid' in change.old_path.lower() and 'clrsysid' in change.field_path.lower():
                structural_logger.info(f"  ⚠️ MmbId moved into ClrSysId: {change.old_path} -> {change.field_path}")
        
        # 4. Find value changes
        common_fields = before_fields.keys() & after_fields.keys()
        for field_path in common_fields:
            before_val = before_fields[field_path]['value']
            after_val = after_fields[field_path]['value']
            
            if isinstance(before_val, (str, int, float, bool)) and isinstance(after_val, (str, int, float, bool)):
                if before_val != after_val:
                    changes.append(StructuralChange(
                        entity=entity_name,
                        change_type='value_changed',
                        field_path=field_path,
                        old_value=before_val,
                        new_value=after_val
                    ))
        
        # 5. Find type changes
        for field_path in common_fields:
            before_type = before_fields[field_path]['type']
            after_type = after_fields[field_path]['type']
            if before_type != after_type:
                changes.append(StructuralChange(
                    entity=entity_name,
                    change_type='type_changed',
                    field_path=field_path,
                    old_value=before_fields[field_path]['value'],
                    new_value=after_fields[field_path]['value'],
                    details={'old_type': before_type, 'new_type': after_type}
                ))
        
        # 6. Detect restructuring patterns
        restructured = self._detect_restructuring(before, after, entity_name)
        changes.extend(restructured)
        
        if self.verbose and changes:
            structural_logger.info(f"  Total changes detected: {len(changes)}")
            structural_logger.info(f"    Added: {len([c for c in changes if c.change_type == 'added'])}")
            structural_logger.info(f"    Removed: {len([c for c in changes if c.change_type == 'removed'])}")
            structural_logger.info(f"    Moved: {len([c for c in changes if c.change_type == 'moved'])}")
        
        self.all_detected_changes.extend(changes)
        return changes
    
    def _get_field_inventory(self, obj: Any, path: str = "") -> Dict[str, Dict]:
        """Get complete inventory of all fields with their paths and values"""
        inventory = {}
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                current_path = f"{path}.{key}" if path else key
                
                inventory[current_path] = {
                    'value': value,
                    'type': type(value).__name__,
                    'depth': current_path.count('.')
                }
                
                # Recursively process nested structures
                if isinstance(value, (dict, list)):
                    nested_inventory = self._get_field_inventory(value, current_path)
                    inventory.update(nested_inventory)
                    
        elif isinstance(obj, list):
            for idx, item in enumerate(obj):
                current_path = f"{path}[{idx}]"
                inventory[current_path] = {
                    'value': item,
                    'type': type(item).__name__,
                    'depth': path.count('.') + 1
                }
                
                if isinstance(item, (dict, list)):
                    nested_inventory = self._get_field_inventory(item, current_path)
                    inventory.update(nested_inventory)
        
        return inventory
    
    def _detect_field_movements(self, before_fields: Dict, after_fields: Dict, 
                                entity_name: str) -> List[StructuralChange]:
        """Detect fields that moved to different locations"""
        changes = []
        
        # Create value-to-path mappings
        before_values = {}
        for path, info in before_fields.items():
            value_str = self._serialize_value(info['value'])
            if value_str and len(value_str) > 3:
                if value_str not in before_values:
                    before_values[value_str] = []
                before_values[value_str].append(path)
        
        after_values = {}
        for path, info in after_fields.items():
            value_str = self._serialize_value(info['value'])
            if value_str and len(value_str) > 3:
                if value_str not in after_values:
                    after_values[value_str] = []
                after_values[value_str].append(path)
        
        # Find moved values
        for value_str, before_paths in before_values.items():
            if value_str in after_values:
                after_paths = after_values[value_str]
                
                for before_path in before_paths:
                    for after_path in after_paths:
                        if self._is_moved_field(before_path, after_path):
                            changes.append(StructuralChange(
                                entity=entity_name,
                                change_type='moved',
                                field_path=after_path,
                                old_path=before_path,
                                new_value=value_str[:100],
                                details={
                                    'depth_change': after_path.count('.') - before_path.count('.'),
                                    'parent_changed': self._get_parent_path(before_path) != self._get_parent_path(after_path)
                                }
                            ))
                            
                            movement_pattern = f"{self._get_field_name(before_path)} -> {self._get_field_name(after_path)}"
                            self.field_movements[entity_name].append(movement_pattern)
        
        return changes
    
    def _detect_restructuring(self, before: Dict, after: Dict, entity_name: str) -> List[StructuralChange]:
        """Detect structural reorganizations"""
        changes = []
        
        restructure_patterns = [
            {
                'name': 'MmbId_to_ClrSysId',
                'before_check': lambda b: self._path_exists(b, 'finInstnId.mmbId') or self._path_exists(b, 'mmbId'),
                'after_check': lambda a: self._path_exists(a, 'finInstnId.clrSysId.mmbId') or self._path_exists(a, 'clrSysId.mmbId'),
                'description': 'MmbId moved inside ClrSysId structure'
            },
            {
                'name': 'Othr_Addition',
                'before_check': lambda b: not self._path_exists(b, 'finInstnId.othr') and not self._path_exists(b, 'othr'),
                'after_check': lambda a: self._path_exists(a, 'finInstnId.othr') or self._path_exists(a, 'othr'),
                'description': 'Other identifier (othr) structure added'
            },
            {
                'name': 'ClrSys_Creation',
                'before_check': lambda b: not self._path_exists(b, 'finInstnId.clrSysId') and not self._path_exists(b, 'clrSysId'),
                'after_check': lambda a: self._path_exists(a, 'finInstnId.clrSysId') or self._path_exists(a, 'clrSysId'),
                'description': 'Clearing System ID structure created'
            }
        ]
        
        for pattern in restructure_patterns:
            try:
                before_matches = pattern['before_check'](before) if before else False
                after_matches = pattern['after_check'](after) if after else False
                
                if not before_matches and after_matches:
                    changes.append(StructuralChange(
                        entity=entity_name,
                        change_type='restructured',
                        field_path=entity_name,
                        details={'pattern': pattern['description']}
                    ))
                    self.structural_patterns[entity_name].add(pattern['description'])
                    
            except Exception:
                continue
        
        return changes
    
    def _serialize_value(self, value: Any) -> Optional[str]:
        """Convert value to string for comparison"""
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return str(value)
        if isinstance(value, dict):
            if len(value) == 0:
                return None
            if len(value) <= 3:
                return json.dumps(value, sort_keys=True, default=str)
        if isinstance(value, list):
            if len(value) == 0:
                return None
            if len(value) == 1:
                return self._serialize_value(value[0])
        return None
    
    def _is_moved_field(self, before_path: str, after_path: str) -> bool:
        """Check if a field has moved to a different location"""
        if before_path == after_path:
            return False
        
        before_field = self._get_field_name(before_path)
        after_field = self._get_field_name(after_path)
        
        if before_field.lower() == after_field.lower():
            return True
        
        before_parent = self._get_parent_path(before_path)
        after_parent = self._get_parent_path(after_path)
        
        return before_parent != after_parent and before_field.lower() in after_field.lower()
    
    def _get_field_name(self, path: str) -> str:
        """Get the field name from a path"""
        path = path.replace(']', '').replace('[', '.')
        parts = path.split('.')
        return parts[-1] if parts else ''
    
    def _get_parent_path(self, path: str) -> str:
        """Get the parent path"""
        path = path.replace(']', '').replace('[', '.')
        parts = path.split('.')
        return '.'.join(parts[:-1]) if len(parts) > 1 else ''
    
    def _path_exists(self, obj: Dict, path: str) -> bool:
        """Check if a path exists in the object"""
        if not obj:
            return False
        parts = path.split('.')
        current = obj
        for part in parts:
            part_lower = part.lower()
            if isinstance(current, dict):
                found = False
                for key in current.keys():
                    if key.lower() == part_lower:
                        current = current[key]
                        found = True
                        break
                if not found:
                    return False
            else:
                return False
        return True
    
    def learn_patterns(self, changes: List[StructuralChange], repair_ids: List[str]):
        """Learn which structural changes correlate with which repairs"""
        for change in changes:
            pattern_key = f"{change.change_type}:{change.field_path}"
            for repair_id in repair_ids:
                self.change_patterns[pattern_key][repair_id] += 1
    
    def get_change_signature(self, changes: List[StructuralChange]) -> Dict:
        """Get a signature of the structural changes"""
        return {
            'total_changes': len(changes),
            'additions': sum(1 for c in changes if c.change_type == 'added'),
            'removals': sum(1 for c in changes if c.change_type == 'removed'),
            'movements': sum(1 for c in changes if c.change_type == 'moved'),
            'restructuring': sum(1 for c in changes if c.change_type == 'restructured'),
            'type_changes': sum(1 for c in changes if c.change_type == 'type_changed'),
            'value_changes': sum(1 for c in changes if c.change_type == 'value_changed')
        }
    
    def print_summary(self):
        """Print comprehensive summary of learned patterns"""
        structural_logger.info(f"\n{'='*70}")
        structural_logger.info("STRUCTURAL LEARNING SUMMARY")
        structural_logger.info(f"{'='*70}")
        structural_logger.info(f"Analyzed {self.transaction_count} transactions")
        structural_logger.info(f"Total changes detected: {len(self.all_detected_changes)}")
        
        # Change type distribution
        change_counts = Counter(c.change_type for c in self.all_detected_changes)
        structural_logger.info("\nChange Type Distribution:")
        for change_type, count in change_counts.most_common():
            structural_logger.info(f"  {change_type}: {count}")
        
        # Movement patterns
        if self.field_movements:
            structural_logger.info("\nField Movement Patterns:")
            for entity, movements in list(self.field_movements.items())[:5]:
                movement_counts = Counter(movements)
                structural_logger.info(f"  {entity}:")
                for pattern, count in movement_counts.most_common(3):
                    structural_logger.info(f"    {pattern}: {count} times")
        
        # Top correlations
        structural_logger.info("\nTop Change-to-Repair Correlations:")
        top_correlations = []
        for pattern_key, repair_counts in self.change_patterns.items():
            for repair_id, count in repair_counts.items():
                if count >= 2:
                    top_correlations.append((pattern_key[:60], repair_id, count))
        
        top_correlations.sort(key=lambda x: x[2], reverse=True)
        for pattern, repair, count in top_correlations[:10]:
            structural_logger.info(f"  {pattern} -> {repair}: {count} times")


# ============================================================================
# REPAIR PATTERN FACTORY
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
        'co_occurring_repairs': Counter(),
        'entity_field_patterns': defaultdict(Counter),
        'structural_changes': Counter()  # Track structural changes
    }


# ============================================================================
# ACE REPAIR LEARNER (Enhanced with Structural Detection)
# ============================================================================

class AceRepairLearner:
    """Enhanced learner that analyzes ACE repairs, diffs, and structural changes"""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.repair_patterns = defaultdict(_repair_pattern_factory)
        self.repair_taxonomy = {}
        self.repair_cooccurrence = defaultdict(Counter)
        self.repair_to_diffs_mapping = defaultdict(list)
        self.entity_repair_associations = defaultdict(Counter)
        self.lookup_tables = {
            'bic_to_name': {},
            'bic_to_address': {},
            'country_extractions': Counter(),
            'clearing_to_bic': {}
        }
        
        # Structural detection
        self.structural_detector = StructuralChangeDetector(verbose=self.config.verbose_logging)
        self.repair_to_structural_patterns = defaultdict(list)
        self.structural_features_cache = {}
        
    def learn_from_transaction(self, txn_id: str, txn_data: Dict) -> Dict:
        """Learn from ACE repairs, diffs, and structural changes"""
        
        # Extract repair IDs
        repairs_in_txn = []
        repair_details = {}
        
        for repair in txn_data.get('ace', []):
            repair_id = safe_get_repair_id(repair.get('id', 'unknown'))
            repair_code = str(repair.get('code', 'I'))
            repair_field = str(repair.get('field', ''))
            repair_text = str(repair.get('text', ''))
            
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
        
        # Process diffs
        diff_features = self._analyze_diffs(txn_data, repairs_in_txn)
        
        # Detect structural changes
        structural_features = self._detect_structural_changes(txn_data, repairs_in_txn, txn_id)
        
        # Learn lookup patterns
        self._learn_lookup_patterns(txn_data, repairs_in_txn)
        
        # Combine all features
        combined_features = {**diff_features, **structural_features}
        
        return combined_features
    
    def _analyze_diffs(self, txn_data: Dict, repairs: List[str]) -> Dict:
        """Analyze explicit diffs from the diffs[] array"""
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
            'specific_repairs_indicated': set()
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
            
            entity_key_str = ensure_hashable(entity_key).lower()
            diff_features['affected_entities'].add(entity_key_str)
            
            # Track repairs for this entity
            for repair_id in repairs:
                self.entity_repair_associations[entity_key_str][repair_id] += 1
            
            # Process each diff
            for diff in entity_data.get('diffs', []):
                action = diff.get('msg', '').lower()
                field_path = ensure_hashable(diff.get('key', ''))
                value = diff.get('val', '')
                
                # Update counts
                if action == 'added':
                    diff_features['num_additions'] += 1
                    if 'ctry' in field_path.lower() and value and len(str(value)) == 2:
                        diff_features['specific_repairs_indicated'].add('6021')
                    if 'bicfi' in field_path.lower():
                        diff_features['specific_repairs_indicated'].add('6035')
                    if 'nm' in field_path.lower():
                        diff_features['specific_repairs_indicated'].add('6036')
                elif action == 'dropped':
                    diff_features['num_drops'] += 1
                elif action in ['transformed', 'edited']:
                    diff_features['num_transformations'] += 1
                
                field_lower = field_path.lower()
                diff_features['affected_fields'].add(field_lower)
                
                # Detect specific change types
                if 'bic' in field_lower:
                    diff_features['has_bic_changes'] = True
                if 'nm' in field_lower or 'name' in field_lower:
                    diff_features['has_name_changes'] = True
                if 'adr' in field_lower or 'pstl' in field_lower:
                    diff_features['has_address_changes'] = True
                if 'ctry' in field_lower:
                    diff_features['has_country_changes'] = True
                    if value and len(str(value)) == 2:
                        self.lookup_tables['country_extractions'][str(value)] += 1
                if 'clrsys' in field_lower or 'mmbid' in field_lower:
                    diff_features['has_clearing_changes'] = True
                
                # Update repair patterns
                for repair_id in repairs:
                    self.repair_patterns[repair_id]['total_diffs'] += 1
                    self.repair_patterns[repair_id]['typical_entities'][entity_key_str] += 1
                    
                    if action == 'added':
                        self.repair_patterns[repair_id]['fields_added'][field_lower] += 1
                    elif action == 'dropped':
                        self.repair_patterns[repair_id]['fields_dropped'][field_lower] += 1
                    elif action in ['transformed', 'edited']:
                        self.repair_patterns[repair_id]['fields_transformed'][field_lower] += 1
        
        return diff_features
    
    def _detect_structural_changes(self, txn_data: Dict, repairs: List[str], txn_id: str) -> Dict:
        """Detect ALL structural changes between before/after states"""
        
        structural_features = {
            'total_structural_changes': 0,
            'field_movements': 0,
            'field_additions': 0,
            'field_removals': 0,
            'restructuring_count': 0,
            'nesting_depth_changes': 0,
            'has_othr_addition': False,
            'has_clrsys_restructure': False,
            'has_mmbid_movement': False,
            'max_nesting_change': 0,
            'structural_complexity': 0
        }
        
        all_changes = []
        
        # Check each entity for structural changes
        entities = ['cdtrAgt', 'dbtrAgt', 'instgAgt', 'instdAgt',
                   'cdtr', 'dbtr', 'intrmyAgt1', 'intrmyAgt2',
                   'cdtrAcct', 'dbtrAcct', 'rmtInf']
        
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
            
            # Detect changes
            changes = self.structural_detector.detect_all_changes(before, after, entity, txn_id)
            all_changes.extend(changes)
            
            # Process detected changes
            for change in changes:
                structural_features['total_structural_changes'] += 1
                
                if change.change_type == 'added':
                    structural_features['field_additions'] += 1
                    if 'othr' in change.field_path.lower():
                        structural_features['has_othr_addition'] = True
                        
                elif change.change_type == 'removed':
                    structural_features['field_removals'] += 1
                    
                elif change.change_type == 'moved':
                    structural_features['field_movements'] += 1
                    if 'mmbid' in change.field_path.lower():
                        structural_features['has_mmbid_movement'] = True
                    if change.details and 'depth_change' in change.details:
                        depth = abs(change.details['depth_change'])
                        structural_features['nesting_depth_changes'] += depth
                        structural_features['max_nesting_change'] = max(
                            structural_features['max_nesting_change'], depth
                        )
                
                elif change.change_type == 'restructured':
                    structural_features['restructuring_count'] += 1
                    if change.details and 'clrsys' in str(change.details).lower():
                        structural_features['has_clrsys_restructure'] = True
                
                # Map change to repairs
                for repair_id in repairs:
                    pattern_key = f"{change.change_type}:{change.field_path}"
                    self.repair_patterns[repair_id]['structural_changes'][pattern_key] += 1
        
        # Calculate complexity score
        structural_features['structural_complexity'] = (
            structural_features['field_movements'] * 2 +
            structural_features['restructuring_count'] * 3 +
            structural_features['max_nesting_change'] +
            (structural_features['field_additions'] + structural_features['field_removals']) * 0.5
        )
        
        # Learn patterns
        if all_changes and repairs:
            self.structural_detector.learn_patterns(all_changes, repairs)
            for repair_id in repairs:
                signature = self.structural_detector.get_change_signature(all_changes)
                self.repair_to_structural_patterns[repair_id].append(signature)
        
        # Cache features
        self.structural_features_cache[txn_id] = structural_features
        
        return structural_features
    
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
            
            # Learn BIC lookups
            clearing = self._find_value_in_dict(before, 'mmbid') or self._find_value_in_dict(before, 'clrsysmmbid')
            bic_after = self._find_value_in_dict(after, 'bicfi') or self._find_value_in_dict(after, 'bic')
            
            if clearing and bic_after and not self._find_value_in_dict(before, 'bicfi'):
                self.lookup_tables['clearing_to_bic'][ensure_hashable(clearing)] = ensure_hashable(bic_after)
            
            # Learn name lookups
            if bic_after:
                name_after = self._find_value_in_dict(after, 'nm')
                if name_after:
                    self.lookup_tables['bic_to_name'][ensure_hashable(bic_after)] = ensure_hashable(name_after)
    
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
        """Print comprehensive learning summary"""
        logger.info("\n" + "="*70)
        logger.info("ACE REPAIR LEARNING SUMMARY")
        logger.info("="*70)
        
        # Repair taxonomy
        logger.info(f"\nLearned {len(self.repair_taxonomy)} unique repairs:")
        sorted_repairs = sorted(self.repair_taxonomy.items(), key=lambda x: x[1]['count'], reverse=True)
        for repair_id, info in sorted_repairs[:10]:
            logger.info(f"  {repair_id}: [{info['code']}] {info['field']} - {info['text'][:50]}...")
            logger.info(f"    Count: {info['count']}")
        
        # Co-occurrence patterns
        logger.info(f"\nRepair Co-occurrence Patterns:")
        for repair_id in list(self.repair_cooccurrence.keys())[:5]:
            co_occurring = self.repair_cooccurrence[repair_id].most_common(3)
            if co_occurring:
                logger.info(f"  {repair_id} often occurs with: {co_occurring}")
        
        # Structural patterns
        self.structural_detector.print_summary()


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

class FeatureExtractor:
    """Extracts features from payment structure, diffs, and structural changes"""
    
    def __init__(self, ace_learner: Optional[AceRepairLearner] = None):
        self.ace_learner = ace_learner
        self.feature_names = []
        self._build_feature_names()
    
    def _build_feature_names(self):
        """Define all feature names"""
        # Basic structural features
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
        
        # ACE-specific features
        self.feature_names.extend([
            'needs_country_extraction', 'needs_bic_lookup', 'needs_name_lookup',
            'needs_address_enrichment', 'multiple_agents_incomplete',
            'clearing_system_present', 'partial_bank_info'
        ])
        
        # Structural change features
        self.feature_names.extend([
            'struct_total_changes', 'struct_field_movements', 
            'struct_field_additions', 'struct_field_removals',
            'struct_restructuring_count', 'struct_nesting_changes',
            'struct_max_nesting_depth', 'struct_complexity_score',
            'struct_has_othr_addition', 'struct_has_clrsys_restructure', 
            'struct_has_mmbid_movement'
        ])
    
    def extract_features(self, payment: Dict, diff_features: Optional[Dict] = None,
                         structural_features: Optional[Dict] = None) -> np.ndarray:
        """Extract all features"""
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
            for _ in range(13):
                features[idx] = 0.0; idx += 1
        
        # ACE-specific features
        has_location_info = has_address or self._has_field(payment, 'townname')
        features[idx] = float(not has_country and has_location_info); idx += 1
        features[idx] = float(has_clearing and not has_bic); idx += 1
        features[idx] = float(has_bic and not has_name); idx += 1
        
        has_partial_address = self._has_field(payment, 'townname') or self._has_field(payment, 'pstlcd')
        features[idx] = float(has_partial_address and not has_address); idx += 1
        
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
        
        features[idx] = float(has_clearing or self._has_field(payment, 'clrsys')); idx += 1
        
        has_some_bank_info = has_bic or has_name or has_clearing
        has_complete_bank_info = has_bic and has_name
        features[idx] = float(has_some_bank_info and not has_complete_bank_info); idx += 1
        
        # Structural features
        if structural_features:
            features[idx] = min(structural_features.get('total_structural_changes', 0) / 20, 1.0); idx += 1
            features[idx] = min(structural_features.get('field_movements', 0) / 10, 1.0); idx += 1
            features[idx] = min(structural_features.get('field_additions', 0) / 15, 1.0); idx += 1
            features[idx] = min(structural_features.get('field_removals', 0) / 15, 1.0); idx += 1
            features[idx] = min(structural_features.get('restructuring_count', 0) / 5, 1.0); idx += 1
            features[idx] = min(structural_features.get('nesting_depth_changes', 0) / 10, 1.0); idx += 1
            features[idx] = min(structural_features.get('max_nesting_change', 0) / 5, 1.0); idx += 1
            features[idx] = min(structural_features.get('structural_complexity', 0) / 50, 1.0); idx += 1
            features[idx] = float(structural_features.get('has_othr_addition', False)); idx += 1
            features[idx] = float(structural_features.get('has_clrsys_restructure', False)); idx += 1
            features[idx] = float(structural_features.get('has_mmbid_movement', False)); idx += 1
        else:
            for _ in range(11):
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
        if depth > 10:
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
    """Process training data with comprehensive learning"""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.ace_learner = AceRepairLearner(self.config)
        self.feature_extractor = FeatureExtractor(self.ace_learner)
        self.repair_vocabulary = {}
        self.idx_to_repair = {}
        self.diff_features_cache = {}
        self.structural_features_cache = {}
    
    def load_transactions_from_path(self, path: str) -> Dict:
        """Load transactions from file or directory"""
        all_transactions = {}
        
        if os.path.isfile(path):
            logger.info(f"Loading single file: {path}")
            all_transactions = self._load_file(path)
        elif os.path.isdir(path):
            logger.info(f"Loading directory: {path}")
            all_transactions = self._load_directory(path)
        else:
            raise ValueError(f"Path does not exist: {path}")
        
        return all_transactions
    
    def _load_file(self, filepath: str) -> Dict:
        """Load transactions from a single file"""
        transactions = {}
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, dict):
                for txn_id, txn_data in data.items():
                    if isinstance(txn_data, dict):
                        transactions[ensure_hashable(txn_id)] = txn_data
                        
            elif isinstance(data, list):
                for idx, item in enumerate(data):
                    if isinstance(item, dict):
                        if len(item) == 1:
                            txn_id = list(item.keys())[0]
                            transactions[ensure_hashable(txn_id)] = item[txn_id]
                        else:
                            transactions[f"txn_{idx}"] = item
            
            logger.info(f"  Loaded {len(transactions)} transactions from {os.path.basename(filepath)}")
            
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
        
        return transactions
    
    def _load_directory(self, dirpath: str) -> Dict:
        """Load all JSON files from a directory"""
        all_transactions = {}
        
        json_files = []
        for root, dirs, files in os.walk(dirpath):
            for filename in files:
                if filename.endswith('.json'):
                    json_files.append(os.path.join(root, filename))
        
        logger.info(f"Found {len(json_files)} JSON files")
        
        for filepath in sorted(json_files):
            file_transactions = self._load_file(filepath)
            all_transactions.update(file_transactions)
        
        return all_transactions
    
    def load_and_process(self, path: str, max_transactions: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """Load and process training data"""
        logger.info(f"Loading data from {path}")
        
        all_transactions = self.load_transactions_from_path(path)
        
        if not all_transactions:
            raise ValueError(f"No transactions loaded from {path}")
        
        logger.info(f"Loaded {len(all_transactions)} total transactions")
        
        if max_transactions and len(all_transactions) > max_transactions:
            logger.info(f"Limiting to {max_transactions} transactions")
            all_transactions = dict(list(all_transactions.items())[:max_transactions])
        
        # First pass: Learn from data
        logger.info("Learning from ACE repairs, diffs, and structural changes...")
        processed_count = 0
        skipped_count = 0
        
        for txn_id, txn_data in all_transactions.items():
            if not isinstance(txn_data, dict):
                continue
            
            if 'ace' not in txn_data or not txn_data['ace']:
                skipped_count += 1
                continue
            
            try:
                # Learn from transaction (includes structural detection)
                combined_features = self.ace_learner.learn_from_transaction(txn_id, txn_data)
                
                # Split features for caching
                diff_features = {k: v for k, v in combined_features.items() 
                               if not k.startswith('struct_') and not k.startswith('total_structural')}
                structural_features = {k: v for k, v in combined_features.items() 
                                     if k.startswith('struct_') or k.startswith('total_structural')}
                
                self.diff_features_cache[txn_id] = diff_features
                self.structural_features_cache[txn_id] = structural_features
                processed_count += 1
                
                if processed_count % 100 == 0:
                    logger.info(f"  Processed {processed_count} transactions...")
                    
            except Exception as e:
                logger.warning(f"Error processing transaction {txn_id}: {e}")
                continue
        
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
                repairs = []
                for r in txn_data.get('ace', []):
                    if isinstance(r, dict) and 'id' in r:
                        repair_id = safe_get_repair_id(r.get('id'))
                        repairs.append(repair_id)
                
                # Extract features
                payment = self._extract_before_state(txn_data)
                diff_features = self.diff_features_cache[txn_id]
                structural_features = self.structural_features_cache.get(txn_id, {})
                
                features = self.feature_extractor.extract_features(
                    payment, diff_features, structural_features
                )
                
                labels = self._repairs_to_labels(repairs)
                
                all_features.append(features)
                all_labels.append(labels)
                all_payments.append(txn_data)
                
            except Exception as e:
                logger.warning(f"Error extracting features for {txn_id}: {e}")
                continue
        
        features_array = np.array(all_features)
        labels_array = np.array(all_labels)
        
        logger.info(f"\nProcessing complete:")
        logger.info(f"  Total transactions: {len(all_transactions)}")
        logger.info(f"  Trainable transactions: {len(features_array)}")
        logger.info(f"  Skipped (no repairs): {skipped_count}")
        logger.info(f"  Feature dimensions: {features_array.shape}")
        logger.info(f"  Unique repairs: {len(self.repair_vocabulary)}")
        
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
                    payment[entity] = entity_data
        
        return payment
    
    def _build_repair_vocabulary(self, data: Dict):
        """Build repair vocabulary"""
        all_repairs = set()
        
        for txn_data in data.values():
            if not isinstance(txn_data, dict):
                continue
            for repair in txn_data.get('ace', []):
                if isinstance(repair, dict) and 'id' in repair:
                    repair_id = safe_get_repair_id(repair.get('id'))
                    all_repairs.add(repair_id)
        
        sorted_repairs = sorted(all_repairs)
        self.repair_vocabulary = {repair_id: idx for idx, repair_id in enumerate(sorted_repairs)}
        self.idx_to_repair = {idx: repair_id for repair_id, idx in self.repair_vocabulary.items()}
        
        logger.info(f"Built vocabulary with {len(self.repair_vocabulary)} unique repairs")
    
    def _repairs_to_labels(self, repairs: List[str]) -> np.ndarray:
        """Convert repair IDs to multi-hot encoding"""
        labels = np.zeros(len(self.repair_vocabulary))
        for repair_id in repairs:
            if repair_id in self.repair_vocabulary:
                labels[self.repair_vocabulary[repair_id]] = 1.0
        return labels


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


# ============================================================================
# MODEL TRAINER
# ============================================================================

class ModelTrainer:
    """Handles model training and evaluation"""
    
    def __init__(self, config: Config):
        self.config = config
        self.processor = DataProcessor(config)
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
                n_estimators=150,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        )
        self.rf_model.fit(X_train, y_train)
        
        # Train Neural Network
        logger.info("Training Neural Network...")
        self.model = RepairNN(
            num_features=features.shape[1],
            num_repairs=labels.shape[1],
            hidden_dim=self.config.hidden_dim,
            dropout=self.config.dropout
        ).to(self.device)
        
        self._train_nn(X_train, y_train, X_val, y_val)
        
        # Evaluate
        logger.info("\nEvaluating on test set...")
        metrics = self._evaluate(X_test, y_test)
        
        # Save models
        self._save_models()
        
        return metrics
    
    def _train_nn(self, X_train, y_train, X_val, y_val):
        """Train neural network"""
        train_dataset = RepairDataset(X_train, y_train)
        val_dataset = RepairDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size)
        
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        criterion = nn.BCELoss()
        
        best_val_loss = float('inf')
        patience = 15
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
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
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
            scheduler.step(val_loss)
            
            if epoch % 5 == 0 or epoch == 1:
                logger.info(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss - 0.001:
                best_val_loss = val_loss
                patience_counter = 0
                os.makedirs(self.config.model_dir, exist_ok=True)
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
        
        # Get RF predictions
        rf_preds = self.rf_model.predict(X_test)
        
        # Calculate metrics
        nn_preds_binary = (preds > 0.5).astype(int)
        
        exact_match_nn = np.mean(np.all(nn_preds_binary == labels, axis=1))
        exact_match_rf = np.mean(np.all(rf_preds == labels, axis=1))
        
        hamming_nn = np.mean(nn_preds_binary == labels)
        hamming_rf = np.mean(rf_preds == labels)
        
        logger.info(f"\nModel Performance:")
        logger.info(f"  Neural Network:")
        logger.info(f"    Exact Match: {exact_match_nn:.2%}")
        logger.info(f"    Hamming Accuracy: {hamming_nn:.2%}")
        logger.info(f"  Random Forest:")
        logger.info(f"    Exact Match: {exact_match_rf:.2%}")
        logger.info(f"    Hamming Accuracy: {hamming_rf:.2%}")
        
        return {
            'exact_match_accuracy': exact_match_nn,
            'hamming_accuracy': hamming_nn,
            'rf_exact_match': exact_match_rf,
            'rf_hamming': hamming_rf
        }
    
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
# PREDICTOR
# ============================================================================

class RepairPredictor:
    """Make predictions on new payments"""
    
    def __init__(self, model_dir: str = './models'):
        self.model_dir = model_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Loading models from {model_dir}")
        
        # Load config
        self.config = Config.load(os.path.join(model_dir, 'config.json'))
        
        # Load processor
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
        """Predict repairs for a payment"""
        
        # Load payment
        with open(payment_file, 'r') as f:
            data = json.load(f)
        
        # Extract payment
        if isinstance(data, dict) and len(data) == 1:
            txn_id = list(data.keys())[0]
            payment = data[txn_id]
            logger.info(f"Processing transaction: {txn_id}")
        else:
            payment = data
        
        # Extract features (no diff/structural features for prediction)
        features = self.processor.feature_extractor.extract_features(payment, None, None)
        
        # Get NN predictions
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        with torch.no_grad():
            nn_probs = self.nn_model(features_tensor)[0].cpu().numpy()
        
        # Get RF predictions
        rf_probs = self.rf_model.predict_proba(features.reshape(1, -1))
        rf_probs = np.array([p[0, 1] if p.shape[1] > 1 else p[0, 0] for p in rf_probs])
        
        # Ensemble
        if use_ensemble:
            ensemble_probs = (nn_probs * 0.6 + rf_probs * 0.4)
        else:
            ensemble_probs = nn_probs
        
        # Get predicted repairs
        predicted_repairs = []
        ace_predictions = []
        
        for idx, prob in enumerate(ensemble_probs):
            if prob > threshold:
                repair_id = self.processor.idx_to_repair[idx]
                
                # Get repair info
                repair_info = self._get_repair_info(repair_id)
                
                predicted_repairs.append({
                    'repair_id': repair_id,
                    'confidence': float(prob),
                    'nn_prob': float(nn_probs[idx]),
                    'rf_prob': float(rf_probs[idx])
                })
                
                ace_predictions.append({
                    'id': repair_id,
                    'code': repair_info['code'],
                    'field': repair_info['field'],
                    'text': repair_info['text']
                })
        
        # Sort by confidence
        predicted_repairs.sort(key=lambda x: x['confidence'], reverse=True)
        ace_predictions.sort(key=lambda x: x['id'])
        
        return {
            'predicted_repairs': predicted_repairs,
            'repair_ids': [r['repair_id'] for r in predicted_repairs],
            'ace': ace_predictions,
            'confidence_summary': {
                'high_confidence': [r for r in predicted_repairs if r['confidence'] > 0.8],
                'medium_confidence': [r for r in predicted_repairs if 0.5 < r['confidence'] <= 0.8],
                'total_repairs': len(predicted_repairs)
            }
        }
    
    def _get_repair_info(self, repair_id: str) -> Dict:
        """Get repair information"""
        if hasattr(self.processor.ace_learner, 'repair_taxonomy'):
            if repair_id in self.processor.ace_learner.repair_taxonomy:
                return self.processor.ace_learner.repair_taxonomy[repair_id]
        
        # Fallback
        return {
            'code': 'I',
            'field': 'UNKNOWN',
            'text': f'Repair {repair_id} applied'
        }


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='ACE Payment Repair Predictor - Complete Version with Structural Detection'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--input', required=True, help='Training data')
    train_parser.add_argument('--max_txns', type=int, help='Max transactions')
    train_parser.add_argument('--epochs', type=int, default=50)
    train_parser.add_argument('--model_dir', default='./models')
    train_parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict repairs')
    predict_parser.add_argument('--input', required=True, help='Input payment')
    predict_parser.add_argument('--model', default='./models')
    predict_parser.add_argument('--threshold', type=float, default=0.5)
    predict_parser.add_argument('--output', help='Output file')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze data')
    analyze_parser.add_argument('--input', required=True, help='Data to analyze')
    analyze_parser.add_argument('--output', help='Output report')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        config = Config()
        config.num_epochs = args.epochs
        config.model_dir = args.model_dir
        config.verbose_logging = args.verbose
        
        trainer = ModelTrainer(config)
        
        if args.max_txns:
            original_method = trainer.processor.load_and_process
            def limited_load_and_process(path):
                return original_method(path, max_transactions=args.max_txns)
            trainer.processor.load_and_process = limited_load_and_process
        
        metrics = trainer.train(args.input)
        logger.info(f"\nTraining complete. Metrics: {metrics}")
        logger.info(f"Check structural_changes.log for detailed change analysis")
    
    elif args.command == 'predict':
        predictor = RepairPredictor(args.model)
        results = predictor.predict(args.input, args.threshold)
        
        logger.info(f"\nPredicted {len(results['predicted_repairs'])} repairs:")
        for repair in results['predicted_repairs']:
            logger.info(f"  {repair['repair_id']}: {repair['confidence']:.2%}")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Results saved to {args.output}")
    
    elif args.command == 'analyze':
        config = Config(verbose_logging=True)
        processor = DataProcessor(config)
        
        transactions = processor.load_transactions_from_path(args.input)
        logger.info(f"Analyzing {len(transactions)} transactions...")
        
        for txn_id, txn_data in list(transactions.items())[:100]:  # Analyze first 100
            if isinstance(txn_data, dict) and 'ace' in txn_data:
                processor.ace_learner.learn_from_transaction(txn_id, txn_data)
        
        processor.ace_learner.print_summary()
        logger.info(f"Check structural_changes.log for detailed analysis")
        
        if args.output:
            summary = {
                'transactions_analyzed': len(transactions),
                'repairs_found': len(processor.ace_learner.repair_taxonomy),
                'structural_changes': len(processor.ace_learner.structural_detector.all_detected_changes)
            }
            with open(args.output, 'w') as f:
                json.dump(summary, f, indent=2)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
