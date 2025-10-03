"""
Comprehensive Rule-Based Payment Repair System
===============================================

Three-phase system for automatic payment repair:

PHASE 1 - LEARN: Extract knowledge from training data
  - Discovers what each ACE rule does by analyzing diffs
  - Builds lookup tables (BIC → bank details) from before/after comparisons
  - Learns transformation patterns (which fields change, how they change)

PHASE 2 - TRAIN: Train ML model to predict rule application
  - Extracts features from payment structure (has BIC? missing name? etc.)
  - Trains neural network: payment features → rule probabilities
  - Multi-label classification (multiple rules can apply to one payment)

PHASE 3 - REPAIR: Apply learned rules to new payments
  - ML predicts which rules apply (e.g., 6021, 6035, 6036)
  - Rule engine executes transformations using learned patterns
  - Lookups substitute values (e.g., BIC → bank name/address)

Usage:
    # Step 1: Learn from training data (discovers all rules and builds lookups)
    python repair_system.py learn --input repairs_large.json --output models/
    
    # Step 2: Train ML predictor (teaches when to apply which rules)
    python repair_system.py train --input repairs_large.json --output models/ --epochs 50
    
    # Step 3: Apply repairs to new payments
    python repair_system.py repair --input payment.json --output repaired.json --model models/

PHASE 1 - LEARN: Extract knowledge from training data
PHASE 2 - TRAIN: Train ML model to predict rule application  
PHASE 3 - REPAIR: Apply learned rules to new payments

Usage:
    python ace_repair_model.py learn --input repairs_large.json
    python ace_repair_model.py train --input repairs_large.json --epochs 100
    python ace_repair_model.py repair --input payment.json --output result.json

Author: Enhanced Three-Phase Implementation
Version: 2.1 - Fixed clearing lookup and BIC learning

Author: Enhanced Three-Phase Implementation
Version: 2.1 - Fixed clearing lookup and BIC learning

Updated Training Loss
  Total loss = 1.0×char_loss + 0.1×change_loss + 1.0×repair_loss
  Example:    = 1.0×0.30    + 0.1×0.05      + 1.0×0.40
           = 0.30 + 0.005 + 0.40 = 0.705

"""
"""
Comprehensive Rule-Based Payment Repair System
===============================================

Three-phase system for automatic payment repair with enhanced learning.

PHASE 1 - LEARN: Extract knowledge from training data
PHASE 2 - TRAIN: Train ML model to predict rule application  
PHASE 3 - REPAIR: Apply learned rules to new payments

Usage:
    python ace_repair_model.py learn --input repairs_large.json
    python ace_repair_model.py train --input repairs_large.json --epochs 100
    python ace_repair_model.py repair --input payment.json --output result.json

Author: Enhanced Three-Phase Implementation
Version: 2.1 - Fixed clearing lookup and BIC learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import json
import argparse
import os
from typing import Dict, List, Optional, Tuple, Set, Any
import numpy as np
from dataclasses import dataclass, asdict
from pathlib import Path
import pickle
import re
from collections import defaultdict, Counter
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Model configuration with all hyperparameters"""
    embedding_dim: int = 128
    hidden_dim: int = 256
    num_lstm_layers: int = 2
    dropout: float = 0.3
    num_attention_heads: int = 8
    repair_attention_heads: int = 4
    batch_size: int = 16
    learning_rate: float = 0.001
    num_epochs: int = 100
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    max_text_length: int = 128
    max_fields: int = 100
    max_repairs: int = 20
    train_split: float = 0.8
    val_split: float = 0.1
    char_loss_weight: float = 1.0
    change_detection_weight: float = 0.1
    repair_prediction_weight: float = 1.0  # FIXED: Increased from 0.2 to 1.0
    model_save_path: str = "./models/ace_model.pt"
    vocab_save_path: str = "./models/ace_vocab.pkl"
    field_vocab_save_path: str = "./models/ace_field_vocab.pkl"
    repair_taxonomy_path: str = "./models/ace_repair_taxonomy.json"
    lookup_tables_path: str = "./models/lookup_tables.pkl"
    
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: str):
        with open(path, 'r') as f:
            return cls(**json.load(f))


# ============================================================================
# PHASE 1: LEARN - LOOKUP TABLE BUILDER (FIXED)
# ============================================================================

class LookupTableBuilder:
    """
    Builds lookup tables by analyzing repair patterns.
    FIXED: Now properly learns from both 'before' and 'after' states,
    and correctly captures clearing system ID -> BIC mappings.
    """
    
    def __init__(self):
        self.bic_to_bank_info = {}
        self.iban_to_info = {}
        self.clearing_system_to_bank = {}
        self.country_code_mappings = {}
        self.field_standardizations = defaultdict(dict)
        self.repair_to_lookup_type = {}
        self.repair_patterns = defaultdict(list)
        self.stats = {
            'total_transactions': 0,
            'total_diffs': 0,
            'lookup_operations': defaultdict(int),
            'field_additions': Counter(),
            'field_edits': Counter(),
            'field_drops': Counter()
        }
    
    def learn_from_transactions(self, transactions_file: str):
        """Main learning phase - extract patterns from all transactions"""
        logger.info("="*70)
        logger.info("PHASE 1: LEARNING FROM REPAIR PATTERNS")
        logger.info("="*70)
        
        with open(transactions_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.stats['total_transactions'] = len(data)
        logger.info(f"Loaded {len(data)} transactions for learning")
        
        processed_count = 0
        for txn_id, txn_data in data.items():
            processed_count += 1
            if processed_count % 100 == 0:
                logger.info(f"Processing transaction {processed_count}/{len(data)}...")
            
            ace_repairs = txn_data.get('ace', [])
            repair_ids = [r['id'] for r in ace_repairs]
            
            for entity_key in ['cdtrAgt', 'dbtrAgt', 'instgAgt', 'cdtr', 'dbtr', 
                              'cdtrAcct', 'dbtrAcct', 'intrmyAgt1', 'intrmyAgt2']:
                if entity_key not in txn_data:
                    continue
                
                entity_data = txn_data[entity_key]
                before = entity_data.get('before', {})
                after = entity_data.get('after', {})
                diffs = entity_data.get('diffs', [])
                
                # FIXED: Learn from existing data in 'before' state
                # This captures data that was already present (not added in diffs)
                self._learn_from_existing_data(before, after, repair_ids, entity_key)
                
                # Learn from diffs (additions/transformations)
                for diff in diffs:
                    self.analyze_diff(diff, before, after, repair_ids, entity_key)
        
        self.print_statistics()
        return self.compile_lookup_tables()
    
    def _learn_from_dropped_bic(self, bic: str, before: Dict, repair_ids: List[str]):
        """
        NEW: Learn BIC info even when BIC is being dropped.
        In cases where BIC exists in 'before' but is dropped in 'after',
        we still want to capture the bank info associated with it.
        """
        if bic not in self.bic_to_bank_info:
            self.bic_to_bank_info[bic] = {}
        
        # Look for name in before state
        name = self._find_value_in_nested_dict(before, ['nm', 'name'])
        if name:
            self.bic_to_bank_info[bic]['name'] = name
        
        # Look for address in before state
        address = self._find_value_in_nested_dict(before, ['adrline', 'address'])
        if address:
            if 'address' not in self.bic_to_bank_info[bic]:
                self.bic_to_bank_info[bic]['address'] = []
            if isinstance(address, list):
                self.bic_to_bank_info[bic]['address'].extend(address)
            else:
                self.bic_to_bank_info[bic]['address'].append(address)
        
        for repair_id in repair_ids:
            self.repair_patterns[repair_id].append({
                'type': 'BIC_dropped',
                'source_field': 'BIC'
            })
    
    def _learn_from_existing_data(self, before: Dict, after: Dict, 
                                   repair_ids: List[str], entity_key: str):
        """
        NEW: Learn from data that exists in 'before' state.
        This captures BIC->name mappings where name already existed.
        """
        # Find BIC in before state
        bic = self._find_bic_in_data(before)
        if bic:
            # Look for bank name in before state
            name = self._find_value_in_nested_dict(before, ['nm', 'name'])
            if name:
                self._learn_bic_lookup(bic, 'name', name, repair_ids)
                self.stats['lookup_operations']['BIC_to_name_existing'] += 1
            
            # Look for address in before state
            address = self._find_value_in_nested_dict(before, ['adrline', 'address'])
            if address:
                self._learn_bic_lookup(bic, 'address', address, repair_ids)
                self.stats['lookup_operations']['BIC_to_address_existing'] += 1
    
    def analyze_diff(self, diff: Dict, before: Dict, after: Dict, 
                     repair_ids: List[str], entity_key: str):
        """Analyze a single diff entry to extract lookup patterns"""
        self.stats['total_diffs'] += 1
        
        action = diff.get('action') or diff.get('msg', 'unknown')
        field_path = diff.get('field') or diff.get('key', '')
        value = diff.get('value') or diff.get('val')
        source = diff.get('source', '')
        
        if action == 'added':
            self.stats['field_additions'][field_path] += 1
        elif action in ['edited', 'transformed']:
            self.stats['field_edits'][field_path] += 1
        elif action == 'dropped':
            self.stats['field_drops'][field_path] += 1
        
        # Handle additions and transformations
        if action in ['added', 'transformed']:
            self._detect_lookup_operation(field_path, value, before, after, 
                                         repair_ids, source, entity_key)
        
        # Handle edits and transformations
        elif action in ['edited', 'transformed']:
            old_value = diff.get('old_value')
            self._detect_standardization(field_path, old_value, value, repair_ids)
        
        # NEW: Handle BIC drops - learn the BIC info before it's dropped
        elif action == 'dropped':
            if 'bic' in field_path.lower() and value:
                # The BIC is being dropped, but we should still learn its info
                # Look for name/address that might exist in 'before'
                self._learn_from_dropped_bic(value, before, repair_ids)
                self.stats['lookup_operations']['BIC_dropped'] += 1
    
    def _detect_lookup_operation(self, field_path: str, value: Any, 
                                  before: Dict, after: Dict, 
                                  repair_ids: List[str], source: str, entity_key: str):
        """Detect if this is a lookup operation and extract the pattern"""
        
        # Pattern 1: Bank name addition (BIC lookup)
        # FIXED: Check both before AND after for BIC (after needed when BIC was just added)
        if 'nm' in field_path.lower() and 'fininstnid' in field_path.lower():
            bic_value = self._find_bic_in_data(before) or self._find_bic_in_data(after)
            if bic_value:
                self._learn_bic_lookup(bic_value, 'name', value, repair_ids)
                self.stats['lookup_operations']['BIC_to_name'] += 1
                return
        
        # Pattern 2: Address addition (BIC lookup)
        # FIXED: Check both before AND after for BIC
        if 'adrline' in field_path.lower() or 'pstladr' in field_path.lower():
            bic_value = self._find_bic_in_data(before) or self._find_bic_in_data(after)
            if bic_value and 'fininstnid' in field_path.lower():
                self._learn_bic_lookup(bic_value, 'address', value, repair_ids)
                self.stats['lookup_operations']['BIC_to_address'] += 1
                return
        
        # Pattern 3: BIC addition from clearing system ID
        if ('bicfi' in field_path.lower() or 'bic' in field_path.lower()) and value:
            clearing_id = self._find_clearing_id_in_data(before)
            if clearing_id:
                self._learn_clearing_lookup(clearing_id, value, repair_ids)
                self.stats['lookup_operations']['Clearing_to_BIC'] += 1
                logger.debug(f"Learned: Clearing ID {clearing_id} -> BIC {value}")
                return
        
        # Pattern 4: Country code extraction from BIC
        if 'ctryofres' in field_path.lower() or 'ctry' in field_path.lower():
            bic = self._find_bic_in_data(before)
            iban = self._find_iban_in_data(before)
            
            if bic and len(bic) >= 6:
                country_from_bic = bic[4:6].upper()
                if country_from_bic == value or value.upper() == country_from_bic:
                    self._learn_country_extraction(bic, value, 'BIC', repair_ids)
                    self.stats['lookup_operations']['Country_from_BIC'] += 1
                    return
            elif iban and len(iban) >= 2:
                country_from_iban = iban[:2].upper()
                if country_from_iban == value.upper():
                    self._learn_country_extraction(iban, value, 'IBAN', repair_ids)
                    self.stats['lookup_operations']['Country_from_IBAN'] += 1
                    return
        
        # Pattern 5: IBAN recognition
        if 'iban' in field_path.lower():
            if value and len(str(value)) >= 15:
                self._learn_iban_lookup(value, field_path, value, repair_ids)
                self.stats['lookup_operations']['IBAN_recognition'] += 1
                return
    
    def _learn_bic_lookup(self, bic: str, field_type: str, value: Any, 
                          repair_ids: List[str]):
        """Learn BIC -> field value mapping"""
        if bic not in self.bic_to_bank_info:
            self.bic_to_bank_info[bic] = {}
        
        if field_type == 'name':
            self.bic_to_bank_info[bic]['name'] = value
        elif field_type == 'address':
            if 'address' not in self.bic_to_bank_info[bic]:
                self.bic_to_bank_info[bic]['address'] = []
            if isinstance(value, list):
                self.bic_to_bank_info[bic]['address'].extend(value)
            else:
                self.bic_to_bank_info[bic]['address'].append(value)
        else:
            self.bic_to_bank_info[bic][field_type] = value
        
        for repair_id in repair_ids:
            self.repair_patterns[repair_id].append({
                'type': 'BIC_lookup',
                'field_type': field_type,
                'source_field': 'BIC'
            })
    
    def _learn_clearing_lookup(self, clearing_id: str, bic: str, repair_ids: List[str]):
        """Learn clearing system ID -> BIC mapping"""
        if clearing_id not in self.clearing_system_to_bank:
            self.clearing_system_to_bank[clearing_id] = {}
        
        self.clearing_system_to_bank[clearing_id]['bic'] = bic
        
        # Link to BIC info if we already have it
        if bic in self.bic_to_bank_info:
            self.clearing_system_to_bank[clearing_id]['bank_info_ref'] = bic
        
        for repair_id in repair_ids:
            self.repair_patterns[repair_id].append({
                'type': 'Clearing_lookup',
                'field_type': 'bic',
                'source_field': 'ClrSysMmbId'
            })
    
    def _learn_country_extraction(self, source_value: str, country: str, 
                                   source_type: str, repair_ids: List[str]):
        """Learn country code extraction patterns"""
        key = f"{source_type}:{source_value[:6]}"
        self.country_code_mappings[key] = country
        
        for repair_id in repair_ids:
            self.repair_patterns[repair_id].append({
                'type': 'Country_extraction',
                'source': source_type
            })
    
    def _learn_iban_lookup(self, iban: str, field_path: str, value: Any, 
                           repair_ids: List[str]):
        """Learn IBAN -> field value mapping"""
        if iban not in self.iban_to_info:
            self.iban_to_info[iban] = {}
        
        field_name = field_path.split('.')[-1].lower()
        self.iban_to_info[iban][field_name] = value
    
    def _detect_standardization(self, field_path: str, old_value: Any, 
                                new_value: Any, repair_ids: List[str]):
        """Detect field value standardization patterns"""
        if old_value != new_value:
            self.field_standardizations[field_path][str(old_value)] = str(new_value)
            
            for repair_id in repair_ids:
                self.repair_patterns[repair_id].append({
                    'type': 'Standardization',
                    'field': field_path
                })
    
    def _find_bic_in_data(self, data: Dict) -> Optional[str]:
        """Recursively search for BIC value in nested dict"""
        if isinstance(data, dict):
            for key, value in data.items():
                key_lower = key.lower() if isinstance(key, str) else str(key)
                if 'bic' in key_lower and isinstance(value, str) and len(value) >= 8:
                    return value.upper()
                if isinstance(value, dict):
                    result = self._find_bic_in_data(value)
                    if result:
                        return result
        return None
    
    def _find_iban_in_data(self, data: Dict) -> Optional[str]:
        """Recursively search for IBAN value in nested dict"""
        if isinstance(data, dict):
            for key, value in data.items():
                key_lower = key.lower() if isinstance(key, str) else str(key)
                if 'iban' in key_lower and isinstance(value, str):
                    return value.upper()
                if isinstance(value, str) and len(value) >= 15 and value[:2].isalpha():
                    return value.upper()
                if isinstance(value, dict):
                    result = self._find_iban_in_data(value)
                    if result:
                        return result
        return None
    
    def _find_clearing_id_in_data(self, data: Dict) -> Optional[str]:
        """Recursively search for clearing system member ID"""
        if isinstance(data, dict):
            for key, value in data.items():
                key_lower = key.lower() if isinstance(key, str) else str(key)
                if 'mmbid' in key_lower and isinstance(value, str):
                    return value
                if isinstance(value, dict):
                    result = self._find_clearing_id_in_data(value)
                    if result:
                        return result
        return None
    
    def _find_value_in_nested_dict(self, data: Dict, search_keys: List[str]) -> Optional[Any]:
        """
        NEW: Find value by searching for any of the given keys in nested dict.
        Used to find names, addresses, etc. in existing data.
        """
        if isinstance(data, dict):
            for key, value in data.items():
                key_lower = key.lower() if isinstance(key, str) else str(key)
                # Check if any search key matches
                for search_key in search_keys:
                    if search_key.lower() in key_lower:
                        if isinstance(value, (str, list)) and value:
                            return value
                # Recurse into nested dicts
                if isinstance(value, dict):
                    result = self._find_value_in_nested_dict(value, search_keys)
                    if result:
                        return result
        return None
    
    def compile_lookup_tables(self) -> Dict:
        """Compile all learned lookup tables"""
        lookup_tables = {
            'bic_to_bank_info': self.bic_to_bank_info,
            'iban_to_info': self.iban_to_info,
            'clearing_system_to_bank': self.clearing_system_to_bank,
            'country_code_mappings': self.country_code_mappings,
            'field_standardizations': dict(self.field_standardizations),
            'repair_patterns': dict(self.repair_patterns),
            'repair_to_lookup_type': self.repair_to_lookup_type,
            'stats': self.stats
        }
        return lookup_tables
    
    def print_statistics(self):
        """Print learning phase statistics with details"""
        logger.info("\n" + "="*70)
        logger.info("LEARNING PHASE STATISTICS")
        logger.info("="*70)
        logger.info(f"Total Transactions: {self.stats['total_transactions']}")
        logger.info(f"Total Diffs Analyzed: {self.stats['total_diffs']}")
        logger.info(f"\nLookup Operations Detected:")
        for op_type, count in sorted(self.stats['lookup_operations'].items()):
            logger.info(f"  {op_type}: {count}")
        
        logger.info(f"\nLookup Tables Built:")
        logger.info(f"  BIC Entries: {len(self.bic_to_bank_info)}")
        if len(self.bic_to_bank_info) <= 20:
            for bic, info in list(self.bic_to_bank_info.items())[:20]:
                logger.info(f"    {bic}: {info.get('name', 'N/A')}")
        
        logger.info(f"  IBAN Entries: {len(self.iban_to_info)}")
        
        logger.info(f"  Clearing System Entries: {len(self.clearing_system_to_bank)}")
        if len(self.clearing_system_to_bank) <= 20:
            for clearing_id, info in self.clearing_system_to_bank.items():
                logger.info(f"    {clearing_id} -> {info.get('bic', 'N/A')}")
        
        logger.info(f"  Country Mappings: {len(self.country_code_mappings)}")
        logger.info(f"  Field Standardizations: {len(self.field_standardizations)}")
        
        logger.info(f"\nTop 10 Most Added Fields:")
        for field, count in self.stats['field_additions'].most_common(10):
            logger.info(f"  {field}: {count}")
        
        logger.info(f"\nTop 10 Most Edited Fields:")
        for field, count in self.stats['field_edits'].most_common(10):
            logger.info(f"  {field}: {count}")
    
    def save(self, path: str):
        """Save lookup tables to file"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        lookup_tables = self.compile_lookup_tables()
        with open(path, 'wb') as f:
            pickle.dump(lookup_tables, f)
        logger.info(f"\nLookup tables saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> Dict:
        """Load lookup tables from file"""
        with open(path, 'rb') as f:
            return pickle.load(f)


# ============================================================================
# VOCABULARIES
# ============================================================================

class Vocabulary:
    """Character-level vocabulary"""
    
    def __init__(self):
        self.char2idx = {
            '<PAD>': 0, '<UNK>': 1, '<MISSING>': 2,
            '<START>': 3, '<END>': 4, '<MASK>': 5
        }
        self.idx2char = {v: k for k, v in self.char2idx.items()}
        self.next_idx = 6
        
        for char in '.-_/@#$%&*()[]{}|\\:;"\',<>?+=~`':
            self.add_char(char)
    
    def add_char(self, char: str):
        if char not in self.char2idx:
            self.char2idx[char] = self.next_idx
            self.idx2char[self.next_idx] = char
            self.next_idx += 1
    
    def build_from_texts(self, texts: List[str]):
        for text in texts:
            if text:
                for char in str(text).upper():
                    self.add_char(char)
    
    def encode(self, text: Optional[str], max_length: int, add_special: bool = True) -> List[int]:
        if not text:
            return [self.char2idx['<MISSING>']] * max_length
        
        text = str(text).upper()[:max_length-2 if add_special else max_length]
        encoded = []
        
        if add_special:
            encoded.append(self.char2idx['<START>'])
        
        encoded.extend([self.char2idx.get(char, self.char2idx['<UNK>']) for char in text])
        
        if add_special and len(encoded) < max_length:
            encoded.append(self.char2idx['<END>'])
        
        encoded += [self.char2idx['<PAD>']] * (max_length - len(encoded))
        return encoded[:max_length]
    
    def decode(self, indices: List[int]) -> str:
        chars = []
        for idx in indices:
            char = self.idx2char.get(idx, '<UNK>')
            if char in ['<PAD>', '<MISSING>', '<START>', '<END>', '<MASK>']:
                continue
            if char != '<UNK>':
                chars.append(char)
        return ''.join(chars)
    
    def __len__(self):
        return len(self.char2idx)
    
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({'char2idx': self.char2idx, 'idx2char': self.idx2char}, f)
    
    @classmethod
    def load(cls, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        vocab = cls()
        vocab.char2idx = data['char2idx']
        vocab.idx2char = data['idx2char']
        vocab.next_idx = max(vocab.idx2char.keys()) + 1
        return vocab


def _field_stats_factory():
    return {'count': 0, 'changes': 0}


class FieldVocabulary:
    """Field name vocabulary with statistics"""
    
    def __init__(self):
        self.field2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2field = {0: '<PAD>', 1: '<UNK>'}
        self.next_idx = 2
        self.field_stats = defaultdict(_field_stats_factory)
    
    def add_field(self, field: str, is_changed: bool = False):
        if field not in self.field2idx:
            self.field2idx[field] = self.next_idx
            self.idx2field[self.next_idx] = field
            self.next_idx += 1
        
        self.field_stats[field]['count'] += 1
        if is_changed:
            self.field_stats[field]['changes'] += 1
    
    def get_idx(self, field: str) -> int:
        return self.field2idx.get(field, self.field2idx['<UNK>'])
    
    def __len__(self):
        return len(self.field2idx)
    
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'field2idx': self.field2idx,
                'idx2field': self.idx2field,
                'field_stats': dict(self.field_stats)
            }, f)
    
    @classmethod
    def load(cls, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        vocab = cls()
        vocab.field2idx = data['field2idx']
        vocab.idx2field = data['idx2field']
        vocab.field_stats = defaultdict(_field_stats_factory, data.get('field_stats', {}))
        vocab.next_idx = max(vocab.idx2field.keys()) + 1
        return vocab


class ACERepairTaxonomy:
    """ACE repair taxonomy manager"""
    
    def __init__(self):
        self.repairs = {}
        self.repair2idx = {'<NONE>': 0, '<UNK>': 1}
        self.idx2repair = {0: '<NONE>', 1: '<UNK>'}
        self.next_idx = 2
        self.field_repair_map = defaultdict(set)
        self.repair_field_map = defaultdict(set)
    
    def add_repair(self, repair_id: str, code: str, field: str, text: str):
        if repair_id not in self.repair2idx:
            self.repair2idx[repair_id] = self.next_idx
            self.idx2repair[self.next_idx] = repair_id
            self.next_idx += 1
        
        if repair_id not in self.repairs:
            self.repairs[repair_id] = {
                'code': code,
                'field': field,
                'text': text,
                'count': 0,
                'related_fields': set()
            }
        
        self.repairs[repair_id]['count'] += 1
        self.repairs[repair_id]['related_fields'].add(field)
        
        if field:
            self.field_repair_map[field].add(repair_id)
            self.repair_field_map[repair_id].add(field)
    
    def get_idx(self, repair_id: str) -> int:
        return self.repair2idx.get(repair_id, 1)
    
    def get_repair_category(self, repair_id: str) -> str:
        if repair_id in self.repairs:
            return self.repairs[repair_id]['code']
        return 'I'
    
    def __len__(self):
        return len(self.repair2idx)
    
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump({
                'repairs': {k: {**v, 'related_fields': list(v['related_fields'])} 
                          for k, v in self.repairs.items()},
                'repair2idx': self.repair2idx,
                'idx2repair': {int(k): v for k, v in self.idx2repair.items()}
            }, f, indent=2)
    
    @classmethod
    def load(cls, path: str):
        with open(path, 'r') as f:
            data = json.load(f)
        taxonomy = cls()
        taxonomy.repairs = {k: {**v, 'related_fields': set(v.get('related_fields', []))} 
                           for k, v in data['repairs'].items()}
        taxonomy.repair2idx = data['repair2idx']
        taxonomy.idx2repair = {int(k): v for k, v in data['idx2repair'].items()}
        taxonomy.next_idx = max(taxonomy.idx2repair.keys()) + 1
        return taxonomy


# ============================================================================
# DATA UTILITIES
# ============================================================================

def flatten_json(data: Dict, parent_key: str = '', sep: str = '.') -> Dict:
    """Flatten nested JSON structure"""
    items = []
    for k, v in data.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_json(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    items.extend(flatten_json(item, f"{new_key}[{i}]", sep=sep).items())
                else:
                    items.append((f"{new_key}[{i}]", str(item)))
        else:
            items.append((new_key, str(v) if v is not None else ''))
    return dict(items)


def normalize_keys(data):
    """Normalize all keys to lowercase"""
    if isinstance(data, dict):
        return {k.lower() if isinstance(k, str) else k: normalize_keys(v) 
                for k, v in data.items()}
    elif isinstance(data, list):
        return [normalize_keys(item) for item in data]
    return data


# ============================================================================
# PHASE 2: TRAIN - DATA PARSER
# ============================================================================

class ACEDataParser:
    """Parse ACE repair data and convert to model input format"""
    
    def __init__(self, config: Config, vocab: Vocabulary, 
                 field_vocab: FieldVocabulary, taxonomy: ACERepairTaxonomy):
        self.config = config
        self.vocab = vocab
        self.field_vocab = field_vocab
        self.taxonomy = taxonomy
    
    def parse_file(self, json_file: str) -> List[Tuple]:
        """Parse repair data file"""
        logger.info(f"Parsing file: {json_file}")
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        transactions = []
        
        for txn_id, txn_data in data.items():
            try:
                ace_repairs = []
                if 'ace' in txn_data:
                    for repair in txn_data['ace']:
                        repair_id = repair.get('id', 'unknown')
                        code = repair.get('code', 'I')
                        field = repair.get('field', '')
                        text = repair.get('text', '')
                        self.taxonomy.add_repair(repair_id, code, field, text)
                        ace_repairs.append(repair_id)
                
                before_data = {}
                after_data = {}
                all_diffs = []
                
                for entity_key in ['cdtrAgt', 'dbtrAgt', 'instgAgt', 'cdtr', 'dbtr', 
                                  'cdtrAcct', 'dbtrAcct', 'intrmyAgt1', 'intrmyAgt2']:
                    if entity_key in txn_data:
                        entity_data = txn_data[entity_key]
                        
                        if 'before' in entity_data:
                            before_flat = flatten_json(
                                normalize_keys(entity_data['before']), 
                                parent_key=entity_key
                            )
                            before_data.update(before_flat)
                        
                        if 'after' in entity_data:
                            after_flat = flatten_json(
                                normalize_keys(entity_data['after']), 
                                parent_key=entity_key
                            )
                            after_data.update(after_flat)
                        
                        if 'diffs' in entity_data:
                            all_diffs.extend(entity_data['diffs'])
                
                if before_data and after_data:
                    transactions.append((
                        txn_id, ace_repairs, before_data, after_data, all_diffs
                    ))
                
            except Exception as e:
                logger.error(f"Error processing transaction {txn_id}: {e}")
                continue
        
        logger.info(f"Successfully parsed {len(transactions)} transactions")
        return transactions
    
    def encode_fields(self, field_dict: Dict) -> Dict[str, torch.Tensor]:
        """Encode fields to tensors for model input"""
        field_names = []
        field_values = []
        
        sorted_fields = sorted(field_dict.items())[:self.config.max_fields]
        
        for field_name, field_value in sorted_fields:
            field_names.append(self.field_vocab.get_idx(field_name))
            field_values.append(self.vocab.encode(field_value, self.config.max_text_length))
        
        while len(field_names) < self.config.max_fields:
            field_names.append(self.field_vocab.get_idx('<PAD>'))
            field_values.append([self.vocab.char2idx['<PAD>']] * self.config.max_text_length)
        
        mask = [1.0] * len(sorted_fields) + [0.0] * (self.config.max_fields - len(sorted_fields))
        
        return {
            'field_names': torch.tensor(field_names, dtype=torch.long),
            'field_values': torch.tensor(field_values, dtype=torch.long),
            'field_mask': torch.tensor(mask, dtype=torch.float32)
        }
    
    def encode_repairs(self, repair_ids: List[str], max_repairs: int) -> torch.Tensor:
        """Encode repair IDs to tensor"""
        encoded = [self.taxonomy.get_idx(rid) for rid in repair_ids[:max_repairs]]
        encoded += [0] * (max_repairs - len(encoded))
        return torch.tensor(encoded, dtype=torch.long)


# ============================================================================
# NEURAL NETWORK MODEL
# ============================================================================

class ACERepairModel(nn.Module):
    """Main repair model with repair prediction capability"""
    
    def __init__(self, config: Config, vocab_size: int, field_vocab_size: int, 
                 num_repairs: int):
        super().__init__()
        self.config = config
        
        self.repair_embedding = nn.Embedding(num_repairs, config.embedding_dim)
        self.field_name_embedding = nn.Embedding(field_vocab_size, config.embedding_dim)
        self.char_embedding = nn.Embedding(vocab_size, config.embedding_dim, padding_idx=0)
        
        self.value_lstm = nn.LSTM(
            config.embedding_dim,
            config.hidden_dim // 2,
            num_layers=config.num_lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=config.dropout if config.num_lstm_layers > 1 else 0
        )
        
        # FIXED: Two repair predictors - one for training (with targets), one for inference (input-only)
        self.repair_predictor_train = nn.Sequential(
            nn.Linear(config.hidden_dim * 3, config.hidden_dim),  # Training: input + target + diff
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, num_repairs),
            nn.Sigmoid()
        )
        
        self.repair_predictor_inference = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),  # Inference: input only
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, num_repairs),
            nn.Sigmoid()
        )
        
        self.field_combiner = nn.Sequential(
            nn.Linear(config.embedding_dim * 2 + config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.LayerNorm(config.hidden_dim)
        )
        
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=config.num_attention_heads,
                dim_feedforward=config.hidden_dim * 4,
                dropout=config.dropout,
                batch_first=True
            )
            for _ in range(4)
        ])
        
        self.change_detector = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.decoder_lstm = nn.LSTM(
            config.hidden_dim,
            config.hidden_dim,
            batch_first=True
        )
        
        self.output_projection = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, vocab_size)
        )
    
    def forward(self, input_dict, predict_repairs=False, target_dict=None):
        """
        Forward pass through the model.
        
        REDESIGNED: Repair predictor now sees both input AND target (during training)
        to learn which repairs are needed based on what transformations occur.
        
        Args:
            input_dict: Input fields (before state)
            predict_repairs: If True, predict which repairs to apply
            target_dict: Target fields (after state) - used during training for repair prediction
        """
        batch_size = input_dict['field_names'].size(0)
        max_fields = input_dict['field_names'].size(1)
        
        # Encode field names
        field_name_emb = self.field_name_embedding(input_dict['field_names'])
        
        # Encode field values (input)
        field_values = input_dict['field_values']
        field_values_flat = field_values.view(batch_size * max_fields, -1)
        char_emb = self.char_embedding(field_values_flat)
        lstm_out, (hidden, _) = self.value_lstm(char_emb)
        value_emb = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        value_emb = value_emb.view(batch_size, max_fields, -1)
        
        # REDESIGNED: Repair prediction based on input-target gap
        if predict_repairs:
            if target_dict is not None:
                # Training mode: Compare input vs target to predict repairs
                target_values = target_dict['field_values']
                target_values_flat = target_values.view(batch_size * max_fields, -1)
                target_char_emb = self.char_embedding(target_values_flat)
                target_lstm_out, (target_hidden, _) = self.value_lstm(target_char_emb)
                target_value_emb = torch.cat([target_hidden[-2], target_hidden[-1]], dim=-1)
                target_value_emb = target_value_emb.view(batch_size, max_fields, -1)
                
                # Compute difference between input and target
                value_diff = target_value_emb - value_emb
                
                # Combine: input + target + difference (768 dims)
                repair_input = torch.cat([
                    value_emb.mean(dim=1),         # What we have
                    target_value_emb.mean(dim=1),  # What we need
                    value_diff.mean(dim=1)         # What's changing
                ], dim=-1)
                
                # Use training predictor (expects 768 dims)
                repair_predictions = self.repair_predictor_train(repair_input)
            else:
                # Inference mode: Only have input (256 dims)
                repair_input = value_emb.mean(dim=1)
                
                # Use inference predictor (expects 256 dims)
                repair_predictions = self.repair_predictor_inference(repair_input)
            
            return {'repair_predictions': repair_predictions}
        
        # Normal forward pass for field transformation
        repair_emb = self.repair_embedding(input_dict['repair_ids'])
        repair_context = repair_emb.mean(dim=1, keepdim=True).expand(-1, max_fields, -1)
        combined = torch.cat([repair_context, field_name_emb, value_emb], dim=-1)
        field_embeddings = self.field_combiner(combined)
        
        field_mask = input_dict['field_mask']
        attention_mask = (field_mask == 0)
        
        enriched = field_embeddings
        for transformer_layer in self.transformer_layers:
            enriched = transformer_layer(enriched, src_key_padding_mask=attention_mask)
        
        change_probabilities = self.change_detector(enriched).squeeze(-1)
        
        enriched_flat = enriched.view(batch_size * max_fields, self.config.hidden_dim)
        decoded_flat = enriched_flat.unsqueeze(1).repeat(1, self.config.max_text_length, 1)
        decoded_flat, _ = self.decoder_lstm(decoded_flat)
        logits_flat = self.output_projection(decoded_flat)
        logits = logits_flat.view(batch_size, max_fields, self.config.max_text_length, -1)
        
        return {
            'field_logits': logits,
            'change_probabilities': change_probabilities
        }


# ============================================================================
# DATASET
# ============================================================================

class ACERepairDataset(Dataset):
    """PyTorch Dataset for ACE repair data"""
    
    def __init__(self, transactions: List[Tuple], parser: ACEDataParser):
        self.parser = parser
        self.transactions = transactions
    
    def __len__(self):
        return len(self.transactions)
    
    def __getitem__(self, idx):
        """Get a single training example"""
        txn_id, repair_ids, before_data, after_data, diffs = self.transactions[idx]
        
        input_encoded = self.parser.encode_fields(before_data)
        output_encoded = self.parser.encode_fields(after_data)
        
        repair_encoded = self.parser.encode_repairs(repair_ids, self.parser.config.max_repairs)
        input_encoded['repair_ids'] = repair_encoded
        
        repair_target = torch.zeros(len(self.parser.taxonomy), dtype=torch.float32)
        for rid in repair_ids:
            idx = self.parser.taxonomy.get_idx(rid)
            if idx > 1:
                repair_target[idx] = 1.0
        output_encoded['repair_targets'] = repair_target
        
        change_mask = self.calculate_change_mask(before_data, after_data)
        output_encoded['change_mask'] = change_mask
        
        return input_encoded, output_encoded
    
    def calculate_change_mask(self, before: Dict, after: Dict) -> torch.Tensor:
        """Calculate which fields changed"""
        change_mask = []
        sorted_fields = sorted(before.keys())[:self.parser.config.max_fields]
        
        for field_name in sorted_fields:
            before_value = before.get(field_name, '')
            after_value = after.get(field_name, '')
            change_mask.append(1.0 if before_value != after_value else 0.0)
        
        change_mask += [0.0] * (self.parser.config.max_fields - len(change_mask))
        return torch.tensor(change_mask, dtype=torch.float32)


# ============================================================================
# TRAINER
# ============================================================================

class ACETrainer:
    """Model trainer with multi-task loss and early stopping"""
    
    def __init__(self, model: ACERepairModel, config: Config, device: str = 'cpu'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.5
        )
        
        self.char_criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
        self.change_criterion = nn.BCELoss(reduction='none')
        self.best_val_loss = float('inf')
        
        # ADDED: Early stopping
        self.patience_counter = 0
        self.early_stop_patience = 20
        self.min_delta = 1e-4  # Minimum improvement to reset patience
    
    def compute_loss(self, outputs, targets):
        """Compute multi-task loss"""
        batch_size = targets['field_values'].size(0)
        max_fields = targets['field_values'].size(1)
        max_length = targets['field_values'].size(2)
        
        logits = outputs['field_logits']
        target_values = targets['field_values']
        field_mask = targets['field_mask']
        
        logits_flat = logits.view(batch_size * max_fields * max_length, -1)
        targets_flat = target_values.view(batch_size * max_fields * max_length)
        
        char_loss_per_token = self.char_criterion(logits_flat, targets_flat)
        char_loss_per_token = char_loss_per_token.view(batch_size, max_fields, max_length)
        
        field_mask_expanded = field_mask.unsqueeze(-1).expand(-1, -1, max_length)
        masked_char_loss = char_loss_per_token * field_mask_expanded
        char_loss = masked_char_loss.sum() / field_mask_expanded.sum().clamp(min=1)
        
        change_predictions = outputs['change_probabilities']
        change_targets = targets.get('change_mask', torch.zeros_like(change_predictions))
        change_loss_per_field = self.change_criterion(change_predictions, change_targets)
        masked_change_loss = change_loss_per_field * field_mask
        change_loss = masked_change_loss.sum() / field_mask.sum().clamp(min=1)
        
        repair_loss = torch.tensor(0.0, device=char_loss.device)
        if 'repair_predictions' in outputs and 'repair_targets' in targets:
            repair_preds = outputs['repair_predictions']
            repair_targets = targets['repair_targets']
            repair_loss = F.binary_cross_entropy(repair_preds, repair_targets)
        
        total_loss = (
            self.config.char_loss_weight * char_loss +
            self.config.change_detection_weight * change_loss +
            self.config.repair_prediction_weight * repair_loss
        )
        
        return total_loss, {
            'char_loss': char_loss.item(),
            'change_loss': change_loss.item(),
            'repair_loss': repair_loss.item() if isinstance(repair_loss, torch.Tensor) else 0.0,
            'total_loss': total_loss.item()
        }
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch with redesigned repair predictor training"""
        self.model.train()
        epoch_losses = defaultdict(float)
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            targets = {k: v.to(self.device) for k, v in targets.items()}
            
            self.optimizer.zero_grad()
            
            # Forward pass 1: Regular training (field predictions)
            outputs = self.model(inputs, predict_repairs=False)
            
            # Forward pass 2: Repair prediction WITH target information (REDESIGNED)
            # Now the model sees both input and target to learn repair patterns
            repair_outputs = self.model(inputs, predict_repairs=True, target_dict=targets)
            outputs['repair_predictions'] = repair_outputs['repair_predictions']
            
            # Compute combined loss
            loss, loss_components = self.compute_loss(outputs, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()
            
            for key, value in loss_components.items():
                epoch_losses[key] += value
            
            if batch_idx % 10 == 0:
                logger.info(f'  Epoch {epoch} Batch {batch_idx}/{len(train_loader)}, '
                           f'Loss: {loss.item():.4f} '
                           f'(char: {loss_components["char_loss"]:.4f}, '
                           f'change: {loss_components["change_loss"]:.4f}, '
                           f'repair: {loss_components["repair_loss"]:.4f})')
        
        for key in epoch_losses:
            epoch_losses[key] /= len(train_loader)
        
        return epoch_losses
    
    def evaluate(self, val_loader):
        """Evaluate on validation set with redesigned repair prediction"""
        self.model.eval()
        epoch_losses = defaultdict(float)
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                targets = {k: v.to(self.device) for k, v in targets.items()}
                
                # Evaluate field transformations
                outputs = self.model(inputs, predict_repairs=False)
                
                # REDESIGNED: Evaluate repair predictor with target information
                repair_outputs = self.model(inputs, predict_repairs=True, target_dict=targets)
                outputs['repair_predictions'] = repair_outputs['repair_predictions']
                
                loss, loss_components = self.compute_loss(outputs, targets)
                
                for key, value in loss_components.items():
                    epoch_losses[key] += value
        
        for key in epoch_losses:
            epoch_losses[key] /= len(val_loader)
        
        return epoch_losses
    
    def train(self, train_loader, val_loader):
        """Main training loop with early stopping"""
        logger.info("="*70)
        logger.info("PHASE 2: TRAINING ACE REPAIR MODEL")
        logger.info("="*70)
        
        for epoch in range(1, self.config.num_epochs + 1):
            logger.info(f"\nEpoch {epoch}/{self.config.num_epochs}")
            
            train_losses = self.train_epoch(train_loader, epoch)
            val_losses = self.evaluate(val_loader)
            self.scheduler.step(val_losses['total_loss'])
            
            logger.info(f"Train Loss: {train_losses['total_loss']:.4f} "
                       f"(char: {train_losses['char_loss']:.4f}, "
                       f"change: {train_losses['change_loss']:.4f}, "
                       f"repair: {train_losses['repair_loss']:.4f})")
            logger.info(f"Val Loss: {val_losses['total_loss']:.4f} "
                       f"(char: {val_losses['char_loss']:.4f}, "
                       f"change: {val_losses['change_loss']:.4f}, "
                       f"repair: {val_losses['repair_loss']:.4f})")
            
            # FIXED: Early stopping logic
            improvement = self.best_val_loss - val_losses['total_loss']
            if improvement > self.min_delta:
                self.best_val_loss = val_losses['total_loss']
                self.patience_counter = 0
                self.save_model()
                logger.info("✓ Best model saved!")
            else:
                self.patience_counter += 1
                logger.info(f"No improvement. Patience: {self.patience_counter}/{self.early_stop_patience}")
                
                if self.patience_counter >= self.early_stop_patience:
                    logger.info(f"\nEarly stopping triggered at epoch {epoch}")
                    logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
                    break
        
        logger.info(f"\nTraining complete! Best val loss: {self.best_val_loss:.4f}")
    
    def save_model(self):
        """Save model checkpoint"""
        os.makedirs(os.path.dirname(self.config.model_save_path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': asdict(self.config)
        }, self.config.model_save_path)


# ============================================================================
# PHASE 3: REPAIR - PREDICTOR WITH ENHANCED LOOKUP SUPPORT (FIXED)
# ============================================================================

class ACERepairer:
    """Apply repairs to new payments using trained model and lookup tables"""
    
    def __init__(self, model_dir: str = './models', device: str = 'cpu'):
        self.device = device
        self.model_dir = model_dir
        
        logger.info(f"Loading model from {model_dir}")
        
        self.config = Config.load(f"{model_dir}/config.json")
        self.vocab = Vocabulary.load(self.config.vocab_save_path)
        self.field_vocab = FieldVocabulary.load(self.config.field_vocab_save_path)
        self.taxonomy = ACERepairTaxonomy.load(self.config.repair_taxonomy_path)
        
        self.lookup_tables = LookupTableBuilder.load(self.config.lookup_tables_path)
        logger.info(f"Loaded lookup tables:")
        logger.info(f"  - BIC entries: {len(self.lookup_tables['bic_to_bank_info'])}")
        logger.info(f"  - Clearing system entries: {len(self.lookup_tables['clearing_system_to_bank'])}")
        logger.info(f"  - IBAN entries: {len(self.lookup_tables['iban_to_info'])}")
        
        self.parser = ACEDataParser(self.config, self.vocab, self.field_vocab, self.taxonomy)
        
        self.model = ACERepairModel(
            self.config, len(self.vocab), len(self.field_vocab), len(self.taxonomy)
        )
        
        checkpoint = torch.load(self.config.model_save_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        logger.info("Model loaded successfully")
    
    def repair(self, payment: Dict, repair_ids: List[str] = None,
               confidence_threshold: float = 0.5, repair_threshold: float = 0.5) -> Dict:
        """Apply repairs to a payment (FIXED: lower default confidence threshold)"""
        logger.info("="*70)
        logger.info("PHASE 3: APPLYING REPAIRS")
        logger.info("="*70)
        
        before_flat = flatten_json(normalize_keys(payment))
        
        if not repair_ids:
            logger.info("Predicting which repairs to apply...")
            repair_ids = self.predict_repairs(before_flat, repair_threshold)
            logger.info(f"Predicted repairs: {repair_ids}")
        else:
            logger.info(f"Using provided repairs: {repair_ids}")
        
        enriched_from_lookups = self.apply_lookup_tables(before_flat, repair_ids)
        
        encoded = self.parser.encode_fields(enriched_from_lookups)
        repair_encoded = self.parser.encode_repairs(repair_ids, self.config.max_repairs)
        encoded['repair_ids'] = repair_encoded
        
        batch_input = {k: v.unsqueeze(0).to(self.device) for k, v in encoded.items()}
        
        with torch.no_grad():
            outputs = self.model(batch_input)
        
        enriched = dict(enriched_from_lookups)
        applied_repairs = []
        
        logits = outputs['field_logits'][0]
        change_probs = outputs['change_probabilities'][0]
        field_names = encoded['field_names']
        field_mask = encoded['field_mask']
        
        for i in range(self.config.max_fields):
            if field_mask[i] == 0:
                continue
            
            field_idx = field_names[i].item()
            field_name = self.field_vocab.idx2field.get(field_idx, '<UNK>')
            
            if field_name in ['<PAD>', '<UNK>']:
                continue
            
            if change_probs[i] >= confidence_threshold:
                predicted_indices = logits[i].argmax(dim=-1).cpu().tolist()
                predicted_value = self.vocab.decode(predicted_indices)
                
                if predicted_value and predicted_value != enriched_from_lookups.get(field_name, ''):
                    enriched[field_name] = predicted_value
                    applied_repairs.append({
                        'field': field_name,
                        'before': before_flat.get(field_name, ''),
                        'after': predicted_value,
                        'source': 'model_prediction',
                        'confidence': float(change_probs[i])
                    })
        
        return {
            'enriched_payment': enriched,
            'applied_repairs': applied_repairs,
            'predicted_repair_ids': repair_ids,
            'repair_descriptions': [
                self.get_repair_description(rid) for rid in repair_ids
            ]
        }
    
    def predict_repairs(self, payment: Dict, threshold: float = 0.5) -> List[str]:
        """Predict which repairs should be applied"""
        encoded = self.parser.encode_fields(payment)
        encoded['repair_ids'] = torch.zeros(self.config.max_repairs, dtype=torch.long)
        
        batch_input = {k: v.unsqueeze(0).to(self.device) for k, v in encoded.items()}
        
        with torch.no_grad():
            outputs = self.model(batch_input, predict_repairs=True)
        
        repair_probs = outputs['repair_predictions'][0].cpu().numpy()
        
        predicted_repairs = []
        for idx, prob in enumerate(repair_probs):
            if prob > threshold and idx in self.taxonomy.idx2repair:
                repair_id = self.taxonomy.idx2repair[idx]
                if repair_id not in ['<NONE>', '<UNK>']:
                    predicted_repairs.append((repair_id, float(prob)))
        
        predicted_repairs.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"  Found {len(predicted_repairs)} repairs above threshold {threshold}")
        for repair_id, prob in predicted_repairs[:10]:
            logger.info(f"    {repair_id}: {prob:.3f} - {self.get_repair_description(repair_id)}")
        
        return [r[0] for r in predicted_repairs]
    
    def get_repair_description(self, repair_id: str) -> str:
        """Get human-readable description for a repair ID"""
        if repair_id in self.taxonomy.repairs:
            r = self.taxonomy.repairs[repair_id]
            return f"[{r['code']}] {r['text']}"
        return "Unknown repair"
    
    def apply_lookup_tables(self, payment: Dict, repair_ids: List[str]) -> Dict:
        """
        Apply lookup table enrichments (FIXED: better clearing ID search).
        This applies deterministic lookups before the model makes predictions.
        """
        enriched = dict(payment)
        applied_lookups = []
        
        # Lookup 1: BIC-based enrichment
        bic = self._find_value_by_key(payment, 'bic')
        if bic and bic in self.lookup_tables['bic_to_bank_info']:
            bank_info = self.lookup_tables['bic_to_bank_info'][bic]
            logger.info(f"Applying BIC lookup for {bic}")
            
            if 'name' in bank_info:
                enriched['lookup.bank_name'] = bank_info['name']
                applied_lookups.append(f"BIC {bic} -> name: {bank_info['name']}")
            
            if 'address' in bank_info:
                for i, addr_line in enumerate(bank_info['address']):
                    enriched[f'lookup.address_{i}'] = addr_line
                applied_lookups.append(f"BIC {bic} -> address: {len(bank_info['address'])} lines")
            
            if len(bic) >= 6:
                country = bic[4:6]
                enriched['lookup.country_from_bic'] = country
                applied_lookups.append(f"BIC {bic} -> country: {country}")
        
        # Lookup 2: Clearing system ID to BIC (FIXED: better search)
        clearing_id = self._find_value_by_key(payment, 'mmbid')
        if not clearing_id:
            # Try alternative search patterns
            for key, value in payment.items():
                if 'clrsys' in key.lower() and 'mmbid' in key.lower():
                    clearing_id = value
                    break
        
        if clearing_id and clearing_id in self.lookup_tables['clearing_system_to_bank']:
            clearing_info = self.lookup_tables['clearing_system_to_bank'][clearing_id]
            logger.info(f"Applying clearing system lookup for {clearing_id}")
            
            if 'bic' in clearing_info:
                resolved_bic = clearing_info['bic']
                enriched['lookup.bic_from_clearing'] = resolved_bic
                applied_lookups.append(f"Clearing ID {clearing_id} -> BIC: {resolved_bic}")
                
                # Chain lookup: use the resolved BIC to get bank info
                if resolved_bic in self.lookup_tables['bic_to_bank_info']:
                    bank_info = self.lookup_tables['bic_to_bank_info'][resolved_bic]
                    if 'name' in bank_info:
                        enriched['lookup.bank_name_from_clearing'] = bank_info['name']
                        applied_lookups.append(f"  -> name: {bank_info['name']}")
                    if 'address' in bank_info:
                        for i, addr_line in enumerate(bank_info['address']):
                            enriched[f'lookup.clearing_address_{i}'] = addr_line
                        applied_lookups.append(f"  -> address: {len(bank_info['address'])} lines")
        else:
            if clearing_id:
                logger.info(f"Clearing ID {clearing_id} not found in lookup tables")
        
        # Lookup 3: IBAN-based enrichment
        iban = self._find_value_by_key(payment, 'iban')
        if iban and len(iban) >= 2:
            country = iban[:2]
            enriched['lookup.country_from_iban'] = country
            applied_lookups.append(f"IBAN {iban[:10]}... -> country: {country}")
            
            if iban in self.lookup_tables['iban_to_info']:
                iban_info = self.lookup_tables['iban_to_info'][iban]
                for key, value in iban_info.items():
                    enriched[f'lookup.iban_{key}'] = value
        
        if applied_lookups:
            logger.info(f"Applied {len(applied_lookups)} lookup operations:")
            for lookup in applied_lookups:
                logger.info(f"  - {lookup}")
        else:
            logger.info("No lookup operations applied (no matching data found)")
        
        return enriched
    
    def _find_value_by_key(self, data: Dict, search_key: str) -> Optional[str]:
        """
        Find value by partial key match - prioritizes exact suffix matches.
        FIXED: Prevents false matches like 'clrsysmmbid.cd' when searching for 'mmbid'
        """
        search_key_lower = search_key.lower()
        
        # First pass: look for keys ending with the search key (most specific)
        for key, value in data.items():
            key_lower = key.lower()
            if key_lower.endswith('.' + search_key_lower) or key_lower == search_key_lower:
                return value
        
        # Second pass: look for keys containing the search key (fallback)
        for key, value in data.items():
            if search_key_lower in key.lower():
                return value
        
        return None


# ============================================================================
# CLI COMMANDS
# ============================================================================

def learn_command(args):
    """Phase 1: Learn lookup tables from training data"""
    logger.info("Starting LEARN phase...")
    builder = LookupTableBuilder()
    builder.learn_from_transactions(args.input)
    
    output_path = args.output or './models/lookup_tables.pkl'
    builder.save(output_path)
    logger.info(f"✓ Learn phase complete! Lookup tables saved to {output_path}")


def train_command(args):
    """Phase 2: Train neural network model"""
    logger.info("Starting TRAIN phase...")
    
    config = Config()
    config.num_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    
    vocab = Vocabulary()
    field_vocab = FieldVocabulary()
    taxonomy = ACERepairTaxonomy()
    
    parser = ACEDataParser(config, vocab, field_vocab, taxonomy)
    transactions = parser.parse_file(args.input)
    
    logger.info(f"Parsed {len(transactions)} transactions")
    
    all_texts = []
    for txn_id, repair_ids, before_data, after_data, diffs in transactions:
        for field, value in {**before_data, **after_data}.items():
            field_vocab.add_field(field, before_data.get(field) != after_data.get(field))
            if value:
                all_texts.append(str(value))
    
    vocab.build_from_texts(all_texts)
    
    logger.info(f"Vocabulary sizes:")
    logger.info(f"  - Characters: {len(vocab)}")
    logger.info(f"  - Fields: {len(field_vocab)}")
    logger.info(f"  - Repairs: {len(taxonomy)}")
    
    os.makedirs('models', exist_ok=True)
    vocab.save(config.vocab_save_path)
    field_vocab.save(config.field_vocab_save_path)
    taxonomy.save(config.repair_taxonomy_path)
    config.save('models/config.json')
    
    # FIXED: Shuffle data before splitting to prevent ordering bias
    import random
    random.seed(42)  # For reproducibility
    random.shuffle(transactions)
    
    n = len(transactions)
    train_size = int(n * config.train_split)
    val_size = int(n * config.val_split)
    
    train_txns = transactions[:train_size]
    val_txns = transactions[train_size:train_size+val_size]
    test_txns = transactions[train_size+val_size:]  # Save test set
    
    logger.info(f"Data split: Train={len(train_txns)}, Val={len(val_txns)}, Test={len(test_txns)}")
    
    train_dataset = ACERepairDataset(train_txns, parser)
    val_dataset = ACERepairDataset(val_txns, parser)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=0)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    model = ACERepairModel(config, len(vocab), len(field_vocab), len(taxonomy))
    trainer = ACETrainer(model, config, device)
    trainer.train(train_loader, val_loader)
    
    logger.info("✓ Train phase complete!")


def repair_command(args):
    """Phase 3: Apply repairs to new payment"""
    logger.info("Starting REPAIR phase...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    repairer = ACERepairer(args.model or './models', device)
    
    with open(args.input, 'r') as f:
        data = json.load(f)
    
    if len(data) == 1:
        txn_id = list(data.keys())[0]
        if txn_id.isdigit() or len(txn_id) > 10:
            logger.info(f"Processing transaction: {txn_id}")
            payment = data[txn_id]
        else:
            payment = data
    else:
        payment = data
    
    repair_ids = None
    if hasattr(args, 'repairs') and args.repairs:
        repair_ids = args.repairs.split(',')
    
    repair_threshold = getattr(args, 'repair_threshold', 0.5)
    confidence_threshold = getattr(args, 'confidence_threshold', 0.5)
    
    results = repairer.repair(
        payment, 
        repair_ids=repair_ids,
        repair_threshold=repair_threshold,
        confidence_threshold=confidence_threshold
    )
    
    logger.info("\n" + "="*70)
    logger.info("REPAIR RESULTS")
    logger.info("="*70)
    logger.info(f"Predicted/Applied Repair IDs: {results['predicted_repair_ids']}")
    logger.info(f"\nRepair Descriptions:")
    for desc in results['repair_descriptions']:
        logger.info(f"  {desc}")
    
    logger.info(f"\nApplied {len(results['applied_repairs'])} field changes:")
    for repair in results['applied_repairs'][:20]:
        logger.info(f"  {repair['field']}:")
        logger.info(f"    Before: {repair['before']}")
        logger.info(f"    After:  {repair['after']}")
        logger.info(f"    Source: {repair['source']}")
        if 'confidence' in repair:
            logger.info(f"    Confidence: {repair['confidence']:.3f}")
    
    if len(results['applied_repairs']) > 20:
        logger.info(f"  ... and {len(results['applied_repairs']) - 20} more changes")
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\n✓ Results saved to {args.output}")
    
    logger.info("✓ Repair phase complete!")


# ============================================================================
# MAIN CLI
# ============================================================================

def main():
    """Main entry point for CLI"""
    parser = argparse.ArgumentParser(
        description='ACE Payment Repair System - Three Phase Architecture',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Phase 1: Learn lookup tables
  python ace_repair_model.py learn --input repair3.json
  
  # Phase 2: Train model
  python ace_repair_model.py train --input repair3.json --epochs 100
  
  # Phase 3: Apply repairs (auto-predict)
  python ace_repair_model.py repair --input input1.json --output result.json
  
  # Phase 3: Apply repairs with lower thresholds for testing
  python ace_repair_model.py repair --input input1.json --confidence_threshold 0.3
  
  # Phase 3: Apply specific repairs
  python ace_repair_model.py repair --input input1.json --repairs 6021,6035,6036
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Learn command
    learn_parser = subparsers.add_parser(
        'learn', 
        help='Phase 1: Build lookup tables from training data'
    )
    learn_parser.add_argument(
        '--input', 
        required=True, 
        help='Input repairs JSON file'
    )
    learn_parser.add_argument(
        '--output', 
        help='Output lookup tables file (default: ./models/lookup_tables.pkl)'
    )
    
    # Train command
    train_parser = subparsers.add_parser(
        'train', 
        help='Phase 2: Train neural network model'
    )
    train_parser.add_argument(
        '--input', 
        required=True, 
        help='Input repairs JSON file'
    )
    train_parser.add_argument(
        '--epochs', 
        type=int, 
        default=100, 
        help='Number of training epochs (default: 100)'
    )
    train_parser.add_argument(
        '--batch_size', 
        type=int, 
        help='Batch size (default: 16)'
    )
    
    # Repair command
    repair_parser = subparsers.add_parser(
        'repair', 
        help='Phase 3: Apply repairs to new payment'
    )
    repair_parser.add_argument(
        '--input', 
        required=True, 
        help='Input payment JSON file'
    )
    repair_parser.add_argument(
        '--output', 
        help='Output file for repaired payment'
    )
    repair_parser.add_argument(
        '--model', 
        default='./models', 
        help='Model directory (default: ./models)'
    )
    repair_parser.add_argument(
        '--repairs', 
        help='Comma-separated repair IDs (auto-predicted if not provided)'
    )
    repair_parser.add_argument(
        '--repair_threshold', 
        type=float, 
        default=0.5, 
        help='Threshold for repair prediction (default: 0.5)'
    )
    repair_parser.add_argument(
        '--confidence_threshold', 
        type=float, 
        default=0.5, 
        help='Threshold for field change confidence (default: 0.5, lowered from 0.7)'
    )
    
    args = parser.parse_args()
    
    if args.command == 'learn':
        learn_command(args)
    elif args.command == 'train':
        train_command(args)
    elif args.command == 'repair':
        repair_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
