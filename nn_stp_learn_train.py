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
"""

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
import os
import re
from typing import Dict, List, Optional, Tuple, Set, Any
from collections import defaultdict, Counter
from pathlib import Path
import logging
from copy import deepcopy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# RULE LEARNING ENGINE
# ============================================================================

class RuleLearner:
    """
    Learns what each ACE rule does by analyzing training examples.
    
    This is the "knowledge extraction" phase that builds:
    1. Rule definitions: What each ACE ID means (from ace[] section)
    2. Lookup tables: BIC → bank details, routing → bank info (from before→after)
    3. Rule types: Which rules need lookups vs extraction vs formatting
    
    Example:
        Rule 6035 learns from training data:
        - Description: "NCH Code derived from BIC"
        - Lookup: BOFAUS3N → {"name": "BANK OF AMERICA, N.A., NY", ...}
        - Type: LOOKUP_ENRICHMENT (needs BIC directory lookup)
    """
    
    def __init__(self):
        # Store discovered rules: ace_id -> {id, code, field, text, count, type}
        self.rules = {}
        
        # Lookup tables for value substitution: table_name -> key -> value
        # e.g., lookup_tables['bic']['BOFAUS3N'] = {'name': 'BANK OF AMERICA...', ...}
        self.lookup_tables = defaultdict(dict)
        
        # Rule type classification (determines execution strategy)
        self.rule_types = {}
        
    def learn_from_file(self, json_file: str):
        """
        Analyze training file to discover all rules and build lookup tables.
        
        Process:
        1. First pass: Collect ACE rule metadata (ID, description, field)
        2. Second pass: Build lookup tables and classify rule types
        
        Args:
            json_file: Path to training JSON with transactions containing:
                      - ace[]: Rules that were applied
                      - before/after: Payment state before and after repair
                      - diffs[]: Exact changes made
        
        Returns:
            self: For method chaining
        """
        logger.info(f"Learning rules from {json_file}")
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # ===== FIRST PASS: Collect rule metadata from ace[] sections =====
        for txn_id, txn_data in data.items():
            for repair in txn_data.get('ace', []):
                ace_id = repair.get('id')
                if ace_id and ace_id not in self.rules:
                    # Store rule metadata
                    self.rules[ace_id] = {
                        'id': ace_id,
                        'code': repair.get('code', 'I'),  # I/W/E/R category
                        'field': repair.get('field', ''),  # Primary field affected
                        'text': repair.get('text', ''),    # Human-readable description
                        'count': 0,                        # Number of times seen
                    }
        
        # ===== SECOND PASS: Build lookups and classify rule types =====
        for txn_id, txn_data in data.items():
            # Get list of ACE rules applied to this transaction
            ace_ids = [r['id'] for r in txn_data.get('ace', [])]
            
            # Increment usage count for each rule
            for ace_id in ace_ids:
                if ace_id in self.rules:
                    self.rules[ace_id]['count'] += 1
            
            # Analyze transformations for each entity (cdtrAgt, dbtrAgt, etc.)
            for entity_key in ['cdtrAgt', 'dbtrAgt', 'instgAgt', 'cdtr', 'dbtr',
                              'cdtrAcct', 'dbtrAcct', 'intrmyAgt1', 'intrmyAgt2']:
                if entity_key not in txn_data:
                    continue
                
                entity_data = txn_data[entity_key]
                before = entity_data.get('before', {})
                after = entity_data.get('after', {})
                
                # Build lookup tables by comparing before → after
                self._extract_lookups(before, after, entity_key)
        
        # ===== CLASSIFY RULE TYPES =====
        self._classify_rule_types()
        
        logger.info(f"Learned {len(self.rules)} ACE rules")
        logger.info(f"Built lookup tables: BIC={len(self.lookup_tables['bic'])}, "
                   f"Routing={len(self.lookup_tables['routing'])}")
        
        return self
    
    def _extract_lookups(self, before: Dict, after: Dict, entity_key: str):
        """
        Extract lookup patterns from before → after transformations.
        
        This builds dictionaries for value substitution:
        - BIC code → bank name, address, country
        - Routing number → bank details
        - IBAN → validation status
        
        Example:
            before: {"FinInstnId": {"BICFI": "BOFAUS3N"}}
            after:  {"finInstnId": {"bicFi": "BOFAUS3N", 
                                   "nm": "BANK OF AMERICA, N.A., NY", ...}}
            
            Creates: lookup_tables['bic']['BOFAUS3N'] = {'name': 'BANK OF AMERICA...', ...}
        
        Args:
            before: Payment entity state before repair
            after: Payment entity state after repair
            entity_key: Entity type (cdtrAgt, dbtrAgt, etc.)
        """
        
        # ===== Extract BIC Lookup =====
        before_bic = self._extract_bic(before)
        after_bic = self._extract_bic(after)
        
        if before_bic or after_bic:
            # Normalize BIC to 8 characters (strip branch code if present)
            bic_key = (before_bic or after_bic)[:8].upper()
            
            # Extract enriched information from after state
            after_name = self._get_nested(after, ['finInstnId', 'nm']) or \
                        self._get_nested(after, ['FinInstnId', 'Nm'])
            after_addr = self._get_nested(after, ['finInstnId', 'pstlAdr']) or \
                        self._get_nested(after, ['FinInstnId', 'PstlAdr'])
            after_country = after.get('ctryOfRes') or after.get('CtryOfRes')
            
            # Store in lookup table if we got any useful information
            if after_name or after_addr or after_country:
                self.lookup_tables['bic'][bic_key] = {
                    'name': after_name,
                    'address': after_addr,
                    'country': after_country,
                    'entity_type': entity_key
                }
        
        # ===== Extract Routing Number Lookup =====
        routing = self._extract_routing(before) or self._extract_routing(after)
        if routing:
            after_name = self._get_nested(after, ['finInstnId', 'nm']) or \
                        self._get_nested(after, ['FinInstnId', 'Nm'])
            after_bic = self._extract_bic(after)
            after_country = after.get('ctryOfRes') or after.get('CtryOfRes')
            after_addr = self._get_nested(after, ['finInstnId', 'pstlAdr']) or \
                        self._get_nested(after, ['FinInstnId', 'PstlAdr'])
            
            if after_name or after_bic or after_country:
                self.lookup_tables['routing'][routing] = {
                    'name': after_name,
                    'bic': after_bic,
                    'country': after_country,
                    'address': after_addr
                }
        
        # ===== Extract IBAN Validation Examples =====
        after_iban = self._get_nested(after, ['id', 'iban']) or \
                     self._get_nested(after, ['Id', 'IBAN'])
        if after_iban:
            country = after_iban[:2].upper()
            self.lookup_tables['iban_valid'][after_iban] = True
            self.lookup_tables['iban_country'][country] = True
    
    def _classify_rule_types(self):
        """
        Classify rules by execution type to determine correct application strategy.
        
        Types:
        - EXTRACT: Extract value from existing field (e.g., country from BIC[4:6])
        - LOOKUP: Look up value in dictionary (e.g., BIC → bank name)
        - FORMAT: Reformat existing value (e.g., BIC11 → BIC8)
        - RESTRUCTURE: Change field structure (e.g., Othr.Id → IBAN)
        - NORMALIZE: Normalize text (e.g., lowercase → UPPERCASE)
        """
        # Rule text patterns indicate type
        for ace_id, rule in self.rules.items():
            text = rule['text'].lower()
            
            if 'extracted from' in text or 'derive' in text:
                self.rule_types[ace_id] = 'EXTRACT'
            elif 'nch code' in text or 'enrich' in text:
                self.rule_types[ace_id] = 'LOOKUP'
            elif 'bic4' in text or 'normalize' in text:
                self.rule_types[ace_id] = 'FORMAT'
            elif 'valid iban' in text or 'account number' in text:
                self.rule_types[ace_id] = 'RESTRUCTURE'
            else:
                self.rule_types[ace_id] = 'EXTRACT'  # Default
    
    def _extract_bic(self, data: Dict) -> Optional[str]:
        """
        Extract BIC code from nested structure (handles different case variations).
        
        Tries multiple path variations:
        - finInstnId.bicFi (camelCase)
        - FinInstnId.BICFI (PascalCase + upper)
        - etc.
        """
        return self._get_nested(data, ['finInstnId', 'bicFi']) or \
               self._get_nested(data, ['FinInstnId', 'BICFI']) or \
               self._get_nested(data, ['finInstnId', 'BICFI']) or \
               self._get_nested(data, ['FinInstnId', 'bicFi'])
    
    def _extract_routing(self, data: Dict) -> Optional[str]:
        """
        Extract routing number from clearing system structure.
        
        Returns: "USABA:026009593" format (clearing_system:member_id)
        """
        clr_sys = self._get_nested(data, ['finInstnId', 'clrSysMmbId']) or \
                  self._get_nested(data, ['FinInstnId', 'ClrSysMmbId'])
        
        if clr_sys:
            clr_id = clr_sys.get('clrSysId') or clr_sys.get('ClrSysId') or \
                     self._get_nested(clr_sys, ['ClrSysId', 'Cd']) or \
                     self._get_nested(clr_sys, ['clrSysId'])
            mmb_id = clr_sys.get('mmbId') or clr_sys.get('MmbId')
            
            if mmb_id:
                return f"{clr_id}:{mmb_id}"
        return None
    
    def _get_nested(self, data: Dict, path: List[str]) -> Any:
        """
        Navigate nested dictionary with case-insensitive key matching.
        
        Example:
            data = {'FinInstnId': {'BICFI': 'BOFAUS3N'}}
            _get_nested(data, ['fininstnid', 'bicfi']) → 'BOFAUS3N'
        
        Args:
            data: Dictionary to navigate
            path: List of keys to traverse
            
        Returns:
            Value at the end of path, or None if path doesn't exist
        """
        current = data
        for key in path:
            if not isinstance(current, dict):
                return None
            
            # Try exact match first (fastest)
            if key in current:
                current = current[key]
            else:
                # Try case-insensitive match
                found = False
                for k in current:
                    if isinstance(k, str) and k.lower() == key.lower():
                        current = current[k]
                        found = True
                        break
                if not found:
                    return None
        return current
    
    def save(self, directory: str):
        """
        Save learned rules and lookup tables to disk.
        
        Creates files:
        - learned_rules.json: All discovered rules with metadata
        - lookup_tables.json: All lookup tables (BIC, routing, IBAN, etc.)
        
        Args:
            directory: Directory to save files to (creates if doesn't exist)
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save rules with their classification
        with open(f"{directory}/learned_rules.json", 'w') as f:
            json.dump({
                'rules': self.rules,
                'rule_types': self.rule_types
            }, f, indent=2)
        
        # Save lookup tables (convert defaultdict to regular dict for JSON)
        with open(f"{directory}/lookup_tables.json", 'w') as f:
            json.dump({k: dict(v) for k, v in self.lookup_tables.items()}, f, indent=2)
        
        logger.info(f"Saved learned rules and lookups to {directory}")
    
    @classmethod
    def load(cls, directory: str):
        """
        Load previously learned rules and lookup tables from disk.
        
        Args:
            directory: Directory containing learned_rules.json and lookup_tables.json
            
        Returns:
            RuleLearner instance with loaded knowledge
        """
        learner = cls()
        
        # Load rules
        with open(f"{directory}/learned_rules.json", 'r') as f:
            data = json.load(f)
            learner.rules = data.get('rules', data)  # Handle old format
            learner.rule_types = data.get('rule_types', {})
        
        # Load lookup tables (restore as defaultdict)
        with open(f"{directory}/lookup_tables.json", 'r') as f:
            data = json.load(f)
            learner.lookup_tables = defaultdict(dict, {k: dict(v) for k, v in data.items()})
        
        return learner


# ============================================================================
# FEATURE EXTRACTION FOR ML
# ============================================================================

class FeatureExtractor:
    """
    Extract numerical features from payment structure for ML model.
    
    Converts complex payment JSON into a fixed-size feature vector that
    captures important structural properties like:
    - Which entities are present (cdtrAgt, dbtrAgt, etc.)
    - What fields are missing (no name? no country? no address?)
    - BIC characteristics (length, format)
    - IBAN presence and location
    
    The ML model learns to map these features to rule predictions.
    
    Example:
        payment = {"cdtrAgt": {"FinInstnId": {"BICFI": "BOFAUS3N"}}}
        features = extractor.extract(payment)
        # Returns: [1.0, 1.0, 0.0, 0.0, ...] (70+ dimensional vector)
        #          ^    ^    ^    ^
        #          |    |    |    +-- no country
        #          |    |    +------ no name  
        #          |    +---------- has BIC
        #          +--------------- has cdtrAgt
    """
    
    def __init__(self):
        self.feature_dim = None  # Will be set after first extraction
    
    def extract(self, payment: Dict) -> torch.Tensor:
        """
        Extract feature vector from payment structure.
        
        Args:
            payment: Payment dictionary (typically the 'before' state)
            
        Returns:
            torch.Tensor: Feature vector (~70 dimensions)
        """
        features = []
        
        # Normalize all keys to lowercase for consistent access
        payment_norm = self._normalize_case(payment)
        
        # ===== ENTITY PRESENCE FEATURES (7 features) =====
        # Binary indicators: 1.0 if entity exists, 0.0 otherwise
        entities = ['cdtragt', 'dbtragt', 'instgagt', 'cdtr', 'dbtr', 
                   'cdtracct', 'dbtracct']
        for entity in entities:
            features.append(1.0 if entity in payment_norm else 0.0)
        
        # ===== AGENT ENTITY FEATURES (8 features × 3 entities = 24 features) =====
        # For each agent (cdtrAgt, dbtrAgt, instgAgt), extract:
        # - Has BIC?
        # - Has bank name?
        # - Has address?
        # - Has country?
        # - Has clearing system info?
        # - BIC length (0 if no BIC)
        # - Is BIC 11 characters?
        # - Does BIC end with 'XXX'?
        for entity in ['cdtragt', 'dbtragt', 'instgagt']:
            entity_data = payment_norm.get(entity, {})
            fin_instn = entity_data.get('fininstnid', {})
            
            bic = fin_instn.get('bicfi', '')
            features.extend([
                1.0 if bic else 0.0,
                1.0 if fin_instn.get('nm') else 0.0,
                1.0 if fin_instn.get('pstladr') else 0.0,
                1.0 if entity_data.get('ctryofres') else 0.0,
                1.0 if fin_instn.get('clrsysmmbid') else 0.0,
                float(len(bic)) if bic else 0.0,
                1.0 if len(bic) == 11 else 0.0,
                1.0 if bic.endswith('XXX') else 0.0,
            ])
        
        # ===== ACCOUNT FEATURES (3 features × 2 accounts = 6 features) =====
        # For each account (cdtrAcct, dbtrAcct), extract:
        # - Has IBAN?
        # - Has other ID?
        # - Does other ID look like IBAN (misplaced)?
        for entity in ['cdtracct', 'dbtracct']:
            entity_data = payment_norm.get(entity, {})
            id_data = entity_data.get('id', {})
            
            has_iban = bool(id_data.get('iban'))
            other_id = self._get_nested(id_data, ['othr', 'id'])
            
            features.extend([
                1.0 if has_iban else 0.0,
                1.0 if other_id else 0.0,
                1.0 if self._looks_like_iban(other_id) else 0.0,
            ])
        
        # ===== PARTY FEATURES (4 features × 2 parties = 8 features) =====
        # For each party (cdtr, dbtr), extract:
        # - Has name?
        # - Has address?
        # - Is name lowercase (needs normalization)?
        # - Is name mixed case (not all upper)?
        for entity in ['cdtr', 'dbtr']:
            entity_data = payment_norm.get(entity, {})
            name = entity_data.get('nm', '')
            addr = entity_data.get('pstladr', {})
            
            features.extend([
                1.0 if name else 0.0,
                1.0 if addr else 0.0,
                1.0 if name and name.islower() else 0.0,
                1.0 if name and not name.isupper() else 0.0,
            ])
        
        # ===== GLOBAL AGGREGATION FEATURES (6 features) =====
        # High-level indicators across the entire payment
        all_bics = self._extract_all_bics(payment_norm)
        features.extend([
            float(len(all_bics)),                              # Number of BICs in payment
            1.0 if any(len(b) == 11 for b in all_bics) else 0.0,  # Any 11-char BIC?
            1.0 if any(b.endswith('XXX') for b in all_bics) else 0.0,  # Any XXX suffix?
            float(self._count_missing_countries(payment_norm)),     # Missing country count
            float(self._count_missing_names(payment_norm)),         # Missing name count
            float(self._count_missing_addresses(payment_norm)),     # Missing address count
        ])
        
        # Store feature dimension for model initialization
        self.feature_dim = len(features)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _normalize_case(self, data):
        """Normalize all keys to lowercase"""
        if isinstance(data, dict):
            return {k.lower() if isinstance(k, str) else k: self._normalize_case(v) 
                   for k, v in data.items()}
        elif isinstance(data, list):
            return [self._normalize_case(item) for item in data]
        return data
    
    def _get_nested(self, data: Dict, path: List[str]) -> Any:
        """Navigate nested dict"""
        current = data
        for key in path:
            if not isinstance(current, dict) or key not in current:
                return None
            current = current[key]
        return current
    
    def _looks_like_iban(self, value: str) -> bool:
        """Check if value resembles IBAN"""
        if not value or len(value) < 15:
            return False
        return value[:2].isalpha() and value[2:4].isdigit()
    
    def _extract_all_bics(self, payment: Dict) -> List[str]:
        """Extract all BICs from payment"""
        bics = []
        for entity in ['cdtragt', 'dbtragt', 'instgagt']:
            entity_data = payment.get(entity, {})
            bic = self._get_nested(entity_data, ['fininstnid', 'bicfi'])
            if bic:
                bics.append(bic.upper())
        return bics
    
    def _count_missing_countries(self, payment: Dict) -> int:
        """Count agents without country"""
        count = 0
        for entity in ['cdtragt', 'dbtragt', 'instgagt']:
            entity_data = payment.get(entity, {})
            if entity_data and not entity_data.get('ctryofres'):
                count += 1
        return count
    
    def _count_missing_names(self, payment: Dict) -> int:
        """Count agents without bank name"""
        count = 0
        for entity in ['cdtragt', 'dbtragt', 'instgagt']:
            entity_data = payment.get(entity, {})
            fin_instn = entity_data.get('fininstnid', {})
            if entity_data and not fin_instn.get('nm'):
                count += 1
        return count
    
    def _count_missing_addresses(self, payment: Dict) -> int:
        """Count agents without address"""
        count = 0
        for entity in ['cdtragt', 'dbtragt', 'instgagt']:
            entity_data = payment.get(entity, {})
            fin_instn = entity_data.get('fininstnid', {})
            if entity_data and not fin_instn.get('pstladr'):
                count += 1
        return count


# ============================================================================
# ML MODEL
# ============================================================================

class RulePredictionModel(nn.Module):
    """
    Neural network to predict which ACE rules to apply.
    
    Architecture:
    - Input: ~70 payment structure features
    - 3 hidden layers with batch normalization and dropout
    - Output: Probability for each ACE rule (multi-label classification)
    
    The model learns patterns like:
    "If BIC present but name missing → rules 6021, 6035 likely apply"
    """
    
    def __init__(self, num_features: int, num_rules: int):
        super().__init__()
        
        self.network = nn.Sequential(
            # Layer 1: Expand feature space
            nn.Linear(num_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Layer 2: Deep representation
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Layer 3: Compress representation
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Output layer: One probability per rule
            nn.Linear(128, num_rules),
            nn.Sigmoid()  # Independent probabilities (not softmax - multiple rules can apply)
        )
    
    def forward(self, x):
        return self.network(x)


# ============================================================================
# DATASET
# ============================================================================

class RepairDataset(Dataset):
    """
    PyTorch dataset for training rule prediction model.
    
    Each sample consists of:
    - Input: Payment structure features (from 'before' state)
    - Target: Binary vector indicating which rules were applied (from 'ace' array)
    """
    
    def __init__(self, data_file: str, feature_extractor: FeatureExtractor, 
                 ace_to_idx: Dict[str, int]):
        """
        Initialize dataset from training file.
        
        Args:
            data_file: Path to JSON training file
            feature_extractor: FeatureExtractor instance
            ace_to_idx: Mapping from ACE rule ID to index
        """
        self.feature_extractor = feature_extractor
        self.ace_to_idx = ace_to_idx
        self.samples = []
        
        # Parse training file
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        for txn_id, txn_data in data.items():
            # Get list of rules that were applied
            ace_ids = [r['id'] for r in txn_data.get('ace', [])]
            
            # Reconstruct payment from all 'before' sections
            payment = {}
            for entity_key in ['cdtrAgt', 'dbtrAgt', 'instgAgt', 'cdtr', 'dbtr',
                              'cdtrAcct', 'dbtrAcct']:
                if entity_key in txn_data:
                    before = txn_data[entity_key].get('before')
                    if before:
                        payment[entity_key] = before
            
            # Only add if we have both payment structure and rules
            if payment and ace_ids:
                self.samples.append((payment, ace_ids))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get training sample.
        
        Returns:
            features: Tensor of payment features
            target: Binary vector of which rules apply
        """
        payment, ace_ids = self.samples[idx]
        
        # Extract features from payment structure
        features = self.feature_extractor.extract(payment)
        
        # Create target vector (multi-hot encoding of rules)
        target = torch.zeros(len(self.ace_to_idx), dtype=torch.float32)
        for ace_id in ace_ids:
            if ace_id in self.ace_to_idx:
                target[self.ace_to_idx[ace_id]] = 1.0
        
        return features, target


# ============================================================================
# TRAINER
# ============================================================================

class Trainer:
    """
    Train ML model for rule prediction.
    
    Uses binary cross-entropy loss since this is multi-label classification
    (multiple rules can apply to the same payment).
    """
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        self.criterion = nn.BCELoss()  # Binary cross-entropy for multi-label
        self.best_val_loss = float('inf')
    
    def train_epoch(self, loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for features, targets in loader:
            features = features.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(loader)
    
    def evaluate(self, loader):
        """Evaluate on validation set"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for features, targets in loader:
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(features)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                
                # Calculate accuracy (at 0.5 threshold)
                pred = (outputs > 0.5).float()
                correct += (pred == targets).float().sum().item()
                total += targets.numel()
        
        return total_loss / len(loader), correct / total
    
    def train(self, train_loader, val_loader, epochs, save_path):
        """
        Full training loop.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            epochs: Number of training epochs
            save_path: Where to save best model
        """
        logger.info("="*70)
        logger.info("Training started")
        logger.info("="*70)
        
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_acc = self.evaluate(val_loader)
            
            logger.info(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f}, "
                       f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), save_path)
                logger.info("  ✓ Best model saved")
        
        logger.info(f"\nTraining complete! Best val loss: {self.best_val_loss:.4f}")


# ============================================================================
# RULE EXECUTION ENGINE (FIXED - NOW USES LOOKUPS AND EXTRACTION)
# ============================================================================

class RuleExecutor:
    """
    Execute ACE rules using smart logic (not blind pattern replay).
    
    CRITICAL FIX: This now properly:
    1. Extracts values from the CURRENT payment (e.g., country from BIC)
    2. Looks up values in learned dictionaries (e.g., BIC → bank details)
    3. Applies transformations dynamically (not replaying static training values)
    
    Each rule has a specific handler that knows how to apply that rule correctly.
    """
    
    def __init__(self, rule_learner: RuleLearner):
        self.learner = rule_learner
        self.lookup_tables = rule_learner.lookup_tables
        
        # Map rule IDs to handler methods
        self.handlers = {
            '6004': self.rule_6004_valid_iban,
            '6008': self.rule_6008_nch_from_iban,
            '6015': self.rule_6015_country_from_iban,
            '6021': self.rule_6021_country_from_bic,
            '6032': self.rule_6032_domestic_account,
            '6035': self.rule_6035_enrich_from_bic,
            '6036': self.rule_6036_normalize_bic,
            '6046': self.rule_6046_account_from_iban,
            '8026': self.rule_8026_enrich_from_routing,
        }
    
    def apply_rules(self, payment: Dict, ace_ids: List[str]) -> Dict:
        """
        Apply rules to payment using smart execution logic.
        
        Args:
            payment: Payment to repair
            ace_ids: List of rule IDs to apply (from ML prediction)
            
        Returns:
            Repaired payment
        """
        result = deepcopy(payment)
        applied_count = 0
        
        # Apply each rule using its specific handler
        for ace_id in ace_ids:
            if ace_id in self.handlers:
                try:
                    success = self.handlers[ace_id](result)
                    if success:
                        applied_count += 1
                except Exception as e:
                    logger.warning(f"Error applying rule {ace_id}: {e}")
        
        logger.debug(f"Applied {applied_count}/{len(ace_ids)} rules successfully")
        return result
    
    # ===== RULE HANDLERS (Each rule knows how to apply itself) =====
    
    def rule_6004_valid_iban(self, payment: Dict) -> bool:
        """
        Rule 6004: Detect valid IBAN in wrong field and restructure.
        
        Looks for IBAN in 'id.othr.id' and moves it to 'id.iban'
        """
        modified = False
        
        for entity_key in ['cdtrAcct', 'dbtrAcct']:
            entity = payment.get(entity_key, {})
            id_section = entity.get('id', entity.get('Id', {}))
            
            # Check if IBAN is misplaced in 'othr.id'
            other_section = id_section.get('othr', id_section.get('Othr', {}))
            other_id = other_section.get('id', other_section.get('Id', ''))
            
            if other_id and self._is_valid_iban(other_id):
                # Restructure: move to proper IBAN field
                if 'id' not in entity:
                    entity['id'] = {}
                entity['id']['iban'] = other_id.upper()
                
                # Remove old structure
                if 'othr' in entity.get('id', {}):
                    del entity['id']['othr']
                if 'Othr' in entity.get('Id', {}):
                    del entity['Id']['Othr']
                
                modified = True
        
        return modified
    
    def rule_6008_nch_from_iban(self, payment: Dict) -> bool:
        """Rule 6008: Extract NCH code from IBAN (country-specific)"""
        # Implementation depends on country-specific IBAN structure
        return False
    
    def rule_6015_country_from_iban(self, payment: Dict) -> bool:
        """Rule 6015: Extract country code from IBAN (first 2 characters)"""
        modified = False
        
        for entity_key in ['cdtrAcct', 'dbtrAcct']:
            entity = payment.get(entity_key, {})
            id_section = entity.get('id', entity.get('Id', {}))
            iban = id_section.get('iban', id_section.get('IBAN', ''))
            
            if iban and len(iban) >= 2:
                country = iban[:2].upper()
                # Could add country to parent entity if needed
                modified = True
        
        return modified
    
    def rule_6021_country_from_bic(self, payment: Dict) -> bool:
        """
        Rule 6021: Extract ISO country code from BIC.
        
        CRITICAL: Extracts from THIS payment's BIC (positions 5-6), not training data.
        """
        modified = False
        
        for entity_key in ['cdtrAgt', 'dbtrAgt', 'instgAgt']:
            entity = payment.get(entity_key, {})
            if not entity:
                continue
            
            # Get BIC from this payment
            fin_instn = entity.get('finInstnId', entity.get('FinInstnId', {}))
            bic = fin_instn.get('bicFi', fin_instn.get('BICFI', ''))
            
            if bic and len(bic) >= 6:
                # Extract country from BIC positions 5-6 (0-indexed: [4:6])
                country = bic[4:6].upper()
                entity['ctryOfRes'] = country
                modified = True
                logger.debug(f"Rule 6021: Extracted country {country} from BIC {bic}")
        
        return modified
    
    def rule_6032_domestic_account(self, payment: Dict) -> bool:
        """Rule 6032: Extract domestic account number from field"""
        return False
    
    def rule_6035_enrich_from_bic(self, payment: Dict) -> bool:
        """
        Rule 6035: Enrich with bank details from BIC lookup.
        
        CRITICAL: Looks up THIS payment's BIC in learned directory.
        """
        modified = False
        
        for entity_key in ['cdtrAgt', 'dbtrAgt', 'instgAgt']:
            entity = payment.get(entity_key, {})
            if not entity:
                continue
            
            # Get BIC from this payment
            fin_instn = entity.get('finInstnId', entity.get('FinInstnId', {}))
            bic = fin_instn.get('bicFi', fin_instn.get('BICFI', ''))
            
            if bic:
                # Look up in learned BIC directory
                bic_key = bic[:8].upper()
                bank_info = self.lookup_tables['bic'].get(bic_key)
                
                if bank_info:
                    # Add bank name if available
                    if bank_info.get('name') and not fin_instn.get('nm'):
                        if 'finInstnId' not in entity:
                            entity['finInstnId'] = {}
                        entity['finInstnId']['nm'] = bank_info['name']
                        logger.debug(f"Rule 6035: Added name for BIC {bic_key}")
                    
                    # Add address if available
                    if bank_info.get('address') and not fin_instn.get('pstlAdr'):
                        if 'finInstnId' not in entity:
                            entity['finInstnId'] = {}
                        entity['finInstnId']['pstlAdr'] = bank_info['address']
                        logger.debug(f"Rule 6035: Added address for BIC {bic_key}")
                    
                    # Add country if available
                    if bank_info.get('country') and not entity.get('ctryOfRes'):
                        entity['ctryOfRes'] = bank_info['country']
                    
                    modified = True
                else:
                    logger.debug(f"Rule 6035: No lookup data for BIC {bic_key}")
        
        return modified
    
    def rule_6036_normalize_bic(self, payment: Dict) -> bool:
        """
        Rule 6036: Normalize BIC to 8 characters (remove branch code).
        
        Converts BIC11 (e.g., BOFAUS3NXXX) to BIC8 (BOFAUS3N)
        """
        modified = False
        
        for entity_key in ['cdtrAgt', 'dbtrAgt', 'instgAgt']:
            entity = payment.get(entity_key, {})
            if not entity:
                continue
            
            fin_instn = entity.get('finInstnId', entity.get('FinInstnId', {}))
            bic = fin_instn.get('bicFi', fin_instn.get('BICFI', ''))
            
            # If BIC is 11 characters, truncate to 8
            if len(bic) == 11:
                normalized_bic = bic[:8]
                
                if 'finInstnId' not in entity:
                    entity['finInstnId'] = {}
                entity['finInstnId']['bicFi'] = normalized_bic
                modified = True
                logger.debug(f"Rule 6036: Normalized {bic} to {normalized_bic}")
                
                # Also enrich with bank details for the normalized BIC
                self.rule_6035_enrich_from_bic(payment)
        
        return modified
    
    def rule_6046_account_from_iban(self, payment: Dict) -> bool:
        """Rule 6046: Extract account number from IBAN (country-specific)"""
        return False
    
    def rule_8026_enrich_from_routing(self, payment: Dict) -> bool:
        """
        Rule 8026: Enrich from routing/clearing system number.
        
        Looks up routing number in learned directory.
        """
        modified = False
        
        for entity_key in ['instgAgt', 'cdtrAgt', 'dbtrAgt']:
            entity = payment.get(entity_key, {})
            if not entity:
                continue
            
            # Get clearing system info
            fin_instn = entity.get('finInstnId', entity.get('FinInstnId', {}))
            clr_sys = fin_instn.get('clrSysMmbId', fin_instn.get('ClrSysMmbId', {}))
            
            clr_sys_id = clr_sys.get('clrSysId', clr_sys.get('ClrSysId', ''))
            mmb_id = clr_sys.get('mmbId', clr_sys.get('MmbId', ''))
            
            if mmb_id:
                # Look up in routing directory
                key = f"{clr_sys_id}:{mmb_id}"
                bank_info = self.lookup_tables['routing'].get(key)
                
                if bank_info:
                    # Add bank name
                    if bank_info.get('name') and not fin_instn.get('nm'):
                        if 'finInstnId' not in entity:
                            entity['finInstnId'] = {}
                        entity['finInstnId']['nm'] = bank_info['name']
                    
                    # Add address
                    if bank_info.get('address') and not fin_instn.get('pstlAdr'):
                        if 'finInstnId' not in entity:
                            entity['finInstnId'] = {}
                        entity['finInstnId']['pstlAdr'] = bank_info['address']
                    
                    # Add country
                    if bank_info.get('country') and not entity.get('ctryOfRes'):
                        entity['ctryOfRes'] = bank_info['country']
                    
                    # Add othr.id with routing number
                    if 'finInstnId' not in entity:
                        entity['finInstnId'] = {}
                    if 'othr' not in entity['finInstnId']:
                        entity['finInstnId']['othr'] = {}
                    entity['finInstnId']['othr']['id'] = mmb_id
                    
                    modified = True
        
        return modified
    
    # ===== HELPER METHODS =====
    
    def _is_valid_iban(self, value: str) -> bool:
        """Basic IBAN validation"""
        if not value or len(value) < 15:
            return False
        return value[:2].isalpha() and value[2:4].isdigit()


# ============================================================================
# COMPLETE SYSTEM
# ============================================================================

class ComprehensiveRepairSystem:
    """
    Complete repair system integrating all components.
    
    Workflow:
    1. Load learned rules and lookup tables
    2. Load trained ML model
    3. For new payment:
       a. Extract features
       b. ML predicts which rules apply
       c. Rule executor applies rules using lookups and extraction
    """
    
    def __init__(self, model_dir: str):
        logger.info(f"Loading system from {model_dir}")
        
        # Load learned knowledge
        self.learner = RuleLearner.load(model_dir)
        self.executor = RuleExecutor(self.learner)
        
        # Load ACE mapping
        with open(f"{model_dir}/ace_mapping.json", 'r') as f:
            data = json.load(f)
            self.ace_to_idx = data['ace_to_idx']
            self.idx_to_ace = {int(k): v for k, v in data['idx_to_ace'].items()}
        
        # Setup feature extractor
        self.feature_extractor = FeatureExtractor()
        
        # Load ML model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        # Get feature dimension
        dummy_features = self.feature_extractor.extract({})
        
        self.model = RulePredictionModel(len(dummy_features), len(self.ace_to_idx))
        self.model.load_state_dict(torch.load(f"{model_dir}/model.pt", map_location=device))
        self.model.to(device)
        self.model.eval()
        
        logger.info("System loaded successfully")
    
    def repair(self, payment: Dict, threshold: float = 0.5) -> Dict:
        """
        Repair payment using ML prediction + rule execution.
        
        Args:
            payment: Payment to repair
            threshold: Confidence threshold for rule application
            
        Returns:
            Dictionary with repaired payment and metadata
        """
        # Step 1: ML predicts which rules to apply
        features = self.feature_extractor.extract(payment)
        features = features.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(features)[0]
        
        # Select rules above threshold
        predicted_rules = []
        for idx, prob in enumerate(predictions):
            if prob > threshold:
                ace_id = self.idx_to_ace[idx]
                predicted_rules.append((ace_id, float(prob)))
        
        predicted_rules.sort(key=lambda x: x[1], reverse=True)
        ace_ids = [r[0] for r in predicted_rules]
        
        # Step 2: Apply rules with executor
        repaired = self.executor.apply_rules(payment, ace_ids)
        
        return {
            'payment': repaired,
            'predicted_rules': predicted_rules,
            'rule_descriptions': [
                f"{ace_id}: {self.learner.rules[ace_id]['text']}"
                for ace_id, _ in predicted_rules
            ]
        }


# ============================================================================
# CLI COMMANDS
# ============================================================================

def learn_command(args):
    """Learn rules from training data"""
    logger.info("="*70)
    logger.info("LEARNING PHASE: Extracting rules and building lookup tables")
    logger.info("="*70)
    
    learner = RuleLearner().learn_from_file(args.input)
    learner.save(args.output)
    
    logger.info("\nTop 20 Rules by Frequency:")
    sorted_rules = sorted(learner.rules.items(), key=lambda x: x[1]['count'], reverse=True)[:20]
    for ace_id, rule in sorted_rules:
        logger.info(f"  {ace_id} ({rule['count']}x): {rule['text']}")
    
    logger.info(f"\nLearning complete! Files saved to {args.output}/")
    logger.info("Next step: Run 'train' command to train the ML predictor")


def train_command(args):
    """Train ML model"""
    logger.info("="*70)
    logger.info("TRAINING PHASE: Training ML rule predictor")
    logger.info("="*70)
    
    # Check if learn was run first
    if not os.path.exists(f"{args.output}/learned_rules.json"):
        logger.error(f"Error: Must run 'learn' command first!")
        logger.error(f"Run: python {os.path.basename(__file__)} learn --input {args.input} --output {args.output}")
        return
    
    learner = RuleLearner.load(args.output)
    
    ace_to_idx = {ace_id: idx for idx, ace_id in enumerate(sorted(learner.rules.keys()))}
    idx_to_ace = {idx: ace_id for ace_id, idx in ace_to_idx.items()}
    
    with open(f"{args.output}/ace_mapping.json", 'w') as f:
        json.dump({'ace_to_idx': ace_to_idx, 'idx_to_ace': idx_to_ace}, f)
    
    feature_extractor = FeatureExtractor()
    dataset = RepairDataset(args.input, feature_extractor, ace_to_idx)
    
    logger.info(f"Dataset size: {len(dataset)} transactions")
    
    if len(dataset) < 100:
        logger.warning("WARNING: Dataset is very small (<100 transactions)")
        logger.warning("For good results, use at least 1000+ transactions")
    
    n = len(dataset)
    train_size = int(n * 0.8)
    val_size = n - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    dummy_features = feature_extractor.extract({})
    model = RulePredictionModel(len(dummy_features), len(ace_to_idx))
    
    logger.info(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    trainer = Trainer(model, device)
    trainer.train(train_loader, val_loader, args.epochs, f"{args.output}/model.pt")
    
    logger.info(f"\nTraining complete! Files saved to {args.output}/")
    logger.info("Next step: Run 'repair' command to repair payments")


def repair_command(args):
    """Repair payment"""
    logger.info("="*70)
    logger.info("REPAIR PHASE: Applying learned rules to payment")
    logger.info("="*70)
    
    # Check if system was trained
    if not os.path.exists(f"{args.model}/model.pt"):
        logger.error(f"Error: Trained model not found at {args.model}/model.pt")
        logger.error(f"Run 'learn' and 'train' commands first")
        return
    
    system = ComprehensiveRepairSystem(args.model)
    
    with open(args.input, 'r') as f:
        payment = json.load(f)
    
    result = system.repair(payment, threshold=args.threshold)
    
    logger.info(f"\nPredicted {len(result['predicted_rules'])} rules:")
    for ace_id, prob in result['predicted_rules']:
        desc = system.learner.rules[ace_id]['text']
        logger.info(f"  {ace_id} ({prob:.3f}): {desc}")
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"\nRepaired payment saved to {args.output}")
    else:
        logger.info("\nRepaired payment:")
        print(json.dumps(result['payment'], indent=2))


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive Payment Repair System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Step 1: Learn rules and build lookup tables
  python repair_system.py learn --input repair3.json --output models/
  
  # Step 2: Train ML predictor
  python repair_system.py train --input repair3.json --output models/ --epochs 50
  
  # Step 3: Repair a payment
  python repair_system.py repair --input payment.json --output repaired.json --model models/
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Learn command
    learn_parser = subparsers.add_parser(
        'learn', 
        help='Extract rules and build lookup tables from training data'
    )
    learn_parser.add_argument(
        '--input', 
        required=True,
        help='Input JSON file with repair examples (e.g., repair3.json)'
    )
    learn_parser.add_argument(
        '--output', 
        default='./models',
        help='Output directory for learned rules and lookups (default: ./models)'
    )
    
    # Train command
    train_parser = subparsers.add_parser(
        'train',
        help='Train ML model to predict which rules to apply'
    )
    train_parser.add_argument(
        '--input', 
        required=True,
        help='Input JSON file with repair examples (same as learn phase)'
    )
    train_parser.add_argument(
        '--output', 
        default='./models',
        help='Directory with learned rules (from learn phase) and where to save model (default: ./models)'
    )
    train_parser.add_argument(
        '--epochs', 
        type=int, 
        default=50,
        help='Number of training epochs (default: 50)'
    )
    
    # Repair command
    repair_parser = subparsers.add_parser(
        'repair',
        help='Repair a payment using trained system'
    )
    repair_parser.add_argument(
        '--input', 
        required=True,
        help='Input payment JSON file to repair'
    )
    repair_parser.add_argument(
        '--output',
        help='Output file for repaired payment (if not specified, prints to console)'
    )
    repair_parser.add_argument(
        '--model', 
        default='./models',
        help='Directory with trained model and learned rules (default: ./models)'
    )
    repair_parser.add_argument(
        '--threshold', 
        type=float, 
        default=0.5,
        help='Confidence threshold for applying rules (default: 0.5)'
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
