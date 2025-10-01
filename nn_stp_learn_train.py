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
    3. Transformation patterns: Exact field changes (from diffs[] section)
    
    Example:
        Rule 6035 learns from training data:
        - Description: "NCH Code derived from BIC"
        - Lookup: BOFAUS3N → {"name": "BANK OF AMERICA, N.A., NY", ...}
        - Pattern: Add field "finInstnId.nm" with value from BIC lookup
    """
    
    def __init__(self):
        # Store discovered rules: ace_id -> {id, code, field, text, count, patterns}
        self.rules = {}
        
        # Lookup tables for value substitution: table_name -> key -> value
        # e.g., lookup_tables['bic']['BOFAUS3N'] = {'name': 'BANK OF AMERICA...', ...}
        self.lookup_tables = defaultdict(dict)
        
        # Transformation patterns: ace_id -> list of field transformations
        # e.g., rule_patterns['6035'] = [{'field_path': 'finInstnId.nm', 'operation': 'added'}, ...]
        self.rule_patterns = defaultdict(list)
        
    def learn_from_file(self, json_file: str):
        """
        Analyze training file to discover all rules and build lookup tables.
        
        Process:
        1. First pass: Collect ACE rule metadata (ID, description, field)
        2. Second pass: Learn transformations and build lookups
        
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
                        'patterns': []                     # Transformation patterns
                    }
        
        # ===== SECOND PASS: Learn transformations and build lookups =====
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
                diffs = entity_data.get('diffs', [])
                
                # Associate diffs with ACE rules that were applied
                for ace_id in ace_ids:
                    for diff in diffs:
                        # Record transformation pattern for this rule
                        pattern = {
                            'entity': entity_key,                  # Which entity (cdtrAgt, etc.)
                            'field_path': diff.get('key', ''),     # Nested field path (finInstnId.nm)
                            'operation': diff.get('msg', ''),      # added/transformed/removed
                            'value': diff.get('val', ''),          # The new value
                            'description': diff.get('desc', '')    # Human-readable description
                        }
                        self.rules[ace_id]['patterns'].append(pattern)
                        self.rule_patterns[ace_id].append(pattern)
                
                # Build lookup tables by comparing before → after
                self._extract_lookups(before, after, entity_key)
        
        logger.info(f"Learned {len(self.rules)} ACE rules")
        logger.info(f"Built {len(self.lookup_tables)} lookup tables")
        
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
        
        Creates two files:
        - learned_rules.json: All discovered rules with their patterns
        - lookup_tables.json: All lookup tables (BIC, routing, IBAN, etc.)
        
        Args:
            directory: Directory to save files to (creates if doesn't exist)
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save rules with all their transformation patterns
        with open(f"{directory}/learned_rules.json", 'w') as f:
            json.dump(self.rules, f, indent=2)
        
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
            learner.rules = json.load(f)
        
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
    """Predict which ACE rules to apply"""
    
    def __init__(self, num_features: int, num_rules: int):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, num_rules),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)


# ============================================================================
# DATASET
# ============================================================================

class RepairDataset(Dataset):
    """Dataset for training"""
    
    def __init__(self, data_file: str, feature_extractor: FeatureExtractor, 
                 ace_to_idx: Dict[str, int]):
        self.feature_extractor = feature_extractor
        self.ace_to_idx = ace_to_idx
        self.samples = []
        
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        for txn_id, txn_data in data.items():
            ace_ids = [r['id'] for r in txn_data.get('ace', [])]
            
            # Reconstruct payment from all 'before' sections
            payment = {}
            for entity_key in ['cdtrAgt', 'dbtrAgt', 'instgAgt', 'cdtr', 'dbtr',
                              'cdtrAcct', 'dbtrAcct']:
                if entity_key in txn_data:
                    before = txn_data[entity_key].get('before')
                    if before:
                        payment[entity_key] = before
            
            if payment and ace_ids:
                self.samples.append((payment, ace_ids))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        payment, ace_ids = self.samples[idx]
        
        features = self.feature_extractor.extract(payment)
        
        target = torch.zeros(len(self.ace_to_idx), dtype=torch.float32)
        for ace_id in ace_ids:
            if ace_id in self.ace_to_idx:
                target[self.ace_to_idx[ace_id]] = 1.0
        
        return features, target


# ============================================================================
# TRAINER
# ============================================================================

class Trainer:
    """Train ML model"""
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.criterion = nn.BCELoss()
    
    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        
        for features, targets in loader:
            features = features.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(loader)
    
    def evaluate(self, loader):
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
                
                pred = (outputs > 0.5).float()
                correct += (pred == targets).float().sum().item()
                total += targets.numel()
        
        return total_loss / len(loader), correct / total
    
    def train(self, train_loader, val_loader, epochs, save_path):
        logger.info("Training started")
        best_val_loss = float('inf')
        
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_acc = self.evaluate(val_loader)
            
            logger.info(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f}, "
                       f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), save_path)
                logger.info("  Model saved")
        
        logger.info("Training complete")


# ============================================================================
# RULE EXECUTION ENGINE
# ============================================================================

class RuleExecutor:
    """Execute ACE rules using learned patterns and lookups"""
    
    def __init__(self, rule_learner: RuleLearner):
        self.learner = rule_learner
        self.lookup_tables = rule_learner.lookup_tables
    
    def apply_rules(self, payment: Dict, ace_ids: List[str]) -> Dict:
        """Apply rules to payment"""
        result = deepcopy(payment)
        
        for ace_id in ace_ids:
            if ace_id not in self.learner.rules:
                continue
            
            rule = self.learner.rules[ace_id]
            patterns = rule['patterns']
            
            # Apply transformations based on learned patterns
            for pattern in patterns:
                self._apply_pattern(result, pattern)
        
        return result
    
    def _apply_pattern(self, payment: Dict, pattern: Dict):
        """Apply a transformation pattern"""
        entity = pattern['entity']
        field_path = pattern['field_path']
        operation = pattern['operation']
        value = pattern['value']
        
        if entity not in payment:
            return
        
        entity_data = payment[entity]
        
        # Parse field path
        keys = field_path.replace('[', '.').replace(']', '').split('.')
        
        if operation == 'added':
            self._set_nested(entity_data, keys, value)
        elif operation == 'transformed':
            self._set_nested(entity_data, keys, value)
        elif operation == 'removed':
            self._delete_nested(entity_data, keys)
    
    def _set_nested(self, data: Dict, keys: List[str], value: Any):
        """Set value in nested structure"""
        for key in keys[:-1]:
            if key.isdigit():
                key = int(key)
                if not isinstance(data, list):
                    return
                while len(data) <= key:
                    data.append({})
                data = data[key]
            else:
                if key not in data:
                    data[key] = {}
                data = data[key]
        
        final_key = keys[-1]
        if final_key.isdigit():
            final_key = int(final_key)
            if isinstance(data, list):
                while len(data) <= final_key:
                    data.append(None)
                data[final_key] = value
        else:
            data[final_key] = value
    
    def _delete_nested(self, data: Dict, keys: List[str]):
        """Delete key in nested structure"""
        for key in keys[:-1]:
            if key not in data:
                return
            data = data[key]
        
        final_key = keys[-1]
        if final_key in data:
            del data[final_key]


# ============================================================================
# COMPLETE SYSTEM
# ============================================================================

class ComprehensiveRepairSystem:
    """Complete repair system"""
    
    def __init__(self, model_dir: str):
        logger.info(f"Loading system from {model_dir}")
        
        self.learner = RuleLearner.load(model_dir)
        self.executor = RuleExecutor(self.learner)
        
        with open(f"{model_dir}/ace_mapping.json", 'r') as f:
            data = json.load(f)
            self.ace_to_idx = data['ace_to_idx']
            self.idx_to_ace = {int(k): v for k, v in data['idx_to_ace'].items()}
        
        self.feature_extractor = FeatureExtractor()
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        # Get feature dimension
        dummy_features = self.feature_extractor.extract({})
        
        self.model = RulePredictionModel(len(dummy_features), len(self.ace_to_idx))
        self.model.load_state_dict(torch.load(f"{model_dir}/model.pt", map_location=device))
        self.model.to(device)
        self.model.eval()
        
        logger.info("System loaded")
    
    def repair(self, payment: Dict, threshold: float = 0.5) -> Dict:
        """Repair payment"""
        features = self.feature_extractor.extract(payment)
        features = features.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(features)[0]
        
        predicted_rules = []
        for idx, prob in enumerate(predictions):
            if prob > threshold:
                ace_id = self.idx_to_ace[idx]
                predicted_rules.append((ace_id, float(prob)))
        
        predicted_rules.sort(key=lambda x: x[1], reverse=True)
        ace_ids = [r[0] for r in predicted_rules]
        
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
# CLI
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
    
    n = len(dataset)
    train_size = int(n * 0.8)
    val_size = n - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    dummy_features = feature_extractor.extract({})
    model = RulePredictionModel(len(dummy_features), len(ace_to_idx))
    
    logger.info(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    
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
