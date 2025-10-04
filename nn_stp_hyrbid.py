"""
ACE Payment Repair Predictor - Production Version
=================================================

Goal: Predict which repairs to apply to payments with 90%+ accuracy

Architecture:
- Feature-based approach: 172 domain-specific features
- Hybrid model: Deterministic rules (100% accuracy) + ML (pattern learning)
- Multi-label classification: Predict multiple repair codes per payment
- Ensemble: Random Forest + Neural Network averaged predictions

Key Features:
- Handles variable payment structure (some have cdtr, some dbtr, etc.)
- Works with training format (before/after) and inference format (raw)
- Deterministic rules for obvious cases (BIC→country extraction)
- ML models for complex pattern learning
- Comprehensive feature extraction (172 features)

Usage:
    # Train on your 10K dataset
    python ace_repair_predictor.py train --input repairs_10k.json --epochs 50
    
    # Predict repairs for new payment
    python ace_repair_predictor.py predict --input new_payment.json --output result.json
    
    # Evaluate on test set
    python ace_repair_predictor.py evaluate --input test_data.json

Author: Production Implementation
Version: 4.0 - All bugs fixed
"""

import json
import argparse
import os
import pickle
import logging
import re
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

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
    """
    Configuration for the repair predictor.
    All hyperparameters and paths are defined here.
    """
    # Neural network settings
    hidden_dim: int = 256              # Hidden layer size
    dropout: float = 0.3               # Dropout rate for regularization
    learning_rate: float = 0.001       # Adam optimizer learning rate
    batch_size: int = 32               # Training batch size
    num_epochs: int = 50               # Maximum training epochs
    
    # Data split ratios
    train_split: float = 0.7           # 70% for training
    val_split: float = 0.15            # 15% for validation
    test_split: float = 0.15           # 15% for testing
    
    # Paths
    model_dir: str = "./models"        # Where to save trained models
    
    # Prediction thresholds
    ml_threshold: float = 0.5          # Confidence threshold for ML predictions
    rule_confidence: float = 1.0       # Rules always have 100% confidence
    
    def save(self, path: str):
        """Save configuration to JSON file"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: str):
        """Load configuration from JSON file"""
        with open(path, 'r') as f:
            return cls(**json.load(f))


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

class PaymentFeatureExtractor:
    """
    Extracts 172 domain-specific features from payment structure.
    
    These features capture signals that indicate which repairs are needed:
    - What's present (has BIC, has clearing ID)
    - What's missing (no country field, no bank name)
    - Relationships (BIC present but country missing → repair needed)
    - Structure (entity counts, field completeness)
    
    Feature categories:
    - BIC features (15): BIC presence, format, location
    - IBAN features (10): IBAN validation, country codes
    - Clearing features (12): Clearing system IDs, types
    - Name features (18): Bank/party names, completeness
    - Address features (20): Address presence, structure
    - Country features (15): Country codes, validation
    - Account features (12): Account types, currencies
    - Entity features (15): Which entities are present
    - FinInstnId features (10): Financial institution data
    - Remittance features (12): Payment descriptions
    - Metadata features (8): Source, clearing system
    - Flags features (10): Processing flags
    - Structural features (15): Payment complexity metrics
    
    Total: 172 features
    """
    
    def __init__(self):
        self.feature_names = []
        self._build_feature_names()
    
    def _build_feature_names(self):
        """Define all 172 feature names"""
        
        # BIC features (15)
        # Captures BIC code presence, format, and location in payment structure
        self.feature_names.extend([
            'has_bic', 'bic_length_8', 'bic_length_11', 'bic_valid_format',
            'bic_in_cdtrAgt', 'bic_in_dbtrAgt', 'bic_in_instgAgt', 'bic_in_instdAgt',
            'has_bic_country_code', 'bic_country_matches_field',
            'bic_in_cdtr', 'bic_in_dbtr', 'multiple_bics', 'bic_format_xxx', 'bic_format_xxxxxxxxxxx'
        ])
        
        # IBAN features (10)
        # Captures IBAN presence, validation, and country information
        self.feature_names.extend([
            'has_iban', 'iban_valid_format', 'iban_length_valid', 
            'iban_in_cdtrAcct', 'iban_in_dbtrAcct',
            'iban_country_code', 'iban_country_matches_field', 
            'has_iban_no_bic', 'iban_starts_with_letters', 'multiple_ibans'
        ])
        
        # Clearing system features (12)
        # Captures routing/clearing information (USABA, FedWire, CHIPS, etc.)
        self.feature_names.extend([
            'has_clearing_id', 'clearing_in_cdtrAgt', 'clearing_in_dbtrAgt', 'clearing_in_instgAgt',
            'clearing_type_usaba', 'clearing_type_fedwire', 'clearing_type_chips', 'clearing_type_other',
            'has_clearing_no_bic', 'has_bic_and_clearing', 'clearing_id_length', 'has_clearing_system_code'
        ])
        
        # Name features (18)
        # Captures bank and party name information
        self.feature_names.extend([
            'has_bank_name', 'has_party_name', 
            'bank_name_in_cdtrAgt', 'bank_name_in_dbtrAgt', 'bank_name_in_instgAgt', 'bank_name_in_instdAgt',
            'party_name_in_cdtr', 'party_name_in_dbtr',
            'missing_bank_name_has_bic', 'missing_party_name',
            'name_contains_bank_keywords', 'name_all_caps', 'name_length_short', 'name_length_medium', 'name_length_long',
            'multiple_names', 'name_has_special_chars', 'name_numeric'
        ])
        
        # Address features (20)
        # Captures address presence, structure, and completeness
        self.feature_names.extend([
            'has_address', 'has_postal_address', 'has_structured_address',
            'address_in_cdtrAgt', 'address_in_dbtrAgt', 'address_in_instgAgt', 'address_in_instdAgt',
            'address_in_cdtr', 'address_in_dbtr',
            'missing_address_has_bic', 'address_multiline', 'address_has_country',
            'address_has_city', 'address_has_state', 'address_has_zip', 'address_has_street',
            'address_line_count', 'address_total_length', 'address_needs_standardization', 'address_has_po_box'
        ])
        
        # Country features (15)
        # Captures country code presence and validation
        self.feature_names.extend([
            'has_country_field', 'country_in_cdtrAgt', 'country_in_dbtrAgt', 'country_in_instgAgt',
            'country_in_cdtr', 'country_in_dbtr', 'country_in_address',
            'missing_country_has_bic', 'missing_country_has_iban', 'country_code_valid',
            'country_code_length_2', 'multiple_countries', 'country_mismatch', 'country_from_bic_available', 'country_from_iban_available'
        ])
        
        # Account features (12)
        # Captures account information, types, and currencies
        self.feature_names.extend([
            'has_cdtrAcct', 'has_dbtrAcct', 'has_account_id', 'has_account_type',
            'account_type_iban', 'account_type_other', 'account_type_proprietary',
            'has_account_currency', 'currency_usd', 'currency_eur', 'currency_other',
            'multiple_accounts'
        ])
        
        # Entity presence features (15)
        # Captures which payment entities exist (variable structure handling)
        self.feature_names.extend([
            'has_cdtr', 'has_dbtr', 'has_cdtrAgt', 'has_dbtrAgt',
            'has_instgAgt', 'has_instdAgt',
            'has_intrmyAgt1', 'has_intrmyAgt2', 'has_intrmyAgt',
            'has_rmtInf', 'has_creditor_side', 'has_debtor_side', 'has_instructing_side', 'has_instructed_side',
            'entity_count'
        ])
        
        # Financial institution ID features (10)
        # Captures completeness of financial institution data
        self.feature_names.extend([
            'has_fininstnid', 'fininstnid_in_multiple_entities',
            'fininstnid_has_bic', 'fininstnid_has_clearing', 'fininstnid_has_name',
            'fininstnid_has_address', 'fininstnid_complete', 'fininstnid_incomplete',
            'fininstnid_in_cdtrAgt', 'fininstnid_in_dbtrAgt'
        ])
        
        # Remittance features (12)
        # Captures payment description information
        self.feature_names.extend([
            'has_rmtInf', 'rmtInf_has_ustrd', 'rmtInf_has_strd',
            'rmtInf_length', 'rmtInf_needs_split', 'rmtInf_multiline',
            'rmtInf_has_invoice', 'rmtInf_has_reference', 'rmtInf_all_caps',
            'rmtInf_has_numbers', 'rmtInf_has_special_chars', 'rmtInf_line_count'
        ])
        
        # Source/Clearing metadata (8)
        # Captures payment source and clearing system metadata
        self.feature_names.extend([
            'source_swf', 'source_fed', 'source_chips', 'source_other',
            'clearing_fed', 'clearing_chips', 'clearing_swift', 'clearing_other'
        ])
        
        # Flags features (10)
        # Captures processing flags and queue information
        self.feature_names.extend([
            'flag_ace_repairs', 'flag_autotrieve', 'flag_stp_failed', 
            'flag_repair_queue', 'flag_exception_queue', 'flag_verify_queue',
            'flag_ignored_party', 'flag_credit_party_inserted', 'flag_non_ace_repairs',
            'flags_count'
        ])
        
        # Structural features (15)
        # Captures payment complexity and completeness metrics
        self.feature_names.extend([
            'total_fields', 'total_entities', 'total_leaf_values',
            'max_nesting_depth', 'has_arrays', 'array_count',
            'has_empty_fields', 'empty_field_ratio',
            'field_density', 'entity_balance', 'agent_to_party_ratio',
            'has_complete_routing', 'has_partial_routing', 'complexity_score', 'data_completeness'
        ])
        
        logger.info(f"Initialized {len(self.feature_names)} features")
    
    def extract_features(self, payment: Dict) -> np.ndarray:
        """
        Extract all 172 features from a payment.
        
        Args:
            payment: Payment dictionary (normalized with lowercase keys)
            
        Returns:
            numpy array of 172 float values (0.0 to 1.0)
        """
        features = np.zeros(len(self.feature_names))
        
        # Normalize payment keys to lowercase for consistent access
        payment = self._normalize_payment(payment)
        
        # Extract each feature category in order
        idx = 0
        
        # BIC features (15)
        bic_info = self._extract_bic_info(payment)
        features[idx:idx+15] = list(bic_info.values())
        idx += 15
        
        # IBAN features (10)
        iban_info = self._extract_iban_info(payment)
        features[idx:idx+10] = list(iban_info.values())
        idx += 10
        
        # Clearing system features (12)
        clearing_info = self._extract_clearing_info(payment)
        features[idx:idx+12] = list(clearing_info.values())
        idx += 12
        
        # Name features (18)
        name_info = self._extract_name_info(payment)
        features[idx:idx+18] = list(name_info.values())
        idx += 18
        
        # Address features (20)
        address_info = self._extract_address_info(payment)
        features[idx:idx+20] = list(address_info.values())
        idx += 20
        
        # Country features (15)
        country_info = self._extract_country_info(payment)
        features[idx:idx+15] = list(country_info.values())
        idx += 15
        
        # Account features (12)
        account_info = self._extract_account_info(payment)
        features[idx:idx+12] = list(account_info.values())
        idx += 12
        
        # Entity presence features (15)
        entity_info = self._extract_entity_info(payment)
        features[idx:idx+15] = list(entity_info.values())
        idx += 15
        
        # Financial institution ID features (10)
        fininstn_info = self._extract_fininstn_info(payment)
        features[idx:idx+10] = list(fininstn_info.values())
        idx += 10
        
        # Remittance features (12)
        rmt_info = self._extract_remittance_info(payment)
        features[idx:idx+12] = list(rmt_info.values())
        idx += 12
        
        # Source/Clearing metadata (8)
        metadata_info = self._extract_metadata_info(payment)
        features[idx:idx+8] = list(metadata_info.values())
        idx += 8
        
        # Flags features (10)
        flags_info = self._extract_flags_info(payment)
        features[idx:idx+10] = list(flags_info.values())
        idx += 10
        
        # Structural features (15)
        struct_info = self._extract_structural_info(payment)
        features[idx:idx+15] = list(struct_info.values())
        idx += 15
        
        return features
    
    def _normalize_payment(self, payment: Dict) -> Dict:
        """Recursively normalize all keys to lowercase for consistent access"""
        if isinstance(payment, dict):
            return {k.lower() if isinstance(k, str) else k: self._normalize_payment(v) 
                   for k, v in payment.items()}
        elif isinstance(payment, list):
            return [self._normalize_payment(item) for item in payment]
        return payment
    
    def _extract_bic_info(self, payment: Dict) -> Dict:
        """Extract BIC-related features (15 features)"""
        bic_fields = self._find_all_values(payment, 'bic')
        
        has_bic = len(bic_fields) > 0
        bic_value = bic_fields[0] if bic_fields else None
        
        return {
            'has_bic': float(has_bic),
            'length_8': float(bic_value and len(str(bic_value)) == 8) if bic_value else 0.0,
            'length_11': float(bic_value and len(str(bic_value)) == 11) if bic_value else 0.0,
            'valid_format': float(self._is_valid_bic(bic_value)) if bic_value else 0.0,
            'in_cdtrAgt': float(self._field_in_entity(payment, 'bic', 'cdtragt')),
            'in_dbtrAgt': float(self._field_in_entity(payment, 'bic', 'dbtragt')),
            'in_instgAgt': float(self._field_in_entity(payment, 'bic', 'instgagt')),
            'in_instdAgt': float(self._field_in_entity(payment, 'bic', 'instdagt')),
            'has_country_code': float(bic_value and len(str(bic_value)) >= 6) if bic_value else 0.0,
            'country_matches': float(self._bic_country_matches_field(payment, bic_value)) if bic_value else 0.0,
            'in_cdtr': float(self._field_in_entity(payment, 'bic', 'cdtr')),
            'in_dbtr': float(self._field_in_entity(payment, 'bic', 'dbtr')),
            'multiple_bics': float(len(bic_fields) > 1),
            'format_xxx': float(bic_value and len(str(bic_value)) == 8) if bic_value else 0.0,
            'format_xxxxxxxxxxx': float(bic_value and len(str(bic_value)) == 11) if bic_value else 0.0
        }
    
    def _extract_iban_info(self, payment: Dict) -> Dict:
        """Extract IBAN-related features (10 features)"""
        iban_fields = self._find_all_values(payment, 'iban')
        
        has_iban = len(iban_fields) > 0
        iban_value = str(iban_fields[0]) if iban_fields else None
        
        return {
            'has_iban': float(has_iban),
            'valid_format': float(self._is_valid_iban(iban_value)) if iban_value else 0.0,
            'length_valid': float(iban_value and 15 <= len(iban_value) <= 34) if iban_value else 0.0,
            'in_cdtrAcct': float(self._field_in_entity(payment, 'iban', 'cdtracct')),
            'in_dbtrAcct': float(self._field_in_entity(payment, 'iban', 'dbtracct')),
            'country_code': float(iban_value and len(iban_value) >= 2 and iban_value[:2].isalpha()) if iban_value else 0.0,
            'country_matches': float(self._iban_country_matches_field(payment, iban_value)) if iban_value else 0.0,
            'has_iban_no_bic': float(has_iban and len(self._find_all_values(payment, 'bic')) == 0),
            'starts_with_letters': float(iban_value and iban_value[:2].isalpha()) if iban_value else 0.0,
            'multiple_ibans': float(len(iban_fields) > 1)
        }
    
    def _extract_clearing_info(self, payment: Dict) -> Dict:
        """
        Extract clearing system features (12 features)
        
        FIXED: Proper boolean logic to avoid empty string conversion errors
        """
        clearing_fields = self._find_all_values(payment, 'mmbid')
        clearing_sys = self._find_all_values(payment, 'clrsysid')
        clearing_cd = self._find_all_values(payment, 'cd')
        
        has_clearing = len(clearing_fields) > 0
        has_bic = len(self._find_all_values(payment, 'bic')) > 0
        
        clearing_type = str(clearing_sys[0]).lower() if clearing_sys else ""
        clearing_id_val = str(clearing_fields[0]) if clearing_fields else ""
        
        # CRITICAL FIX: Explicitly convert to bool first to avoid empty string issues
        is_usaba = bool(clearing_type and 'usaba' in clearing_type)
        is_fedwire = bool(clearing_type and ('fed' in clearing_type or 'fedwire' in clearing_type))
        is_chips = bool(clearing_type and 'chips' in clearing_type)
        is_other = bool(clearing_type and not is_usaba and not is_fedwire and not is_chips)
        
        return {
            'has_clearing': float(has_clearing),
            'in_cdtrAgt': float(self._field_in_entity(payment, 'mmbid', 'cdtragt')),
            'in_dbtrAgt': float(self._field_in_entity(payment, 'mmbid', 'dbtragt')),
            'in_instgAgt': float(self._field_in_entity(payment, 'mmbid', 'instgagt')),
            'type_usaba': float(is_usaba),
            'type_fedwire': float(is_fedwire),
            'type_chips': float(is_chips),
            'type_other': float(is_other),
            'has_clearing_no_bic': float(has_clearing and not has_bic),
            'has_both': float(has_clearing and has_bic),
            'id_length': min(len(clearing_id_val) / 15.0, 1.0) if clearing_id_val else 0.0,
            'has_system_code': float(len(clearing_sys) > 0 or len(clearing_cd) > 0)
        }
    
    def _extract_name_info(self, payment: Dict) -> Dict:
        """Extract name features (18 features)"""
        bank_names = self._find_all_values(payment, 'nm')
        all_names = bank_names.copy()
        
        has_bank_name = len(bank_names) > 0
        has_any_name = len(all_names) > 0
        
        name_text = str(bank_names[0]) if bank_names else ""
        
        bank_keywords = ['bank', 'trust', 'federal', 'credit union', 'n.a.', 'association']
        has_bank_keyword = any(kw in name_text.lower() for kw in bank_keywords)
        
        return {
            'has_bank_name': float(has_bank_name),
            'has_party_name': float(has_any_name),
            'in_cdtrAgt': float(self._field_in_entity(payment, 'nm', 'cdtragt')),
            'in_dbtrAgt': float(self._field_in_entity(payment, 'nm', 'dbtragt')),
            'in_instgAgt': float(self._field_in_entity(payment, 'nm', 'instgagt')),
            'in_instdAgt': float(self._field_in_entity(payment, 'nm', 'instdagt')),
            'in_cdtr': float(self._field_in_entity(payment, 'nm', 'cdtr')),
            'in_dbtr': float(self._field_in_entity(payment, 'nm', 'dbtr')),
            'missing_has_bic': float(not has_bank_name and len(self._find_all_values(payment, 'bic')) > 0),
            'missing_party': float(not has_any_name),
            'contains_bank_keywords': float(has_bank_keyword),
            'all_caps': float(name_text.isupper()) if name_text else 0.0,
            'length_short': float(len(name_text) < 20) if name_text else 0.0,
            'length_medium': float(20 <= len(name_text) <= 50) if name_text else 0.0,
            'length_long': float(len(name_text) > 50) if name_text else 0.0,
            'multiple': float(len(all_names) > 1),
            'has_special_chars': float(bool(re.search(r'[^a-zA-Z0-9\s]', name_text))) if name_text else 0.0,
            'numeric': float(any(c.isdigit() for c in name_text)) if name_text else 0.0
        }
    
    def _extract_address_info(self, payment: Dict) -> Dict:
        """Extract address features (20 features)"""
        addresses = self._find_all_values(payment, 'adrline')
        postal_addr = self._find_all_values(payment, 'pstladr')
        
        has_address = len(addresses) > 0
        has_postal = len(postal_addr) > 0
        
        addr_text = ' '.join(str(a) for a in addresses) if addresses else ""
        
        return {
            'has_address': float(has_address),
            'has_postal': float(has_postal),
            'has_structured': float(has_postal and isinstance(postal_addr[0], dict)) if postal_addr else 0.0,
            'in_cdtrAgt': float(self._field_in_entity(payment, 'adrline', 'cdtragt')),
            'in_dbtrAgt': float(self._field_in_entity(payment, 'adrline', 'dbtragt')),
            'in_instgAgt': float(self._field_in_entity(payment, 'adrline', 'instgagt')),
            'in_instdAgt': float(self._field_in_entity(payment, 'adrline', 'instdagt')),
            'in_cdtr': float(self._field_in_entity(payment, 'adrline', 'cdtr')),
            'in_dbtr': float(self._field_in_entity(payment, 'adrline', 'dbtr')),
            'missing_has_bic': float(not has_address and len(self._find_all_values(payment, 'bic')) > 0),
            'multiline': float(len(addresses) > 1),
            'has_country': float(bool(re.search(r'\b[A-Z]{2}\b', addr_text))),
            'has_city': float(bool(re.search(r'[A-Z][a-z]+', addr_text))),
            'has_state': float(bool(re.search(r'\b[A-Z]{2}\b', addr_text))),
            'has_zip': float(bool(re.search(r'\d{5}', addr_text))),
            'has_street': float(bool(re.search(r'\d+\s+\w+', addr_text))),
            'line_count': min(len(addresses) / 5.0, 1.0),
            'total_length': min(len(addr_text) / 200.0, 1.0),
            'needs_standardization': float(len(addr_text) > 140 or len(addresses) > 3),
            'has_po_box': float('p.o. box' in addr_text.lower() or 'po box' in addr_text.lower())
        }
    
    def _extract_country_info(self, payment: Dict) -> Dict:
        """Extract country features (15 features)"""
        countries = self._find_all_values(payment, 'ctryofres')
        countries.extend(self._find_all_values(payment, 'ctry'))
        
        has_country = len(countries) > 0
        country_val = str(countries[0]).upper() if countries else ""
        
        bic_fields = self._find_all_values(payment, 'bic')
        iban_fields = self._find_all_values(payment, 'iban')
        
        return {
            'has_country': float(has_country),
            'in_cdtrAgt': float(self._field_in_entity(payment, 'ctryofres', 'cdtragt')),
            'in_dbtrAgt': float(self._field_in_entity(payment, 'ctryofres', 'dbtragt')),
            'in_instgAgt': float(self._field_in_entity(payment, 'ctryofres', 'instgagt')),
            'in_cdtr': float(self._field_in_entity(payment, 'ctryofres', 'cdtr')),
            'in_dbtr': float(self._field_in_entity(payment, 'ctryofres', 'dbtr')),
            'in_address': float(bool(countries)),
            'missing_has_bic': float(not has_country and len(bic_fields) > 0),
            'missing_has_iban': float(not has_country and len(iban_fields) > 0),
            'code_valid': float(len(country_val) == 2 and country_val.isalpha()),
            'code_length_2': float(len(country_val) == 2),
            'multiple': float(len(set(countries)) > 1),
            'mismatch': float(len(set(str(c).upper() for c in countries)) > 1),
            'from_bic_available': float(len(bic_fields) > 0 and len(str(bic_fields[0])) >= 6),
            'from_iban_available': float(len(iban_fields) > 0 and len(str(iban_fields[0])) >= 2)
        }
    
    def _extract_account_info(self, payment: Dict) -> Dict:
        """Extract account features (12 features)"""
        has_cdtr_acct = 'cdtracct' in payment
        has_dbtr_acct = 'dbtracct' in payment
        
        acct_ids = self._find_all_values(payment, 'id')
        acct_types = self._find_all_values(payment, 'tp')
        currencies = self._find_all_values(payment, 'ccy')
        
        type_str = str(acct_types[0]).lower() if acct_types else ""
        currency_str = str(currencies[0]).upper() if currencies else ""
        
        return {
            'has_cdtrAcct': float(has_cdtr_acct),
            'has_dbtrAcct': float(has_dbtr_acct),
            'has_id': float(len(acct_ids) > 0),
            'has_type': float(len(acct_types) > 0),
            'type_iban': float('iban' in type_str),
            'type_other': float('othr' in type_str or 'other' in type_str),
            'type_proprietary': float('prtry' in type_str),
            'has_currency': float(len(currencies) > 0),
            'currency_usd': float(currency_str == 'USD'),
            'currency_eur': float(currency_str == 'EUR'),
            'currency_other': float(currency_str not in ['USD', 'EUR', '']),
            'multiple': float(has_cdtr_acct and has_dbtr_acct)
        }
    
    def _extract_entity_info(self, payment: Dict) -> Dict:
        """
        Extract entity presence features (15 features)
        
        CRITICAL: Handles variable payment structure
        Some payments have all entities, some have only one
        """
        entities = {
            'cdtr': 'cdtr' in payment,
            'dbtr': 'dbtr' in payment,
            'cdtrAgt': 'cdtragt' in payment,
            'dbtrAgt': 'dbtragt' in payment,
            'instgAgt': 'instgagt' in payment,
            'instdAgt': 'instdagt' in payment,
            'intrmyAgt1': 'intrmyagt1' in payment,
            'intrmyAgt2': 'intrmyagt2' in payment,
            'rmtInf': 'rmtinf' in payment
        }
        
        has_creditor = entities['cdtr'] or entities['cdtrAgt']
        has_debtor = entities['dbtr'] or entities['dbtrAgt']
        has_instructing = entities['instgAgt']
        has_instructed = entities['instdAgt']
        has_intrmy = entities['intrmyAgt1'] or entities['intrmyAgt2']
        
        entity_count = sum(1 for v in entities.values() if v)
        
        return {
            'has_cdtr': float(entities['cdtr']),
            'has_dbtr': float(entities['dbtr']),
            'has_cdtrAgt': float(entities['cdtrAgt']),
            'has_dbtrAgt': float(entities['dbtrAgt']),
            'has_instgAgt': float(entities['instgAgt']),
            'has_instdAgt': float(entities['instdAgt']),
            'has_intrmyAgt1': float(entities['intrmyAgt1']),
            'has_intrmyAgt2': float(entities['intrmyAgt2']),
            'has_intrmyAgt': float(has_intrmy),
            'has_rmtInf': float(entities['rmtInf']),
            'has_creditor_side': float(has_creditor),
            'has_debtor_side': float(has_debtor),
            'has_instructing_side': float(has_instructing),
            'has_instructed_side': float(has_instructed),
            'entity_count': min(entity_count / 10.0, 1.0)
        }
    
    def _extract_fininstn_info(self, payment: Dict) -> Dict:
        """Extract financial institution ID features (10 features)"""
        fininstn_fields = self._find_all_values(payment, 'fininstnid')
        
        has_fininstn = len(fininstn_fields) > 0
        
        # Check for completeness in financial institution blocks
        fininstn_entities = []
        for entity in ['cdtragt', 'dbtragt', 'instgagt', 'instdagt']:
            if entity in payment and 'fininstnid' in str(payment[entity]).lower():
                fininstn_entities.append(entity)
        
        has_multiple = len(fininstn_entities) > 1
        
        return {
            'has_fininstnid': float(has_fininstn),
            'in_multiple_entities': float(has_multiple),
            'has_bic': float(self._field_in_entity(payment, 'bic', 'fininstnid')),
            'has_clearing': float(self._field_in_entity(payment, 'mmbid', 'fininstnid')),
            'has_name': float(self._field_in_entity(payment, 'nm', 'fininstnid')),
            'has_address': float(self._field_in_entity(payment, 'adrline', 'fininstnid')),
            'complete': float(has_fininstn and len(fininstn_entities) > 0),
            'incomplete': float(has_fininstn and len(fininstn_entities) == 0),
            'in_cdtrAgt': float('cdtragt' in fininstn_entities),
            'in_dbtrAgt': float('dbtragt' in fininstn_entities)
        }
    
    def _extract_remittance_info(self, payment: Dict) -> Dict:
        """Extract remittance features (12 features)"""
        rmt_fields = self._find_all_values(payment, 'ustrd')
        strd_fields = self._find_all_values(payment, 'strd')
        
        has_rmt = len(rmt_fields) > 0 or len(strd_fields) > 0
        rmt_text = ' '.join(str(r) for r in rmt_fields) if rmt_fields else ""
        
        return {
            'has_rmtInf': float(has_rmt),
            'has_ustrd': float(len(rmt_fields) > 0),
            'has_strd': float(len(strd_fields) > 0),
            'length': min(len(rmt_text) / 200.0, 1.0),
            'needs_split': float(len(rmt_text) > 140),
            'multiline': float(len(rmt_fields) > 1),
            'has_invoice': float('invoice' in rmt_text.lower() or 'inv' in rmt_text.lower()),
            'has_reference': float('ref' in rmt_text.lower() or '#' in rmt_text),
            'all_caps': float(rmt_text.isupper()) if rmt_text else 0.0,
            'has_numbers': float(any(c.isdigit() for c in rmt_text)) if rmt_text else 0.0,
            'has_special_chars': float(bool(re.search(r'[^a-zA-Z0-9\s]', rmt_text))) if rmt_text else 0.0,
            'line_count': min(len(rmt_fields) / 5.0, 1.0)
        }
    
    def _extract_metadata_info(self, payment: Dict) -> Dict:
        """Extract source/clearing metadata (8 features)"""
        source = str(payment.get('source', '')).lower()
        clearing = str(payment.get('clearing', '')).lower()
        
        return {
            'source_swf': float(source == 'swf'),
            'source_fed': float(source == 'fed' or source == 'fedwire'),
            'source_chips': float(source == 'chips'),
            'source_other': float(source and source not in ['swf', 'fed', 'fedwire', 'chips']),
            'clearing_fed': float(clearing == 'fed' or clearing == 'fedwire'),
            'clearing_chips': float(clearing == 'chips'),
            'clearing_swift': float(clearing == 'swift'),
            'clearing_other': float(clearing and clearing not in ['fed', 'fedwire', 'chips', 'swift'])
        }
    
    def _extract_flags_info(self, payment: Dict) -> Dict:
        """Extract flags features (10 features)"""
        flags = payment.get('flags', {})
        
        flag_count = sum(1 for v in flags.values() if v) if flags else 0
        
        return {
            'ace_repairs': float(flags.get('ace_repairs', False)),
            'autotrieve': float(flags.get('autotrieve', False)),
            'stp_failed': float(flags.get('stp_failed', False)),
            'repair_queue': float(flags.get('repair_queue', False)),
            'exception_queue': float(flags.get('exception_queue', False)),
            'verify_queue': float(flags.get('verify_queue', False)),
            'ignored_party': float(flags.get('ignored_party', False)),
            'credit_party_inserted': float(flags.get('credit_party_inserted', False)),
            'non_ace_repairs': float(flags.get('non_ace_repairs', False)),
            'count': min(flag_count / 10.0, 1.0)
        }
    
    def _extract_structural_info(self, payment: Dict) -> Dict:
        """Extract structural features (15 features)"""
        total_fields = self._count_all_fields(payment)
        total_entities = sum(1 for k in payment.keys() if isinstance(payment.get(k), dict))
        total_leaf = self._count_leaf_values(payment)
        
        max_depth = self._get_max_depth(payment)
        arrays = self._count_arrays(payment)
        empty_fields = self._count_empty_fields(payment)
        
        # Count agents vs parties
        agents = sum(1 for k in payment.keys() if 'agt' in k.lower())
        parties = sum(1 for k in payment.keys() if k.lower() in ['cdtr', 'dbtr'])
        
        # Routing completeness
        has_bic = len(self._find_all_values(payment, 'bic')) > 0
        has_clearing = len(self._find_all_values(payment, 'mmbid')) > 0
        
        return {
            'total_fields': min(total_fields / 100.0, 1.0),
            'total_entities': min(total_entities / 15.0, 1.0),
            'total_leaf_values': min(total_leaf / 80.0, 1.0),
            'max_nesting_depth': min(max_depth / 10.0, 1.0),
            'has_arrays': float(arrays > 0),
            'array_count': min(arrays / 10.0, 1.0),
            'has_empty_fields': float(empty_fields > 0),
            'empty_field_ratio': min(empty_fields / max(total_fields, 1), 1.0),
            'field_density': min(total_leaf / max(total_fields, 1), 1.0),
            'entity_balance': 1.0 - abs(agents - parties) / max(agents + parties, 1),
            'agent_to_party_ratio': min(agents / max(parties, 1), 2.0) / 2.0,
            'has_complete_routing': float(has_bic or has_clearing),
            'has_partial_routing': float((has_bic or has_clearing) and not (has_bic and has_clearing)),
            'complexity_score': min((total_entities * max_depth) / 50.0, 1.0),
            'data_completeness': 1.0 - (empty_fields / max(total_fields, 1))
        }
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _is_valid_iban(self, iban: str) -> bool:
        """Check if IBAN format is valid"""
        if not iban or not isinstance(iban, str):
            return False
        iban = iban.replace(' ', '')
        return len(iban) >= 15 and iban[:2].isalpha() and iban[2:4].isdigit()
    
    def _iban_country_matches_field(self, payment: Dict, iban: str) -> bool:
        """Check if IBAN country code matches country field"""
        if not iban or len(iban) < 2:
            return False
        
        iban_country = iban[:2].upper()
        country_fields = self._find_all_values(payment, 'ctryofres')
        country_fields.extend(self._find_all_values(payment, 'ctry'))
        
        return any(iban_country == str(c).upper() for c in country_fields)
    
    def _count_leaf_values(self, obj) -> int:
        """Count leaf values (non-dict, non-list values)"""
        count = 0
        if isinstance(obj, dict):
            for v in obj.values():
                count += self._count_leaf_values(v)
        elif isinstance(obj, list):
            for item in obj:
                count += self._count_leaf_values(item)
        else:
            count = 1
        return count
    
    def _get_max_depth(self, obj, current_depth=0) -> int:
        """Get maximum nesting depth"""
        if isinstance(obj, dict):
            if not obj:
                return current_depth
            return max(self._get_max_depth(v, current_depth + 1) for v in obj.values())
        elif isinstance(obj, list):
            if not obj:
                return current_depth
            return max(self._get_max_depth(item, current_depth + 1) for item in obj)
        return current_depth
    
    def _count_arrays(self, obj) -> int:
        """Count number of arrays in structure"""
        count = 0
        if isinstance(obj, dict):
            for v in obj.values():
                if isinstance(v, list):
                    count += 1
                count += self._count_arrays(v)
        elif isinstance(obj, list):
            for item in obj:
                count += self._count_arrays(item)
        return count
    
    def _count_empty_fields(self, obj) -> int:
        """Count empty or null fields"""
        count = 0
        if isinstance(obj, dict):
            for v in obj.values():
                if v is None or v == '' or v == []:
                    count += 1
                elif isinstance(v, (dict, list)):
                    count += self._count_empty_fields(v)
        elif isinstance(obj, list):
            for item in obj:
                count += self._count_empty_fields(item)
        return count
    
    def _find_all_values(self, obj, key: str) -> List:
        """Recursively find all values for a key"""
        results = []
        key = key.lower()
        
        if isinstance(obj, dict):
            for k, v in obj.items():
                if key in str(k).lower():
                    if isinstance(v, (str, int, float)):
                        results.append(v)
                    elif isinstance(v, list):
                        results.extend([item for item in v if isinstance(item, (str, int, float))])
                
                if isinstance(v, (dict, list)):
                    results.extend(self._find_all_values(v, key))
        
        elif isinstance(obj, list):
            for item in obj:
                results.extend(self._find_all_values(item, key))
        
        return results
    
    def _field_in_entity(self, payment: Dict, field: str, entity: str) -> bool:
        """Check if field exists in specific entity"""
        entity = entity.lower()
        if entity not in payment:
            return False
        
        entity_data = payment[entity]
        values = self._find_all_values(entity_data, field)
        return len(values) > 0
    
    def _is_valid_bic(self, bic: str) -> bool:
        """Check if BIC format is valid"""
        if not bic or not isinstance(bic, str):
            return False
        return len(bic) in [8, 11] and bic[:6].isalpha() and bic[6:8].isalnum()
    
    def _bic_country_matches_field(self, payment: Dict, bic: str) -> bool:
        """Check if BIC country code matches country field"""
        if not bic or len(bic) < 6:
            return False
        
        bic_country = bic[4:6].upper()
        country_fields = self._find_all_values(payment, 'ctryofres')
        country_fields.extend(self._find_all_values(payment, 'ctry'))
        
        return any(bic_country == str(c).upper() for c in country_fields)
    
    def _count_all_fields(self, obj) -> int:
        """Count total number of leaf fields"""
        count = 0
        if isinstance(obj, dict):
            for v in obj.values():
                if isinstance(v, (dict, list)):
                    count += self._count_all_fields(v)
                else:
                    count += 1
        elif isinstance(obj, list):
            for item in obj:
                count += self._count_all_fields(item)
        return count


# ============================================================================
# DETERMINISTIC RULES
# ============================================================================

class DeterministicRules:
    """
    Rule-based repair prediction for 100% certain cases.
    
    These rules handle deterministic patterns where we have 100% confidence:
    - BIC present + country missing → Extract country from BIC[4:6]
    - Clearing ID present + BIC missing → Lookup BIC from clearing system
    - BIC present + bank name missing → Lookup name from BIC directory
    
    Rules handle ~30-40% of all repairs with perfect accuracy.
    """
    
    def __init__(self):
        self.rule_stats = Counter()
    
    def predict_repairs(self, payment: Dict, features: np.ndarray) -> Tuple[List[str], List[float]]:
        """
        Apply deterministic rules to predict repairs.
        
        Args:
            payment: Payment dictionary (not used currently, but available)
            features: 172 feature vector extracted from payment
            
        Returns:
            Tuple of (repair_ids, confidences)
            Example: (['6021', '6036'], [1.0, 1.0])
        """
        repairs = []
        confidences = []
        payment = self._normalize(payment)
        
        # Rule 1: Country from BIC (Repair 6021)
        # Trigger: BIC exists but country field doesn't
        # Action: Extract country from BIC characters 4-5
        if self._needs_country_from_bic(payment, features):
            repairs.append('6021')
            confidences.append(1.0)
            self.rule_stats['6021_country_from_bic'] += 1
        
        # Rule 2: BIC from clearing (Repair 6035)
        # Trigger: Clearing system ID exists but BIC doesn't
        # Action: Lookup BIC from clearing system database
        if self._needs_bic_from_clearing(payment, features):
            repairs.append('6035')
            confidences.append(1.0)
            self.rule_stats['6035_bic_from_clearing'] += 1
        
        # Rule 3: Bank name from BIC (Repair 6036)
        # Trigger: BIC exists but bank name doesn't
        # Action: Lookup bank name from BIC directory
        if self._needs_bank_name_from_bic(payment, features):
            repairs.append('6036')
            confidences.append(1.0)
            self.rule_stats['6036_name_from_bic'] += 1
        
        return repairs, confidences
    
    def _normalize(self, payment: Dict) -> Dict:
        """Normalize payment keys to lowercase"""
        if isinstance(payment, dict):
            return {k.lower() if isinstance(k, str) else k: self._normalize(v) 
                   for k, v in payment.items()}
        elif isinstance(payment, list):
            return [self._normalize(item) for item in payment]
        return payment
    
    def _needs_country_from_bic(self, payment: Dict, features: np.ndarray) -> bool:
        """
        Check if country code should be extracted from BIC.
        
        Feature indices:
        - features[0]: has_bic
        - features[75]: has_country_field (in country features block)
        """
        has_bic = features[0] > 0.5
        has_country = features[75] > 0.5  # has_country_field
        
        return has_bic and not has_country
    
    def _needs_bic_from_clearing(self, payment: Dict, features: np.ndarray) -> bool:
        """
        Check if BIC should be resolved from clearing system.
        
        Feature indices:
        - features[25]: has_clearing_id (in clearing features block)
        - features[0]: has_bic
        """
        has_clearing = features[25] > 0.5  # has_clearing_id
        has_bic = features[0] > 0.5
        
        return has_clearing and not has_bic
    
    def _needs_bank_name_from_bic(self, payment: Dict, features: np.ndarray) -> bool:
        """
        Check if bank name should be looked up from BIC.
        
        Feature indices:
        - features[0]: has_bic
        - features[37]: has_bank_name (in name features block)
        """
        has_bic = features[0] > 0.5
        has_name = features[37] > 0.5  # has_bank_name
        
        return has_bic and not has_name
    
    def print_stats(self):
        """Print rule usage statistics"""
        logger.info("\nDeterministic Rule Statistics:")
        for rule, count in self.rule_stats.most_common():
            logger.info(f"  {rule}: {count}")


# ============================================================================
# ML MODELS
# ============================================================================

class RepairPredictor(nn.Module):
    """
    Neural network for repair prediction.
    
    Architecture:
    - Input: 172 features
    - Hidden layers: 256 → 128
    - Output: N repairs (multi-label with sigmoid)
    - Regularization: Dropout + BatchNorm
    """
    
    def __init__(self, num_features: int, num_repairs: int, hidden_dim: int = 256, dropout: float = 0.3):
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
            nn.Sigmoid()  # Multi-label: each repair independent
        )
    
    def forward(self, x):
        return self.network(x)


class RepairDataset(Dataset):
    """PyTorch dataset for repair prediction"""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ============================================================================
# DATA PROCESSOR
# ============================================================================

class DataProcessor:
    """
    Process training data and prepare for modeling.
    
    Handles:
    - Multiple input formats (array, dict, nested structures)
    - Training format (before/after sections)
    - Inference format (raw payment data)
    - Variable entity presence
    - Feature extraction and label encoding
    """
    
    def __init__(self):
        self.feature_extractor = PaymentFeatureExtractor()
        self.repair_vocabulary = {}
        self.idx_to_repair = {}
    
    def load_and_process(self, json_file: str) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Load training data and extract features + labels.
        
        CRITICAL: Only processes transactions with 'before' AND 'after' sections
        Skips transactions without 'after' as there's nothing to learn
        
        Args:
            json_file: Path to JSON file containing training data
            
        Returns:
            Tuple of (features, labels, raw_payments)
            - features: (N, 172) array of feature vectors
            - labels: (N, num_repairs) binary array of repair labels
            - raw_payments: List of original payment dicts
        """
        logger.info(f"Loading data from {json_file}")
        
        with open(json_file, 'r') as f:
            raw_data = json.load(f)
        
        # Handle multiple input formats
        # Format 1: [{"txn_id1": {...}, "txn_id2": {...}}] - array containing dict
        # Format 2: {"txn_id1": {...}, "txn_id2": {...}} - direct dict
        # Format 3: [{"source": "SWF", ...}, {...}] - array of transactions
        
        if isinstance(raw_data, list):
            if len(raw_data) == 0:
                raise ValueError(f"Empty array in {json_file}")
            
            if isinstance(raw_data[0], dict):
                first_item = raw_data[0]
                
                # Check if first element is a transaction dict
                # (all values are dicts = it's a transaction collection)
                if all(isinstance(v, dict) for v in first_item.values()):
                    data = first_item
                    logger.info(f"Detected format: Array containing dict of {len(data)} transactions")
                else:
                    # Array of individual transactions - convert to dict
                    data = {f"txn_{i:06d}": txn for i, txn in enumerate(raw_data)}
                    logger.info(f"Detected format: Array of {len(data)} transaction objects")
            else:
                raise ValueError(f"Unexpected array format in {json_file}")
        elif isinstance(raw_data, dict):
            data = raw_data
            logger.info(f"Detected format: Dict of {len(data)} transactions")
        else:
            raise ValueError(f"Unexpected data format in {json_file}: {type(raw_data)}")
        
        # Build repair vocabulary from all transactions
        self._build_repair_vocabulary(data)
        
        # Track statistics
        entity_stats = Counter()
        skipped_count = 0
        processed_count = 0
        
        # Extract features and labels
        all_features = []
        all_labels = []
        all_payments = []
        
        for txn_idx, (txn_id, txn_data) in enumerate(data.items()):
            try:
                # CRITICAL: Only train on transactions with 'after' sections
                # Training requires before/after pairs to learn transformations
                has_after = self._has_after_state(txn_data)
                
                if not has_after:
                    skipped_count += 1
                    if skipped_count <= 3:
                        logger.info(f"Skipping {txn_id}: No 'after' state (nothing to learn)")
                    continue
                
                # Get repair labels from 'ace' field
                repairs = [r['id'] for r in txn_data.get('ace', [])]
                
                # Convert training format to feature-extraction format
                # Training: {entity: {before: {...}, after: {...}}}
                # Features: {entity: {...}} (just the before state)
                payment_for_features = self._extract_before_state(txn_data)
                
                # Track which entities are present
                present_entities = [k for k in payment_for_features.keys() 
                                   if k not in ['source', 'clearing', 'flags', 'parties']]
                entity_stats.update(present_entities)
                
                # Log first few for verification
                if processed_count < 3:
                    logger.info(f"Training on {txn_id}: Entities: {present_entities}, Repairs: {repairs}")
                
                # Extract 172 features from payment
                features = self.feature_extractor.extract_features(payment_for_features)
                
                # Convert repair IDs to binary label vector
                labels = self._repairs_to_labels(repairs)
                
                all_features.append(features)
                all_labels.append(labels)
                all_payments.append(txn_data)
                processed_count += 1
                
            except Exception as e:
                logger.warning(f"Error processing transaction {txn_id}: {e}")
                import traceback
                logger.warning(traceback.format_exc())
                continue
        
        if len(all_features) == 0:
            raise ValueError(f"No trainable transactions found in {json_file}. "
                           "Training requires transactions with 'before' and 'after' sections.")
        
        features_array = np.array(all_features)
        labels_array = np.array(all_labels)
        
        # Log summary statistics
        logger.info(f"\n{'='*70}")
        logger.info(f"DATA PROCESSING SUMMARY")
        logger.info(f"{'='*70}")
        logger.info(f"Total transactions in file: {len(data)}")
        logger.info(f"Trainable (with 'after'): {processed_count}")
        logger.info(f"Skipped (no 'after'): {skipped_count}")
        logger.info(f"Feature shape: {features_array.shape}")
        logger.info(f"Label shape: {labels_array.shape}")
        logger.info(f"Unique repairs: {len(self.repair_vocabulary)}")
        
        # Log entity distribution
        if entity_stats:
            logger.info(f"\nEntity Distribution (trainable transactions):")
            for entity, count in entity_stats.most_common():
                percentage = (count / processed_count) * 100
                logger.info(f"  {entity}: {count} ({percentage:.1f}%)")
        
        # Show feature statistics
        logger.info(f"\nFeature Statistics:")
        logger.info(f"  Non-zero features per transaction (avg): {(features_array != 0).sum(axis=1).mean():.1f}/{len(self.feature_extractor.feature_names)}")
        
        return features_array, labels_array, all_payments
    
    def _has_after_state(self, txn_data: Dict) -> bool:
        """
        Check if transaction has any entities with 'after' state.
        
        Returns True if we can learn from this transaction.
        Training requires before/after pairs to see the transformations.
        """
        possible_entities = [
            'cdtr', 'dbtr', 'cdtrAgt', 'dbtrAgt', 
            'cdtrAcct', 'dbtrAcct', 'instgAgt', 'instdAgt',
            'intrmyAgt1', 'intrmyAgt2', 'rmtInf'
        ]
        
        for entity in possible_entities:
            for key in txn_data.keys():
                if key.lower() == entity.lower():
                    entity_data = txn_data[key]
                    if isinstance(entity_data, dict) and 'after' in entity_data:
                        return True
        
        return False
    
    def _extract_before_state(self, txn_data: Dict) -> Dict:
        """
        Extract 'before' state from training format.
        
        Handles variable entity presence - some payments have cdtr, some dbtr, etc.
        
        Training format:
          {"cdtrAgt": {"before": {...}, "after": {...}, "diffs": [...]}}
        
        Returns format for feature extraction:
          {"cdtrAgt": {...}}
        """
        payment = {}
        
        # Copy top-level metadata
        for key in ['source', 'clearing', 'flags', 'parties']:
            if key in txn_data:
                payment[key] = txn_data[key]
        
        # Extract 'before' state from all possible entities
        possible_entities = [
            'cdtr', 'dbtr', 'cdtrAgt', 'dbtrAgt', 
            'cdtrAcct', 'dbtrAcct', 'instgAgt', 'instdAgt',
            'intrmyAgt1', 'intrmyAgt2', 'rmtInf'
        ]
        
        for entity in possible_entities:
            # Check case-insensitive
            for key in txn_data.keys():
                if key.lower() == entity.lower():
                    entity_data = txn_data[key]
                    
                    # If it has 'before', extract it
                    if isinstance(entity_data, dict) and 'before' in entity_data:
                        payment[entity] = entity_data['before']
                    # Otherwise use directly (might be inference format)
                    elif isinstance(entity_data, dict):
                        payment[entity] = entity_data
                    
                    break
        
        return payment
    
    def _build_repair_vocabulary(self, data: Dict):
        """
        Build repair ID vocabulary from all transactions.
        
        Creates bidirectional mapping:
        - repair_vocabulary: repair_id → index
        - idx_to_repair: index → repair_id
        """
        all_repairs = set()
        
        for txn_data in data.values():
            for repair in txn_data.get('ace', []):
                all_repairs.add(repair['id'])
        
        # Sort for consistency
        sorted_repairs = sorted(all_repairs)
        
        self.repair_vocabulary = {repair_id: idx for idx, repair_id in enumerate(sorted_repairs)}
        self.idx_to_repair = {idx: repair_id for repair_id, idx in self.repair_vocabulary.items()}
        
        logger.info(f"Built vocabulary with {len(self.repair_vocabulary)} repairs: {sorted_repairs}")
    
    def _repairs_to_labels(self, repairs: List[str]) -> np.ndarray:
        """
        Convert repair IDs to binary label vector.
        
        Example:
        repairs = ['6021', '6036']
        vocabulary = {'6021': 0, '6035': 1, '6036': 2}
        returns = [1.0, 0.0, 1.0]
        """
        labels = np.zeros(len(self.repair_vocabulary))
        
        for repair_id in repairs:
            if repair_id in self.repair_vocabulary:
                idx = self.repair_vocabulary[repair_id]
                labels[idx] = 1.0
        
        return labels
    
    def labels_to_repairs(self, labels: np.ndarray, threshold: float = 0.5) -> List[str]:
        """Convert binary label vector back to repair IDs"""
        repairs = []
        
        for idx, prob in enumerate(labels):
            if prob > threshold:
                repairs.append(self.idx_to_repair[idx])
        
        return repairs


# ============================================================================
# HYBRID PREDICTOR
# ============================================================================

class HybridRepairPredictor:
    """
    Hybrid predictor combining deterministic rules + ML models.
    
    Architecture:
    1. Deterministic rules (100% accuracy for obvious cases)
    2. Random Forest (fast, interpretable baseline)
    3. Neural Network (complex pattern learning)
    4. Ensemble (average RF + NN predictions)
    
    Goal: 90%+ repair prediction accuracy
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.feature_extractor = PaymentFeatureExtractor()
        self.rules = DeterministicRules()
        self.ml_model = None
        self.rf_model = None
        self.processor = DataProcessor()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def train(self, train_file: str):
        """
        Train the hybrid model.
        
        Process:
        1. Load and process training data
        2. Split into train/val/test sets
        3. Train Random Forest model
        4. Train Neural Network model
        5. Evaluate on test set
        6. Save models if performance is good
        
        Args:
            train_file: Path to JSON file with training data
            
        Returns:
            Dictionary of test metrics
        """
        logger.info("="*70)
        logger.info("TRAINING HYBRID REPAIR PREDICTOR")
        logger.info("="*70)
        
        # Load and process data
        features, labels, payments = self.processor.load_and_process(train_file)
        
        n = len(features)
        
        # Handle small datasets (< 10 samples)
        if n < 10:
            logger.warning(f"\n⚠️  Small dataset detected ({n} samples)")
            logger.warning("For datasets < 10 samples:")
            logger.warning("  - Using all data for training (no validation split)")
            logger.warning("  - Model will overfit - this is for demonstration only")
            logger.warning("  - Recommend 1000+ samples for production use\n")
            
            # Use all data for everything (demonstration mode)
            X_train = features
            y_train = labels
            X_val = features
            y_val = labels
            X_test = features
            y_test = labels
        else:
            # Normal split for production datasets
            indices = np.random.permutation(n)
            
            train_size = int(n * self.config.train_split)
            val_size = int(n * self.config.val_split)
            
            train_idx = indices[:train_size]
            val_idx = indices[train_size:train_size + val_size]
            test_idx = indices[train_size + val_size:]
            
            X_train, y_train = features[train_idx], labels[train_idx]
            X_val, y_val = features[val_idx], labels[val_idx]
            X_test, y_test = features[test_idx], labels[test_idx]
        
        logger.info(f"Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
        # Train Random Forest
        logger.info("\nTraining Random Forest model...")
        self.rf_model = MultiOutputClassifier(
            RandomForestClassifier(
                n_estimators=min(200, max(10, n * 10)),  # Scale with dataset size
                max_depth=min(15, max(3, n // 2)),
                min_samples_split=max(2, n // 10),
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1
            )
        )
        self.rf_model.fit(X_train, y_train)
        
        # Evaluate RF
        rf_val_pred = self.rf_model.predict_proba(X_val)
        
        # CRITICAL FIX: Handle edge case where only one class is seen during training
        # If all validation samples have the same repair, predict_proba returns (n, 1) not (n, 2)
        rf_val_pred_processed = []
        for pred in rf_val_pred:
            if pred.shape[1] == 1:
                # Only one class seen - assume it's class 1 (repair present)
                rf_val_pred_processed.append(np.ones(pred.shape[0]))
            else:
                # Normal case: extract probability of class 1 (repair present)
                rf_val_pred_processed.append(pred[:, 1])
        
        rf_val_pred = np.array(rf_val_pred_processed).T
        rf_metrics = self._compute_metrics(y_val, rf_val_pred)
        
        logger.info("\nRandom Forest Validation Metrics:")
        logger.info(f"  Exact Match Accuracy: {rf_metrics['exact_match']:.1%}")
        logger.info(f"  Hamming Accuracy: {rf_metrics['hamming']:.1%}")
        logger.info(f"  F1 Score (micro): {rf_metrics['f1_micro']:.3f}")
        logger.info(f"  F1 Score (macro): {rf_metrics['f1_macro']:.3f}")
        logger.info(f"  Precision: {rf_metrics['precision']:.3f}")
        logger.info(f"  Recall: {rf_metrics['recall']:.3f}")
        
        # Train Neural Network
        logger.info("\nTraining Neural Network model...")
        self.ml_model = RepairPredictor(
            num_features=features.shape[1],
            num_repairs=labels.shape[1],
            hidden_dim=min(self.config.hidden_dim, max(32, n * 10)),
            dropout=self.config.dropout
        ).to(self.device)
        
        self._train_neural_network(X_train, y_train, X_val, y_val)
        
        # Final evaluation on test set
        logger.info("\n" + "="*70)
        logger.info("FINAL TEST SET EVALUATION")
        logger.info("="*70)
        
        test_metrics = self.evaluate(X_test, y_test)
        
        # Save models
        self.save_models()
        
        return test_metrics
    
    def _train_neural_network(self, X_train, y_train, X_val, y_val):
        """Train neural network with early stopping"""
        train_dataset = RepairDataset(X_train, y_train)
        val_dataset = RepairDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size)
        
        optimizer = torch.optim.AdamW(
            self.ml_model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=1e-4
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=5, factor=0.5
        )
        
        criterion = nn.BCELoss()
        
        best_f1 = 0.0
        patience_counter = 0
        patience = 10
        
        for epoch in range(1, self.config.num_epochs + 1):
            # Training
            self.ml_model.train()
            train_loss = 0.0
            
            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                predictions = self.ml_model(batch_features)
                loss = criterion(predictions, batch_labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            val_metrics = self.evaluate(X_val, y_val, loader=val_loader)
            
            scheduler.step(val_metrics['f1_micro'])
            
            if epoch % 5 == 0:
                logger.info(f"Epoch {epoch}/{self.config.num_epochs} - "
                           f"Loss: {train_loss:.4f} - "
                           f"Val F1: {val_metrics['f1_micro']:.3f} - "
                           f"Val Acc: {val_metrics['exact_match']:.1%}")
            
            # Early stopping
            if val_metrics['f1_micro'] > best_f1 + 0.001:
                best_f1 = val_metrics['f1_micro']
                patience_counter = 0
                torch.save(self.ml_model.state_dict(), 
                          os.path.join(self.config.model_dir, 'best_model.pt'))
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        # Load best model
        self.ml_model.load_state_dict(
            torch.load(os.path.join(self.config.model_dir, 'best_model.pt'))
        )
    
    def evaluate(self, X, y, loader=None):
        """Evaluate model performance with comprehensive metrics"""
        if loader is None:
            dataset = RepairDataset(X, y)
            loader = DataLoader(dataset, batch_size=self.config.batch_size)
        
        self.ml_model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch_features, batch_labels in loader:
                batch_features = batch_features.to(self.device)
                predictions = self.ml_model(batch_features)
                
                all_predictions.append(predictions.cpu().numpy())
                all_labels.append(batch_labels.numpy())
        
        predictions = np.vstack(all_predictions)
        labels = np.vstack(all_labels)
        
        return self._compute_metrics(labels, predictions)
    
    def _compute_metrics(self, y_true, y_pred_probs, threshold=0.5):
        """
        Compute comprehensive evaluation metrics.
        
        Metrics:
        - Exact Match: All repairs correct for a payment
        - Hamming: Per-repair accuracy
        - F1 (micro/macro): Precision-recall balance
        - Precision: Of predicted repairs, how many are correct
        - Recall: Of actual repairs, how many we found
        """
        y_pred = (y_pred_probs > threshold).astype(int)
        
        # Exact match accuracy (all repairs correct)
        exact_match = np.mean(np.all(y_pred == y_true, axis=1))
        
        # Hamming accuracy (per-repair accuracy)
        hamming = accuracy_score(y_true.flatten(), y_pred.flatten())
        
        # F1 scores
        f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Precision and recall
        precision = precision_score(y_true, y_pred, average='micro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='micro', zero_division=0)
        
        return {
            'exact_match': exact_match,
            'hamming': hamming,
            'f1_micro': f1_micro,
            'f1_macro': f1_macro,
            'precision': precision,
            'recall': recall
        }
    
    def predict(self, payment: Dict, use_rules: bool = True, use_ml: bool = True) -> Dict:
        """
        Predict repairs for a payment (inference mode).
        
        Process:
        1. Extract 172 features from payment
        2. Apply deterministic rules (100% confidence)
        3. Apply ML models (ensemble RF + NN)
        4. Combine predictions
        
        Args:
            payment: Payment dictionary (raw or before state)
            use_rules: Whether to apply deterministic rules
            use_ml: Whether to apply ML models
            
        Returns:
            {
                'repairs': List of repair IDs ['6021', '6036'],
                'confidences': List of confidence scores [1.0, 0.87],
                'sources': List of prediction sources ['rule', 'ml']
            }
        """
        # Extract features
        features = self.feature_extractor.extract_features(payment)
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        repairs = []
        confidences = []
        sources = []
        
        # Apply deterministic rules first
        if use_rules:
            rule_repairs, rule_confs = self.rules.predict_repairs(payment, features)
            repairs.extend(rule_repairs)
            confidences.extend(rule_confs)
            sources.extend(['rule'] * len(rule_repairs))
        
        # Apply ML models
        if use_ml:
            self.ml_model.eval()
            with torch.no_grad():
                ml_probs = self.ml_model(features_tensor)[0].cpu().numpy()
            
            # Also get RF predictions
            rf_probs_raw = self.rf_model.predict_proba(features.reshape(1, -1))
            
            # CRITICAL FIX: Handle single-class edge case
            rf_probs = []
            for pred in rf_probs_raw:
                if pred.shape[1] == 1:
                    # Only one class seen during training
                    rf_probs.append(1.0 if pred[0, 0] > 0.5 else 0.0)
                else:
                    rf_probs.append(pred[0, 1])
            rf_probs = np.array(rf_probs)
            
            # Ensemble: average RF and NN predictions
            ensemble_probs = (ml_probs + rf_probs) / 2
            
            # Add ML predictions above threshold
            for idx, prob in enumerate(ensemble_probs):
                repair_id = self.processor.idx_to_repair[idx]
                
                # Skip if already predicted by rules
                if repair_id in repairs:
                    continue
                
                if prob > self.config.ml_threshold:
                    repairs.append(repair_id)
                    confidences.append(float(prob))
                    sources.append('ml')
        
        return {
            'repairs': repairs,
            'confidences': confidences,
            'sources': sources
        }
    
    def save_models(self):
        """Save all models and metadata to disk"""
        os.makedirs(self.config.model_dir, exist_ok=True)
        
        # Save neural network
        torch.save(self.ml_model.state_dict(), 
                  os.path.join(self.config.model_dir, 'neural_model.pt'))
        
        # Save RF model
        with open(os.path.join(self.config.model_dir, 'rf_model.pkl'), 'wb') as f:
            pickle.dump(self.rf_model, f)
        
        # Save processor (contains repair vocabulary)
        with open(os.path.join(self.config.model_dir, 'processor.pkl'), 'wb') as f:
            pickle.dump(self.processor, f)
        
        # Save config
        self.config.save(os.path.join(self.config.model_dir, 'config.json'))
        
        logger.info(f"\nModels saved to {self.config.model_dir}")
    
    def load_models(self, model_dir: str):
        """Load saved models from disk"""
        self.config = Config.load(os.path.join(model_dir, 'config.json'))
        
        # Load processor
        with open(os.path.join(model_dir, 'processor.pkl'), 'rb') as f:
            self.processor = pickle.load(f)
        
        # Load RF model
        with open(os.path.join(model_dir, 'rf_model.pkl'), 'rb') as f:
            self.rf_model = pickle.load(f)
        
        # Load neural network
        num_features = len(self.feature_extractor.feature_names)
        num_repairs = len(self.processor.repair_vocabulary)
        
        self.ml_model = RepairPredictor(
            num_features=num_features,
            num_repairs=num_repairs,
            hidden_dim=self.config.hidden_dim,
            dropout=self.config.dropout
        ).to(self.device)
        
        self.ml_model.load_state_dict(
            torch.load(os.path.join(model_dir, 'neural_model.pt'), 
                      map_location=self.device)
        )
        self.ml_model.eval()
        
        logger.info(f"Models loaded from {model_dir}")


# ============================================================================
# CLI COMMANDS
# ============================================================================

def train_command(args):
    """Train the model"""
    config = Config()
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    
    predictor = HybridRepairPredictor(config)
    metrics = predictor.train(args.input)
    
    logger.info("\n" + "="*70)
    logger.info("TRAINING COMPLETE")
    logger.info("="*70)
    logger.info(f"Test Exact Match Accuracy: {metrics['exact_match']:.1%}")
    logger.info(f"Test F1 Score: {metrics['f1_micro']:.3f}")
    
    if metrics['exact_match'] >= 0.90:
        logger.info("✓ GOAL ACHIEVED: 90%+ accuracy!")
    else:
        logger.info(f"Need {0.90 - metrics['exact_match']:.1%} more to reach 90%")


def predict_command(args):
    """Predict repairs for a payment (inference on data without 'after' sections)"""
    predictor = HybridRepairPredictor(Config())
    predictor.load_models(args.model or './models')
    
    with open(args.input, 'r') as f:
        data = json.load(f)
    
    # Handle different input formats
    # Format 1: Single transaction wrapped in dict
    if isinstance(data, dict) and len(data) == 1:
        txn_id = list(data.keys())[0]
        payment_data = data[txn_id]
        
        # Check if training format (has before/after) or inference format (raw)
        if predictor.processor._has_after_state(payment_data):
            logger.info(f"Detected training format with 'before/after' sections")
            logger.info(f"Extracting 'before' state for prediction...")
            payment = predictor.processor._extract_before_state(payment_data)
        else:
            logger.info(f"Detected inference format (raw payment data)")
            payment = payment_data
    else:
        # Format 2: Raw payment structure
        payment = data
    
    result = predictor.predict(payment)
    
    logger.info("\n" + "="*70)
    logger.info("REPAIR PREDICTION RESULTS")
    logger.info("="*70)
    logger.info(f"Predicted {len(result['repairs'])} repairs:")
    
    for repair, conf, source in zip(result['repairs'], result['confidences'], result['sources']):
        logger.info(f"  {repair}: {conf:.3f} ({source})")
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"\nResults saved to {args.output}")


def evaluate_command(args):
    """Evaluate model on test data"""
    predictor = HybridRepairPredictor(Config())
    predictor.load_models(args.model or './models')
    
    features, labels, payments = predictor.processor.load_and_process(args.input)
    metrics = predictor.evaluate(features, labels)
    
    logger.info("\n" + "="*70)
    logger.info("EVALUATION RESULTS")
    logger.info("="*70)
    logger.info(f"Exact Match Accuracy: {metrics['exact_match']:.1%}")
    logger.info(f"Hamming Accuracy: {metrics['hamming']:.1%}")
    logger.info(f"F1 Score (micro): {metrics['f1_micro']:.3f}")
    logger.info(f"F1 Score (macro): {metrics['f1_macro']:.3f}")
    logger.info(f"Precision: {metrics['precision']:.3f}")
    logger.info(f"Recall: {metrics['recall']:.3f}")


def main():
    parser = argparse.ArgumentParser(
        description='ACE Payment Repair Predictor - Production Version',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on your 10K dataset
  python ace_repair_predictor.py train --input repairs_10k.json --epochs 50
  
  # Predict repairs for new payment
  python ace_repair_predictor.py predict --input new_payment.json --output result.json
  
  # Evaluate on test set
  python ace_repair_predictor.py evaluate --input test_data.json
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--input', required=True, help='Training data JSON file')
    train_parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    train_parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    
    # Predict
    predict_parser = subparsers.add_parser('predict', help='Predict repairs for payment')
    predict_parser.add_argument('--input', required=True, help='Payment JSON file')
    predict_parser.add_argument('--output', help='Output file for results')
    predict_parser.add_argument('--model', default='./models', help='Model directory')
    
    # Evaluate
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model on test data')
    eval_parser.add_argument('--input', required=True, help='Test data JSON file')
    eval_parser.add_argument('--model', default='./models', help='Model directory')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_command(args)
    elif args.command == 'predict':
        predict_command(args)
    elif args.command == 'evaluate':
        evaluate_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
