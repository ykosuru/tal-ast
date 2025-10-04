"""
ace_repair_predictor.py
Author: Yekesa Kosuru
========================

Enhanced ACE Payment Repair Predictor with:
- Discovered pattern-based rules
- Class-weighted loss for imbalanced repairs
- Per-repair detailed metrics
- Support for analysis integration

Usage:
    # Train with enhancements
    python ace_repair_predictor_enhanced.py train --input repairs_10k.json --analysis analysis_report.json --epochs 50
    
    # Train without analysis (uses baseline rules only)
    python ace_repair_predictor_enhanced.py train --input repairs_10k.json --epochs 50
    
    # Predict repairs
    python ace_repair_predictor_enhanced.py predict --input payment.json --output result.json
    
    # Evaluate with detailed per-repair metrics
    python ace_repair_predictor_enhanced.py evaluate --input test_data.json --detailed
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
    """Configuration for the repair predictor"""
    # Neural network settings
    hidden_dim: int = 256
    dropout: float = 0.3
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 50
    
    # Data split ratios
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    
    # Paths
    model_dir: str = "./models"
    
    # Prediction thresholds
    ml_threshold: float = 0.5
    rule_confidence: float = 1.0
    
    # Class weighting
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
# FEATURE EXTRACTION (Same as original)
# ============================================================================

class PaymentFeatureExtractor:
    """Extracts 172 domain-specific features from payment structure"""
    
    def __init__(self):
        self.feature_names = []
        self._build_feature_names()
    
    def _build_feature_names(self):
        """Define all 172 feature names"""
        # BIC features (15)
        self.feature_names.extend([
            'has_bic', 'bic_length_8', 'bic_length_11', 'bic_valid_format',
            'bic_in_cdtrAgt', 'bic_in_dbtrAgt', 'bic_in_instgAgt', 'bic_in_instdAgt',
            'has_bic_country_code', 'bic_country_matches_field',
            'bic_in_cdtr', 'bic_in_dbtr', 'multiple_bics', 'bic_format_xxx', 'bic_format_xxxxxxxxxxx'
        ])
        
        # IBAN features (10)
        self.feature_names.extend([
            'has_iban', 'iban_valid_format', 'iban_length_valid', 
            'iban_in_cdtrAcct', 'iban_in_dbtrAcct',
            'iban_country_code', 'iban_country_matches_field', 
            'has_iban_no_bic', 'iban_starts_with_letters', 'multiple_ibans'
        ])
        
        # Clearing system features (12)
        self.feature_names.extend([
            'has_clearing_id', 'clearing_in_cdtrAgt', 'clearing_in_dbtrAgt', 'clearing_in_instgAgt',
            'clearing_type_usaba', 'clearing_type_fedwire', 'clearing_type_chips', 'clearing_type_other',
            'has_clearing_no_bic', 'has_bic_and_clearing', 'clearing_id_length', 'has_clearing_system_code'
        ])
        
        # Name features (18)
        self.feature_names.extend([
            'has_bank_name', 'has_party_name', 
            'bank_name_in_cdtrAgt', 'bank_name_in_dbtrAgt', 'bank_name_in_instgAgt', 'bank_name_in_instdAgt',
            'party_name_in_cdtr', 'party_name_in_dbtr',
            'missing_bank_name_has_bic', 'missing_party_name',
            'name_contains_bank_keywords', 'name_all_caps', 'name_length_short', 'name_length_medium', 'name_length_long',
            'multiple_names', 'name_has_special_chars', 'name_numeric'
        ])
        
        # Address features (20)
        self.feature_names.extend([
            'has_address', 'has_postal_address', 'has_structured_address',
            'address_in_cdtrAgt', 'address_in_dbtrAgt', 'address_in_instgAgt', 'address_in_instdAgt',
            'address_in_cdtr', 'address_in_dbtr',
            'missing_address_has_bic', 'address_multiline', 'address_has_country',
            'address_has_city', 'address_has_state', 'address_has_zip', 'address_has_street',
            'address_line_count', 'address_total_length', 'address_needs_standardization', 'address_has_po_box'
        ])
        
        # Country features (15)
        self.feature_names.extend([
            'has_country_field', 'country_in_cdtrAgt', 'country_in_dbtrAgt', 'country_in_instgAgt',
            'country_in_cdtr', 'country_in_dbtr', 'country_in_address',
            'missing_country_has_bic', 'missing_country_has_iban', 'country_code_valid',
            'country_code_length_2', 'multiple_countries', 'country_mismatch', 'country_from_bic_available', 'country_from_iban_available'
        ])
        
        # Account features (12)
        self.feature_names.extend([
            'has_cdtrAcct', 'has_dbtrAcct', 'has_account_id', 'has_account_type',
            'account_type_iban', 'account_type_other', 'account_type_proprietary',
            'has_account_currency', 'currency_usd', 'currency_eur', 'currency_other',
            'multiple_accounts'
        ])
        
        # Entity presence features (15)
        self.feature_names.extend([
            'has_cdtr', 'has_dbtr', 'has_cdtrAgt', 'has_dbtrAgt',
            'has_instgAgt', 'has_instdAgt',
            'has_intrmyAgt1', 'has_intrmyAgt2', 'has_intrmyAgt',
            'has_rmtInf', 'has_creditor_side', 'has_debtor_side', 'has_instructing_side', 'has_instructed_side',
            'entity_count'
        ])
        
        # Financial institution ID features (10)
        self.feature_names.extend([
            'has_fininstnid', 'fininstnid_in_multiple_entities',
            'fininstnid_has_bic', 'fininstnid_has_clearing', 'fininstnid_has_name',
            'fininstnid_has_address', 'fininstnid_complete', 'fininstnid_incomplete',
            'fininstnid_in_cdtrAgt', 'fininstnid_in_dbtrAgt'
        ])
        
        # Remittance features (12)
        self.feature_names.extend([
            'has_rmtInf', 'rmtInf_has_ustrd', 'rmtInf_has_strd',
            'rmtInf_length', 'rmtInf_needs_split', 'rmtInf_multiline',
            'rmtInf_has_invoice', 'rmtInf_has_reference', 'rmtInf_all_caps',
            'rmtInf_has_numbers', 'rmtInf_has_special_chars', 'rmtInf_line_count'
        ])
        
        # Source/Clearing metadata (8)
        self.feature_names.extend([
            'source_swf', 'source_fed', 'source_chips', 'source_other',
            'clearing_fed', 'clearing_chips', 'clearing_swift', 'clearing_other'
        ])
        
        # Flags features (10)
        self.feature_names.extend([
            'flag_ace_repairs', 'flag_autotrieve', 'flag_stp_failed', 
            'flag_repair_queue', 'flag_exception_queue', 'flag_verify_queue',
            'flag_ignored_party', 'flag_credit_party_inserted', 'flag_non_ace_repairs',
            'flags_count'
        ])
        
        # Structural features (15)
        self.feature_names.extend([
            'total_fields', 'total_entities', 'total_leaf_values',
            'max_nesting_depth', 'has_arrays', 'array_count',
            'has_empty_fields', 'empty_field_ratio',
            'field_density', 'entity_balance', 'agent_to_party_ratio',
            'has_complete_routing', 'has_partial_routing', 'complexity_score', 'data_completeness'
        ])
        
        logger.info(f"Initialized {len(self.feature_names)} features")
    
    def extract_features(self, payment: Dict) -> np.ndarray:
        """Extract all 172 features from a payment"""
        features = np.zeros(len(self.feature_names))
        payment = self._normalize_payment(payment)
        
        idx = 0
        bic_info = self._extract_bic_info(payment)
        features[idx:idx+15] = list(bic_info.values())
        idx += 15
        
        iban_info = self._extract_iban_info(payment)
        features[idx:idx+10] = list(iban_info.values())
        idx += 10
        
        clearing_info = self._extract_clearing_info(payment)
        features[idx:idx+12] = list(clearing_info.values())
        idx += 12
        
        name_info = self._extract_name_info(payment)
        features[idx:idx+18] = list(name_info.values())
        idx += 18
        
        address_info = self._extract_address_info(payment)
        features[idx:idx+20] = list(address_info.values())
        idx += 20
        
        country_info = self._extract_country_info(payment)
        features[idx:idx+15] = list(country_info.values())
        idx += 15
        
        account_info = self._extract_account_info(payment)
        features[idx:idx+12] = list(account_info.values())
        idx += 12
        
        entity_info = self._extract_entity_info(payment)
        features[idx:idx+15] = list(entity_info.values())
        idx += 15
        
        fininstn_info = self._extract_fininstn_info(payment)
        features[idx:idx+10] = list(fininstn_info.values())
        idx += 10
        
        rmt_info = self._extract_remittance_info(payment)
        features[idx:idx+12] = list(rmt_info.values())
        idx += 12
        
        metadata_info = self._extract_metadata_info(payment)
        features[idx:idx+8] = list(metadata_info.values())
        idx += 8
        
        flags_info = self._extract_flags_info(payment)
        features[idx:idx+10] = list(flags_info.values())
        idx += 10
        
        struct_info = self._extract_structural_info(payment)
        features[idx:idx+15] = list(struct_info.values())
        idx += 15
        
        return features
    
    def _normalize_payment(self, payment: Dict) -> Dict:
        if isinstance(payment, dict):
            return {k.lower() if isinstance(k, str) else k: self._normalize_payment(v) 
                   for k, v in payment.items()}
        elif isinstance(payment, list):
            return [self._normalize_payment(item) for item in payment]
        return payment
    
    # Include all the extraction methods from original script
    # (I'll include a few key ones, but in practice you'd copy all from original)
    
    def _extract_bic_info(self, payment: Dict) -> Dict:
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
    
    # Copy all other _extract_* methods from original script
    # For brevity, I'm showing structure - you'd include all methods
    
    def _extract_iban_info(self, payment: Dict) -> Dict:
        # Copy from original
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
        clearing_fields = self._find_all_values(payment, 'mmbid')
        clearing_sys = self._find_all_values(payment, 'clrsysid')
        clearing_cd = self._find_all_values(payment, 'cd')
        
        has_clearing = len(clearing_fields) > 0
        has_bic = len(self._find_all_values(payment, 'bic')) > 0
        
        clearing_type = str(clearing_sys[0]).lower() if clearing_sys else ""
        clearing_id_val = str(clearing_fields[0]) if clearing_fields else ""
        
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
    
    # Continue with all other extraction methods...
    # (Copy from original script - too long to include all here)
    
    def _extract_name_info(self, payment: Dict) -> Dict:
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
        fininstn_fields = self._find_all_values(payment, 'fininstnid')
        has_fininstn = len(fininstn_fields) > 0
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
            'source_other': float(bool(source) and source not in ['swf', 'fed', 'fedwire', 'chips']),
            'clearing_fed': float(clearing == 'fed' or clearing == 'fedwire'),
            'clearing_chips': float(clearing == 'chips'),
            'clearing_swift': float(clearing == 'swift'),
            'clearing_other': float(bool(clearing) and clearing not in ['fed', 'fedwire', 'chips', 'swift'])
        }
    
    def _extract_flags_info(self, payment: Dict) -> Dict:
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
        total_fields = self._count_all_fields(payment)
        total_entities = sum(1 for k in payment.keys() if isinstance(payment.get(k), dict))
        total_leaf = self._count_leaf_values(payment)
        max_depth = self._get_max_depth(payment)
        arrays = self._count_arrays(payment)
        empty_fields = self._count_empty_fields(payment)
        agents = sum(1 for k in payment.keys() if 'agt' in k.lower())
        parties = sum(1 for k in payment.keys() if k.lower() in ['cdtr', 'dbtr'])
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
    
    # Helper methods
    def _is_valid_bic(self, bic: str) -> bool:
        if not bic or not isinstance(bic, str):
            return False
        return len(bic) in [8, 11] and bic[:6].isalpha() and bic[6:8].isalnum()
    
    def _is_valid_iban(self, iban: str) -> bool:
        if not iban or not isinstance(iban, str):
            return False
        iban = iban.replace(' ', '')
        return len(iban) >= 15 and iban[:2].isalpha() and iban[2:4].isdigit()
    
    def _bic_country_matches_field(self, payment: Dict, bic: str) -> bool:
        if not bic or len(bic) < 6:
            return False
        bic_country = bic[4:6].upper()
        country_fields = self._find_all_values(payment, 'ctryofres')
        country_fields.extend(self._find_all_values(payment, 'ctry'))
        return any(bic_country == str(c).upper() for c in country_fields)
    
    def _iban_country_matches_field(self, payment: Dict, iban: str) -> bool:
        if not iban or len(iban) < 2:
            return False
        iban_country = iban[:2].upper()
        country_fields = self._find_all_values(payment, 'ctryofres')
        country_fields.extend(self._find_all_values(payment, 'ctry'))
        return any(iban_country == str(c).upper() for c in country_fields)
    
    def _count_leaf_values(self, obj) -> int:
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
        entity = entity.lower()
        if entity not in payment:
            return False
        entity_data = payment[entity]
        values = self._find_all_values(entity_data, field)
        return len(values) > 0
    
    def _count_all_fields(self, obj) -> int:
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
# ENHANCED DETERMINISTIC RULES
# ============================================================================

class EnhancedDeterministicRules:
    
    """Enhanced rule-based system using discovered patterns"""
    
    def __init__(self, analysis: Optional[Dict] = None):
        self.rule_stats = Counter()
        self.discovered_rules = []
        self.debug_mode = True  # Enable debug logging
        
        if analysis:
            self.discovered_rules = analysis.get('deterministic_rules', [])
            logger.info(f"Loaded {len(self.discovered_rules)} discovered rules from analysis")
            for rule in self.discovered_rules:
                logger.info(f"  - {rule['repair_id']}: {rule['description']} (confidence: {rule['confidence']:.1%})")
    
    def predict_repairs(self, payment: Dict, features: np.ndarray) -> Tuple[List[str], List[float]]:
        """Apply enhanced deterministic rules with debug logging"""
        repairs = []
        confidences = []
        payment = self._normalize(payment)
        
        if self.debug_mode:
            logger.info("\n" + "="*60)
            logger.info("RULE EVALUATION DEBUG")
            logger.info("="*60)
            logger.info(f"Feature[0] has_bic: {features[0]:.3f}")
            logger.info(f"Feature[25] has_clearing: {features[25]:.3f}")
            logger.info(f"Feature[37] has_name: {features[37]:.3f}")
            logger.info(f"Feature[75] has_country: {features[75]:.3f}")
            logger.info(f"Feature[15] has_iban: {features[15]:.3f}")
        
        # Rule 1: Country from BIC
        if self._needs_country_from_bic(payment, features):
            repairs.append('6021')
            confidences.append(1.0)
            self.rule_stats['6021_country_from_bic'] += 1
            if self.debug_mode:
                logger.info("✓ Rule fired: 6021 (Country from BIC)")
        
        # Rule 2: BIC from clearing - ENHANCED
        if self._needs_bic_from_clearing(payment, features):
            repairs.append('6035')
            confidences.append(1.0)
            self.rule_stats['6035_bic_from_clearing'] += 1
            if self.debug_mode:
                logger.info("✓ Rule fired: 6035 (BIC from clearing)")
                # Show which clearing ID was found
                clearing_ids = self._find_all_values(payment, 'mmbid')
                logger.info(f"  Clearing IDs found: {clearing_ids}")
        elif self.debug_mode:
            logger.info("✗ Rule 6035 not fired (BIC from clearing)")
            logger.info(f"  has_clearing={features[25]:.3f}, has_bic={features[0]:.3f}")
        
        # Rule 3: Bank name from BIC
        if self._needs_bank_name_from_bic(payment, features):
            repairs.append('6036')
            confidences.append(1.0)
            self.rule_stats['6036_name_from_bic'] += 1
            if self.debug_mode:
                logger.info("✓ Rule fired: 6036 (Name from BIC)")
        
        # Enhanced rules from analysis
        if self._needs_address_standardization(payment, features):
            repair_added = False
            for rule in self.discovered_rules:
                desc = rule.get('description', '').lower()
                if 'address' in desc or 'adrline' in desc:
                    if rule['repair_id'] not in repairs:
                        repairs.append(rule['repair_id'])
                        confidences.append(rule['confidence'])
                        self.rule_stats[f"{rule['repair_id']}_address_std"] += 1
                        repair_added = True
                        if self.debug_mode:
                            logger.info(f"✓ Rule fired: {rule['repair_id']} (Address standardization)")
                        break
        
        if self._needs_remittance_split(payment, features):
            for rule in self.discovered_rules:
                desc = rule.get('description', '').lower()
                if 'remittance' in desc or 'rmtinf' in desc or 'ustrd' in desc:
                    if rule['repair_id'] not in repairs:
                        repairs.append(rule['repair_id'])
                        confidences.append(rule['confidence'])
                        self.rule_stats[f"{rule['repair_id']}_rmt_split"] += 1
                        if self.debug_mode:
                            logger.info(f"✓ Rule fired: {rule['repair_id']} (Remittance split)")
                        break
        
        if self._needs_country_from_iban(payment, features):
            if '6021' not in repairs:
                repairs.append('6021')
                confidences.append(0.95)
                self.rule_stats['6021_country_from_iban'] += 1
                if self.debug_mode:
                    logger.info("✓ Rule fired: 6021 (Country from IBAN)")
        
        # Apply high-confidence discovered rules
        for rule in self.discovered_rules:
            if rule['confidence'] >= 0.95 and rule['repair_id'] not in repairs:
                if self._check_rule_condition(payment, features, rule):
                    repairs.append(rule['repair_id'])
                    confidences.append(rule['confidence'])
                    self.rule_stats[f"{rule['repair_id']}_discovered"] += 1
                    if self.debug_mode:
                        logger.info(f"✓ Rule fired: {rule['repair_id']} (Discovered rule)")
        
        if self.debug_mode:
            logger.info(f"\nTotal rules fired: {len(repairs)}")
            logger.info("="*60 + "\n")
        
        return repairs, confidences
    
    def _needs_address_standardization(self, payment: Dict, features: np.ndarray) -> bool:
        addresses = self._find_all_values(payment, 'adrline')
        if not addresses:
            return False
        total_len = sum(len(str(a)) for a in addresses)
        return total_len > 140 or len(addresses) > 3
    
    def _needs_remittance_split(self, payment: Dict, features: np.ndarray) -> bool:
        rmt_fields = self._find_all_values(payment, 'ustrd')
        if not rmt_fields:
            return False
        for rmt in rmt_fields:
            if len(str(rmt)) > 140:
                return True
        return False
    
    def _needs_country_from_iban(self, payment: Dict, features: np.ndarray) -> bool:
        has_iban = features[15] > 0.5
        has_country = features[75] > 0.5
        has_bic = features[0] > 0.5
        return has_iban and not has_bic and not has_country
    
    def _check_rule_condition(self, payment: Dict, features: np.ndarray, rule: Dict) -> bool:
        condition = rule.get('condition', '')
        
        if 'AND' in condition:
            parts = [p.strip() for p in condition.split('AND')]
            return all(self._eval_condition_part(payment, features, part) for part in parts)
        else:
            return self._eval_condition_part(payment, features, condition)
    
    def _eval_condition_part(self, payment: Dict, features: np.ndarray, condition: str) -> bool:
        condition = condition.lower()
        
        if 'not has_bic' in condition or 'not has_country' in condition:
            if 'bic' in condition:
                return features[0] < 0.5
            if 'country' in condition:
                return features[75] < 0.5
        
        if 'has_bic' in condition:
            return features[0] > 0.5
        if 'has_clearing' in condition:
            return features[25] > 0.5
        if 'has_name' in condition:
            return features[37] > 0.5
        if 'changes_in_' in condition:
            entity = condition.split('changes_in_')[1]
            return entity in payment
        
        return False
    
    def _normalize(self, payment: Dict) -> Dict:
        if isinstance(payment, dict):
            return {k.lower() if isinstance(k, str) else k: self._normalize(v) 
                   for k, v in payment.items()}
        elif isinstance(payment, list):
            return [self._normalize(item) for item in payment]
        return payment
    
    def _needs_country_from_bic(self, payment: Dict, features: np.ndarray) -> bool:
        has_bic = features[0] > 0.5
        has_country = features[75] > 0.5
        return has_bic and not has_country
    
    def _needs_bic_from_clearing(self, payment: Dict, features: np.ndarray) -> bool:
        has_clearing = features[25] > 0.5
        has_bic = features[0] > 0.5
        return has_clearing and not has_bic
    
    def _needs_bank_name_from_bic(self, payment: Dict, features: np.ndarray) -> bool:
        has_bic = features[0] > 0.5
        has_name = features[37] > 0.5
        return has_bic and not has_name
    
    def _find_all_values(self, obj, key: str) -> List:
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
    
    def print_stats(self):
        logger.info("\nEnhanced Rule Statistics:")
        for rule, count in self.rule_stats.most_common():
            logger.info(f"  {rule}: {count}")


# ============================================================================
# PER-REPAIR METRICS
# ============================================================================

class PerRepairMetrics:
    """Compute and report per-repair-code metrics"""
    
    @staticmethod
    def compute_detailed_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                 repair_vocabulary: Dict) -> Dict:
        """Compute metrics for each repair code"""
        metrics = {'per_repair': {}, 'overall': {}}
        
        idx_to_repair = {v: k for k, v in repair_vocabulary.items()}
        
        # Overall metrics
        exact_match = np.mean(np.all(y_pred == y_true, axis=1))
        hamming = np.mean(y_pred == y_true)
        
        metrics['overall'] = {
            'exact_match_accuracy': float(exact_match),
            'hamming_accuracy': float(hamming),
            'total_samples': len(y_true)
        }
        
        # Per-repair metrics
        for idx, repair_id in idx_to_repair.items():
            y_true_repair = y_true[:, idx]
            y_pred_repair = y_pred[:, idx]
            
            tp = np.sum((y_true_repair == 1) & (y_pred_repair == 1))
            fp = np.sum((y_true_repair == 0) & (y_pred_repair == 1))
            fn = np.sum((y_true_repair == 1) & (y_pred_repair == 0))
            tn = np.sum((y_true_repair == 0) & (y_pred_repair == 0))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            accuracy = (tp + tn) / len(y_true_repair)
            
            support = int(np.sum(y_true_repair))
            
            metrics['per_repair'][repair_id] = {
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'accuracy': float(accuracy),
                'support': support,
                'true_positives': int(tp),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_negatives': int(tn)
            }
        
        return metrics
    
    @staticmethod
    def print_detailed_report(metrics: Dict):
        """Print detailed per-repair report"""
        logger.info("\n" + "="*70)
        logger.info("DETAILED PER-REPAIR METRICS")
        logger.info("="*70)
        
        overall = metrics['overall']
        logger.info(f"\nOverall Performance:")
        logger.info(f"  Exact Match Accuracy: {overall['exact_match_accuracy']:.1%}")
        logger.info(f"  Hamming Accuracy: {overall['hamming_accuracy']:.1%}")
        logger.info(f"  Total Samples: {overall['total_samples']}")
        
        logger.info(f"\nPer-Repair Performance:")
        logger.info(f"{'Repair':<10} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Accuracy':<12} {'Support':<10}")
        logger.info("-" * 70)
        
        sorted_repairs = sorted(metrics['per_repair'].items(), 
                               key=lambda x: x[1]['support'], reverse=True)
        
        for repair_id, m in sorted_repairs:
            logger.info(
                f"{repair_id:<10} "
                f"{m['precision']:>10.3f}  "
                f"{m['recall']:>10.3f}  "
                f"{m['f1_score']:>10.3f}  "
                f"{m['accuracy']:>10.3f}  "
                f"{m['support']:>8d}"
            )
        
        logger.info(f"\nRepairs Needing Attention (F1 < 0.7 and Support > 10):")
        problem_repairs = [
            (rid, m) for rid, m in metrics['per_repair'].items()
            if m['f1_score'] < 0.7 and m['support'] > 10
        ]
        
        if problem_repairs:
            for repair_id, m in sorted(problem_repairs, key=lambda x: x[1]['f1_score']):
                logger.info(f"  {repair_id}: F1={m['f1_score']:.3f}, Support={m['support']}")
                logger.info(f"    Precision={m['precision']:.3f}, Recall={m['recall']:.3f}")
        else:
            logger.info("  None - All repairs performing well!")


# ============================================================================
# CLASS WEIGHTS
# ============================================================================

def compute_class_weights(y_train: np.ndarray, max_weight: float = 10.0) -> torch.Tensor:
    """Compute class weights for imbalanced repairs"""
    pos_counts = y_train.sum(axis=0)
    neg_counts = len(y_train) - pos_counts
    
    # Avoid division by zero
    pos_counts = np.maximum(pos_counts, 1)
    neg_counts = np.maximum(neg_counts, 1)
    
    # Weight = neg_count / pos_count
    weights = neg_counts / pos_counts
    
    # Cap weights
    weights = np.minimum(weights, max_weight)
    
    return torch.FloatTensor(weights)


# ============================================================================
# ML MODELS (Same as original)
# ============================================================================

class RepairPredictor(nn.Module):
    """Neural network for repair prediction"""
    
    def __init__(self, num_features: int, num_repairs: int, hidden_dim: int = 256, 
                 dropout: float = 0.3, use_batchnorm: bool = True):
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(num_features, hidden_dim))
        layers.append(nn.ReLU())
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(hidden_dim, hidden_dim // 2))
        layers.append(nn.ReLU())
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dim // 2))
        layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(hidden_dim // 2, num_repairs))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
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
# DATA PROCESSOR (Same as original)
# ============================================================================

class DataProcessor:
    """Process training data and prepare for modeling"""
    
    def __init__(self):
        self.feature_extractor = PaymentFeatureExtractor()
        self.repair_vocabulary = {}
        self.idx_to_repair = {}
    
    def load_and_process(self, json_file: str) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """Load training data and extract features + labels"""
        logger.info(f"Loading data from {json_file}")
        
        with open(json_file, 'r') as f:
            raw_data = json.load(f)
        
        if isinstance(raw_data, list):
            if len(raw_data) == 0:
                raise ValueError(f"Empty array in {json_file}")
            
            if isinstance(raw_data[0], dict):
                first_item = raw_data[0]
                
                # Check if values are dicts (transaction objects)
                if all(isinstance(v, dict) for v in first_item.values()):
                    # Merge all dicts in the array
                    data = {}
                    for item in raw_data:
                        if isinstance(item, dict):
                            data.update(item)
                    logger.info(f"Detected format: Array of {len(raw_data)} dicts merged into {len(data)} transactions")
                else:
                    data = {f"txn_{i:06d}": txn for i, txn in enumerate(raw_data)}
                    logger.info(f"Detected format: Array of {len(data)} transaction objects")
                    
        elif isinstance(raw_data, dict):
            data = raw_data
            logger.info(f"Detected format: Dict of {len(data)} transactions")
        else:
            raise ValueError(f"Unexpected data format in {json_file}: {type(raw_data)}")
        
        self._build_repair_vocabulary(data)
        
        entity_stats = Counter()
        skipped_count = 0
        processed_count = 0
        
        all_features = []
        all_labels = []
        all_payments = []
        
        for txn_idx, (txn_id, txn_data) in enumerate(data.items()):
            try:
                has_after = self._has_after_state(txn_data)
                
                if not has_after:
                    skipped_count += 1
                    if skipped_count <= 3:
                        logger.info(f"Skipping {txn_id}: No 'after' state")
                    continue
                
                repairs = [r['id'] for r in txn_data.get('ace', [])]
                payment_for_features = self._extract_before_state(txn_data)
                
                present_entities = [k for k in payment_for_features.keys() 
                                   if k not in ['source', 'clearing', 'flags', 'parties']]
                entity_stats.update(present_entities)
                
                if processed_count < 3:
                    logger.info(f"Training on {txn_id}: Entities: {present_entities}, Repairs: {repairs}")
                
                features = self.feature_extractor.extract_features(payment_for_features)
                labels = self._repairs_to_labels(repairs)
                
                all_features.append(features)
                all_labels.append(labels)
                all_payments.append(txn_data)
                processed_count += 1
                
            except Exception as e:
                logger.warning(f"Error processing transaction {txn_id}: {e}")
                continue
        
        if len(all_features) == 0:
            raise ValueError(f"No trainable transactions found in {json_file}")
        
        features_array = np.array(all_features)
        labels_array = np.array(all_labels)
        
        logger.info(f"\n{'='*70}")
        logger.info(f"DATA PROCESSING SUMMARY")
        logger.info(f"{'='*70}")
        logger.info(f"Total transactions: {len(data)}")
        logger.info(f"Trainable: {processed_count}")
        logger.info(f"Skipped: {skipped_count}")
        logger.info(f"Feature shape: {features_array.shape}")
        logger.info(f"Label shape: {labels_array.shape}")
        logger.info(f"Unique repairs: {len(self.repair_vocabulary)}")
        
        return features_array, labels_array, all_payments
    
    def _has_after_state(self, txn_data: Dict) -> bool:
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
        payment = {}
        
        for key in ['source', 'clearing', 'flags', 'parties']:
            if key in txn_data:
                payment[key] = txn_data[key]
        
        possible_entities = [
            'cdtr', 'dbtr', 'cdtrAgt', 'dbtrAgt', 
            'cdtrAcct', 'dbtrAcct', 'instgAgt', 'instdAgt',
            'intrmyAgt1', 'intrmyAgt2', 'rmtInf'
        ]
        
        for entity in possible_entities:
            for key in txn_data.keys():
                if key.lower() == entity.lower():
                    entity_data = txn_data[key]
                    
                    if isinstance(entity_data, dict) and 'before' in entity_data:
                        payment[entity] = entity_data['before']
                    elif isinstance(entity_data, dict):
                        payment[entity] = entity_data
                    
                    break
        
        return payment
    
    def _build_repair_vocabulary(self, data: Dict):
        all_repairs = set()
        
        for txn_data in data.values():
            for repair in txn_data.get('ace', []):
                all_repairs.add(repair['id'])
        
        sorted_repairs = sorted(all_repairs)
        
        self.repair_vocabulary = {repair_id: idx for idx, repair_id in enumerate(sorted_repairs)}
        self.idx_to_repair = {idx: repair_id for repair_id, idx in self.repair_vocabulary.items()}
        
        logger.info(f"Built vocabulary with {len(self.repair_vocabulary)} repairs: {sorted_repairs}")
    
    def _repairs_to_labels(self, repairs: List[str]) -> np.ndarray:
        labels = np.zeros(len(self.repair_vocabulary))
        
        for repair_id in repairs:
            if repair_id in self.repair_vocabulary:
                idx = self.repair_vocabulary[repair_id]
                labels[idx] = 1.0
        
        return labels
    
    def labels_to_repairs(self, labels: np.ndarray, threshold: float = 0.5) -> List[str]:
        repairs = []
        
        for idx, prob in enumerate(labels):
            if prob > threshold:
                repairs.append(self.idx_to_repair[idx])
        
        return repairs


# ============================================================================
# ENHANCED HYBRID PREDICTOR
# ============================================================================

class EnhancedHybridPredictor:
    """Enhanced hybrid predictor with class weights and detailed metrics"""
    
    def __init__(self, config: Config, analysis: Optional[Dict] = None):
        self.config = config
        self.feature_extractor = PaymentFeatureExtractor()
        self.rules = EnhancedDeterministicRules(analysis)
        self.ml_model = None
        self.rf_model = None
        self.processor = DataProcessor()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_weights = None
        self.analysis = analysis

    def debug_features(self, payment: Dict) -> None:
        """Debug helper to show extracted features"""
        features = self.feature_extractor.extract_features(payment)
        
        logger.info("\n" + "="*60)
        logger.info("FEATURE EXTRACTION DEBUG")
        logger.info("="*60)
        
        important_features = {
            0: 'has_bic',
            15: 'has_iban',
            25: 'has_clearing_id',
            37: 'has_bank_name',
            75: 'has_country_field'
        }
        
        for idx, name in important_features.items():
            logger.info(f"Feature[{idx:3d}] {name:20s}: {features[idx]:.3f}")
        
        # Show what was found in payment
        payment_norm = self.feature_extractor._normalize_payment(payment)
        bics = self.feature_extractor._find_all_values(payment_norm, 'bic')
        clearings = self.feature_extractor._find_all_values(payment_norm, 'mmbid')
        countries = self.feature_extractor._find_all_values(payment_norm, 'ctryofres')
        
        logger.info(f"\nValues found in payment:")
        logger.info(f"  BICs: {bics}")
        logger.info(f"  Clearing IDs: {clearings}")
        logger.info(f"  Countries: {countries}")
        logger.info("="*60 + "\n")
    
    def train(self, train_file: str):
        """Train the enhanced hybrid model"""
        logger.info("="*70)
        logger.info("TRAINING ENHANCED HYBRID REPAIR PREDICTOR")
        logger.info("="*70)
        
        features, labels, payments = self.processor.load_and_process(train_file)
        
        n = len(features)
        
        if n < 10:
            logger.warning(f"\nSmall dataset detected ({n} samples)")
            X_train = features
            y_train = labels
            X_val = features
            y_val = labels
            X_test = features
            y_test = labels
        else:
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
        
        # Compute class weights
        if self.config.use_class_weights:
            self.class_weights = compute_class_weights(y_train, self.config.max_class_weight)
            logger.info(f"\nClass Weights:")
            for idx, weight in enumerate(self.class_weights):
                repair_id = self.processor.idx_to_repair[idx]
                support = int(y_train[:, idx].sum())
                logger.info(f"  {repair_id}: {weight:.2f}x (support: {support})")
        
        # Train Random Forest
        logger.info("\nTraining Random Forest...")
        self.rf_model = MultiOutputClassifier(
            RandomForestClassifier(
                n_estimators=min(200, max(10, n * 10)),
                max_depth=min(15, max(3, n // 2)),
                min_samples_split=max(2, n // 10),
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1
            )
        )
        self.rf_model.fit(X_train, y_train)
        
        # Train Neural Network
        logger.info("\nTraining Neural Network...")
        use_batchnorm = n >= 10
        
        self.ml_model = RepairPredictor(
            num_features=features.shape[1],
            num_repairs=labels.shape[1],
            hidden_dim=min(self.config.hidden_dim, max(32, n * 10)),
            dropout=self.config.dropout,
            use_batchnorm=use_batchnorm
        ).to(self.device)
        
        self._train_neural_network(X_train, y_train, X_val, y_val)
        
        # Final evaluation
        logger.info("\n" + "="*70)
        logger.info("FINAL TEST SET EVALUATION")
        logger.info("="*70)
        
        test_metrics = self.evaluate_detailed(X_test, y_test)
        
        # Save models
        self.save_models()
        
        return test_metrics
    
    def _train_neural_network(self, X_train, y_train, X_val, y_val):
        """Train neural network with class weights"""
        train_dataset = RepairDataset(X_train, y_train)
        val_dataset = RepairDataset(X_val, y_val)
        
        batch_size = min(self.config.batch_size, max(1, len(X_train) // 2))
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        optimizer = torch.optim.AdamW(
            self.ml_model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=1e-4
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=5, factor=0.5
        )
        
        # Use weighted loss if class weights available
        if self.class_weights is not None and self.config.use_class_weights:
            # Expand weights to match batch size
            criterion = nn.BCELoss(reduction='none')
        else:
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
                
                if self.class_weights is not None and self.config.use_class_weights:
                    # Apply per-sample weights
                    loss_per_sample = criterion(predictions, batch_labels)
                    weights = self.class_weights.to(self.device)
                    weighted_loss = (loss_per_sample * weights).mean()
                    loss = weighted_loss
                else:
                    loss = criterion(predictions, batch_labels)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            val_predictions = []
            val_labels = []
            
            self.ml_model.eval()
            with torch.no_grad():
                for batch_features, batch_labels in val_loader:
                    batch_features = batch_features.to(self.device)
                    predictions = self.ml_model(batch_features)
                    val_predictions.append(predictions.cpu().numpy())
                    val_labels.append(batch_labels.numpy())
            
            val_predictions = np.vstack(val_predictions)
            val_labels = np.vstack(val_labels)
            val_predictions_binary = (val_predictions > 0.5).astype(int)
            
            val_metrics = PerRepairMetrics.compute_detailed_metrics(
                val_labels, val_predictions_binary, self.processor.repair_vocabulary
            )
            
            val_f1 = val_metrics['overall']['hamming_accuracy']
            
            scheduler.step(val_f1)
            
            if epoch % 5 == 0:
                logger.info(f"Epoch {epoch}/{self.config.num_epochs} - "
                           f"Loss: {train_loss:.4f} - "
                           f"Val Hamming Acc: {val_f1:.3f} - "
                           f"Val Exact Match: {val_metrics['overall']['exact_match_accuracy']:.3f}")
            
            # Early stopping
            if val_f1 > best_f1 + 0.001:
                best_f1 = val_f1
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
    
    def evaluate_detailed(self, X, y):
        """Evaluate with detailed per-repair metrics"""
        self.ml_model.eval()
        
        dataset = RepairDataset(X, y)
        loader = DataLoader(dataset, batch_size=self.config.batch_size)
        
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
        predictions_binary = (predictions > 0.5).astype(int)
        
        metrics = PerRepairMetrics.compute_detailed_metrics(
            labels, predictions_binary, self.processor.repair_vocabulary
        )
        
        PerRepairMetrics.print_detailed_report(metrics)
        
        # Print rule stats
        self.rules.print_stats()
        
        return metrics
    
    def predict(self, payment: Dict, use_rules: bool = True, use_ml: bool = True) -> Dict:
        """Predict repairs with dynamic thresholds for rare repairs"""
        features = self.feature_extractor.extract_features(payment)
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        repairs = []
        confidences = []
        sources = []
        
        # Apply rules first (highest priority)
        if use_rules:
            rule_repairs, rule_confs = self.rules.predict_repairs(payment, features)
            repairs.extend(rule_repairs)
            confidences.extend(rule_confs)
            sources.extend(['rule'] * len(rule_repairs))
            
            logger.info(f"\nRule-based predictions: {len(rule_repairs)} repairs")
            for r, c in zip(rule_repairs, rule_confs):
                logger.info(f"  {r}: {c:.3f}")
        
        # Apply ML with dynamic thresholds
        if use_ml:
            self.ml_model.eval()
            with torch.no_grad():
                ml_probs = self.ml_model(features_tensor)[0].cpu().numpy()
            
            # RF predictions
            rf_probs_raw = self.rf_model.predict_proba(features.reshape(1, -1))
            
            rf_probs = []
            for pred in rf_probs_raw:
                if pred.shape[1] == 1:
                    rf_probs.append(1.0 if pred[0, 0] > 0.5 else 0.0)
                else:
                    rf_probs.append(pred[0, 1])
            rf_probs = np.array(rf_probs)
            
            # Ensemble
            ensemble_probs = (ml_probs + rf_probs) / 2
            
            logger.info(f"\nML predictions (before threshold):")
            
            for idx, prob in enumerate(ensemble_probs):
                repair_id = self.processor.idx_to_repair[idx]
                
                # Skip if already predicted by rules
                if repair_id in repairs:
                    continue
                
                # Dynamic threshold based on training support
                # For rare repairs (support < 10), use lower threshold
                support = self.class_weights[idx].item() if self.class_weights is not None else 1.0
                
                # Inverse relationship: high weight = low support = lower threshold
                if support >= 5.0:  # Very rare (< 10 examples)
                    threshold = 0.2
                elif support >= 2.0:  # Rare (< 100 examples)
                    threshold = 0.35
                else:
                    threshold = self.config.ml_threshold  # 0.5
                
                if prob > threshold:
                    repairs.append(repair_id)
                    confidences.append(float(prob))
                    sources.append('ml')
                    logger.info(f"  {repair_id}: {prob:.3f} (threshold: {threshold:.2f}, support_weight: {support:.2f})")
        
        return {
            'repairs': repairs,
            'confidences': confidences,
            'sources': sources
        }
    
    def save_models(self):
        """Save all models and metadata"""
        os.makedirs(self.config.model_dir, exist_ok=True)
        
        # Save model architecture info - more robust detection
        # Count BatchNorm layers to determine if it was used
        has_batchnorm = any(isinstance(layer, nn.BatchNorm1d) for layer in self.ml_model.network)
        
        model_info = {
            'num_features': self.ml_model.network[0].in_features,
            'num_repairs': self.ml_model.network[-2].out_features,  # -2 because -1 is Sigmoid
            'hidden_dim': self.ml_model.network[0].out_features,
            'use_batchnorm': has_batchnorm
        }
        
        with open(os.path.join(self.config.model_dir, 'model_info.json'), 'w') as f:
            json.dump(model_info, f, indent=2)
        
        torch.save(self.ml_model.state_dict(), 
                os.path.join(self.config.model_dir, 'neural_model.pt'))
        
        with open(os.path.join(self.config.model_dir, 'rf_model.pkl'), 'wb') as f:
            pickle.dump(self.rf_model, f)
        
        with open(os.path.join(self.config.model_dir, 'processor.pkl'), 'wb') as f:
            pickle.dump(self.processor, f)
        
        if self.class_weights is not None:
            with open(os.path.join(self.config.model_dir, 'class_weights.pkl'), 'wb') as f:
                pickle.dump(self.class_weights, f)
        
        if self.analysis is not None:
            with open(os.path.join(self.config.model_dir, 'analysis.json'), 'w') as f:
                json.dump(self.analysis, f, indent=2)
        
        self.config.save(os.path.join(self.config.model_dir, 'config.json'))
        
        logger.info(f"\nModels saved to {self.config.model_dir}")
        logger.info(f"Architecture: {model_info['num_features']} -> {model_info['hidden_dim']} -> "
                    f"{model_info['hidden_dim']//2} -> {model_info['num_repairs']}")
        logger.info(f"BatchNorm: {model_info['use_batchnorm']}")


    def load_models(self, model_dir: str):
        """Load saved models from disk"""
        self.config = Config.load(os.path.join(model_dir, 'config.json'))
        
        with open(os.path.join(model_dir, 'processor.pkl'), 'rb') as f:
            self.processor = pickle.load(f)
        
        with open(os.path.join(model_dir, 'rf_model.pkl'), 'rb') as f:
            self.rf_model = pickle.load(f)
        
        # Load class weights if available
        weights_path = os.path.join(model_dir, 'class_weights.pkl')
        if os.path.exists(weights_path):
            with open(weights_path, 'rb') as f:
                self.class_weights = pickle.load(f)
                logger.info(f"Loaded class weights")
        
        # Load analysis if available
        analysis_path = os.path.join(model_dir, 'analysis.json')
        if os.path.exists(analysis_path):
            with open(analysis_path, 'r') as f:
                self.analysis = json.load(f)
                self.rules = EnhancedDeterministicRules(self.analysis)
                logger.info(f"Loaded analysis with {len(self.analysis.get('deterministic_rules', []))} rules")
        
        # Load model architecture info
        model_info_path = os.path.join(model_dir, 'model_info.json')
        if not os.path.exists(model_info_path):
            raise FileNotFoundError(
                f"Model info file not found: {model_info_path}\n"
                "This might be an old model. Please retrain."
            )
        
        with open(model_info_path, 'r') as f:
            model_info = json.load(f)
        
        logger.info(f"Loading model with architecture: {model_info['num_features']} -> "
                    f"{model_info['hidden_dim']} -> {model_info['hidden_dim']//2} -> "
                    f"{model_info['num_repairs']}")
        
        # Create model with saved architecture
        self.ml_model = RepairPredictor(
            num_features=model_info['num_features'],
            num_repairs=model_info['num_repairs'],
            hidden_dim=model_info['hidden_dim'],
            dropout=self.config.dropout,
            use_batchnorm=model_info['use_batchnorm']
        ).to(self.device)
        
        self.ml_model.load_state_dict(
            torch.load(os.path.join(model_dir, 'neural_model.pt'), 
                    map_location=self.device)
        )
        self.ml_model.eval()
        
        logger.info(f"Models loaded successfully from {model_dir}")


# ============================================================================
# CLI COMMANDS
# ============================================================================

def debug_features(self, payment: Dict) -> None:
    """Debug helper to show extracted features"""
    features = self.feature_extractor.extract_features(payment)
    
    logger.info("\n" + "="*60)
    logger.info("FEATURE EXTRACTION DEBUG")
    logger.info("="*60)
    
    important_features = {
        0: 'has_bic',
        15: 'has_iban',
        25: 'has_clearing_id',
        37: 'has_bank_name',
        75: 'has_country_field'
    }
    
    for idx, name in important_features.items():
        logger.info(f"Feature[{idx:3d}] {name:20s}: {features[idx]:.3f}")
    
    # Show what was found in payment
    payment_norm = self.feature_extractor._normalize_payment(payment)
    bics = self.feature_extractor._find_all_values(payment_norm, 'bic')
    clearings = self.feature_extractor._find_all_values(payment_norm, 'mmbid')
    countries = self.feature_extractor._find_all_values(payment_norm, 'ctryofres')
    
    logger.info(f"\nValues found in payment:")
    logger.info(f"  BICs: {bics}")
    logger.info(f"  Clearing IDs: {clearings}")
    logger.info(f"  Countries: {countries}")
    logger.info("="*60 + "\n")
    
def train_command(args):
    """Train the enhanced model"""
    config = Config()
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.use_class_weights = not args.no_class_weights
    
    # Load analysis if provided
    analysis = None
    if args.analysis:
        logger.info(f"Loading analysis from {args.analysis}")
        with open(args.analysis, 'r') as f:
            analysis = json.load(f)
    
    predictor = EnhancedHybridPredictor(config, analysis)
    metrics = predictor.train(args.input)
    
    logger.info("\n" + "="*70)
    logger.info("TRAINING COMPLETE")
    logger.info("="*70)
    logger.info(f"Test Exact Match Accuracy: {metrics['overall']['exact_match_accuracy']:.1%}")
    logger.info(f"Test Hamming Accuracy: {metrics['overall']['hamming_accuracy']:.1%}")
    
    if metrics['overall']['exact_match_accuracy'] >= 0.90:
        logger.info("GOAL ACHIEVED: 90%+ exact match accuracy!")
    else:
        gap = 0.90 - metrics['overall']['exact_match_accuracy']
        logger.info(f"Need {gap:.1%} more to reach 90% exact match")


def predict_command(args):
    """Predict repairs for a payment"""
    predictor = EnhancedHybridPredictor(Config())
    predictor.load_models(args.model or './models')
    
    with open(args.input, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, dict) and len(data) == 1:
        txn_id = list(data.keys())[0]
        payment_data = data[txn_id]
        
        if predictor.processor._has_after_state(payment_data):
            logger.info(f"Detected training format - extracting 'before' state")
            payment = predictor.processor._extract_before_state(payment_data)
        else:
            logger.info(f"Detected inference format")
            payment = payment_data
    else:
        payment = data
    
    # Debug features
    predictor.debug_features(payment)
    
    # Predict
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
    """Evaluate model with detailed metrics"""
    predictor = EnhancedHybridPredictor(Config())
    predictor.load_models(args.model or './models')
    
    features, labels, payments = predictor.processor.load_and_process(args.input)
    
    if args.detailed:
        metrics = predictor.evaluate_detailed(features, labels)
    else:
        # Quick evaluation
        self.ml_model.eval()
        dataset = RepairDataset(features, labels)
        loader = DataLoader(dataset, batch_size=predictor.config.batch_size)
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch_features, batch_labels in loader:
                batch_features = batch_features.to(predictor.device)
                predictions = predictor.ml_model(batch_features)
                all_predictions.append(predictions.cpu().numpy())
                all_labels.append(batch_labels.numpy())
        
        predictions = np.vstack(all_predictions)
        labels_np = np.vstack(all_labels)
        predictions_binary = (predictions > 0.5).astype(int)
        
        exact_match = np.mean(np.all(predictions_binary == labels_np, axis=1))
        hamming = np.mean(predictions_binary == labels_np)
        
        logger.info("\n" + "="*70)
        logger.info("EVALUATION RESULTS")
        logger.info("="*70)
        logger.info(f"Exact Match Accuracy: {exact_match:.1%}")
        logger.info(f"Hamming Accuracy: {hamming:.1%}")


def main():
    parser = argparse.ArgumentParser(
        description='Enhanced ACE Payment Repair Predictor',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train
    train_parser = subparsers.add_parser('train', help='Train the enhanced model')
    train_parser.add_argument('--input', required=True, help='Training data JSON file')
    train_parser.add_argument('--analysis', help='Analysis report JSON from analyzer')
    train_parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    train_parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    train_parser.add_argument('--no-class-weights', action='store_true', help='Disable class weighting')
    
    # Predict
    predict_parser = subparsers.add_parser('predict', help='Predict repairs for payment')
    predict_parser.add_argument('--input', required=True, help='Payment JSON file')
    predict_parser.add_argument('--output', help='Output file for results')
    predict_parser.add_argument('--model', default='./models', help='Model directory')
    
    # Evaluate
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model')
    eval_parser.add_argument('--input', required=True, help='Test data JSON file')
    eval_parser.add_argument('--model', default='./models', help='Model directory')
    eval_parser.add_argument('--detailed', action='store_true', help='Show detailed per-repair metrics')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_command(args)
    elif args.command == 'predict':
        predict_command(args)
    elif args.command == 'evaluate':
        eval_parser(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
