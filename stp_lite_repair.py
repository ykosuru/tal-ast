"""
stp_lite_repair.py
Author: Yekesa Kosuru
========================

ACE Payment Repair Predictor using Rule-Based + XGBoost Hybrid Approach

Key features:
- XGBoost handles small datasets (3,000 samples) well
- Built-in regularization prevents overfitting
- 10-100x faster than neural networks
- Interpretable feature importance
- Ensemble with Random Forest for diversity

Architecture:
1. Deterministic Rules (100% confidence) - applied first
2. XGBoost (primary ML model) - gradient boosting
3. Random Forest (ensemble member) - for diversity
4. Ensemble voting for final prediction

Usage:
    # Analyze repair distribution first (CRITICAL!)
    python stp_lite_repair.py analyze --input ./data_directory
    
    # Train on directory of JSON files
    python stp_lite_repair.py train --input ./data_directory --estimators 500
    
    # Predict repairs for single payment
    python stp_lite_repair.py predict --input payment.json --output result.json
    
    # Evaluate with detailed metrics
    python stp_lite_repair.py evaluate --input ./test_data_directory --detailed
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
import glob

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from xgboost import XGBClassifier

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
    """Configuration optimized for small datasets with XGBoost"""
    
    # XGBoost settings (optimized for tabular data)
    xgb_n_estimators: int = 500
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.05
    xgb_min_child_weight: int = 3
    xgb_gamma: float = 0.1  # Regularization
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    xgb_reg_alpha: float = 0.1  # L1 regularization
    xgb_reg_lambda: float = 1.0  # L2 regularization
    
    # Random Forest settings (ensemble diversity)
    rf_n_estimators: int = 200
    rf_max_depth: int = 12
    rf_min_samples_split: int = 10
    
    # Data split ratios
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    
    # Paths
    model_dir: str = "./models"
    
    # Prediction thresholds
    ml_threshold: float = 0.5  # Fixed threshold
    rule_confidence: float = 1.0
    min_confidence_rare_repairs: float = 0.7  # Higher bar for rare repairs
    
    # Class weighting
    use_class_weights: bool = True
    
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
# FEATURE EXTRACTION (Same as before - no changes needed)
# ============================================================================

class PaymentFeatureExtractor:
    """
    Extracts 172 domain-specific features from payment structure.
    
    Feature categories:
    - BIC (15), IBAN (10), Clearing (12)
    - Names (18), Addresses (20), Countries (15)
    - Accounts (12), Entities (15), FinInstn (10)
    - Remittance (12), Metadata (8), Flags (10)
    - Structural (15)
    """
    
    def __init__(self):
        self.feature_names = []
        self._build_feature_names()
    
    def _build_feature_names(self):
        """Define all 172 feature names in order"""
        
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
        """Extract all 172 features from a payment transaction"""
        features = np.zeros(len(self.feature_names))
        payment = self._normalize_payment(payment)
        
        idx = 0
        
        # Extract all feature groups
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
        """Recursively lowercase all string keys"""
        if isinstance(payment, dict):
            return {k.lower() if isinstance(k, str) else k: self._normalize_payment(v) 
                   for k, v in payment.items()}
        elif isinstance(payment, list):
            return [self._normalize_payment(item) for item in payment]
        return payment
    
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
    
    def _extract_iban_info(self, payment: Dict) -> Dict:
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
# DETERMINISTIC RULES (100% Confidence)
# ============================================================================

class DeterministicRules:
    """
    Rule-based system for high-confidence repairs.
    
    Rules are applied FIRST before ML models.
    These have 100% accuracy when applicable.
    """
    
    def __init__(self, analysis: Optional[Dict] = None):
        self.rule_stats = Counter()
        self.discovered_rules = []
        self.debug_mode = True
        
        if analysis:
            self.discovered_rules = analysis.get('deterministic_rules', [])
            logger.info(f"Loaded {len(self.discovered_rules)} discovered rules")
    
    def predict_repairs(self, payment: Dict, features: np.ndarray) -> Tuple[List[str], List[float]]:
        """Apply deterministic rules and return repairs with confidences"""
        repairs = []
        confidences = []
        payment = self._normalize(payment)
        
        if self.debug_mode:
            logger.info("\n" + "="*60)
            logger.info("RULE EVALUATION")
            logger.info("="*60)
            logger.info(f"has_bic: {features[0]:.1f}, has_clearing: {features[25]:.1f}, "
                       f"has_name: {features[37]:.1f}, has_country: {features[75]:.1f}")
        
        # Rule 1: Country from BIC (6021)
        if self._needs_country_from_bic(payment, features):
            repairs.append('6021')
            confidences.append(1.0)
            self.rule_stats['6021_country_from_bic'] += 1
            if self.debug_mode:
                logger.info("✓ Rule: 6021 (Country from BIC)")
        
        # Rule 2: BIC from clearing ID (6035)
        if self._needs_bic_from_clearing(payment, features):
            repairs.append('6035')
            confidences.append(1.0)
            self.rule_stats['6035_bic_from_clearing'] += 1
            if self.debug_mode:
                logger.info("✓ Rule: 6035 (BIC from clearing)")
        
        # Rule 3: Bank name from BIC (6036)
        if self._needs_bank_name_from_bic(payment, features):
            repairs.append('6036')
            confidences.append(1.0)
            self.rule_stats['6036_name_from_bic'] += 1
            if self.debug_mode:
                logger.info("✓ Rule: 6036 (Name from BIC)")
        
        # Apply discovered rules with high confidence
        for rule in self.discovered_rules:
            if rule['confidence'] >= 0.95 and rule['repair_id'] not in repairs:
                if self._check_rule_condition(payment, features, rule):
                    repairs.append(rule['repair_id'])
                    confidences.append(rule['confidence'])
                    self.rule_stats[f"{rule['repair_id']}_discovered"] += 1
                    if self.debug_mode:
                        logger.info(f"✓ Rule: {rule['repair_id']} (Discovered)")
        
        if self.debug_mode:
            logger.info(f"Total rule predictions: {len(repairs)}")
            logger.info("="*60 + "\n")
        
        return repairs, confidences
    
    def _normalize(self, payment: Dict) -> Dict:
        """Recursively normalize payment keys to lowercase"""
        if isinstance(payment, dict):
            return {k.lower() if isinstance(k, str) else k: self._normalize(v) 
                   for k, v in payment.items()}
        elif isinstance(payment, list):
            return [self._normalize(item) for item in payment]
        return payment
    
    def _needs_country_from_bic(self, payment: Dict, features: np.ndarray) -> bool:
        """Check if country should be derived from BIC"""
        return features[0] > 0.5 and features[75] < 0.5  # has_bic and not has_country
    
    def _needs_bic_from_clearing(self, payment: Dict, features: np.ndarray) -> bool:
        """Check if BIC should be looked up from clearing ID"""
        return features[25] > 0.5 and features[0] < 0.5  # has_clearing and not has_bic
    
    def _needs_bank_name_from_bic(self, payment: Dict, features: np.ndarray) -> bool:
        """Check if bank name should be looked up from BIC"""
        return features[0] > 0.5 and features[37] < 0.5  # has_bic and not has_name
    
    def _check_rule_condition(self, payment: Dict, features: np.ndarray, rule: Dict) -> bool:
        """Evaluate rule condition from discovered rules"""
        condition = rule.get('condition', '')
        if 'AND' in condition:
            parts = [p.strip() for p in condition.split('AND')]
            return all(self._eval_condition_part(payment, features, part) for part in parts)
        return self._eval_condition_part(payment, features, condition)
    
    def _eval_condition_part(self, payment: Dict, features: np.ndarray, condition: str) -> bool:
        """Evaluate a single condition part"""
        condition = condition.lower()
        if 'not has_bic' in condition:
            return features[0] < 0.5
        if 'not has_country' in condition:
            return features[75] < 0.5
        if 'has_bic' in condition:
            return features[0] > 0.5
        if 'has_clearing' in condition:
            return features[25] > 0.5
        if 'has_name' in condition:
            return features[37] > 0.5
        return False
    
    def print_stats(self):
        """Print rule firing statistics"""
        logger.info("\nRule Statistics:")
        for rule, count in self.rule_stats.most_common():
            logger.info(f"  {rule}: {count}")


# ============================================================================
# PER-REPAIR METRICS
# ============================================================================

class PerRepairMetrics:
    """Compute and report detailed per-repair-code metrics"""
    
    @staticmethod
    def compute_detailed_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                 repair_vocabulary: Dict) -> Dict:
        """
        Compute precision, recall, F1 for each repair code.
        
        Args:
            y_true: Ground truth labels (samples x repairs)
            y_pred: Predicted labels (samples x repairs)
            repair_vocabulary: Mapping of repair_id to index
            
        Returns:
            Dictionary with overall and per-repair metrics
        """
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
        """Print formatted per-repair report"""
        logger.info("\n" + "="*80)
        logger.info("DETAILED PER-REPAIR METRICS")
        logger.info("="*80)
        
        overall = metrics['overall']
        logger.info(f"\nOverall Performance:")
        logger.info(f"  Exact Match Accuracy: {overall['exact_match_accuracy']:.1%}")
        logger.info(f"  Hamming Accuracy: {overall['hamming_accuracy']:.1%}")
        logger.info(f"  Total Samples: {overall['total_samples']}")
        
        logger.info(f"\nPer-Repair Performance:")
        logger.info(f"{'Repair':<10} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Accuracy':<12} {'Support':<10}")
        logger.info("-" * 80)
        
        # Sort by support (most common first)
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
        
        # Highlight problem repairs
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
# DATA PROCESSOR (WITH DIRECTORY SUPPORT)
# ============================================================================

class DataProcessor:
    """
    Load and process payment data from files or directories.
    
    Supports:
    - Single JSON file: {"txn_id": {...}, ...}
    - Directory of JSON files: each file contains {"txn_id": {...}, ...}
    - Automatic transaction merging from multiple files
    """
    
    def __init__(self):
        self.feature_extractor = PaymentFeatureExtractor()
        self.repair_vocabulary = {}
        self.idx_to_repair = {}
    
    def load_and_process(self, path: str) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Load data from file/directory and extract features + labels.
        
        Args:
            path: Path to JSON file or directory
            
        Returns:
            (features, labels, payments)
        """
        logger.info(f"Loading data from {path}")
        
        # Get list of JSON files
        if os.path.isfile(path):
            json_files = [path]
            logger.info(f"Single file mode")
        elif os.path.isdir(path):
            json_files = glob.glob(os.path.join(path, '*.json'))
            logger.info(f"Directory mode: {len(json_files)} JSON files found")
        else:
            raise ValueError(f"Path does not exist: {path}")
        
        if len(json_files) == 0:
            raise ValueError(f"No JSON files found at {path}")
        
        # Load all transactions from all files
        all_transactions = {}
        for json_file in json_files:
            logger.info(f"  Loading {os.path.basename(json_file)}...")
            try:
                with open(json_file, 'r') as f:
                    file_data = json.load(f)
                
                if isinstance(file_data, dict):
                    all_transactions.update(file_data)
                    logger.info(f"    Added {len(file_data)} transactions")
                else:
                    logger.warning(f"    Skipping: unexpected format")
            except Exception as e:
                logger.warning(f"    Error: {e}")
                continue
        
        logger.info(f"Total transactions loaded: {len(all_transactions)}")
        
        # Build repair vocabulary
        self._build_repair_vocabulary(all_transactions)
        
        # Process transactions
        all_features = []
        all_labels = []
        all_payments = []
        
        skipped = 0
        processed = 0
        
        for txn_id, txn_data in all_transactions.items():
            try:
                # Check for 'after' state (required for training)
                if not self._has_after_state(txn_data):
                    skipped += 1
                    continue
                
                # Extract repairs
                repairs = [r['id'] for r in txn_data.get('ace', [])]
                
                # Extract 'before' state for features
                payment = self._extract_before_state(txn_data)
                
                # Extract features and labels
                features = self.feature_extractor.extract_features(payment)
                labels = self._repairs_to_labels(repairs)
                
                all_features.append(features)
                all_labels.append(labels)
                all_payments.append(txn_data)
                processed += 1
                
                # Progress logging
                if processed % 1000 == 0:
                    logger.info(f"  Processed {processed}/{len(all_transactions)}...")
                
            except Exception as e:
                logger.warning(f"Error processing {txn_id}: {e}")
                continue
        
        if len(all_features) == 0:
            raise ValueError("No trainable transactions found")
        
        features_array = np.array(all_features)
        labels_array = np.array(all_labels)
        
        # Summary
        logger.info(f"\n{'='*70}")
        logger.info(f"DATA LOADING SUMMARY")
        logger.info(f"{'='*70}")
        logger.info(f"Files processed: {len(json_files)}")
        logger.info(f"Total transactions: {len(all_transactions)}")
        logger.info(f"Trainable: {processed}")
        logger.info(f"Skipped (no 'after' state): {skipped}")
        logger.info(f"Feature shape: {features_array.shape}")
        logger.info(f"Label shape: {labels_array.shape}")
        logger.info(f"Unique repairs: {len(self.repair_vocabulary)}")
        logger.info(f"{'='*70}\n")
        
        return features_array, labels_array, all_payments
    
    def analyze_repair_distribution(self, path: str) -> Dict:
        """
        Analyze repair distribution in dataset.
        CRITICAL for understanding what model can/cannot learn.
        
        Returns:
            Dictionary with repair statistics
        """
        logger.info(f"Analyzing repair distribution in {path}")
        
        # Load files
        if os.path.isfile(path):
            json_files = [path]
        elif os.path.isdir(path):
            json_files = glob.glob(os.path.join(path, '*.json'))
        else:
            raise ValueError(f"Path does not exist: {path}")
        
        # Collect all transactions
        all_transactions = {}
        for json_file in json_files:
            with open(json_file, 'r') as f:
                file_data = json.load(f)
            if isinstance(file_data, dict):
                all_transactions.update(file_data)
        
        # Analyze repairs
        repair_counts = Counter()
        repair_co_occurrence = defaultdict(Counter)
        
        for txn_data in all_transactions.values():
            repairs = [r['id'] for r in txn_data.get('ace', [])]
            repair_counts.update(repairs)
            
            # Track co-occurrence
            for i, r1 in enumerate(repairs):
                for r2 in repairs[i+1:]:
                    repair_co_occurrence[r1][r2] += 1
                    repair_co_occurrence[r2][r1] += 1
        
        total_txns = len(all_transactions)
        
        # Print detailed report
        logger.info("\n" + "="*70)
        logger.info("REPAIR DISTRIBUTION ANALYSIS")
        logger.info("="*70)
        logger.info(f"Total transactions: {total_txns}")
        logger.info(f"Unique repair codes: {len(repair_counts)}")
        logger.info("\nRepair Frequency:")
        logger.info(f"{'Repair':<10} {'Count':<10} {'Percentage':<12} {'ML Feasibility'}")
        logger.info("-" * 70)
        
        for repair, count in repair_counts.most_common():
            percentage = count / total_txns * 100
            
            # Determine ML feasibility
            if count >= 100:
                feasibility = "✓ Excellent"
            elif count >= 50:
                feasibility = "✓ Good"
            elif count >= 20:
                feasibility = "⚠ Moderate"
            elif count >= 10:
                feasibility = "⚠ Difficult"
            else:
                feasibility = "✗ Too rare - need rules"
            
            logger.info(f"{repair:<10} {count:<10} {percentage:>10.1f}%  {feasibility}")
        
        # Identify rare repairs
        rare_repairs = [r for r, c in repair_counts.items() if c < 10]
        if rare_repairs:
            logger.info(f"\n⚠ WARNING: {len(rare_repairs)} repairs have < 10 examples:")
            logger.info(f"  {', '.join(rare_repairs)}")
            logger.info(f"  ML models CANNOT reliably learn these.")
            logger.info(f"  Options: (1) Get more data, (2) Create expert rules, (3) Accept missed predictions")
        
        # Common co-occurrences
        logger.info("\nTop Repair Co-occurrences:")
        all_pairs = []
        for r1, partners in repair_co_occurrence.items():
            for r2, count in partners.items():
                if r1 < r2:  # Avoid duplicates
                    all_pairs.append((r1, r2, count))
        
        for r1, r2, count in sorted(all_pairs, key=lambda x: x[2], reverse=True)[:10]:
            logger.info(f"  {r1} + {r2}: {count} times ({count/total_txns*100:.1f}%)")
        
        logger.info("="*70 + "\n")
        
        return {
            'total_transactions': total_txns,
            'repair_counts': dict(repair_counts),
            'rare_repairs': rare_repairs,
            'co_occurrence': {k: dict(v) for k, v in repair_co_occurrence.items()}
        }
    
    def _has_after_state(self, txn_data: Dict) -> bool:
        """Check if transaction has 'after' state"""
        entities = ['cdtr', 'dbtr', 'cdtrAgt', 'dbtrAgt', 'cdtrAcct', 'dbtrAcct',
                   'instgAgt', 'instdAgt', 'intrmyAgt1', 'intrmyAgt2', 'rmtInf']
        
        for entity in entities:
            for key in txn_data.keys():
                if key.lower() == entity.lower():
                    entity_data = txn_data[key]
                    if isinstance(entity_data, dict) and 'after' in entity_data:
                        return True
        return False
    
    def _extract_before_state(self, txn_data: Dict) -> Dict:
        """Extract 'before' state from transaction"""
        payment = {}
        
        # Copy metadata
        for key in ['source', 'clearing', 'flags', 'parties']:
            if key in txn_data:
                payment[key] = txn_data[key]
        
        # Extract 'before' from entities
        entities = ['cdtr', 'dbtr', 'cdtrAgt', 'dbtrAgt', 'cdtrAcct', 'dbtrAcct',
                   'instgAgt', 'instdAgt', 'intrmyAgt1', 'intrmyAgt2', 'rmtInf']
        
        for entity in entities:
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
        """Build mapping of repair IDs to indices"""
        all_repairs = set()
        
        for txn_data in data.values():
            for repair in txn_data.get('ace', []):
                all_repairs.add(repair['id'])
        
        sorted_repairs = sorted(all_repairs)
        
        self.repair_vocabulary = {repair_id: idx for idx, repair_id in enumerate(sorted_repairs)}
        self.idx_to_repair = {idx: repair_id for repair_id, idx in self.repair_vocabulary.items()}
        
        logger.info(f"Built vocabulary with {len(self.repair_vocabulary)} repairs")
    
    def _repairs_to_labels(self, repairs: List[str]) -> np.ndarray:
        """Convert repair IDs to multi-hot labels"""
        labels = np.zeros(len(self.repair_vocabulary))
        
        for repair_id in repairs:
            if repair_id in self.repair_vocabulary:
                idx = self.repair_vocabulary[repair_id]
                labels[idx] = 1.0
        
        return labels
    
    def labels_to_repairs(self, labels: np.ndarray, threshold: float = 0.5) -> List[str]:
        """Convert multi-hot labels to repair IDs"""
        repairs = []
        
        for idx, prob in enumerate(labels):
            if prob > threshold:
                repairs.append(self.idx_to_repair[idx])
        
        return repairs


# ============================================================================
# HYBRID PREDICTOR (RULES + XGBOOST + RANDOM FOREST)
# ============================================================================

class HybridPredictor:
    """
    Hybrid predictor optimized for small datasets using XGBoost.
    
    Architecture:
    1. Deterministic Rules (100% confidence) - applied first
    2. XGBoost (primary ML model) - gradient boosting
    3. Random Forest (ensemble diversity)
    4. Ensemble voting for final prediction
    
    Why XGBoost:
    - Excellent for tabular data
    - Handles small datasets well (3,000 samples)
    - Built-in regularization prevents overfitting
    - Fast training
    - Interpretable feature importance
    """
    
    def __init__(self, config: Config, analysis: Optional[Dict] = None):
        self.config = config
        self.feature_extractor = PaymentFeatureExtractor()
        self.rules = DeterministicRules(analysis)
        self.xgb_model = None
        self.rf_model = None
        self.processor = DataProcessor()
        self.analysis = analysis
        self.repair_support = {}  # Track support for each repair
    
    def train(self, train_file: str):
        """
        Train the hybrid model.
        
        Process:
        1. Load and analyze data
        2. Split train/val/test
        3. Train XGBoost (primary)
        4. Train Random Forest (diversity)
        5. Evaluate and save
        """
        logger.info("="*70)
        logger.info("TRAINING HYBRID PREDICTOR (XGBOOST + RULES)")
        logger.info("="*70)
        
        # Load data
        features, labels, payments = self.processor.load_and_process(train_file)
        
        n = len(features)
        
        # Analyze repair distribution
        logger.info("\nAnalyzing repair distribution...")
        self._analyze_repair_support(labels)
        
        # Split data
        if n < 10:
            logger.warning(f"Very small dataset ({n} samples) - using same data for train/val/test")
            X_train = X_val = X_test = features
            y_train = y_val = y_test = labels
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
        
        logger.info(f"\nData split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
        # Compute scale_pos_weight for each repair (handles class imbalance)
        scale_pos_weights = {}
        for idx in range(y_train.shape[1]):
            pos_count = y_train[:, idx].sum()
            neg_count = len(y_train) - pos_count
            if pos_count > 0:
                scale_pos_weights[idx] = neg_count / pos_count
            else:
                scale_pos_weights[idx] = 1.0
        
        # Train XGBoost (primary model)
        logger.info("\n" + "="*70)
        logger.info("TRAINING XGBOOST (Primary Model)")
        logger.info("="*70)
        
        xgb_estimators = []
        for idx in range(labels.shape[1]):
            repair_id = self.processor.idx_to_repair[idx]
            
            xgb = XGBClassifier(
                n_estimators=self.config.xgb_n_estimators,
                max_depth=self.config.xgb_max_depth,
                learning_rate=self.config.xgb_learning_rate,
                min_child_weight=self.config.xgb_min_child_weight,
                gamma=self.config.xgb_gamma,
                subsample=self.config.xgb_subsample,
                colsample_bytree=self.config.xgb_colsample_bytree,
                reg_alpha=self.config.xgb_reg_alpha,
                reg_lambda=self.config.xgb_reg_lambda,
                scale_pos_weight=scale_pos_weights[idx],  # Handle imbalance
                random_state=42,
                eval_metric='logloss',
                early_stopping_rounds=50,
                verbosity=0
            )
            
            # Train with validation set for early stopping
            if len(X_val) > 0:
                xgb.fit(
                    X_train, y_train[:, idx],
                    eval_set=[(X_val, y_val[:, idx])],
                    verbose=False
                )
            else:
                xgb.fit(X_train, y_train[:, idx])
            
            xgb_estimators.append(xgb)
            
            if (idx + 1) % 5 == 0:
                logger.info(f"  Trained {idx + 1}/{labels.shape[1]} XGBoost models...")
        
        self.xgb_model = xgb_estimators
        logger.info("✓ XGBoost training complete")
        
        # Evaluate XGBoost on validation set
        if len(X_val) > 0:
            xgb_val_pred = self._predict_ml(X_val, use_xgb=True, use_rf=False)
            xgb_val_pred_binary = (xgb_val_pred > 0.5).astype(int)
            xgb_exact_match = np.mean(np.all(xgb_val_pred_binary == y_val, axis=1))
            xgb_hamming = np.mean(xgb_val_pred_binary == y_val)
            
            logger.info(f"XGBoost validation - Exact Match: {xgb_exact_match:.1%}, Hamming: {xgb_hamming:.1%}")
        
        # Train Random Forest (ensemble diversity)
        logger.info("\n" + "="*70)
        logger.info("TRAINING RANDOM FOREST (Ensemble Diversity)")
        logger.info("="*70)
        
        self.rf_model = MultiOutputClassifier(
            RandomForestClassifier(
                n_estimators=self.config.rf_n_estimators,
                max_depth=self.config.rf_max_depth,
                min_samples_split=self.config.rf_min_samples_split,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1,
                verbose=0
            )
        )
        
        logger.info("Training Random Forest...")
        self.rf_model.fit(X_train, y_train)
        logger.info("✓ Random Forest training complete")
        
        # Evaluate Random Forest
        if len(X_val) > 0:
            rf_val_pred = self._predict_ml(X_val, use_xgb=False, use_rf=True)
            rf_val_pred_binary = (rf_val_pred > 0.5).astype(int)
            rf_exact_match = np.mean(np.all(rf_val_pred_binary == y_val, axis=1))
            rf_hamming = np.mean(rf_val_pred_binary == y_val)
            
            logger.info(f"Random Forest validation - Exact Match: {rf_exact_match:.1%}, Hamming: {rf_hamming:.1%}")
        
        # Evaluate ensemble
        if len(X_val) > 0:
            ensemble_val_pred = self._predict_ml(X_val, use_xgb=True, use_rf=True)
            ensemble_val_pred_binary = (ensemble_val_pred > 0.5).astype(int)
            ensemble_exact_match = np.mean(np.all(ensemble_val_pred_binary == y_val, axis=1))
            ensemble_hamming = np.mean(ensemble_val_pred_binary == y_val)
            
            logger.info(f"Ensemble validation - Exact Match: {ensemble_exact_match:.1%}, Hamming: {ensemble_hamming:.1%}")
        
        # Final test evaluation
        logger.info("\n" + "="*70)
        logger.info("FINAL TEST SET EVALUATION")
        logger.info("="*70)
        
        test_metrics = self.evaluate_detailed(X_test, y_test)
        
        # Save models
        self.save_models()
        
        return test_metrics
    
    def _analyze_repair_support(self, labels: np.ndarray):
        """Analyze and log repair support in training data"""
        logger.info("\nRepair Support in Training Data:")
        logger.info(f"{'Repair':<10} {'Count':<10} {'Percentage':<12} {'ML Feasibility'}")
        logger.info("-" * 70)
        
        for idx, repair_id in self.processor.idx_to_repair.items():
            count = int(labels[:, idx].sum())
            percentage = count / len(labels) * 100
            
            self.repair_support[repair_id] = count
            
            # Assess feasibility
            if count >= 100:
                feasibility = "✓ Excellent"
            elif count >= 50:
                feasibility = "✓ Good"
            elif count >= 20:
                feasibility = "⚠ Moderate"
            elif count >= 10:
                feasibility = "⚠ Difficult"
            else:
                feasibility = "✗ Too rare"
            
            logger.info(f"{repair_id:<10} {count:<10} {percentage:>10.1f}%  {feasibility}")
        
        # Warn about rare repairs
        rare_repairs = [r for r, c in self.repair_support.items() if c < 10]
        if rare_repairs:
            logger.info(f"\n⚠ WARNING: {len(rare_repairs)} repairs have < 10 training examples:")
            logger.info(f"  {', '.join(rare_repairs)}")
            logger.info(f"  These will be difficult for ML to learn. Consider adding expert rules.")
    
    def _predict_ml(self, X: np.ndarray, use_xgb: bool = True, use_rf: bool = True) -> np.ndarray:
        """
        Get ML predictions (XGBoost and/or Random Forest).
        
        Args:
            X: Features
            use_xgb: Use XGBoost predictions
            use_rf: Use Random Forest predictions
            
        Returns:
            Probability predictions
        """
        predictions = []
        
        if use_xgb and self.xgb_model is not None:
            xgb_probs = []
            for estimator in self.xgb_model:
                pred = estimator.predict_proba(X)
                if pred.shape[1] == 2:
                    xgb_probs.append(pred[:, 1])
                else:
                    xgb_probs.append(pred[:, 0])
            xgb_probs = np.column_stack(xgb_probs)
            predictions.append(xgb_probs)
        
        if use_rf and self.rf_model is not None:
            rf_probs = []
            for estimator in self.rf_model.estimators_:
                pred = estimator.predict_proba(X)
                if pred.shape[1] == 2:
                    rf_probs.append(pred[:, 1])
                else:
                    rf_probs.append(pred[:, 0])
            rf_probs = np.column_stack(rf_probs)
            predictions.append(rf_probs)
        
        if len(predictions) == 0:
            raise ValueError("No models available for prediction")
        
        # Ensemble: average predictions
        return np.mean(predictions, axis=0)
    
    def predict(self, payment: Dict, use_rules: bool = True, use_ml: bool = True) -> Dict:
        """
        Predict repairs for a payment.
        
        Process:
        1. Apply deterministic rules (100% confidence)
        2. Apply ML models with fixed threshold
        3. Filter by minimum confidence for rare repairs
        
        Args:
            payment: Payment transaction
            use_rules: Apply rules
            use_ml: Apply ML models
            
        Returns:
            Dictionary with repairs, confidences, sources
        """
        features = self.feature_extractor.extract_features(payment)
        
        repairs = []
        confidences = []
        sources = []
        
        # Apply rules first (100% confidence)
        if use_rules:
            rule_repairs, rule_confs = self.rules.predict_repairs(payment, features)
            repairs.extend(rule_repairs)
            confidences.extend(rule_confs)
            sources.extend(['rule'] * len(rule_repairs))
            
            logger.info(f"\nRule predictions: {len(rule_repairs)} repairs")
            for r, c in zip(rule_repairs, rule_confs):
                logger.info(f"  {r}: {c:.3f} (rule)")
        
        # Apply ML models
        if use_ml:
            ml_probs = self._predict_ml(features.reshape(1, -1))[0]
            
            logger.info(f"\nML predictions:")
            
            for idx, prob in enumerate(ml_probs):
                repair_id = self.processor.idx_to_repair[idx]
                
                # Skip if already predicted by rules
                if repair_id in repairs:
                    continue
                
                # Fixed threshold
                threshold = self.config.ml_threshold  # 0.5
                
                # Higher bar for rare repairs (avoid false positives)
                support = self.repair_support.get(repair_id, 0)
                if support < 10 and prob < self.config.min_confidence_rare_repairs:
                    # Rare repair needs >= 0.7 confidence
                    continue
                
                if prob > threshold:
                    repairs.append(repair_id)
                    confidences.append(float(prob))
                    sources.append('ml')
                    logger.info(f"  {repair_id}: {prob:.3f} (ml, support={support})")
        
        return {
            'repairs': repairs,
            'confidences': confidences,
            'sources': sources
        }
    
    def evaluate_detailed(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Evaluate with detailed per-repair metrics"""
        # Get ML predictions
        predictions = self._predict_ml(X)
        predictions_binary = (predictions > self.config.ml_threshold).astype(int)
        
        # Compute metrics
        metrics = PerRepairMetrics.compute_detailed_metrics(
            y, predictions_binary, self.processor.repair_vocabulary
        )
        
        PerRepairMetrics.print_detailed_report(metrics)
        self.rules.print_stats()
        
        return metrics
    
    def get_feature_importance(self, top_n: int = 20) -> Dict:
        """
        Get feature importance from XGBoost models.
        
        Args:
            top_n: Number of top features to return per repair
            
        Returns:
            Dictionary mapping repair_id to top features
        """
        if self.xgb_model is None:
            raise ValueError("Model not trained yet")
        
        importance_by_repair = {}
        
        for idx, estimator in enumerate(self.xgb_model):
            repair_id = self.processor.idx_to_repair[idx]
            
            # Get feature importance
            importance = estimator.feature_importances_
            
            # Get top N features
            top_indices = np.argsort(importance)[::-1][:top_n]
            top_features = [
                (self.feature_extractor.feature_names[i], importance[i])
                for i in top_indices
            ]
            
            importance_by_repair[repair_id] = top_features
        
        return importance_by_repair
    
    def print_feature_importance(self, repair_id: Optional[str] = None):
        """Print feature importance for a repair or all repairs"""
        importance = self.get_feature_importance()
        
        if repair_id:
            if repair_id not in importance:
                logger.error(f"Repair {repair_id} not found")
                return
            
            logger.info(f"\nTop Features for Repair {repair_id}:")
            logger.info("-" * 50)
            for feature, score in importance[repair_id]:
                logger.info(f"  {feature:<40} {score:.4f}")
        else:
            for rid, features in importance.items():
                logger.info(f"\nTop Features for Repair {rid}:")
                logger.info("-" * 50)
                for feature, score in features[:10]:  # Top 10 for each
                    logger.info(f"  {feature:<40} {score:.4f}")
    
    def save_models(self):
        """Save all models and metadata"""
        os.makedirs(self.config.model_dir, exist_ok=True)
        
        # Save XGBoost
        with open(os.path.join(self.config.model_dir, 'xgb_model.pkl'), 'wb') as f:
            pickle.dump(self.xgb_model, f)
        
        # Save Random Forest
        with open(os.path.join(self.config.model_dir, 'rf_model.pkl'), 'wb') as f:
            pickle.dump(self.rf_model, f)
        
        # Save processor
        with open(os.path.join(self.config.model_dir, 'processor.pkl'), 'wb') as f:
            pickle.dump(self.processor, f)
        
        # Save repair support
        with open(os.path.join(self.config.model_dir, 'repair_support.pkl'), 'wb') as f:
            pickle.dump(self.repair_support, f)
        
        # Save analysis
        if self.analysis:
            with open(os.path.join(self.config.model_dir, 'analysis.json'), 'w') as f:
                json.dump(self.analysis, f, indent=2)
        
        # Save config
        self.config.save(os.path.join(self.config.model_dir, 'config.json'))
        
        logger.info(f"\nModels saved to {self.config.model_dir}")
    
    def load_models(self, model_dir: str):
        """Load saved models"""
        # Load config
        self.config = Config.load(os.path.join(model_dir, 'config.json'))
        
        # Load processor
        with open(os.path.join(model_dir, 'processor.pkl'), 'rb') as f:
            self.processor = pickle.load(f)
        
        # Load XGBoost
        with open(os.path.join(model_dir, 'xgb_model.pkl'), 'rb') as f:
            self.xgb_model = pickle.load(f)
        
        # Load Random Forest
        with open(os.path.join(model_dir, 'rf_model.pkl'), 'rb') as f:
            self.rf_model = pickle.load(f)
        
        # Load repair support
        support_path = os.path.join(model_dir, 'repair_support.pkl')
        if os.path.exists(support_path):
            with open(support_path, 'rb') as f:
                self.repair_support = pickle.load(f)
        
        # Load analysis
        analysis_path = os.path.join(model_dir, 'analysis.json')
        if os.path.exists(analysis_path):
            with open(analysis_path, 'r') as f:
                self.analysis = json.load(f)
                self.rules = DeterministicRules(self.analysis)
        
        logger.info(f"Models loaded from {model_dir}")


# ============================================================================
# CLI COMMANDS
# ============================================================================

def train_command(args):
    """Train the model"""
    config = Config()
    config.xgb_n_estimators = args.estimators
    
    # Load analysis if provided
    analysis = None
    if args.analysis:
        with open(args.analysis, 'r') as f:
            analysis = json.load(f)
    
    predictor = HybridPredictor(config, analysis)
    metrics = predictor.train(args.input)
    
    logger.info("\n" + "="*70)
    logger.info("TRAINING COMPLETE")
    logger.info("="*70)
    logger.info(f"Test Exact Match: {metrics['overall']['exact_match_accuracy']:.1%}")
    logger.info(f"Test Hamming: {metrics['overall']['hamming_accuracy']:.1%}")
    
    if metrics['overall']['exact_match_accuracy'] >= 0.90:
        logger.info("🎯 GOAL ACHIEVED: 90%+ exact match!")
    else:
        gap = 0.90 - metrics['overall']['exact_match_accuracy']
        logger.info(f"Gap to 90%: {gap:.1%}")
    
    # Print feature importance for top repairs
    if args.feature_importance:
        logger.info("\n" + "="*70)
        logger.info("FEATURE IMPORTANCE")
        logger.info("="*70)
        predictor.print_feature_importance()


def analyze_command(args):
    """Analyze repair distribution"""
    processor = DataProcessor()
    analysis = processor.analyze_repair_distribution(args.input)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(analysis, f, indent=2)
        logger.info(f"Analysis saved to {args.output}")


def predict_command(args):
    """Predict repairs"""
    predictor = HybridPredictor(Config())
    predictor.load_models(args.model or './models')
    
    # Load payment
    with open(args.input, 'r') as f:
        data = json.load(f)
    
    # Handle format
    if isinstance(data, dict) and len(data) == 1:
        txn_id = list(data.keys())[0]
        payment_data = data[txn_id]
        
        if predictor.processor._has_after_state(payment_data):
            payment = predictor.processor._extract_before_state(payment_data)
        else:
            payment = payment_data
    else:
        payment = data
    
    # Predict
    result = predictor.predict(payment)
    
    # Display
    logger.info("\n" + "="*70)
    logger.info("REPAIR PREDICTIONS")
    logger.info("="*70)
    logger.info(f"Predicted {len(result['repairs'])} repairs:")
    
    for repair, conf, source in zip(result['repairs'], result['confidences'], result['sources']):
        logger.info(f"  {repair}: {conf:.3f} ({source})")
    
    # Save
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"\nResults saved to {args.output}")


def evaluate_command(args):
    """Evaluate model"""
    predictor = HybridPredictor(Config())
    predictor.load_models(args.model or './models')
    
    features, labels, payments = predictor.processor.load_and_process(args.input)
    
    if args.detailed:
        predictor.evaluate_detailed(features, labels)
    else:
        predictions = predictor._predict_ml(features)
        predictions_binary = (predictions > 0.5).astype(int)
        
        exact_match = np.mean(np.all(predictions_binary == labels, axis=1))
        hamming = np.mean(predictions_binary == labels)
        
        logger.info("\n" + "="*70)
        logger.info("EVALUATION RESULTS")
        logger.info("="*70)
        logger.info(f"Exact Match: {exact_match:.1%}")
        logger.info(f"Hamming: {hamming:.1%}")


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description='ACE Payment Repair Predictor (XGBoost + Rules)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze your data FIRST (CRITICAL!)
  python stp_lite_repair.py analyze --input ./data
  
  # Train on directory
  python stp_lite_repair.py train --input ./data --estimators 500
  
  # Train with feature importance
  python stp_lite_repair.py train --input ./data --feature-importance
  
  # Predict for payment
  python stp_lite_repair.py predict --input payment.json
  
  # Evaluate with details
  python stp_lite_repair.py evaluate --input ./test_data --detailed
        """
    )
    
    subparsers = parser.add_subparsers(dest='command')
    
    # TRAIN
    train_parser = subparsers.add_parser('train', help='Train model')
    train_parser.add_argument('--input', required=True, help='Training data path')
    train_parser.add_argument('--analysis', help='Analysis JSON (optional)')
    train_parser.add_argument('--estimators', type=int, default=500, help='XGBoost estimators')
    train_parser.add_argument('--feature-importance', action='store_true', help='Show feature importance')
    
    # ANALYZE
    analyze_parser = subparsers.add_parser('analyze', help='Analyze repair distribution')
    analyze_parser.add_argument('--input', required=True, help='Data path')
    analyze_parser.add_argument('--output', help='Save analysis JSON')
    
    # PREDICT
    predict_parser = subparsers.add_parser('predict', help='Predict repairs')
    predict_parser.add_argument('--input', required=True, help='Payment JSON')
    predict_parser.add_argument('--output', help='Save results')
    predict_parser.add_argument('--model', default='./models', help='Model directory')
    
    # EVALUATE
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model')
    eval_parser.add_argument('--input', required=True, help='Test data path')
    eval_parser.add_argument('--model', default='./models', help='Model directory')
    eval_parser.add_argument('--detailed', action='store_true', help='Detailed metrics')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_command(args)
    elif args.command == 'analyze':
        analyze_command(args)
    elif args.command == 'predict':
        predict_command(args)
    elif args.command == 'evaluate':
        evaluate_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

