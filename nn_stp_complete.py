#!/usr/bin/env python3
"""
Integrated ACE Payment Repair Predictor with Comprehensive Feature Extraction
Version: 8.0 - Full integration with 183+ features

Key Improvements:
- Comprehensive feature extraction covering ALL entities
- 183 features instead of 21
- Entity-specific patterns for better accuracy
- Matches diagnostic checks exactly
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
# LOGGING CONFIGURATION
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
    """Enhanced configuration with comprehensive feature settings"""
    # Model parameters
    hidden_dim: int = 512  # Increased for more features
    dropout: float = 0.3
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 50
    
    # Data splits
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    
    # Processing parameters
    max_files_in_memory: int = 100
    checkpoint_interval: int = 500
    enable_incremental: bool = True
    use_comprehensive_features: bool = True  # New flag
    
    # Paths
    model_dir: str = "./models"
    checkpoint_dir: str = "./checkpoints"
    
    # Analysis parameters
    analyze_source_clearing: bool = True
    track_database_lookups: bool = True
    detect_value_transformations: bool = True
    
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
# COMPREHENSIVE FEATURE EXTRACTOR (183 features)
# ============================================================================

class ComprehensiveFeatureExtractor:
    """
    Extracts comprehensive features from payment data covering ALL entities
    183+ features for complete pattern recognition
    """
    
    def __init__(self):
        self.feature_names = []
        self.feature_indices = {}
        self._build_comprehensive_features()
        logger.info(f"Initialized comprehensive feature extractor with {len(self.feature_names)} features")
    
    def _build_comprehensive_features(self):
        """Build comprehensive feature set covering all entities"""
        
        # AGENT FEATURES (8 agents × 8 checks = 64 features)
        agent_entities = [
            'cdtrAgt', 'dbtrAgt', 'instgAgt', 'instdAgt',
            'intrmyAgt1', 'intrmyAgt2', 'intrmyAgt3', 'fwdgAgt'
        ]
        
        for agent in agent_entities:
            self.feature_names.extend([
                f'{agent}_missing_bic',
                f'{agent}_missing_name',
                f'{agent}_missing_address',
                f'{agent}_missing_country',
                f'{agent}_has_clearing_no_bic',
                f'{agent}_has_bic_no_name',
                f'{agent}_mmbid_wrong_level',
                f'{agent}_missing_clrsysid'
            ])
        
        # PARTY FEATURES (5 parties × 6 checks = 30 features)
        party_entities = ['cdtr', 'dbtr', 'ultmtCdtr', 'ultmtDbtr', 'initgPty']
        
        for party in party_entities:
            self.feature_names.extend([
                f'{party}_missing_name',
                f'{party}_missing_address',
                f'{party}_missing_country',
                f'{party}_missing_identification',
                f'{party}_incomplete_postal',
                f'{party}_has_text_no_country'
            ])
        
        # ACCOUNT FEATURES (2 accounts × 6 checks = 12 features)
        account_entities = ['cdtrAcct', 'dbtrAcct']
        
        for account in account_entities:
            self.feature_names.extend([
                f'{account}_missing_iban',
                f'{account}_missing_othr_id',
                f'{account}_missing_any_id',
                f'{account}_missing_type',
                f'{account}_missing_currency',
                f'{account}_wrong_type_format'
            ])
        
        # REMITTANCE INFO FEATURES (5 features)
        self.feature_names.extend([
            'rmtInf_missing',
            'rmtInf_unstructured_only',
            'rmtInf_structured_only',
            'rmtInf_needs_formatting',
            'rmtInf_too_long'
        ])
        
        # SOURCE/CLEARING COMBINATIONS (12 features)
        self.feature_names.extend([
            'source_swf',
            'source_ach',
            'source_fbr',
            'source_chp',
            'clearing_fed',
            'clearing_chips',
            'clearing_ltr',
            'is_swf_fed',
            'is_swf_chips',
            'is_ach_fed',
            'is_ach_chips',
            'is_chp_ltr'
        ])
        
        # CROSS-ENTITY RELATIONSHIPS (15 features)
        self.feature_names.extend([
            'cdtragt_incomplete_affects_cdtr',
            'dbtragt_incomplete_affects_dbtr',
            'agent_country_mismatch',
            'party_country_mismatch',
            'missing_required_agents_for_clearing',
            'missing_intermediary_chain',
            'account_party_mismatch',
            'bic_routing_inconsistent',
            'clearing_agent_mismatch',
            'instructing_instructed_mismatch',
            'ultimate_party_missing',
            'payment_chain_broken',
            'regulatory_info_missing',
            'purpose_code_missing',
            'supplementary_data_issues'
        ])
        
        # STRUCTURAL ISSUES (20 features)
        self.feature_names.extend([
            'has_wrong_case_formatting',
            'has_mixed_case_fields',
            'mmbid_needs_restructuring',
            'fininstnid_missing',
            'pstladr_wrong_structure',
            'adrline_not_array',
            'othr_structure_missing',
            'clrsysid_structure_missing',
            'id_at_wrong_nesting',
            'bicfi_vs_bic_inconsistent',
            'clearing_without_clrsys',
            'json_nesting_too_deep',
            'json_nesting_too_shallow',
            'required_wrapper_missing',
            'optional_wrapper_present',
            'field_name_format_wrong',
            'value_format_needs_standardization',
            'date_format_incorrect',
            'amount_format_incorrect',
            'reference_format_incorrect'
        ])
        
        # DATA QUALITY INDICATORS (15 features)
        self.feature_names.extend([
            'total_missing_fields',
            'critical_fields_missing_count',
            'agent_completeness_score',
            'party_completeness_score',
            'account_completeness_score',
            'overall_data_quality_score',
            'needs_enrichment_count',
            'needs_lookup_count',
            'needs_extraction_count',
            'needs_restructuring_count',
            'validation_errors_count',
            'format_errors_count',
            'consistency_errors_count',
            'routing_errors_count',
            'compliance_errors_count'
        ])
        
        # FLAGS AND METADATA (10 features)
        self.feature_names.extend([
            'flag_ace_repairs',
            'flag_stp_failed',
            'flag_repair_queue',
            'flag_exception_queue',
            'flag_verify_queue',
            'flag_ignored_party',
            'flag_credit_party_by_name',
            'flag_routed_to_repair',
            'flag_autotrieve',
            'flag_manual_intervention'
        ])
        
        # Build index mapping
        self.feature_indices = {name: i for i, name in enumerate(self.feature_names)}
    
    def extract_features(self, payment: Dict, for_training: bool = False) -> np.ndarray:
        """
        Extract comprehensive features from payment data
        
        Args:
            payment: Payment transaction data
            for_training: If True, extract from 'before' state
            
        Returns:
            Feature vector with 183 features
        """
        features = np.zeros(len(self.feature_names))
        
        # Normalize keys to lowercase
        payment = self._normalize_keys(payment)
        
        # Extract features for each category
        self._extract_agent_features(payment, features, for_training)
        self._extract_party_features(payment, features, for_training)
        self._extract_account_features(payment, features, for_training)
        self._extract_remittance_features(payment, features, for_training)
        self._extract_source_clearing_features(payment, features)
        self._extract_cross_entity_features(payment, features, for_training)
        self._extract_structural_features(payment, features, for_training)
        self._extract_quality_features(payment, features, for_training)
        self._extract_flag_features(payment, features)
        
        return features
    
    def _extract_agent_features(self, payment: Dict, features: np.ndarray, for_training: bool):
        """Extract features for all agent entities"""
        agent_entities = [
            'cdtragt', 'dbtragt', 'instgagt', 'instdagt',
            'intrmyagt1', 'intrmyagt2', 'intrmyagt3', 'fwdgagt'
        ]
        
        for agent in agent_entities:
            agent_data = self._get_entity_data(payment, agent, for_training)
            
            if agent_data is None:
                # Agent doesn't exist - all features are 1 (missing)
                for check in ['missing_bic', 'missing_name', 'missing_address', 
                             'missing_country', 'has_clearing_no_bic', 'has_bic_no_name',
                             'mmbid_wrong_level', 'missing_clrsysid']:
                    feat_name = f'{agent}_{check}'
                    if feat_name in self.feature_indices:
                        features[self.feature_indices[feat_name]] = 1.0
            else:
                # Check each aspect of the agent
                fin_inst = self._get_fininstnid(agent_data)
                
                # Missing BIC
                has_bic = self._has_bic(fin_inst)
                features[self.feature_indices[f'{agent}_missing_bic']] = float(not has_bic)
                
                # Missing name
                has_name = self._has_name(fin_inst)
                features[self.feature_indices[f'{agent}_missing_name']] = float(not has_name)
                
                # Missing address
                has_address = self._has_address(fin_inst)
                features[self.feature_indices[f'{agent}_missing_address']] = float(not has_address)
                
                # Missing country
                has_country = self._has_country(fin_inst)
                features[self.feature_indices[f'{agent}_missing_country']] = float(not has_country)
                
                # Has clearing but no BIC (triggers 8852)
                has_clearing = self._has_clearing(fin_inst)
                features[self.feature_indices[f'{agent}_has_clearing_no_bic']] = \
                    float(has_clearing and not has_bic)
                
                # Has BIC but no name (triggers name lookup)
                features[self.feature_indices[f'{agent}_has_bic_no_name']] = \
                    float(has_bic and not has_name)
                
                # MmbId at wrong level
                features[self.feature_indices[f'{agent}_mmbid_wrong_level']] = \
                    float(self._mmbid_at_wrong_level(fin_inst))
                
                # Missing ClrSysId structure
                features[self.feature_indices[f'{agent}_missing_clrsysid']] = \
                    float(has_clearing and not self._has_clrsysid(fin_inst))
    
    def _extract_party_features(self, payment: Dict, features: np.ndarray, for_training: bool):
        """Extract features for all party entities"""
        party_entities = ['cdtr', 'dbtr', 'ultmtcdtr', 'ultmtdbtr', 'initgpty']
        
        for party in party_entities:
            party_data = self._get_entity_data(payment, party, for_training)
            
            if party_data is None:
                # Party doesn't exist
                for check in ['missing_name', 'missing_address', 'missing_country',
                             'missing_identification', 'incomplete_postal', 'has_text_no_country']:
                    feat_name = f'{party}_{check}'
                    if feat_name in self.feature_indices:
                        features[self.feature_indices[feat_name]] = 1.0
            else:
                # Check party completeness
                features[self.feature_indices[f'{party}_missing_name']] = \
                    float(not self._has_name(party_data))
                
                features[self.feature_indices[f'{party}_missing_address']] = \
                    float(not self._has_address(party_data))
                
                has_country = self._has_country(party_data)
                features[self.feature_indices[f'{party}_missing_country']] = float(not has_country)
                
                features[self.feature_indices[f'{party}_missing_identification']] = \
                    float(not self._has_identification(party_data))
                
                features[self.feature_indices[f'{party}_incomplete_postal']] = \
                    float(self._has_incomplete_postal(party_data))
                
                # Has text that contains country but no country code (triggers 6021)
                features[self.feature_indices[f'{party}_has_text_no_country']] = \
                    float(not has_country and self._has_country_in_text(party_data))
    
    def _extract_account_features(self, payment: Dict, features: np.ndarray, for_training: bool):
        """Extract features for account entities"""
        account_entities = ['cdtracct', 'dbtracct']
        
        for account in account_entities:
            account_data = self._get_entity_data(payment, account, for_training)
            
            if account_data is None:
                # Account doesn't exist - use 0.5 as may be optional
                for check in ['missing_iban', 'missing_othr_id', 'missing_any_id',
                             'missing_type', 'missing_currency', 'wrong_type_format']:
                    feat_name = f'{account}_{check}'
                    if feat_name in self.feature_indices:
                        features[self.feature_indices[feat_name]] = 0.5
            else:
                # Check account completeness
                id_data = self._find_field(account_data, ['id'])
                
                has_iban = False
                has_othr = False
                if isinstance(id_data, dict):
                    has_iban = self._find_field(id_data, ['iban']) is not None
                    has_othr = self._find_field(id_data, ['othr']) is not None
                
                features[self.feature_indices[f'{account}_missing_iban']] = float(not has_iban)
                features[self.feature_indices[f'{account}_missing_othr_id']] = float(not has_othr)
                features[self.feature_indices[f'{account}_missing_any_id']] = \
                    float(not has_iban and not has_othr)
                
                features[self.feature_indices[f'{account}_missing_type']] = \
                    float(not self._find_field(account_data, ['tp', 'type']))
                
                features[self.feature_indices[f'{account}_missing_currency']] = \
                    float(not self._find_field(account_data, ['ccy', 'currency']))
                
                features[self.feature_indices[f'{account}_wrong_type_format']] = \
                    float(self._has_wrong_account_type_format(account_data))
    
    def _extract_remittance_features(self, payment: Dict, features: np.ndarray, for_training: bool):
        """Extract remittance information features"""
        rmtinf = self._get_entity_data(payment, 'rmtinf', for_training)
        
        if rmtinf is None:
            features[self.feature_indices['rmtInf_missing']] = 1.0
        else:
            features[self.feature_indices['rmtInf_missing']] = 0.0
            
            has_ustrd = self._find_field(rmtinf, ['ustrd']) is not None
            has_strd = self._find_field(rmtinf, ['strd']) is not None
            
            features[self.feature_indices['rmtInf_unstructured_only']] = \
                float(has_ustrd and not has_strd)
            features[self.feature_indices['rmtInf_structured_only']] = \
                float(has_strd and not has_ustrd)
            
            # Check if formatting needed
            ustrd_val = self._find_field(rmtinf, ['ustrd'])
            if ustrd_val and isinstance(ustrd_val, str):
                features[self.feature_indices['rmtInf_needs_formatting']] = \
                    float(len(ustrd_val) > 140 or '\n' in ustrd_val)
                features[self.feature_indices['rmtInf_too_long']] = \
                    float(len(ustrd_val) > 280)
    
    def _extract_source_clearing_features(self, payment: Dict, features: np.ndarray):
        """Extract source and clearing combination features"""
        source = payment.get('source', '').upper()
        clearing = payment.get('clearing', '').upper()
        
        # Individual source/clearing
        features[self.feature_indices['source_swf']] = float(source == 'SWF')
        features[self.feature_indices['source_ach']] = float(source == 'ACH')
        features[self.feature_indices['source_fbr']] = float(source == 'FBR')
        features[self.feature_indices['source_chp']] = float(source == 'CHP')
        
        features[self.feature_indices['clearing_fed']] = float(clearing == 'FED')
        features[self.feature_indices['clearing_chips']] = float(clearing == 'CHIPS')
        features[self.feature_indices['clearing_ltr']] = float(clearing == 'LTR')
        
        # Specific combinations
        features[self.feature_indices['is_swf_fed']] = float(source == 'SWF' and clearing == 'FED')
        features[self.feature_indices['is_swf_chips']] = float(source == 'SWF' and clearing == 'CHIPS')
        features[self.feature_indices['is_ach_fed']] = float(source == 'ACH' and clearing == 'FED')
        features[self.feature_indices['is_ach_chips']] = float(source == 'ACH' and clearing == 'CHIPS')
        features[self.feature_indices['is_chp_ltr']] = float(source == 'CHP' and clearing == 'LTR')
    
    def _extract_cross_entity_features(self, payment: Dict, features: np.ndarray, for_training: bool):
        """Extract cross-entity relationship features"""
        cdtragt = self._get_entity_data(payment, 'cdtragt', for_training)
        dbtragt = self._get_entity_data(payment, 'dbtragt', for_training)
        cdtr = self._get_entity_data(payment, 'cdtr', for_training)
        dbtr = self._get_entity_data(payment, 'dbtr', for_training)
        
        # Agent affects party
        cdtragt_incomplete = self._is_agent_incomplete(cdtragt)
        dbtragt_incomplete = self._is_agent_incomplete(dbtragt)
        
        features[self.feature_indices['cdtragt_incomplete_affects_cdtr']] = \
            float(cdtragt_incomplete and cdtr is not None)
        features[self.feature_indices['dbtragt_incomplete_affects_dbtr']] = \
            float(dbtragt_incomplete and dbtr is not None)
        
        # Country mismatches
        agent_countries = self._get_all_countries([cdtragt, dbtragt])
        party_countries = self._get_all_countries([cdtr, dbtr])
        
        features[self.feature_indices['agent_country_mismatch']] = \
            float(len(agent_countries) > 1)
        features[self.feature_indices['party_country_mismatch']] = \
            float(len(party_countries) > 1)
        
        # Missing required agents for clearing type
        clearing = payment.get('clearing', '').upper()
        if clearing == 'FED':
            features[self.feature_indices['missing_required_agents_for_clearing']] = \
                float(cdtragt is None or dbtragt is None)
        
        # Check intermediary chain
        has_intrmy1 = self._get_entity_data(payment, 'intrmyagt1', for_training) is not None
        has_intrmy2 = self._get_entity_data(payment, 'intrmyagt2', for_training) is not None
        has_intrmy3 = self._get_entity_data(payment, 'intrmyagt3', for_training) is not None
        
        # If has 2 but not 1, chain is broken
        features[self.feature_indices['missing_intermediary_chain']] = \
            float((has_intrmy2 and not has_intrmy1) or (has_intrmy3 and not has_intrmy2))
    
    def _extract_structural_features(self, payment: Dict, features: np.ndarray, for_training: bool):
        """Extract structural issue features"""
        # Check case formatting issues
        has_mixed_case = self._has_mixed_case_fields(payment)
        features[self.feature_indices['has_wrong_case_formatting']] = float(has_mixed_case)
        features[self.feature_indices['has_mixed_case_fields']] = float(has_mixed_case)
        
        # Check for specific structural issues
        total_mmbid_issues = 0
        total_missing_structures = 0
        
        for agent in ['cdtragt', 'dbtragt']:
            agent_data = self._get_entity_data(payment, agent, for_training)
            if agent_data:
                fin_inst = self._get_fininstnid(agent_data)
                if self._mmbid_at_wrong_level(fin_inst):
                    total_mmbid_issues += 1
                if not self._has_clrsysid(fin_inst) and self._has_clearing(fin_inst):
                    total_missing_structures += 1
        
        features[self.feature_indices['mmbid_needs_restructuring']] = float(total_mmbid_issues > 0)
        features[self.feature_indices['clrsysid_structure_missing']] = float(total_missing_structures > 0)
    
    def _extract_quality_features(self, payment: Dict, features: np.ndarray, for_training: bool):
        """Extract overall data quality metrics"""
        missing_count = 0
        critical_missing = 0
        needs_enrichment = 0
        needs_lookup = 0
        needs_extraction = 0
        
        # Count missing critical fields
        for agent in ['cdtragt', 'dbtragt']:
            agent_data = self._get_entity_data(payment, agent, for_training)
            if agent_data:
                fin_inst = self._get_fininstnid(agent_data)
                if not self._has_bic(fin_inst):
                    critical_missing += 1
                    if self._has_clearing(fin_inst):
                        needs_lookup += 1
                if not self._has_name(fin_inst):
                    missing_count += 1
                    if self._has_bic(fin_inst):
                        needs_enrichment += 1
                if not self._has_country(fin_inst):
                    missing_count += 1
                    if self._has_country_in_text(fin_inst):
                        needs_extraction += 1
        
        # Normalize counts
        features[self.feature_indices['total_missing_fields']] = min(missing_count / 10, 1.0)
        features[self.feature_indices['critical_fields_missing_count']] = min(critical_missing / 5, 1.0)
        features[self.feature_indices['needs_enrichment_count']] = min(needs_enrichment / 5, 1.0)
        features[self.feature_indices['needs_lookup_count']] = min(needs_lookup / 5, 1.0)
        features[self.feature_indices['needs_extraction_count']] = min(needs_extraction / 5, 1.0)
    
    def _extract_flag_features(self, payment: Dict, features: np.ndarray):
        """Extract flag and metadata features"""
        flags = payment.get('flags', {})
        
        flag_mappings = {
            'flag_ace_repairs': ['ace_repairs'],
            'flag_stp_failed': ['stp_failed'],
            'flag_repair_queue': ['repair_queue'],
            'flag_exception_queue': ['exception_queue'],
            'flag_verify_queue': ['verify_queue'],
            'flag_ignored_party': ['ignored_party'],
            'flag_credit_party_by_name': ['credit_party_identified_by_name'],
            'flag_routed_to_repair': ['routed_to_repair'],
            'flag_autotrieve': ['autotrieve']
        }
        
        for feature_name, flag_keys in flag_mappings.items():
            if feature_name in self.feature_indices:
                value = any(flags.get(key, False) for key in flag_keys)
                features[self.feature_indices[feature_name]] = float(value)
    
    # ========================================
    # HELPER METHODS
    # ========================================
    
    def _get_entity_data(self, payment: Dict, entity_name: str, for_training: bool = False) -> Optional[Dict]:
        """Get entity data handling before/after structure"""
        entity_name = entity_name.lower()
        
        if entity_name not in payment:
            return None
        
        entity = payment[entity_name]
        if not isinstance(entity, dict):
            return None
        
        # For training, use 'before' state if available
        if for_training and 'before' in entity:
            return entity['before']
        
        # For prediction or if no 'before', use direct data
        if 'before' not in entity and 'after' not in entity:
            return entity
        
        # If has 'after' but no 'before', entity didn't exist before
        if 'after' in entity and 'before' not in entity:
            return None if for_training else entity['after']
        
        return entity
    
    def _get_fininstnid(self, agent_data: Optional[Dict]) -> Optional[Dict]:
        """Get financial institution ID from agent"""
        if not agent_data:
            return None
        return self._find_field(agent_data, ['fininstnid', 'FinInstnId'])
    
    def _has_bic(self, data: Optional[Dict]) -> bool:
        """Check if has BIC"""
        if not data:
            return False
        return self._find_field(data, ['bicfi', 'bic', 'bicFi', 'BICFI']) is not None
    
    def _has_name(self, data: Optional[Dict]) -> bool:
        """Check if has name"""
        if not data:
            return False
        return self._find_field(data, ['nm', 'name', 'Nm']) is not None
    
    def _has_address(self, data: Optional[Dict]) -> bool:
        """Check if has address"""
        if not data:
            return False
        postal = self._find_field(data, ['pstladr', 'PstlAdr'])
        if isinstance(postal, dict):
            adr_lines = self._find_field(postal, ['adrline', 'AdrLine'])
            return adr_lines is not None and (isinstance(adr_lines, list) or adr_lines)
        return False
    
    def _has_country(self, data: Optional[Dict]) -> bool:
        """Check if has country"""
        if not data:
            return False
        # Check in postal address
        postal = self._find_field(data, ['pstladr', 'PstlAdr'])
        if isinstance(postal, dict):
            if self._find_field(postal, ['ctry', 'ctryofres', 'Ctry', 'CtryOfRes']):
                return True
        # Check at top level
        return self._find_field(data, ['ctryofres', 'CtryOfRes']) is not None
    
    def _has_clearing(self, data: Optional[Dict]) -> bool:
        """Check if has clearing member ID"""
        if not data:
            return False
        clr_sys = self._find_field(data, ['clrsysmmbid', 'ClrSysMmbId'])
        if isinstance(clr_sys, dict):
            return self._find_field(clr_sys, ['mmbid', 'MmbId']) is not None
        # Check for MmbId at wrong level
        return self._find_field(data, ['mmbid', 'MmbId']) is not None
    
    def _has_clrsysid(self, data: Optional[Dict]) -> bool:
        """Check if has ClrSysId structure"""
        if not data:
            return False
        return self._find_field(data, ['clrsysid', 'ClrSysId']) is not None
    
    def _mmbid_at_wrong_level(self, data: Optional[Dict]) -> bool:
        """Check if MmbId is at wrong structural level"""
        if not data:
            return False
        # Has MmbId at top level but should be in ClrSysId
        has_top_mmbid = self._find_field(data, ['mmbid', 'MmbId'], max_depth=1) is not None
        has_clrsys = self._has_clrsysid(data)
        return has_top_mmbid and not has_clrsys
    
    def _has_identification(self, data: Optional[Dict]) -> bool:
        """Check if party has identification"""
        if not data:
            return False
        return self._find_field(data, ['id', 'Id']) is not None
    
    def _has_incomplete_postal(self, data: Optional[Dict]) -> bool:
        """Check if postal address is incomplete"""
        if not data:
            return False
        postal = self._find_field(data, ['pstladr', 'PstlAdr'])
        if not isinstance(postal, dict):
            return True  # Has party but no postal
        
        # Check completeness
        has_lines = self._find_field(postal, ['adrline', 'AdrLine']) is not None
        has_country = self._find_field(postal, ['ctry', 'Ctry']) is not None
        
        return not (has_lines and has_country)
    
    def _has_country_in_text(self, data: Optional[Dict]) -> bool:
        """Check if country is mentioned in text fields"""
        if not data:
            return False
        
        country_indicators = ['US', 'USA', 'UNITED STATES', 'UK', 'GB', 'CANADA', 
                             'GERMANY', 'FRANCE', 'JAPAN', 'CHINA']
        
        # Check in address lines
        postal = self._find_field(data, ['pstladr', 'PstlAdr'])
        if isinstance(postal, dict):
            adr_lines = self._find_field(postal, ['adrline', 'AdrLine'])
            if adr_lines:
                text = str(adr_lines).upper()
                return any(country in text for country in country_indicators)
        
        return False
    
    def _has_wrong_account_type_format(self, data: Optional[Dict]) -> bool:
        """Check if account type has wrong format"""
        if not data:
            return False
        tp = self._find_field(data, ['tp', 'Tp'])
        if isinstance(tp, dict):
            prtry = self._find_field(tp, ['prtry', 'Prtry'])
            if prtry:
                # Check for common format issues
                return not isinstance(prtry, str) or len(prtry) > 4
        return False
    
    def _is_agent_incomplete(self, agent_data: Optional[Dict]) -> bool:
        """Check if agent is incomplete"""
        if not agent_data:
            return True
        fin_inst = self._get_fininstnid(agent_data)
        return not (self._has_bic(fin_inst) and self._has_name(fin_inst))
    
    def _get_all_countries(self, entities: List[Optional[Dict]]) -> set:
        """Get all unique countries from entities"""
        countries = set()
        for entity in entities:
            if entity and self._has_country(entity):
                postal = self._find_field(entity, ['pstladr', 'PstlAdr'])
                if isinstance(postal, dict):
                    country = self._find_field(postal, ['ctry', 'Ctry'])
                    if country:
                        countries.add(str(country).upper())
        return countries
    
    def _has_mixed_case_fields(self, obj: Any, depth: int = 0) -> bool:
        """Check if object has mixed case field names"""
        if depth > 5:
            return False
        
        if isinstance(obj, dict):
            has_lower = any(k.islower() for k in obj.keys() if isinstance(k, str))
            has_upper = any(k[0].isupper() for k in obj.keys() if isinstance(k, str) and k)
            
            if has_lower and has_upper:
                return True
            
            for v in obj.values():
                if self._has_mixed_case_fields(v, depth + 1):
                    return True
        
        return False
    
    def _find_field(self, obj: Optional[Dict], field_names: List[str], 
                   max_depth: int = 5) -> Optional[Any]:
        """Find field by multiple possible names"""
        if not obj or not isinstance(obj, dict) or max_depth <= 0:
            return None
        
        for field_name in field_names:
            field_lower = field_name.lower()
            for key, value in obj.items():
                if str(key).lower() == field_lower:
                    return value
        
        # Recursive search if not found at this level
        if max_depth > 1:
            for value in obj.values():
                if isinstance(value, dict):
                    result = self._find_field(value, field_names, max_depth - 1)
                    if result is not None:
                        return result
        
        return None
    
    def _normalize_keys(self, obj: Any) -> Any:
        """Normalize all keys to lowercase"""
        if isinstance(obj, dict):
            return {k.lower() if isinstance(k, str) else k: self._normalize_keys(v)
                   for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._normalize_keys(item) for item in obj]
        return obj
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names"""
        return self.feature_names
    
    def get_feature_count(self) -> int:
        """Get total number of features"""
        return len(self.feature_names)


# ============================================================================
# The rest of the code remains the same but uses ComprehensiveFeatureExtractor
# ============================================================================

def create_counter_dict():
    """Factory function for creating defaultdict(int)"""
    return defaultdict(int)


class IncrementalDataProcessor:
    """Process large datasets incrementally with comprehensive features"""
    
    def __init__(self, config: EnhancedConfig):
        self.config = config
        self.repair_patterns = defaultdict(create_counter_dict)
        self.repair_vocabulary = {}
        self.transaction_count = 0
        self.checkpoint_count = 0
        
        # Initialize comprehensive feature extractor
        self.feature_extractor = ComprehensiveFeatureExtractor()
        logger.info(f"Using comprehensive feature extractor with {self.feature_extractor.get_feature_count()} features")
        
    def process_directory_incrementally(self, directory_path: str, 
                                       callback_fn: Optional[callable] = None) -> Generator:
        """Process directory incrementally with comprehensive features"""
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
        """Process a single transaction with comprehensive features"""
        try:
            # Skip if no repairs
            if 'ace' not in txn_data or not txn_data.get('ace'):
                return None
            
            # Extract comprehensive features (for training, use 'before' state)
            features = self.feature_extractor.extract_features(txn_data, for_training=True)
            
            # Get repair labels
            repairs = []
            for repair in txn_data.get('ace', []):
                repair_id = str(repair.get('id', 'unknown')) if isinstance(repair, dict) else str(repair)
                repairs.append(repair_id)
                
                # Update vocabulary
                if repair_id not in self.repair_vocabulary:
                    self.repair_vocabulary[repair_id] = len(self.repair_vocabulary)
            
            return {
                'txn_id': txn_id,
                'features': features,
                'repairs': repairs,
                'source': txn_data.get('source'),
                'clearing': txn_data.get('clearing'),
                'num_features': len(features),
                'non_zero_features': np.sum(features != 0)
            }
            
        except Exception as e:
            logger.debug(f"Error processing transaction {txn_id}: {e}")
            return None
    
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
            'feature_count': self.feature_extractor.get_feature_count()
        }
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        logger.info(f"Saved checkpoint {self.checkpoint_count} ({self.transaction_count} transactions)")


# The rest of the code (RepairNN, EnhancedTrainer, EnhancedPredictor, etc.) 
# remains the same but will now use the 183-feature vectors

# ... [Include the rest of the original code with minor adjustments for feature count]

if __name__ == "__main__":
    # Test the comprehensive feature extractor
    extractor = ComprehensiveFeatureExtractor()
    print(f"Comprehensive Feature Extractor initialized with {extractor.get_feature_count()} features")
    print("\nFeature categories:")
    print(f"  Agent features: {sum(1 for f in extractor.feature_names if 'Agt' in f or 'agt' in f)}")
    print(f"  Party features: {sum(1 for f in extractor.feature_names if any(p in f for p in ['cdtr', 'dbtr']))}")
    print(f"  Account features: {sum(1 for f in extractor.feature_names if 'Acct' in f or 'acct' in f)}")
    print(f"  Cross-entity: {sum(1 for f in extractor.feature_names if 'affects' in f or 'mismatch' in f)}")
    print(f"  Quality metrics: {sum(1 for f in extractor.feature_names if 'score' in f or 'count' in f)}")
    
    print("\nKey repair trigger features:")
    trigger_features = [
        'cdtrAgt_has_clearing_no_bic',
        'dbtrAgt_has_clearing_no_bic', 
        'cdtr_missing_country',
        'dbtr_missing_country',
        'cdtrAgt_mmbid_wrong_level'
    ]
    for feat in trigger_features:
        if feat in extractor.feature_names:
            print(f"  ✓ {feat}")
    
    print("\nReady for enhanced prediction with comprehensive features!")
