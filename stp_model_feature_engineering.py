"""
Feature Engineering for IFML Payment Data.
Transforms parsed IFML features into ML-ready format with encoding and derived features.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction import FeatureHasher
import pickle
from pathlib import Path


class IFMLFeatureEngineer:
    """
    Transforms raw IFML features into ML-ready format.
    Handles categorical encoding, missing values, and derived features.
    """
    
    # Columns that should be one-hot encoded (low cardinality)
    # NOTE: source_code, incoming_msg_type, incoming_format removed - spurious correlation
    ONEHOT_COLS = [
        'bank_operation_code'
    ]
    
    # Columns that should be target/frequency encoded (high cardinality)
    # NOTE: primary_currency removed - spurious correlation with error codes
    FREQUENCY_COLS = [
        'originator_country', 'beneficiary_country',
        'cdt_country', 'bnf_country', 'dbt_country', 'intm_country',
        # BIC country codes
        'orig_bic_country', 'send_bic_country', 'dbt_bic_country',
        'cdt_bic_country', 'intm_bic_country', 'bnf_bic_country',
        # IBAN country codes
        'orig_iban_country', 'send_iban_country', 'dbt_iban_country',
        'cdt_iban_country', 'intm_iban_country', 'bnf_iban_country',
    ]
    
    # Boolean columns - keep as-is
    BOOL_COLS = [
        'has_instructed_amount', 'amount_mismatch', 'is_cross_border',
        'has_intermediary', 'has_beneficiary_bank', 'has_bank_info',
        'orig_present', 'orig_has_id', 'orig_has_bic', 'orig_has_account', 'orig_has_name',
        'send_present', 'send_has_id', 'send_has_bic', 'send_has_account', 'send_has_name',
        'dbt_present', 'dbt_has_id', 'dbt_has_bic', 'dbt_has_account', 'dbt_has_name',
        'cdt_present', 'cdt_has_id', 'cdt_has_bic', 'cdt_has_account', 'cdt_has_name',
        'intm_present', 'intm_has_id', 'intm_has_bic', 'intm_has_account', 'intm_has_name',
        'bnf_present', 'bnf_has_id', 'bnf_has_bic', 'bnf_has_account', 'bnf_has_name',
        # BIC/IBAN country match flags
        'orig_bic_iban_match', 'send_bic_iban_match', 'dbt_bic_iban_match',
        'cdt_bic_iban_match', 'intm_bic_iban_match', 'bnf_bic_iban_match',
        # BIC validation flags
        'orig_bic_valid_format', 'send_bic_valid_format', 'dbt_bic_valid_format',
        'cdt_bic_valid_format', 'intm_bic_valid_format', 'bnf_bic_valid_format',
        'orig_bic_valid_country', 'send_bic_valid_country', 'dbt_bic_valid_country',
        'cdt_bic_valid_country', 'intm_bic_valid_country', 'bnf_bic_valid_country',
        # IBAN validation flags
        'orig_iban_valid_format', 'send_iban_valid_format', 'dbt_iban_valid_format',
        'cdt_iban_valid_format', 'intm_iban_valid_format', 'bnf_iban_valid_format',
        'orig_iban_checksum_valid', 'send_iban_checksum_valid', 'dbt_iban_checksum_valid',
        'cdt_iban_checksum_valid', 'intm_iban_checksum_valid', 'bnf_iban_checksum_valid',
        # BIC-Party country match
        'orig_bic_party_country_match', 'send_bic_party_country_match', 'dbt_bic_party_country_match',
        'cdt_bic_party_country_match', 'intm_bic_party_country_match', 'bnf_bic_party_country_match',
        # Account type detection flags
        'orig_is_clabe', 'send_is_clabe', 'dbt_is_clabe', 'cdt_is_clabe', 'intm_is_clabe', 'bnf_is_clabe',
        'orig_is_fedaba', 'send_is_fedaba', 'dbt_is_fedaba', 'cdt_is_fedaba', 'intm_is_fedaba', 'bnf_is_fedaba',
        'orig_is_chips_aba', 'send_is_chips_aba', 'dbt_is_chips_aba', 'cdt_is_chips_aba', 'intm_is_chips_aba', 'bnf_is_chips_aba',
        'orig_is_chips_uid', 'send_is_chips_uid', 'dbt_is_chips_uid', 'cdt_is_chips_uid', 'intm_is_chips_uid', 'bnf_is_chips_uid',
        # Account numeric only
        'orig_account_numeric', 'send_account_numeric', 'dbt_account_numeric',
        'cdt_account_numeric', 'intm_account_numeric', 'bnf_account_numeric',
        # Address-based domestic/international flags
        'orig_is_domestic', 'send_is_domestic', 'dbt_is_domestic',
        'cdt_is_domestic', 'intm_is_domestic', 'bnf_is_domestic',
        'orig_is_international', 'send_is_international', 'dbt_is_international',
        'cdt_is_international', 'intm_is_international', 'bnf_is_international',
        # NCH/Clearing code flags (for 8026, 8895)
        'orig_has_nch', 'send_has_nch', 'dbt_has_nch',
        'cdt_has_nch', 'intm_has_nch', 'bnf_has_nch',
        'orig_nch_valid', 'send_nch_valid', 'dbt_nch_valid',
        'cdt_nch_valid', 'intm_nch_valid', 'bnf_nch_valid',
        'orig_fedaba_checksum_valid', 'send_fedaba_checksum_valid', 'dbt_fedaba_checksum_valid',
        'cdt_fedaba_checksum_valid', 'intm_fedaba_checksum_valid', 'bnf_fedaba_checksum_valid',
        'orig_has_adr_bank_id', 'send_has_adr_bank_id', 'dbt_has_adr_bank_id',
        'cdt_has_adr_bank_id', 'intm_has_adr_bank_id', 'bnf_has_adr_bank_id',
        # IBAN requirement flags (for 8004)
        'orig_has_iban', 'send_has_iban', 'dbt_has_iban',
        'cdt_has_iban', 'intm_has_iban', 'bnf_has_iban',
        'orig_needs_iban', 'send_needs_iban', 'dbt_needs_iban',
        'cdt_needs_iban', 'intm_needs_iban', 'bnf_needs_iban',
        # NCH validation applicability (for 8895)
        'orig_nch_validation_applicable', 'send_nch_validation_applicable', 'dbt_nch_validation_applicable',
        'cdt_nch_validation_applicable', 'intm_nch_validation_applicable', 'bnf_nch_validation_applicable',
        'ordi_nch_validation_applicable', 'acwi_nch_validation_applicable',
        # === REPAIR DETECTION FEATURES (for 9XXX) ===
        # Account cleaning (9002, 9009, 9014, 9015)
        'orig_account_has_dirty_chars', 'send_account_has_dirty_chars', 'dbt_account_has_dirty_chars',
        'cdt_account_has_dirty_chars', 'intm_account_has_dirty_chars', 'bnf_account_has_dirty_chars',
        'ordi_account_has_dirty_chars', 'acwi_account_has_dirty_chars',
        'orig_name_has_dirty_chars', 'send_name_has_dirty_chars', 'dbt_name_has_dirty_chars',
        'cdt_name_has_dirty_chars', 'intm_name_has_dirty_chars', 'bnf_name_has_dirty_chars',
        'ordi_name_has_dirty_chars', 'acwi_name_has_dirty_chars',
        'orig_account_has_spaces', 'send_account_has_spaces', 'dbt_account_has_spaces',
        'cdt_account_has_spaces', 'intm_account_has_spaces', 'bnf_account_has_spaces',
        'ordi_account_has_spaces', 'acwi_account_has_spaces',
        'orig_account_has_special_chars', 'send_account_has_special_chars', 'dbt_account_has_special_chars',
        'cdt_account_has_special_chars', 'intm_account_has_special_chars', 'bnf_account_has_special_chars',
        'ordi_account_has_special_chars', 'acwi_account_has_special_chars',
        # IBAN derivation (9004, 9007)
        'orig_is_iban_derivable', 'send_is_iban_derivable', 'dbt_is_iban_derivable',
        'cdt_is_iban_derivable', 'intm_is_iban_derivable', 'bnf_is_iban_derivable',
        'ordi_is_iban_derivable', 'acwi_is_iban_derivable',
        'orig_has_bban_in_iban_country', 'send_has_bban_in_iban_country', 'dbt_has_bban_in_iban_country',
        'cdt_has_bban_in_iban_country', 'intm_has_bban_in_iban_country', 'bnf_has_bban_in_iban_country',
        'ordi_has_bban_in_iban_country', 'acwi_has_bban_in_iban_country',
        # BIC derivation (9005, 9008)
        'orig_is_bic_derivable', 'send_is_bic_derivable', 'dbt_is_bic_derivable',
        'cdt_is_bic_derivable', 'intm_is_bic_derivable', 'bnf_is_bic_derivable',
        'ordi_is_bic_derivable', 'acwi_is_bic_derivable',
        'orig_has_nch_no_bic', 'send_has_nch_no_bic', 'dbt_has_nch_no_bic',
        'cdt_has_nch_no_bic', 'intm_has_nch_no_bic', 'bnf_has_nch_no_bic',
        'ordi_has_nch_no_bic', 'acwi_has_nch_no_bic',
        'orig_has_iban_no_bic', 'send_has_iban_no_bic', 'dbt_has_iban_no_bic',
        'cdt_has_iban_no_bic', 'intm_has_iban_no_bic', 'bnf_has_iban_no_bic',
        'ordi_has_iban_no_bic', 'acwi_has_iban_no_bic',
        # NCH repair (9000, 9021, 9028)
        'orig_nch_has_dirty_chars', 'send_nch_has_dirty_chars', 'dbt_nch_has_dirty_chars',
        'cdt_nch_has_dirty_chars', 'intm_nch_has_dirty_chars', 'bnf_nch_has_dirty_chars',
        'ordi_nch_has_dirty_chars', 'acwi_nch_has_dirty_chars',
        'orig_nch_needs_formatting', 'send_nch_needs_formatting', 'dbt_nch_needs_formatting',
        'cdt_nch_needs_formatting', 'intm_nch_needs_formatting', 'bnf_nch_needs_formatting',
        'ordi_nch_needs_formatting', 'acwi_nch_needs_formatting',
        # Multiple/duplicate (9017, 9018)
        'orig_has_multiple_ids', 'send_has_multiple_ids', 'dbt_has_multiple_ids',
        'cdt_has_multiple_ids', 'intm_has_multiple_ids', 'bnf_has_multiple_ids',
        'ordi_has_multiple_ids', 'acwi_has_multiple_ids',
        'orig_has_duplicate_info', 'send_has_duplicate_info', 'dbt_has_duplicate_info',
        'cdt_has_duplicate_info', 'intm_has_duplicate_info', 'bnf_has_duplicate_info',
        'ordi_has_duplicate_info', 'acwi_has_duplicate_info',
        # Account formatting (9022)
        'orig_account_needs_length_fix', 'send_account_needs_length_fix', 'dbt_account_needs_length_fix',
        'cdt_account_needs_length_fix', 'intm_account_needs_length_fix', 'bnf_account_needs_length_fix',
        'ordi_account_needs_length_fix', 'acwi_account_needs_length_fix',
        # IBAN formatting (9006, 9012)
        'orig_iban_needs_formatting', 'send_iban_needs_formatting', 'dbt_iban_needs_formatting',
        'cdt_iban_needs_formatting', 'intm_iban_needs_formatting', 'bnf_iban_needs_formatting',
        'ordi_iban_needs_formatting', 'acwi_iban_needs_formatting',
        # ID cleaning (9019) - identifier contains non-alphanumeric chars
        'orig_id_needs_cleaning', 'send_id_needs_cleaning', 'dbt_id_needs_cleaning',
        'cdt_id_needs_cleaning', 'intm_id_needs_cleaning', 'bnf_id_needs_cleaning',
        'ordi_id_needs_cleaning', 'acwi_id_needs_cleaning',
        'orig_iban_needs_cleaning', 'send_iban_needs_cleaning', 'dbt_iban_needs_cleaning',
        'cdt_iban_needs_cleaning', 'intm_iban_needs_cleaning', 'bnf_iban_needs_cleaning',
        'ordi_iban_needs_cleaning', 'acwi_iban_needs_cleaning',
        'orig_any_id_needs_cleaning', 'send_any_id_needs_cleaning', 'dbt_any_id_needs_cleaning',
        'cdt_any_id_needs_cleaning', 'intm_any_id_needs_cleaning', 'bnf_any_id_needs_cleaning',
        'ordi_any_id_needs_cleaning', 'acwi_any_id_needs_cleaning',
    ]
    
    # Numeric columns
    NUMERIC_COLS = [
        'primary_amount', 'instructed_amount', 'bank_info_count', 'amount_count',
        'orig_address_lines', 'send_address_lines', 'dbt_address_lines',
        'cdt_address_lines', 'intm_address_lines', 'bnf_address_lines',
        # BIC lengths (4, 8, or 11 typically)
        'orig_bic_length', 'send_bic_length', 'dbt_bic_length',
        'cdt_bic_length', 'intm_bic_length', 'bnf_bic_length',
        # Account lengths
        'orig_account_length', 'send_account_length', 'dbt_account_length',
        'cdt_account_length', 'intm_account_length', 'bnf_account_length',
        # NCH source count (for 8026 - inconsistency detection)
        'orig_nch_sources', 'send_nch_sources', 'dbt_nch_sources',
        'cdt_nch_sources', 'intm_nch_sources', 'bnf_nch_sources',
    ]
    
    # Account type columns (categorical)
    ACCOUNT_TYPE_COLS = [
        'cdt_account_type', 'dbt_account_type', 'bnf_account_type'
    ]
    
    # NCH type columns (categorical) - FEDABA, CHIPS, SORTCODE, etc.
    NCH_TYPE_COLS = [
        'orig_nch_type', 'send_nch_type', 'dbt_nch_type',
        'cdt_nch_type', 'intm_nch_type', 'bnf_nch_type',
        'orig_adr_bank_id_type', 'send_adr_bank_id_type', 'dbt_adr_bank_id_type',
        'cdt_adr_bank_id_type', 'intm_adr_bank_id_type', 'bnf_adr_bank_id_type',
    ]
    
    def __init__(self):
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.frequency_maps: Dict[str, Dict[str, float]] = {}
        self.feature_columns: List[str] = []
        self.fitted = False
        
    def fit(self, df: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit encoders on training data.
        
        Args:
            df: DataFrame with raw IFML features
            y: Optional target for target encoding (future use)
        """
        # Fit one-hot encoding columns
        for col in self.ONEHOT_COLS:
            if col in df.columns:
                le = LabelEncoder()
                # Include 'unknown' for unseen values
                values = df[col].fillna('__missing__').astype(str).tolist() + ['__unknown__']
                le.fit(values)
                self.label_encoders[col] = le
        
        # Fit frequency encoding for high-cardinality columns
        for col in self.FREQUENCY_COLS:
            if col in df.columns:
                freq = df[col].value_counts(normalize=True).to_dict()
                self.frequency_maps[col] = freq
        
        # Fit account type encoding
        for col in self.ACCOUNT_TYPE_COLS:
            if col in df.columns:
                le = LabelEncoder()
                values = df[col].fillna('__missing__').astype(str).tolist() + ['__unknown__']
                le.fit(values)
                self.label_encoders[col] = le
        
        # Fit NCH type encoding
        for col in self.NCH_TYPE_COLS:
            if col in df.columns:
                le = LabelEncoder()
                values = df[col].fillna('__missing__').astype(str).tolist() + ['__unknown__']
                le.fit(values)
                self.label_encoders[col] = le
        
        self.fitted = True
        
        # Generate feature columns list
        self._build_feature_columns(df)
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform raw features to ML-ready format.
        """
        if not self.fitted:
            raise ValueError("FeatureEngineer must be fitted before transform")
        
        result = pd.DataFrame(index=df.index)
        
        # Boolean features - convert to int
        for col in self.BOOL_COLS:
            if col in df.columns:
                result[col] = df[col].fillna(False).astype(int)
            else:
                result[col] = 0
        
        # Numeric features - fill missing with 0
        for col in self.NUMERIC_COLS:
            if col in df.columns:
                result[col] = df[col].fillna(0).astype(float)
            else:
                result[col] = 0.0
        
        # One-hot encoded features
        for col in self.ONEHOT_COLS:
            if col in df.columns and col in self.label_encoders:
                le = self.label_encoders[col]
                values = df[col].fillna('__missing__').astype(str)
                # Handle unseen categories
                values = values.apply(lambda x: x if x in le.classes_ else '__unknown__')
                result[f'{col}_encoded'] = le.transform(values)
        
        # Frequency encoded features
        for col in self.FREQUENCY_COLS:
            if col in df.columns and col in self.frequency_maps:
                freq_map = self.frequency_maps[col]
                values = df[col].fillna('__missing__').astype(str)
                # Use 0 frequency for unseen values
                result[f'{col}_freq'] = values.apply(lambda x: freq_map.get(x, 0.0))
        
        # Account type encoding
        for col in self.ACCOUNT_TYPE_COLS:
            if col in df.columns and col in self.label_encoders:
                le = self.label_encoders[col]
                values = df[col].fillna('__missing__').astype(str)
                values = values.apply(lambda x: x if x in le.classes_ else '__unknown__')
                result[f'{col}_encoded'] = le.transform(values)
        
        # NCH type encoding
        for col in self.NCH_TYPE_COLS:
            if col in df.columns and col in self.label_encoders:
                le = self.label_encoders[col]
                values = df[col].fillna('__missing__').astype(str)
                values = values.apply(lambda x: x if x in le.classes_ else '__unknown__')
                result[f'{col}_encoded'] = le.transform(values)
        
        # Add derived features
        result = self._add_derived_features(df, result)
        
        return result
    
    def fit_transform(self, df: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(df, y)
        return self.transform(df)
    
    def _add_derived_features(self, raw_df: pd.DataFrame, result_df: pd.DataFrame) -> pd.DataFrame:
        """Add derived/interaction features."""
        
        # Missing field pattern as bitmask
        party_present_cols = [c for c in result_df.columns if c.endswith('_present')]
        if party_present_cols:
            result_df['party_count'] = result_df[party_present_cols].sum(axis=1)
        
        # BIC availability pattern
        bic_cols = [c for c in result_df.columns if c.endswith('_has_bic')]
        if bic_cols:
            result_df['bic_count'] = result_df[bic_cols].sum(axis=1)
        
        # Account availability pattern
        account_cols = [c for c in result_df.columns if c.endswith('_has_account')]
        if account_cols:
            result_df['account_count'] = result_df[account_cols].sum(axis=1)
        
        # Address completeness
        address_cols = [c for c in result_df.columns if c.endswith('_address_lines')]
        if address_cols:
            result_df['total_address_lines'] = result_df[address_cols].sum(axis=1)
            result_df['avg_address_lines'] = result_df[address_cols].mean(axis=1)
        
        # Amount features
        if 'primary_amount' in result_df.columns:
            result_df['amount_bucket'] = pd.cut(
                result_df['primary_amount'],
                bins=[0, 1000, 10000, 100000, 1000000, float('inf')],
                labels=[0, 1, 2, 3, 4]
            ).astype(float).fillna(0)
        
        # Country match features
        if 'cdt_country' in raw_df.columns and 'bnf_country' in raw_df.columns:
            result_df['cdt_bnf_country_match'] = (
                raw_df['cdt_country'].fillna('') == raw_df['bnf_country'].fillna('')
            ).astype(int)
        
        # IBAN derivability hint - if credit party has BBAN but not in IBAN country
        if 'cdt_account_type' in raw_df.columns:
            result_df['has_bban'] = (raw_df['cdt_account_type'] == 'BBAN').astype(int)
        
        # Missing critical fields indicator
        result_df['missing_cdt_account'] = (
            (result_df.get('cdt_present', 0) == 1) & 
            (result_df.get('cdt_has_account', 0) == 0)
        ).astype(int)
        
        result_df['missing_bnf_bic'] = (
            (result_df.get('bnf_present', 0) == 1) & 
            (result_df.get('bnf_has_bic', 0) == 0)
        ).astype(int)
        
        # === 8XXX Code Specific Features ===
        
        # 8004: IBAN cannot be derived - needs IBAN but doesn't have it
        iban_cols = [c for c in result_df.columns if c.endswith('_has_iban')]
        needs_iban_cols = [c for c in result_df.columns if c.endswith('_needs_iban')]
        if iban_cols and needs_iban_cols:
            result_df['iban_count'] = result_df[iban_cols].sum(axis=1)
            result_df['needs_iban_count'] = result_df[needs_iban_cols].sum(axis=1)
            # Missing IBAN when needed
            result_df['missing_required_iban'] = (
                (result_df['needs_iban_count'] > 0) & 
                (result_df['iban_count'] == 0)
            ).astype(int)
        
        # 8022: IBAN/BIC country mismatch - count mismatches
        bic_iban_match_cols = [c for c in result_df.columns if c.endswith('_bic_iban_match')]
        if bic_iban_match_cols:
            # Count explicit mismatches (where both exist and don't match)
            result_df['bic_iban_mismatch_count'] = (
                result_df[bic_iban_match_cols].eq(False).sum(axis=1)
            )
        
        # 8026: NCH inconsistency - multiple NCH sources
        nch_sources_cols = [c for c in result_df.columns if c.endswith('_nch_sources')]
        if nch_sources_cols:
            result_df['total_nch_sources'] = result_df[nch_sources_cols].sum(axis=1)
            result_df['has_multiple_nch_sources'] = (result_df['total_nch_sources'] > 1).astype(int)
        
        # 8895: Invalid NCH - has NCH but not valid
        nch_cols = [c for c in result_df.columns if c.endswith('_has_nch')]
        nch_valid_cols = [c for c in result_df.columns if c.endswith('_nch_valid')]
        if nch_cols and nch_valid_cols:
            result_df['nch_count'] = result_df[nch_cols].sum(axis=1)
            result_df['nch_valid_count'] = result_df[nch_valid_cols].sum(axis=1)
            result_df['has_invalid_nch'] = (
                (result_df['nch_count'] > 0) & 
                (result_df['nch_valid_count'] < result_df['nch_count'])
            ).astype(int)
        
        # 8894: Invalid IBAN - has IBAN but format/checksum invalid
        iban_format_cols = [c for c in result_df.columns if c.endswith('_iban_valid_format')]
        iban_checksum_cols = [c for c in result_df.columns if c.endswith('_iban_checksum_valid')]
        if iban_format_cols and iban_checksum_cols:
            result_df['iban_format_valid_count'] = result_df[iban_format_cols].sum(axis=1)
            result_df['iban_checksum_valid_count'] = result_df[iban_checksum_cols].sum(axis=1)
            # Has IBAN but invalid
            if 'iban_count' in result_df.columns:
                result_df['has_invalid_iban'] = (
                    (result_df['iban_count'] > 0) & 
                    ((result_df['iban_format_valid_count'] < result_df['iban_count']) |
                     (result_df['iban_checksum_valid_count'] < result_df['iban_count']))
                ).astype(int)
        
        # 8001: Invalid BIC - has BIC but format invalid
        bic_format_cols = [c for c in result_df.columns if c.endswith('_bic_valid_format')]
        if bic_format_cols:
            result_df['bic_format_valid_count'] = result_df[bic_format_cols].sum(axis=1)
            if 'bic_count' in result_df.columns:
                result_df['has_invalid_bic'] = (
                    (result_df['bic_count'] > 0) & 
                    (result_df['bic_format_valid_count'] < result_df['bic_count'])
                ).astype(int)
        
        # Domestic vs International payment indicator
        domestic_cols = [c for c in result_df.columns if c.endswith('_is_domestic')]
        intl_cols = [c for c in result_df.columns if c.endswith('_is_international')]
        if domestic_cols and intl_cols:
            result_df['is_any_domestic'] = (result_df[domestic_cols].sum(axis=1) > 0).astype(int)
            result_df['is_any_international'] = (result_df[intl_cols].sum(axis=1) > 0).astype(int)
        
        # NCH validation applicability (for 8895 - only fires on domestic US payments)
        nch_applicable_cols = [c for c in result_df.columns if c.endswith('_nch_validation_applicable')]
        if nch_applicable_cols:
            result_df['nch_validation_applicable'] = (result_df[nch_applicable_cols].sum(axis=1) > 0).astype(int)
        
        return result_df
    
    def _build_feature_columns(self, df: pd.DataFrame):
        """Build list of expected feature columns."""
        self.feature_columns = []
        
        # Bool columns
        self.feature_columns.extend([c for c in self.BOOL_COLS if c in df.columns or True])
        
        # Numeric columns
        self.feature_columns.extend([c for c in self.NUMERIC_COLS if c in df.columns or True])
        
        # Encoded columns
        for col in self.ONEHOT_COLS:
            self.feature_columns.append(f'{col}_encoded')
        
        for col in self.FREQUENCY_COLS:
            self.feature_columns.append(f'{col}_freq')
        
        for col in self.ACCOUNT_TYPE_COLS:
            self.feature_columns.append(f'{col}_encoded')
        
        for col in self.NCH_TYPE_COLS:
            self.feature_columns.append(f'{col}_encoded')
        
        # Derived columns
        derived = [
            'party_count', 'bic_count', 'account_count',
            'total_address_lines', 'avg_address_lines', 'amount_bucket',
            'cdt_bnf_country_match', 'has_bban',
            'missing_cdt_account', 'missing_bnf_bic',
            # 8XXX specific derived features
            'iban_count', 'needs_iban_count', 'missing_required_iban',
            'bic_iban_mismatch_count',
            'total_nch_sources', 'has_multiple_nch_sources',
            'nch_count', 'nch_valid_count', 'has_invalid_nch',
            'iban_format_valid_count', 'iban_checksum_valid_count', 'has_invalid_iban',
            'bic_format_valid_count', 'has_invalid_bic',
            'is_any_domestic', 'is_any_international',
            'nch_validation_applicable'  # For 8895 - only domestic US
        ]
        self.feature_columns.extend(derived)
    
    def save(self, filepath: str):
        """Save fitted encoders to file."""
        state = {
            'label_encoders': self.label_encoders,
            'frequency_maps': self.frequency_maps,
            'feature_columns': self.feature_columns,
            'fitted': self.fitted
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
    
    def load(self, filepath: str):
        """Load fitted encoders from file."""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        self.label_encoders = state['label_encoders']
        self.frequency_maps = state['frequency_maps']
        self.feature_columns = state['feature_columns']
        self.fitted = state['fitted']
        return self


class ErrorCodeLabelEncoder:
    """
    Handles multi-label encoding for ACE error codes.
    Supports both single-label and multi-label scenarios.
    """
    
    def __init__(self, min_samples: int = 5):
        """
        Args:
            min_samples: Minimum samples required to include a code as a class.
                        Codes with fewer samples are grouped into 'rare' class.
        """
        self.min_samples = min_samples
        self.code_to_idx: Dict[str, int] = {}
        self.idx_to_code: Dict[int, str] = {}
        self.rare_codes: List[str] = []
        self.code_counts: Dict[str, int] = {}
        self.n_classes: int = 0
        self.fitted = False
    
    def fit(self, code_lists: List[List[str]]):
        """
        Fit encoder on list of code lists.
        
        Args:
            code_lists: List of lists, where each inner list contains
                       codes for one sample.
        """
        # Count occurrences
        from collections import Counter
        all_codes = [code for codes in code_lists for code in codes]
        self.code_counts = Counter(all_codes)
        
        # Separate frequent and rare codes
        frequent_codes = sorted([
            code for code, count in self.code_counts.items() 
            if count >= self.min_samples
        ])
        self.rare_codes = [
            code for code, count in self.code_counts.items() 
            if count < self.min_samples
        ]
        
        # Build mapping (include 'RARE' class if there are rare codes)
        self.code_to_idx = {code: idx for idx, code in enumerate(frequent_codes)}
        
        if self.rare_codes:
            self.code_to_idx['__RARE__'] = len(frequent_codes)
        
        # Add 'NO_ERROR' class for clean transactions
        self.code_to_idx['__NO_ERROR__'] = len(self.code_to_idx)
        
        self.idx_to_code = {idx: code for code, idx in self.code_to_idx.items()}
        self.n_classes = len(self.code_to_idx)
        self.fitted = True
        
        return self
    
    def transform_multilabel(self, code_lists: List[List[str]]) -> np.ndarray:
        """
        Transform to multi-label binary matrix.
        
        Returns:
            np.ndarray of shape (n_samples, n_classes) with 0/1 values.
        """
        if not self.fitted:
            raise ValueError("Encoder must be fitted before transform")
        
        result = np.zeros((len(code_lists), self.n_classes), dtype=np.float32)
        
        for i, codes in enumerate(code_lists):
            if not codes:
                # No errors - mark NO_ERROR class
                result[i, self.code_to_idx['__NO_ERROR__']] = 1
            else:
                for code in codes:
                    if code in self.code_to_idx:
                        result[i, self.code_to_idx[code]] = 1
                    elif code in self.rare_codes and '__RARE__' in self.code_to_idx:
                        result[i, self.code_to_idx['__RARE__']] = 1
        
        return result
    
    def transform_single_label(self, code_lists: List[List[str]], 
                               priority: str = 'error') -> np.ndarray:
        """
        Transform to single-label by selecting one code per sample.
        
        Args:
            code_lists: List of code lists
            priority: 'error' to prioritize error codes (E severity),
                     'first' to take first code.
        
        Returns:
            np.ndarray of shape (n_samples,) with class indices.
        """
        if not self.fitted:
            raise ValueError("Encoder must be fitted before transform")
        
        result = np.zeros(len(code_lists), dtype=np.int32)
        
        for i, codes in enumerate(code_lists):
            if not codes:
                result[i] = self.code_to_idx['__NO_ERROR__']
            else:
                # Take first code that maps to a known class
                selected = None
                for code in codes:
                    if code in self.code_to_idx:
                        selected = code
                        break
                
                if selected:
                    result[i] = self.code_to_idx[selected]
                elif self.rare_codes and '__RARE__' in self.code_to_idx:
                    result[i] = self.code_to_idx['__RARE__']
                else:
                    result[i] = self.code_to_idx['__NO_ERROR__']
        
        return result
    
    def inverse_transform(self, indices: np.ndarray) -> List[str]:
        """Convert class indices back to code strings."""
        return [self.idx_to_code.get(idx, '__UNKNOWN__') for idx in indices]
    
    def get_class_weights(self) -> Dict[int, float]:
        """Compute balanced class weights for training."""
        total = sum(self.code_counts.values())
        weights = {}
        
        for code, idx in self.code_to_idx.items():
            if code == '__RARE__':
                count = sum(self.code_counts.get(c, 0) for c in self.rare_codes)
            elif code == '__NO_ERROR__':
                count = max(1, total // 10)  # Estimate
            else:
                count = self.code_counts.get(code, 1)
            
            weights[idx] = total / (len(self.code_to_idx) * max(count, 1))
        
        return weights
    
    def save(self, filepath: str):
        """Save encoder state."""
        state = {
            'code_to_idx': self.code_to_idx,
            'idx_to_code': self.idx_to_code,
            'rare_codes': self.rare_codes,
            'code_counts': self.code_counts,
            'n_classes': self.n_classes,
            'min_samples': self.min_samples,
            'fitted': self.fitted
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
    
    def load(self, filepath: str):
        """Load encoder state."""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        self.__dict__.update(state)
        return self
    
    def describe(self) -> str:
        """Get summary of encoder state."""
        lines = [
            f"Error Code Label Encoder",
            f"  Total classes: {self.n_classes}",
            f"  Frequent codes: {len(self.code_to_idx) - (2 if self.rare_codes else 1)}",
            f"  Rare codes (grouped): {len(self.rare_codes)}",
            f"  Minimum samples threshold: {self.min_samples}",
            f"\nCode distribution:"
        ]
        
        for code, count in sorted(self.code_counts.items(), key=lambda x: -x[1])[:20]:
            status = "frequent" if code in self.code_to_idx else "rare"
            lines.append(f"    {code}: {count} ({status})")
        
        if len(self.code_counts) > 20:
            lines.append(f"    ... and {len(self.code_counts) - 20} more codes")
        
        return '\n'.join(lines)


if __name__ == '__main__':
    # Test feature engineering
    sample_data = pd.DataFrame([
        {
            'source_code': 'SWF',
            'incoming_msg_type': '103',
            'bank_operation_code': 'CRED',
            'primary_currency': 'USD',
            'primary_amount': 12000.0,
            'is_cross_border': True,
            'cdt_present': True,
            'cdt_has_account': True,
            'cdt_account_type': 'BBAN',
            'cdt_country': 'JP',
            'bnf_country': 'JP',
        },
        {
            'source_code': 'SWF',
            'incoming_msg_type': '103',
            'bank_operation_code': 'CRED',
            'primary_currency': 'EUR',
            'primary_amount': 5000.0,
            'is_cross_border': False,
            'cdt_present': True,
            'cdt_has_account': False,
            'cdt_account_type': None,
            'cdt_country': 'DE',
            'bnf_country': 'DE',
        }
    ])
    
    engineer = IFMLFeatureEngineer()
    transformed = engineer.fit_transform(sample_data)
    
    print("Transformed features:")
    print(transformed.columns.tolist())
    print("\nSample values:")
    print(transformed.iloc[0])
