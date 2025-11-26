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
    ONEHOT_COLS = [
        'source_code', 'incoming_msg_type', 'incoming_format',
        'bank_operation_code'
    ]
    
    # Columns that should be target/frequency encoded (high cardinality)
    FREQUENCY_COLS = [
        'primary_currency', 'originator_country', 'beneficiary_country',
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
    ]
    
    # Numeric columns
    NUMERIC_COLS = [
        'primary_amount', 'instructed_amount', 'bank_info_count', 'amount_count',
        'orig_address_lines', 'send_address_lines', 'dbt_address_lines',
        'cdt_address_lines', 'intm_address_lines', 'bnf_address_lines',
        # BIC lengths (4, 8, or 11 typically)
        'orig_bic_length', 'send_bic_length', 'dbt_bic_length',
        'cdt_bic_length', 'intm_bic_length', 'bnf_bic_length',
    ]
    
    # Account type columns (categorical)
    ACCOUNT_TYPE_COLS = [
        'cdt_account_type', 'dbt_account_type', 'bnf_account_type'
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
        
        # Derived columns
        derived = [
            'party_count', 'bic_count', 'account_count',
            'total_address_lines', 'avg_address_lines', 'amount_bucket',
            'cdt_bnf_country_match', 'has_bban',
            'missing_cdt_account', 'missing_bnf_bic'
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
