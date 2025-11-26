"""
Data Pipeline for IFML Error Code Prediction.
Handles loading, pairing, and preprocessing of IFML request/response data.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from collections import defaultdict
import glob
import re

from ifml_parser import IFMLParser, IFMLResponseParser, IFMLFeatures
from feature_engineering import IFMLFeatureEngineer, ErrorCodeLabelEncoder


@dataclass
class PaymentRecord:
    """A single payment with its request features and response codes."""
    transaction_id: str
    request_features: dict
    response_codes: List[dict]
    error_codes_only: List[str]  # Just the codes (e.g., ['8004', '7452'])
    severity_map: Dict[str, str]  # code -> severity


class IFMLDataPipeline:
    """
    End-to-end data pipeline for IFML error code prediction.
    Handles multiple data loading patterns and creates ML-ready datasets.
    """
    
    def __init__(self):
        self.parser = IFMLParser()
        self.response_parser = IFMLResponseParser()
        self.feature_engineer = IFMLFeatureEngineer()
        self.label_encoder = ErrorCodeLabelEncoder()
        self.records: List[PaymentRecord] = []
    
    def load_paired_files(self, request_dir: str, response_dir: str,
                          request_pattern: str = "*.json",
                          response_pattern: str = "*.json") -> int:
        """
        Load paired request/response files from directories.
        Assumes files are matched by transaction ID in filename.
        
        Returns number of records loaded.
        """
        request_files = list(Path(request_dir).glob(request_pattern))
        response_files = {Path(f).stem: f for f in Path(response_dir).glob(response_pattern)}
        
        loaded = 0
        for req_file in request_files:
            txn_id = req_file.stem
            
            if txn_id in response_files:
                record = self._load_pair(req_file, response_files[txn_id])
                if record:
                    self.records.append(record)
                    loaded += 1
        
        return loaded
    
    def load_combined_files(self, data_dir: str, pattern: str = "*.json") -> int:
        """
        Load files that contain both request and response.
        Each file should have a transaction ID as key with Request/Response nested.
        """
        files = list(Path(data_dir).glob(pattern))
        loaded = 0
        
        for filepath in files:
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                # Handle different structures
                records = self._extract_records_from_combined(data)
                self.records.extend(records)
                loaded += len(records)
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
        
        return loaded
    
    def load_single_file(self, filepath: str) -> int:
        """
        Load a single JSON file containing multiple payments.
        
        Expected format:
        {
            "2025092900000192": {
                "Request": { "IFML": { ... } },
                "Response": { "IFML": { ... } }
            },
            "2025092900000193": {
                "Request": { ... },
                "Response": { ... }
            },
            ...
        }
        
        Returns number of records loaded.
        """
        filepath = Path(filepath)
        loaded = 0
        errors = []
        
        print(f"Loading from {filepath}...")
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        for txn_id, payment_data in data.items():
            try:
                if not isinstance(payment_data, dict):
                    continue
                
                # Extract Request and Response
                request_data = payment_data.get('Request')
                response_data = payment_data.get('Response')
                
                if not request_data:
                    # Maybe the IFML is directly nested
                    if 'IFML' in payment_data:
                        request_data = payment_data
                        response_data = None
                    else:
                        continue
                
                # Parse request
                features = self.parser.parse(request_data)
                feature_dict = self.parser.to_dict(features)
                
                # Parse response if present
                if response_data:
                    txn_uid, codes = self.response_parser.parse(response_data)
                else:
                    txn_uid = txn_id
                    codes = []
                
                record = PaymentRecord(
                    transaction_id=txn_uid or txn_id,
                    request_features=feature_dict,
                    response_codes=codes,
                    error_codes_only=[c['code'] for c in codes if c.get('code')],
                    severity_map={c['code']: c['severity'] for c in codes if c.get('code')}
                )
                
                self.records.append(record)
                loaded += 1
                
            except Exception as e:
                errors.append(f"{txn_id}: {str(e)}")
        
        print(f"  Loaded {loaded} payments")
        if errors:
            print(f"  Errors ({len(errors)}):")
            for err in errors[:5]:
                print(f"    {err}")
            if len(errors) > 5:
                print(f"    ... and {len(errors) - 5} more")
        
        return loaded
    
    def load_from_dataframe(self, df: pd.DataFrame,
                            request_col: str = 'request',
                            response_col: str = 'response',
                            txn_id_col: str = 'transaction_id') -> int:
        """
        Load from a DataFrame where each row contains JSON strings
        for request and response.
        """
        loaded = 0
        
        for idx, row in df.iterrows():
            try:
                request_data = json.loads(row[request_col]) if isinstance(row[request_col], str) else row[request_col]
                response_data = json.loads(row[response_col]) if isinstance(row[response_col], str) else row[response_col]
                
                record = self._process_pair(
                    row.get(txn_id_col, str(idx)),
                    request_data,
                    response_data
                )
                
                if record:
                    self.records.append(record)
                    loaded += 1
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
        
        return loaded
    
    def load_sample_data(self, request_json: dict, response_jsons: List[dict]) -> int:
        """
        Load from in-memory JSON objects (for testing).
        
        Args:
            request_json: Single request JSON
            response_jsons: List of response JSONs (different outcomes)
        """
        loaded = 0
        
        # Parse request once
        features = self.parser.parse(request_json)
        feature_dict = self.parser.to_dict(features)
        
        for i, response_json in enumerate(response_jsons):
            txn_uid, codes = self.response_parser.parse(response_json)
            
            record = PaymentRecord(
                transaction_id=txn_uid or f"sample_{i}",
                request_features=feature_dict,
                response_codes=codes,
                error_codes_only=[c['code'] for c in codes if c['code']],
                severity_map={c['code']: c['severity'] for c in codes if c['code']}
            )
            
            self.records.append(record)
            loaded += 1
        
        return loaded
    
    def _load_pair(self, request_file: Path, response_file: Path) -> Optional[PaymentRecord]:
        """Load a single request/response pair."""
        try:
            with open(request_file, 'r') as f:
                request_data = json.load(f)
            with open(response_file, 'r') as f:
                response_data = json.load(f)
            
            return self._process_pair(request_file.stem, request_data, response_data)
        except Exception as e:
            print(f"Error loading pair {request_file}: {e}")
            return None
    
    def _process_pair(self, txn_id: str, request_data: dict, 
                      response_data: dict) -> Optional[PaymentRecord]:
        """Process a request/response pair into a PaymentRecord."""
        # Parse request
        features = self.parser.parse(request_data)
        feature_dict = self.parser.to_dict(features)
        
        # Parse response
        txn_uid, codes = self.response_parser.parse(response_data)
        
        return PaymentRecord(
            transaction_id=txn_uid or txn_id,
            request_features=feature_dict,
            response_codes=codes,
            error_codes_only=[c['code'] for c in codes if c['code']],
            severity_map={c['code']: c['severity'] for c in codes if c['code']}
        )
    
    def _extract_records_from_combined(self, data: dict) -> List[PaymentRecord]:
        """Extract records from combined request/response JSON structures."""
        records = []
        
        # Check if top-level keys are transaction IDs
        for key, value in data.items():
            if isinstance(value, dict):
                if 'Request' in value and 'Response' in value:
                    # Structure: {txn_id: {Request: {...}, Response: {...}}}
                    record = self._process_pair(key, value['Request'], value['Response'])
                    if record:
                        records.append(record)
                elif 'Request' in value:
                    # Structure: {txn_id: {Request: {...}}} - no response yet
                    features = self.parser.parse(value)
                    feature_dict = self.parser.to_dict(features)
                    record = PaymentRecord(
                        transaction_id=key,
                        request_features=feature_dict,
                        response_codes=[],
                        error_codes_only=[],
                        severity_map={}
                    )
                    records.append(record)
        
        return records
    
    def create_dataset(self, 
                       filter_severity: Optional[List[str]] = None,
                       min_code_samples: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create ML-ready dataset from loaded records.
        
        Args:
            filter_severity: If provided, only include codes with these severities.
                           E.g., ['E'] for errors only, ['E', 'W'] for errors and warnings.
            min_code_samples: Minimum samples for a code to be a separate class.
        
        Returns:
            Tuple of (X_raw, X_transformed, y_multilabel)
        """
        if not self.records:
            raise ValueError("No records loaded. Call a load_* method first.")
        
        # Extract raw features
        raw_features = []
        code_lists = []
        
        for record in self.records:
            raw_features.append(record.request_features)
            
            # Filter codes by severity if requested
            if filter_severity:
                codes = [
                    code for code in record.error_codes_only
                    if record.severity_map.get(code) in filter_severity
                ]
            else:
                codes = record.error_codes_only
            
            code_lists.append(codes)
        
        # Create DataFrames
        X_raw = pd.DataFrame(raw_features)
        
        # Fit and transform features
        X_transformed = self.feature_engineer.fit_transform(X_raw)
        
        # Fit and transform labels
        self.label_encoder = ErrorCodeLabelEncoder(min_samples=min_code_samples)
        self.label_encoder.fit(code_lists)
        y_multilabel = pd.DataFrame(
            self.label_encoder.transform_multilabel(code_lists),
            columns=[self.label_encoder.idx_to_code[i] for i in range(self.label_encoder.n_classes)]
        )
        
        return X_raw, X_transformed, y_multilabel
    
    def create_single_label_dataset(self,
                                    target_code: Optional[str] = None,
                                    filter_severity: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create dataset for single error code prediction.
        
        Args:
            target_code: Specific code to predict (binary classification).
                        If None, predicts all codes (multiclass).
            filter_severity: Filter codes by severity.
        
        Returns:
            Tuple of (X_transformed, y)
        """
        X_raw, X_transformed, y_multilabel = self.create_dataset(
            filter_severity=filter_severity
        )
        
        if target_code:
            # Binary classification for specific code
            if target_code in y_multilabel.columns:
                y = y_multilabel[target_code]
            else:
                print(f"Warning: Code {target_code} not found. Using all codes.")
                y = y_multilabel.max(axis=1)  # Any error
        else:
            # Multi-class: take primary code
            code_lists = [record.error_codes_only for record in self.records]
            y = pd.Series(self.label_encoder.transform_single_label(code_lists))
        
        return X_transformed, y
    
    def get_code_statistics(self) -> pd.DataFrame:
        """Get statistics about error code distribution."""
        # Use nested defaultdict to handle any severity code
        code_counts = defaultdict(lambda: defaultdict(int))
        all_severities = set()
        
        for record in self.records:
            for code in record.error_codes_only:
                severity = record.severity_map.get(code, 'I')
                all_severities.add(severity)
                code_counts[code]['total'] += 1
                code_counts[code][severity] += 1
        
        stats = []
        for code, counts in sorted(code_counts.items(), key=lambda x: -x[1]['total']):
            row = {
                'code': code,
                'total': counts['total'],
                'pct': counts['total'] / len(self.records) * 100
            }
            # Add counts for each severity found
            for sev in sorted(all_severities):
                row[sev] = counts.get(sev, 0)
            stats.append(row)
        
        return pd.DataFrame(stats)
    
    def save_encoders(self, output_dir: str):
        """Save fitted encoders for inference."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.feature_engineer.save(output_path / 'feature_engineer.pkl')
        self.label_encoder.save(output_path / 'label_encoder.pkl')
    
    def load_encoders(self, encoder_dir: str):
        """Load fitted encoders."""
        encoder_path = Path(encoder_dir)
        
        self.feature_engineer.load(encoder_path / 'feature_engineer.pkl')
        self.label_encoder.load(encoder_path / 'label_encoder.pkl')
    
    def transform_new_request(self, request_json: dict) -> np.ndarray:
        """
        Transform a new request for inference.
        Requires encoders to be fitted/loaded.
        """
        features = self.parser.parse(request_json)
        feature_dict = self.parser.to_dict(features)
        
        X_raw = pd.DataFrame([feature_dict])
        X_transformed = self.feature_engineer.transform(X_raw)
        
        return X_transformed.values
    
    def describe(self) -> str:
        """Get summary of loaded data."""
        if not self.records:
            return "No records loaded."
        
        total_codes = sum(len(r.error_codes_only) for r in self.records)
        records_with_errors = sum(1 for r in self.records if r.error_codes_only)
        
        lines = [
            f"IFML Data Pipeline Summary",
            f"=" * 40,
            f"Total records: {len(self.records)}",
            f"Records with codes: {records_with_errors} ({records_with_errors/len(self.records)*100:.1f}%)",
            f"Total code occurrences: {total_codes}",
            f"Average codes per record: {total_codes/len(self.records):.2f}",
            f"",
            f"Top 10 codes:"
        ]
        
        stats = self.get_code_statistics()
        for _, row in stats.head(10).iterrows():
            # Build severity breakdown dynamically
            sev_parts = []
            for col in row.index:
                if col not in ['code', 'total', 'pct'] and row[col] > 0:
                    sev_parts.append(f"{col}:{int(row[col])}")
            sev_str = ', '.join(sev_parts) if sev_parts else 'none'
            lines.append(f"  {row['code']}: {row['total']} ({row['pct']:.1f}%) [{sev_str}]")
        
        return '\n'.join(lines)


def create_synthetic_training_data(n_samples: int = 1000,
                                   n_codes: int = 20,
                                   imbalance_ratio: float = 10.0) -> Tuple[pd.DataFrame, List[List[str]]]:
    """
    Create synthetic training data for testing the pipeline.
    Simulates real IFML feature distributions and error code patterns.
    
    Args:
        n_samples: Number of samples to generate
        n_codes: Number of unique error codes
        imbalance_ratio: Ratio of most common to least common code
    
    Returns:
        Tuple of (features_df, code_lists)
    """
    np.random.seed(42)
    
    # Generate code frequencies (Zipf-like distribution)
    codes = [f'{6000 + i:04d}' if i < 10 else f'{7000 + i:04d}' if i < 15 else f'{8000 + i:04d}' 
             for i in range(n_codes)]
    code_probs = 1.0 / np.arange(1, n_codes + 1) ** 0.8
    code_probs /= code_probs.sum()
    
    features = []
    code_lists = []
    
    for i in range(n_samples):
        # Generate features
        is_cross_border = np.random.random() < 0.3
        has_intermediary = np.random.random() < 0.2
        has_iban = np.random.random() < 0.6
        
        feature = {
            'source_code': np.random.choice(['SWF', 'FED', 'CHI'], p=[0.6, 0.3, 0.1]),
            'incoming_msg_type': np.random.choice(['103', '202', '199'], p=[0.7, 0.2, 0.1]),
            'bank_operation_code': np.random.choice(['CRED', 'DEBT', 'SPAY'], p=[0.5, 0.3, 0.2]),
            'primary_currency': np.random.choice(['USD', 'EUR', 'GBP', 'JPY'], p=[0.5, 0.25, 0.15, 0.1]),
            'primary_amount': np.random.lognormal(9, 2),  # Log-normal amount distribution
            'is_cross_border': is_cross_border,
            'has_intermediary': has_intermediary,
            'has_beneficiary_bank': np.random.random() < 0.7,
            'originator_country': np.random.choice(['US', 'GB', 'DE', 'JP', 'CN']),
            'beneficiary_country': np.random.choice(['US', 'GB', 'DE', 'JP', 'CN']),
            'cdt_present': True,
            'cdt_has_account': has_iban,
            'cdt_account_type': 'IBAN' if has_iban else 'BBAN',
            'cdt_has_bic': np.random.random() < 0.8,
            'bnf_present': np.random.random() < 0.7,
            'bnf_has_bic': np.random.random() < 0.6,
            'cdt_address_lines': np.random.randint(0, 5),
            'bnf_address_lines': np.random.randint(0, 4),
        }
        features.append(feature)
        
        # Generate codes based on feature patterns
        sample_codes = []
        
        # Simulate realistic code triggers
        if not has_iban and np.random.random() < 0.7:
            sample_codes.append('8004')  # IBAN derivation error
        
        if is_cross_border and np.random.random() < 0.4:
            sample_codes.append('7452')  # Cross-border warning
        
        if not feature['cdt_has_bic'] and np.random.random() < 0.5:
            sample_codes.append('6007')  # BIC extraction info
        
        # Add some random codes
        n_random = np.random.poisson(0.5)
        if n_random > 0:
            random_codes = np.random.choice(codes, size=min(n_random, 3), 
                                           p=code_probs, replace=False)
            sample_codes.extend(random_codes.tolist())
        
        # Always add info code
        if np.random.random() < 0.8:
            sample_codes.append('602')
        
        code_lists.append(list(set(sample_codes)))
    
    return pd.DataFrame(features), code_lists


if __name__ == '__main__':
    # Test with synthetic data
    print("Testing data pipeline with synthetic data...")
    
    features_df, code_lists = create_synthetic_training_data(n_samples=500)
    
    # Create pipeline and load data
    pipeline = IFMLDataPipeline()
    
    # Simulate records
    for i, (_, row) in enumerate(features_df.iterrows()):
        record = PaymentRecord(
            transaction_id=f"TXN{i:06d}",
            request_features=row.to_dict(),
            response_codes=[{'code': c, 'severity': 'E' if c.startswith('8') else 'W' if c.startswith('7') else 'I', 'info': ''} for c in code_lists[i]],
            error_codes_only=code_lists[i],
            severity_map={c: 'E' if c.startswith('8') else 'W' if c.startswith('7') else 'I' for c in code_lists[i]}
        )
        pipeline.records.append(record)
    
    print(pipeline.describe())
    print("\n" + "=" * 40)
    
    # Create dataset
    X_raw, X_transformed, y_multilabel = pipeline.create_dataset(min_code_samples=10)
    
    print(f"\nDataset shapes:")
    print(f"  X_raw: {X_raw.shape}")
    print(f"  X_transformed: {X_transformed.shape}")
    print(f"  y_multilabel: {y_multilabel.shape}")
    
    print(f"\nLabel encoder summary:")
    print(pipeline.label_encoder.describe())
