#!/usr/bin/env python3
"""
================================================================================
ACE ERROR CODE DATA COLLECTOR & DECISION TREE BUILDER
================================================================================

PURPOSE:
    Collect features and actual error codes from IFML request/response pairs,
    then build decision trees to understand which features trigger each code.

USAGE:
    # Collect data from directory of IFML files
    python ace_data_collector.py collect --input-dir ./ifml_data --output dataset.csv
    
    # Build decision trees from collected data
    python ace_data_collector.py analyze --input dataset.csv --output-dir ./trees
    
    # Process single file pair
    python ace_data_collector.py single --request req.json --response resp.json
    
    # Show feature importance for a specific code
    python ace_data_collector.py explain --input dataset.csv --code 8894

WORKFLOW:
    1. Place IFML request/response pairs in a directory
       - Naming convention: *_request.json / *_response.json
       - Or: requests/*.json / responses/*.json subdirectories
    
    2. Run collect to extract features and actual codes
    
    3. Run analyze to build decision trees
    
    4. Review generated rules and feature importance

OUTPUT:
    - CSV dataset with all features and fired codes
    - Decision tree visualizations (text and optional graphviz)
    - Feature importance rankings per code
    - Suggested precondition rules

AUTHOR: ACE Pelican ML Team
VERSION: 1.0 (December 2025)
================================================================================
"""

import json
import csv
import os
import sys
import re
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from datetime import datetime

# Try to import ML libraries (optional)
try:
    import numpy as np
    from sklearn.tree import DecisionTreeClassifier, export_text
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("Note: scikit-learn not installed. Decision tree building disabled.")
    print("      Install with: pip install scikit-learn numpy")

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Try to import our parser and enhanced features
try:
    from ifml_parser_v3 import IFMLParser
    PARSER_VERSION = 'v3'
except ImportError:
    try:
        from ifml_parser_v2 import IFMLParser
        PARSER_VERSION = 'v2'
    except ImportError:
        print("ERROR: Could not import IFMLParser. Make sure ifml_parser_v3.py is in the same directory.")
        sys.exit(1)

# Try to import enhanced features
try:
    from enhanced_features import EnhancedFeatureExtractor
    ENHANCED_AVAILABLE = True
except ImportError:
    ENHANCED_AVAILABLE = False
    print("Note: enhanced_features.py not found. Using base parser only.")


# =============================================================================
# CONSTANTS
# =============================================================================

# Codes that require directory lookup - EXCLUDE from training
DIRECTORY_DEPENDENT_CODES = {
    # 8XXX directory-dependent
    '8035', '8036', '8464', '8465', '8472',
    # 9XXX directory-dependent
    '9004', '9005', '9007', '9008', '9013', '9023', '9032',
    '9475', '9476', '9477', '9478', '9479', '9480', '9481',
    '9485', '9486', '9490', '9961', '9985', '9986',
}

# Codes that are unpredictable (system/config/security)
UNPREDICTABLE_CODES = {
    '8003',  # File name derivation
    '8034',  # Forced Debit
    '8851',  # Field size (schema)
    '8853',  # Number format (schema)
    '8905',  # Hash mismatch
    '8906',  # Wrong flow
    '9439',  # No Pattern Found
    '9999',  # Field Repair (generic)
}

# All codes to exclude
EXCLUDED_CODES = DIRECTORY_DEPENDENT_CODES | UNPREDICTABLE_CODES

# Trainable 8XXX codes (deterministic from message)
TRAINABLE_8XXX = {
    '8001', '8004', '8005', '8006', '8007', '8022', '8023', '8024',
    '8025', '8026', '8027', '8028', '8029', '8030', '8033', '8124',
    '8852', '8892', '8894', '8895', '8896', '8897', '8898',
}

# Trainable 9XXX codes
TRAINABLE_9XXX = {
    '9000', '9002', '9006', '9009', '9012', '9014', '9015', '9017',
    '9018', '9019', '9020', '9021', '9022', '9024', '9025', '9028',
}

# Party suffixes
PARTY_SUFFIXES = ['ORGPTY', 'SNDBNK', 'DBTPTY', 'CDTPTY', 'INTBNK', 'BNFBNK', 'BNPPTY']

# Feature prefixes for parties
PARTY_PREFIXES = ['orig', 'send', 'dbt', 'cdt', 'intm', 'bnf', 'bnp']


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class CodeOccurrence:
    """Record of an error code occurrence."""
    code: str
    party: Optional[str]
    info: str
    
    @property
    def full_code(self) -> str:
        if self.party:
            return f"{self.code}_{self.party}"
        return self.code
    
    @property 
    def base_code(self) -> str:
        return self.code.split('_')[0]


@dataclass
class ProcessedRecord:
    """A single processed IFML record with features and codes."""
    transaction_id: str
    features: Dict[str, Any]
    actual_codes: List[str]  # Full codes like "8894_BNFBNK"
    base_codes: Set[str]     # Just base codes like "8894"
    source_file: str


# =============================================================================
# RESPONSE PARSER
# =============================================================================

class ACEResponseParser:
    """Parse ACE response to extract fired error codes."""
    
    def parse_file(self, filepath: str) -> Tuple[Optional[str], List[CodeOccurrence]]:
        """Parse response file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return self.parse(data)
    
    def parse(self, data: dict) -> Tuple[Optional[str], List[CodeOccurrence]]:
        """
        Parse response and extract error codes.
        Returns (transaction_id, list of CodeOccurrence).
        """
        message = self._find_message(data)
        
        if not message:
            # Try finding at File level
            file_node = self._find_file(data)
            if file_node:
                audit_trail = file_node.get('AuditTrail', {})
                transaction_id = None
            else:
                return None, []
        else:
            basic_payment = message.get('BasicPayment', {})
            transaction_id = (
                basic_payment.get('TransactionUID') or 
                basic_payment.get('TransactionID') or
                message.get('TransactionUID')
            )
            audit_trail = message.get('AuditTrail', {})
        
        # Extract codes from MsgStatus
        msg_status = audit_trail.get('MsgStatus', [])
        if isinstance(msg_status, dict):
            msg_status = [msg_status]
        
        codes = []
        for status in msg_status:
            code = status.get('Code', '')
            if not code:
                continue
            
            info = status.get('InformationalData', '')
            party = self._extract_party(info)
            
            codes.append(CodeOccurrence(
                code=code,
                party=party,
                info=info
            ))
        
        return transaction_id, codes
    
    def _find_message(self, data: dict) -> Optional[dict]:
        """Find Message node."""
        def safe_get(d, *keys):
            current = d
            for key in keys:
                if not isinstance(current, dict):
                    return None
                current = current.get(key)
                if current is None:
                    return None
            return current if isinstance(current, dict) else None
        
        patterns = [
            ('Response', 'IFML', 'File', 'Message'),
            ('IFML', 'File', 'Message'),
            ('File', 'Message'),
            ('Message',),
        ]
        
        for path in patterns:
            result = safe_get(data, *path)
            if result:
                return result
        
        # Try with transaction ID key
        if isinstance(data, dict) and len(data) == 1:
            key = list(data.keys())[0]
            nested = data[key]
            for path in patterns:
                result = safe_get(nested, *path)
                if result:
                    return result
        
        return None
    
    def _find_file(self, data: dict) -> Optional[dict]:
        """Find File node."""
        def safe_get(d, *keys):
            current = d
            for key in keys:
                if not isinstance(current, dict):
                    return None
                current = current.get(key)
            return current if isinstance(current, dict) else None
        
        patterns = [
            ('Response', 'IFML', 'File'),
            ('IFML', 'File'),
            ('File',),
        ]
        
        for path in patterns:
            result = safe_get(data, *path)
            if result:
                return result
        
        return None
    
    def _extract_party(self, info: str) -> Optional[str]:
        """Extract party suffix from informational data."""
        if not info:
            return None
        
        # Pattern: starts with party suffix
        for suffix in PARTY_SUFFIXES:
            if info.startswith(suffix):
                return suffix
        
        # Pattern: contains party suffix
        for suffix in PARTY_SUFFIXES:
            if suffix in info:
                return suffix
        
        return None


# =============================================================================
# DATA COLLECTOR
# =============================================================================

class ACEDataCollector:
    """Collect and process IFML data for analysis."""
    
    def __init__(self, use_enhanced: bool = True):
        self.parser = IFMLParser()
        self.response_parser = ACEResponseParser()
        self.records: List[ProcessedRecord] = []
        
        # Use enhanced features if available
        self.use_enhanced = use_enhanced and ENHANCED_AVAILABLE
        if self.use_enhanced:
            self.enhanced_extractor = EnhancedFeatureExtractor()
            print(f"Using enhanced feature extraction ({PARSER_VERSION})")
        else:
            self.enhanced_extractor = None
            print(f"Using base parser only ({PARSER_VERSION})")
    
    def _extract_features(self, ifml_data: dict) -> Dict[str, Any]:
        """Extract features using enhanced extractor if available."""
        if self.use_enhanced and self.enhanced_extractor:
            return self.enhanced_extractor.extract(ifml_data)
        else:
            features_obj = self.parser.parse(ifml_data)
            return self.parser.to_dict(features_obj)
    
    def process_combined_file(self, filepath: str) -> int:
        """
        Process a single JSON file containing multiple request/response pairs.
        
        Supports formats:
        1. Dict keyed by transaction ID: {"TX001": {"Request": {...}, "Response": {...}}, ...}
        2. Array of transactions: [{"Request": {...}, "Response": {...}}, ...]
        3. Wrapper with Transactions array: {"Transactions": [...]}
        
        Returns number of records processed.
        """
        try:
            with open(filepath) as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return 0
        
        processed = 0
        
        # Format 1: Dict keyed by transaction ID
        if isinstance(data, dict):
            # Check for wrapper keys
            if 'Transactions' in data:
                data = data['Transactions']
            elif 'transactions' in data:
                data = data['transactions']
            elif 'Data' in data:
                data = data['Data']
            
            if isinstance(data, dict):
                # Each key is a transaction ID
                for tx_id, tx_data in data.items():
                    if tx_id.startswith('_'):  # Skip metadata keys
                        continue
                    
                    if isinstance(tx_data, dict):
                        record = self._process_combined_entry(tx_id, tx_data, filepath)
                        if record:
                            self.records.append(record)
                            processed += 1
            
            elif isinstance(data, list):
                # Array of transactions
                for idx, tx_data in enumerate(data):
                    tx_id = self._extract_tx_id(tx_data, idx)
                    record = self._process_combined_entry(tx_id, tx_data, filepath)
                    if record:
                        self.records.append(record)
                        processed += 1
        
        # Format 2: Top-level array
        elif isinstance(data, list):
            for idx, tx_data in enumerate(data):
                tx_id = self._extract_tx_id(tx_data, idx)
                record = self._process_combined_entry(tx_id, tx_data, filepath)
                if record:
                    self.records.append(record)
                    processed += 1
        
        return processed
    
    def _extract_tx_id(self, tx_data: Any, fallback_idx: int) -> str:
        """Extract transaction ID from transaction data."""
        if isinstance(tx_data, dict):
            # Check common ID fields
            for key in ['TransactionID', 'TransactionUID', 'transaction_id', 'id', 'ID']:
                if key in tx_data:
                    return str(tx_data[key])
            
            # Check if single key (the key is the TX ID)
            if len(tx_data) == 1:
                return list(tx_data.keys())[0]
            
            # Try to find in nested Request
            request = tx_data.get('Request') or tx_data.get('request')
            if request:
                return self._extract_tx_id(request, fallback_idx)
        
        return f"TX_{fallback_idx:06d}"
    
    def _process_combined_entry(self, tx_id: str, tx_data: dict, 
                                 source_file: str) -> Optional[ProcessedRecord]:
        """Process a single request/response entry from a combined file."""
        try:
            # Find Request and Response in the data
            request_data = None
            response_data = None
            
            # Direct Request/Response keys
            if 'Request' in tx_data:
                request_data = tx_data['Request']
            elif 'request' in tx_data:
                request_data = tx_data['request']
            
            if 'Response' in tx_data:
                response_data = tx_data['Response']
            elif 'response' in tx_data:
                response_data = tx_data['response']
            
            # Sometimes the whole tx_data IS the request, with Response nested
            if not request_data and 'IFML' in tx_data:
                request_data = tx_data
            
            # Nested under tx_id key: {"TX001": {"Request": {...}}}
            if not request_data and len(tx_data) == 1:
                inner = list(tx_data.values())[0]
                if isinstance(inner, dict):
                    if 'Request' in inner:
                        request_data = inner['Request']
                        response_data = inner.get('Response')
                    elif 'IFML' in inner:
                        request_data = inner
            
            if not request_data:
                return None
            
            # Wrap request in expected format if needed
            if 'IFML' in request_data and tx_id not in request_data:
                request_data = {tx_id: {'Request': request_data}}
            elif 'Request' not in str(request_data)[:100]:
                request_data = {tx_id: {'Request': request_data}}
            
            # Parse request using enhanced extractor
            features = self._extract_features(request_data)
            
            # Parse response (if available)
            actual_codes = []
            base_codes = set()
            
            if response_data:
                # Wrap response in expected format if needed
                if 'IFML' in response_data or 'Response' not in str(response_data)[:100]:
                    response_data = {tx_id: {'Response': response_data}}
                
                _, code_occurrences = self.response_parser.parse(response_data)
                
                for occ in code_occurrences:
                    if occ.base_code not in EXCLUDED_CODES:
                        actual_codes.append(occ.full_code)
                        base_codes.add(occ.base_code)
            
            return ProcessedRecord(
                transaction_id=tx_id,
                features=features,
                actual_codes=actual_codes,
                base_codes=base_codes,
                source_file=f"{source_file}#{tx_id}"
            )
            
        except Exception as e:
            print(f"Error processing {tx_id} in {source_file}: {e}")
            return None
    
    def process_pair(self, request_path: str, response_path: str) -> Optional[ProcessedRecord]:
        """Process a request/response pair."""
        try:
            # Parse request
            with open(request_path) as f:
                request_data = json.load(f)
            
            # Use enhanced feature extraction
            features = self._extract_features(request_data)
            
            # Parse response
            tx_id, code_occurrences = self.response_parser.parse_file(response_path)
            
            # Filter out excluded codes
            filtered_codes = []
            base_codes = set()
            
            for occ in code_occurrences:
                if occ.base_code not in EXCLUDED_CODES:
                    filtered_codes.append(occ.full_code)
                    base_codes.add(occ.base_code)
            
            record = ProcessedRecord(
                transaction_id=tx_id or features.get('transaction_id', 'unknown'),
                features=features,
                actual_codes=filtered_codes,
                base_codes=base_codes,
                source_file=request_path
            )
            
            self.records.append(record)
            return record
            
        except Exception as e:
            print(f"Error processing {request_path}: {e}")
            return None
    
    def process_directory(self, input_dir: str, 
                         request_pattern: str = '*request*.json',
                         response_pattern: str = '*response*.json') -> int:
        """
        Process all IFML files in a directory.
        
        Supports multiple directory structures:
        1. Combined files: Each JSON contains multiple {TX: {Request, Response}} pairs
        2. Same directory with naming convention: *_request.json / *_response.json
        3. Subdirectories: requests/*.json / responses/*.json
        4. Transaction ID matching: {txid}_req.json / {txid}_resp.json
        
        The method auto-detects the format.
        """
        input_path = Path(input_dir)
        processed = 0
        
        # First, try processing all JSON files as combined files
        all_json_files = list(input_path.glob('*.json'))
        
        # Check if files contain combined request/response pairs
        combined_processed = 0
        for json_file in all_json_files:
            # Peek at the file to determine format
            try:
                with open(json_file) as f:
                    # Read first 2000 chars to detect format
                    peek = f.read(2000)
                    
                # Check if this looks like a combined file
                # (contains both Request and Response at similar levels)
                has_request = '"Request"' in peek or '"request"' in peek or '"IFML"' in peek
                has_response = '"Response"' in peek or '"response"' in peek or '"AuditTrail"' in peek
                
                if has_request and has_response:
                    count = self.process_combined_file(str(json_file))
                    if count > 0:
                        print(f"  Processed {count} transactions from {json_file.name}")
                        combined_processed += count
            except Exception as e:
                pass  # Not a combined file, try other methods
        
        if combined_processed > 0:
            print(f"Total from combined files: {combined_processed}")
            return combined_processed
        
        # Strategy 2: requests/responses subdirectories
        req_dir = input_path / 'requests'
        resp_dir = input_path / 'responses'
        
        if req_dir.exists() and resp_dir.exists():
            for req_file in req_dir.glob('*.json'):
                # Try to find matching response
                resp_file = resp_dir / req_file.name.replace('request', 'response')
                if not resp_file.exists():
                    resp_file = resp_dir / req_file.name
                
                if resp_file.exists():
                    if self.process_pair(str(req_file), str(resp_file)):
                        processed += 1
            return processed
        
        # Strategy 3: Same directory with naming patterns
        request_files = list(input_path.glob(request_pattern))
        
        for req_file in request_files:
            # Try different response file patterns
            base_name = req_file.stem
            possible_resp_names = [
                base_name.replace('request', 'response'),
                base_name.replace('req', 'resp'),
                base_name.replace('_request', '_response'),
                base_name + '_response',
            ]
            
            resp_file = None
            for resp_name in possible_resp_names:
                candidate = req_file.parent / f"{resp_name}.json"
                if candidate.exists():
                    resp_file = candidate
                    break
            
            if resp_file:
                if self.process_pair(str(req_file), str(resp_file)):
                    processed += 1
        
        if processed > 0:
            return processed
        
        # Strategy 4: All JSON files, match by transaction ID
        all_files = list(input_path.glob('*.json'))
        requests = {}
        responses = {}
        
        for f in all_files:
            try:
                with open(f) as fp:
                    data = json.load(fp)
                
                # Detect if request or response
                if 'Request' in str(data) or 'request' in f.name.lower():
                    # Extract transaction ID
                    if isinstance(data, dict) and len(data) == 1:
                        tx_id = list(data.keys())[0]
                        requests[tx_id] = str(f)
                elif 'Response' in str(data) or 'response' in f.name.lower():
                    if isinstance(data, dict) and len(data) == 1:
                        tx_id = list(data.keys())[0]
                        responses[tx_id] = str(f)
            except:
                pass
        
        # Match by transaction ID
        for tx_id, req_path in requests.items():
            if tx_id in responses:
                if self.process_pair(req_path, responses[tx_id]):
                    processed += 1
        
        return processed
    
    def to_dataframe_dict(self) -> Tuple[List[str], List[Dict]]:
        """
        Convert records to a format suitable for DataFrame/CSV.
        Returns (column_names, list of row dicts).
        """
        if not self.records:
            return [], []
        
        # Collect all feature names
        all_features = set()
        all_codes = set()
        
        for record in self.records:
            all_features.update(record.features.keys())
            all_codes.update(record.base_codes)
        
        # Filter to trainable codes only
        trainable_codes = sorted(all_codes - EXCLUDED_CODES)
        feature_names = sorted(all_features)
        
        # Build column names
        columns = ['transaction_id', 'source_file'] + feature_names + \
                  [f'code_{c}' for c in trainable_codes] + ['all_codes']
        
        # Build rows
        rows = []
        for record in self.records:
            row = {
                'transaction_id': record.transaction_id,
                'source_file': record.source_file,
            }
            
            # Add features
            for feat in feature_names:
                val = record.features.get(feat)
                # Convert to simple types for CSV
                if isinstance(val, bool):
                    row[feat] = 1 if val else 0
                elif val is None:
                    row[feat] = ''
                else:
                    row[feat] = val
            
            # Add code columns (1 if fired, 0 if not)
            for code in trainable_codes:
                row[f'code_{code}'] = 1 if code in record.base_codes else 0
            
            # Add all codes as string
            row['all_codes'] = ','.join(sorted(record.actual_codes))
            
            rows.append(row)
        
        return columns, rows
    
    def save_csv(self, output_path: str):
        """Save collected data to CSV."""
        columns, rows = self.to_dataframe_dict()
        
        if not rows:
            print("No data to save!")
            return
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"Saved {len(rows)} records to {output_path}")
        print(f"Columns: {len(columns)} ({len(columns) - len(rows[0])} features)")
    
    def get_code_statistics(self) -> Dict[str, int]:
        """Get count of each error code."""
        stats = defaultdict(int)
        for record in self.records:
            for code in record.base_codes:
                stats[code] += 1
        return dict(sorted(stats.items(), key=lambda x: -x[1]))


# =============================================================================
# DECISION TREE BUILDER
# =============================================================================

class DecisionTreeBuilder:
    """Build decision trees to understand error code triggers."""
    
    def __init__(self):
        self.trees: Dict[str, DecisionTreeClassifier] = {}
        self.feature_names: List[str] = []
        self.feature_importance: Dict[str, Dict[str, float]] = {}
    
    def load_csv(self, csv_path: str) -> Tuple[List[Dict], List[str], List[str]]:
        """Load data from CSV file."""
        rows = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            columns = reader.fieldnames
            for row in reader:
                rows.append(row)
        
        # Separate feature columns from code columns
        feature_cols = [c for c in columns if not c.startswith('code_') 
                       and c not in ['transaction_id', 'source_file', 'all_codes']]
        code_cols = [c for c in columns if c.startswith('code_')]
        
        return rows, feature_cols, code_cols
    
    def build_tree(self, rows: List[Dict], feature_cols: List[str], 
                   target_code: str, max_depth: int = 5,
                   min_samples_leaf: int = 5) -> Optional[DecisionTreeClassifier]:
        """Build a decision tree for a specific code."""
        if not ML_AVAILABLE:
            print("scikit-learn not available. Cannot build tree.")
            return None
        
        # Prepare data
        X = []
        y = []
        
        code_col = f'code_{target_code}'
        if code_col not in rows[0]:
            print(f"Code {target_code} not found in data")
            return None
        
        for row in rows:
            # Extract features (convert to numeric)
            features = []
            for col in feature_cols:
                val = row.get(col, 0)
                if val == '' or val is None:
                    features.append(0)
                elif isinstance(val, str):
                    try:
                        features.append(float(val))
                    except:
                        features.append(0)
                else:
                    features.append(float(val))
            
            X.append(features)
            y.append(int(row.get(code_col, 0)))
        
        X = np.array(X)
        y = np.array(y)
        
        # Check if we have enough positive samples
        positive_count = y.sum()
        if positive_count < min_samples_leaf:
            print(f"Not enough positive samples for {target_code}: {positive_count}")
            return None
        
        # Build tree
        tree = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight='balanced',
            random_state=42
        )
        
        tree.fit(X, y)
        
        self.trees[target_code] = tree
        self.feature_names = feature_cols
        
        # Calculate feature importance
        importance = dict(zip(feature_cols, tree.feature_importances_))
        self.feature_importance[target_code] = {
            k: v for k, v in sorted(importance.items(), key=lambda x: -x[1]) if v > 0
        }
        
        return tree
    
    def build_all_trees(self, csv_path: str, max_depth: int = 5,
                        min_samples: int = 10) -> Dict[str, str]:
        """Build trees for all codes with sufficient samples."""
        rows, feature_cols, code_cols = self.load_csv(csv_path)
        
        results = {}
        
        for code_col in code_cols:
            code = code_col.replace('code_', '')
            
            # Count positive samples
            positive = sum(1 for r in rows if int(r.get(code_col, 0)) == 1)
            
            if positive < min_samples:
                continue
            
            print(f"\nBuilding tree for {code} ({positive} positive samples)...")
            
            tree = self.build_tree(rows, feature_cols, code, max_depth)
            
            if tree:
                # Get tree rules
                rules = export_text(tree, feature_names=feature_cols, max_depth=max_depth)
                results[code] = rules
                
                # Show top features
                print(f"  Top features for {code}:")
                for feat, imp in list(self.feature_importance[code].items())[:5]:
                    print(f"    {feat}: {imp:.3f}")
        
        return results
    
    def extract_rules(self, code: str) -> List[Dict]:
        """
        Extract human-readable rules from a decision tree.
        Returns list of rule dicts with conditions and outcome.
        """
        if code not in self.trees:
            return []
        
        tree = self.trees[code]
        rules = []
        
        def recurse(node_id: int, conditions: List[str], depth: int = 0):
            tree_ = tree.tree_
            
            # Check if leaf
            if tree_.feature[node_id] == -2:  # Leaf node
                # Get class prediction
                values = tree_.value[node_id][0]
                predicted_class = np.argmax(values)
                confidence = values[predicted_class] / values.sum()
                
                if predicted_class == 1:  # Fires
                    rules.append({
                        'conditions': conditions.copy(),
                        'fires': True,
                        'confidence': confidence,
                        'samples': int(values.sum())
                    })
                return
            
            # Get split info
            feature_idx = tree_.feature[node_id]
            feature_name = self.feature_names[feature_idx]
            threshold = tree_.threshold[node_id]
            
            # Left child (<=)
            left_conditions = conditions + [f"{feature_name} <= {threshold:.2f}"]
            recurse(tree_.children_left[node_id], left_conditions, depth + 1)
            
            # Right child (>)
            right_conditions = conditions + [f"{feature_name} > {threshold:.2f}"]
            recurse(tree_.children_right[node_id], right_conditions, depth + 1)
        
        recurse(0, [])
        
        # Sort by confidence
        rules.sort(key=lambda x: -x['confidence'])
        
        return rules
    
    def generate_precondition_rules(self, code: str) -> Dict:
        """
        Generate precondition_rules.json format from decision tree.
        """
        rules = self.extract_rules(code)
        
        if not rules:
            return {}
        
        # Find the most confident rule
        best_rule = rules[0]
        
        require_true = []
        require_false = []
        
        for condition in best_rule['conditions']:
            # Parse condition
            match = re.match(r'(\w+)\s*(<=|>)\s*([\d.]+)', condition)
            if match:
                feature, op, val = match.groups()
                val = float(val)
                
                # For boolean features (0/1)
                if val == 0.5:
                    if op == '>':
                        require_true.append(feature)
                    else:
                        require_false.append(feature)
        
        return {
            'require_true': require_true,
            'require_false': require_false,
            'confidence': best_rule['confidence'],
            'samples': best_rule['samples'],
            'full_conditions': best_rule['conditions']
        }


# =============================================================================
# ANALYSIS UTILITIES
# =============================================================================

def analyze_feature_correlation(csv_path: str, target_code: str):
    """Analyze which features correlate with a specific code."""
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    code_col = f'code_{target_code}'
    if code_col not in rows[0]:
        print(f"Code {target_code} not found")
        return
    
    # Separate positive and negative samples
    positive = [r for r in rows if int(r.get(code_col, 0)) == 1]
    negative = [r for r in rows if int(r.get(code_col, 0)) == 0]
    
    print(f"\nAnalysis for code {target_code}:")
    print(f"  Positive samples: {len(positive)}")
    print(f"  Negative samples: {len(negative)}")
    print(f"  Ratio: {len(positive)/(len(positive)+len(negative))*100:.1f}%")
    
    if not positive:
        return
    
    # Find distinguishing features
    feature_cols = [c for c in rows[0].keys() 
                   if not c.startswith('code_') 
                   and c not in ['transaction_id', 'source_file', 'all_codes']]
    
    print(f"\n  Features that differ between positive/negative:")
    
    differences = []
    for col in feature_cols:
        try:
            pos_vals = [float(r.get(col, 0) or 0) for r in positive]
            neg_vals = [float(r.get(col, 0) or 0) for r in negative]
            
            pos_mean = sum(pos_vals) / len(pos_vals) if pos_vals else 0
            neg_mean = sum(neg_vals) / len(neg_vals) if neg_vals else 0
            
            diff = abs(pos_mean - neg_mean)
            if diff > 0.1:  # Significant difference
                differences.append((col, pos_mean, neg_mean, diff))
        except:
            pass
    
    # Sort by difference
    differences.sort(key=lambda x: -x[3])
    
    for col, pos_mean, neg_mean, diff in differences[:20]:
        print(f"    {col}: pos={pos_mean:.2f}, neg={neg_mean:.2f}, diff={diff:.2f}")


def show_code_summary(csv_path: str):
    """Show summary of codes in dataset."""
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    code_cols = [c for c in rows[0].keys() if c.startswith('code_')]
    
    print(f"\nCode Summary ({len(rows)} total records):")
    print("-" * 50)
    
    stats = []
    for col in code_cols:
        code = col.replace('code_', '')
        count = sum(1 for r in rows if int(r.get(col, 0)) == 1)
        if count > 0:
            stats.append((code, count, count/len(rows)*100))
    
    stats.sort(key=lambda x: -x[1])
    
    for code, count, pct in stats:
        trainable = "✓" if code in TRAINABLE_8XXX or code in TRAINABLE_9XXX else "○"
        print(f"  {trainable} {code}: {count:5d} ({pct:5.1f}%)")
    
    print("-" * 50)
    print("✓ = Trainable (deterministic)")
    print("○ = May need directory lookup")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='ACE Error Code Data Collector & Decision Tree Builder',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect data from IFML files
  python ace_data_collector.py collect -i ./ifml_data -o dataset.csv
  
  # Build decision trees
  python ace_data_collector.py analyze -i dataset.csv -o ./trees
  
  # Analyze specific code
  python ace_data_collector.py explain -i dataset.csv -c 8894
  
  # Show dataset summary
  python ace_data_collector.py summary -i dataset.csv
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command')
    
    # Collect command
    collect_parser = subparsers.add_parser('collect', help='Collect data from IFML files')
    collect_parser.add_argument('-i', '--input-dir', required=True,
                               help='Directory containing IFML files')
    collect_parser.add_argument('-o', '--output', required=True,
                               help='Output CSV file')
    collect_parser.add_argument('--combined', action='store_true',
                               help='Treat each file as containing multiple request/response pairs')
    collect_parser.add_argument('--request-pattern', default='*request*.json',
                               help='Glob pattern for request files (ignored if --combined)')
    collect_parser.add_argument('--response-pattern', default='*response*.json',
                               help='Glob pattern for response files (ignored if --combined)')
    collect_parser.add_argument('--recursive', '-r', action='store_true',
                               help='Recursively process subdirectories')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Build decision trees')
    analyze_parser.add_argument('-i', '--input', required=True,
                               help='Input CSV file from collect')
    analyze_parser.add_argument('-o', '--output-dir', default='./trees',
                               help='Output directory for trees')
    analyze_parser.add_argument('--max-depth', type=int, default=5,
                               help='Maximum tree depth')
    analyze_parser.add_argument('--min-samples', type=int, default=10,
                               help='Minimum samples to build tree')
    
    # Explain command
    explain_parser = subparsers.add_parser('explain', help='Explain specific code')
    explain_parser.add_argument('-i', '--input', required=True,
                               help='Input CSV file')
    explain_parser.add_argument('-c', '--code', required=True,
                               help='Error code to explain')
    explain_parser.add_argument('--max-depth', type=int, default=5,
                               help='Maximum tree depth')
    
    # Summary command
    summary_parser = subparsers.add_parser('summary', help='Show dataset summary')
    summary_parser.add_argument('-i', '--input', required=True,
                               help='Input CSV file')
    
    # Single command
    single_parser = subparsers.add_parser('single', help='Process single file pair')
    single_parser.add_argument('--request', required=True,
                              help='Request JSON file')
    single_parser.add_argument('--response', required=True,
                              help='Response JSON file')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Execute command
    if args.command == 'collect':
        print(f"Collecting data from {args.input_dir}...")
        collector = ACEDataCollector()
        
        input_path = Path(args.input_dir)
        
        if args.combined or getattr(args, 'recursive', False):
            # Process as combined files
            glob_pattern = '**/*.json' if getattr(args, 'recursive', False) else '*.json'
            json_files = list(input_path.glob(glob_pattern))
            
            total_count = 0
            for json_file in json_files:
                count = collector.process_combined_file(str(json_file))
                if count > 0:
                    print(f"  {json_file.name}: {count} transactions")
                    total_count += count
            
            print(f"\nTotal processed: {total_count} transactions from {len(json_files)} files")
            count = total_count
        else:
            count = collector.process_directory(args.input_dir, 
                                               args.request_pattern,
                                               args.response_pattern)
            print(f"Processed {count} file pairs")
        
        if count > 0:
            print("\nCode statistics:")
            for code, cnt in list(collector.get_code_statistics().items())[:20]:
                print(f"  {code}: {cnt}")
            if len(collector.get_code_statistics()) > 20:
                print(f"  ... and {len(collector.get_code_statistics()) - 20} more codes")
            
            collector.save_csv(args.output)
    
    elif args.command == 'analyze':
        if not ML_AVAILABLE:
            print("ERROR: scikit-learn required for analysis")
            print("Install with: pip install scikit-learn numpy")
            return
        
        builder = DecisionTreeBuilder()
        results = builder.build_all_trees(args.input, args.max_depth, args.min_samples)
        
        # Save results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for code, rules in results.items():
            rule_file = output_dir / f"tree_{code}.txt"
            with open(rule_file, 'w') as f:
                f.write(f"Decision Tree for {code}\n")
                f.write("=" * 50 + "\n\n")
                f.write(rules)
                
                # Add precondition format
                f.write("\n\nSuggested Precondition Rules:\n")
                f.write("-" * 50 + "\n")
                precond = builder.generate_precondition_rules(code)
                f.write(json.dumps(precond, indent=2))
        
        print(f"\nSaved {len(results)} tree files to {args.output_dir}")
    
    elif args.command == 'explain':
        if ML_AVAILABLE:
            builder = DecisionTreeBuilder()
            rows, feature_cols, code_cols = builder.load_csv(args.input)
            
            tree = builder.build_tree(rows, feature_cols, args.code, args.max_depth)
            
            if tree:
                print(f"\nDecision Tree for {args.code}:")
                print("=" * 50)
                print(export_text(tree, feature_names=feature_cols, max_depth=args.max_depth))
                
                print("\nTop Features by Importance:")
                for feat, imp in list(builder.feature_importance[args.code].items())[:10]:
                    print(f"  {feat}: {imp:.3f}")
                
                print("\nSuggested Precondition Rules:")
                precond = builder.generate_precondition_rules(args.code)
                print(json.dumps(precond, indent=2))
        
        # Also show correlation analysis
        analyze_feature_correlation(args.input, args.code)
    
    elif args.command == 'summary':
        show_code_summary(args.input)
    
    elif args.command == 'single':
        collector = ACEDataCollector()
        record = collector.process_pair(args.request, args.response)
        
        if record:
            print(f"\nTransaction: {record.transaction_id}")
            print(f"Fired Codes: {record.actual_codes}")
            print(f"\nKey Features:")
            for key, val in sorted(record.features.items()):
                if val and val != 0 and val != '' and val != False:
                    print(f"  {key}: {val}")


if __name__ == '__main__':
    main()
