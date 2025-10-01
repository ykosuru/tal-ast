"""
Enhanced Repair Learning Model
===================================

Learns from repair taxonomy and before/after transformations with improved
repair context handling, validation, and prediction capabilities.

Usage:
    python nn_stp_model.py train --input repair1.json --epochs 50
    python nn_stp_model.py predict --input test.json --model models/ace_model.pt
"""

# Standard library imports for core Python functionality
import torch  # PyTorch deep learning framework
import torch.nn as nn  # Neural network modules
import torch.nn.functional as F  # Functional interface for neural network operations
from torch.utils.data import Dataset, DataLoader  # Data handling utilities
import json  # JSON file parsing and writing
import argparse  # Command-line argument parsing
import os  # Operating system interface for file/directory operations
from typing import Dict, List, Optional, Tuple, Set  # Type hints for better code documentation
import numpy as np  # Numerical computing library
from dataclasses import dataclass, asdict  # Simplified class definitions
from pathlib import Path  # Object-oriented filesystem paths
import pickle  # Python object serialization
import re  # Regular expression operations
from collections import defaultdict, Counter  # Specialized container datatypes
import logging  # Logging facility for Python
from datetime import datetime  # Date and time handling

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

# Setup logging with timestamp, module name, level, and message
logging.basicConfig(
    level=logging.INFO,  # Set minimum logging level to INFO
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'  # Format for log messages
)
# Create a logger instance for this module
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION DATA CLASS
# ============================================================================

@dataclass  # Decorator to create a data class with automatic __init__, __repr__, etc.
class Config:
    """Model configuration - centralizes all hyperparameters and settings"""
    
    # Model architecture parameters
    embedding_dim: int = 128  # Dimension of embedding vectors for characters/fields/repairs
    hidden_dim: int = 256  # Hidden dimension for LSTM and transformer layers
    num_lstm_layers: int = 2  # Number of stacked LSTM layers
    dropout: float = 0.3  # Dropout probability for regularization
    num_attention_heads: int = 8  # Number of attention heads in transformer
    repair_attention_heads: int = 4  # Number of attention heads for repair-field interaction
    
    # Training parameters
    batch_size: int = 16  # Number of samples per training batch
    learning_rate: float = 0.001  # Initial learning rate for optimizer
    num_epochs: int = 100  # Maximum number of training epochs
    weight_decay: float = 1e-4  # L2 regularization factor
    grad_clip: float = 1.0  # Maximum gradient norm for clipping
    
    # Data parameters
    max_text_length: int = 128  # Maximum character length for field values
    max_fields: int = 100  # Maximum number of fields per transaction
    max_repairs: int = 20  # Maximum number of repairs per transaction
    train_split: float = 0.8  # Proportion of data for training
    val_split: float = 0.1  # Proportion of data for validation
    
    # Loss function weights - controls importance of each loss component
    char_loss_weight: float = 1.0  # Weight for character-level prediction loss
    change_detection_weight: float = 0.1  # Weight for field change detection loss
    repair_prediction_weight: float = 0.2  # Weight for repair prediction loss
    
    # File paths for saving/loading model components
    model_save_path: str = "./models/ace_model.pt"  # Path to save trained model
    vocab_save_path: str = "./models/ace_vocab.pkl"  # Path to save character vocabulary
    field_vocab_save_path: str = "./models/ace_field_vocab.pkl"  # Path to save field vocabulary
    repair_taxonomy_path: str = "./models/ace_repair_taxonomy.json"  # Path to save repair taxonomy
    validator_rules_path: str = "./models/ace_validator_rules.json"  # Path to save validation rules
    
    def save(self, path: str):
        """Save configuration to JSON file"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Write config as JSON with indentation for readability
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: str):
        """Load configuration from JSON file"""
        # Read JSON file and create Config instance
        with open(path, 'r') as f:
            return cls(**json.load(f))


# ============================================================================
# VALIDATION RULES FOR REPAIR OUTPUTS
# ============================================================================

class RepairValidator:
    """Validates repair outputs based on business rules and format requirements"""
    
    def __init__(self):
        # ISO 3166-1 alpha-2 country codes - used to validate country fields
        # This is a subset for demonstration; in production, use complete list
        self.valid_iso_codes = {
            'US', 'GB', 'DE', 'FR', 'IT', 'ES', 'NL', 'BE', 'CH', 'AT',
            'SE', 'NO', 'DK', 'FI', 'PT', 'IE', 'LU', 'GR', 'PL', 'CZ',
            'HU', 'RO', 'BG', 'HR', 'SI', 'SK', 'EE', 'LV', 'LT', 'MT',
            'CY', 'CA', 'MX', 'BR', 'AR', 'CL', 'CO', 'PE', 'JP', 'CN',
            'IN', 'AU', 'NZ', 'SG', 'HK', 'KR', 'TW', 'TH', 'MY', 'ID'
        }
        
        # BIC (Bank Identifier Code) validation pattern
        # Format: 4 letters (bank code) + 2 letters (country) + 2 alphanumeric (location) + optional 3 alphanumeric (branch)
        self.bic_pattern = re.compile(r'^[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}([A-Z0-9]{3})?$')
        
        # IBAN (International Bank Account Number) validation patterns by country
        # Each country has specific length and format requirements
        self.iban_patterns = {
            'DE': re.compile(r'^DE\d{2}\d{8}\d{10}$'),  # Germany: DE + 2 check + 8 bank + 10 account
            'GB': re.compile(r'^GB\d{2}[A-Z]{4}\d{6}\d{8}$'),  # UK: GB + 2 check + 4 bank + 6 sort + 8 account
            'FR': re.compile(r'^FR\d{2}\d{5}\d{5}[A-Z0-9]{11}\d{2}$'),  # France format
            'US': re.compile(r'^US\d{2}\d{4}\d{13}$'),  # US format (hypothetical)
        }
        
        # Map field types to their validation functions
        self.field_validators = {
            'iso_country': self.validate_iso_country,
            'bic': self.validate_bic,
            'iban': self.validate_iban,
            'amount': self.validate_amount,
            'date': self.validate_date,
            'postal_code': self.validate_postal_code
        }
    
    def validate_iso_country(self, value: str) -> bool:
        """
        Validate ISO country code
        Returns True if value is a valid 2-letter ISO country code
        """
        return value.upper() in self.valid_iso_codes
    
    def validate_bic(self, value: str) -> bool:
        """
        Validate BIC/SWIFT code format
        Returns True if value matches BIC pattern (8 or 11 characters)
        """
        if not value:  # Empty value is invalid
            return False
        return bool(self.bic_pattern.match(value.upper()))  # Check against regex pattern
    
    def validate_iban(self, value: str) -> bool:
        """
        Validate IBAN format based on country-specific patterns
        Returns True if IBAN format is valid for the country
        """
        if not value or len(value) < 15:  # Basic length check
            return False
        country_code = value[:2].upper()  # Extract country code (first 2 chars)
        if country_code in self.iban_patterns:  # If we have specific pattern for country
            return bool(self.iban_patterns[country_code].match(value.upper()))
        # Generic validation for other countries - just check length
        return len(value) >= 15 and len(value) <= 34
    
    def validate_amount(self, value: str) -> bool:
        """
        Validate amount format (should be parseable as float)
        Handles both comma and dot as decimal separator
        """
        try:
            float(value.replace(',', '.'))  # Try to convert to float
            return True
        except:
            return False
    
    def validate_date(self, value: str) -> bool:
        """
        Validate date format - accepts ISO 8601 and other common formats
        Returns True if value matches any accepted date pattern
        """
        patterns = [
            r'^\d{4}-\d{2}-\d{2}$',  # YYYY-MM-DD
            r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}',  # ISO 8601 with time
            r'^\d{8}$'  # YYYYMMDD
        ]
        return any(re.match(p, value) for p in patterns)
    
    def validate_postal_code(self, value: str, country: str = None) -> bool:
        """
        Validate postal code format
        Simple validation - can be extended with country-specific rules
        """
        if not value:
            return False
        # Basic check: postal codes are typically 3-10 characters
        return len(value) >= 3 and len(value) <= 10
    
    def infer_field_type(self, field_name: str) -> Optional[str]:
        """
        Infer validation type from field name using keywords
        Returns the type of validation to apply based on field name patterns
        """
        field_lower = field_name.lower()  # Convert to lowercase for comparison
        
        # Check for country-related fields
        if 'country' in field_lower or 'ctry' in field_lower:
            return 'iso_country'
        # Check for BIC/SWIFT fields
        elif 'bic' in field_lower or 'swift' in field_lower:
            return 'bic'
        # Check for IBAN/account fields
        elif 'iban' in field_lower or 'acct' in field_lower:
            return 'iban'
        # Check for amount fields
        elif 'amt' in field_lower or 'amount' in field_lower:
            return 'amount'
        # Check for date fields
        elif 'date' in field_lower or 'dt' in field_lower:
            return 'date'
        # Check for postal/zip code fields
        elif 'postal' in field_lower or 'zip' in field_lower:
            return 'postal_code'
        
        return None  # No specific validation type identified
    
    def validate_field(self, field_name: str, value: str) -> Tuple[bool, str]:
        """
        Validate a field value based on inferred type
        Returns (is_valid, message) tuple
        """
        # Determine validation type from field name
        field_type = self.infer_field_type(field_name)
        
        # Apply appropriate validation if type was identified
        if field_type and field_type in self.field_validators:
            is_valid = self.field_validators[field_type](value)
            # Create descriptive message based on validation result
            message = f"Valid {field_type}" if is_valid else f"Invalid {field_type}: {value}"
            return is_valid, message
        
        # No validation rules apply - consider valid by default
        return True, "No validation rules"
    
    def validate_repairs(self, before: Dict, after: Dict, repairs: List[str]) -> Dict:
        """
        Validate all repairs applied to a transaction
        Compares before/after states and validates changed fields
        Returns detailed validation results dictionary
        """
        # Initialize validation results structure
        validation_results = {
            'valid': True,  # Overall validation status
            'errors': [],  # List of validation errors
            'warnings': [],  # List of validation warnings
            'field_validations': {}  # Per-field validation details
        }
        
        # Check each field in the after state
        for field_name, after_value in after.items():
            before_value = before.get(field_name, '')  # Get before value (empty if new field)
            
            # Only validate fields that changed
            if after_value != before_value:
                # Validate the new value
                is_valid, message = self.validate_field(field_name, after_value)
                # Store validation details for this field
                validation_results['field_validations'][field_name] = {
                    'valid': is_valid,
                    'message': message,
                    'before': before_value,
                    'after': after_value
                }
                
                # Update overall validation status if field is invalid
                if not is_valid:
                    validation_results['valid'] = False
                    validation_results['errors'].append(
                        f"Field {field_name}: {message}"
                    )
        
        return validation_results


# ============================================================================
# REPAIR TAXONOMY WITH RELATIONSHIP TRACKING
# ============================================================================

class RepairTaxonomy:
    """Enhanced repair taxonomy with relationship tracking between repairs and fields"""
    
    def __init__(self):
        # Main repair information storage
        # Format: repair_id -> {code, field, text, count, related_fields}
        self.repairs = {}
        
        # Bidirectional mapping between repair IDs and indices for neural network
        self.repair2idx = {'<NONE>': 0, '<UNK>': 1}  # Special tokens
        self.idx2repair = {0: '<NONE>', 1: '<UNK>'}
        self.next_idx = 2  # Next available index
        
        # Track relationships between repairs and fields
        self.repair_dependencies = defaultdict(set)  # repair_id -> set of dependent repairs
        self.field_repair_map = defaultdict(set)  # field -> set of repair_ids that affect it
        self.repair_field_map = defaultdict(set)  # repair_id -> set of fields it affects
    
    def add_repair(self, repair_id: str, code: str, field: str, text: str):
        """
        Register a repair from taxonomy
        Tracks repair information and field relationships
        """
        # Add repair to index mapping if new
        if repair_id not in self.repair2idx:
            self.repair2idx[repair_id] = self.next_idx
            self.idx2repair[self.next_idx] = repair_id
            self.next_idx += 1
        
        # Initialize repair information if new
        if repair_id not in self.repairs:
            self.repairs[repair_id] = {
                'code': code,  # Repair category (I/W/E/R)
                'field': field,  # Primary field affected
                'text': text,  # Description of repair
                'count': 0,  # Number of occurrences
                'related_fields': set()  # All fields this repair has affected
            }
        
        # Update repair statistics
        self.repairs[repair_id]['count'] += 1
        self.repairs[repair_id]['related_fields'].add(field)
        
        # Track bidirectional field-repair relationships
        if field:
            self.field_repair_map[field].add(repair_id)  # Field -> repairs
            self.repair_field_map[repair_id].add(field)  # Repair -> fields
    
    def add_repair_dependency(self, repair_id: str, depends_on: str):
        """
        Add dependency between repairs
        Some repairs may require other repairs to be applied first
        """
        self.repair_dependencies[repair_id].add(depends_on)
    
    def get_repair_description(self, repair_id: str) -> str:
        """
        Get human-readable description of a repair
        Format: [Category] Description (field: FieldName)
        """
        if repair_id in self.repairs:
            r = self.repairs[repair_id]
            return f"[{r['code']}] {r['text']} (field: {r['field']})"
        return "Unknown repair"
    
    def get_repairs_for_field(self, field: str) -> List[str]:
        """Get all repair IDs that can affect a specific field"""
        return list(self.field_repair_map.get(field, []))
    
    def get_fields_for_repair(self, repair_id: str) -> List[str]:
        """Get all fields that can be affected by a specific repair"""
        return list(self.repair_field_map.get(repair_id, []))
    
    def get_repair_category(self, repair_id: str) -> str:
        """
        Get repair category code (I/W/E/R)
        I=Information, W=Warning, E=Error, R=Repair
        """
        if repair_id in self.repairs:
            return self.repairs[repair_id]['code']
        return 'I'  # Default to Information
    
    def get_idx(self, repair_id: str) -> int:
        """Convert repair ID to index for neural network"""
        return self.repair2idx.get(repair_id, 1)  # Return UNK index if not found
    
    def __len__(self):
        """Return total number of repairs in taxonomy"""
        return len(self.repair2idx)
    
    def save(self, path: str):
        """Save taxonomy to JSON file"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            # Convert sets to lists for JSON serialization
            json.dump({
                'repairs': {k: {**v, 'related_fields': list(v['related_fields'])} 
                          for k, v in self.repairs.items()},
                'repair2idx': self.repair2idx,
                'idx2repair': {int(k): v for k, v in self.idx2repair.items()},
                'repair_dependencies': {k: list(v) for k, v in self.repair_dependencies.items()},
                'field_repair_map': {k: list(v) for k, v in self.field_repair_map.items()},
                'repair_field_map': {k: list(v) for k, v in self.repair_field_map.items()}
            }, f, indent=2)
    
    @classmethod
    def load(cls, path: str):
        """Load taxonomy from JSON file"""
        with open(path, 'r') as f:
            data = json.load(f)
        taxonomy = cls()
        # Restore sets from lists
        taxonomy.repairs = {k: {**v, 'related_fields': set(v.get('related_fields', []))} 
                           for k, v in data['repairs'].items()}
        taxonomy.repair2idx = data['repair2idx']
        taxonomy.idx2repair = {int(k): v for k, v in data['idx2repair'].items()}
        taxonomy.repair_dependencies = defaultdict(set, {k: set(v) for k, v in data.get('repair_dependencies', {}).items()})
        taxonomy.field_repair_map = defaultdict(set, {k: set(v) for k, v in data.get('field_repair_map', {}).items()})
        taxonomy.repair_field_map = defaultdict(set, {k: set(v) for k, v in data.get('repair_field_map', {}).items()})
        taxonomy.next_idx = max(taxonomy.idx2repair.keys()) + 1
        return taxonomy


# ============================================================================
# CHARACTER-LEVEL VOCABULARY WITH SPECIAL TOKENS
# ============================================================================

class Vocabulary:
    """Enhanced character-level vocabulary with special token handling"""
    
    def __init__(self):
        # Initialize special tokens with reserved indices
        self.char2idx = {
            '<PAD>': 0,      # Padding token for sequence alignment
            '<UNK>': 1,      # Unknown character token
            '<MISSING>': 2,  # Token for missing/null values
            '<START>': 3,    # Sequence start marker
            '<END>': 4,      # Sequence end marker
            '<MASK>': 5      # Masking token for future use
        }
        # Create reverse mapping
        self.idx2char = {v: k for k, v in self.char2idx.items()}
        self.next_idx = 6  # Next available index
        
        # Add common special characters that appear in financial data
        for char in '.-_/@#$%&*()[]{}|\\:;"\',<>?+=~`':
            self.add_char(char)
    
    def add_char(self, char: str):
        """Add a new character to vocabulary if not already present"""
        if char not in self.char2idx:
            self.char2idx[char] = self.next_idx
            self.idx2char[self.next_idx] = char
            self.next_idx += 1
    
    def build_from_texts(self, texts: List[str]):
        """Build vocabulary from a list of text strings"""
        for text in texts:
            if text:  # Skip empty texts
                # Convert to uppercase for consistency and add each character
                for char in str(text).upper():
                    self.add_char(char)
    
    def encode(self, text: Optional[str], max_length: int, add_special: bool = True) -> List[int]:
        """
        Encode text string to list of character indices
        Args:
            text: Input text to encode
            max_length: Maximum sequence length
            add_special: Whether to add START/END tokens
        Returns:
            List of character indices
        """
        # Handle missing/null values
        if not text:
            return [self.char2idx['<MISSING>']] * max_length
        
        # Convert to uppercase and truncate if needed
        # Reserve space for special tokens if add_special is True
        text = str(text).upper()[:max_length-2 if add_special else max_length]
        
        encoded = []
        # Add START token if requested
        if add_special:
            encoded.append(self.char2idx['<START>'])
        
        # Encode each character
        encoded.extend([self.char2idx.get(char, self.char2idx['<UNK>']) for char in text])
        
        # Add END token if requested and there's space
        if add_special and len(encoded) < max_length:
            encoded.append(self.char2idx['<END>'])
        
        # Pad to max_length with PAD tokens
        encoded += [self.char2idx['<PAD>']] * (max_length - len(encoded))
        
        # Ensure we don't exceed max_length
        return encoded[:max_length]
    
    def decode(self, indices: List[int]) -> str:
        """
        Decode list of indices back to text string
        Skips special tokens in the output
        """
        chars = []
        for idx in indices:
            char = self.idx2char.get(idx, '<UNK>')
            # Skip special tokens in decoded output
            if char in ['<PAD>', '<MISSING>', '<START>', '<END>', '<MASK>']:
                continue
            if char != '<UNK>':
                chars.append(char)
        return ''.join(chars)
    
    def __len__(self):
        """Return vocabulary size"""
        return len(self.char2idx)
    
    def save(self, path: str):
        """Save vocabulary to pickle file"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({'char2idx': self.char2idx, 'idx2char': self.idx2char}, f)
    
    @classmethod
    def load(cls, path: str):
        """Load vocabulary from pickle file"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        vocab = cls()
        vocab.char2idx = data['char2idx']
        vocab.idx2char = data['idx2char']
        vocab.next_idx = max(vocab.idx2char.keys()) + 1
        return vocab


# ============================================================================
# FIELD VOCABULARY WITH STATISTICS TRACKING
# ============================================================================

def _field_stats_factory():
    """Factory function for field statistics default values"""
    return {'count': 0, 'changes': 0}

class FieldVocabulary:
    """Vocabulary for field names with change statistics"""
    
    def __init__(self):
        # Initialize with special tokens
        self.field2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2field = {0: '<PAD>', 1: '<UNK>'}
        self.next_idx = 2
        
        # Track statistics for each field
        # Helps model learn which fields are more likely to change
        # Use a function instead of lambda for pickling compatibility
        self.field_stats = defaultdict(_field_stats_factory)
    
    def add_field(self, field: str, is_changed: bool = False):
        """
        Add field to vocabulary and update statistics
        Args:
            field: Field name to add
            is_changed: Whether this field changed in the current transaction
        """
        # Add to vocabulary if new
        if field not in self.field2idx:
            self.field2idx[field] = self.next_idx
            self.idx2field[self.next_idx] = field
            self.next_idx += 1
        
        # Update field statistics
        self.field_stats[field]['count'] += 1
        if is_changed:
            self.field_stats[field]['changes'] += 1
    
    def get_idx(self, field: str) -> int:
        """Get index for field name, return UNK index if not found"""
        return self.field2idx.get(field, self.field2idx['<UNK>'])
    
    def get_change_probability(self, field: str) -> float:
        """
        Calculate historical probability that a field changes
        Useful for initializing model biases
        """
        if field in self.field_stats and self.field_stats[field]['count'] > 0:
            return self.field_stats[field]['changes'] / self.field_stats[field]['count']
        return 0.0
    
    def __len__(self):
        """Return vocabulary size"""
        return len(self.field2idx)
    
    def save(self, path: str):
        """Save vocabulary and statistics to pickle file"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'field2idx': self.field2idx, 
                'idx2field': self.idx2field,
                'field_stats': dict(self.field_stats)
            }, f)
    
    @classmethod
    def load(cls, path: str):
        """Load vocabulary and statistics from pickle file"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        vocab = cls()
        vocab.field2idx = data['field2idx']
        vocab.idx2field = data['idx2field']
        # Restore defaultdict with factory function instead of lambda
        vocab.field_stats = defaultdict(_field_stats_factory, 
                                       data.get('field_stats', {}))
        vocab.next_idx = max(vocab.idx2field.keys()) + 1
        return vocab


# ============================================================================
# DATA PROCESSING UTILITIES
# ============================================================================

def flatten_json(data: Dict, parent_key: str = '', sep: str = '.') -> Dict:
    """
    Flatten nested JSON structure into flat key-value pairs
    Example: {'a': {'b': 1}} -> {'a.b': 1}
    
    Args:
        data: Nested dictionary to flatten
        parent_key: Prefix for keys (used in recursion)
        sep: Separator between nested keys
    Returns:
        Flattened dictionary
    """
    items = []
    # Iterate through each key-value pair
    for k, v in data.items():
        # Create new key with parent prefix if it exists
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, dict):
            # Recursively flatten nested dictionaries
            items.extend(flatten_json(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            # Handle lists by adding index to key
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    # Recursively flatten dict items in list
                    items.extend(flatten_json(item, f"{new_key}[{i}]", sep=sep).items())
                else:
                    # Add simple list items with index
                    items.append((f"{new_key}[{i}]", str(item)))
        else:
            # Add simple key-value pairs, converting None to empty string
            items.append((new_key, str(v) if v is not None else ''))
    return dict(items)


def normalize_keys(data):
    """
    Recursively normalize all keys in a nested structure to lowercase
    Ensures consistent key naming across different data sources
    """
    if isinstance(data, dict):
        # Normalize dictionary keys
        return {k.lower() if isinstance(k, str) else k: normalize_keys(v) 
                for k, v in data.items()}
    elif isinstance(data, list):
        # Recursively process list items
        return [normalize_keys(item) for item in data]
    else:
        # Return non-dict/list values as-is
        return data


# ============================================================================
# DATA PARSER WITH VALIDATION
# ============================================================================

class DataParser:
    """Enhanced data parser with validation and error handling"""
    
    def __init__(self, config: Config, vocab: Vocabulary, 
                 field_vocab: FieldVocabulary, taxonomy: RepairTaxonomy,
                 validator: RepairValidator = None):
        """
        Initialize parser with all necessary components
        Args:
            config: Model configuration
            vocab: Character vocabulary
            field_vocab: Field name vocabulary
            taxonomy: repair taxonomy
            validator: Optional repair validator (creates default if not provided)
        """
        self.config = config
        self.vocab = vocab
        self.field_vocab = field_vocab
        self.taxonomy = taxonomy
        self.validator = validator or RepairValidator()
    
    def validate_transaction_data(self, txn_data: Dict) -> bool:
        """
        Validate that transaction data has required structure
        Checks for repairs and at least one entity with before/after data
        """
        required_keys = ['ace']  # Must have repair information
        # Check if at least one entity exists in the transaction
        has_entity_data = any(k in txn_data for k in 
                              ['cdtrAgt', 'dbtrAgt', 'instgAgt', 'cdtr', 'dbtr'])
        return all(k in txn_data for k in required_keys) and has_entity_data
    
    def parse_file(self, json_file: str) -> List[Tuple]:
        """
        Parse repair file with comprehensive validation
        
        Args:
            json_file: Path to JSON file containing repair data
            
        Returns:
            List of tuples: (transaction_id, repair_ids, before_data, after_data, validation_results)
        """
        logger.info(f"Parsing file: {json_file}")
        
        # Load JSON file with error handling
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load JSON file: {e}")
            raise
        
        transactions = []  # Store parsed transactions
        invalid_count = 0  # Track invalid transactions
        
        # Process each transaction in the file
        for txn_id, txn_data in data.items():
            # Validate transaction structure
            if not self.validate_transaction_data(txn_data):
                logger.warning(f"Skipping invalid transaction: {txn_id}")
                invalid_count += 1
                continue
            
            try:
                # Extract repairs from transaction
                ace_repairs = []  # List of repair IDs
                repair_categories = Counter()  # Count repairs by category
                
                if 'ace' in txn_data:
                    # Process each repair in the array
                    for repair in txn_data['ace']:
                        repair_id = repair.get('id', 'unknown')
                        code = repair.get('code', 'I')  # Default to Information
                        field = repair.get('field', '')
                        text = repair.get('text', '')
                        
                        # Register repair in taxonomy
                        self.taxonomy.add_repair(repair_id, code, field, text)
                        ace_repairs.append(repair_id)
                        repair_categories[code] += 1  # Track category distribution
                
                # Initialize storage for before/after data
                before_data = {}
                after_data = {}
                
                # Process each entity type that might have repairs
                for entity_key in ['cdtrAgt', 'dbtrAgt', 'instgAgt', 'cdtr', 'dbtr', 
                                  'cdtrAcct', 'dbtrAcct', 'intrmyAgt1', 'intrmyAgt2']:
                    if entity_key in txn_data:
                        entity_data = txn_data[entity_key]
                        
                        # Process 'before' state
                        if 'before' in entity_data:
                            try:
                                # Flatten and normalize the nested structure
                                before_flat = flatten_json(
                                    normalize_keys(entity_data['before']), 
                                    parent_key=entity_key  # Prefix with entity type
                                )
                                before_data.update(before_flat)
                            except Exception as e:
                                logger.warning(f"Failed to process before data for {entity_key}: {e}")
                        
                        # Process 'after' state
                        if 'after' in entity_data:
                            try:
                                # Flatten and normalize the nested structure
                                after_flat = flatten_json(
                                    normalize_keys(entity_data['after']), 
                                    parent_key=entity_key  # Prefix with entity type
                                )
                                after_data.update(after_flat)
                            except Exception as e:
                                logger.warning(f"Failed to process after data for {entity_key}: {e}")
                
                # Only add transaction if we have both before and after data
                if before_data and after_data:
                    # Validate the repairs
                    validation_results = self.validator.validate_repairs(
                        before_data, after_data, ace_repairs
                    )
                    
                    # Add complete transaction tuple
                    transactions.append((
                        txn_id, ace_repairs, before_data, after_data, validation_results
                    ))
                    
                    # Log progress every 100 transactions
                    if len(transactions) % 100 == 0:
                        logger.info(f"Processed {len(transactions)} transactions")
                        logger.info(f"Repair categories in last transaction: {dict(repair_categories)}")
                
            except Exception as e:
                logger.error(f"Error processing transaction {txn_id}: {e}")
                continue
        
        # Log final statistics
        logger.info(f"Successfully parsed {len(transactions)} transactions")
        logger.info(f"Skipped {invalid_count} invalid transactions")
        
        return transactions
    
    def encode_fields(self, field_dict: Dict) -> Dict[str, torch.Tensor]:
        """
        Encode field dictionary to tensors for neural network
        
        Args:
            field_dict: Dictionary of field names to values
            
        Returns:
            Dictionary containing encoded tensors
        """
        field_names = []  # Encoded field name indices
        field_values = []  # Encoded field value character sequences
        
        # Sort fields for consistency and limit to max_fields
        sorted_fields = sorted(field_dict.items())[:self.config.max_fields]
        
        # Encode each field
        for field_name, field_value in sorted_fields:
            # Convert field name to index
            field_names.append(self.field_vocab.get_idx(field_name))
            # Encode field value as character sequence
            field_values.append(self.vocab.encode(field_value, self.config.max_text_length))
        
        # Pad to max_fields if necessary
        while len(field_names) < self.config.max_fields:
            field_names.append(self.field_vocab.get_idx('<PAD>'))
            field_values.append([self.vocab.char2idx['<PAD>']] * self.config.max_text_length)
        
        # Create mask to indicate which fields are real (1.0) vs padding (0.0)
        mask = [1.0] * len(sorted_fields) + [0.0] * (self.config.max_fields - len(sorted_fields))
        
        return {
            'field_names': torch.tensor(field_names, dtype=torch.long),
            'field_values': torch.tensor(field_values, dtype=torch.long),
            'field_mask': torch.tensor(mask, dtype=torch.float32)
        }
    
    def encode_repairs(self, repair_ids: List[str], max_repairs: int) -> torch.Tensor:
        """
        Encode repair IDs to tensor
        
        Args:
            repair_ids: List of repair ID strings
            max_repairs: Maximum number of repairs to encode
            
        Returns:
            Tensor of repair indices
        """
        # Convert repair IDs to indices, limit to max_repairs
        encoded = [self.taxonomy.get_idx(rid) for rid in repair_ids[:max_repairs]]
        # Pad with zeros (NONE tokens) if necessary
        encoded += [0] * (max_repairs - len(encoded))
        return torch.tensor(encoded, dtype=torch.long)


# ============================================================================
# NEURAL NETWORK COMPONENTS
# ============================================================================

class RepairAttention(nn.Module):
    """
    Attention mechanism for learning repair-field interactions
    Allows model to learn which repairs affect which fields
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 4):
        """
        Initialize repair attention module
        Args:
            embed_dim: Dimension of embeddings
            num_heads: Number of attention heads
        """
        super().__init__()
        # Multi-head attention for repair-field interactions
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True  # Expect batch dimension first
        )
        # Layer normalization for stability
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, field_emb, repair_emb, repair_mask=None):
        """
        Apply attention between fields and repairs
        
        Args:
            field_emb: Field embeddings (batch, num_fields, embed_dim)
            repair_emb: Repair embeddings (batch, num_repairs, embed_dim)
            repair_mask: Boolean mask for padding (batch, num_repairs)
            
        Returns:
            Tuple of (attended field embeddings, attention weights)
        """
        # Fields attend to repairs: fields are queries, repairs are keys/values
        attn_output, attn_weights = self.multihead_attn(
            field_emb,  # Queries: what does each field need?
            repair_emb,  # Keys: what does each repair offer?
            repair_emb,  # Values: repair information to aggregate
            key_padding_mask=repair_mask  # Mask padded repairs
        )
        
        # Residual connection and normalization
        output = self.norm(field_emb + attn_output)
        
        return output, attn_weights


class RepairPredictor(nn.Module):
    """
    Predicts which repairs should be applied to a payment
    Used during inference to determine repair strategy
    """
    
    def __init__(self, hidden_dim: int, num_repairs: int):
        """
        Initialize repair predictor
        Args:
            hidden_dim: Dimension of hidden representations
            num_repairs: Total number of possible repairs
        """
        super().__init__()
        # Multi-layer classifier for repair prediction
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),  # Reduce dimensionality
            nn.ReLU(),  # Non-linearity
            nn.Dropout(0.3),  # Regularization
            nn.Linear(hidden_dim // 2, num_repairs),  # Output layer
            nn.Sigmoid()  # Output probabilities [0,1] for each repair
        )
    
    def forward(self, field_representations):
        """
        Predict repair probabilities from field representations
        
        Args:
            field_representations: Field embeddings (batch, num_fields, hidden_dim)
            
        Returns:
            Repair probabilities (batch, num_repairs)
        """
        # Pool field representations by averaging
        pooled = field_representations.mean(dim=1)  # (batch, hidden_dim)
        # Predict repair probabilities
        return self.classifier(pooled)  # (batch, num_repairs)


class RepairModel(nn.Module):
    """
    Enhanced neural network that learns repair patterns
    Combines repair context, field encoding, and transformation prediction
    """
    
    def __init__(self, config: Config, vocab_size: int, field_vocab_size: int, 
                 num_repairs: int):
        """
        Initialize the repair model
        
        Args:
            config: Model configuration
            vocab_size: Size of character vocabulary
            field_vocab_size: Size of field name vocabulary
            num_repairs: Total number of repairs in taxonomy
        """
        super().__init__()
        self.config = config
        
        # ===== Embedding Layers =====
        # Embed repair IDs into continuous space
        self.repair_embedding = nn.Embedding(num_repairs, config.embedding_dim)
        # Embed repair categories (I/W/E/R) separately
        self.repair_category_embedding = nn.Embedding(5, config.embedding_dim // 4)
        # Projection layer for combining repair and category embeddings
        self.repair_projection = nn.Linear(
            config.embedding_dim + config.embedding_dim // 4, 
            config.embedding_dim
        )
        # Embed field names
        self.field_name_embedding = nn.Embedding(field_vocab_size, config.embedding_dim)
        # Embed characters for field values
        self.char_embedding = nn.Embedding(vocab_size, config.embedding_dim, padding_idx=0)
        
        # ===== Field Value Encoder =====
        # Bidirectional LSTM to encode character sequences of field values
        self.value_lstm = nn.LSTM(
            config.embedding_dim,  # Input size
            config.hidden_dim // 2,  # Hidden size (halved because bidirectional)
            num_layers=config.num_lstm_layers,  # Stack multiple LSTM layers
            bidirectional=True,  # Process sequence in both directions
            batch_first=True,  # Batch dimension first
            dropout=config.dropout if config.num_lstm_layers > 1 else 0  # Inter-layer dropout
        )
        
        # ===== Repair Attention =====
        # Learn which repairs affect which fields
        self.repair_attention = RepairAttention(
            embed_dim=config.embedding_dim,
            num_heads=config.repair_attention_heads
        )
        
        # ===== Repair Predictor =====
        # Predict which repairs to apply
        self.repair_predictor = RepairPredictor(config.hidden_dim, num_repairs)
        
        # ===== Field Combiner =====
        # Combine field name, value, and repair context
        self.field_combiner = nn.Sequential(
            # Input: field_emb + name_emb + value_emb
            nn.Linear(config.embedding_dim * 2 + config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.LayerNorm(config.hidden_dim)
        )
        
        # ===== Transformer Layers =====
        # Model field interactions and dependencies
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,  # Model dimension
                nhead=config.num_attention_heads,  # Number of attention heads
                dim_feedforward=config.hidden_dim * 4,  # FFN dimension
                dropout=config.dropout,  # Dropout rate
                batch_first=True  # Batch dimension first
            )
            for _ in range(4)  # Stack 4 transformer layers
        ])
        
        # ===== Change Detection =====
        # Predict whether each field should change
        self.change_detector = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid()  # Output probability of change
        )
        
        # ===== Field Decoder =====
        # Decode transformed field representations to character sequences
        # Note: Set dropout to 0 when num_layers=1 to avoid warning
        decoder_dropout = config.dropout if config.num_lstm_layers > 1 else 0
        self.decoder_lstm = nn.LSTM(
            config.hidden_dim, 
            config.hidden_dim, 
            batch_first=True,
            num_layers=1,  # Single layer decoder
            dropout=0  # No dropout for single layer LSTM
        )
        
        # Project to vocabulary size for character prediction
        self.output_projection = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, vocab_size)  # Output logits for each character
        )
        
        # ===== Confidence Scorer =====
        # Predict confidence for each field transformation
        self.confidence_scorer = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid()  # Output confidence [0,1]
        )
    
    def encode_repairs(self, repair_ids, repair_categories=None):
        """
        Encode repairs with optional category information
        
        Args:
            repair_ids: Tensor of repair IDs (batch, num_repairs)
            repair_categories: Optional tensor of repair categories (batch, num_repairs)
            
        Returns:
            Repair embeddings (batch, num_repairs, embed_dim)
        """
        # Get base repair embeddings (batch, num_repairs, embed_dim)
        repair_emb = self.repair_embedding(repair_ids)
        
        # Combine with category embeddings if provided
        if repair_categories is not None:
            # Embed categories - gives (batch, num_repairs, category_embed_dim)
            category_emb = self.repair_category_embedding(repair_categories)
            # Concatenate repair and category embeddings along the last dimension
            combined = torch.cat([repair_emb, category_emb], dim=-1)
            # Project back to embedding dimension using the pre-defined layer
            repair_emb = self.repair_projection(combined)
        
        return repair_emb
    
    def forward(self, input_dict, return_repair_predictions=False):
        """
        Forward pass through the model
        
        Args:
            input_dict: Dictionary containing:
                - field_names: Field name indices (batch, max_fields)
                - field_values: Character sequences (batch, max_fields, max_length)
                - field_mask: Valid field mask (batch, max_fields)
                - repair_ids: Repair IDs (batch, max_repairs)
                - repair_categories: Optional repair categories
            return_repair_predictions: Whether to predict repairs
            
        Returns:
            Dictionary containing model outputs
        """
        # Extract batch dimensions
        batch_size = input_dict['field_names'].size(0)
        max_fields = input_dict['field_names'].size(1)
        
        # ===== Encode Repairs =====
        repair_ids = input_dict['repair_ids']
        repair_categories = input_dict.get('repair_categories', None)
        repair_mask = (repair_ids == 0)  # True for padding positions
        
        # Get repair embeddings with categories
        repair_emb = self.encode_repairs(repair_ids, repair_categories)
        
        # ===== Encode Field Names =====
        field_name_emb = self.field_name_embedding(input_dict['field_names'])
        
        # ===== Encode Field Values =====
        field_values = input_dict['field_values']
        # Flatten batch and field dimensions for LSTM processing
        field_values_flat = field_values.view(batch_size * max_fields, -1)
        # Embed characters
        char_emb = self.char_embedding(field_values_flat)
        # Process through bidirectional LSTM
        lstm_out, (hidden, _) = self.value_lstm(char_emb)
        # Concatenate final hidden states from both directions
        value_emb = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        # Reshape to separate batch and field dimensions
        value_emb = value_emb.view(batch_size, max_fields, -1)
        
        # ===== Apply Repair Attention =====
        # Learn which repairs affect each field
        field_with_repair_context, attn_weights = self.repair_attention(
            field_name_emb, repair_emb, repair_mask
        )
        
        # ===== Combine Field Representations =====
        # Concatenate repair context, field name, and value embeddings
        combined = torch.cat([field_with_repair_context, field_name_emb, value_emb], dim=-1)
        field_embeddings = self.field_combiner(combined)
        
        # ===== Predict Repairs (Optional) =====
        repair_predictions = None
        if return_repair_predictions:
            repair_predictions = self.repair_predictor(field_embeddings)
        
        # ===== Apply Transformer Layers =====
        # Model field interactions
        field_mask = input_dict['field_mask']
        attention_mask = (field_mask == 0)  # True for positions to mask
        
        enriched = field_embeddings
        for transformer_layer in self.transformer_layers:
            enriched = transformer_layer(enriched, src_key_padding_mask=attention_mask)
        
        # ===== Detect Changes =====
        # Predict which fields should change
        change_probabilities = self.change_detector(enriched).squeeze(-1)
        
        # ===== Decode to Character Sequences =====
        # Flatten for decoding
        enriched_flat = enriched.view(batch_size * max_fields, self.config.hidden_dim)
        # Expand to sequence length
        decoded_flat = enriched_flat.unsqueeze(1).repeat(1, self.config.max_text_length, 1)
        # Process through decoder LSTM
        decoded_flat, _ = self.decoder_lstm(decoded_flat)
        # Project to character logits
        logits_flat = self.output_projection(decoded_flat)
        # Reshape to separate dimensions
        logits = logits_flat.view(batch_size, max_fields, self.config.max_text_length, -1)
        
        # ===== Calculate Confidence =====
        confidence_flat = self.confidence_scorer(enriched_flat)
        confidence = confidence_flat.view(batch_size, max_fields)
        
        # ===== Prepare Outputs =====
        outputs = {
            'field_logits': logits,  # Character predictions
            'field_confidence': confidence,  # Confidence scores
            'change_probabilities': change_probabilities,  # Change detection
            'repair_attention_weights': attn_weights  # Attention weights for interpretability
        }
        
        if return_repair_predictions:
            outputs['repair_predictions'] = repair_predictions
        
        return outputs


# ============================================================================
# DATASET FOR TRAINING
# ============================================================================

class RepairDataset(Dataset):
    """PyTorch Dataset for repair data with validation tracking"""
    
    def __init__(self, transactions: List[Tuple], parser: DataParser, 
                 include_validation: bool = True):
        """
        Initialize dataset
        
        Args:
            transactions: List of parsed transactions
            parser: DataParser instance
            include_validation: Whether to include validation results
        """
        self.parser = parser
        self.transactions = transactions
        self.include_validation = include_validation
    
    def __len__(self):
        """Return number of transactions in dataset"""
        return len(self.transactions)
    
    def __getitem__(self, idx):
        """
        Get a single transaction for training
        
        Args:
            idx: Index of transaction
            
        Returns:
            Tuple of (input_dict, output_dict) for training
        """
        # Unpack transaction data
        txn_id, repair_ids, before_data, after_data, validation_results = self.transactions[idx]
        
        # Encode before state (input)
        input_encoded = self.encode_fields(before_data)
        # Encode after state (target)
        output_encoded = self.encode_fields(after_data)
        
        # Encode repair IDs
        repair_encoded = self.parser.encode_repairs(repair_ids, self.parser.config.max_repairs)
        input_encoded['repair_ids'] = repair_encoded
        
        # Add repair categories for better learning
        repair_categories = []
        for rid in repair_ids[:self.parser.config.max_repairs]:
            # Get category code and map to index
            category = self.parser.taxonomy.get_repair_category(rid)
            category_map = {'I': 0, 'W': 1, 'E': 2, 'R': 3}
            repair_categories.append(category_map.get(category, 4))  # 4 for unknown
        
        # Pad categories to max_repairs
        repair_categories += [4] * (self.parser.config.max_repairs - len(repair_categories))
        input_encoded['repair_categories'] = torch.tensor(repair_categories, dtype=torch.long)
        
        # Calculate which fields changed
        change_mask = self.calculate_change_mask(before_data, after_data)
        output_encoded['change_mask'] = change_mask
        
        return input_encoded, output_encoded
    
    def encode_fields(self, field_dict: Dict) -> Dict[str, torch.Tensor]:
        """
        Encode field dictionary to tensors
        
        Args:
            field_dict: Dictionary of field names to values
            
        Returns:
            Dictionary of encoded tensors
        """
        field_names = []
        field_values = []
        
        # Sort and limit fields
        sorted_fields = sorted(field_dict.items())[:self.parser.config.max_fields]
        
        # Encode each field
        for field_name, field_value in sorted_fields:
            field_names.append(self.parser.field_vocab.get_idx(field_name))
            field_values.append(
                self.parser.vocab.encode(field_value, self.parser.config.max_text_length)
            )
        
        # Pad to max_fields
        while len(field_names) < self.parser.config.max_fields:
            field_names.append(self.parser.field_vocab.get_idx('<PAD>'))
            field_values.append([self.parser.vocab.char2idx['<PAD>']] * self.parser.config.max_text_length)
        
        # Create mask for valid fields
        mask = [1.0] * len(sorted_fields) + [0.0] * (self.parser.config.max_fields - len(sorted_fields))
        
        return {
            'field_names': torch.tensor(field_names, dtype=torch.long),
            'field_values': torch.tensor(field_values, dtype=torch.long),
            'field_mask': torch.tensor(mask, dtype=torch.float32)
        }
    
    def calculate_change_mask(self, before: Dict, after: Dict) -> torch.Tensor:
        """
        Calculate binary mask indicating which fields changed
        
        Args:
            before: Before state dictionary
            after: After state dictionary
            
        Returns:
            Binary mask tensor (1.0 for changed, 0.0 for unchanged/padding)
        """
        change_mask = []
        sorted_fields = sorted(before.keys())[:self.parser.config.max_fields]
        
        # Check each field for changes
        for field_name in sorted_fields:
            before_value = before.get(field_name, '')
            after_value = after.get(field_name, '')
            change_mask.append(1.0 if before_value != after_value else 0.0)
        
        # Pad mask to max_fields
        change_mask += [0.0] * (self.parser.config.max_fields - len(change_mask))
        
        return torch.tensor(change_mask, dtype=torch.float32)


# ============================================================================
# TRAINING LOGIC WITH MULTIPLE LOSS COMPONENTS
# ============================================================================

class Trainer:
    """Enhanced trainer with multiple loss components and comprehensive logging"""
    
    def __init__(self, model: RepairModel, config: Config, device: str = 'cpu'):
        """
        Initialize trainer
        
        Args:
            model: RepairModel instance
            config: Training configuration
            device: Device to train on ('cpu' or 'cuda')
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # ===== Optimizer =====
        # AdamW optimizer with weight decay for regularization
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # ===== Learning Rate Scheduler =====
        # Reduce learning rate when validation loss plateaus
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min',  # Minimize loss
            patience=10,  # Wait 10 epochs before reducing
            factor=0.5  # Reduce LR by half
        )
        
        # ===== Loss Functions =====
        # Character-level cross-entropy loss
        self.char_criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
        # Binary cross-entropy for change detection
        self.change_criterion = nn.BCELoss(reduction='none')
        # Binary cross-entropy for repair prediction
        self.repair_criterion = nn.BCELoss()
        
        # ===== Training State =====
        self.best_val_loss = float('inf')
        self.training_history = []
    
    def compute_loss(self, outputs, targets, inputs=None):
        """
        Compute combined loss with multiple components
        
        Args:
            outputs: Model outputs dictionary
            targets: Target values dictionary
            inputs: Input values dictionary (optional)
            
        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        # Extract dimensions
        batch_size = targets['field_values'].size(0)
        max_fields = targets['field_values'].size(1)
        max_length = targets['field_values'].size(2)
        
        # ===== Character-Level Loss =====
        logits = outputs['field_logits']
        target_values = targets['field_values']
        field_mask = targets['field_mask']
        
        # Flatten for loss computation
        logits_flat = logits.view(batch_size * max_fields * max_length, -1)
        targets_flat = target_values.view(batch_size * max_fields * max_length)
        
        # Compute per-token loss
        char_loss_per_token = self.char_criterion(logits_flat, targets_flat)
        char_loss_per_token = char_loss_per_token.view(batch_size, max_fields, max_length)
        
        # Apply field mask to ignore padding
        field_mask_expanded = field_mask.unsqueeze(-1).expand(-1, -1, max_length)
        masked_char_loss = char_loss_per_token * field_mask_expanded
        # Average over valid positions
        char_loss = masked_char_loss.sum() / field_mask_expanded.sum().clamp(min=1)
        
        # ===== Change Detection Loss =====
        change_predictions = outputs['change_probabilities']
        change_targets = targets.get('change_mask', torch.zeros_like(change_predictions))
        # Compute binary cross-entropy per field
        change_loss_per_field = self.change_criterion(change_predictions, change_targets)
        # Apply field mask
        masked_change_loss = change_loss_per_field * field_mask
        # Average over valid fields
        change_loss = masked_change_loss.sum() / field_mask.sum().clamp(min=1)
        
        # ===== Repair Prediction Loss =====
        # Currently not implemented as it requires ground truth repair labels
        repair_loss = torch.tensor(0.0).to(self.device)
        if 'repair_predictions' in outputs:
            # Would compute loss here if we had repair targets
            pass
        
        # ===== Combined Loss =====
        # Weight each loss component according to config
        total_loss = (
            self.config.char_loss_weight * char_loss +
            self.config.change_detection_weight * change_loss +
            self.config.repair_prediction_weight * repair_loss
        )
        
        # Return total loss and individual components for logging
        return total_loss, {
            'char_loss': char_loss.item(),
            'change_loss': change_loss.item(),
            'repair_loss': repair_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def train_epoch(self, train_loader, epoch):
        """
        Train for one epoch
        
        Args:
            train_loader: DataLoader for training data
            epoch: Current epoch number
            
        Returns:
            Dictionary of average losses for the epoch
        """
        self.model.train()  # Set model to training mode
        epoch_losses = defaultdict(float)  # Accumulate losses
        
        # Iterate through batches
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # Move data to device (GPU/CPU)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            targets = {k: v.to(self.device) for k, v in targets.items()}
            
            # Zero gradients from previous step
            self.optimizer.zero_grad()
            
            # Forward pass through model
            outputs = self.model(inputs)
            
            # Compute loss with all components
            loss, loss_components = self.compute_loss(outputs, targets, inputs)
            
            # Backward pass - compute gradients
            loss.backward()
            
            # Clip gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            
            # Update model parameters
            self.optimizer.step()
            
            # Track losses for averaging
            for key, value in loss_components.items():
                epoch_losses[key] += value
            
            # Log progress every 10 batches
            if batch_idx % 10 == 0:
                logger.info(
                    f'  Epoch {epoch} Batch {batch_idx}/{len(train_loader)}, '
                    f'Loss: {loss.item():.4f} '
                    f'(char: {loss_components["char_loss"]:.4f}, '
                    f'change: {loss_components["change_loss"]:.4f})'
                )
        
        # Average losses over all batches
        for key in epoch_losses:
            epoch_losses[key] /= len(train_loader)
        
        return epoch_losses
    
    def evaluate(self, val_loader):
        """
        Evaluate model on validation set
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            Dictionary of average losses
        """
        self.model.eval()  # Set model to evaluation mode
        epoch_losses = defaultdict(float)  # Accumulate losses
        
        # Disable gradient computation for efficiency
        with torch.no_grad():
            for inputs, targets in val_loader:
                # Move data to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                targets = {k: v.to(self.device) for k, v in targets.items()}
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Compute losses
                loss, loss_components = self.compute_loss(outputs, targets, inputs)
                
                # Accumulate losses
                for key, value in loss_components.items():
                    epoch_losses[key] += value
        
        # Average losses over all batches
        for key in epoch_losses:
            epoch_losses[key] /= len(val_loader)
        
        return epoch_losses
    
    def train(self, train_loader, val_loader):
        """
        Complete training loop
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
        """
        # Print training header
        logger.info("="*70)
        logger.info("TRAINING ENHANCED REPAIR MODEL")
        logger.info("="*70)
        
        # Training loop over epochs
        for epoch in range(1, self.config.num_epochs + 1):
            logger.info(f"\nEpoch {epoch}/{self.config.num_epochs}")
            logger.info("-"*70)
            
            # Train for one epoch
            train_losses = self.train_epoch(train_loader, epoch)
            
            # Evaluate on validation set
            val_losses = self.evaluate(val_loader)
            
            # Adjust learning rate based on validation loss
            self.scheduler.step(val_losses['total_loss'])
            
            # Log epoch results
            logger.info(f"Train - Total: {train_losses['total_loss']:.4f}, "
                       f"Char: {train_losses['char_loss']:.4f}, "
                       f"Change: {train_losses['change_loss']:.4f}")
            logger.info(f"Val   - Total: {val_losses['total_loss']:.4f}, "
                       f"Char: {val_losses['char_loss']:.4f}, "
                       f"Change: {val_losses['change_loss']:.4f}")
            
            # Track training history for analysis
            self.training_history.append({
                'epoch': epoch,
                'train': train_losses,
                'val': val_losses
            })
            
            # Save best model based on validation loss
            if val_losses['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_losses['total_loss']
                self.save_model()
                logger.info("✓ Best model saved!")
        
        # Training complete
        logger.info(f"\nTraining complete! Best val loss: {self.best_val_loss:.4f}")
        self.save_training_history()
    
    def save_model(self):
        """Save model checkpoint with all training state"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.config.model_save_path), exist_ok=True)
        
        # Save complete training state
        torch.save({
            'model_state_dict': self.model.state_dict(),  # Model parameters
            'optimizer_state_dict': self.optimizer.state_dict(),  # Optimizer state
            'scheduler_state_dict': self.scheduler.state_dict(),  # Scheduler state
            'best_val_loss': self.best_val_loss,  # Best validation loss
            'config': asdict(self.config)  # Configuration for reproducibility
        }, self.config.model_save_path)
        
        logger.info(f"Model saved to {self.config.model_save_path}")
    
    def save_training_history(self):
        """Save training history for analysis and visualization"""
        # Create history filename from model path
        history_path = self.config.model_save_path.replace('.pt', '_history.json')
        
        # Save as JSON for easy loading and plotting
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        logger.info(f"Training history saved to {history_path}")


# ============================================================================
# INFERENCE WITH REPAIR PREDICTION AND VALIDATION
# ============================================================================

class Predictor:
    """Enhanced predictor with repair prediction, validation, and explainability"""
    
    def __init__(self, model_dir: str = './models', device: str = 'cpu'):
        """
        Initialize predictor by loading trained model and components
        
        Args:
            model_dir: Directory containing saved model and vocabularies
            device: Device for inference ('cpu' or 'cuda')
        """
        self.device = device
        self.model_dir = model_dir
        
        logger.info(f"Loading model from {model_dir}")
        
        # ===== Load Configuration and Vocabularies =====
        self.config = Config.load(f"{model_dir}/config.json")
        self.vocab = Vocabulary.load(self.config.vocab_save_path)
        self.field_vocab = FieldVocabulary.load(self.config.field_vocab_save_path)
        self.taxonomy = RepairTaxonomy.load(self.config.repair_taxonomy_path)
        self.validator = RepairValidator()
        
        # Initialize parser with loaded components
        self.parser = DataParser(
            self.config, self.vocab, self.field_vocab, 
            self.taxonomy, self.validator
        )
        
        # ===== Load Model =====
        self.model = RepairModel(
            self.config,
            len(self.vocab),
            len(self.field_vocab),
            len(self.taxonomy)
        )
        
        # Load trained weights
        checkpoint = torch.load(self.config.model_save_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()  # Set to evaluation mode
        
        logger.info("Model loaded successfully")
    
    def predict_repairs(self, payment: Dict, threshold: float = 0.5) -> List[str]:
        """
        First stage: Predict which repairs should be applied to a payment
        
        Args:
            payment: Input payment dictionary
            threshold: Probability threshold for selecting repairs
            
        Returns:
            List of predicted repair IDs
        """
        # Flatten and normalize payment structure
        before_flat = flatten_json(normalize_keys(payment))
        
        # Encode payment fields
        encoded = self.parser.encode_fields(before_flat)
        
        # Add dummy repair IDs for initial prediction
        # Model will predict which repairs are actually needed
        encoded['repair_ids'] = torch.zeros(self.config.max_repairs, dtype=torch.long)
        
        # Prepare batch (add batch dimension)
        batch_input = {k: v.unsqueeze(0).to(self.device) for k, v in encoded.items()}
        
        # Get repair predictions from model
        with torch.no_grad():
            outputs = self.model(batch_input, return_repair_predictions=True)
        
        # Extract repair probabilities
        repair_probs = outputs['repair_predictions'][0].cpu().numpy()
        
        # Select repairs above threshold
        predicted_repairs = []
        for idx, prob in enumerate(repair_probs):
            if prob > threshold and idx in self.taxonomy.idx2repair:
                repair_id = self.taxonomy.idx2repair[idx]
                # Skip special tokens
                if repair_id not in ['<NONE>', '<UNK>']:
                    predicted_repairs.append((repair_id, float(prob)))
        
        # Sort by probability (highest first)
        predicted_repairs.sort(key=lambda x: x[1], reverse=True)
        
        # Return just the repair IDs
        return [r[0] for r in predicted_repairs]
    
    def predict(self, payment: Dict, repair_ids: List[str] = None, 
                confidence_threshold: float = 0.7) -> Dict:
        """
        Second stage: Apply predicted repairs to transform the payment
        
        Args:
            payment: Input payment dictionary
            repair_ids: List of repair IDs to apply (if None, predicts them)
            confidence_threshold: Minimum confidence to apply a field change
            
        Returns:
            Dictionary containing:
                - enriched_payment: Transformed payment
                - applied_repairs: List of applied changes
                - predicted_repair_ids: Which repairs were predicted
                - repair_descriptions: Human-readable repair descriptions
                - validation: Validation results
        """
        
        # ===== Stage 1: Predict Repairs if Not Provided =====
        if repair_ids is None:
            repair_ids = self.predict_repairs(payment)
            logger.info(f"Predicted repairs: {repair_ids}")
        
        # ===== Prepare Input =====
        # Flatten and normalize payment
        before_flat = flatten_json(normalize_keys(payment))
        
        # Encode payment fields
        encoded = self.parser.encode_fields(before_flat)
        
        # Encode the repair IDs to apply
        repair_encoded = self.parser.encode_repairs(repair_ids, self.config.max_repairs)
        encoded['repair_ids'] = repair_encoded
        
        # Add repair categories for better context
        repair_categories = []
        for rid in repair_ids[:self.config.max_repairs]:
            category = self.taxonomy.get_repair_category(rid)
            category_map = {'I': 0, 'W': 1, 'E': 2, 'R': 3}
            repair_categories.append(category_map.get(category, 4))
        
        # Pad categories to max_repairs
        repair_categories += [4] * (self.config.max_repairs - len(repair_categories))
        encoded['repair_categories'] = torch.tensor(repair_categories, dtype=torch.long)
        
        # Prepare batch
        batch_input = {k: v.unsqueeze(0).to(self.device) for k, v in encoded.items()}
        
        # ===== Stage 2: Apply Repairs =====
        with torch.no_grad():
            outputs = self.model(batch_input)
        
        # ===== Decode Results =====
        enriched = dict(before_flat)  # Start with original values
        applied_repairs = []  # Track what changed
        
        # Extract model outputs
        logits = outputs['field_logits'][0]  # Remove batch dimension
        confidence = outputs['field_confidence'][0]
        change_probs = outputs['change_probabilities'][0]
        field_names = encoded['field_names']
        field_mask = encoded['field_mask']
        
        # Process each field
        for i in range(self.config.max_fields):
            # Skip padding
            if field_mask[i] == 0:
                continue
            
            # Get field name
            field_idx = field_names[i].item()
            field_name = self.field_vocab.idx2field.get(field_idx, '<UNK>')
            
            # Skip special tokens
            if field_name in ['<PAD>', '<UNK>']:
                continue
            
            # Check if field should change based on model predictions
            if change_probs[i] >= 0.5 and confidence[i] >= confidence_threshold:
                # Decode predicted character sequence
                predicted_indices = logits[i].argmax(dim=-1).cpu().tolist()
                predicted_value = self.vocab.decode(predicted_indices)
                
                # Only apply if value actually changed
                if predicted_value and predicted_value != before_flat.get(field_name, ''):
                    # Validate the predicted value
                    is_valid, message = self.validator.validate_field(field_name, predicted_value)
                    
                    if is_valid:
                        # Apply the change
                        enriched[field_name] = predicted_value
                        
                        # Record the applied repair
                        applied_repairs.append({
                            'field': field_name,
                            'before': before_flat.get(field_name, ''),
                            'after': predicted_value,
                            'confidence': float(confidence[i]),
                            'change_probability': float(change_probs[i]),
                            'validation': message
                        })
                    else:
                        # Log validation failure
                        logger.warning(f"Invalid repair for {field_name}: {message}")
        
        # ===== Validate All Repairs =====
        validation_results = self.validator.validate_repairs(before_flat, enriched, repair_ids)
        
        # ===== Prepare Results =====
        results = {
            'enriched_payment': enriched,  # Final transformed payment
            'applied_repairs': applied_repairs,  # List of changes made
            'predicted_repair_ids': repair_ids,  # Which repairs were applied
            'repair_descriptions': [  # Human-readable descriptions
                self.taxonomy.get_repair_description(rid) for rid in repair_ids
            ],
            'validation': validation_results,  # Validation status
            'statistics': {  # Additional statistics
                'total_repairs_predicted': len(repair_ids),
                'fields_changed': len(applied_repairs),
                'validation_passed': validation_results['valid']
            }
        }
        
        return results


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def train_command(args):
    """
    Handle the train command
    Trains the repair model on provided data
    """
    # Initialize configuration
    config = Config()
    config.num_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    
    # ===== Initialize Components =====
    logger.info("Initializing model components...")
    vocab = Vocabulary()
    field_vocab = FieldVocabulary()
    taxonomy = RepairTaxonomy()
    validator = RepairValidator()
    
    # Create parser
    parser = DataParser(config, vocab, field_vocab, taxonomy, validator)
    
    # ===== Parse Training Data =====
    logger.info(f"Parsing {args.input}...")
    transactions = parser.parse_file(args.input)
    logger.info(f"Found {len(transactions)} valid transactions")
    
    # ===== Build Vocabularies =====
    logger.info("Building vocabularies...")
    all_texts = []  # Collect all text values
    all_fields = set()  # Collect all field names
    
    # Process each transaction to build vocabularies
    for txn_id, repair_ids, before_data, after_data, validation in transactions:
        # Process before state
        for field, value in before_data.items():
            all_fields.add(field)
            if value:
                all_texts.append(str(value))
            # Track whether field changed
            field_vocab.add_field(field, before_data.get(field) != after_data.get(field))
        
        # Process after state
        for field, value in after_data.items():
            all_fields.add(field)
            if value:
                all_texts.append(str(value))
    
    # Build character vocabulary from all texts
    vocab.build_from_texts(all_texts)
    
    # Log vocabulary sizes
    logger.info(f"  Characters: {len(vocab)}")
    logger.info(f"  Fields: {len(field_vocab)}")
    logger.info(f"  Repairs: {len(taxonomy)}")
    
    # ===== Show Repair Statistics =====
    logger.info("\nTop 10 Repairs by frequency:")
    sorted_repairs = sorted(
        taxonomy.repairs.items(), 
        key=lambda x: x[1]['count'], 
        reverse=True
    )[:10]
    
    for repair_id, info in sorted_repairs:
        logger.info(f"  {repair_id} [{info['code']}]: {info['text']} (count: {info['count']})")
    
    # ===== Save Model Components =====
    logger.info("\nSaving vocabularies and configuration...")
    os.makedirs('models', exist_ok=True)
    vocab.save(config.vocab_save_path)
    field_vocab.save(config.field_vocab_save_path)
    taxonomy.save(config.repair_taxonomy_path)
    config.save('models/config.json')
    
    # ===== Split Data =====
    n = len(transactions)
    train_size = int(n * config.train_split)  # 80% for training
    val_size = int(n * config.val_split)  # 10% for validation
    
    train_txns = transactions[:train_size]
    val_txns = transactions[train_size:train_size+val_size]
    test_txns = transactions[train_size+val_size:]  # Remaining 10% for test
    
    logger.info(f"\nData split: Train={len(train_txns)}, Val={len(val_txns)}, Test={len(test_txns)}")
    
    # ===== Create Datasets and DataLoaders =====
    train_dataset = RepairDataset(train_txns, parser)
    val_dataset = RepairDataset(val_txns, parser)
    
    # Set num_workers to 0 to avoid multiprocessing issues on some systems
    # Can be increased if the system supports it
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,  # Shuffle training data
        num_workers=0  # Use main process for data loading to avoid pickling issues
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size,
        num_workers=0  # Use main process for data loading
    )
    
    # ===== Initialize Model =====
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    model = RepairModel(config, len(vocab), len(field_vocab), len(taxonomy))
    
    # Log model size
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")
    
    # ===== Train Model =====
    trainer = Trainer(model, config, device)
    trainer.train(train_loader, val_loader)
    
    logger.info("Training completed successfully!")


def predict_command(args):
    """
    Handle the predict command
    Uses trained model to predict repairs and transform payments
    """
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load trained model
    predictor = Predictor(args.model or './models', device)
    
    # Load input payment
    logger.info(f"Loading payment from {args.input}")
    with open(args.input, 'r') as f:
        payment = json.load(f)
    
    # Make predictions
    logger.info("Applying repairs...")
    results = predictor.predict(
        payment, 
        confidence_threshold=args.confidence_threshold
    )
    
    # ===== Display Results =====
    logger.info("\n" + "="*70)
    logger.info("PREDICTION RESULTS")
    logger.info("="*70)
    
    # Show predicted repairs
    logger.info(f"\nPredicted Repairs ({len(results['predicted_repair_ids'])}):")
    for repair_id, desc in zip(
        results['predicted_repair_ids'], 
        results['repair_descriptions']
    ):
        logger.info(f"  - {repair_id}: {desc}")
    
    # Show applied changes
    logger.info(f"\nApplied Changes ({len(results['applied_repairs'])}):")
    for repair in results['applied_repairs']:
        logger.info(f"  Field: {repair['field']}")
        logger.info(f"    Before: {repair['before']}")
        logger.info(f"    After:  {repair['after']}")
        logger.info(f"    Confidence: {repair['confidence']:.2%}")
        logger.info(f"    Change Probability: {repair['change_probability']:.2%}")
    
    # Show validation status
    validation = results['validation']
    logger.info(f"\nValidation: {'PASSED' if validation['valid'] else 'FAILED'}")
    if validation['errors']:
        logger.info("Errors:")
        for error in validation['errors']:
            logger.info(f"  - {error}")
    
    # Show statistics
    stats = results['statistics']
    logger.info(f"\nStatistics:")
    logger.info(f"  Total repairs predicted: {stats['total_repairs_predicted']}")
    logger.info(f"  Fields changed: {stats['fields_changed']}")
    logger.info(f"  Validation passed: {stats['validation_passed']}")
    
    # ===== Save Output =====
    if args.output:
        with open(args.output, 'w') as f:
            # Save enriched payment and repair details
            json.dump({
                'enriched_payment': results['enriched_payment'],
                'applied_repairs': results['applied_repairs'],
                'validation': validation,
                'statistics': stats
            }, f, indent=2)
        logger.info(f"\nResults saved to {args.output}")


def main():
    """
    Main entry point for the CLI
    Parses arguments and routes to appropriate command
    """
    # Create main parser
    parser = argparse.ArgumentParser(
        description='Enhanced Repair Learning Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Train model:
    python ace_repair_model_enhanced.py train --input repairs.json --epochs 100
    
  Make predictions:
    python ace_repair_model_enhanced.py predict --input payment.json --output result.json
        """
    )
    
    # Create subparsers for commands
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # ===== Train Command =====
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument(
        '--input', 
        required=True, 
        help='Input JSON file with repairs'
    )
    train_parser.add_argument(
        '--epochs', 
        type=int, 
        default=100, 
        help='Number of training epochs (default: 100)'
    )
    train_parser.add_argument(
        '--batch_size', 
        type=int, 
        help='Batch size for training (default: from config)'
    )
    
    # ===== Predict Command =====
    predict_parser = subparsers.add_parser('predict', help='Make predictions')
    predict_parser.add_argument(
        '--input', 
        required=True, 
        help='Input payment JSON file'
    )
    predict_parser.add_argument(
        '--output', 
        help='Output file path for results'
    )
    predict_parser.add_argument(
        '--model', 
        default='./models', 
        help='Model directory path (default: ./models)'
    )
    predict_parser.add_argument(
        '--confidence_threshold', 
        type=float, 
        default=0.7,
        help='Confidence threshold for applying changes (default: 0.7)'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Route to appropriate command
    if args.command == 'train':
        train_command(args)
    elif args.command == 'predict':
        predict_command(args)
    else:
        # No command specified - print help
        parser.print_help()


# Entry point when script is run directly
if __name__ == "__main__":
    main()

