#!/usr/bin/env python3
"""
ISO 20022 Payment Entity Validator with Structure Comparison
Validates payment entities and shows structure differences
"""

import json
import re
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass
from enum import Enum


class ValidationLevel(Enum):
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


@dataclass
class ValidationResult:
    """Result of a validation check"""
    level: ValidationLevel
    field: str
    message: str
    
    def __str__(self):
        return f"[{self.level.value}] {self.field}: {self.message}"


# ISO 20022 Structure Specifications (hardcoded)
ISO20022_SPECS = {
    'cdtr': {
        'name': 'Creditor (Beneficiary)',
        'required_fields': ['Nm'],
        'optional_fields': ['PstlAdr', 'Id', 'CtryOfRes'],
        'structure': {
            'Nm': {'type': 'string', 'max_length': 140, 'description': 'Name'},
            'PstlAdr': {'type': 'object', 'description': 'Postal Address'},
            'CtryOfRes': {'type': 'string', 'max_length': 2, 'description': 'Country Code (ISO 3166)'}
        }
    },
    'dbtr': {
        'name': 'Debtor (Originator)',
        'required_fields': ['Nm'],
        'optional_fields': ['PstlAdr', 'Id', 'CtryOfRes'],
        'structure': {
            'Nm': {'type': 'string', 'max_length': 140, 'description': 'Name'},
            'PstlAdr': {'type': 'object', 'description': 'Postal Address'},
            'CtryOfRes': {'type': 'string', 'max_length': 2, 'description': 'Country Code (ISO 3166)'}
        }
    },
    'cdtrAgt': {
        'name': 'Creditor Agent (Beneficiary Bank)',
        'required_fields': ['FinInstnId'],
        'optional_fields': ['BrnchId', 'CtryOfRes'],
        'structure': {
            'FinInstnId': {
                'type': 'object',
                'description': 'Financial Institution ID',
                'fields': {
                    'BICFI': {'type': 'string', 'format': 'BIC (8 or 11 chars)', 'optional': True},
                    'ClrSysMmbId': {'type': 'object', 'optional': True},
                    'Nm': {'type': 'string', 'max_length': 140, 'optional': True},
                    'PstlAdr': {'type': 'object', 'optional': True},
                    'Othr': {'type': 'object', 'optional': True}
                }
            },
            'CtryOfRes': {'type': 'string', 'max_length': 2, 'description': 'Country Code'}
        }
    },
    'dbtrAgt': {
        'name': 'Debtor Agent (Ordering Bank)',
        'required_fields': ['FinInstnId'],
        'optional_fields': ['BrnchId', 'CtryOfRes'],
        'structure': {
            'FinInstnId': {
                'type': 'object',
                'description': 'Financial Institution ID',
                'fields': {
                    'BICFI': {'type': 'string', 'format': 'BIC (8 or 11 chars)', 'optional': True},
                    'ClrSysMmbId': {'type': 'object', 'optional': True},
                    'Nm': {'type': 'string', 'max_length': 140, 'optional': True},
                    'PstlAdr': {'type': 'object', 'optional': True},
                    'Othr': {'type': 'object', 'optional': True}
                }
            }
        }
    },
    'cdtrAcct': {
        'name': 'Creditor Account',
        'required_fields': ['Id'],
        'optional_fields': ['Tp', 'Ccy', 'Nm'],
        'structure': {
            'Id': {
                'type': 'object',
                'description': 'Account Identification (IBAN OR Othr required)',
                'fields': {
                    'IBAN': {'type': 'string', 'format': 'IBAN (15-34 chars)', 'optional': True},
                    'Othr': {
                        'type': 'object',
                        'optional': True,
                        'fields': {
                            'Id': {'type': 'string', 'max_length': 34, 'description': 'Account number'}
                        }
                    }
                }
            },
            'Tp': {'type': 'object', 'description': 'Account Type'},
            'Ccy': {'type': 'string', 'max_length': 3, 'description': 'Currency Code (ISO 4217)'},
            'Nm': {'type': 'string', 'max_length': 70, 'description': 'Account Name'}
        }
    },
    'dbtrAcct': {
        'name': 'Debtor Account',
        'required_fields': ['Id'],
        'optional_fields': ['Tp', 'Ccy', 'Nm'],
        'structure': {
            'Id': {
                'type': 'object',
                'description': 'Account Identification (IBAN OR Othr required)',
                'fields': {
                    'IBAN': {'type': 'string', 'format': 'IBAN (15-34 chars)', 'optional': True},
                    'Othr': {
                        'type': 'object',
                        'optional': True,
                        'fields': {
                            'Id': {'type': 'string', 'max_length': 34, 'description': 'Account number'}
                        }
                    }
                }
            },
            'Tp': {'type': 'object', 'description': 'Account Type'},
            'Ccy': {'type': 'string', 'max_length': 3, 'description': 'Currency Code (ISO 4217)'}
        }
    },
    'instgAgt': {
        'name': 'Instructing Agent',
        'required_fields': ['FinInstnId'],
        'optional_fields': ['BrnchId'],
        'structure': {
            'FinInstnId': {'type': 'object', 'description': 'Financial Institution ID'}
        }
    },
    'instdAgt': {
        'name': 'Instructed Agent',
        'required_fields': ['FinInstnId'],
        'optional_fields': ['BrnchId'],
        'structure': {
            'FinInstnId': {'type': 'object', 'description': 'Financial Institution ID'}
        }
    },
    'rmtInf': {
        'name': 'Remittance Information',
        'required_fields': [],
        'optional_fields': ['Ustrd', 'Strd'],
        'structure': {
            'Ustrd': {'type': 'string or array', 'max_length': 140, 'description': 'Unstructured text'},
            'Strd': {'type': 'object', 'description': 'Structured remittance'}
        }
    }
}


class ISO20022Validator:
    """Validates payment entities against ISO 20022 specifications"""
    
    def __init__(self):
        self.results: List[ValidationResult] = []
        self.structure_analysis = {}
    
    def add_error(self, field: str, message: str):
        self.results.append(ValidationResult(ValidationLevel.ERROR, field, message))
    
    def add_warning(self, field: str, message: str):
        self.results.append(ValidationResult(ValidationLevel.WARNING, field, message))
    
    def add_info(self, field: str, message: str):
        self.results.append(ValidationResult(ValidationLevel.INFO, field, message))
    
    def get_errors(self) -> List[ValidationResult]:
        return [r for r in self.results if r.level == ValidationLevel.ERROR]
    
    def get_warnings(self) -> List[ValidationResult]:
        return [r for r in self.results if r.level == ValidationLevel.WARNING]
    
    def is_valid(self) -> bool:
        return len(self.get_errors()) == 0
    
    def clear(self):
        self.results.clear()
        self.structure_analysis.clear()
    
    # ========================================================================
    # STRUCTURE ANALYSIS
    # ========================================================================
    
    def extract_structure(self, obj: Any, max_depth: int = 4) -> Dict:
        """Extract structure from an object"""
        def traverse(o, depth=0):
            if depth > max_depth:
                return "..."
            
            if isinstance(o, dict):
                return {k: traverse(v, depth + 1) for k, v in o.items()}
            elif isinstance(o, list):
                if len(o) > 0:
                    return [traverse(o[0], depth + 1)]
                return []
            else:
                return type(o).__name__
        
        return traverse(obj)
    
    def get_all_fields(self, obj: Any, prefix: str = "") -> Set[str]:
        """Get all field paths from an object"""
        fields = set()
        
        def traverse(o, path):
            if isinstance(o, dict):
                for k, v in o.items():
                    field_path = f"{path}.{k}" if path else k
                    fields.add(field_path)
                    traverse(v, field_path)
            elif isinstance(o, list) and len(o) > 0:
                traverse(o[0], f"{path}[0]")
        
        traverse(obj, prefix)
        return fields
    
    def normalize_field_name(self, field: str) -> str:
        """Normalize field names (handle PascalCase and camelCase)"""
        # Convert first letter to uppercase for comparison
        if field and field[0].islower():
            return field[0].upper() + field[1:]
        return field
    
    def compare_structures(self, entity_name: str, actual: Dict, spec: Dict) -> Dict:
        """Compare actual structure with ISO 20022 spec"""
        comparison = {
            'entity': entity_name,
            'spec_name': spec['name'],
            'actual_fields': set(),
            'expected_required': set(spec['required_fields']),
            'expected_optional': set(spec['optional_fields']),
            'missing_required': set(),
            'missing_optional': set(),
            'extra_fields': set(),
            'present_fields': set()
        }
        
        # Get all actual fields (normalized)
        actual_fields_raw = self.get_all_fields(actual)
        actual_fields_normalized = set()
        
        for field in actual_fields_raw:
            # Get top-level field name
            top_field = field.split('.')[0].split('[')[0]
            normalized = self.normalize_field_name(top_field)
            actual_fields_normalized.add(normalized)
            comparison['actual_fields'].add(field)
        
        # Check required fields
        for req_field in spec['required_fields']:
            if req_field not in actual_fields_normalized:
                # Check camelCase version
                camel = req_field[0].lower() + req_field[1:] if req_field else req_field
                if camel not in [f.split('.')[0] for f in actual_fields_raw]:
                    comparison['missing_required'].add(req_field)
                else:
                    comparison['present_fields'].add(camel)
            else:
                comparison['present_fields'].add(req_field)
        
        # Check optional fields
        all_expected = comparison['expected_required'] | comparison['expected_optional']
        for opt_field in spec['optional_fields']:
            if opt_field in actual_fields_normalized:
                comparison['present_fields'].add(opt_field)
        
        # Check for extra fields
        for actual_field in actual_fields_normalized:
            if actual_field not in all_expected:
                comparison['extra_fields'].add(actual_field)
        
        return comparison
    
    # ========================================================================
    # VALIDATION (simplified for structure focus)
    # ========================================================================
    
    def validate_payment(self, payment: Dict) -> Dict:
        """Validate entire payment structure"""
        self.clear()
        
        if not isinstance(payment, dict):
            self.add_error("payment", "Payment must be a dictionary")
            return self.get_summary()
        
        # Analyze each entity
        entities_to_check = [
            ('cdtr', 'Cdtr'),
            ('dbtr', 'Dbtr'),
            ('cdtrAgt', 'CdtrAgt'),
            ('dbtrAgt', 'DbtrAgt'),
            ('cdtrAcct', 'CdtrAcct'),
            ('dbtrAcct', 'DbtrAcct'),
            ('instgAgt', 'InstgAgt'),
            ('instdAgt', 'InstdAgt'),
            ('rmtInf', 'RmtInf')
        ]
        
        for camel_name, pascal_name in entities_to_check:
            entity_data = None
            entity_key = None
            
            if camel_name in payment:
                entity_data = payment[camel_name]
                entity_key = camel_name
            elif pascal_name in payment:
                entity_data = payment[pascal_name]
                entity_key = pascal_name
            
            if entity_data and camel_name in ISO20022_SPECS:
                spec = ISO20022_SPECS[camel_name]
                comparison = self.compare_structures(entity_key, entity_data, spec)
                self.structure_analysis[entity_key] = comparison
                
                # Add validation messages based on comparison
                if comparison['missing_required']:
                    for field in comparison['missing_required']:
                        self.add_error(f"{entity_key}.{field}", f"Required field missing")
                
                if comparison['extra_fields']:
                    for field in comparison['extra_fields']:
                        self.add_info(f"{entity_key}.{field}", f"Extra field (not in ISO 20022 spec)")
        
        return self.get_summary()
    
    def get_summary(self) -> Dict:
        """Get validation summary"""
        errors = self.get_errors()
        warnings = self.get_warnings()
        
        return {
            'valid': len(errors) == 0,
            'error_count': len(errors),
            'warning_count': len(warnings),
            'errors': [str(e) for e in errors],
            'warnings': [str(w) for w in warnings],
            'all_results': [str(r) for r in self.results],
            'structure_analysis': self.structure_analysis
        }
    
    def print_structure_comparison(self, payment: Dict):
        """Print detailed structure comparison"""
        print("\n" + "="*80)
        print("STRUCTURE COMPARISON: INPUT vs ISO 20022 SPECIFICATION")
        print("="*80)
        
        entities_to_check = [
            ('cdtr', 'Cdtr'),
            ('dbtr', 'Dbtr'),
            ('cdtrAgt', 'CdtrAgt'),
            ('dbtrAgt', 'DbtrAgt'),
            ('cdtrAcct', 'CdtrAcct'),
            ('dbtrAcct', 'DbtrAcct'),
            ('instgAgt', 'InstgAgt'),
            ('instdAgt', 'InstdAgt'),
            ('rmtInf', 'RmtInf')
        ]
        
        for camel_name, pascal_name in entities_to_check:
            entity_data = None
            entity_key = None
            
            if camel_name in payment:
                entity_data = payment[camel_name]
                entity_key = camel_name
            elif pascal_name in payment:
                entity_data = payment[pascal_name]
                entity_key = pascal_name
            
            if camel_name in ISO20022_SPECS:
                spec = ISO20022_SPECS[camel_name]
                
                print(f"\n{'─'*80}")
                print(f"Entity: {entity_key if entity_key else camel_name} ({spec['name']})")
                print(f"{'─'*80}")
                
                if entity_data:
                    # Show input structure
                    print("\n  INPUT STRUCTURE:")
                    self._print_dict(entity_data, indent=4)
                    
                    # Show ISO spec
                    print("\n  ISO 20022 SPECIFICATION:")
                    print(f"    Required fields: {', '.join(spec['required_fields']) if spec['required_fields'] else 'None'}")
                    print(f"    Optional fields: {', '.join(spec['optional_fields']) if spec['optional_fields'] else 'None'}")
                    
                    # Show detailed spec structure
                    if 'structure' in spec:
                        print("\n    Expected Structure:")
                        for field_name, field_spec in spec['structure'].items():
                            desc = field_spec.get('description', '')
                            field_type = field_spec.get('type', 'unknown')
                            max_len = field_spec.get('max_length', '')
                            length_str = f" (max {max_len})" if max_len else ""
                            print(f"      {field_name}: {field_type}{length_str} - {desc}")
                            
                            # Show nested fields if any
                            if 'fields' in field_spec:
                                for nested, nested_spec in field_spec['fields'].items():
                                    opt = " [OPTIONAL]" if nested_spec.get('optional') else " [REQUIRED]"
                                    n_desc = nested_spec.get('description', '')
                                    print(f"        └─ {nested}: {nested_spec.get('type', 'unknown')}{opt} - {n_desc}")
                    
                    # Show differences
                    if entity_key in self.structure_analysis:
                        comp = self.structure_analysis[entity_key]
                        
                        print("\n  COMPARISON:")
                        print(f"    ✓ Present fields: {', '.join(sorted(comp['present_fields'])) if comp['present_fields'] else 'None'}")
                        
                        if comp['missing_required']:
                            print(f"    ✗ Missing REQUIRED: {', '.join(sorted(comp['missing_required']))}")
                        
                        if comp['extra_fields']:
                            print(f"    ⚠ Extra fields (not in spec): {', '.join(sorted(comp['extra_fields']))}")
                else:
                    print("\n  ✗ NOT FOUND IN INPUT")
                    print("\n  ISO 20022 SPECIFICATION:")
                    print(f"    Required fields: {', '.join(spec['required_fields'])}")
                    print(f"    Optional fields: {', '.join(spec['optional_fields'])}")
    
    def _print_dict(self, d, indent=0):
        """Pretty print dictionary structure"""
        if not isinstance(d, dict):
            print(" " * indent + str(d))
            return
        
        for key, value in d.items():
            if isinstance(value, dict):
                print(" " * indent + f"{key}:")
                self._print_dict(value, indent + 2)
            elif isinstance(value, list):
                print(" " * indent + f"{key}: [")
                for item in value:
                    if isinstance(item, dict):
                        self._print_dict(item, indent + 2)
                    else:
                        print(" " * (indent + 2) + str(item))
                print(" " * indent + "]")
            else:
                print(" " * indent + f"{key}: {value}")


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='ISO 20022 Payment Validator')
    parser.add_argument('input', help='Input JSON file containing payment data')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed structure comparison')
    
    args = parser.parse_args()
    
    # Load payment
    with open(args.input, 'r') as f:
        data = json.load(f)
    
    # Extract payment (handle wrapped format)
    if isinstance(data, dict):
        if len(data) == 1:
            # Wrapped format: { "txn_id": { payment } }
            payment = data[list(data.keys())[0]]
        else:
            # Direct format
            payment = data
    else:
        print("Error: Invalid JSON format")
        return
    
    # Validate
    validator = ISO20022Validator()
    summary = validator.validate_payment(payment)
    
    # Print results
    print("\n" + "="*80)
    print("ISO 20022 VALIDATION RESULTS")
    print("="*80)
    
    print(f"\nStatus: {'✓ VALID' if summary['valid'] else '✗ INVALID'}")
    print(f"Errors: {summary['error_count']}")
    print(f"Warnings: {summary['warning_count']}")
    
    if summary['errors']:
        print("\nERRORS:")
        for error in summary['errors']:
            print(f"  {error}")
    
    if summary['warnings']:
        print("\nWARNINGS:")
        for warning in summary['warnings']:
            print(f"  {warning}")
    
    # Always show structure comparison in verbose mode
    if args.verbose or True:  # Always show for now
        validator.print_structure_comparison(payment)
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
