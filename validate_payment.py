#!/usr/bin/env python3
"""
ISO 20022 Validator with Context-Aware Repair Lookup
Shows spec, validates structure, suggests ACE repairs
Version 8
"""

import json
import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass
from enum import Enum


class ValidationLevel(Enum):
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"
    PASS = "PASS"


@dataclass
class ValidationResult:
    """Result of a validation check"""
    level: ValidationLevel
    field: str
    message: str
    value: Any = None
    
    def __str__(self):
        if self.level == ValidationLevel.PASS:
            return f"  ✓ {self.field}: {self.message}"
        
        value_str = f" → '{self.value}'" if self.value else ""
        icon = {"ERROR": "✗", "WARNING": "⚠", "INFO": "ℹ"}.get(self.level.value, "•")
        return f"  {icon} {self.field}: {self.message}{value_str}"


@dataclass
class RepairInfo:
    """Information about a specific repair"""
    code: str
    field: str
    text: str
    entity: str
    repair_id: str
    
    def matches_error(self, error_message: str, field_name: str) -> float:
        """Calculate match score between this repair and a validation error"""
        score = 0.0
        error_lower = error_message.lower()
        field_lower = field_name.lower()
        repair_field_lower = self.field.lower()
        repair_text_lower = self.text.lower()
        
        # Exact field match
        if repair_field_lower in field_lower or field_lower in repair_field_lower:
            score += 0.4
        
        # Field mentioned in repair text
        if field_lower in repair_text_lower:
            score += 0.2
        
        # Error type matching
        error_keywords = {
            'missing': ['extracted', 'looked up', 'added', 'required'],
            'required': ['extracted', 'looked up', 'added'],
            'country': ['country', 'ctryofres', 'cdtpty'],
            'bic': ['bic', 'bicfi'],
            'name': ['name', 'nm'],
            'address': ['address', 'pstladr'],
            'routing': ['routing', 'clrsysmmbid'],
            'iban': ['iban'],
            'account': ['account', 'acct'],
            'dashes': ['dashes', 'format'],
        }
        
        for error_type, keywords in error_keywords.items():
            if error_type in error_lower:
                for keyword in keywords:
                    if keyword in repair_text_lower or keyword in repair_field_lower:
                        score += 0.3
                        break
        
        return min(score, 1.0)
    
    def get_severity(self) -> str:
        """Get severity level based on repair code"""
        code_upper = self.code.upper()
        if code_upper == 'W':
            return 'WARNING'
        elif code_upper == 'I':
            return 'INFO'
        elif code_upper == 'E':
            return 'ERROR'
        else:
            return 'INFO'


class RepairLookup:
    """Context-aware repair lookup from pickle file"""
    
    def __init__(self, lookup_path: str):
        if not Path(lookup_path).exists():
            raise FileNotFoundError(f"Repair lookup file not found: {lookup_path}")
        
        with open(lookup_path, 'rb') as f:
            data = pickle.load(f)
        
        self.repairs_by_entity_field = data['repairs_by_entity_field']
        self.repairs_by_id = data['repairs_by_id']
        self.all_repairs = data['all_repairs']
        self.stats = data['stats']
    
    def find_repair(self, entity: str, field: str, error_message: str = "", 
                    context: Optional[Dict] = None, threshold: float = 0.3) -> Optional[RepairInfo]:
        """
        Find best matching repair for an error with context awareness
        
        Args:
            entity: Entity name (e.g., 'cdtrAgt')
            field: Field name (e.g., 'BICFI')
            error_message: Error message from validator
            context: Dict with 'source', 'clearing', 'parties'
            threshold: Minimum match score
        """
        entity = entity.lower()
        field_normalized = self.normalize_field_name(field)
        
        # Get candidate repairs
        candidates = []
        
        if entity in self.repairs_by_entity_field:
            # Exact field match
            if field_normalized in self.repairs_by_entity_field[entity]:
                candidates.extend(self.repairs_by_entity_field[entity][field_normalized])
            
            # Partial field match (for nested fields)
            for repair_field, repairs in self.repairs_by_entity_field[entity].items():
                if repair_field in field_normalized or field_normalized in repair_field:
                    for repair in repairs:
                        if repair not in candidates:
                            candidates.append(repair)
        
        if not candidates:
            return None
        
        # Score each candidate
        scored_repairs = []
        for repair in candidates:
            score = repair.matches_error(error_message, field)
            
            # Boost score if context matches (if context filtering is needed in future)
            # For now, we use all repairs regardless of context
            
            if score >= threshold:
                scored_repairs.append((score, repair))
        
        if not scored_repairs:
            return candidates[0] if candidates else None
        
        # Return best match
        scored_repairs.sort(reverse=True, key=lambda x: x[0])
        return scored_repairs[0][1]
    
    def normalize_field_name(self, name: str) -> str:
        """Normalize field name"""
        if '.' in name:
            name = name.split('.')[-1]
        return name.upper()


# ISO 20022 Specifications
ISO20022_SPECS = {
    'cdtr': {
        'name': 'Creditor (Beneficiary)',
        'description': 'Party to which an amount of money is due',
        'required': ['Nm'],
        'optional': ['PstlAdr', 'Id', 'CtryOfRes'],
        'fields': {
            'Nm': {'type': 'string', 'min': 1, 'max': 140, 'desc': 'Name of the creditor'},
            'PstlAdr': {'type': 'PostalAddress', 'desc': 'Postal address of the creditor'},
            'CtryOfRes': {'type': 'CountryCode', 'format': 'ISO 3166 (2 chars)', 'desc': 'Country of residence'}
        }
    },
    'dbtr': {
        'name': 'Debtor (Originator)',
        'description': 'Party that owes an amount of money',
        'required': ['Nm'],
        'optional': ['PstlAdr', 'Id', 'CtryOfRes'],
        'fields': {
            'Nm': {'type': 'string', 'min': 1, 'max': 140, 'desc': 'Name of the debtor'},
            'PstlAdr': {'type': 'PostalAddress', 'desc': 'Postal address of the debtor'},
            'CtryOfRes': {'type': 'CountryCode', 'format': 'ISO 3166 (2 chars)', 'desc': 'Country of residence'}
        }
    },
    'cdtrAgt': {
        'name': 'Creditor Agent (Beneficiary Bank)',
        'description': 'Financial institution servicing the creditor account',
        'required': ['FinInstnId'],
        'optional': ['BrnchId', 'CtryOfRes'],
        'fields': {
            'FinInstnId': {'type': 'FinancialInstitutionIdentification', 'desc': 'Unique identification of the financial institution'},
            'CtryOfRes': {'type': 'CountryCode', 'format': 'ISO 3166 (2 chars)', 'desc': 'Country where the financial institution is resident'}
        }
    },
    'dbtrAgt': {
        'name': 'Debtor Agent (Ordering Bank)',
        'description': 'Financial institution servicing the debtor account',
        'required': ['FinInstnId'],
        'optional': ['BrnchId', 'CtryOfRes'],
        'fields': {
            'FinInstnId': {'type': 'FinancialInstitutionIdentification', 'desc': 'Unique identification of the financial institution'},
            'CtryOfRes': {'type': 'CountryCode', 'format': 'ISO 3166 (2 chars)', 'desc': 'Country where the financial institution is resident'}
        }
    },
    'cdtrAcct': {
        'name': 'Creditor Account',
        'description': 'Account to be credited',
        'required': ['Id'],
        'optional': ['Tp', 'Ccy', 'Nm'],
        'fields': {
            'Id': {'type': 'AccountIdentification', 'desc': 'Unique identification (IBAN OR Othr required)'},
            'Tp': {'type': 'AccountType', 'desc': 'Type of account'},
            'Ccy': {'type': 'CurrencyCode', 'format': 'ISO 4217 (3 chars)', 'desc': 'Currency of the account'},
            'Nm': {'type': 'string', 'min': 1, 'max': 70, 'desc': 'Name of the account'}
        },
        'nested': {
            'Id': {
                'required_one_of': ['IBAN', 'Othr'],
                'fields': {
                    'IBAN': {'type': 'string', 'format': 'ISO 13616 (15-34 chars)', 'desc': 'International Bank Account Number'},
                    'Othr': {'type': 'GenericAccountIdentification', 'desc': 'Other account identification'}
                }
            },
            'Id.Othr': {
                'required': ['Id'],
                'fields': {
                    'Id': {'type': 'string', 'min': 1, 'max': 34, 'desc': 'Account number', 'no_dashes': True}
                }
            }
        }
    },
    'dbtrAcct': {
        'name': 'Debtor Account',
        'description': 'Account to be debited',
        'required': ['Id'],
        'optional': ['Tp', 'Ccy', 'Nm'],
        'fields': {
            'Id': {'type': 'AccountIdentification', 'desc': 'Unique identification (IBAN OR Othr required)'},
            'Tp': {'type': 'AccountType', 'desc': 'Type of account'},
            'Ccy': {'type': 'CurrencyCode', 'format': 'ISO 4217 (3 chars)', 'desc': 'Currency of the account'},
            'Nm': {'type': 'string', 'min': 1, 'max': 70, 'desc': 'Name of the account'}
        },
        'nested': {
            'Id': {
                'required_one_of': ['IBAN', 'Othr'],
                'fields': {
                    'IBAN': {'type': 'string', 'format': 'ISO 13616 (15-34 chars)', 'desc': 'International Bank Account Number'},
                    'Othr': {'type': 'GenericAccountIdentification', 'desc': 'Other account identification'}
                }
            },
            'Id.Othr': {
                'required': ['Id'],
                'fields': {
                    'Id': {'type': 'string', 'min': 1, 'max': 34, 'desc': 'Account number', 'no_dashes': True}
                }
            }
        }
    },
    'instgAgt': {
        'name': 'Instructing Agent',
        'description': 'Agent that instructs the next party in the chain to carry out the instruction',
        'required': ['FinInstnId'],
        'optional': ['BrnchId'],
        'fields': {
            'FinInstnId': {'type': 'FinancialInstitutionIdentification', 'desc': 'Unique identification of the financial institution'}
        }
    },
    'instdAgt': {
        'name': 'Instructed Agent',
        'description': 'Agent that is instructed by the previous party in the chain to carry out the instruction',
        'required': ['FinInstnId'],
        'optional': ['BrnchId'],
        'fields': {
            'FinInstnId': {'type': 'FinancialInstitutionIdentification', 'desc': 'Unique identification of the financial institution'}
        }
    },
    'intrmyAgt1': {
        'name': 'Intermediary Agent 1',
        'description': 'First intermediary agent in the payment chain',
        'required': ['FinInstnId'],
        'optional': ['BrnchId'],
        'fields': {
            'FinInstnId': {'type': 'FinancialInstitutionIdentification', 'desc': 'Unique identification of the financial institution'}
        }
    },
    'intrmyAgt2': {
        'name': 'Intermediary Agent 2',
        'description': 'Second intermediary agent in the payment chain',
        'required': ['FinInstnId'],
        'optional': ['BrnchId'],
        'fields': {
            'FinInstnId': {'type': 'FinancialInstitutionIdentification', 'desc': 'Unique identification of the financial institution'}
        }
    },
    'intrmyAgt3': {
        'name': 'Intermediary Agent 3',
        'description': 'Third intermediary agent in the payment chain',
        'required': ['FinInstnId'],
        'optional': ['BrnchId'],
        'fields': {
            'FinInstnId': {'type': 'FinancialInstitutionIdentification', 'desc': 'Unique identification of the financial institution'}
        }
    },
    'rmtInf': {
        'name': 'Remittance Information',
        'description': 'Information supplied to enable matching/reconciliation',
        'required': [],
        'optional': ['Ustrd', 'Strd'],
        'fields': {
            'Ustrd': {'type': 'string or array', 'min': 1, 'max': 140, 'desc': 'Unstructured remittance information'}
        }
    }
}


class ISO20022Validator:
    """ISO 20022 validator with context-aware ACE repair suggestions"""
    
    def __init__(self, context_spec_path: Optional[str] = None,
                 repair_lookup_path: Optional[str] = None,
                 use_context: bool = False,
                 merge_specs: bool = False):
        self.results: List[ValidationResult] = []
        self.entity_results: Dict[str, List[ValidationResult]] = {}
        self.context_specs = {}
        self.use_context = use_context
        self.merge_specs = merge_specs
        self.active_spec = ISO20022_SPECS.copy()
        self.matched_context = None
        self.payment_context = None  # Store payment context for repair lookup
        
        # Repair lookup
        self.repair_lookup: Optional[RepairLookup] = None
        self.repair_suggestions: Dict[str, List[Tuple[ValidationResult, RepairInfo]]] = {}
        
        # Load repair lookup if provided
        if repair_lookup_path and Path(repair_lookup_path).exists():
            try:
                self.repair_lookup = RepairLookup(repair_lookup_path)
                print(f"Loaded repair lookup: {len(self.repair_lookup.all_repairs)} repairs")
            except Exception as e:
                print(f"Warning: Could not load repair lookup: {e}")
        
        # Load context-aware spec if provided
        if context_spec_path and Path(context_spec_path).exists():
            self.load_context_spec(context_spec_path)
    
    def load_context_spec(self, spec_path: str):
        """Load context-aware specifications from JSON"""
        with open(spec_path, 'r') as f:
            self.context_specs = json.load(f)
    
    def match_context(self, payment: Dict) -> Optional[str]:
        """Match payment to a context specification"""
        if not self.context_specs:
            return None
        
        source = payment.get('source', 'UNKNOWN')
        clearing = payment.get('clearing', 'UNKNOWN')
        parties = payment.get('parties', {})
        
        active_parties = sorted([k for k, v in parties.items() if v])
        parties_str = ','.join(active_parties) if active_parties else 'none'
        
        context_key = f"{source}|{clearing}|{parties_str}"
        if context_key in self.context_specs:
            return context_key
        
        for spec_key, spec in self.context_specs.items():
            parts = spec_key.split('|')
            spec_source = parts[0] if len(parts) > 0 else ''
            spec_clearing = parts[1] if len(parts) > 1 else ''
            spec_parties_str = parts[2] if len(parts) > 2 else ''
            spec_parties = set(spec_parties_str.split(',')) if spec_parties_str else set()
            
            if source == spec_source and clearing == spec_clearing:
                payment_parties = set(active_parties)
                if payment_parties == spec_parties or payment_parties.issubset(spec_parties):
                    return spec_key
        
        return None
    
    def apply_context_spec(self, context_key: str):
        """Apply context-specific requirements to active spec"""
        if context_key not in self.context_specs:
            return
        
        context_spec = self.context_specs[context_key]
        self.matched_context = {
            'key': context_key,
            'source': context_spec['source'],
            'clearing': context_spec['clearing'],
            'parties': context_spec['parties'],
            'payment_count': context_spec.get('payment_count', 'N/A')
        }
        
        for entity_name, entity_spec in context_spec.get('entities', {}).items():
            if entity_name in self.active_spec:
                if self.merge_specs:
                    iso_req = set(self.active_spec[entity_name]['required'])
                    ctx_req = set(entity_spec.get('required', []))
                    self.active_spec[entity_name]['required'] = sorted(iso_req | ctx_req)
                else:
                    self.active_spec[entity_name]['required'] = entity_spec.get('required', [])
                    self.active_spec[entity_name]['optional'] = entity_spec.get('optional', [])
    
    def add_result(self, entity: str, level: ValidationLevel, field: str, message: str, value: Any = None):
        result = ValidationResult(level, field, message, value)
        if entity not in self.entity_results:
            self.entity_results[entity] = []
        self.entity_results[entity].append(result)
        self.results.append(result)
    
    def clear(self):
        self.results.clear()
        self.entity_results.clear()
        self.repair_suggestions.clear()
    
    # ========================================================================
    # REPAIR LOOKUP FUNCTIONS
    # ========================================================================
    
    def find_repairs_for_errors(self):
        """Find context-aware repair suggestions for all validation errors"""
        if not self.repair_lookup:
            return
        
        self.repair_suggestions.clear()
        
        for entity_name in self.entity_results:
            for result in self.entity_results[entity_name]:
                if result.level in [ValidationLevel.ERROR, ValidationLevel.WARNING]:
                    repair = self.repair_lookup.find_repair(
                        entity_name,
                        result.field,
                        result.message,
                        context=self.payment_context
                    )
                    
                    if repair:
                        if entity_name not in self.repair_suggestions:
                            self.repair_suggestions[entity_name] = []
                        self.repair_suggestions[entity_name].append((result, repair))
    
    def print_repair_suggestions(self):
        """Print ACE repair code suggestions"""
        if not self.repair_suggestions:
            return
        
        print(f"\n{'='*80}")
        print("ACE REPAIR SUGGESTIONS")
        print(f"{'='*80}")
        
        # Show context if available
        if self.payment_context:
            print(f"\nPayment Context:")
            print(f"  Source: {self.payment_context.get('source', 'N/A')}")
            print(f"  Clearing: {self.payment_context.get('clearing', 'N/A')}")
            print(f"  Parties: {', '.join(self.payment_context.get('parties', []))}")
        
        total_suggestions = sum(len(repairs) for repairs in self.repair_suggestions.values())
        print(f"\nFound {total_suggestions} ACE repair(s)\n")
        
        for entity_name in sorted(self.repair_suggestions.keys()):
            repairs = self.repair_suggestions[entity_name]
            
            print(f"{'-'*80}")
            print(f"Entity: {entity_name}")
            print(f"{'-'*80}\n")
            
            for result, repair in repairs:
                # Format ACE code with repair ID and severity
                ace_id = repair.repair_id if repair.repair_id else "UNKNOWN"
                severity = repair.code if repair.code else "?"
                
                print(f"  Field: {result.field}")
                print(f"  Error: {result.message}")
                print(f"  ACE {ace_id} ({severity}): {repair.text}")
                print(f"  [Entity: {repair.entity}, Repair Field: {repair.field}]")
                print()
    
    # ========================================================================
    # FIELD VALIDATORS
    # ========================================================================
    
    def check_string(self, entity: str, field: str, value: Any, min_len: int, max_len: int, 
                     no_dashes: bool = False) -> bool:
        """Check string field compliance"""
        if not isinstance(value, str):
            self.add_result(entity, ValidationLevel.ERROR, field, f"Must be string, got {type(value).__name__}", value)
            return False
        
        if not value.strip():
            self.add_result(entity, ValidationLevel.ERROR, field, "Cannot be empty", value)
            return False
        
        if len(value) < min_len:
            self.add_result(entity, ValidationLevel.ERROR, field, f"Length {len(value)} < minimum {min_len}", value)
            return False
        
        if len(value) > max_len:
            self.add_result(entity, ValidationLevel.ERROR, field, f"Length {len(value)} > maximum {max_len}", value)
            return False
        
        if no_dashes and '-' in value:
            self.add_result(entity, ValidationLevel.ERROR, field, "Dashes not allowed", value)
            return False
        
        if value != value.strip():
            self.add_result(entity, ValidationLevel.WARNING, field, "Has leading/trailing whitespace", value)
        
        self.add_result(entity, ValidationLevel.PASS, field, f"Valid ({len(value)} chars)")
        return True
    
    def check_currency(self, entity: str, field: str, value: Any) -> bool:
        """Check currency code"""
        if not isinstance(value, str):
            self.add_result(entity, ValidationLevel.ERROR, field, "Must be string", value)
            return False
        
        if len(value) != 3:
            self.add_result(entity, ValidationLevel.ERROR, field, f"Must be exactly 3 chars (ISO 4217), got {len(value)}", value)
            return False
        
        if not value.isalpha():
            self.add_result(entity, ValidationLevel.ERROR, field, "Must be letters only", value)
            return False
        
        if not value.isupper():
            self.add_result(entity, ValidationLevel.WARNING, field, "Should be uppercase", value)
        
        self.add_result(entity, ValidationLevel.PASS, field, f"Valid ISO 4217 code")
        return True
    
    def check_country(self, entity: str, field: str, value: Any) -> bool:
        """Check country code"""
        if not isinstance(value, str):
            self.add_result(entity, ValidationLevel.ERROR, field, "Must be string", value)
            return False
        
        if len(value) != 2:
            self.add_result(entity, ValidationLevel.ERROR, field, f"Must be exactly 2 chars (ISO 3166), got {len(value)}", value)
            return False
        
        if not value.isalpha():
            self.add_result(entity, ValidationLevel.ERROR, field, "Must be letters only", value)
            return False
        
        if not value.isupper():
            self.add_result(entity, ValidationLevel.WARNING, field, "Should be uppercase", value)
        
        self.add_result(entity, ValidationLevel.PASS, field, f"Valid ISO 3166 code")
        return True
    
    def check_bic(self, entity: str, field: str, value: Any) -> bool:
        """Check BIC/SWIFT code"""
        if not isinstance(value, str):
            self.add_result(entity, ValidationLevel.ERROR, field, "Must be string", value)
            return False
        
        bic = value.replace(' ', '')
        
        if '-' in bic:
            self.add_result(entity, ValidationLevel.ERROR, field, "BIC cannot contain dashes", value)
            return False
        
        if len(bic) not in [8, 11]:
            self.add_result(entity, ValidationLevel.ERROR, field, f"Must be 8 or 11 chars, got {len(bic)}", value)
            return False
        
        if not bic[:4].isalpha():
            self.add_result(entity, ValidationLevel.ERROR, field, "Bank code (chars 1-4) must be letters", value)
            return False
        
        if not bic[4:6].isalpha():
            self.add_result(entity, ValidationLevel.ERROR, field, "Country code (chars 5-6) must be letters", value)
            return False
        
        if not bic[6:8].isalnum():
            self.add_result(entity, ValidationLevel.ERROR, field, "Location code (chars 7-8) must be alphanumeric", value)
            return False
        
        self.add_result(entity, ValidationLevel.PASS, field, f"Valid BIC ({len(bic)} chars)")
        return True
    
    def check_routing(self, entity: str, field: str, value: Any) -> bool:
        """Check US routing number"""
        if not isinstance(value, str):
            self.add_result(entity, ValidationLevel.ERROR, field, "Must be string", value)
            return False
        
        if '-' in value:
            self.add_result(entity, ValidationLevel.ERROR, field, "Routing number cannot contain dashes", value)
            return False
        
        if len(value) != 9:
            self.add_result(entity, ValidationLevel.ERROR, field, f"Must be exactly 9 digits, got {len(value)}", value)
            return False
        
        if not value.isdigit():
            self.add_result(entity, ValidationLevel.ERROR, field, "Must be numeric only", value)
            return False
        
        self.add_result(entity, ValidationLevel.PASS, field, "Valid ABA routing number")
        return True
    
    # ========================================================================
    # ENTITY VALIDATORS
    # ========================================================================
    
    def validate_party(self, entity_key: str, party: Dict, spec: Dict) -> bool:
        """Validate party entity"""
        for req_field in spec.get('required', []):
            key = self._find_key(party, [req_field, req_field[0].lower() + req_field[1:]])
            
            if not key:
                key_insensitive = self._find_key_case_insensitive(party, req_field)
                if key_insensitive:
                    self.add_result(entity_key, ValidationLevel.WARNING, req_field, 
                                  f"Found as '{key_insensitive}' - case mismatch (expected '{req_field}')")
                    key = key_insensitive
                else:
                    self.add_result(entity_key, ValidationLevel.ERROR, req_field, "REQUIRED field missing")
                    continue
            
            field_spec = spec['fields'].get(req_field, {})
            if field_spec.get('type') == 'string':
                self.check_string(entity_key, key, party[key], 
                                field_spec.get('min', 1), field_spec.get('max', 140))
            elif field_spec.get('type') == 'CountryCode':
                self.check_country(entity_key, key, party[key])
            elif field_spec.get('type') == 'PostalAddress':
                self.validate_postal_address(entity_key, key, party[key])
        
        for opt_field in spec.get('optional', []):
            key = self._find_key(party, [opt_field, opt_field[0].lower() + opt_field[1:]])
            
            if not key:
                key = self._find_key_case_insensitive(party, opt_field)
                if key:
                    self.add_result(entity_key, ValidationLevel.WARNING, opt_field,
                                  f"Found as '{key}' - case mismatch (expected '{opt_field}')")
            
            if key:
                if 'PstlAdr' in opt_field or 'pstlAdr' in opt_field:
                    self.validate_postal_address(entity_key, key, party[key])
                elif 'CtryOfRes' in opt_field:
                    self.check_country(entity_key, key, party[key])
        
        return True
    
    def validate_postal_address(self, entity_key: str, base_path: str, address: Dict) -> bool:
        """Validate postal address"""
        if not isinstance(address, dict):
            self.add_result(entity_key, ValidationLevel.ERROR, base_path, "Must be object")
            return False
        
        adr_key = self._find_key(address, ['AdrLine', 'adrLine'])
        if adr_key:
            adr_lines = address[adr_key]
            if isinstance(adr_lines, list):
                if len(adr_lines) > 7:
                    self.add_result(entity_key, ValidationLevel.ERROR, f"{base_path}.{adr_key}", 
                                  f"Maximum 7 lines, got {len(adr_lines)}")
                for i, line in enumerate(adr_lines):
                    self.check_string(entity_key, f"{base_path}.{adr_key}[{i}]", line, 1, 70)
            else:
                self.check_string(entity_key, f"{base_path}.{adr_key}", adr_lines, 1, 70)
        
        return True
    
    def validate_agent(self, entity_key: str, agent: Dict, spec: Dict) -> bool:
        """Validate financial institution agent"""
        for req_field in spec.get('required', []):
            key = self._find_key(agent, [req_field, req_field[0].lower() + req_field[1:]])
            
            if not key:
                key_insensitive = self._find_key_case_insensitive(agent, req_field)
                if key_insensitive:
                    self.add_result(entity_key, ValidationLevel.WARNING, req_field,
                        f"Found as '{key_insensitive}' - case mismatch (expected '{req_field}')")
                    key = key_insensitive
                else:
                    self.add_result(entity_key, ValidationLevel.ERROR, req_field, "REQUIRED field missing")
                    continue
        
        fin_key = self._find_key(agent, ['FinInstnId', 'finInstnId'])
        if not fin_key:
            self.add_result(entity_key, ValidationLevel.ERROR, 'FinInstnId', "REQUIRED field missing")
            return False
        
        fin_instn = agent[fin_key]
        if not isinstance(fin_instn, dict):
            self.add_result(entity_key, ValidationLevel.ERROR, fin_key, "Must be object")
            return False
        
        bic_key = self._find_key(fin_instn, ['BICFI', 'bicFi', 'BIC', 'bic'])
        if bic_key:
            self.check_bic(entity_key, f"{fin_key}.{bic_key}", fin_instn[bic_key])
        
        clr_key = self._find_key(fin_instn, ['ClrSysMmbId', 'clrSysMmbId'])
        if clr_key:
            self.validate_clearing_system(entity_key, f"{fin_key}.{clr_key}", fin_instn[clr_key])
        
        nm_key = self._find_key(fin_instn, ['Nm', 'nm'])
        if nm_key:
            self.check_string(entity_key, f"{fin_key}.{nm_key}", fin_instn[nm_key], 1, 140)
        
        addr_key = self._find_key(fin_instn, ['PstlAdr', 'pstlAdr'])
        if addr_key:
            self.validate_postal_address(entity_key, f"{fin_key}.{addr_key}", fin_instn[addr_key])
        
        return True
    
    def validate_clearing_system(self, entity_key: str, base_path: str, clr_sys: Dict) -> bool:
        """Validate clearing system member ID"""
        if not isinstance(clr_sys, dict):
            self.add_result(entity_key, ValidationLevel.ERROR, base_path, "Must be object")
            return False
        
        clr_id_key = self._find_key(clr_sys, ['ClrSysId', 'clrSysId'])
        clr_code = None
        if clr_id_key:
            clr_sys_id = clr_sys[clr_id_key]
            if isinstance(clr_sys_id, dict):
                cd_key = self._find_key(clr_sys_id, ['Cd', 'cd'])
                if cd_key:
                    clr_code = clr_sys_id[cd_key]
            elif isinstance(clr_sys_id, str):
                clr_code = clr_sys_id
        
        mmb_key = self._find_key(clr_sys, ['MmbId', 'mmbId'])
        if not mmb_key:
            self.add_result(entity_key, ValidationLevel.ERROR, f"{base_path}.MmbId", "REQUIRED field missing")
            return False
        
        if clr_code == 'USABA':
            self.check_routing(entity_key, f"{base_path}.{mmb_key}", clr_sys[mmb_key])
        else:
            self.check_string(entity_key, f"{base_path}.{mmb_key}", clr_sys[mmb_key], 1, 35, no_dashes=True)
        
        return True
    
    def validate_account(self, entity_key: str, account: Dict, spec: Dict) -> bool:
        """Validate account"""
        for req_field in spec.get('required', []):
            key = self._find_key(account, [req_field, req_field[0].lower() + req_field[1:]])
            
            if not key:
                key = self._find_key_case_insensitive(account, req_field)
                if key:
                    self.add_result(entity_key, ValidationLevel.WARNING, req_field,
                                  f"Found as '{key}' - case mismatch (expected '{req_field}')")
                else:
                    self.add_result(entity_key, ValidationLevel.ERROR, req_field, "REQUIRED field missing")
                    if req_field in ['Id', 'id']:
                        return False
        
        id_key = self._find_key(account, ['Id', 'id'])
        if not id_key:
            self.add_result(entity_key, ValidationLevel.ERROR, 'Id', "REQUIRED field missing")
            return False
        
        acct_id = account[id_key]
        if not isinstance(acct_id, dict):
            self.add_result(entity_key, ValidationLevel.ERROR, id_key, "Must be object")
            return False
        
        iban_key = self._find_key(acct_id, ['IBAN', 'iban'])
        oth_key = self._find_key(acct_id, ['Othr', 'othr'])
        
        if not iban_key and not oth_key:
            self.add_result(entity_key, ValidationLevel.ERROR, f"{id_key}", "Must have IBAN OR Othr")
            return False
        
        if oth_key:
            other = acct_id[oth_key]
            if isinstance(other, dict):
                other_id_key = self._find_key(other, ['Id', 'id'])
                if not other_id_key:
                    self.add_result(entity_key, ValidationLevel.ERROR, f"{id_key}.{oth_key}.Id", "REQUIRED field missing")
                else:
                    self.check_string(entity_key, f"{id_key}.{oth_key}.{other_id_key}", 
                                    other[other_id_key], 1, 34, no_dashes=True)
        
        ccy_key = self._find_key(account, ['Ccy', 'ccy'])
        if ccy_key:
            self.check_currency(entity_key, ccy_key, account[ccy_key])
        
        tp_key = self._find_key(account, ['Tp', 'tp'])
        if tp_key:
            self.add_result(entity_key, ValidationLevel.PASS, tp_key, "Present")
        
        return True
    
    # ========================================================================
    # MAIN VALIDATION
    # ========================================================================
    
    def _find_key(self, obj: Dict, candidates: List[str]) -> Optional[str]:
        """Find first matching key from candidates"""
        for key in candidates:
            if key in obj:
                return key
        return None
    
    def _find_key_case_insensitive(self, obj: Dict, target: str) -> Optional[str]:
        """Find key in object with case-insensitive matching"""
        if not isinstance(obj, dict):
            return None
        target_lower = target.lower()
        for key in obj.keys():
            if key.lower() == target_lower:
                return key
        return None
    
    def validate_payment(self, payment: Dict) -> Dict:
        """Validate entire payment and find repair suggestions"""
        self.clear()
        
        # Extract and store payment context
        source = payment.get('source', 'UNKNOWN')
        clearing = payment.get('clearing', 'UNKNOWN')
        parties = payment.get('parties', {})
        active_parties = sorted([k for k, v in parties.items() if v])
        
        self.payment_context = {
            'source': source,
            'clearing': clearing,
            'parties': active_parties
        }
        
        # Match context if using context-aware validation
        if self.use_context and self.context_specs:
            context_key = self.match_context(payment)
            if context_key:
                self.apply_context_spec(context_key)
            else:
                self.add_result('_system', ValidationLevel.WARNING, 'context', 
                              'No matching context found, using default spec')
        
        entities = [
            ('cdtr', 'Cdtr', self.validate_party),
            ('dbtr', 'Dbtr', self.validate_party),
            ('cdtrAgt', 'CdtrAgt', self.validate_agent),
            ('dbtrAgt', 'DbtrAgt', self.validate_agent),
            ('cdtrAcct', 'CdtrAcct', self.validate_account),
            ('dbtrAcct', 'DbtrAcct', self.validate_account),
            ('instgAgt', 'InstgAgt', self.validate_agent),
            ('instdAgt', 'InstdAgt', self.validate_agent),
            ('intrmyAgt1', 'IntrmyAgt1', self.validate_agent),
            ('intrmyAgt2', 'IntrmyAgt2', self.validate_agent),
            ('intrmyAgt3', 'IntrmyAgt3', self.validate_agent),
        ]
        
        for camel, pascal, validator in entities:
            if camel in self.active_spec:
                spec = self.active_spec[camel]
                key = self._find_key(payment, [camel, pascal])
                
                if key:
                    validator(key, payment[key], spec)
                else:
                    self.add_result(camel, ValidationLevel.INFO, camel, "Entity not found in input")
        
        # Find repair suggestions after validation
        if self.repair_lookup:
            self.find_repairs_for_errors()
        
        return self.get_summary()
    
    def print_entity_validation(self, payment: Dict):
        """Print detailed entity-by-entity validation"""
        print("\n" + "="*80)
        print(f"VALIDATION REPORT")
        print("="*80)
        
        if self.matched_context:
            print(f"\nContext: {self.matched_context['source']} | {self.matched_context['clearing']}")
            print(f"Parties: {', '.join(self.matched_context['parties'])}")
        
        for entity_name, spec in self.active_spec.items():
            print(f"\n{'='*80}")
            print(f"ENTITY: {spec['name']}")
            print(f"{'='*80}")
            
            print(f"\nREQUIREMENTS:")
            print(f"  Required: {', '.join(spec['required']) if spec['required'] else 'None'}")
            
            key = self._find_key(payment, [entity_name, entity_name.capitalize()])
            if key:
                print(f"\nINPUT DATA:")
                self._print_structure(payment[key], indent=2)
            else:
                print(f"\nINPUT DATA: ✗ NOT FOUND")
            
            print(f"\nVALIDATION RESULTS:")
            if entity_name in self.entity_results or key in self.entity_results:
                results = self.entity_results.get(entity_name, []) + self.entity_results.get(key, [])
                if results:
                    for result in results:
                        print(str(result))
                else:
                    print("  ℹ No validation performed")
            else:
                print("  ℹ Entity not validated")
    
    def _print_structure(self, obj: Any, indent: int = 0):
        """Print object structure"""
        prefix = " " * indent
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    print(f"{prefix}{k}:")
                    self._print_structure(v, indent + 2)
                else:
                    print(f"{prefix}{k}: {v}")
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                print(f"{prefix}[{i}]:")
                self._print_structure(item, indent + 2)
        else:
            print(f"{prefix}{obj}")
    
    def get_summary(self) -> Dict:
        """Get validation summary"""
        errors = [r for r in self.results if r.level == ValidationLevel.ERROR]
        warnings = [r for r in self.results if r.level == ValidationLevel.WARNING]
        passed = [r for r in self.results if r.level == ValidationLevel.PASS]
        
        return {
            'valid': len(errors) == 0,
            'fields_checked': len(self.results),
            'fields_passed': len(passed),
            'error_count': len(errors),
            'warning_count': len(warnings),
            'errors': [str(e) for e in errors],
            'warnings': [str(w) for w in warnings],
            'repair_suggestions': len(self.repair_suggestions)
        }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='ISO 20022 Validator with ACE Repair Suggestions')
    parser.add_argument('input', help='Input JSON file')
    parser.add_argument('--context-spec', help='Path to context_specs.json')
    parser.add_argument('--repair-lookup', help='Path to repair_lookup.pkl')
    parser.add_argument('--use-context', action='store_true')
    parser.add_argument('--merge', action='store_true')
    
    args = parser.parse_args()
    
    if args.use_context and not args.context_spec:
        parser.error("--use-context requires --context-spec")
    
    with open(args.input, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        if len(data) == 1:
            payment = data[list(data.keys())[0]]
        else:
            payment = data
    else:
        print("Error: Invalid JSON")
        return
    
    validator = ISO20022Validator(
        context_spec_path=args.context_spec,
        repair_lookup_path=args.repair_lookup,
        use_context=args.use_context,
        merge_specs=args.merge
    )
    
    summary = validator.validate_payment(payment)
    validator.print_entity_validation(payment)
    
    # Print ACE repair suggestions
    if validator.repair_lookup:
        validator.print_repair_suggestions()
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Status: {'✓ VALID' if summary['valid'] else '✗ INVALID'}")
    print(f"Errors: {summary['error_count']}")
    print(f"Warnings: {summary['warning_count']}")
    if validator.repair_lookup:
        print(f"ACE Repairs: {summary['repair_suggestions']}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
