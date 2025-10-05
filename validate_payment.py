#!/usr/bin/env python3
"""
ISO 20022 Comprehensive Entity Validator
Shows spec, validates structure, checks every field
Version 2
"""

import json
import re
from typing import Dict, List, Optional, Any, Set
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
    """Comprehensive ISO 20022 validator with entity-by-entity output"""
    
    def __init__(self):
        self.results: List[ValidationResult] = []
        self.entity_results: Dict[str, List[ValidationResult]] = {}
    
    def add_result(self, entity: str, level: ValidationLevel, field: str, message: str, value: Any = None):
        result = ValidationResult(level, field, message, value)
        if entity not in self.entity_results:
            self.entity_results[entity] = []
        self.entity_results[entity].append(result)
        self.results.append(result)
    
    def clear(self):
        self.results.clear()
        self.entity_results.clear()
    
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
        # Check required fields
        for req_field in spec['required']:
            key = self._find_key(party, [req_field, req_field[0].lower() + req_field[1:]])
            if not key:
                self.add_result(entity_key, ValidationLevel.ERROR, req_field, "REQUIRED field missing")
            else:
                field_spec = spec['fields'].get(req_field, {})
                if field_spec.get('type') == 'string':
                    self.check_string(entity_key, key, party[key], 
                                    field_spec.get('min', 1), field_spec.get('max', 140))
                elif field_spec.get('type') == 'CountryCode':
                    self.check_country(entity_key, key, party[key])
        
        # Check optional fields if present
        for opt_field in spec['optional']:
            key = self._find_key(party, [opt_field, opt_field[0].lower() + opt_field[1:]])
            if key:
                if opt_field == 'PstlAdr' or opt_field == 'pstlAdr':
                    self.validate_postal_address(entity_key, key, party[key])
                elif opt_field == 'CtryOfRes':
                    self.check_country(entity_key, key, party[key])
        
        return True
    
    def validate_postal_address(self, entity_key: str, base_path: str, address: Dict) -> bool:
        """Validate postal address"""
        if not isinstance(address, dict):
            self.add_result(entity_key, ValidationLevel.ERROR, base_path, "Must be object")
            return False
        
        # Address Lines
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
        # FinInstnId required
        fin_key = self._find_key(agent, ['FinInstnId', 'finInstnId'])
        if not fin_key:
            self.add_result(entity_key, ValidationLevel.ERROR, 'FinInstnId', "REQUIRED field missing")
            return False
        
        fin_instn = agent[fin_key]
        if not isinstance(fin_instn, dict):
            self.add_result(entity_key, ValidationLevel.ERROR, fin_key, "Must be object")
            return False
        
        # Check BIC
        bic_key = self._find_key(fin_instn, ['BICFI', 'bicFi', 'BIC', 'bic'])
        if bic_key:
            self.check_bic(entity_key, f"{fin_key}.{bic_key}", fin_instn[bic_key])
        
        # Check Clearing System
        clr_key = self._find_key(fin_instn, ['ClrSysMmbId', 'clrSysMmbId'])
        if clr_key:
            self.validate_clearing_system(entity_key, f"{fin_key}.{clr_key}", fin_instn[clr_key])
        
        # Check Name
        nm_key = self._find_key(fin_instn, ['Nm', 'nm'])
        if nm_key:
            self.check_string(entity_key, f"{fin_key}.{nm_key}", fin_instn[nm_key], 1, 140)
        
        # Check Address
        addr_key = self._find_key(fin_instn, ['PstlAdr', 'pstlAdr'])
        if addr_key:
            self.validate_postal_address(entity_key, f"{fin_key}.{addr_key}", fin_instn[addr_key])
        
        return True
    
    def validate_clearing_system(self, entity_key: str, base_path: str, clr_sys: Dict) -> bool:
        """Validate clearing system member ID"""
        if not isinstance(clr_sys, dict):
            self.add_result(entity_key, ValidationLevel.ERROR, base_path, "Must be object")
            return False
        
        # Get clearing system code
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
        
        # Member ID required
        mmb_key = self._find_key(clr_sys, ['MmbId', 'mmbId'])
        if not mmb_key:
            self.add_result(entity_key, ValidationLevel.ERROR, f"{base_path}.MmbId", "REQUIRED field missing")
            return False
        
        # Validate based on clearing system
        if clr_code == 'USABA':
            self.check_routing(entity_key, f"{base_path}.{mmb_key}", clr_sys[mmb_key])
        else:
            self.check_string(entity_key, f"{base_path}.{mmb_key}", clr_sys[mmb_key], 1, 35, no_dashes=True)
        
        return True
    
    def validate_account(self, entity_key: str, account: Dict, spec: Dict) -> bool:
        """Validate account"""
        # Id required
        id_key = self._find_key(account, ['Id', 'id'])
        if not id_key:
            self.add_result(entity_key, ValidationLevel.ERROR, 'Id', "REQUIRED field missing")
            return False
        
        acct_id = account[id_key]
        if not isinstance(acct_id, dict):
            self.add_result(entity_key, ValidationLevel.ERROR, id_key, "Must be object")
            return False
        
        # IBAN or Othr required
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
        
        # Currency
        ccy_key = self._find_key(account, ['Ccy', 'ccy'])
        if ccy_key:
            self.check_currency(entity_key, ccy_key, account[ccy_key])
        
        # Type
        tp_key = self._find_key(account, ['Tp', 'tp'])
        if tp_key:
            self.add_result(entity_key, ValidationLevel.PASS, tp_key, "Present")
        
        return True
    
    # ========================================================================
    # MAIN VALIDATION
    # ========================================================================
    
    def validate_payment(self, payment: Dict) -> Dict:
        """Validate entire payment"""
        self.clear()
        
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
            if camel in ISO20022_SPECS:
                spec = ISO20022_SPECS[camel]
                key = self._find_key(payment, [camel, pascal])
                
                if key:
                    validator(key, payment[key], spec)
                else:
                    self.add_result(camel, ValidationLevel.INFO, camel, "Entity not found in input")
        
        return self.get_summary()
    
    def print_entity_validation(self, payment: Dict):
        """Print detailed entity-by-entity validation"""
        print("\n" + "="*80)
        print("ENTITY-BY-ENTITY VALIDATION REPORT")
        print("="*80)
        
        for entity_name, spec in ISO20022_SPECS.items():
            print(f"\n{'='*80}")
            print(f"ENTITY: {spec['name']}")
            print(f"{'='*80}")
            print(f"Description: {spec.get('description', 'N/A')}")
            
            # Show ISO spec
            print(f"\nISO 20022 SPECIFICATION:")
            print(f"  Required fields: {', '.join(spec['required']) if spec['required'] else 'None'}")
            print(f"  Optional fields: {', '.join(spec['optional']) if spec['optional'] else 'None'}")
            
            print(f"\n  Field Details:")
            for field_name, field_spec in spec.get('fields', {}).items():
                ftype = field_spec.get('type', 'unknown')
                fformat = field_spec.get('format', '')
                format_str = f" ({fformat})" if fformat else ""
                min_max = ""
                if 'min' in field_spec and 'max' in field_spec:
                    min_max = f" [{field_spec['min']}-{field_spec['max']} chars]"
                desc = field_spec.get('desc', '')
                print(f"    • {field_name}: {ftype}{format_str}{min_max}")
                print(f"      {desc}")
            
            # Show nested structure if any
            if 'nested' in spec:
                print(f"\n  Nested Structure:")
                for nest_path, nest_spec in spec['nested'].items():
                    print(f"    {nest_path}:")
                    if 'required_one_of' in nest_spec:
                        print(f"      Required (one of): {', '.join(nest_spec['required_one_of'])}")
                    if 'required' in nest_spec:
                        print(f"      Required: {', '.join(nest_spec['required'])}")
                    for nf, nf_spec in nest_spec.get('fields', {}).items():
                        print(f"        • {nf}: {nf_spec.get('desc', '')}")
            
            # Show input data
            key = self._find_key(payment, [entity_name, entity_name.capitalize()])
            if key:
                print(f"\nINPUT DATA:")
                self._print_structure(payment[key], indent=2)
            else:
                print(f"\nINPUT DATA: ✗ NOT FOUND")
            
            # Show validation results
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
    
    def _find_key(self, obj: Dict, candidates: List[str]) -> Optional[str]:
        """Find first matching key"""
        for key in candidates:
            if key in obj:
                return key
        return None
    
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
            'warnings': [str(w) for w in warnings]
        }


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='ISO 20022 Entity Validator')
    parser.add_argument('input', help='Input JSON file')
    
    args = parser.parse_args()
    
    with open(args.input, 'r') as f:
        data = json.load(f)
    
    # Extract payment
    if isinstance(data, dict):
        if len(data) == 1:
            payment = data[list(data.keys())[0]]
        else:
            payment = data
    else:
        print("Error: Invalid JSON")
        return
    
    # Validate
    validator = ISO20022Validator()
    summary = validator.validate_payment(payment)
    
    # Print entity-by-entity report
    validator.print_entity_validation(payment)
    
    # Print summary
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}")
    print(f"Status: {'✓ VALID' if summary['valid'] else '✗ INVALID'}")
    print(f"Fields Checked: {summary['fields_checked']}")
    print(f"Fields Passed: {summary['fields_passed']}")
    print(f"Errors: {summary['error_count']}")
    print(f"Warnings: {summary['warning_count']}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
