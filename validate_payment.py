#!/usr/bin/env python3
"""
ISO 20022 Deep Field Validator
Validates every field with format, length, and character restrictions
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


@dataclass
class ValidationResult:
    """Result of a validation check"""
    level: ValidationLevel
    field: str
    message: str
    value: Any = None
    
    def __str__(self):
        value_str = f" (value: '{self.value}')" if self.value else ""
        return f"[{self.level.value}] {self.field}: {self.message}{value_str}"


class ISO20022Validator:
    """Deep validator for ISO 20022 payment entities"""
    
    def __init__(self):
        self.results: List[ValidationResult] = []
        self.field_count = 0
        self.valid_count = 0
        self.error_count = 0
    
    def add_error(self, field: str, message: str, value: Any = None):
        self.results.append(ValidationResult(ValidationLevel.ERROR, field, message, value))
        self.error_count += 1
    
    def add_warning(self, field: str, message: str, value: Any = None):
        self.results.append(ValidationResult(ValidationLevel.WARNING, field, message, value))
    
    def add_info(self, field: str, message: str, value: Any = None):
        self.results.append(ValidationResult(ValidationLevel.INFO, field, message, value))
    
    def add_valid(self, field: str):
        self.valid_count += 1
    
    def get_errors(self) -> List[ValidationResult]:
        return [r for r in self.results if r.level == ValidationLevel.ERROR]
    
    def get_warnings(self) -> List[ValidationResult]:
        return [r for r in self.results if r.level == ValidationLevel.WARNING]
    
    def is_valid(self) -> bool:
        return len(self.get_errors()) == 0
    
    def clear(self):
        self.results.clear()
        self.field_count = 0
        self.valid_count = 0
        self.error_count = 0
    
    # ========================================================================
    # DEEP FIELD VALIDATORS
    # ========================================================================
    
    def validate_string_field(self, value: Any, field: str, min_len: int = 1, max_len: int = 140, 
                             pattern: str = None, allow_dash: bool = True, 
                             numeric_only: bool = False, alpha_only: bool = False) -> bool:
        """Deep validation of string field"""
        self.field_count += 1
        
        # Type check
        if not isinstance(value, str):
            self.add_error(field, f"Must be string, got {type(value).__name__}", value)
            return False
        
        # Empty check
        if not value.strip():
            self.add_error(field, "Cannot be empty or whitespace only", value)
            return False
        
        # Length check
        if len(value) < min_len:
            self.add_error(field, f"Length {len(value)} is below minimum {min_len}", value)
            return False
        
        if len(value) > max_len:
            self.add_error(field, f"Length {len(value)} exceeds maximum {max_len}", value)
            return False
        
        # Dash check
        if not allow_dash and '-' in value:
            self.add_error(field, f"Dashes not allowed in this field", value)
            return False
        
        # Numeric check
        if numeric_only:
            if not value.replace('-', '').replace(' ', '').isdigit():
                self.add_error(field, f"Must contain only digits", value)
                return False
            
            # Check for dashes in numeric fields
            if '-' in value:
                self.add_warning(field, f"Numeric field contains dashes", value)
        
        # Alpha check
        if alpha_only:
            if not value.replace(' ', '').replace('-', '').isalpha():
                self.add_error(field, f"Must contain only letters", value)
                return False
        
        # Pattern check
        if pattern:
            if not re.match(pattern, value):
                self.add_error(field, f"Does not match required pattern: {pattern}", value)
                return False
        
        # Check for common issues
        if value != value.strip():
            self.add_warning(field, f"Has leading/trailing whitespace", value)
        
        if '  ' in value:
            self.add_warning(field, f"Contains multiple consecutive spaces", value)
        
        # Check for non-printable characters
        if any(ord(c) < 32 or ord(c) > 126 for c in value if c not in ['\n', '\r', '\t']):
            self.add_warning(field, f"Contains non-printable characters", value)
        
        self.add_valid(field)
        return True
    
    def validate_currency(self, value: Any, field: str) -> bool:
        """Validate ISO 4217 currency code"""
        self.field_count += 1
        
        if not isinstance(value, str):
            self.add_error(field, f"Currency must be string", value)
            return False
        
        if len(value) != 3:
            self.add_error(field, f"Currency must be exactly 3 characters (ISO 4217), got {len(value)}", value)
            return False
        
        if not value.isalpha():
            self.add_error(field, f"Currency must contain only letters", value)
            return False
        
        if not value.isupper():
            self.add_warning(field, f"Currency should be uppercase (ISO 4217)", value)
        
        # Common currencies check
        common_currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CHF', 'CAD', 'AUD', 'CNY']
        if value.upper() not in common_currencies:
            self.add_info(field, f"Uncommon currency code: {value}", value)
        
        self.add_valid(field)
        return True
    
    def validate_country_code(self, value: Any, field: str) -> bool:
        """Validate ISO 3166 country code"""
        self.field_count += 1
        
        if not isinstance(value, str):
            self.add_error(field, f"Country code must be string", value)
            return False
        
        if len(value) != 2:
            self.add_error(field, f"Country code must be exactly 2 characters (ISO 3166), got {len(value)}", value)
            return False
        
        if not value.isalpha():
            self.add_error(field, f"Country code must contain only letters", value)
            return False
        
        if not value.isupper():
            self.add_warning(field, f"Country code should be uppercase (ISO 3166)", value)
        
        self.add_valid(field)
        return True
    
    def validate_bic(self, value: Any, field: str) -> bool:
        """Validate BIC/SWIFT code"""
        self.field_count += 1
        
        if not isinstance(value, str):
            self.add_error(field, f"BIC must be string", value)
            return False
        
        # Remove spaces
        bic = value.replace(' ', '')
        
        if len(bic) not in [8, 11]:
            self.add_error(field, f"BIC must be 8 or 11 characters, got {len(bic)}", value)
            return False
        
        # Check for dashes (not allowed in BIC)
        if '-' in bic:
            self.add_error(field, f"BIC cannot contain dashes", value)
            return False
        
        # Format: AAAABBCCXXX
        if not bic[:4].isalpha():
            self.add_error(field, f"BIC bank code (first 4 chars) must be letters", value)
            return False
        
        if not bic[4:6].isalpha():
            self.add_error(field, f"BIC country code (chars 5-6) must be letters", value)
            return False
        
        if not bic[6:8].isalnum():
            self.add_error(field, f"BIC location code (chars 7-8) must be alphanumeric", value)
            return False
        
        if len(bic) == 11:
            if not bic[8:11].isalnum():
                self.add_error(field, f"BIC branch code (chars 9-11) must be alphanumeric", value)
                return False
        
        # Check if uppercase
        if not bic.isupper():
            self.add_warning(field, f"BIC should be uppercase", value)
        
        self.add_valid(field)
        return True
    
    def validate_iban(self, value: Any, field: str) -> bool:
        """Validate IBAN"""
        self.field_count += 1
        
        if not isinstance(value, str):
            self.add_error(field, f"IBAN must be string", value)
            return False
        
        # Remove spaces
        iban = value.replace(' ', '')
        
        # Check for dashes (not standard in IBAN)
        if '-' in iban:
            self.add_warning(field, f"IBAN should not contain dashes", value)
            iban = iban.replace('-', '')
        
        # Length: 15-34 characters
        if len(iban) < 15 or len(iban) > 34:
            self.add_error(field, f"IBAN must be 15-34 characters, got {len(iban)}", value)
            return False
        
        # First 2 chars: country code
        if not iban[:2].isalpha():
            self.add_error(field, f"IBAN must start with 2-letter country code", value)
            return False
        
        # Next 2 chars: check digits
        if not iban[2:4].isdigit():
            self.add_error(field, f"IBAN chars 3-4 must be check digits (numeric)", value)
            return False
        
        # Rest should be alphanumeric
        if not iban[4:].isalnum():
            self.add_error(field, f"IBAN contains invalid characters after check digits", value)
            return False
        
        if not iban.isupper():
            self.add_warning(field, f"IBAN should be uppercase", value)
        
        self.add_valid(field)
        return True
    
    def validate_routing_number(self, value: Any, field: str) -> bool:
        """Validate US ABA routing number"""
        self.field_count += 1
        
        if not isinstance(value, str):
            self.add_error(field, f"Routing number must be string", value)
            return False
        
        # Check for dashes (not allowed)
        if '-' in value:
            self.add_error(field, f"Routing number cannot contain dashes", value)
            return False
        
        # Must be exactly 9 digits
        if len(value) != 9:
            self.add_error(field, f"US routing number must be exactly 9 digits, got {len(value)}", value)
            return False
        
        if not value.isdigit():
            self.add_error(field, f"Routing number must be numeric only", value)
            return False
        
        self.add_valid(field)
        return True
    
    def validate_account_number(self, value: Any, field: str, max_len: int = 34) -> bool:
        """Validate account number"""
        self.field_count += 1
        
        if not isinstance(value, str):
            self.add_error(field, f"Account number must be string", value)
            return False
        
        if not value.strip():
            self.add_error(field, f"Account number cannot be empty", value)
            return False
        
        if len(value) > max_len:
            self.add_error(field, f"Account number exceeds maximum length {max_len}, got {len(value)}", value)
            return False
        
        # Check for dashes in numeric-only account numbers
        if value.replace('-', '').isdigit() and '-' in value:
            self.add_warning(field, f"Numeric account number contains dashes", value)
        
        # Account numbers should be alphanumeric
        if not value.replace('-', '').replace(' ', '').isalnum():
            self.add_warning(field, f"Account number contains special characters", value)
        
        self.add_valid(field)
        return True
    
    # ========================================================================
    # ENTITY VALIDATORS
    # ========================================================================
    
    def validate_party(self, party: Dict, base_path: str, party_type: str) -> bool:
        """Validate Party (Creditor/Debtor)"""
        if not isinstance(party, dict):
            self.add_error(base_path, f"{party_type} must be object/dict")
            return False
        
        valid = True
        
        # Name - REQUIRED
        name_key = self._find_key(party, ['Nm', 'nm'])
        if not name_key:
            self.add_error(f"{base_path}.Nm", "Name is REQUIRED")
            valid = False
        else:
            self.validate_string_field(party[name_key], f"{base_path}.{name_key}", max_len=140)
        
        # Postal Address - OPTIONAL
        addr_key = self._find_key(party, ['PstlAdr', 'pstlAdr'])
        if addr_key:
            self.validate_postal_address(party[addr_key], f"{base_path}.{addr_key}")
        
        # Country - OPTIONAL
        ctry_key = self._find_key(party, ['CtryOfRes', 'ctryOfRes'])
        if ctry_key:
            self.validate_country_code(party[ctry_key], f"{base_path}.{ctry_key}")
        
        return valid
    
    def validate_postal_address(self, address: Dict, base_path: str) -> bool:
        """Validate Postal Address"""
        if not isinstance(address, dict):
            self.add_error(base_path, "Postal address must be object/dict")
            return False
        
        # Country
        ctry_key = self._find_key(address, ['Ctry', 'ctry'])
        if ctry_key:
            self.validate_country_code(address[ctry_key], f"{base_path}.{ctry_key}")
        
        # Address Lines - max 7 lines, max 70 chars each
        adr_key = self._find_key(address, ['AdrLine', 'adrLine'])
        if adr_key:
            adr_lines = address[adr_key]
            if isinstance(adr_lines, list):
                if len(adr_lines) > 7:
                    self.add_error(f"{base_path}.{adr_key}", f"Maximum 7 address lines allowed, got {len(adr_lines)}")
                
                for i, line in enumerate(adr_lines):
                    self.validate_string_field(line, f"{base_path}.{adr_key}[{i}]", max_len=70)
            elif isinstance(adr_lines, str):
                self.validate_string_field(adr_lines, f"{base_path}.{adr_key}", max_len=70)
        
        # Post Code - max 16 chars
        pst_key = self._find_key(address, ['PstCd', 'pstCd'])
        if pst_key:
            self.validate_string_field(address[pst_key], f"{base_path}.{pst_key}", max_len=16)
        
        # Town Name - max 35 chars
        twn_key = self._find_key(address, ['TwnNm', 'twnNm'])
        if twn_key:
            self.validate_string_field(address[twn_key], f"{base_path}.{twn_key}", max_len=35)
        
        # Street Name - max 70 chars
        str_key = self._find_key(address, ['StrtNm', 'strtNm'])
        if str_key:
            self.validate_string_field(address[str_key], f"{base_path}.{str_key}", max_len=70)
        
        # Building Number - max 16 chars
        bld_key = self._find_key(address, ['BldgNb', 'bldgNb'])
        if bld_key:
            self.validate_string_field(address[bld_key], f"{base_path}.{bld_key}", max_len=16)
        
        return True
    
    def validate_agent(self, agent: Dict, base_path: str, agent_type: str) -> bool:
        """Validate Financial Institution Agent"""
        if not isinstance(agent, dict):
            self.add_error(base_path, f"{agent_type} must be object/dict")
            return False
        
        # FinInstnId - REQUIRED
        fin_key = self._find_key(agent, ['FinInstnId', 'finInstnId'])
        if not fin_key:
            self.add_error(f"{base_path}.FinInstnId", "Financial Institution ID is REQUIRED")
            return False
        
        fin_instn = agent[fin_key]
        if not isinstance(fin_instn, dict):
            self.add_error(f"{base_path}.{fin_key}", "Must be object/dict")
            return False
        
        has_id = False
        
        # BIC/SWIFT
        bic_key = self._find_key(fin_instn, ['BICFI', 'bicFi', 'BIC', 'bic'])
        if bic_key:
            self.validate_bic(fin_instn[bic_key], f"{base_path}.{fin_key}.{bic_key}")
            has_id = True
        
        # Clearing System Member ID
        clr_key = self._find_key(fin_instn, ['ClrSysMmbId', 'clrSysMmbId'])
        if clr_key:
            self.validate_clearing_system_member_id(fin_instn[clr_key], f"{base_path}.{fin_key}.{clr_key}")
            has_id = True
        
        # Other ID
        oth_key = self._find_key(fin_instn, ['Othr', 'othr'])
        if oth_key:
            other = fin_instn[oth_key]
            if isinstance(other, dict):
                id_key = self._find_key(other, ['Id', 'id'])
                if id_key:
                    self.validate_string_field(other[id_key], f"{base_path}.{fin_key}.{oth_key}.{id_key}", max_len=35)
                    has_id = True
        
        if not has_id:
            self.add_warning(f"{base_path}.{fin_key}", "No identification (BIC, ClrSysMmbId, or Othr) provided")
        
        # Name - max 140 chars
        nm_key = self._find_key(fin_instn, ['Nm', 'nm'])
        if nm_key:
            self.validate_string_field(fin_instn[nm_key], f"{base_path}.{fin_key}.{nm_key}", max_len=140)
        
        # Postal Address
        addr_key = self._find_key(fin_instn, ['PstlAdr', 'pstlAdr'])
        if addr_key:
            self.validate_postal_address(fin_instn[addr_key], f"{base_path}.{fin_key}.{addr_key}")
        
        # Country of Residence
        ctry_key = self._find_key(agent, ['CtryOfRes', 'ctryOfRes'])
        if ctry_key:
            self.validate_country_code(agent[ctry_key], f"{base_path}.{ctry_key}")
        
        return True
    
    def validate_clearing_system_member_id(self, clr_sys: Dict, base_path: str) -> bool:
        """Validate Clearing System Member ID"""
        if not isinstance(clr_sys, dict):
            self.add_error(base_path, "Must be object/dict")
            return False
        
        # Clearing System ID
        clr_id_key = self._find_key(clr_sys, ['ClrSysId', 'clrSysId'])
        clr_sys_code = None
        
        if clr_id_key:
            clr_sys_id = clr_sys[clr_id_key]
            if isinstance(clr_sys_id, dict):
                cd_key = self._find_key(clr_sys_id, ['Cd', 'cd'])
                if cd_key:
                    clr_sys_code = clr_sys_id[cd_key]
                    self.validate_string_field(clr_sys_code, f"{base_path}.{clr_id_key}.{cd_key}", max_len=35)
            elif isinstance(clr_sys_id, str):
                clr_sys_code = clr_sys_id
                self.validate_string_field(clr_sys_code, f"{base_path}.{clr_id_key}", max_len=35)
        
        # Member ID - REQUIRED
        mmb_key = self._find_key(clr_sys, ['MmbId', 'mmbId'])
        if not mmb_key:
            self.add_error(f"{base_path}.MmbId", "Member ID is REQUIRED")
            return False
        
        mmb_id = clr_sys[mmb_key]
        
        # Validate based on clearing system
        if clr_sys_code == 'USABA':
            self.validate_routing_number(mmb_id, f"{base_path}.{mmb_key}")
        else:
            self.validate_string_field(mmb_id, f"{base_path}.{mmb_key}", max_len=35, numeric_only=True, allow_dash=False)
        
        return True
    
    def validate_account(self, account: Dict, base_path: str, account_type: str) -> bool:
        """Validate Account"""
        if not isinstance(account, dict):
            self.add_error(base_path, f"{account_type} must be object/dict")
            return False
        
        # Id - REQUIRED
        id_key = self._find_key(account, ['Id', 'id'])
        if not id_key:
            self.add_error(f"{base_path}.Id", "Account ID is REQUIRED")
            return False
        
        acct_id = account[id_key]
        if not isinstance(acct_id, dict):
            self.add_error(f"{base_path}.{id_key}", "Account ID must be object/dict")
            return False
        
        # IBAN or Othr - at least one required
        iban_key = self._find_key(acct_id, ['IBAN', 'iban'])
        oth_key = self._find_key(acct_id, ['Othr', 'othr'])
        
        if not iban_key and not oth_key:
            self.add_error(f"{base_path}.{id_key}", "Must have either IBAN or Othr identification")
            return False
        
        if iban_key:
            self.validate_iban(acct_id[iban_key], f"{base_path}.{id_key}.{iban_key}")
        
        if oth_key:
            other = acct_id[oth_key]
            if isinstance(other, dict):
                other_id_key = self._find_key(other, ['Id', 'id'])
                if not other_id_key:
                    self.add_error(f"{base_path}.{id_key}.{oth_key}.Id", "ID is REQUIRED in Othr")
                else:
                    self.validate_account_number(other[other_id_key], f"{base_path}.{id_key}.{oth_key}.{other_id_key}")
        
        # Type - OPTIONAL, max 35 chars
        tp_key = self._find_key(account, ['Tp', 'tp'])
        if tp_key:
            tp = account[tp_key]
            if isinstance(tp, dict):
                prtry_key = self._find_key(tp, ['Prtry', 'prtry'])
                if prtry_key:
                    self.validate_string_field(tp[prtry_key], f"{base_path}.{tp_key}.{prtry_key}", max_len=35)
        
        # Currency - OPTIONAL
        ccy_key = self._find_key(account, ['Ccy', 'ccy'])
        if ccy_key:
            self.validate_currency(account[ccy_key], f"{base_path}.{ccy_key}")
        
        # Name - OPTIONAL, max 70 chars
        nm_key = self._find_key(account, ['Nm', 'nm'])
        if nm_key:
            self.validate_string_field(account[nm_key], f"{base_path}.{nm_key}", max_len=70)
        
        return True
    
    def validate_remittance_info(self, rmt_inf: Any, base_path: str) -> bool:
        """Validate Remittance Information"""
        if isinstance(rmt_inf, str):
            self.validate_string_field(rmt_inf, base_path, max_len=140)
            return True
        
        if isinstance(rmt_inf, dict):
            ustrd_key = self._find_key(rmt_inf, ['Ustrd', 'ustrd'])
            if ustrd_key:
                ustrd = rmt_inf[ustrd_key]
                if isinstance(ustrd, str):
                    self.validate_string_field(ustrd, f"{base_path}.{ustrd_key}", max_len=140)
                elif isinstance(ustrd, list):
                    for i, line in enumerate(ustrd):
                        self.validate_string_field(line, f"{base_path}.{ustrd_key}[{i}]", max_len=140)
            return True
        
        if isinstance(rmt_inf, list):
            for i, item in enumerate(rmt_inf):
                if isinstance(item, dict):
                    ustrd_key = self._find_key(item, ['Ustrd', 'ustrd'])
                    if ustrd_key:
                        ustrd = item[ustrd_key]
                        if isinstance(ustrd, str):
                            self.validate_string_field(ustrd, f"{base_path}[{i}].{ustrd_key}", max_len=140)
                        elif isinstance(ustrd, list):
                            for j, line in enumerate(ustrd):
                                self.validate_string_field(line, f"{base_path}[{i}].{ustrd_key}[{j}]", max_len=140)
                elif isinstance(item, str):
                    self.validate_string_field(item, f"{base_path}[{i}]", max_len=140)
            return True
        
        return True
    
    # ========================================================================
    # MAIN VALIDATION
    # ========================================================================
    
    def validate_payment(self, payment: Dict) -> Dict:
        """Validate entire payment with deep field validation"""
        self.clear()
        
        if not isinstance(payment, dict):
            self.add_error("payment", "Payment must be object/dict")
            return self.get_summary()
        
        # Validate each entity
        entities = [
            ('cdtr', 'Cdtr', self.validate_party, 'Creditor'),
            ('dbtr', 'Dbtr', self.validate_party, 'Debtor'),
            ('cdtrAgt', 'CdtrAgt', self.validate_agent, 'Creditor Agent'),
            ('dbtrAgt', 'DbtrAgt', self.validate_agent, 'Debtor Agent'),
            ('cdtrAcct', 'CdtrAcct', self.validate_account, 'Creditor Account'),
            ('dbtrAcct', 'DbtrAcct', self.validate_account, 'Debtor Account'),
            ('instgAgt', 'InstgAgt', self.validate_agent, 'Instructing Agent'),
            ('instdAgt', 'InstdAgt', self.validate_agent, 'Instructed Agent'),
        ]
        
        for camel, pascal, validator, label in entities:
            key = self._find_key(payment, [camel, pascal])
            if key:
                validator(payment[key], key, label)
        
        # Remittance Info
        rmt_key = self._find_key(payment, ['rmtInf', 'RmtInf'])
        if rmt_key:
            self.validate_remittance_info(payment[rmt_key], rmt_key)
        
        return self.get_summary()
    
    def _find_key(self, obj: Dict, candidates: List[str]) -> Optional[str]:
        """Find first matching key from candidates"""
        for key in candidates:
            if key in obj:
                return key
        return None
    
    def get_summary(self) -> Dict:
        """Get validation summary"""
        errors = self.get_errors()
        warnings = self.get_warnings()
        
        return {
            'valid': len(errors) == 0,
            'fields_checked': self.field_count,
            'fields_valid': self.valid_count,
            'error_count': len(errors),
            'warning_count': len(warnings),
            'errors': [str(e) for e in errors],
            'warnings': [str(w) for w in warnings],
            'all_results': [str(r) for r in self.results]
        }


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='ISO 20022 Deep Field Validator')
    parser.add_argument('input', help='Input JSON file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show all validation results')
    
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
        print("Error: Invalid JSON format")
        return
    
    # Validate
    validator = ISO20022Validator()
    summary = validator.validate_payment(payment)
    
    # Print results
    print("\n" + "="*80)
    print("ISO 20022 DEEP FIELD VALIDATION RESULTS")
    print("="*80)
    
    print(f"\nStatus: {'✓ VALID' if summary['valid'] else '✗ INVALID'}")
    print(f"Fields Checked: {summary['fields_checked']}")
    print(f"Fields Valid: {summary['fields_valid']}")
    print(f"Errors: {summary['error_count']}")
    print(f"Warnings: {summary['warning_count']}")
    
    if summary['errors']:
        print("\n" + "="*80)
        print("ERRORS")
        print("="*80)
        for error in summary['errors']:
            print(f"  {error}")
    
    if summary['warnings']:
        print("\n" + "="*80)
        print("WARNINGS")
        print("="*80)
        for warning in summary['warnings']:
            print(f"  {warning}")
    
    if args.verbose:
        print("\n" + "="*80)
        print("ALL VALIDATION RESULTS")
        print("="*80)
        for result in summary['all_results']:
            print(f"  {result}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
