#!/usr/bin/env python3
"""
ISO 20022 Payment Entity Validator
Validates payment entities against ISO 20022 pacs.008 specifications
"""

import json
import re
from typing import Dict, List, Optional, Any, Tuple
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


class ISO20022Validator:
    """Validates payment entities against ISO 20022 specifications"""
    
    def __init__(self):
        self.results: List[ValidationResult] = []
    
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
    
    # ========================================================================
    # HELPER VALIDATORS
    # ========================================================================
    
    def validate_string(self, value: Any, field: str, min_len: int = 1, max_len: int = 140) -> bool:
        """Validate string field"""
        if not isinstance(value, str):
            self.add_error(field, f"Must be a string, got {type(value).__name__}")
            return False
        
        if len(value) < min_len:
            self.add_error(field, f"Minimum length is {min_len}, got {len(value)}")
            return False
        
        if len(value) > max_len:
            self.add_error(field, f"Maximum length is {max_len}, got {len(value)}")
            return False
        
        return True
    
    def validate_currency(self, value: Any, field: str) -> bool:
        """Validate ISO 4217 currency code"""
        if not isinstance(value, str):
            self.add_error(field, f"Currency must be a string")
            return False
        
        if len(value) != 3:
            self.add_error(field, f"Currency must be 3 characters (ISO 4217)")
            return False
        
        if not value.isupper():
            self.add_warning(field, f"Currency should be uppercase")
        
        return True
    
    def validate_country_code(self, value: Any, field: str) -> bool:
        """Validate ISO 3166 country code"""
        if not isinstance(value, str):
            self.add_error(field, f"Country code must be a string")
            return False
        
        if len(value) != 2:
            self.add_error(field, f"Country code must be 2 characters (ISO 3166)")
            return False
        
        if not value.isupper():
            self.add_warning(field, f"Country code should be uppercase")
        
        return True
    
    def validate_bic(self, value: Any, field: str) -> bool:
        """Validate BIC/SWIFT code"""
        if not isinstance(value, str):
            self.add_error(field, f"BIC must be a string")
            return False
        
        # BIC: 8 or 11 characters
        if len(value) not in [8, 11]:
            self.add_error(field, f"BIC must be 8 or 11 characters, got {len(value)}")
            return False
        
        # Format: AAAABBCCXXX
        # AAAA: Bank code (4 letters)
        # BB: Country code (2 letters)
        # CC: Location code (2 alphanumeric)
        # XXX: Branch code (3 alphanumeric, optional)
        
        if not value[:4].isalpha():
            self.add_error(field, f"BIC bank code (first 4 chars) must be letters")
            return False
        
        if not value[4:6].isalpha():
            self.add_error(field, f"BIC country code (chars 5-6) must be letters")
            return False
        
        if not value[6:8].isalnum():
            self.add_error(field, f"BIC location code (chars 7-8) must be alphanumeric")
            return False
        
        return True
    
    def validate_iban(self, value: Any, field: str) -> bool:
        """Validate IBAN"""
        if not isinstance(value, str):
            self.add_error(field, f"IBAN must be a string")
            return False
        
        # Remove spaces
        iban = value.replace(' ', '')
        
        # Length: 15-34 characters
        if len(iban) < 15 or len(iban) > 34:
            self.add_error(field, f"IBAN must be 15-34 characters, got {len(iban)}")
            return False
        
        # First 2 chars: country code
        if not iban[:2].isalpha():
            self.add_error(field, f"IBAN must start with 2-letter country code")
            return False
        
        # Next 2 chars: check digits
        if not iban[2:4].isdigit():
            self.add_error(field, f"IBAN chars 3-4 must be check digits")
            return False
        
        return True
    
    def validate_routing_number(self, value: Any, field: str) -> bool:
        """Validate US routing number (ABA)"""
        if not isinstance(value, str):
            self.add_error(field, f"Routing number must be a string")
            return False
        
        # US routing number: 9 digits
        if len(value) != 9:
            self.add_error(field, f"US routing number must be 9 digits, got {len(value)}")
            return False
        
        if not value.isdigit():
            self.add_error(field, f"Routing number must be numeric")
            return False
        
        return True
    
    # ========================================================================
    # PARTY VALIDATORS
    # ========================================================================
    
    def validate_party(self, party: Dict, base_path: str, party_type: str) -> bool:
        """
        Validate Party (Creditor or Debtor)
        Structure:
        - Nm: Name [1..1] max 140
        - PstlAdr: Postal Address [0..1]
        - Id: Identification [0..1]
        - CtryOfRes: Country of Residence [0..1]
        """
        if not isinstance(party, dict):
            self.add_error(base_path, f"{party_type} must be a dictionary")
            return False
        
        valid = True
        
        # Name - REQUIRED
        if 'Nm' not in party and 'nm' not in party:
            self.add_error(f"{base_path}.Nm", "Name is required")
            valid = False
        else:
            name_key = 'Nm' if 'Nm' in party else 'nm'
            self.validate_string(party[name_key], f"{base_path}.{name_key}", max_len=140)
        
        # Postal Address - OPTIONAL
        if 'PstlAdr' in party:
            self.validate_postal_address(party['PstlAdr'], f"{base_path}.PstlAdr")
        elif 'pstlAdr' in party:
            self.validate_postal_address(party['pstlAdr'], f"{base_path}.pstlAdr")
        
        # Country of Residence - OPTIONAL
        if 'CtryOfRes' in party:
            self.validate_country_code(party['CtryOfRes'], f"{base_path}.CtryOfRes")
        elif 'ctryOfRes' in party:
            self.validate_country_code(party['ctryOfRes'], f"{base_path}.ctryOfRes")
        
        return valid
    
    def validate_postal_address(self, address: Dict, base_path: str) -> bool:
        """
        Validate Postal Address
        Structure:
        - Ctry: Country [0..1]
        - AdrLine: Address Line [0..7] max 70 each
        - StrtNm: Street Name [0..1]
        - BldgNb: Building Number [0..1]
        - PstCd: Post Code [0..1]
        - TwnNm: Town Name [0..1]
        """
        if not isinstance(address, dict):
            self.add_error(base_path, "Postal address must be a dictionary")
            return False
        
        # Country
        if 'Ctry' in address:
            self.validate_country_code(address['Ctry'], f"{base_path}.Ctry")
        elif 'ctry' in address:
            self.validate_country_code(address['ctry'], f"{base_path}.ctry")
        
        # Address Lines
        if 'AdrLine' in address:
            adr_lines = address['AdrLine']
            if isinstance(adr_lines, list):
                if len(adr_lines) > 7:
                    self.add_error(f"{base_path}.AdrLine", f"Maximum 7 address lines, got {len(adr_lines)}")
                for i, line in enumerate(adr_lines):
                    self.validate_string(line, f"{base_path}.AdrLine[{i}]", max_len=70)
            elif isinstance(adr_lines, str):
                self.validate_string(adr_lines, f"{base_path}.AdrLine", max_len=70)
        elif 'adrLine' in address:
            adr_lines = address['adrLine']
            if isinstance(adr_lines, list):
                if len(adr_lines) > 7:
                    self.add_error(f"{base_path}.adrLine", f"Maximum 7 address lines, got {len(adr_lines)}")
                for i, line in enumerate(adr_lines):
                    self.validate_string(line, f"{base_path}.adrLine[{i}]", max_len=70)
        
        # Post Code
        if 'PstCd' in address:
            self.validate_string(address['PstCd'], f"{base_path}.PstCd", max_len=16)
        elif 'pstCd' in address:
            self.validate_string(address['pstCd'], f"{base_path}.pstCd", max_len=16)
        
        # Town Name
        if 'TwnNm' in address:
            self.validate_string(address['TwnNm'], f"{base_path}.TwnNm", max_len=35)
        elif 'twnNm' in address:
            self.validate_string(address['twnNm'], f"{base_path}.twnNm", max_len=35)
        
        return True
    
    # ========================================================================
    # AGENT VALIDATORS
    # ========================================================================
    
    def validate_agent(self, agent: Dict, base_path: str, agent_type: str) -> bool:
        """
        Validate Financial Institution (Agent)
        Structure:
        - FinInstnId: Financial Institution Identification [1..1]
          - BICFI: BIC [0..1]
          - ClrSysMmbId: Clearing System Member ID [0..1]
          - Nm: Name [0..1]
          - PstlAdr: Postal Address [0..1]
          - Othr: Other [0..1]
        - BrnchId: Branch Identification [0..1]
        """
        if not isinstance(agent, dict):
            self.add_error(base_path, f"{agent_type} must be a dictionary")
            return False
        
        # FinInstnId - REQUIRED
        fin_instn_key = None
        if 'FinInstnId' in agent:
            fin_instn_key = 'FinInstnId'
        elif 'finInstnId' in agent:
            fin_instn_key = 'finInstnId'
        else:
            self.add_error(f"{base_path}.FinInstnId", "Financial Institution ID is required")
            return False
        
        fin_instn = agent[fin_instn_key]
        self.validate_financial_institution_id(fin_instn, f"{base_path}.{fin_instn_key}")
        
        # Country of Residence - OPTIONAL
        if 'CtryOfRes' in agent:
            self.validate_country_code(agent['CtryOfRes'], f"{base_path}.CtryOfRes")
        elif 'ctryOfRes' in agent:
            self.validate_country_code(agent['ctryOfRes'], f"{base_path}.ctryOfRes")
        
        return True
    
    def validate_financial_institution_id(self, fin_instn: Dict, base_path: str) -> bool:
        """Validate Financial Institution Identification"""
        if not isinstance(fin_instn, dict):
            self.add_error(base_path, "Financial Institution ID must be a dictionary")
            return False
        
        # At least one identification method required
        has_id = False
        
        # BIC/SWIFT
        if 'BICFI' in fin_instn:
            self.validate_bic(fin_instn['BICFI'], f"{base_path}.BICFI")
            has_id = True
        elif 'bicFi' in fin_instn:
            self.validate_bic(fin_instn['bicFi'], f"{base_path}.bicFi")
            has_id = True
        
        # Clearing System Member ID
        if 'ClrSysMmbId' in fin_instn:
            self.validate_clearing_system_member_id(fin_instn['ClrSysMmbId'], f"{base_path}.ClrSysMmbId")
            has_id = True
        elif 'clrSysMmbId' in fin_instn:
            self.validate_clearing_system_member_id(fin_instn['clrSysMmbId'], f"{base_path}.clrSysMmbId")
            has_id = True
        
        # Other ID
        if 'Othr' in fin_instn:
            self.validate_other_identification(fin_instn['Othr'], f"{base_path}.Othr")
            has_id = True
        elif 'othr' in fin_instn:
            self.validate_other_identification(fin_instn['othr'], f"{base_path}.othr")
            has_id = True
        
        if not has_id:
            self.add_warning(base_path, "No identification method provided (BIC, ClrSysMmbId, or Othr)")
        
        # Name - OPTIONAL but recommended
        if 'Nm' in fin_instn:
            self.validate_string(fin_instn['Nm'], f"{base_path}.Nm", max_len=140)
        elif 'nm' in fin_instn:
            self.validate_string(fin_instn['nm'], f"{base_path}.nm", max_len=140)
        
        # Postal Address - OPTIONAL
        if 'PstlAdr' in fin_instn:
            self.validate_postal_address(fin_instn['PstlAdr'], f"{base_path}.PstlAdr")
        elif 'pstlAdr' in fin_instn:
            self.validate_postal_address(fin_instn['pstlAdr'], f"{base_path}.pstlAdr")
        
        return True
    
    def validate_clearing_system_member_id(self, clr_sys: Dict, base_path: str) -> bool:
        """Validate Clearing System Member Identification"""
        if not isinstance(clr_sys, dict):
            self.add_error(base_path, "Clearing System Member ID must be a dictionary")
            return False
        
        # ClrSysId - OPTIONAL
        if 'ClrSysId' in clr_sys:
            clr_sys_id = clr_sys['ClrSysId']
            if isinstance(clr_sys_id, dict):
                # Can have Cd (Code) or Prtry (Proprietary)
                if 'Cd' in clr_sys_id:
                    self.validate_string(clr_sys_id['Cd'], f"{base_path}.ClrSysId.Cd", max_len=35)
                elif 'Prtry' in clr_sys_id:
                    self.validate_string(clr_sys_id['Prtry'], f"{base_path}.ClrSysId.Prtry", max_len=35)
            elif isinstance(clr_sys_id, str):
                # Direct code (common in camelCase format)
                self.validate_string(clr_sys_id, f"{base_path}.ClrSysId", max_len=35)
        elif 'clrSysId' in clr_sys:
            clr_sys_id = clr_sys['clrSysId']
            if isinstance(clr_sys_id, str):
                self.validate_string(clr_sys_id, f"{base_path}.clrSysId", max_len=35)
        
        # MmbId - REQUIRED
        if 'MmbId' not in clr_sys and 'mmbId' not in clr_sys:
            self.add_error(f"{base_path}.MmbId", "Member ID is required")
            return False
        
        mmb_key = 'MmbId' if 'MmbId' in clr_sys else 'mmbId'
        mmb_id = clr_sys[mmb_key]
        
        # Validate based on clearing system
        clr_sys_code = None
        if 'ClrSysId' in clr_sys:
            if isinstance(clr_sys['ClrSysId'], dict) and 'Cd' in clr_sys['ClrSysId']:
                clr_sys_code = clr_sys['ClrSysId']['Cd']
            elif isinstance(clr_sys['ClrSysId'], str):
                clr_sys_code = clr_sys['ClrSysId']
        elif 'clrSysId' in clr_sys:
            clr_sys_code = clr_sys['clrSysId']
        
        if clr_sys_code == 'USABA':
            # US ABA routing number
            self.validate_routing_number(mmb_id, f"{base_path}.{mmb_key}")
        else:
            self.validate_string(mmb_id, f"{base_path}.{mmb_key}", max_len=35)
        
        return True
    
    def validate_other_identification(self, other: Dict, base_path: str) -> bool:
        """Validate Other Identification"""
        if not isinstance(other, dict):
            self.add_error(base_path, "Other identification must be a dictionary")
            return False
        
        # Id - REQUIRED
        if 'Id' not in other and 'id' not in other:
            self.add_error(f"{base_path}.Id", "ID is required")
            return False
        
        id_key = 'Id' if 'Id' in other else 'id'
        self.validate_string(other[id_key], f"{base_path}.{id_key}", max_len=35)
        
        return True
    
    # ========================================================================
    # ACCOUNT VALIDATORS
    # ========================================================================
    
    def validate_account(self, account: Dict, base_path: str, account_type: str) -> bool:
        """
        Validate Account
        Structure:
        - Id: Identification [1..1]
          - IBAN: IBAN [0..1] OR
          - Othr: Other [0..1]
            - Id: ID [1..1]
            - SchmeNm: Scheme Name [0..1]
        - Tp: Type [0..1]
        - Ccy: Currency [0..1]
        - Nm: Name [0..1]
        """
        if not isinstance(account, dict):
            self.add_error(base_path, f"{account_type} must be a dictionary")
            return False
        
        # Id - REQUIRED
        if 'Id' not in account and 'id' not in account:
            self.add_error(f"{base_path}.Id", "Account ID is required")
            return False
        
        id_key = 'Id' if 'Id' in account else 'id'
        acct_id = account[id_key]
        
        if not isinstance(acct_id, dict):
            self.add_error(f"{base_path}.{id_key}", "Account ID must be a dictionary")
            return False
        
        # IBAN or Other - at least one required
        has_iban = 'IBAN' in acct_id or 'iban' in acct_id
        has_other = 'Othr' in acct_id or 'othr' in acct_id
        
        if not has_iban and not has_other:
            self.add_error(f"{base_path}.{id_key}", "Must have either IBAN or Othr identification")
            return False
        
        if has_iban:
            iban_key = 'IBAN' if 'IBAN' in acct_id else 'iban'
            self.validate_iban(acct_id[iban_key], f"{base_path}.{id_key}.{iban_key}")
        
        if has_other:
            other_key = 'Othr' if 'Othr' in acct_id else 'othr'
            other = acct_id[other_key]
            
            if not isinstance(other, dict):
                self.add_error(f"{base_path}.{id_key}.{other_key}", "Othr must be a dictionary")
            else:
                # Id - REQUIRED in Othr
                if 'Id' not in other and 'id' not in other:
                    self.add_error(f"{base_path}.{id_key}.{other_key}.Id", "ID is required in Othr")
                else:
                    other_id_key = 'Id' if 'Id' in other else 'id'
                    self.validate_string(other[other_id_key], f"{base_path}.{id_key}.{other_key}.{other_id_key}", max_len=34)
        
        # Type - OPTIONAL
        if 'Tp' in account:
            tp = account['Tp']
            if isinstance(tp, dict):
                if 'Prtry' in tp:
                    self.validate_string(tp['Prtry'], f"{base_path}.Tp.Prtry", max_len=35)
                elif 'Cd' in tp:
                    self.validate_string(tp['Cd'], f"{base_path}.Tp.Cd", max_len=4)
        elif 'tp' in account:
            tp = account['tp']
            if isinstance(tp, dict):
                if 'prtry' in tp:
                    self.validate_string(tp['prtry'], f"{base_path}.tp.prtry", max_len=35)
        
        # Currency - OPTIONAL
        if 'Ccy' in account:
            self.validate_currency(account['Ccy'], f"{base_path}.Ccy")
        elif 'ccy' in account:
            self.validate_currency(account['ccy'], f"{base_path}.ccy")
        
        # Name - OPTIONAL
        if 'Nm' in account:
            self.validate_string(account['Nm'], f"{base_path}.Nm", max_len=70)
        elif 'nm' in account:
            self.validate_string(account['nm'], f"{base_path}.nm", max_len=70)
        
        return True
    
    # ========================================================================
    # REMITTANCE INFO VALIDATOR
    # ========================================================================
    
    def validate_remittance_info(self, rmt_inf: Any, base_path: str) -> bool:
        """
        Validate Remittance Information
        Structure (can be array or single object):
        - Ustrd: Unstructured [0..n] max 140 each
        - Strd: Structured [0..n]
        """
        # Can be dict, list, or direct string
        if isinstance(rmt_inf, str):
            self.validate_string(rmt_inf, base_path, max_len=140)
            return True
        
        if isinstance(rmt_inf, dict):
            # Single remittance info object
            if 'Ustrd' in rmt_inf:
                ustrd = rmt_inf['Ustrd']
                if isinstance(ustrd, str):
                    self.validate_string(ustrd, f"{base_path}.Ustrd", max_len=140)
                elif isinstance(ustrd, list):
                    for i, line in enumerate(ustrd):
                        self.validate_string(line, f"{base_path}.Ustrd[{i}]", max_len=140)
            elif 'ustrd' in rmt_inf:
                ustrd = rmt_inf['ustrd']
                if isinstance(ustrd, str):
                    self.validate_string(ustrd, f"{base_path}.ustrd", max_len=140)
                elif isinstance(ustrd, list):
                    for i, line in enumerate(ustrd):
                        self.validate_string(line, f"{base_path}.ustrd[{i}]", max_len=140)
            
            return True
        
        if isinstance(rmt_inf, list):
            # Array of remittance info
            for i, item in enumerate(rmt_inf):
                if isinstance(item, dict):
                    if 'Ustrd' in item or 'ustrd' in item:
                        ustrd_key = 'Ustrd' if 'Ustrd' in item else 'ustrd'
                        ustrd = item[ustrd_key]
                        if isinstance(ustrd, str):
                            self.validate_string(ustrd, f"{base_path}[{i}].{ustrd_key}", max_len=140)
                        elif isinstance(ustrd, list):
                            for j, line in enumerate(ustrd):
                                self.validate_string(line, f"{base_path}[{i}].{ustrd_key}[{j}]", max_len=140)
                elif isinstance(item, str):
                    self.validate_string(item, f"{base_path}[{i}]", max_len=140)
            
            return True
        
        self.add_warning(base_path, f"Unexpected remittance info type: {type(rmt_inf).__name__}")
        return True
    
    # ========================================================================
    # MAIN VALIDATION METHOD
    # ========================================================================
    
    def validate_payment(self, payment: Dict) -> Dict:
        """
        Validate entire payment structure
        Returns validation summary
        """
        self.clear()
        
        if not isinstance(payment, dict):
            self.add_error("payment", "Payment must be a dictionary")
            return self.get_summary()
        
        # Validate Creditor (Beneficiary)
        if 'cdtr' in payment:
            self.validate_party(payment['cdtr'], 'cdtr', 'Creditor')
        elif 'Cdtr' in payment:
            self.validate_party(payment['Cdtr'], 'Cdtr', 'Creditor')
        else:
            self.add_info('cdtr', 'Creditor not found (may be optional depending on message type)')
        
        # Validate Debtor (Originator)
        if 'dbtr' in payment:
            self.validate_party(payment['dbtr'], 'dbtr', 'Debtor')
        elif 'Dbtr' in payment:
            self.validate_party(payment['Dbtr'], 'Dbtr', 'Debtor')
        else:
            self.add_info('dbtr', 'Debtor not found (may be optional depending on message type)')
        
        # Validate Creditor Agent (Beneficiary Bank)
        if 'cdtrAgt' in payment:
            self.validate_agent(payment['cdtrAgt'], 'cdtrAgt', 'Creditor Agent')
        elif 'CdtrAgt' in payment:
            self.validate_agent(payment['CdtrAgt'], 'CdtrAgt', 'Creditor Agent')
        
        # Validate Debtor Agent (Ordering Bank)
        if 'dbtrAgt' in payment:
            self.validate_agent(payment['dbtrAgt'], 'dbtrAgt', 'Debtor Agent')
        elif 'DbtrAgt' in payment:
            self.validate_agent(payment['DbtrAgt'], 'DbtrAgt', 'Debtor Agent')
        
        # Validate Creditor Account
        if 'cdtrAcct' in payment:
            self.validate_account(payment['cdtrAcct'], 'cdtrAcct', 'Creditor Account')
        elif 'CdtrAcct' in payment:
            self.validate_account(payment['CdtrAcct'], 'CdtrAcct', 'Creditor Account')
        
        # Validate Debtor Account
        if 'dbtrAcct' in payment:
            self.validate_account(payment['dbtrAcct'], 'dbtrAcct', 'Debtor Account')
        elif 'DbtrAcct' in payment:
            self.validate_account(payment['DbtrAcct'], 'DbtrAcct', 'Debtor Account')
        
        # Validate Instructing Agent
        if 'instgAgt' in payment:
            self.validate_agent(payment['instgAgt'], 'instgAgt', 'Instructing Agent')
        elif 'InstgAgt' in payment:
            self.validate_agent(payment['InstgAgt'], 'InstgAgt', 'Instructing Agent')
        
        # Validate Instructed Agent
        if 'instdAgt' in payment:
            self.validate_agent(payment['instdAgt'], 'instdAgt', 'Instructed Agent')
        elif 'InstdAgt' in payment:
            self.validate_agent(payment['InstdAgt'], 'InstdAgt', 'Instructed Agent')
        
        # Validate Intermediary Agent 1
        if 'intrmyAgt1' in payment:
            self.validate_agent(payment['intrmyAgt1'], 'intrmyAgt1', 'Intermediary Agent 1')
        elif 'IntrmyAgt1' in payment:
            self.validate_agent(payment['IntrmyAgt1'], 'IntrmyAgt1', 'Intermediary Agent 1')
        
        # Validate Intermediary Agent 2
        if 'intrmyAgt2' in payment:
            self.validate_agent(payment['intrmyAgt2'], 'intrmyAgt2', 'Intermediary Agent 2')
        elif 'IntrmyAgt2' in payment:
            self.validate_agent(payment['IntrmyAgt2'], 'IntrmyAgt2', 'Intermediary Agent 2')
        
        # Validate Remittance Information
        if 'rmtInf' in payment:
            self.validate_remittance_info(payment['rmtInf'], 'rmtInf')
        elif 'RmtInf' in payment:
            self.validate_remittance_info(payment['RmtInf'], 'RmtInf')
        
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
            'all_results': [str(r) for r in self.results]
        }


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='ISO 20022 Payment Validator')
    parser.add_argument('input', help='Input JSON file containing payment data')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show all validation results')
    
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
    print("\n" + "="*70)
    print("ISO 20022 VALIDATION RESULTS")
    print("="*70)
    
    print(f"\nStatus: {'✓ VALID' if summary['valid'] else '✗ INVALID'}")
    print(f"Errors: {summary['error_count']}")
    print(f"Warnings: {summary['warning_count']}")
    
    if summary['errors']:
        print("\n" + "="*70)
        print("ERRORS")
        print("="*70)
        for error in summary['errors']:
            print(f"  {error}")
    
    if summary['warnings']:
        print("\n" + "="*70)
        print("WARNINGS")
        print("="*70)
        for warning in summary['warnings']:
            print(f"  {warning}")
    
    if args.verbose:
        print("\n" + "="*70)
        print("ALL VALIDATION RESULTS")
        print("="*70)
        for result in summary['all_results']:
            print(f"  {result}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
