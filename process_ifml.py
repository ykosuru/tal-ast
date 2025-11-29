#!/usr/bin/env python3
"""
ACE Pelican IFML Processor and Rule Verification

This script processes your actual IFML payment data in the format:
{
  "payment_id": {
    "Request": { "IFML": { "File": { "Message": { ... } } } },
    "Response": { ... }  // Contains ACE codes
  }
}

Usage:
    # Process a single JSON file
    python process_ifml_payments.py --file payment.json --output report.json
    
    # Process entire directory of JSON files
    python process_ifml_payments.py --dir /path/to/payments/ --output report.json
    
    # Verbose mode
    python process_ifml_payments.py --dir /path/to/payments/ --verbose --limit 100
"""

import json
import argparse
import re
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict


# =============================================================================
# IFML PARSING - Extract features from your actual IFML structure
# =============================================================================

class IFMLFeatureExtractor:
    """
    Extract features from IFML payment messages in your actual format.
    Maps IFML party structures to normalized feature names.
    """
    
    # Map IFML party names to our standard prefixes
    PARTY_MAPPING = {
        'OriginatingPartyInf': 'orig',
        'OrderingBankInf': 'ordi',
        'SendingBankInf': 'send',
        'DebitPartyInf': 'dbt',
        'CreditPartyInf': 'cdt',
        'BeneficiaryBankInf': 'bnf_bank',
        'BeneficiaryPartyInf': 'bnf',
        'IntermediaryBankInf': 'intm',
        'AccountWithInf': 'acwi',
    }
    
    # IBAN country list (countries that require IBAN)
    IBAN_COUNTRIES = {
        'AD', 'AE', 'AL', 'AT', 'AZ', 'BA', 'BE', 'BG', 'BH', 'BR', 'BY', 
        'CH', 'CR', 'CY', 'CZ', 'DE', 'DK', 'DO', 'EE', 'EG', 'ES', 'FI', 
        'FO', 'FR', 'GB', 'GE', 'GI', 'GL', 'GR', 'GT', 'HR', 'HU', 'IE', 
        'IL', 'IQ', 'IS', 'IT', 'JO', 'KW', 'KZ', 'LB', 'LC', 'LI', 'LT', 
        'LU', 'LV', 'MC', 'MD', 'ME', 'MK', 'MR', 'MT', 'MU', 'NL', 'NO', 
        'PK', 'PL', 'PS', 'PT', 'QA', 'RO', 'RS', 'SA', 'SC', 'SE', 'SI', 
        'SK', 'SM', 'ST', 'SV', 'TL', 'TN', 'TR', 'UA', 'VA', 'VG', 'XK'
    }
    
    def __init__(self):
        self.features = {}
    
    def extract(self, ifml: Dict) -> Dict:
        """Extract all features from IFML message"""
        self.features = {}
        
        # Navigate to Message level
        message = self._get_message(ifml)
        if not message:
            return self.features
        
        # Extract basic payment info
        self._extract_basic_payment(message.get('BasicPayment', []))
        
        # Extract party info
        party_inf = message.get('PartyInf', {})
        for ifml_party_name, prefix in self.PARTY_MAPPING.items():
            party_data = party_inf.get(ifml_party_name, {})
            if party_data:
                self._extract_party_features(party_data, prefix)
            else:
                self.features[f'{prefix}_present'] = False
        
        return self.features
    
    def _get_message(self, ifml: Dict) -> Optional[Dict]:
        """Navigate to Message level in IFML structure"""
        try:
            # Handle nested structure: IFML.File.Message
            if 'IFML' in ifml:
                ifml = ifml['IFML']
            if 'File' in ifml:
                ifml = ifml['File']
            if 'Message' in ifml:
                return ifml['Message']
            return ifml
        except (KeyError, TypeError):
            return None
    
    def _extract_basic_payment(self, basic_payment: Any):
        """Extract basic payment info"""
        if isinstance(basic_payment, list) and basic_payment:
            basic_payment = basic_payment[0]
        if not isinstance(basic_payment, dict):
            return
        
        # Source code, message type, etc.
        self.features['source_code'] = basic_payment.get('SourceCode', '')
        self.features['incoming_msg_type'] = basic_payment.get('IncomingMsgType', '')
        self.features['incoming_format'] = basic_payment.get('IncomingFormat', '')
        
        # Currency
        amounts = basic_payment.get('MonetaryAmount', [])
        if amounts:
            for amt in amounts:
                if amt.get('Type') == 'Amount':
                    self.features['currency'] = amt.get('Currency', '')
                    break
    
    def _extract_party_features(self, party_data: Dict, prefix: str):
        """Extract features for a single party"""
        self.features[f'{prefix}_present'] = True
        
        # Get BasicPartyInf or AccountPartyInf
        basic_party = party_data.get('BasicPartyInf', {})
        account_party = party_data.get('AccountPartyInf', {})
        
        # Merge - some parties have one or the other
        if not basic_party and account_party:
            basic_party = account_party.get('BasicPartyInf', account_party)
        
        # Extract ID info
        self._extract_id_info(basic_party, account_party, prefix)
        
        # Extract address info
        self._extract_address_info(basic_party, prefix)
        self._extract_address_info(account_party, prefix)
        
        # Extract country
        self._extract_country(party_data, basic_party, prefix)
        
        # Determine if party needs IBAN based on country
        country = self.features.get(f'{prefix}_country', '')
        self.features[f'{prefix}_needs_iban'] = country in self.IBAN_COUNTRIES
        
        # Check for name
        address_inf = basic_party.get('AddressInf', [])
        if address_inf:
            self.features[f'{prefix}_has_name'] = True
            self.features[f'{prefix}_address_lines'] = len(address_inf)
        else:
            self.features[f'{prefix}_has_name'] = False
            self.features[f'{prefix}_address_lines'] = 0
    
    def _extract_id_info(self, basic_party: Dict, account_party: Dict, prefix: str):
        """Extract ID information (BIC, IBAN, NCH, Account)"""
        
        # Check multiple locations for ID
        id_info = basic_party.get('ID', {})
        if isinstance(id_info, str):
            # Sometimes ID is just a string
            id_info = {'text': id_info}
        
        # Also check AcctIDInf
        acct_id_inf = account_party.get('AcctIDInf', {}) if account_party else {}
        acct_id = acct_id_inf.get('ID', {})
        if isinstance(acct_id, str):
            acct_id = {'text': acct_id}
        
        # ID Type: S=SWIFT/BIC, P=Proprietary, D=Domestic, etc.
        id_type = id_info.get('Type', '')
        id_text = id_info.get('text', '')
        
        acct_type = acct_id.get('Type', '')
        acct_text = acct_id.get('text', '')
        
        # AdrBankID (often contains BIC)
        adr_bank_id = basic_party.get('AdrBankID', '')
        
        # Analyze what we have
        self._classify_identifiers(prefix, id_type, id_text, acct_type, acct_text, adr_bank_id)
    
    def _classify_identifiers(self, prefix: str, id_type: str, id_text: str, 
                              acct_type: str, acct_text: str, adr_bank_id: str):
        """Classify identifiers into BIC, IBAN, NCH, Account"""
        
        # Initialize defaults
        self.features[f'{prefix}_has_bic'] = False
        self.features[f'{prefix}_has_iban'] = False
        self.features[f'{prefix}_has_nch'] = False
        self.features[f'{prefix}_has_account'] = False
        
        bic = None
        iban = None
        nch = None
        account = None
        
        # Check ID based on type
        if id_type == 'S':  # SWIFT BIC
            bic = id_text
        elif id_type == 'P':  # Proprietary - could be account
            account = id_text
        elif id_type == 'D':  # Domestic account
            account = id_text
        
        # Check account ID
        if acct_type == 'S':
            bic = acct_text
        elif acct_type in ('P', 'D', ''):
            if acct_text:
                account = acct_text
        
        # Parse composite IDs (e.g., "BNPATHBK/AC1016850017)")
        for text in [id_text, acct_text]:
            if text:
                parsed = self._parse_composite_id(text)
                if parsed.get('bic'):
                    bic = parsed['bic']
                if parsed.get('account'):
                    account = parsed['account']
        
        # AdrBankID is often a BIC
        if adr_bank_id and self._looks_like_bic(adr_bank_id):
            bic = adr_bank_id
        
        # Detect IBAN
        if account and self._looks_like_iban(account):
            iban = account
            account = None
        
        # Detect NCH (routing numbers)
        if account and self._looks_like_nch(account):
            nch = account
            account = None
        
        # Store results
        if bic:
            self.features[f'{prefix}_has_bic'] = True
            self.features[f'{prefix}_bic'] = bic
            self._validate_bic(prefix, bic)
        
        if iban:
            self.features[f'{prefix}_has_iban'] = True
            self.features[f'{prefix}_iban'] = iban
            self._validate_iban(prefix, iban)
        
        if nch:
            self.features[f'{prefix}_has_nch'] = True
            self.features[f'{prefix}_nch'] = nch
            self._validate_nch(prefix, nch)
        
        if account:
            self.features[f'{prefix}_has_account'] = True
            self.features[f'{prefix}_account'] = account
            self._validate_account(prefix, account)
    
    def _parse_composite_id(self, text: str) -> Dict:
        """Parse composite ID like 'BNPATHBK/(AC1016850017)'"""
        result = {}
        
        # Pattern: BIC/(account) or BIC/account
        match = re.match(r'^([A-Z]{4}[A-Z]{2}[A-Z0-9]{2,5})[/\(](.+?)[\)]?$', text)
        if match:
            result['bic'] = match.group(1)
            result['account'] = match.group(2).lstrip('AC')
            return result
        
        # Just BIC
        if self._looks_like_bic(text):
            result['bic'] = text
        
        return result
    
    def _looks_like_bic(self, text: str) -> bool:
        """Check if text looks like a BIC"""
        if not text:
            return False
        # BIC: 8 or 11 chars, format: AAAABBCC or AAAABBCCDDD
        text = text.strip()
        if len(text) not in (8, 11):
            return False
        return bool(re.match(r'^[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}([A-Z0-9]{3})?$', text))
    
    def _looks_like_iban(self, text: str) -> bool:
        """Check if text looks like an IBAN"""
        if not text:
            return False
        text = text.replace(' ', '').replace('-', '').upper()
        # IBAN: 2 letter country + 2 check digits + up to 30 alphanumeric
        if len(text) < 15 or len(text) > 34:
            return False
        return bool(re.match(r'^[A-Z]{2}[0-9]{2}[A-Z0-9]+$', text))
    
    def _looks_like_nch(self, text: str) -> bool:
        """Check if text looks like an NCH/routing number"""
        if not text:
            return False
        text = text.strip()
        # FEDABA: 9 digits, CHIPS: 6 digits, SORTCODE: 6 digits
        if re.match(r'^\d{9}$', text):  # FEDABA
            return True
        if re.match(r'^\d{6}$', text):  # CHIPS/SORTCODE
            return True
        return False
    
    def _validate_bic(self, prefix: str, bic: str):
        """Validate BIC and set features"""
        bic = bic.strip().upper()
        self.features[f'{prefix}_bic_length'] = len(bic)
        
        # Format validation
        valid_format = bool(re.match(r'^[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}([A-Z0-9]{3})?$', bic))
        self.features[f'{prefix}_bic_valid_format'] = valid_format
        
        # Country validation (chars 5-6)
        if len(bic) >= 6:
            country = bic[4:6]
            self.features[f'{prefix}_bic_country'] = country
            # Simple ISO check (not exhaustive)
            valid_country = bool(re.match(r'^[A-Z]{2}$', country))
            self.features[f'{prefix}_bic_valid_country'] = valid_country
        else:
            self.features[f'{prefix}_bic_valid_country'] = False
    
    def _validate_iban(self, prefix: str, iban: str):
        """Validate IBAN and set features"""
        iban = iban.replace(' ', '').replace('-', '').upper()
        
        # Format validation
        valid_format = bool(re.match(r'^[A-Z]{2}[0-9]{2}[A-Z0-9]+$', iban))
        self.features[f'{prefix}_iban_valid_format'] = valid_format
        
        # Extract country
        if len(iban) >= 2:
            self.features[f'{prefix}_iban_country'] = iban[:2]
        
        # Checksum validation (mod 97)
        checksum_valid = self._validate_iban_checksum(iban)
        self.features[f'{prefix}_iban_checksum_valid'] = checksum_valid
        
        # Check if needs formatting
        original_iban = self.features.get(f'{prefix}_iban', '')
        self.features[f'{prefix}_iban_needs_formatting'] = (
            ' ' in original_iban or 
            '-' in original_iban or 
            original_iban != original_iban.upper()
        )
        
        # BIC/IBAN country match
        bic_country = self.features.get(f'{prefix}_bic_country', '')
        iban_country = self.features.get(f'{prefix}_iban_country', '')
        if bic_country and iban_country:
            self.features[f'{prefix}_bic_iban_match'] = (bic_country == iban_country)
        else:
            self.features[f'{prefix}_bic_iban_match'] = True  # Can't mismatch if one missing
    
    def _validate_iban_checksum(self, iban: str) -> bool:
        """Validate IBAN mod-97 checksum"""
        try:
            if len(iban) < 5:
                return False
            # Move first 4 chars to end
            rearranged = iban[4:] + iban[:4]
            # Convert letters to numbers (A=10, B=11, etc.)
            numeric = ''
            for char in rearranged:
                if char.isalpha():
                    numeric += str(ord(char) - ord('A') + 10)
                else:
                    numeric += char
            # Check mod 97
            return int(numeric) % 97 == 1
        except (ValueError, TypeError):
            return False
    
    def _validate_nch(self, prefix: str, nch: str):
        """Validate NCH/routing number"""
        nch = re.sub(r'[^0-9]', '', nch)  # Keep only digits
        
        self.features[f'{prefix}_nch_has_dirty_chars'] = (
            self.features.get(f'{prefix}_nch', '') != nch
        )
        
        # Determine NCH type
        if len(nch) == 9:
            self.features[f'{prefix}_nch_type'] = 'FEDABA'
            self.features[f'{prefix}_nch_valid'] = True
            # ABA checksum
            checksum_valid = self._validate_aba_checksum(nch)
            self.features[f'{prefix}_fedaba_checksum_valid'] = checksum_valid
        elif len(nch) == 6:
            self.features[f'{prefix}_nch_type'] = 'CHIPS_OR_SORT'
            self.features[f'{prefix}_nch_valid'] = True
            self.features[f'{prefix}_fedaba_checksum_valid'] = True  # N/A
        else:
            self.features[f'{prefix}_nch_type'] = 'OTHER'
            self.features[f'{prefix}_nch_valid'] = False
            self.features[f'{prefix}_fedaba_checksum_valid'] = True
        
        # Default other NCH features
        self.features[f'{prefix}_nch_sources'] = 1
        self.features[f'{prefix}_nch_consistent'] = True
        self.features[f'{prefix}_nch_needs_formatting'] = False
        self.features[f'{prefix}_nch_validation_applicable'] = (
            self.features.get(f'{prefix}_country', '') == 'US' or
            self.features.get('is_domestic', False)
        )
    
    def _validate_aba_checksum(self, aba: str) -> bool:
        """Validate ABA routing number checksum"""
        try:
            if len(aba) != 9 or not aba.isdigit():
                return False
            # ABA checksum: 3*d1 + 7*d2 + d3 + 3*d4 + 7*d5 + d6 + 3*d7 + 7*d8 + d9 ≡ 0 (mod 10)
            weights = [3, 7, 1, 3, 7, 1, 3, 7, 1]
            total = sum(int(d) * w for d, w in zip(aba, weights))
            return total % 10 == 0
        except (ValueError, TypeError):
            return False
    
    def _validate_account(self, prefix: str, account: str):
        """Validate generic account number"""
        # Check for dirty characters
        clean = re.sub(r'[^A-Za-z0-9]', '', account)
        self.features[f'{prefix}_account_has_dirty_chars'] = (account != clean)
        self.features[f'{prefix}_account_has_spaces'] = ' ' in account
        self.features[f'{prefix}_account_has_special_chars'] = bool(re.search(r'[^A-Za-z0-9\s]', account))
        
        # Account type guess
        self.features[f'{prefix}_account_type'] = 'OTHER'
        self.features[f'{prefix}_account_valid_format'] = len(clean) > 0
        self.features[f'{prefix}_account_needs_length_fix'] = False
        
        # Check for CLABE (Mexico - 18 digits)
        if len(clean) == 18 and clean.isdigit():
            self.features[f'{prefix}_is_clabe'] = True
        else:
            self.features[f'{prefix}_is_clabe'] = False
    
    def _extract_address_info(self, party_data: Dict, prefix: str):
        """Extract address information"""
        address_inf = party_data.get('AddressInf', [])
        if address_inf and isinstance(address_inf, list):
            # Concatenate address lines
            address_text = ' '.join(
                line.get('text', '') for line in address_inf if isinstance(line, dict)
            )
            self.features[f'{prefix}_address_text'] = address_text
    
    def _extract_country(self, party_data: Dict, basic_party: Dict, prefix: str):
        """Extract country from various possible fields"""
        country = (
            party_data.get('Country') or
            party_data.get('ResidenceCountry') or
            party_data.get('MailingCountry') or
            basic_party.get('Country') or
            ''
        )
        self.features[f'{prefix}_country'] = country
        self.features[f'{prefix}_is_domestic'] = (country == 'US')


# =============================================================================
# ACE CODE EXTRACTION
# =============================================================================

def extract_ace_codes(response: Dict) -> Set[str]:
    """Extract ACE codes from response"""
    codes = set()
    
    # Try various possible locations for ACE codes in response
    # Adjust these based on your actual response structure
    
    def find_codes(obj, path=''):
        """Recursively find codes in response"""
        if isinstance(obj, dict):
            # Check for code-related keys
            for key in ['ACECodes', 'ErrorCodes', 'Codes', 'errors', 'codes', 
                       'PelicanCodes', 'ValidationErrors', 'RepairCodes']:
                if key in obj:
                    val = obj[key]
                    if isinstance(val, list):
                        for item in val:
                            if isinstance(item, str):
                                codes.add(extract_base_code(item))
                            elif isinstance(item, dict):
                                # Code might be in a nested structure
                                code = item.get('Code') or item.get('code') or item.get('ErrorCode')
                                if code:
                                    codes.add(extract_base_code(str(code)))
                    elif isinstance(val, str):
                        for c in val.split(','):
                            codes.add(extract_base_code(c.strip()))
            
            # Recurse
            for k, v in obj.items():
                find_codes(v, f'{path}.{k}')
        
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                find_codes(item, f'{path}[{i}]')
    
    find_codes(response)
    return codes


def extract_base_code(code: str) -> str:
    """Extract 4-digit base code"""
    match = re.match(r'^(\d{4})', str(code).strip())
    return match.group(1) if match else str(code).strip()


# =============================================================================
# DETERMINISTIC RULES
# =============================================================================

PARTY_PREFIXES = ['orig', 'ordi', 'send', 'dbt', 'cdt', 'bnf', 'bnf_bank', 'intm', 'acwi']


class DeterministicRule:
    """Base class for rules"""
    def __init__(self, code: str, description: str):
        self.code = code
        self.description = description
    
    def check(self, features: Dict, party: str) -> Tuple[bool, bool, Dict]:
        """Returns (should_emit, precondition_met, relevant_features)"""
        raise NotImplementedError
    
    def _get(self, features: Dict, party: str, key: str, default=None):
        return features.get(f'{party}_{key}', default)


# 8XXX Validation Rules
class Rule8001(DeterministicRule):
    def __init__(self):
        super().__init__('8001', 'Invalid BIC')
    
    def check(self, features: Dict, party: str) -> Tuple[bool, bool, Dict]:
        has_bic = self._get(features, party, 'has_bic', False)
        if not has_bic:
            return False, False, {}
        
        valid_format = self._get(features, party, 'bic_valid_format', True)
        valid_country = self._get(features, party, 'bic_valid_country', True)
        should_emit = not valid_format or not valid_country
        
        return should_emit, True, {
            f'{party}_has_bic': has_bic,
            f'{party}_bic_valid_format': valid_format,
            f'{party}_bic_valid_country': valid_country,
            f'{party}_bic': self._get(features, party, 'bic', ''),
        }


class Rule8894(DeterministicRule):
    def __init__(self):
        super().__init__('8894', 'Invalid IBAN')
    
    def check(self, features: Dict, party: str) -> Tuple[bool, bool, Dict]:
        has_iban = self._get(features, party, 'has_iban', False)
        if not has_iban:
            return False, False, {}
        
        valid_format = self._get(features, party, 'iban_valid_format', True)
        valid_checksum = self._get(features, party, 'iban_checksum_valid', True)
        should_emit = not valid_format or not valid_checksum
        
        return should_emit, True, {
            f'{party}_has_iban': has_iban,
            f'{party}_iban_valid_format': valid_format,
            f'{party}_iban_checksum_valid': valid_checksum,
            f'{party}_iban': self._get(features, party, 'iban', ''),
        }


class Rule8895(DeterministicRule):
    def __init__(self):
        super().__init__('8895', 'Invalid NCH')
    
    def check(self, features: Dict, party: str) -> Tuple[bool, bool, Dict]:
        has_nch = self._get(features, party, 'has_nch', False)
        applicable = self._get(features, party, 'nch_validation_applicable', False)
        if not has_nch or not applicable:
            return False, False, {}
        
        valid = self._get(features, party, 'nch_valid', True)
        checksum_valid = self._get(features, party, 'fedaba_checksum_valid', True)
        should_emit = not valid or not checksum_valid
        
        return should_emit, True, {
            f'{party}_has_nch': has_nch,
            f'{party}_nch_validation_applicable': applicable,
            f'{party}_nch_valid': valid,
            f'{party}_fedaba_checksum_valid': checksum_valid,
        }


class Rule8022(DeterministicRule):
    def __init__(self):
        super().__init__('8022', 'BIC/IBAN country mismatch')
    
    def check(self, features: Dict, party: str) -> Tuple[bool, bool, Dict]:
        has_iban = self._get(features, party, 'has_iban', False)
        has_bic = self._get(features, party, 'has_bic', False)
        if not has_iban or not has_bic:
            return False, False, {}
        
        match = self._get(features, party, 'bic_iban_match', True)
        should_emit = not match
        
        return should_emit, True, {
            f'{party}_has_iban': has_iban,
            f'{party}_has_bic': has_bic,
            f'{party}_bic_iban_match': match,
            f'{party}_bic_country': self._get(features, party, 'bic_country', ''),
            f'{party}_iban_country': self._get(features, party, 'iban_country', ''),
        }


# 9XXX No-Directory Rules
class Rule9002(DeterministicRule):
    def __init__(self):
        super().__init__('9002', 'Account cleaned')
    
    def check(self, features: Dict, party: str) -> Tuple[bool, bool, Dict]:
        present = self._get(features, party, 'present', False)
        has_account = self._get(features, party, 'has_account', False)
        dirty = self._get(features, party, 'account_has_dirty_chars', False)
        
        precondition = present and has_account and dirty
        return precondition, precondition, {
            f'{party}_present': present,
            f'{party}_has_account': has_account,
            f'{party}_account_has_dirty_chars': dirty,
        }


class Rule9006(DeterministicRule):
    def __init__(self):
        super().__init__('9006', 'IBAN formatted')
    
    def check(self, features: Dict, party: str) -> Tuple[bool, bool, Dict]:
        present = self._get(features, party, 'present', False)
        has_iban = self._get(features, party, 'has_iban', False)
        needs_format = self._get(features, party, 'iban_needs_formatting', False)
        
        precondition = present and has_iban and needs_format
        return precondition, precondition, {
            f'{party}_present': present,
            f'{party}_has_iban': has_iban,
            f'{party}_iban_needs_formatting': needs_format,
        }


class Rule9000(DeterministicRule):
    def __init__(self):
        super().__init__('9000', 'NCH cleaned')
    
    def check(self, features: Dict, party: str) -> Tuple[bool, bool, Dict]:
        has_nch = self._get(features, party, 'has_nch', False)
        dirty = self._get(features, party, 'nch_has_dirty_chars', False)
        
        precondition = has_nch and dirty
        return precondition, precondition, {
            f'{party}_has_nch': has_nch,
            f'{party}_nch_has_dirty_chars': dirty,
        }


# All rules to check
RULES = [
    Rule8001(),
    Rule8894(),
    Rule8895(),
    Rule8022(),
    Rule9002(),
    Rule9006(),
    Rule9000(),
]


# =============================================================================
# VERIFICATION
# =============================================================================

@dataclass
class PaymentVerification:
    payment_id: str
    features: Dict
    actual_codes: Set[str]
    checks: int = 0
    matches: int = 0
    false_positives: List[Dict] = field(default_factory=list)
    false_negatives: List[Dict] = field(default_factory=list)
    true_positives: List[Dict] = field(default_factory=list)


def verify_payment(payment_id: str, features: Dict, actual_codes: Set[str], 
                   rules: List[DeterministicRule], verbose: bool = False) -> PaymentVerification:
    """Verify rules for a payment"""
    
    result = PaymentVerification(
        payment_id=payment_id,
        features=features,
        actual_codes=actual_codes
    )
    
    for rule in rules:
        for party in PARTY_PREFIXES:
            should_emit, precondition, relevant = rule.check(features, party)
            actual = rule.code in actual_codes
            
            result.checks += 1
            
            if should_emit == actual:
                result.matches += 1
                if should_emit:
                    result.true_positives.append({
                        'code': rule.code,
                        'party': party,
                        'description': rule.description,
                        'features': relevant,
                    })
            else:
                item = {
                    'code': rule.code,
                    'party': party,
                    'description': rule.description,
                    'precondition_met': precondition,
                    'features': relevant,
                }
                if should_emit and not actual:
                    result.false_positives.append(item)
                else:
                    result.false_negatives.append(item)
    
    if verbose and (result.false_positives or result.false_negatives):
        print(f"\n{'='*60}")
        print(f"Payment: {payment_id}")
        print(f"Actual codes: {actual_codes}")
        if result.false_positives:
            print(f"\nFalse Positives:")
            for fp in result.false_positives[:3]:
                print(f"  {fp['code']} ({fp['party']}): {fp['description']}")
                print(f"    {fp['features']}")
        if result.false_negatives:
            print(f"\nFalse Negatives:")
            for fn in result.false_negatives[:3]:
                print(f"  {fn['code']} ({fn['party']}): {fn['description']}")
                print(f"    {fn['features']}")
    
    return result


# =============================================================================
# FILE PROCESSING
# =============================================================================

def process_payment_file(filepath: Path, verbose: bool = False) -> List[PaymentVerification]:
    """Process a single JSON file containing payment(s)"""
    results = []
    
    with open(filepath) as f:
        data = json.load(f)
    
    # Handle different file structures
    if isinstance(data, dict):
        # Could be {payment_id: {Request:..., Response:...}}
        for payment_id, payment_data in data.items():
            if isinstance(payment_data, dict) and 'Request' in payment_data:
                result = process_single_payment(payment_id, payment_data, verbose)
                if result:
                    results.append(result)
    
    elif isinstance(data, list):
        # List of payments
        for i, payment_data in enumerate(data):
            payment_id = payment_data.get('payment_id', payment_data.get('id', f'payment_{i}'))
            result = process_single_payment(payment_id, payment_data, verbose)
            if result:
                results.append(result)
    
    return results


def process_single_payment(payment_id: str, payment_data: Dict, 
                           verbose: bool = False) -> Optional[PaymentVerification]:
    """Process a single payment request/response pair"""
    
    # Extract IFML from request
    request = payment_data.get('Request', payment_data)
    ifml = request.get('IFML', request)
    
    # Extract features
    extractor = IFMLFeatureExtractor()
    features = extractor.extract(ifml)
    
    if not features:
        return None
    
    # Extract ACE codes from response
    response = payment_data.get('Response', {})
    actual_codes = extract_ace_codes(response)
    
    # Verify
    result = verify_payment(payment_id, features, actual_codes, RULES, verbose)
    
    return result


def process_directory(dirpath: Path, verbose: bool = False, 
                      limit: Optional[int] = None) -> List[PaymentVerification]:
    """Process all JSON files in a directory"""
    results = []
    json_files = list(dirpath.glob('*.json'))
    
    print(f"Found {len(json_files)} JSON files")
    
    for i, filepath in enumerate(json_files):
        if limit and len(results) >= limit:
            break
        
        if (i + 1) % 10 == 0:
            print(f"  Processing file {i+1}/{len(json_files)}...", end='\r')
        
        try:
            file_results = process_payment_file(filepath, verbose)
            results.extend(file_results)
        except Exception as e:
            print(f"\nError processing {filepath}: {e}")
    
    print(f"\nProcessed {len(results)} payments from {len(json_files)} files")
    return results


# =============================================================================
# REPORTING
# =============================================================================

def generate_report(results: List[PaymentVerification]) -> Dict:
    """Generate summary report"""
    
    total_checks = sum(r.checks for r in results)
    total_matches = sum(r.matches for r in results)
    
    code_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    
    for r in results:
        for tp in r.true_positives:
            code_stats[tp['code']]['tp'] += 1
        for fp in r.false_positives:
            code_stats[fp['code']]['fp'] += 1
        for fn in r.false_negatives:
            code_stats[fn['code']]['fn'] += 1
    
    # Calculate metrics
    for code, stats in code_stats.items():
        tp, fp, fn = stats['tp'], stats['fp'], stats['fn']
        stats['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        stats['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    
    return {
        'summary': {
            'total_payments': len(results),
            'total_checks': total_checks,
            'total_matches': total_matches,
            'accuracy': total_matches / total_checks if total_checks > 0 else 0,
            'payments_with_discrepancies': sum(1 for r in results if r.false_positives or r.false_negatives),
        },
        'code_stats': dict(code_stats),
        'discrepancies': [
            {
                'payment_id': r.payment_id,
                'actual_codes': list(r.actual_codes),
                'false_positives': r.false_positives,
                'false_negatives': r.false_negatives,
            }
            for r in results if r.false_positives or r.false_negatives
        ][:100],  # Limit to 100 for report size
    }


def print_report(report: Dict):
    """Print report to console"""
    s = report['summary']
    
    print("\n" + "=" * 70)
    print("ACE PELICAN RULE VERIFICATION REPORT")
    print("=" * 70)
    
    print(f"\n📊 SUMMARY")
    print(f"   Payments: {s['total_payments']}")
    print(f"   Checks: {s['total_checks']}")
    print(f"   Matches: {s['total_matches']}")
    print(f"   Accuracy: {s['accuracy']:.2%}")
    print(f"   With discrepancies: {s['payments_with_discrepancies']}")
    
    print(f"\n📋 CODE METRICS")
    print(f"   {'Code':<8} {'TP':<6} {'FP':<6} {'FN':<6} {'Prec':<10} {'Recall':<10}")
    print(f"   {'-'*50}")
    
    for code in sorted(report['code_stats'].keys()):
        st = report['code_stats'][code]
        if st['tp'] + st['fp'] + st['fn'] > 0:
            print(f"   {code:<8} {st['tp']:<6} {st['fp']:<6} {st['fn']:<6} "
                  f"{st['precision']:<10.2%} {st['recall']:<10.2%}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Process IFML payments and verify ACE rules')
    parser.add_argument('--file', help='Single JSON file to process')
    parser.add_argument('--dir', help='Directory of JSON files to process')
    parser.add_argument('--output', '-o', help='Output JSON report path')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--limit', type=int, help='Limit number of payments')
    
    args = parser.parse_args()
    
    if not args.file and not args.dir:
        parser.print_help()
        return
    
    # Process
    if args.file:
        results = process_payment_file(Path(args.file), args.verbose)
    else:
        results = process_directory(Path(args.dir), args.verbose, args.limit)
    
    # Report
    report = generate_report(results)
    print_report(report)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n📄 Saved to: {args.output}")


if __name__ == '__main__':
    main()
