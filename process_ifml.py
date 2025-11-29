#!/usr/bin/env python3
"""
ACE Pelican IFML Verification Tool

Parses IFML JSON files, applies deterministic 8XXX/9XXX rules,
compares predictions against actual Response codes.

Usage:
    python verify_ifml.py /path/to/ifml/directory
    python verify_ifml.py /path/to/ifml/directory --verbose
    python verify_ifml.py /path/to/single_file.json
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict


# =============================================================================
# CONSTANTS
# =============================================================================

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

PARTY_PREFIXES = list(PARTY_MAPPING.values())

# Countries using IBAN
IBAN_COUNTRIES = {
    'AD', 'AE', 'AL', 'AT', 'AZ', 'BA', 'BE', 'BG', 'BH', 'BR', 'BY',
    'CH', 'CR', 'CY', 'CZ', 'DE', 'DK', 'DO', 'EE', 'EG', 'ES', 'FI',
    'FO', 'FR', 'GB', 'GE', 'GI', 'GL', 'GR', 'GT', 'HR', 'HU', 'IE',
    'IL', 'IQ', 'IS', 'IT', 'JO', 'KW', 'KZ', 'LB', 'LC', 'LI', 'LT',
    'LU', 'LV', 'MC', 'MD', 'ME', 'MK', 'MR', 'MT', 'MU', 'NL', 'NO',
    'PK', 'PL', 'PS', 'PT', 'QA', 'RO', 'RS', 'SA', 'SC', 'SE', 'SI',
    'SK', 'SM', 'ST', 'SV', 'TL', 'TN', 'TR', 'UA', 'VA', 'VG', 'XK'
}

# Countries where IBAN derivation is supported
IBAN_DERIVATION_COUNTRIES = {
    'DE', 'FR', 'GB', 'ES', 'IT', 'NL', 'BE', 'AT', 'CH', 'PT', 'IE',
    'FI', 'NO', 'SE', 'DK', 'PL', 'CZ', 'HU', 'SK', 'SI', 'HR', 'RO',
    'BG', 'GR', 'CY', 'MT', 'LU', 'EE', 'LV', 'LT'
}

# Valid ISO country codes
VALID_COUNTRY_CODES = {
    'US', 'GB', 'DE', 'FR', 'IT', 'ES', 'NL', 'BE', 'AT', 'CH', 'SE',
    'NO', 'DK', 'FI', 'IE', 'PT', 'GR', 'PL', 'CZ', 'HU', 'RO', 'BG',
    'HR', 'SK', 'SI', 'EE', 'LV', 'LT', 'CY', 'MT', 'LU', 'JP', 'CN',
    'IN', 'AU', 'NZ', 'CA', 'MX', 'BR', 'AR', 'CL', 'CO', 'PE', 'VE',
    'SG', 'HK', 'TW', 'KR', 'TH', 'MY', 'ID', 'PH', 'VN', 'AE', 'SA',
    'IL', 'TR', 'ZA', 'EG', 'NG', 'KE', 'RU', 'UA', 'BY', 'KZ'
}


# =============================================================================
# FEATURE EXTRACTOR
# =============================================================================

class FeatureExtractor:
    """Extract features from IFML for rule evaluation."""
    
    def __init__(self):
        self.features = {}
        self.all_ibans = []
        self.all_bbans = []
        self.all_nchs = []
        self.all_accounts = []
        self.all_bic4s = []
    
    def extract(self, ifml: Dict) -> Dict:
        """Extract all features from IFML message."""
        self.features = {}
        self.all_ibans = []
        self.all_bbans = []
        self.all_nchs = []
        self.all_accounts = []
        self.all_bic4s = []
        
        message = self._navigate_to_message(ifml)
        if not message:
            return self.features
        
        # Extract party information
        party_inf = message.get('PartyInf', {})
        for ifml_name, prefix in PARTY_MAPPING.items():
            party_data = party_inf.get(ifml_name, {})
            if party_data:
                # CRITICAL: Check if party is a list (multiple entries = 9017/9018)
                if isinstance(party_data, list):
                    if len(party_data) > 1:
                        # Multiple party entries of same type - triggers 9017/9018
                        self.features[f'{prefix}_has_multiple_ids'] = True
                        self.features[f'{prefix}_has_duplicate_info'] = True
                    # Process first entry for features, but mark as present
                    self.features[f'{prefix}_present'] = True
                    party_data = party_data[0] if party_data else {}
                    self._extract_party(party_data, prefix)
                else:
                    self._extract_party(party_data, prefix)
            else:
                self.features[f'{prefix}_present'] = False
        
        # Track intermediary presence for 9024
        self.features['has_intermediary'] = self.features.get('intm_present', False)
        
        # Cross-party consistency
        self._check_consistency()
        
        return self.features
    
    def _navigate_to_message(self, data: Dict) -> Optional[Dict]:
        """Navigate to Message node in IFML structure."""
        try:
            if 'IFML' in data:
                data = data['IFML']
            if 'File' in data:
                data = data['File']
            if 'Message' in data:
                return data['Message']
            return data
        except (KeyError, TypeError):
            return None
    
    def _extract_party(self, party_data: Dict, prefix: str):
        """Extract features for a single party."""
        self.features[f'{prefix}_present'] = True
        
        # Check if party_data itself is a list (multiple party entries - 9017/9018)
        if isinstance(party_data, list):
            if len(party_data) > 1:
                self.features[f'{prefix}_has_multiple_ids'] = True
                self.features[f'{prefix}_has_duplicate_info'] = True
            party_data = party_data[0] if party_data else {}
        
        basic = party_data.get('BasicPartyInf', {})
        account = party_data.get('AccountPartyInf', {})
        if not basic and account:
            basic = account.get('BasicPartyInf', account)
        
        # Check if basic is a list (multiple BasicPartyInf entries - 9017/9018)
        if isinstance(basic, list):
            if len(basic) > 1:
                self.features[f'{prefix}_has_multiple_ids'] = True
                self.features[f'{prefix}_has_duplicate_info'] = True
            basic = basic[0] if basic else {}
        
        # Check if account is a list (multiple AccountPartyInf entries - 9017/9018)
        if isinstance(account, list):
            if len(account) > 1:
                self.features[f'{prefix}_has_multiple_ids'] = True
                self.features[f'{prefix}_has_duplicate_info'] = True
            account = account[0] if account else {}
        
        # Extract identifiers
        self._extract_identifiers(basic, account, prefix)
        
        # Extract country
        country = (
            party_data.get('Country') or
            party_data.get('ResidenceCountry') or
            basic.get('Country') or ''
        )
        self.features[f'{prefix}_country'] = country
        self.features[f'{prefix}_country_valid'] = country in VALID_COUNTRY_CODES if country else True
        self.features[f'{prefix}_needs_iban'] = country in IBAN_COUNTRIES
        self.features[f'{prefix}_iban_derivation_supported'] = country in IBAN_DERIVATION_COUNTRIES
        
        # Address info
        address_inf = basic.get('AddressInf', [])
        self.features[f'{prefix}_has_name'] = bool(address_inf)
        self.features[f'{prefix}_address_lines'] = len(address_inf) if isinstance(address_inf, list) else 0
        
        # Check for embedded identifiers in address
        if address_inf:
            addr_text = ' '.join(
                line.get('text', '') for line in address_inf if isinstance(line, dict)
            )
            self.features[f'{prefix}_has_fedaba_in_address'] = bool(re.search(r'\b\d{9}\b', addr_text))
            self.features[f'{prefix}_has_chips_in_address'] = bool(re.search(r'\b\d{6}\b', addr_text))
        
        # Check for duplicate values (same BIC in ID and AdrBankID)
        self._check_duplicate_values(basic, account, prefix)
    
    def _check_duplicate_values(self, basic: Dict, account: Dict, prefix: str):
        """Check for duplicate values across different fields (triggers 9018)."""
        values = []
        
        # Collect ID value
        id_info = basic.get('ID', {})
        if isinstance(id_info, dict):
            id_text = id_info.get('text') or id_info.get('#text', '')
            if id_text:
                values.append(id_text)
        elif isinstance(id_info, str):
            values.append(id_info)
        
        # Collect AdrBankID value
        adr_bank_id = basic.get('AdrBankID', {})
        if isinstance(adr_bank_id, dict):
            adr_text = adr_bank_id.get('text') or adr_bank_id.get('#text', '')
            if adr_text:
                values.append(adr_text)
        elif isinstance(adr_bank_id, str):
            values.append(adr_bank_id)
        
        # Collect account ID value
        if account:
            acct_id = account.get('AcctIDInf', {}).get('ID', {})
            if isinstance(acct_id, dict):
                acct_text = acct_id.get('text') or acct_id.get('#text', '')
                if acct_text:
                    values.append(acct_text)
            elif isinstance(acct_id, str):
                values.append(acct_id)
        
        # Check for duplicates
        if len(values) != len(set(values)):
            self.features[f'{prefix}_has_duplicate_info'] = True
    
    def _extract_identifiers(self, basic: Dict, account: Dict, prefix: str):
        """Extract and classify identifiers (BIC, IBAN, NCH, Account)."""
        # Initialize
        for key in ['has_bic', 'has_iban', 'has_nch', 'has_account', 'has_bban', 'has_domestic_account']:
            self.features[f'{prefix}_{key}'] = False
        
        # Initialize multiple/duplicate detection
        self.features[f'{prefix}_has_multiple_ids'] = False
        self.features[f'{prefix}_has_duplicate_info'] = False
        self.features[f'{prefix}_has_id'] = False
        self.features[f'{prefix}_id_has_dirty_chars'] = False
        
        # Get ID fields - check if it's a list (multiple IDs)
        id_info = basic.get('ID', {})
        
        # CRITICAL: Check for multiple IDs (9017, 9018 detection)
        if isinstance(id_info, list):
            if len(id_info) > 1:
                self.features[f'{prefix}_has_multiple_ids'] = True
                self.features[f'{prefix}_has_duplicate_info'] = True
            id_info = id_info[0] if id_info else {}
        
        if isinstance(id_info, str):
            id_info = {'text': id_info}
        
        acct_id = {}
        if account:
            acct_id_inf = account.get('AcctIDInf', {}) or account.get('AcctIDInfo', {})
            acct_id = acct_id_inf.get('ID', {})
            
            # Check for multiple account IDs too
            if isinstance(acct_id, list):
                if len(acct_id) > 1:
                    self.features[f'{prefix}_has_multiple_ids'] = True
                    self.features[f'{prefix}_has_duplicate_info'] = True
                acct_id = acct_id[0] if acct_id else {}
            
            if isinstance(acct_id, str):
                acct_id = {'text': acct_id}
        
        id_type = id_info.get('Type') or id_info.get('@Type', '')
        id_text = id_info.get('text') or id_info.get('#text', '')
        acct_type = acct_id.get('Type') or acct_id.get('@Type', '')
        acct_text = acct_id.get('text') or acct_id.get('#text', '')
        
        # AdrBankID can be in BasicPartyInf OR AccountPartyInf
        adr_bank_id = basic.get('AdrBankID', '') or (account.get('AdrBankID', '') if account else '')
        
        # Check for has_id and dirty chars (for 9019)
        if id_text:
            self.features[f'{prefix}_has_id'] = True
            # Dirty chars in ID: unusual characters that need cleaning
            if re.search(r'[^\w\s/\-\(\)]', id_text):
                self.features[f'{prefix}_id_has_dirty_chars'] = True
        
        # Classify each identifier
        bic, iban, nch, account_num = None, None, None, None
        
        # Type S = SWIFT/BIC
        if id_type == 'S' and id_text:
            bic = id_text
        elif id_type == 'BIC' and id_text:
            bic = id_text
        elif id_type in ('P', 'D') and id_text:
            account_num = id_text
            if id_type == 'D':
                self.features[f'{prefix}_has_domestic_account'] = True
        elif id_type == 'IBAN' and id_text:
            iban = id_text
        
        if acct_type == 'S' and acct_text:
            bic = acct_text
        elif acct_type == 'BIC' and acct_text:
            bic = acct_text
        elif acct_type == 'IBAN' and acct_text:
            iban = acct_text
        elif acct_text:
            account_num = acct_text
            if acct_type == 'D':
                self.features[f'{prefix}_has_domestic_account'] = True
        
        # Parse composite IDs like "BNPATHBK/(AC1016850017)"
        for text in [id_text, acct_text]:
            if text:
                match = re.match(r'^([A-Z]{4}[A-Z]{2}[A-Z0-9]{2,5})[/\(](.+?)[\)]?$', text)
                if match:
                    bic = match.group(1)
                    account_num = match.group(2).lstrip('AC')
        
        # AdrBankID often contains BIC or NCH
        if adr_bank_id:
            if isinstance(adr_bank_id, dict):
                adr_type = adr_bank_id.get('Type') or adr_bank_id.get('@Type', '')
                adr_text = adr_bank_id.get('text') or adr_bank_id.get('#text', '')
                
                if adr_type in ['FEDABA', 'FW', 'FEDWIRE']:
                    nch = adr_text
                    self.features[f'{prefix}_nch_type'] = 'FEDABA'
                elif adr_type in ['CHIPS', 'CH', 'CHIPSABA']:
                    nch = adr_text
                    self.features[f'{prefix}_nch_type'] = 'CHIPS'
                elif adr_type in ['SORTCODE', 'SC', 'UKSC']:
                    nch = adr_text
                    self.features[f'{prefix}_nch_type'] = 'SORTCODE'
                elif self._is_bic(adr_text):
                    bic = adr_text
                elif adr_text and adr_text.isdigit():
                    # Numeric AdrBankID is likely an NCH
                    nch = adr_text
                    self.features[f'{prefix}_nch_type'] = 'NCH'
            elif isinstance(adr_bank_id, str):
                # Plain string AdrBankID
                if self._is_bic(adr_bank_id):
                    bic = adr_bank_id
                elif adr_bank_id.isdigit() and len(adr_bank_id) in (3, 6, 9):
                    # Numeric - likely NCH (BSB=3, CHIPS=6, FEDABA=9)
                    nch = adr_bank_id
                    if len(adr_bank_id) == 9:
                        self.features[f'{prefix}_nch_type'] = 'FEDABA'
                    elif len(adr_bank_id) == 6:
                        self.features[f'{prefix}_nch_type'] = 'CHIPS_OR_SORT'
                    else:
                        self.features[f'{prefix}_nch_type'] = 'BSB'  # Australian 3-digit
        
        # Classify account type if not already classified
        if account_num and not iban and not nch:
            if self._is_iban(account_num):
                iban = account_num
                account_num = None
            elif self._is_nch(account_num):
                nch = account_num
                account_num = None
        
        # Store and validate
        if bic:
            self._validate_bic(prefix, bic)
        if iban:
            self._validate_iban(prefix, iban)
        if nch:
            self._validate_nch(prefix, nch)
        if account_num:
            self._validate_account(prefix, account_num)
    
    def _is_bic(self, text: str) -> bool:
        text = text.strip()
        return len(text) in (8, 11) and bool(re.match(r'^[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}([A-Z0-9]{3})?$', text))
    
    def _is_iban(self, text: str) -> bool:
        text = text.replace(' ', '').replace('-', '').upper()
        return 15 <= len(text) <= 34 and bool(re.match(r'^[A-Z]{2}[0-9]{2}[A-Z0-9]+$', text))
    
    def _is_nch(self, text: str) -> bool:
        clean = re.sub(r'[^0-9]', '', text)
        # NCH can be: 3 (AU BSB short), 6 (UK sort/CHIPS), 9 (FEDABA)
        return len(clean) in (3, 6, 9)
    
    def _validate_bic(self, prefix: str, bic: str):
        bic = bic.strip().upper()
        self.features[f'{prefix}_has_bic'] = True
        self.features[f'{prefix}_bic'] = bic
        self.features[f'{prefix}_bic_length'] = len(bic)
        
        # BIC4 (first 4 chars)
        bic4 = bic[:4] if len(bic) >= 4 else ''
        self.features[f'{prefix}_bic4'] = bic4
        self.features[f'{prefix}_bic4_valid'] = bool(re.match(r'^[A-Z]{4}$', bic4))
        if bic4:
            self.all_bic4s.append(bic4)
        
        # Full format
        self.features[f'{prefix}_bic_valid_format'] = bool(
            re.match(r'^[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}([A-Z0-9]{3})?$', bic)
        )
        
        # Country (chars 5-6)
        if len(bic) >= 6:
            country = bic[4:6]
            self.features[f'{prefix}_bic_country'] = country
            self.features[f'{prefix}_bic_valid_country'] = country in VALID_COUNTRY_CODES
        else:
            self.features[f'{prefix}_bic_country'] = ''
            self.features[f'{prefix}_bic_valid_country'] = False
    
    def _validate_iban(self, prefix: str, iban: str):
        original = iban
        iban = iban.replace(' ', '').replace('-', '').upper()
        
        self.features[f'{prefix}_has_iban'] = True
        self.features[f'{prefix}_iban'] = iban
        self.all_ibans.append(iban)
        
        # Format validation
        self.features[f'{prefix}_iban_valid_format'] = bool(
            re.match(r'^[A-Z]{2}[0-9]{2}[A-Z0-9]+$', iban)
        )
        
        # Country
        if len(iban) >= 2:
            self.features[f'{prefix}_iban_country'] = iban[:2]
        
        # Length check for specific countries
        iban_lengths = {
            'DE': 22, 'FR': 27, 'GB': 22, 'ES': 24, 'IT': 27, 'NL': 18, 'BE': 16,
            'AT': 20, 'CH': 21, 'SE': 24, 'NO': 15, 'DK': 18, 'FI': 18, 'PL': 28,
            'PT': 25, 'IE': 22, 'GR': 27, 'CZ': 24, 'HU': 28, 'RO': 24, 'SK': 24,
            'LU': 20, 'HR': 21, 'SI': 19, 'BG': 22, 'LT': 20, 'LV': 21, 'EE': 20,
        }
        country = iban[:2] if len(iban) >= 2 else ''
        expected_len = iban_lengths.get(country)
        if expected_len and len(iban) != expected_len:
            self.features[f'{prefix}_iban_wrong_length'] = True
        else:
            self.features[f'{prefix}_iban_wrong_length'] = False
        
        # Checksum (mod-97)
        try:
            rearranged = iban[4:] + iban[:4]
            numeric = ''.join(str(ord(c) - ord('A') + 10) if c.isalpha() else c for c in rearranged)
            self.features[f'{prefix}_iban_checksum_valid'] = int(numeric) % 97 == 1
        except:
            self.features[f'{prefix}_iban_checksum_valid'] = False
        
        # Needs formatting
        self.features[f'{prefix}_iban_needs_formatting'] = (
            ' ' in original or '-' in original or original != original.upper()
        )
        
        # BIC/IBAN country match
        bic_country = self.features.get(f'{prefix}_bic_country', '')
        iban_country = self.features.get(f'{prefix}_iban_country', '')
        self.features[f'{prefix}_bic_iban_match'] = (
            not bic_country or not iban_country or bic_country == iban_country
        )
        
        # BBAN (from IBAN)
        if len(iban) > 4:
            self.features[f'{prefix}_has_bban'] = True
            self.features[f'{prefix}_bban_from_iban'] = iban[4:]
            self.all_bbans.append(iban[4:])
    
    def _validate_nch(self, prefix: str, nch: str):
        original = nch
        clean = re.sub(r'[^0-9]', '', nch)
        
        self.features[f'{prefix}_has_nch'] = True
        self.features[f'{prefix}_nch'] = nch
        self.all_nchs.append(clean)
        
        self.features[f'{prefix}_nch_has_dirty_chars'] = (original != clean)
        
        if len(clean) == 9:
            self.features[f'{prefix}_nch_type'] = 'FEDABA'
            self.features[f'{prefix}_is_fedaba'] = True
            self.features[f'{prefix}_nch_valid'] = True
            # ABA checksum
            try:
                weights = [3, 7, 1, 3, 7, 1, 3, 7, 1]
                total = sum(int(d) * w for d, w in zip(clean, weights))
                self.features[f'{prefix}_fedaba_checksum_valid'] = (total % 10 == 0)
            except:
                self.features[f'{prefix}_fedaba_checksum_valid'] = False
            self.features[f'{prefix}_nch_needs_formatting'] = (original != clean)
        elif len(clean) == 6:
            self.features[f'{prefix}_nch_type'] = 'CHIPS_OR_SORT'
            self.features[f'{prefix}_is_fedaba'] = False
            self.features[f'{prefix}_nch_valid'] = True
            self.features[f'{prefix}_fedaba_checksum_valid'] = True
            self.features[f'{prefix}_nch_needs_formatting'] = False
        elif len(clean) == 3:
            # Australian BSB (short form) or other 3-digit codes
            self.features[f'{prefix}_nch_type'] = 'BSB'
            self.features[f'{prefix}_is_fedaba'] = False
            self.features[f'{prefix}_nch_valid'] = True
            self.features[f'{prefix}_fedaba_checksum_valid'] = True
            self.features[f'{prefix}_nch_needs_formatting'] = False
        else:
            self.features[f'{prefix}_nch_type'] = 'OTHER'
            self.features[f'{prefix}_is_fedaba'] = False
            self.features[f'{prefix}_nch_valid'] = False
            self.features[f'{prefix}_fedaba_checksum_valid'] = True
            self.features[f'{prefix}_nch_needs_formatting'] = False
        
        # NCH validation applicable for US or AU
        country = self.features.get(f'{prefix}_country', '')
        self.features[f'{prefix}_nch_validation_applicable'] = (country in ['US', 'AU'])
        self.features[f'{prefix}_nch_in_wrong_field'] = False
    
    def _validate_account(self, prefix: str, account: str):
        clean = re.sub(r'[^A-Za-z0-9]', '', account)
        
        self.features[f'{prefix}_has_account'] = True
        self.features[f'{prefix}_account'] = account
        self.all_accounts.append(clean)
        
        self.features[f'{prefix}_account_has_dirty_chars'] = (account != clean)
        self.features[f'{prefix}_account_valid_format'] = len(clean) > 0
        self.features[f'{prefix}_account_needs_length_fix'] = False
        self.features[f'{prefix}_account_wrong_length'] = False
        self.features[f'{prefix}_is_clabe'] = (len(clean) == 18 and clean.isdigit())
        
        # Check for wrong length based on account type
        acct_type = self.features.get(f'{prefix}_account_type', '')
        if acct_type == 'CLABE' and len(clean) != 18:
            self.features[f'{prefix}_account_wrong_length'] = True
        elif acct_type == 'FEDABA' and len(clean) != 9:
            self.features[f'{prefix}_account_wrong_length'] = True
        
        # Domestic account validation (US)
        country = self.features.get(f'{prefix}_country', '')
        if country == 'US':
            self.features[f'{prefix}_domestic_account_valid'] = (
                clean.isdigit() and 9 <= len(clean) <= 17
            )
        else:
            self.features[f'{prefix}_domestic_account_valid'] = True
    
    def _check_consistency(self):
        """Check cross-party consistency for 8023-8029 codes."""
        # IBAN consistency - only inconsistent if >1 AND different values
        unique_ibans = set(self.all_ibans)
        self.features['iban_count'] = len(self.all_ibans)
        self.features['ibans_consistent'] = len(unique_ibans) <= 1
        
        # BBAN consistency
        unique_bbans = set(self.all_bbans)
        self.features['bban_count'] = len(self.all_bbans)
        self.features['bbans_consistent'] = len(unique_bbans) <= 1
        
        # NCH consistency
        unique_nchs = set(self.all_nchs)
        self.features['nch_count'] = len(self.all_nchs)
        self.features['nchs_consistent'] = len(unique_nchs) <= 1
        
        # Account consistency
        unique_accounts = set(self.all_accounts)
        self.features['account_count'] = len(self.all_accounts)
        self.features['accounts_consistent'] = len(unique_accounts) <= 1
        
        # BIC4 consistency - need >1 DIFFERENT values to be inconsistent
        unique_bic4s = set(self.all_bic4s)
        self.features['bic4_count'] = len(self.all_bic4s)
        self.features['bic4s_consistent'] = len(unique_bic4s) <= 1


# =============================================================================
# RULE ENGINE - Deterministic 8XXX and 9XXX Rules
# =============================================================================

class RuleEngine:
    """Apply deterministic rules to predict 8XXX and 9XXX codes."""
    
    def __init__(self, features: Dict):
        self.f = features
        self.codes = set()
        self.directory_eligible = []  # List of (code, party, condition) for directory lookups
    
    def predict(self) -> Set[str]:
        """Run all deterministic rules and return predicted codes."""
        self.codes = set()
        self.directory_eligible = []
        
        # Message-level consistency checks
        self._check_message_level()
        
        # Party-level checks
        for party in PARTY_PREFIXES:
            if not self.f.get(f'{party}_present', False):
                continue
            self._check_8xxx(party)
            self._check_9xxx(party)
            self._check_directory_eligible(party)
        
        return self.codes
    
    def get_directory_eligible(self) -> List[Tuple[str, str, str]]:
        """Return list of directory-dependent codes that are eligible."""
        return self.directory_eligible
    
    def _get(self, party: str, key: str, default=None):
        return self.f.get(f'{party}_{key}', default)
    
    def _emit(self, code: str):
        self.codes.add(code)
    
    def _emit_eligible(self, code: str, party: str, condition: str):
        """Record a directory-dependent code eligibility."""
        self.directory_eligible.append((code, party, condition))
    
    # =========================================================================
    # MESSAGE-LEVEL CHECKS (8023-8029)
    # =========================================================================
    
    def _check_message_level(self):
        # 8023: IBAN inconsistency
        if self.f.get('iban_count', 0) > 1 and not self.f.get('ibans_consistent', True):
            self._emit('8023')
        
        # 8024: BBAN inconsistency
        if self.f.get('bban_count', 0) > 1 and not self.f.get('bbans_consistent', True):
            self._emit('8024')
        
        # 8026: NCH inconsistency
        if self.f.get('nch_count', 0) > 1 and not self.f.get('nchs_consistent', True):
            self._emit('8026')
        
        # 8028: BIC4 inconsistency - within same party, not cross-party
        # Different parties having different BIC4s is expected
        # This check would need party-level multiple BIC4 detection
        pass
        
        # 8029: Account inconsistency
        if self.f.get('account_count', 0) > 1 and not self.f.get('accounts_consistent', True):
            self._emit('8029')
    
    # =========================================================================
    # 8XXX VALIDATION ERRORS
    # =========================================================================
    
    def _check_8xxx(self, party: str):
        # 8001: Invalid BIC
        if self._get(party, 'has_bic', False):
            if not self._get(party, 'bic_valid_format', True):
                self._emit('8001')
            elif not self._get(party, 'bic_valid_country', True):
                self._emit('8001')
        
        # 8005: Invalid BIC4
        if self._get(party, 'has_bic', False):
            if not self._get(party, 'bic4_valid', True):
                self._emit('8005')
        
        # 8006: Invalid country code
        if self._get(party, 'country') and not self._get(party, 'country_valid', True):
            self._emit('8006')
        
        # 8022: IBAN/BIC country mismatch OR BIC/NCH inconsistency
        # Can fire when:
        # 1. IBAN country != BIC country
        # 2. BIC country != Party country
        # 3. BIC present with NCH from different country
        # Note: Only fire if BIC is valid format (otherwise 8001 fires)
        if self._get(party, 'has_bic', False) and self._get(party, 'bic_valid_format', False):
            bic_country = self._get(party, 'bic_country', '')
            
            # Check IBAN vs BIC
            if self._get(party, 'has_iban', False):
                if not self._get(party, 'bic_iban_match', True):
                    self._emit('8022')
            
            # Check BIC vs party country
            party_country = self._get(party, 'country', '')
            if bic_country and party_country and bic_country != party_country:
                self._emit('8022')
            
            # Check BIC vs NCH country (US NCH with non-US BIC, etc.)
            if self._get(party, 'has_nch', False):
                nch_type = self._get(party, 'nch_type', '')
                # FEDABA implies US
                if nch_type == 'FEDABA' and bic_country and bic_country != 'US':
                    self._emit('8022')
                # BSB implies AU
                if nch_type == 'BSB' and bic_country and bic_country != 'AU':
                    self._emit('8022')
        
        # 8030: IBAN derivation not supported
        if (self._get(party, 'needs_iban', False) and 
            not self._get(party, 'has_iban', False) and
            self._get(party, 'has_account', False) and
            not self._get(party, 'iban_derivation_supported', True)):
            self._emit('8030')
        
        # 8892: Invalid account number
        if self._get(party, 'has_account', False):
            if not self._get(party, 'account_valid_format', True):
                self._emit('8892')
        
        # 8894: Invalid IBAN
        if self._get(party, 'has_iban', False):
            if not self._get(party, 'iban_valid_format', True):
                self._emit('8894')
            elif not self._get(party, 'iban_checksum_valid', True):
                self._emit('8894')
        
        # 8895: Invalid NCH (US only)
        if self._get(party, 'has_nch', False) and self._get(party, 'nch_validation_applicable', False):
            if not self._get(party, 'nch_valid', True):
                self._emit('8895')
            elif not self._get(party, 'fedaba_checksum_valid', True):
                self._emit('8895')
        
        # 8896: Invalid domestic account
        if self._get(party, 'has_domestic_account', False):
            if not self._get(party, 'domestic_account_valid', True):
                self._emit('8896')
        
        # 8898: IBAN checksum failed (format OK but checksum bad)
        if self._get(party, 'has_iban', False):
            if self._get(party, 'iban_valid_format', True) and not self._get(party, 'iban_checksum_valid', True):
                self._emit('8898')
        
        # 8852: Incorrect length of attribute
        # Check for account/IBAN/BIC with wrong length
        # Note: Don't fire 8852 if 8001 already fires for invalid format
        if self._get(party, 'has_account', False):
            if self._get(party, 'account_wrong_length', False):
                self._emit('8852')
        if self._get(party, 'has_iban', False):
            if self._get(party, 'iban_wrong_length', False):
                self._emit('8852')
        # BIC length check - only fire if format is valid but length is unusual
        if self._get(party, 'has_bic', False):
            bic_len = self._get(party, 'bic_length', 0)
            bic_valid_format = self._get(party, 'bic_valid_format', True)
            # 8852 for BIC only when format looks like BIC but length is off
            # (valid format catches the length check, so we don't double-fire)
            if bic_len > 0 and bic_len not in (8, 11) and not bic_valid_format:
                # If bic_valid_format is False, 8001 will fire
                # Only fire 8852 if it's a length-specific issue with otherwise valid chars
                bic_val = self._get(party, 'bic', '')
                if bic_val and re.match(r'^[A-Z0-9]+$', bic_val):
                    pass  # Let 8001 handle it
    
    # =========================================================================
    # 9XXX REPAIRS (Deterministic - No Directory)
    # =========================================================================
    
    def _check_9xxx(self, party: str):
        # 9000: NCH code cleaned
        if self._get(party, 'has_nch', False) and self._get(party, 'nch_has_dirty_chars', False):
            self._emit('9000')
        
        # 9002: Account number cleaned (non-domestic)
        if (self._get(party, 'has_account', False) and 
            self._get(party, 'account_has_dirty_chars', False) and
            not self._get(party, 'has_domestic_account', False)):
            self._emit('9002')
        
        # 9006: IBAN cleaned
        if self._get(party, 'has_iban', False) and self._get(party, 'iban_needs_formatting', False):
            self._emit('9006')
        
        # 9014: BBAN cleaned
        if self._get(party, 'has_bban', False):
            bban = self._get(party, 'bban_from_iban', '')
            if bban and re.search(r'[^A-Za-z0-9]', bban):
                self._emit('9014')
        
        # 9015: Domestic account cleaned
        if self._get(party, 'has_domestic_account', False) and self._get(party, 'account_has_dirty_chars', False):
            self._emit('9015')
        
        # 9017: Multiple party info
        if self._get(party, 'has_multiple_ids', False):
            self._emit('9017')
        # Also check if multiple address entries (some systems store multiple party info in addresses)
        if self._get(party, 'address_lines', 0) > 4:
            # More than 4 address lines might indicate multiple party info merged
            pass  # Too aggressive - don't emit
        
        # 9018: Duplicate removed
        if self._get(party, 'has_duplicate_info', False):
            self._emit('9018')
        
        # 9019: Party identifier cleaned (dirty chars in party ID)
        if self._get(party, 'has_id', False):
            if self._get(party, 'id_has_dirty_chars', False):
                self._emit('9019')
        
        # 9021: FEDABA formatted
        if (self._get(party, 'has_nch', False) and 
            self._get(party, 'is_fedaba', False) and 
            self._get(party, 'nch_needs_formatting', False)):
            self._emit('9021')
        
        # 9022: Account length fixed
        if self._get(party, 'has_account', False) and self._get(party, 'account_needs_length_fix', False):
            self._emit('9022')
        
        # 9024: Intermediary added/pushed up
        # This fires when intermediary routing is performed
        # From preconditions: party present + has_bic + [directory lookup for intermediary]
        # Since this is directory-dependent, we should emit as eligible rather than deterministic
        # But the precondition says require_false: has_intermediary, which is confusing
        # For now, emit if beneficiary bank has BIC (implies routing might need intermediary)
        # This is a soft rule - the actual determination is directory-dependent
        if party == 'bnf':
            if self._get(party, 'has_bic', False):
                # This is directory-dependent - we can't know deterministically
                # Move to directory_eligible instead
                pass
        
        # 9028: NCH code strip off
        if self._get(party, 'has_nch', False) and self._get(party, 'nch_in_wrong_field', False):
            self._emit('9028')
    
    # =========================================================================
    # DIRECTORY-DEPENDENT ELIGIBILITY (for --include-directory mode)
    # =========================================================================
    
    def _check_directory_eligible(self, party: str):
        """Check eligibility for directory-dependent codes."""
        
        # 8004: IBAN cannot be derived (emitted when derivation FAILS)
        if (self._get(party, 'needs_iban', False) and 
            not self._get(party, 'has_iban', False) and
            self._get(party, 'has_account', False)):
            if self._get(party, 'iban_derivation_supported', True):
                self._emit_eligible('9004', party, 
                    f"IBAN derivation eligible (has account, needs IBAN) → IF FOUND: 9004, ELSE: 8004+9999")
            else:
                # Country doesn't support derivation - 8030 already emitted deterministically
                pass
        
        # 9005: BIC from NCH
        if self._get(party, 'has_nch', False) and not self._get(party, 'has_bic', False):
            self._emit_eligible('9005', party,
                f"BIC from NCH eligible (has NCH, no BIC) → IF FOUND: 9005, ELSE: 9999")
        
        # 9008: BIC from IBAN
        if self._get(party, 'has_iban', False) and not self._get(party, 'has_bic', False):
            self._emit_eligible('9008', party,
                f"BIC from IBAN eligible (has IBAN, no BIC) → IF FOUND: 9008, ELSE: 9999")
        
        # 9013: Name/Address repair
        if (self._get(party, 'has_name', False) and 
            not self._get(party, 'has_bic', False) and
            self._get(party, 'address_lines', 0) >= 1):
            self._emit_eligible('9013', party,
                f"Name/Address repair eligible → IF FOUND: 9013, ELSE: 9999")
        
        # 9477: FED ABA from Name/Address (NOT BIC expansion!)
        if self._get(party, 'has_fedaba_in_address', False):
            self._emit_eligible('9477', party,
                f"FEDABA in address detected → IF FOUND: 9477, ELSE: 9999")
        
        # 9476: CHIPS ABA from Name/Address
        if self._get(party, 'has_chips_in_address', False):
            self._emit_eligible('9476', party,
                f"CHIPS ABA in address detected → IF FOUND: 9476, ELSE: 9999")
        
        # 9479: Account Number Cleaned (directory enrichment)
        if self._get(party, 'has_bic', False):
            self._emit_eligible('9479', party,
                f"Party enrichment eligible (has BIC) → IF FOUND: 9479, ELSE: 9999")
        
        # 9961: BIC from Name/Address (fuzzy)
        if (self._get(party, 'has_name', False) and 
            not self._get(party, 'has_bic', False) and
            self._get(party, 'address_lines', 0) >= 2):
            self._emit_eligible('9961', party,
                f"BIC from Name/Address eligible (fuzzy) → IF FOUND: 9961, ELSE: 9999")
        
        # 9970: D-A using BIC from Name/Address
        if self._get(party, 'has_name', False) and not self._get(party, 'has_bic', False):
            # Check if BIC pattern exists in address
            pass  # Would need to detect BIC in address text
        
        # 9985: BIC from CHIPS ABA
        if self._get(party, 'has_nch', False) and self._get(party, 'nch_type') == 'CHIPS_OR_SORT':
            self._emit_eligible('9985', party,
                f"BIC from CHIPS ABA eligible → IF FOUND: 9985, ELSE: 9999")


# =============================================================================
# RESPONSE CODE EXTRACTOR
# =============================================================================

# Directory-dependent codes - cannot predict deterministically
# These require external directory lookups to determine success/failure
DIRECTORY_DEPENDENT_CODES = {
    # 8XXX that depend on directory lookup outcome
    '8004',  # IBAN cannot be derived - emitted when derivation FAILS
    '8036',  # FCDA name matching - requires CIF
    '8464',  # Target channel not derived
    '8465',  # Product code not found
    '8472',  # Fee code not derived
    
    # 9XXX that require directory lookups
    '9004', '9005', '9007', '9008', '9013', '9023', '9024', '9029', '9032',
    '9475', '9476', '9477', '9478', '9479', '9480', '9481', '9482',
    '9484', '9485', '9486', '9901', '9910', '9917', '9918', '9932',
    '9935', '9936', '9961', '9970', '9978', '9979', '9980', '9981',
    '9982', '9983', '9984', '9985', '9986', '9999',
}


def extract_actual_codes(response: Dict, include_directory_dependent: bool = False) -> Set[str]:
    """Extract actual ACE codes from Response, filtering to 8XXX/9XXX only."""
    codes = set()
    
    def find_codes(obj):
        if isinstance(obj, dict):
            for key in ['ACECodes', 'ErrorCodes', 'Codes', 'codes', 'errors',
                       'PelicanCodes', 'ValidationErrors', 'RepairCodes',
                       'ErrorCode', 'Code', 'code', 'ViolationCode', 'violationCode',
                       'ViolationCodes', 'AceViolation', 'aceViolation']:
                if key in obj:
                    val = obj[key]
                    if isinstance(val, list):
                        for item in val:
                            if isinstance(item, str):
                                code = extract_base_code(item)
                                if code:
                                    codes.add(code)
                            elif isinstance(item, dict):
                                c = item.get('Code') or item.get('code') or item.get('ErrorCode') or item.get('ViolationCode')
                                if c:
                                    code = extract_base_code(str(c))
                                    if code:
                                        codes.add(code)
                    elif isinstance(val, str):
                        for c in val.split(','):
                            code = extract_base_code(c.strip())
                            if code:
                                codes.add(code)
                    elif isinstance(val, (int, float)):
                        code = extract_base_code(str(int(val)))
                        if code:
                            codes.add(code)
            for v in obj.values():
                find_codes(v)
        elif isinstance(obj, list):
            for item in obj:
                find_codes(item)
    
    find_codes(response)
    
    # Filter to 8XXX and 9XXX only
    result = {c for c in codes if c.startswith('8') or c.startswith('9')}
    
    # Optionally exclude directory-dependent codes
    if not include_directory_dependent:
        result = result - DIRECTORY_DEPENDENT_CODES
    
    return result


def extract_base_code(code: str) -> Optional[str]:
    """Extract 4-digit base code."""
    match = re.match(r'^(\d{4})', str(code).strip())
    return match.group(1) if match else None


# =============================================================================
# FILE PROCESSOR
# =============================================================================

@dataclass
class VerificationResult:
    trn_id: str
    predicted: Set[str]
    actual: Set[str]
    passed: bool = False
    directory_codes: Set[str] = field(default_factory=set)
    directory_eligible: List[Tuple[str, str, str]] = field(default_factory=list)
    features: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        self.passed = (self.predicted == self.actual)


def get_transaction_id(data: Dict) -> str:
    """Try to extract transaction ID from IFML."""
    # Common locations for transaction ID
    try:
        if 'Request' in data:
            req = data['Request']
            if 'IFML' in req:
                ifml = req['IFML']
                if 'File' in ifml:
                    file_data = ifml['File']
                    if 'Message' in file_data:
                        msg = file_data['Message']
                        # Try BasicPayment
                        bp = msg.get('BasicPayment', [])
                        if bp and isinstance(bp, list) and bp:
                            bp = bp[0]
                        if isinstance(bp, dict):
                            for key in ['TransactionReference', 'TrnRef', 'Reference', 'MsgId', 'TRNID']:
                                if key in bp:
                                    return str(bp[key])
    except:
        pass
    return "UNKNOWN"


def process_payment(payment_id: str, request_data: Dict, response_data: Dict, 
                    include_directory_dependent: bool = False,
                    debug: bool = False) -> VerificationResult:
    """Process a single payment: extract features, predict, compare."""
    # Navigate to IFML
    ifml = request_data.get('IFML', request_data)
    
    # Extract features
    extractor = FeatureExtractor()
    features = extractor.extract(ifml)
    
    # Predict codes
    engine = RuleEngine(features)
    predicted = engine.predict()
    directory_eligible = engine.get_directory_eligible()
    
    # Extract actual codes (8XXX/9XXX only, excluding directory-dependent by default)
    actual = extract_actual_codes(response_data, include_directory_dependent)
    
    # Also get all actual codes for reporting
    all_actual = extract_actual_codes(response_data, include_directory_dependent=True)
    directory_codes = all_actual - actual
    
    result = VerificationResult(trn_id=payment_id, predicted=predicted, actual=actual)
    result.directory_codes = directory_codes  # Store for reporting
    result.directory_eligible = directory_eligible  # Store eligibility predictions
    result.features = features  # Store for debug output
    
    # Debug output for failures
    if debug and predicted != actual:
        print(f"\n=== DEBUG: {payment_id} ===")
        print("Key features:")
        for key in sorted(features.keys()):
            val = features[key]
            if val and val != False and val != 0 and val != '':
                if 'has_' in key or 'multiple' in key or 'duplicate' in key or 'valid' in key or 'match' in key or 'nch' in key:
                    print(f"  {key}: {val}")
        print(f"Predicted: {sorted(predicted)}")
        print(f"Actual: {sorted(actual)}")
    
    return result


def process_file(filepath: Path, verbose: bool = False, debug: bool = False) -> List[VerificationResult]:
    """Process a single JSON file containing IFML data."""
    results = []
    
    with open(filepath) as f:
        data = json.load(f)
    
    # Handle different file structures
    if isinstance(data, dict):
        # Check if it's a single payment with Request/Response
        if 'Request' in data and 'Response' in data:
            trn_id = get_transaction_id(data)
            result = process_payment(trn_id, data['Request'], data['Response'], INCLUDE_DIRECTORY, debug)
            results.append(result)
        
        # Check if it's keyed by payment ID
        else:
            for key, value in data.items():
                if isinstance(value, dict):
                    if 'Request' in value and 'Response' in value:
                        result = process_payment(key, value['Request'], value['Response'], INCLUDE_DIRECTORY, debug)
                        results.append(result)
                    elif 'IFML' in value:
                        # Just request, no response to compare
                        pass
    
    elif isinstance(data, list):
        # Array of payments
        for i, item in enumerate(data):
            if isinstance(item, dict) and 'Request' in item and 'Response' in item:
                trn_id = get_transaction_id(item) or f"payment_{i}"
                result = process_payment(trn_id, item['Request'], item['Response'], INCLUDE_DIRECTORY, debug)
                results.append(result)
    
    return results


def process_directory(dirpath: Path, verbose: bool = False, debug: bool = False) -> List[VerificationResult]:
    """Process all JSON files in a directory."""
    results = []
    json_files = sorted(dirpath.glob('*.json'))
    
    for filepath in json_files:
        try:
            file_results = process_file(filepath, verbose, debug)
            results.extend(file_results)
        except Exception as e:
            if verbose:
                print(f"Error processing {filepath.name}: {e}")
    
    return results


# =============================================================================
# OUTPUT
# =============================================================================

def print_results(results: List[VerificationResult], verbose: bool = False):
    """Print verification results."""
    passed = 0
    failed = 0
    
    for result in results:
        if result.passed:
            if verbose:
                dir_info = f" [Dir: {sorted(result.directory_codes)}]" if result.directory_codes else ""
                print(f"{result.trn_id} PASS{dir_info}")
            else:
                print(f"{result.trn_id} PASS")
            passed += 1
        else:
            failed += 1
            print(f"\n{result.trn_id} FAIL")
            print(f"  {'PREDICTED':<40} {'ACTUAL':<40}")
            print(f"  {'-'*40} {'-'*40}")
            
            # Align codes side by side
            pred_list = sorted(result.predicted)
            actual_list = sorted(result.actual)
            max_len = max(len(pred_list), len(actual_list))
            
            for i in range(max_len):
                pred = pred_list[i] if i < len(pred_list) else ''
                actual = actual_list[i] if i < len(actual_list) else ''
                
                # Mark differences
                pred_mark = '  ' if pred in result.actual else '+ '
                actual_mark = '  ' if actual in result.predicted else '- '
                
                print(f"  {pred_mark}{pred:<38} {actual_mark}{actual:<38}")
            
            # Summary
            only_predicted = result.predicted - result.actual
            only_actual = result.actual - result.predicted
            if only_predicted:
                print(f"  FP (predicted, not actual): {sorted(only_predicted)}")
            if only_actual:
                print(f"  FN (actual, not predicted): {sorted(only_actual)}")
            
            # Show directory-dependent codes that were excluded
            if result.directory_codes:
                print(f"  [Excluded directory-dependent: {sorted(result.directory_codes)}]")
            print()
        
        # Show directory eligibility predictions when --include-directory is enabled
        if INCLUDE_DIRECTORY and result.directory_eligible:
            print(f"\n  📋 Directory Lookup Predictions for {result.trn_id}:")
            for code, party, condition in result.directory_eligible:
                print(f"     [{party}] {condition}")
            
            # Match predictions against actual directory codes
            if result.directory_codes:
                print(f"\n  📊 Actual Directory Codes: {sorted(result.directory_codes)}")
                eligible_codes = {code for code, _, _ in result.directory_eligible}
                matched = eligible_codes & result.directory_codes
                if matched:
                    print(f"     ✓ Correctly predicted eligible: {sorted(matched)}")
                unmatched_actual = result.directory_codes - eligible_codes - {'9999'}
                if unmatched_actual:
                    print(f"     ✗ Actual but not predicted eligible: {sorted(unmatched_actual)}")
            print()
    
    # Summary
    total = len(results)
    if total > 0:
        print(f"\n{'='*60}")
        print(f"SUMMARY: {passed}/{total} passed ({100*passed/total:.1f}%), {failed} failed")
        if not INCLUDE_DIRECTORY:
            print(f"(Directory-dependent codes excluded from comparison)")
        else:
            print(f"(Directory-dependent codes INCLUDED in comparison)")
        print(f"{'='*60}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python verify_ifml.py <path> [--verbose] [--include-directory] [--debug]")
        print("  <path> can be a JSON file or directory of JSON files")
        print("  --verbose, -v      Show more details")
        print("  --include-directory  Include directory-dependent codes in comparison")
        print("  --debug            Show feature extraction details for each payment")
        sys.exit(1)
    
    path = Path(sys.argv[1])
    verbose = '--verbose' in sys.argv or '-v' in sys.argv
    include_directory = '--include-directory' in sys.argv
    debug = '--debug' in sys.argv
    
    if not path.exists():
        print(f"Error: {path} does not exist")
        sys.exit(1)
    
    # Store flags globally for process functions
    global INCLUDE_DIRECTORY, DEBUG_MODE
    INCLUDE_DIRECTORY = include_directory
    DEBUG_MODE = debug
    
    if path.is_file():
        results = process_file(path, verbose, debug)
    else:
        results = process_directory(path, verbose, debug)
    
    if not results:
        print("No payments processed. Check file format.")
        sys.exit(1)
    
    print_results(results, verbose)


# Global flag
INCLUDE_DIRECTORY = False
DEBUG_MODE = False


if __name__ == '__main__':
    main()
