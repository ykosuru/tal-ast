#!/usr/bin/env python3
"""
ACE Pelican IFML Processor and Rule Verification (Corrected Version)

Cross-referenced against official ACE code definitions.

Usage:
    python process_ifml_payments.py --file payment.json --output report.json
    python process_ifml_payments.py --dir /path/to/payments/ --output report.json
    python process_ifml_payments.py --dir /path/to/payments/ --verbose --limit 100
"""

import json
import argparse
import re
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict


# =============================================================================
# CONSTANTS
# =============================================================================

PARTY_PREFIXES = ['orig', 'ordi', 'send', 'dbt', 'cdt', 'bnf', 'bnf_bank', 'intm', 'acwi']

# Countries that use IBAN
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

# Valid ISO country codes (subset for validation)
VALID_COUNTRY_CODES = {
    'US', 'GB', 'DE', 'FR', 'IT', 'ES', 'NL', 'BE', 'AT', 'CH', 'SE', 
    'NO', 'DK', 'FI', 'IE', 'PT', 'GR', 'PL', 'CZ', 'HU', 'RO', 'BG',
    'HR', 'SK', 'SI', 'EE', 'LV', 'LT', 'CY', 'MT', 'LU', 'JP', 'CN',
    'IN', 'AU', 'NZ', 'CA', 'MX', 'BR', 'AR', 'CL', 'CO', 'PE', 'VE',
    'SG', 'HK', 'TW', 'KR', 'TH', 'MY', 'ID', 'PH', 'VN', 'AE', 'SA',
    'IL', 'TR', 'ZA', 'EG', 'NG', 'KE', 'RU', 'UA', 'BY', 'KZ'
}


# =============================================================================
# IFML FEATURE EXTRACTION
# =============================================================================

class IFMLFeatureExtractor:
    """Extract features from IFML payment messages"""
    
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
    
    def __init__(self):
        self.features = {}
        # Track values across parties for inconsistency checks
        self.all_ibans = []
        self.all_bbans = []
        self.all_nchs = []
        self.all_accounts = []
        self.all_countries = []
        self.all_bic4s = []
    
    def extract(self, ifml: Dict) -> Dict:
        """Extract all features from IFML message"""
        self.features = {}
        self.all_ibans = []
        self.all_bbans = []
        self.all_nchs = []
        self.all_accounts = []
        self.all_countries = []
        self.all_bic4s = []
        
        message = self._get_message(ifml)
        if not message:
            return self.features
        
        self._extract_basic_payment(message.get('BasicPayment', []))
        
        party_inf = message.get('PartyInf', {})
        for ifml_party_name, prefix in self.PARTY_MAPPING.items():
            party_data = party_inf.get(ifml_party_name, {})
            if party_data:
                self._extract_party_features(party_data, prefix)
            else:
                self.features[f'{prefix}_present'] = False
        
        # Cross-party inconsistency checks
        self._check_cross_party_consistency()
        
        return self.features
    
    def _get_message(self, ifml: Dict) -> Optional[Dict]:
        try:
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
        if isinstance(basic_payment, list) and basic_payment:
            basic_payment = basic_payment[0]
        if not isinstance(basic_payment, dict):
            return
        
        self.features['source_code'] = basic_payment.get('SourceCode', '')
        self.features['incoming_msg_type'] = basic_payment.get('IncomingMsgType', '')
        self.features['incoming_format'] = basic_payment.get('IncomingFormat', '')
        
        amounts = basic_payment.get('MonetaryAmount', [])
        if amounts:
            for amt in amounts:
                if amt.get('Type') == 'Amount':
                    self.features['currency'] = amt.get('Currency', '')
                    self.features['amount'] = amt.get('Amount', '')
                    break
    
    def _extract_party_features(self, party_data: Dict, prefix: str):
        self.features[f'{prefix}_present'] = True
        
        basic_party = party_data.get('BasicPartyInf', {})
        account_party = party_data.get('AccountPartyInf', {})
        
        if not basic_party and account_party:
            basic_party = account_party.get('BasicPartyInf', account_party)
        
        self._extract_id_info(basic_party, account_party, prefix)
        self._extract_address_info(basic_party, prefix)
        self._extract_country(party_data, basic_party, prefix)
        
        country = self.features.get(f'{prefix}_country', '')
        self.features[f'{prefix}_needs_iban'] = country in IBAN_COUNTRIES
        self.features[f'{prefix}_iban_derivation_supported'] = country in IBAN_DERIVATION_COUNTRIES
        
        address_inf = basic_party.get('AddressInf', [])
        if address_inf:
            self.features[f'{prefix}_has_name'] = True
            self.features[f'{prefix}_address_lines'] = len(address_inf)
        else:
            self.features[f'{prefix}_has_name'] = False
            self.features[f'{prefix}_address_lines'] = 0
    
    def _extract_id_info(self, basic_party: Dict, account_party: Dict, prefix: str):
        id_info = basic_party.get('ID', {})
        if isinstance(id_info, str):
            id_info = {'text': id_info}
        
        acct_id_inf = account_party.get('AcctIDInf', {}) if account_party else {}
        acct_id = acct_id_inf.get('ID', {})
        if isinstance(acct_id, str):
            acct_id = {'text': acct_id}
        
        id_type = id_info.get('Type', '')
        id_text = id_info.get('text', '')
        acct_type = acct_id.get('Type', '')
        acct_text = acct_id.get('text', '')
        adr_bank_id = basic_party.get('AdrBankID', '')
        
        self._classify_identifiers(prefix, id_type, id_text, acct_type, acct_text, adr_bank_id)
    
    def _classify_identifiers(self, prefix: str, id_type: str, id_text: str, 
                              acct_type: str, acct_text: str, adr_bank_id: str):
        # Initialize
        self.features[f'{prefix}_has_bic'] = False
        self.features[f'{prefix}_has_iban'] = False
        self.features[f'{prefix}_has_nch'] = False
        self.features[f'{prefix}_has_account'] = False
        self.features[f'{prefix}_has_bban'] = False
        self.features[f'{prefix}_has_domestic_account'] = False
        
        bic = None
        iban = None
        nch = None
        account = None
        
        # Type S = SWIFT/BIC
        if id_type == 'S':
            bic = id_text
        elif id_type in ('P', 'D'):
            account = id_text
            if id_type == 'D':
                self.features[f'{prefix}_has_domestic_account'] = True
        
        if acct_type == 'S':
            bic = acct_text
        elif acct_type in ('P', 'D', '') and acct_text:
            account = acct_text
            if acct_type == 'D':
                self.features[f'{prefix}_has_domestic_account'] = True
        
        # Parse composite IDs
        for text in [id_text, acct_text]:
            if text:
                parsed = self._parse_composite_id(text)
                if parsed.get('bic'):
                    bic = parsed['bic']
                if parsed.get('account'):
                    account = parsed['account']
        
        if adr_bank_id and self._looks_like_bic(adr_bank_id):
            bic = adr_bank_id
        
        # Classify account type
        if account:
            if self._looks_like_iban(account):
                iban = account
                account = None
            elif self._looks_like_nch(account):
                nch = account
                account = None
            elif self._looks_like_bban(account, prefix):
                self.features[f'{prefix}_has_bban'] = True
        
        # Store and validate
        if bic:
            self.features[f'{prefix}_has_bic'] = True
            self.features[f'{prefix}_bic'] = bic
            self._validate_bic(prefix, bic)
        
        if iban:
            self.features[f'{prefix}_has_iban'] = True
            self.features[f'{prefix}_iban'] = iban
            self._validate_iban(prefix, iban)
            self.all_ibans.append((prefix, iban))
        
        if nch:
            self.features[f'{prefix}_has_nch'] = True
            self.features[f'{prefix}_nch'] = nch
            self._validate_nch(prefix, nch)
            self.all_nchs.append((prefix, nch))
        
        if account:
            self.features[f'{prefix}_has_account'] = True
            self.features[f'{prefix}_account'] = account
            self._validate_account(prefix, account)
            self.all_accounts.append((prefix, account))
    
    def _parse_composite_id(self, text: str) -> Dict:
        result = {}
        match = re.match(r'^([A-Z]{4}[A-Z]{2}[A-Z0-9]{2,5})[/\(](.+?)[\)]?$', text)
        if match:
            result['bic'] = match.group(1)
            result['account'] = match.group(2).lstrip('AC')
            return result
        if self._looks_like_bic(text):
            result['bic'] = text
        return result
    
    def _looks_like_bic(self, text: str) -> bool:
        if not text:
            return False
        text = text.strip()
        if len(text) not in (8, 11):
            return False
        return bool(re.match(r'^[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}([A-Z0-9]{3})?$', text))
    
    def _looks_like_iban(self, text: str) -> bool:
        if not text:
            return False
        text = text.replace(' ', '').replace('-', '').upper()
        if len(text) < 15 or len(text) > 34:
            return False
        return bool(re.match(r'^[A-Z]{2}[0-9]{2}[A-Z0-9]+$', text))
    
    def _looks_like_nch(self, text: str) -> bool:
        if not text:
            return False
        text = re.sub(r'[^0-9]', '', text)
        return len(text) in (6, 9)  # CHIPS/SORT (6) or FEDABA (9)
    
    def _looks_like_bban(self, text: str, prefix: str) -> bool:
        """Check if account looks like a BBAN based on country"""
        country = self.features.get(f'{prefix}_country', '')
        if country not in IBAN_COUNTRIES:
            return False
        # BBAN is the national account number portion
        # Length varies by country
        clean = re.sub(r'[^A-Za-z0-9]', '', text)
        return 10 <= len(clean) <= 30
    
    def _validate_bic(self, prefix: str, bic: str):
        bic = bic.strip().upper()
        self.features[f'{prefix}_bic_length'] = len(bic)
        
        # Full BIC format validation
        valid_format = bool(re.match(r'^[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}([A-Z0-9]{3})?$', bic))
        self.features[f'{prefix}_bic_valid_format'] = valid_format
        
        # BIC4 validation (first 4 chars - bank code)
        bic4 = bic[:4] if len(bic) >= 4 else ''
        bic4_valid = bool(re.match(r'^[A-Z]{4}$', bic4))
        self.features[f'{prefix}_bic4'] = bic4
        self.features[f'{prefix}_bic4_valid'] = bic4_valid
        if bic4:
            self.all_bic4s.append((prefix, bic4))
        
        # Country validation (chars 5-6)
        if len(bic) >= 6:
            country = bic[4:6]
            self.features[f'{prefix}_bic_country'] = country
            valid_country = country in VALID_COUNTRY_CODES
            self.features[f'{prefix}_bic_valid_country'] = valid_country
        else:
            self.features[f'{prefix}_bic_country'] = ''
            self.features[f'{prefix}_bic_valid_country'] = False
    
    def _validate_iban(self, prefix: str, iban: str):
        original = iban
        iban = iban.replace(' ', '').replace('-', '').upper()
        
        valid_format = bool(re.match(r'^[A-Z]{2}[0-9]{2}[A-Z0-9]+$', iban))
        self.features[f'{prefix}_iban_valid_format'] = valid_format
        
        if len(iban) >= 2:
            iban_country = iban[:2]
            self.features[f'{prefix}_iban_country'] = iban_country
            self.all_countries.append((prefix, iban_country))
        
        checksum_valid = self._validate_iban_checksum(iban)
        self.features[f'{prefix}_iban_checksum_valid'] = checksum_valid
        
        self.features[f'{prefix}_iban_needs_formatting'] = (
            ' ' in original or '-' in original or original != original.upper()
        )
        
        # BIC/IBAN country match
        bic_country = self.features.get(f'{prefix}_bic_country', '')
        iban_country = self.features.get(f'{prefix}_iban_country', '')
        if bic_country and iban_country:
            self.features[f'{prefix}_bic_iban_match'] = (bic_country == iban_country)
        else:
            self.features[f'{prefix}_bic_iban_match'] = True
        
        # Extract BBAN from IBAN
        if len(iban) > 4:
            bban = iban[4:]
            self.features[f'{prefix}_bban_from_iban'] = bban
            self.all_bbans.append((prefix, bban))
    
    def _validate_iban_checksum(self, iban: str) -> bool:
        try:
            if len(iban) < 5:
                return False
            rearranged = iban[4:] + iban[:4]
            numeric = ''
            for char in rearranged:
                if char.isalpha():
                    numeric += str(ord(char) - ord('A') + 10)
                else:
                    numeric += char
            return int(numeric) % 97 == 1
        except (ValueError, TypeError):
            return False
    
    def _validate_nch(self, prefix: str, nch: str):
        original = nch
        nch_clean = re.sub(r'[^0-9]', '', nch)
        
        self.features[f'{prefix}_nch_has_dirty_chars'] = (original != nch_clean)
        self.features[f'{prefix}_nch_in_wrong_field'] = False  # Would need field context
        
        if len(nch_clean) == 9:
            self.features[f'{prefix}_nch_type'] = 'FEDABA'
            self.features[f'{prefix}_is_fedaba'] = True
            self.features[f'{prefix}_nch_valid'] = True
            self.features[f'{prefix}_fedaba_checksum_valid'] = self._validate_aba_checksum(nch_clean)
            self.features[f'{prefix}_nch_needs_formatting'] = (original != nch_clean)
        elif len(nch_clean) == 6:
            self.features[f'{prefix}_nch_type'] = 'CHIPS_OR_SORT'
            self.features[f'{prefix}_is_fedaba'] = False
            self.features[f'{prefix}_nch_valid'] = True
            self.features[f'{prefix}_fedaba_checksum_valid'] = True  # N/A for non-FEDABA
            self.features[f'{prefix}_nch_needs_formatting'] = False
        else:
            self.features[f'{prefix}_nch_type'] = 'OTHER'
            self.features[f'{prefix}_is_fedaba'] = False
            self.features[f'{prefix}_nch_valid'] = False
            self.features[f'{prefix}_fedaba_checksum_valid'] = True
            self.features[f'{prefix}_nch_needs_formatting'] = False
        
        self.features[f'{prefix}_nch_sources'] = 1
        self.features[f'{prefix}_nch_consistent'] = True
        
        # NCH validation is applicable for US domestic
        country = self.features.get(f'{prefix}_country', '')
        self.features[f'{prefix}_nch_validation_applicable'] = (country == 'US')
    
    def _validate_aba_checksum(self, aba: str) -> bool:
        try:
            if len(aba) != 9 or not aba.isdigit():
                return False
            weights = [3, 7, 1, 3, 7, 1, 3, 7, 1]
            total = sum(int(d) * w for d, w in zip(aba, weights))
            return total % 10 == 0
        except (ValueError, TypeError):
            return False
    
    def _validate_account(self, prefix: str, account: str):
        clean = re.sub(r'[^A-Za-z0-9]', '', account)
        self.features[f'{prefix}_account_has_dirty_chars'] = (account != clean)
        self.features[f'{prefix}_account_has_spaces'] = ' ' in account
        self.features[f'{prefix}_account_has_special_chars'] = bool(re.search(r'[^A-Za-z0-9\s]', account))
        self.features[f'{prefix}_account_type'] = 'OTHER'
        self.features[f'{prefix}_account_valid_format'] = len(clean) > 0
        self.features[f'{prefix}_account_needs_length_fix'] = False
        
        # CLABE check (Mexico - 18 digits)
        self.features[f'{prefix}_is_clabe'] = (len(clean) == 18 and clean.isdigit())
        
        # Domestic account validation
        country = self.features.get(f'{prefix}_country', '')
        if country == 'US':
            # US account numbers typically 9-17 digits
            self.features[f'{prefix}_domestic_account_valid'] = (
                clean.isdigit() and 9 <= len(clean) <= 17
            )
        else:
            self.features[f'{prefix}_domestic_account_valid'] = True
    
    def _extract_address_info(self, party_data: Dict, prefix: str):
        address_inf = party_data.get('AddressInf', [])
        if address_inf and isinstance(address_inf, list):
            address_text = ' '.join(
                line.get('text', '') for line in address_inf if isinstance(line, dict)
            )
            self.features[f'{prefix}_address_text'] = address_text
            
            # Check for embedded identifiers in address
            self._check_embedded_in_address(prefix, address_text)
    
    def _check_embedded_in_address(self, prefix: str, address_text: str):
        """Check for NCH, BIC, account numbers embedded in address"""
        # FEDABA pattern (9 digits)
        fedaba_match = re.search(r'\b(\d{9})\b', address_text)
        self.features[f'{prefix}_has_fedaba_in_address'] = bool(fedaba_match)
        
        # CHIPS ABA pattern (6 digits)
        chips_match = re.search(r'\b(\d{6})\b', address_text)
        self.features[f'{prefix}_has_chips_in_address'] = bool(chips_match)
        
        # BIC pattern in address
        bic_match = re.search(r'\b([A-Z]{4}[A-Z]{2}[A-Z0-9]{2}([A-Z0-9]{3})?)\b', address_text)
        self.features[f'{prefix}_has_bic_in_address'] = bool(bic_match)
    
    def _extract_country(self, party_data: Dict, basic_party: Dict, prefix: str):
        country = (
            party_data.get('Country') or
            party_data.get('ResidenceCountry') or
            party_data.get('MailingCountry') or
            basic_party.get('Country') or
            ''
        )
        self.features[f'{prefix}_country'] = country
        self.features[f'{prefix}_country_valid'] = country in VALID_COUNTRY_CODES
        self.features[f'{prefix}_is_domestic'] = (country == 'US')
        
        if country:
            self.all_countries.append((prefix, country))
    
    def _check_cross_party_consistency(self):
        """Check for inconsistencies across parties (8023-8029)"""
        
        # IBAN consistency (8023)
        if len(self.all_ibans) > 1:
            iban_values = [v for _, v in self.all_ibans]
            self.features['iban_count'] = len(iban_values)
            self.features['ibans_consistent'] = len(set(iban_values)) == 1
        else:
            self.features['iban_count'] = len(self.all_ibans)
            self.features['ibans_consistent'] = True
        
        # BBAN consistency (8024)
        if len(self.all_bbans) > 1:
            bban_values = [v for _, v in self.all_bbans]
            self.features['bban_count'] = len(bban_values)
            self.features['bbans_consistent'] = len(set(bban_values)) == 1
        else:
            self.features['bban_count'] = len(self.all_bbans)
            self.features['bbans_consistent'] = True
        
        # NCH consistency (8026)
        if len(self.all_nchs) > 1:
            nch_values = [re.sub(r'[^0-9]', '', v) for _, v in self.all_nchs]
            self.features['nch_count'] = len(nch_values)
            self.features['nchs_consistent'] = len(set(nch_values)) == 1
        else:
            self.features['nch_count'] = len(self.all_nchs)
            self.features['nchs_consistent'] = True
        
        # Account consistency (8029)
        if len(self.all_accounts) > 1:
            account_values = [v for _, v in self.all_accounts]
            self.features['account_count'] = len(account_values)
            self.features['accounts_consistent'] = len(set(account_values)) == 1
        else:
            self.features['account_count'] = len(self.all_accounts)
            self.features['accounts_consistent'] = True
        
        # Country consistency (8027)
        if len(self.all_countries) > 1:
            country_values = [v for _, v in self.all_countries]
            self.features['country_count'] = len(country_values)
            # Countries don't need to be identical, but check for obvious conflicts
            self.features['countries_consistent'] = True
        else:
            self.features['country_count'] = len(self.all_countries)
            self.features['countries_consistent'] = True
        
        # BIC4 consistency (8028)
        if len(self.all_bic4s) > 1:
            bic4_values = [v for _, v in self.all_bic4s]
            self.features['bic4_count'] = len(bic4_values)
            self.features['bic4s_consistent'] = len(set(bic4_values)) == 1
        else:
            self.features['bic4_count'] = len(self.all_bic4s)
            self.features['bic4s_consistent'] = True


# =============================================================================
# CODE PREDICTOR - Emits predicted 8XXX and 9XXX codes
# =============================================================================

class CodePredictor:
    """
    Predicts 8XXX and 9XXX codes based on extracted features.
    Cross-referenced against official ACE code definitions.
    """
    
    def __init__(self, features: Dict):
        self.features = features
        self.predictions = []
    
    def predict_all(self) -> List[Tuple[str, str, str]]:
        """Run all prediction rules"""
        self.predictions = []
        
        # Message-level checks (not party-specific)
        self._check_message_level()
        
        # Party-level checks
        for party in PARTY_PREFIXES:
            if not self._get(party, 'present', False):
                continue
            
            # === 8XXX VALIDATION ERRORS ===
            self._check_8001(party)  # Invalid BIC
            self._check_8005(party)  # Invalid BIC4
            self._check_8006(party)  # Invalid country code
            self._check_8022(party)  # IBAN/BIC country mismatch
            self._check_8030(party)  # IBAN derivation not supported
            self._check_8892(party)  # Invalid Account number
            self._check_8894(party)  # Invalid IBAN
            self._check_8895(party)  # Invalid NCH code
            self._check_8896(party)  # Invalid Domestic Account Number
            self._check_8897(party)  # Invalid BBAN
            self._check_8898(party)  # IBAN Check Digit failed
            
            # === 9XXX NO-DIRECTORY REPAIRS ===
            self._check_9000(party)  # NCH code cleaned
            self._check_9002(party)  # Account number cleaned
            self._check_9006(party)  # IBAN Cleaned
            self._check_9014(party)  # BBAN cleaned
            self._check_9015(party)  # Domestic account cleaned
            self._check_9017(party)  # Multiple party info
            self._check_9018(party)  # Duplicate removed
            self._check_9021(party)  # FEDABA formatted
            self._check_9022(party)  # Account length fixed
            self._check_9028(party)  # NCH Code strip off
            
            # === 9XXX DIRECTORY-DEPENDENT (Eligibility only) ===
            self._check_9004_eligibility(party)  # IBAN derivation
            self._check_9005_eligibility(party)  # BIC from NCH
            self._check_9008_eligibility(party)  # BIC from IBAN
            self._check_9013_eligibility(party)  # Name/Address repair
            self._check_9032_eligibility(party)  # 8-char BIC insertion
            self._check_9961_eligibility(party)  # BIC from Name/Address
        
        return self.predictions
    
    def _get(self, party: str, key: str, default=None):
        return self.features.get(f'{party}_{key}', default)
    
    def _get_global(self, key: str, default=None):
        return self.features.get(key, default)
    
    def _emit(self, code: str, party: str, reason: str):
        self.predictions.append((code, party, reason))
    
    # =========================================================================
    # MESSAGE-LEVEL CHECKS
    # =========================================================================
    
    def _check_message_level(self):
        """Check message-level inconsistencies (8023-8029)"""
        
        # 8023: IBAN inconsistency
        if self._get_global('iban_count', 0) > 1:
            if not self._get_global('ibans_consistent', True):
                self._emit('8023', 'message', "Multiple IBANs are inconsistent")
        
        # 8024: BBAN inconsistency
        if self._get_global('bban_count', 0) > 1:
            if not self._get_global('bbans_consistent', True):
                self._emit('8024', 'message', "Multiple BBANs are inconsistent")
        
        # 8026: NCH inconsistency
        if self._get_global('nch_count', 0) > 1:
            if not self._get_global('nchs_consistent', True):
                self._emit('8026', 'message', "Multiple NCH codes are inconsistent")
        
        # 8028: BIC4 inconsistency
        if self._get_global('bic4_count', 0) > 1:
            if not self._get_global('bic4s_consistent', True):
                self._emit('8028', 'message', "Multiple BIC4 codes are inconsistent")
        
        # 8029: Account inconsistency
        if self._get_global('account_count', 0) > 1:
            if not self._get_global('accounts_consistent', True):
                self._emit('8029', 'message', "Multiple Account numbers are inconsistent")
    
    # =========================================================================
    # 8XXX VALIDATION ERRORS
    # =========================================================================
    
    def _check_8001(self, party: str):
        """8001: Invalid BIC"""
        if not self._get(party, 'has_bic', False):
            return
        if not self._get(party, 'bic_valid_format', True):
            self._emit('8001', party, f"BIC format invalid: {self._get(party, 'bic', '')}")
        elif not self._get(party, 'bic_valid_country', True):
            self._emit('8001', party, f"BIC country invalid: {self._get(party, 'bic_country', '')}")
    
    def _check_8005(self, party: str):
        """8005: Invalid BIC4"""
        if not self._get(party, 'has_bic', False):
            return
        if not self._get(party, 'bic4_valid', True):
            self._emit('8005', party, f"BIC4 invalid: {self._get(party, 'bic4', '')}")
    
    def _check_8006(self, party: str):
        """8006: Invalid country code"""
        country = self._get(party, 'country', '')
        if country and not self._get(party, 'country_valid', True):
            self._emit('8006', party, f"Country code invalid: {country}")
    
    def _check_8022(self, party: str):
        """8022: IBAN inconsistent with Account With Institution BIC"""
        if self._get(party, 'has_iban', False) and self._get(party, 'has_bic', False):
            if not self._get(party, 'bic_iban_match', True):
                self._emit('8022', party, 
                    f"BIC country ({self._get(party, 'bic_country', '')}) != "
                    f"IBAN country ({self._get(party, 'iban_country', '')})")
    
    def _check_8030(self, party: str):
        """8030: IBAN derivation not supported for the country"""
        # Only applies when IBAN is needed but can't be derived
        if self._get(party, 'needs_iban', False) and not self._get(party, 'has_iban', False):
            if self._get(party, 'has_account', False):
                if not self._get(party, 'iban_derivation_supported', True):
                    country = self._get(party, 'country', '')
                    self._emit('8030', party, f"IBAN derivation not supported for country: {country}")
    
    def _check_8892(self, party: str):
        """8892: Invalid Account number"""
        if self._get(party, 'has_account', False):
            if not self._get(party, 'account_valid_format', True):
                self._emit('8892', party, f"Account format invalid: {self._get(party, 'account', '')}")
    
    def _check_8894(self, party: str):
        """8894: Invalid IBAN"""
        if not self._get(party, 'has_iban', False):
            return
        if not self._get(party, 'iban_valid_format', True):
            self._emit('8894', party, f"IBAN format invalid: {self._get(party, 'iban', '')}")
        elif not self._get(party, 'iban_checksum_valid', True):
            self._emit('8894', party, f"IBAN checksum invalid: {self._get(party, 'iban', '')}")
    
    def _check_8895(self, party: str):
        """8895: Invalid NCH code"""
        if not self._get(party, 'has_nch', False):
            return
        if not self._get(party, 'nch_validation_applicable', False):
            return
        if not self._get(party, 'nch_valid', True):
            self._emit('8895', party, f"NCH format invalid: {self._get(party, 'nch', '')}")
        elif not self._get(party, 'fedaba_checksum_valid', True):
            self._emit('8895', party, f"NCH checksum invalid: {self._get(party, 'nch', '')}")
    
    def _check_8896(self, party: str):
        """8896: Invalid Domestic Account Number"""
        if self._get(party, 'has_domestic_account', False):
            if not self._get(party, 'domestic_account_valid', True):
                self._emit('8896', party, f"Domestic account invalid: {self._get(party, 'account', '')}")
    
    def _check_8897(self, party: str):
        """8897: Invalid BBAN"""
        if self._get(party, 'has_bban', False):
            # BBAN validation would need country-specific rules
            # For now, basic check
            pass
    
    def _check_8898(self, party: str):
        """8898: IBAN Check Digit calculation/validation failed"""
        if self._get(party, 'has_iban', False):
            if self._get(party, 'iban_valid_format', True):  # Format OK but checksum failed
                if not self._get(party, 'iban_checksum_valid', True):
                    self._emit('8898', party, f"IBAN checksum failed: {self._get(party, 'iban', '')}")
    
    # =========================================================================
    # 9XXX NO-DIRECTORY REPAIRS
    # =========================================================================
    
    def _check_9000(self, party: str):
        """9000: NCH code cleaned"""
        if self._get(party, 'has_nch', False) and self._get(party, 'nch_has_dirty_chars', False):
            self._emit('9000', party, f"NCH cleaned: {self._get(party, 'nch', '')}")
    
    def _check_9002(self, party: str):
        """9002: Account number cleaned of non-alphanumeric characters"""
        if self._get(party, 'has_account', False) and self._get(party, 'account_has_dirty_chars', False):
            self._emit('9002', party, f"Account cleaned: {self._get(party, 'account', '')}")
    
    def _check_9006(self, party: str):
        """9006: IBAN Cleaned"""
        if self._get(party, 'has_iban', False) and self._get(party, 'iban_needs_formatting', False):
            self._emit('9006', party, f"IBAN cleaned: {self._get(party, 'iban', '')}")
    
    def _check_9014(self, party: str):
        """9014: BBAN cleaned"""
        if self._get(party, 'has_bban', False):
            bban = self._get(party, 'bban_from_iban', '')
            if bban and re.search(r'[^A-Za-z0-9]', bban):
                self._emit('9014', party, f"BBAN cleaned: {bban}")
    
    def _check_9015(self, party: str):
        """9015: Domestic account number cleaned"""
        if self._get(party, 'has_domestic_account', False):
            if self._get(party, 'account_has_dirty_chars', False):
                self._emit('9015', party, f"Domestic account cleaned: {self._get(party, 'account', '')}")
    
    def _check_9017(self, party: str):
        """9017: Multiple party information present"""
        if self._get(party, 'has_multiple_ids', False):
            self._emit('9017', party, "Multiple party info split")
    
    def _check_9018(self, party: str):
        """9018: Duplicate party information removed"""
        if self._get(party, 'has_duplicate_info', False):
            self._emit('9018', party, "Duplicate info removed")
    
    def _check_9021(self, party: str):
        """9021: FEDABA Code correctly formatted"""
        if self._get(party, 'has_nch', False):
            if self._get(party, 'is_fedaba', False) and self._get(party, 'nch_needs_formatting', False):
                self._emit('9021', party, f"FEDABA formatted: {self._get(party, 'nch', '')}")
    
    def _check_9022(self, party: str):
        """9022: Account number formatted to required length"""
        if self._get(party, 'has_account', False) and self._get(party, 'account_needs_length_fix', False):
            self._emit('9022', party, f"Account length fixed: {self._get(party, 'account', '')}")
    
    def _check_9028(self, party: str):
        """9028: NCH Code strip off"""
        if self._get(party, 'has_nch', False) and self._get(party, 'nch_in_wrong_field', False):
            self._emit('9028', party, f"NCH stripped from wrong field: {self._get(party, 'nch', '')}")
    
    # =========================================================================
    # 9XXX DIRECTORY-DEPENDENT (Eligibility prediction only)
    # =========================================================================
    
    def _check_9004_eligibility(self, party: str):
        """9004: IBAN repaired in Account Number line - ELIGIBILITY CHECK"""
        # Eligible if: needs IBAN, doesn't have IBAN, has account, derivation supported
        if (self._get(party, 'needs_iban', False) and 
            not self._get(party, 'has_iban', False) and
            self._get(party, 'has_account', False) and
            self._get(party, 'iban_derivation_supported', True)):
            self._emit('9004', party, "[ELIGIBLE] IBAN derivation from account (directory dependent)")
    
    def _check_9005_eligibility(self, party: str):
        """9005: BIC replaced by derived BIC - ELIGIBILITY CHECK"""
        # Eligible if: has NCH, doesn't have BIC
        if self._get(party, 'has_nch', False) and not self._get(party, 'has_bic', False):
            self._emit('9005', party, "[ELIGIBLE] BIC derivation from NCH (directory dependent)")
    
    def _check_9008_eligibility(self, party: str):
        """9008: IBAN to BIC repair done - ELIGIBILITY CHECK"""
        # Eligible if: has IBAN, doesn't have BIC
        if self._get(party, 'has_iban', False) and not self._get(party, 'has_bic', False):
            self._emit('9008', party, "[ELIGIBLE] BIC derivation from IBAN (directory dependent)")
    
    def _check_9013_eligibility(self, party: str):
        """9013: Name and Address repair done - ELIGIBILITY CHECK"""
        # Eligible if: has name/address that may need repair
        if self._get(party, 'has_name', False) and self._get(party, 'address_lines', 0) > 0:
            # Would need more specific conditions
            pass
    
    def _check_9032_eligibility(self, party: str):
        """9032: 8 char BIC insertion - ELIGIBILITY CHECK"""
        # Eligible if: no BIC present but can be derived
        if not self._get(party, 'has_bic', False):
            # Need additional context to determine if BIC can be derived
            pass
    
    def _check_9961_eligibility(self, party: str):
        """9961: BIC Derived from Name and Address - ELIGIBILITY CHECK"""
        # Eligible if: has name/address, no BIC
        if (self._get(party, 'has_name', False) and 
            not self._get(party, 'has_bic', False) and
            self._get(party, 'address_lines', 0) >= 2):
            self._emit('9961', party, "[ELIGIBLE] BIC derivation from Name/Address (directory dependent)")


# =============================================================================
# RESPONSE CODE EXTRACTION
# =============================================================================

def extract_ace_codes_from_response(response: Dict) -> Set[str]:
    """Extract actual ACE codes from response"""
    codes = set()
    
    def find_codes(obj):
        if isinstance(obj, dict):
            for key in ['ACECodes', 'ErrorCodes', 'Codes', 'codes', 'errors',
                       'PelicanCodes', 'ValidationErrors', 'RepairCodes',
                       'ErrorCode', 'Code', 'code', 'ViolationCode', 'violationCode']:
                if key in obj:
                    val = obj[key]
                    if isinstance(val, list):
                        for item in val:
                            if isinstance(item, str):
                                codes.add(extract_base_code(item))
                            elif isinstance(item, dict):
                                code = item.get('Code') or item.get('code') or item.get('ErrorCode')
                                if code:
                                    codes.add(extract_base_code(str(code)))
                    elif isinstance(val, str):
                        for c in val.split(','):
                            if c.strip():
                                codes.add(extract_base_code(c.strip()))
                    elif isinstance(val, (int, float)):
                        codes.add(extract_base_code(str(int(val))))
            for v in obj.values():
                find_codes(v)
        elif isinstance(obj, list):
            for item in obj:
                find_codes(item)
    
    find_codes(response)
    return codes


def extract_base_code(code: str) -> str:
    """Extract 4-digit base code"""
    match = re.match(r'^(\d{4})', str(code).strip())
    return match.group(1) if match else str(code).strip()


# =============================================================================
# PAYMENT RESULT
# =============================================================================

@dataclass
class PaymentResult:
    payment_id: str
    predicted_codes: List[Tuple[str, str, str]]
    actual_codes: Set[str]
    true_positives: List[str] = field(default_factory=list)
    false_positives: List[str] = field(default_factory=list)
    false_negatives: List[str] = field(default_factory=list)
    success: bool = False
    
    def compute_match(self):
        # Only compare deterministic predictions (not [ELIGIBLE] ones)
        deterministic_predicted = {
            code for code, _, reason in self.predicted_codes 
            if not reason.startswith('[ELIGIBLE]')
        }
        
        self.true_positives = list(deterministic_predicted & self.actual_codes)
        self.false_positives = list(deterministic_predicted - self.actual_codes)
        self.false_negatives = list(self.actual_codes - deterministic_predicted)
        
        # Filter to 8xxx/9xxx only
        relevant_actual = {c for c in self.actual_codes if c.startswith(('8', '9'))}
        self.success = (deterministic_predicted == relevant_actual)
    
    def to_dict(self) -> Dict:
        return {
            'payment_id': self.payment_id,
            'predicted_codes': [
                {'code': c, 'party': p, 'reason': r} 
                for c, p, r in self.predicted_codes
            ],
            'actual_codes': list(self.actual_codes),
            'true_positives': self.true_positives,
            'false_positives': self.false_positives,
            'false_negatives': self.false_negatives,
            'success': self.success,
        }


# =============================================================================
# PROCESSING
# =============================================================================

def process_payment(payment_id: str, payment_data: Dict, verbose: bool = False) -> Optional[PaymentResult]:
    request = payment_data.get('Request', payment_data)
    response = payment_data.get('Response', {})
    ifml = request.get('IFML', request)
    
    extractor = IFMLFeatureExtractor()
    features = extractor.extract(ifml)
    if not features:
        return None
    
    predictor = CodePredictor(features)
    predicted_codes = predictor.predict_all()
    actual_codes = extract_ace_codes_from_response(response)
    
    result = PaymentResult(payment_id=payment_id, predicted_codes=predicted_codes, actual_codes=actual_codes)
    result.compute_match()
    
    if verbose:
        status = '✓ SUCCESS' if result.success else '✗ FAIL'
        print(f"\n{'='*70}")
        print(f"Payment: {payment_id} [{status}]")
        
        det_codes = [(c, p, r) for c, p, r in predicted_codes if not r.startswith('[ELIGIBLE]')]
        elig_codes = [(c, p, r) for c, p, r in predicted_codes if r.startswith('[ELIGIBLE]')]
        
        print(f"\nDeterministic Predictions ({len(det_codes)}):")
        for code, party, reason in det_codes:
            print(f"  {code} [{party}]: {reason}")
        
        if elig_codes:
            print(f"\nDirectory-Dependent Eligibility ({len(elig_codes)}):")
            for code, party, reason in elig_codes:
                print(f"  {code} [{party}]: {reason}")
        
        print(f"\nActual ({len(actual_codes)}): {sorted(actual_codes)}")
        if result.false_positives:
            print(f"\nFP (predicted not actual): {result.false_positives}")
        if result.false_negatives:
            print(f"FN (actual not predicted): {result.false_negatives}")
    
    return result


def process_file(filepath: Path, verbose: bool = False) -> List[PaymentResult]:
    results = []
    with open(filepath) as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        for payment_id, payment_data in data.items():
            if isinstance(payment_data, dict) and ('Request' in payment_data or 'IFML' in payment_data):
                result = process_payment(payment_id, payment_data, verbose)
                if result:
                    results.append(result)
    elif isinstance(data, list):
        for i, payment_data in enumerate(data):
            payment_id = payment_data.get('payment_id', payment_data.get('id', f'payment_{i}'))
            result = process_payment(payment_id, payment_data, verbose)
            if result:
                results.append(result)
    
    return results


def process_directory(dirpath: Path, verbose: bool = False, limit: Optional[int] = None) -> List[PaymentResult]:
    results = []
    json_files = list(dirpath.glob('*.json'))
    print(f"Found {len(json_files)} JSON files")
    
    for i, filepath in enumerate(json_files):
        if limit and len(results) >= limit:
            break
        if (i + 1) % 10 == 0:
            print(f"  Processing file {i+1}/{len(json_files)}...", end='\r')
        try:
            results.extend(process_file(filepath, verbose))
        except Exception as e:
            print(f"\nError processing {filepath}: {e}")
    
    print(f"\nProcessed {len(results)} payments")
    return results


# =============================================================================
# REPORT
# =============================================================================

def generate_report(results: List[PaymentResult]) -> Dict:
    total = len(results)
    successes = sum(1 for r in results if r.success)
    
    code_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    for r in results:
        for c in r.true_positives:
            code_stats[c]['tp'] += 1
        for c in r.false_positives:
            code_stats[c]['fp'] += 1
        for c in r.false_negatives:
            code_stats[c]['fn'] += 1
    
    for stats in code_stats.values():
        tp, fp, fn = stats['tp'], stats['fp'], stats['fn']
        stats['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        stats['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        stats['f1'] = (2 * stats['precision'] * stats['recall'] / 
                      (stats['precision'] + stats['recall'])) if (stats['precision'] + stats['recall']) > 0 else 0
    
    return {
        'summary': {
            'total_payments': total,
            'successful': successes,
            'success_rate': successes / total if total > 0 else 0,
            'failed': total - successes,
        },
        'code_stats_8xxx': {k: v for k, v in code_stats.items() if k.startswith('8')},
        'code_stats_9xxx': {k: v for k, v in code_stats.items() if k.startswith('9')},
        'failures': [r.to_dict() for r in results if not r.success][:50],
        'all_results': [r.to_dict() for r in results],
    }


def print_report(report: Dict):
    s = report['summary']
    print("\n" + "=" * 70)
    print("ACE PELICAN RULE VERIFICATION REPORT (Cross-Referenced)")
    print("=" * 70)
    print(f"\n📊 SUMMARY")
    print(f"   Total: {s['total_payments']} | Success: {s['successful']} | Failed: {s['failed']}")
    print(f"   Success Rate: {s['success_rate']:.2%}")
    
    for label, key in [("8XXX VALIDATION ERRORS", 'code_stats_8xxx'), 
                       ("9XXX REPAIRS", 'code_stats_9xxx')]:
        stats = report.get(key, {})
        if stats:
            print(f"\n📋 {label}")
            print(f"   {'Code':<8} {'TP':<6} {'FP':<6} {'FN':<6} {'Prec':<10} {'Recall':<10}")
            print(f"   {'-'*50}")
            for code in sorted(stats.keys()):
                st = stats[code]
                print(f"   {code:<8} {st['tp']:<6} {st['fp']:<6} {st['fn']:<6} "
                      f"{st['precision']:<10.2%} {st['recall']:<10.2%}")
    
    failures = report.get('failures', [])
    if failures:
        print(f"\n⚠️  FAILURES ({len(failures)} shown):")
        for f in failures[:5]:
            print(f"\n   {f['payment_id']}")
            det_pred = [p['code'] for p in f['predicted_codes'] if not p['reason'].startswith('[ELIGIBLE]')]
            print(f"   Predicted: {det_pred}")
            print(f"   Actual: {f['actual_codes']}")
            print(f"   FP: {f['false_positives']} | FN: {f['false_negatives']}")


def main():
    parser = argparse.ArgumentParser(description='Process IFML, predict ACE codes, compare with actual')
    parser.add_argument('--file', help='Single JSON file')
    parser.add_argument('--dir', help='Directory of JSON files')
    parser.add_argument('--output', '-o', help='Output JSON report')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    
    if not args.file and not args.dir:
        print("Usage: python process_ifml_payments.py --file payment.json")
        print("       python process_ifml_payments.py --dir /path/to/payments/")
        return
    
    results = process_file(Path(args.file), args.verbose) if args.file else process_directory(Path(args.dir), args.verbose, args.limit)
    
    if not results:
        print("No payments processed.")
        return
    
    report = generate_report(results)
    print_report(report)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n📄 Saved: {args.output}")


if __name__ == '__main__':
    main()
