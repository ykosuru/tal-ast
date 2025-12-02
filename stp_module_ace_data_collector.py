#!/usr/bin/env python3
"""
================================================================================
ACE DATA COLLECTOR v2.0
================================================================================

Collects IFML payment data and builds decision trees for ACE code prediction.

KEY CHANGE: Excludes spurious correlation features from decision tree analysis:
- Amount features (primary_amount, instructed_amount, etc.)
- Transaction IDs (transaction_uid - meaningless numeric)
- Message type (correlation, not causation)
- Source code, incoming format (operational, not validation-related)

These features may correlate with codes in the dataset but do NOT cause the codes
to fire. Including them creates misleading decision trees.

Usage:
    # Collect data from combined JSON files
    python ace_data_collector_v2.py collect -i /path/to/data -o dataset.csv
    
    # Build decision trees (excluding spurious features)
    python ace_data_collector_v2.py analyze -i dataset.csv -o ./trees --prefix 8
    
    # Build trees for 9XXX repair codes
    python ace_data_collector_v2.py analyze -i dataset.csv -o ./trees --prefix 9

Author: ACE Pelican Team
Version: 2.0 - Excludes spurious correlations
================================================================================
"""

import json
import csv
import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Tuple
from collections import defaultdict
from dataclasses import dataclass, field
import warnings

# Suppress sklearn warnings
warnings.filterwarnings('ignore')

# =============================================================================
# SPURIOUS FEATURES TO EXCLUDE FROM DECISION TREES
# =============================================================================

# These features correlate with codes but do NOT cause them
# Including them creates misleading trees (e.g., "amount > 275M -> invalid IBAN")
EXCLUDED_FEATURES = {
    # Amount features - payment value doesn't determine validation
    'primary_amount',
    'instructed_amount',
    'amount_mismatch',
    'has_instructed_amount',
    
    # Transaction identifiers - meaningless numeric correlation
    'transaction_uid',
    'transaction_id',
    
    # Message metadata - operational, not validation-related
    'message_type',
    'incoming_msg_type',
    'source_code',
    'incoming_format',
    
    # Derived ID features that are just length proxies
    # (keep specific validation features like bic_valid_format)
}

# Features to KEEP for decision trees (validation-relevant)
# These are the features that actually determine if a code fires
VALIDATION_FEATURES = {
    # BIC validation (8001, 8005, 8006)
    'has_bic', 'bic_valid_format', 'bic_valid_country', 'bic4_valid',
    'bic_length', 'bic_country',
    
    # IBAN validation (8004, 8894, 8898)
    'has_iban', 'iban_valid_format', 'iban_checksum_valid',
    'needs_iban', 'iban_country',
    
    # NCH/ABA validation (8895, 8026)
    'has_nch', 'nch_valid', 'fedaba_checksum_valid',
    'has_adr_bank_id', 'nch_validation_applicable',
    
    # Account validation (8892, 8896)
    'has_account', 'account_valid', 'account_type',
    
    # Country validation (8006, 8027)
    'has_country', 'country_valid', 'country',
    
    # Currency validation (8124)
    'has_currency', 'currency_valid',
    
    # Cross-party consistency (8022, 8023, 8026)
    'wirekey_bic_bnp_iban_match', 'bic_iban_country_match',
    
    # Party presence (context)
    'present', 'has_id', 'has_name',
    
    # Domestic/international (context for 8895, 8896)
    'is_domestic', 'is_international',
    
    # 9XXX repair detection
    'intm_has_multiple', 'intm_has_redundant_info', 'intm_count',
    'intm_entries_share_adr_bank_id', 'intm_entries_share_country',
    'intm_entries_share_bic_prefix',
    'account_has_dirty_chars', 'name_has_dirty_chars',
    'iban_needs_formatting', 'nch_needs_formatting',
    'has_multiple_ids', 'has_duplicate_info',
    'any_id_needs_cleaning', 'id_needs_cleaning',
    
    # Structural features (relevant for 9017, 9018)
    'bank_party_count', 'non_bank_party_count', 'party_count',
    'has_intermediary', 'address_line_count',
}

# =============================================================================
# CONSTANTS
# =============================================================================

VALID_COUNTRY_CODES = {
    'AD', 'AE', 'AF', 'AG', 'AI', 'AL', 'AM', 'AO', 'AQ', 'AR', 'AS', 'AT', 'AU', 'AW', 'AX', 'AZ',
    'BA', 'BB', 'BD', 'BE', 'BF', 'BG', 'BH', 'BI', 'BJ', 'BL', 'BM', 'BN', 'BO', 'BQ', 'BR', 'BS',
    'BT', 'BV', 'BW', 'BY', 'BZ', 'CA', 'CC', 'CD', 'CF', 'CG', 'CH', 'CI', 'CK', 'CL', 'CM', 'CN',
    'CO', 'CR', 'CU', 'CV', 'CW', 'CX', 'CY', 'CZ', 'DE', 'DJ', 'DK', 'DM', 'DO', 'DZ', 'EC', 'EE',
    'EG', 'EH', 'ER', 'ES', 'ET', 'FI', 'FJ', 'FK', 'FM', 'FO', 'FR', 'GA', 'GB', 'GD', 'GE', 'GF',
    'GG', 'GH', 'GI', 'GL', 'GM', 'GN', 'GP', 'GQ', 'GR', 'GS', 'GT', 'GU', 'GW', 'GY', 'HK', 'HM',
    'HN', 'HR', 'HT', 'HU', 'ID', 'IE', 'IL', 'IM', 'IN', 'IO', 'IQ', 'IR', 'IS', 'IT', 'JE', 'JM',
    'JO', 'JP', 'KE', 'KG', 'KH', 'KI', 'KM', 'KN', 'KP', 'KR', 'KW', 'KY', 'KZ', 'LA', 'LB', 'LC',
    'LI', 'LK', 'LR', 'LS', 'LT', 'LU', 'LV', 'LY', 'MA', 'MC', 'MD', 'ME', 'MF', 'MG', 'MH', 'MK',
    'ML', 'MM', 'MN', 'MO', 'MP', 'MQ', 'MR', 'MS', 'MT', 'MU', 'MV', 'MW', 'MX', 'MY', 'MZ', 'NA',
    'NC', 'NE', 'NF', 'NG', 'NI', 'NL', 'NO', 'NP', 'NR', 'NU', 'NZ', 'OM', 'PA', 'PE', 'PF', 'PG',
    'PH', 'PK', 'PL', 'PM', 'PN', 'PR', 'PS', 'PT', 'PW', 'PY', 'QA', 'RE', 'RO', 'RS', 'RU', 'RW',
    'SA', 'SB', 'SC', 'SD', 'SE', 'SG', 'SH', 'SI', 'SJ', 'SK', 'SL', 'SM', 'SN', 'SO', 'SR', 'SS',
    'ST', 'SV', 'SX', 'SY', 'SZ', 'TC', 'TD', 'TF', 'TG', 'TH', 'TJ', 'TK', 'TL', 'TM', 'TN', 'TO',
    'TR', 'TT', 'TV', 'TW', 'TZ', 'UA', 'UG', 'UM', 'US', 'UY', 'UZ', 'VA', 'VC', 'VE', 'VG', 'VI',
    'VN', 'VU', 'WF', 'WS', 'XK', 'YE', 'YT', 'ZA', 'ZM', 'ZW'
}

IBAN_LENGTHS = {
    'AL': 28, 'AD': 24, 'AT': 20, 'AZ': 28, 'BH': 22, 'BY': 28, 'BE': 16, 'BA': 20,
    'BR': 29, 'BG': 22, 'CR': 22, 'HR': 21, 'CY': 28, 'CZ': 24, 'DK': 18, 'DO': 28,
    'TL': 23, 'EE': 20, 'FO': 18, 'FI': 18, 'FR': 27, 'GE': 22, 'DE': 22, 'GI': 23,
    'GR': 27, 'GL': 18, 'GT': 28, 'HU': 28, 'IS': 26, 'IQ': 23, 'IE': 22, 'IL': 23,
    'IT': 27, 'JO': 30, 'KZ': 20, 'XK': 20, 'KW': 30, 'LV': 21, 'LB': 28, 'LI': 21,
    'LT': 20, 'LU': 20, 'MK': 19, 'MT': 31, 'MR': 27, 'MU': 30, 'MC': 27, 'MD': 24,
    'ME': 22, 'NL': 18, 'NO': 15, 'PK': 24, 'PS': 29, 'PL': 28, 'PT': 25, 'QA': 29,
    'RO': 24, 'SM': 27, 'SA': 24, 'RS': 22, 'SC': 31, 'SK': 24, 'SI': 19, 'ES': 24,
    'SE': 24, 'CH': 21, 'TN': 24, 'TR': 26, 'AE': 23, 'GB': 22, 'VA': 22, 'VG': 24,
}

IBAN_COUNTRIES = set(IBAN_LENGTHS.keys())

PARTY_PREFIXES = ['orig', 'send', 'dbt', 'cdt', 'intm', 'bnf', 'bnp', 'acwi', 'ordi']

PARTY_MAPPING = {
    'OriginatingPartyInf': 'orig', 'OriginatingPartyInfo': 'orig',
    'SendingBankInf': 'send', 'SendingBankInfo': 'send',
    'DebitPartyInf': 'dbt', 'DebitPartyInfo': 'dbt',
    'CreditPartyInf': 'cdt', 'CreditPartyInfo': 'cdt',
    'IntermediaryBankInf': 'intm', 'IntermediaryBankInfo': 'intm',
    'BeneficiaryBankInf': 'bnf', 'BeneficiaryBankInfo': 'bnf',
    'BeneficiaryPartyInf': 'bnp', 'BeneficiaryPartyInfo': 'bnp',
    'AccountWithInstitution': 'acwi',
    'OrderingInstitution': 'ordi',
}


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_bic(bic: str) -> Tuple[bool, bool]:
    """Returns (format_valid, country_valid)"""
    if not bic:
        return False, False
    bic = bic.upper().strip()
    if len(bic) not in (8, 11):
        return False, False
    pattern = r'^[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}([A-Z0-9]{3})?$'
    format_valid = bool(re.match(pattern, bic))
    country_code = bic[4:6] if len(bic) >= 6 else ''
    country_valid = country_code in VALID_COUNTRY_CODES
    return format_valid, country_valid


def validate_iban(iban: str) -> Tuple[bool, bool]:
    """Returns (format_valid, checksum_valid)"""
    if not iban:
        return False, False
    iban = iban.upper().replace(' ', '').replace('-', '')
    if len(iban) < 5:
        return False, False
    country_code = iban[:2]
    if not country_code.isalpha():
        return False, False
    expected_length = IBAN_LENGTHS.get(country_code)
    if expected_length:
        format_valid = len(iban) == expected_length
    else:
        format_valid = 15 <= len(iban) <= 34
    try:
        rearranged = iban[4:] + iban[:4]
        numeric = ''
        for char in rearranged:
            if char.isdigit():
                numeric += char
            elif char.isalpha():
                numeric += str(ord(char) - ord('A') + 10)
            else:
                return format_valid, False
        checksum_valid = int(numeric) % 97 == 1
    except (ValueError, OverflowError):
        checksum_valid = False
    return format_valid, checksum_valid


def validate_fedaba(aba: str) -> Tuple[bool, bool]:
    """Returns (format_valid, checksum_valid)"""
    if not aba:
        return False, False
    aba = aba.strip()
    if len(aba) != 9 or not aba.isdigit():
        return False, False
    try:
        weights = [3, 7, 1, 3, 7, 1, 3, 7, 1]
        total = sum(int(d) * w for d, w in zip(aba, weights))
        checksum_valid = (total % 10 == 0)
    except (ValueError, TypeError):
        checksum_valid = False
    return True, checksum_valid


# =============================================================================
# FEATURE EXTRACTOR
# =============================================================================

class FeatureExtractor:
    """Extract features from IFML data for ML analysis."""
    
    def __init__(self, debug: bool = False):
        self.features = {}
        self.debug = debug
    
    def extract(self, data: Dict) -> Dict:
        """Extract all features from IFML request."""
        self.features = {}
        
        message = self._find_message(data)
        if not message:
            return self.features
        
        basic_payment = message.get('BasicPayment', {})
        if not basic_payment:
            return self.features
        
        self._parse_basic_info(basic_payment)
        self._parse_amounts(basic_payment)
        self._parse_parties(basic_payment)
        self._parse_intermediary_array(basic_payment)
        self._compute_derived_features()
        
        return self.features
    
    def _find_message(self, data: Dict) -> Optional[Dict]:
        """Navigate to Message node."""
        paths = [
            ['Request', 'IFML', 'File', 'Message'],
            ['IFML', 'File', 'Message'],
            ['File', 'Message'],
            ['Message'],
        ]
        for path in paths:
            current = data
            for key in path:
                if isinstance(current, dict):
                    current = current.get(key)
                else:
                    current = None
                    break
            if current and isinstance(current, dict):
                return current
        
        # Try with transaction ID wrapper
        if isinstance(data, dict) and len(data) == 1:
            key = list(data.keys())[0]
            nested = data[key]
            for path in paths:
                current = nested
                for key in path:
                    if isinstance(current, dict):
                        current = current.get(key)
                    else:
                        current = None
                        break
                if current and isinstance(current, dict):
                    return current
        return None
    
    def _parse_basic_info(self, basic_payment: Dict):
        """Parse basic transaction info."""
        self.features['transaction_id'] = basic_payment.get('TransactionID', '')
        self.features['transaction_uid'] = basic_payment.get('TransactionUID', '')
        self.features['source_code'] = basic_payment.get('SourceCode', '')
        self.features['incoming_msg_type'] = basic_payment.get('IncomingMsgType', '')
        
        # Convert message type to numeric for tree
        msg_type = basic_payment.get('IncomingMsgType', '')
        try:
            self.features['message_type'] = int(msg_type) if msg_type else 0
        except ValueError:
            self.features['message_type'] = 0
    
    def _parse_amounts(self, basic_payment: Dict):
        """Parse monetary amounts."""
        amounts = basic_payment.get('MonetaryAmount', [])
        if isinstance(amounts, dict):
            amounts = [amounts]
        
        self.features['primary_amount'] = 0.0
        self.features['primary_currency'] = ''
        self.features['has_currency'] = False
        self.features['currency_valid'] = False
        
        for amt in amounts:
            amt_type = amt.get('@Type') or amt.get('Type') or 'Unknown'
            currency = amt.get('@Currency') or amt.get('Currency') or ''
            amount_str = str(amt.get('Amount', '0'))
            
            if amt_type == 'Amount':
                try:
                    self.features['primary_amount'] = float(amount_str.replace(',', ''))
                except ValueError:
                    pass
                self.features['primary_currency'] = currency
                self.features['has_currency'] = bool(currency)
                # Simple currency validation (3 uppercase letters)
                self.features['currency_valid'] = bool(re.match(r'^[A-Z]{3}$', currency))
    
    def _parse_parties(self, basic_payment: Dict):
        """Parse all party information."""
        party_info = basic_payment.get('PartyInfo') or basic_payment.get('PartyInf') or {}
        
        if not isinstance(party_info, dict):
            return
        
        # Initialize party counts
        bank_count = 0
        non_bank_count = 0
        
        processed = set()
        for ifml_name, prefix in PARTY_MAPPING.items():
            party_data = party_info.get(ifml_name)
            if party_data:
                if isinstance(party_data, list):
                    party_data = party_data[0] if party_data else None
                if party_data and isinstance(party_data, dict):
                    if prefix not in processed:
                        self._extract_party(party_data, prefix)
                        processed.add(prefix)
                        
                        # Count party types
                        if prefix in ('send', 'intm', 'bnf'):
                            bank_count += 1
                        else:
                            non_bank_count += 1
        
        self.features['bank_party_count'] = bank_count
        self.features['non_bank_party_count'] = non_bank_count
        self.features['party_count'] = bank_count + non_bank_count
    
    def _extract_party(self, party_data: Dict, prefix: str):
        """Extract features from a single party."""
        self.features[f'{prefix}_present'] = True
        
        basic = (
            party_data.get('BasicPartyInfo') or
            party_data.get('BasicPartyInf') or
            party_data.get('BasicIDInfo') or
            party_data.get('AccountPartyInfo') or
            party_data.get('AccountPartyInf') or
            party_data.get('BasicPartyBankInfo') or
            party_data.get('BasicPartyBankInf') or
            party_data
        )
        
        if isinstance(basic, list):
            basic = basic[0] if basic else {}
        
        if not isinstance(basic, dict):
            return
        
        # Parse ID field
        id_field = basic.get('ID')
        bic, iban = None, None
        id_value = ''
        
        if id_field:
            self.features[f'{prefix}_has_id'] = True
            if isinstance(id_field, dict):
                id_type = (id_field.get('Type') or id_field.get('@Type') or '').upper()
                id_value = id_field.get('text') or id_field.get('#text') or ''
                
                if id_type in ('S', 'SWIFT', 'BIC'):
                    bic = id_value
                elif id_type == 'IBAN':
                    iban = id_value
                elif self._looks_like_bic(id_value):
                    bic = id_value
                elif self._looks_like_iban(id_value):
                    iban = id_value
            else:
                id_value = str(id_field)
                if self._looks_like_iban(id_value):
                    iban = id_value
                elif self._looks_like_bic(id_value):
                    bic = id_value
            
            self.features[f'{prefix}_id_length'] = len(id_value)
            
            # Check if ID needs cleaning (for 9019)
            if id_value:
                cleaned = re.sub(r'[^A-Za-z0-9]', '', id_value)
                self.features[f'{prefix}_id_needs_cleaning'] = (cleaned != id_value)
        
        # BIC features
        if bic:
            self.features[f'{prefix}_has_bic'] = True
            self.features[f'{prefix}_bic_length'] = len(bic)
            fmt_valid, ctry_valid = validate_bic(bic)
            self.features[f'{prefix}_bic_valid_format'] = fmt_valid
            self.features[f'{prefix}_bic_valid_country'] = ctry_valid
            self.features[f'{prefix}_bic4_valid'] = bic[:4].isalpha() if len(bic) >= 4 else False
            if len(bic) >= 6:
                self.features[f'{prefix}_bic_country'] = bic[4:6].upper()
        
        # IBAN features
        if iban:
            self.features[f'{prefix}_has_iban'] = True
            fmt_valid, cksum_valid = validate_iban(iban)
            self.features[f'{prefix}_iban_valid_format'] = fmt_valid
            self.features[f'{prefix}_iban_checksum_valid'] = cksum_valid
            if len(iban) >= 2:
                self.features[f'{prefix}_iban_country'] = iban[:2].upper()
            
            # Check if IBAN needs formatting (for 9006)
            cleaned = iban.upper().replace(' ', '').replace('-', '')
            self.features[f'{prefix}_iban_needs_formatting'] = (cleaned != iban)
        
        # Account info
        acct_info = basic.get('AcctIDInfo') or basic.get('AcctIDInf')
        if acct_info and isinstance(acct_info, dict):
            self.features[f'{prefix}_has_account'] = True
            acct_id = acct_info.get('ID', {})
            if isinstance(acct_id, dict):
                acct_type = acct_id.get('@Type') or acct_id.get('Type')
                acct_value = acct_id.get('#text') or acct_id.get('text') or ''
                self.features[f'{prefix}_account_type'] = acct_type
                self.features[f'{prefix}_account_length'] = len(acct_value)
                
                if acct_type == 'IBAN' and acct_value:
                    self.features[f'{prefix}_has_iban'] = True
                    fmt_valid, cksum_valid = validate_iban(acct_value)
                    self.features[f'{prefix}_iban_valid_format'] = fmt_valid
                    self.features[f'{prefix}_iban_checksum_valid'] = cksum_valid
                
                # Check for dirty chars (for 9002)
                if acct_value:
                    self.features[f'{prefix}_account_has_dirty_chars'] = bool(
                        re.search(r'[^a-zA-Z0-9]', acct_value)
                    )
        
        # AdrBankID (NCH)
        adr_bank_id = basic.get('AdrBankID')
        if adr_bank_id:
            self.features[f'{prefix}_has_adr_bank_id'] = True
            if isinstance(adr_bank_id, dict):
                adr_type = adr_bank_id.get('@Type') or adr_bank_id.get('Type')
                adr_value = adr_bank_id.get('#text') or adr_bank_id.get('text') or ''
            else:
                adr_type = None
                adr_value = str(adr_bank_id)
            
            # FEDABA validation
            if adr_type in ['FEDABA', 'FW', 'FEDWIRE'] or (adr_value and len(adr_value) == 9 and adr_value.isdigit()):
                self.features[f'{prefix}_has_nch'] = True
                self.features[f'{prefix}_nch_type'] = 'FEDABA'
                fmt_valid, cksum_valid = validate_fedaba(adr_value)
                self.features[f'{prefix}_nch_valid'] = fmt_valid
                self.features[f'{prefix}_fedaba_checksum_valid'] = cksum_valid
                self.features[f'{prefix}_nch_validation_applicable'] = True
        
        # Country
        country = basic.get('Country') or party_data.get('Country')
        if country:
            self.features[f'{prefix}_has_country'] = True
            self.features[f'{prefix}_country'] = country
            self.features[f'{prefix}_country_valid'] = country.upper() in VALID_COUNTRY_CODES
        
        # Name
        name_field = basic.get('Name')
        if name_field:
            self.features[f'{prefix}_has_name'] = True
            if isinstance(name_field, dict):
                name_value = name_field.get('#text') or name_field.get('text') or ''
            else:
                name_value = str(name_field)
            
            if name_value:
                self.features[f'{prefix}_name_has_dirty_chars'] = bool(
                    re.search(r'[@#$%^*=+\[\]{}|\\<>]', name_value)
                )
        
        # Address
        address_info = basic.get('AddressInfo') or basic.get('AddressInf') or party_data.get('AddressInfo')
        if address_info:
            if isinstance(address_info, list):
                self.features[f'{prefix}_address_line_count'] = len(address_info)
                # Extract country from address
                self._extract_address_country(address_info, prefix)
        
        # IBAN requirement
        party_country = (
            self.features.get(f'{prefix}_bic_country') or
            self.features.get(f'{prefix}_iban_country') or
            self.features.get(f'{prefix}_country') or
            self.features.get(f'{prefix}_address_country')
        )
        if party_country and party_country.upper() in IBAN_COUNTRIES:
            has_iban = self.features.get(f'{prefix}_has_iban', False)
            has_account = self.features.get(f'{prefix}_has_account', False)
            if not has_iban and not has_account:
                self.features[f'{prefix}_needs_iban'] = True
    
    def _extract_address_country(self, address_info: list, prefix: str):
        """Extract country from address lines."""
        if not address_info:
            return
        
        # Check last address line for country code
        for item in reversed(address_info):
            if isinstance(item, dict):
                text = item.get('text') or item.get('Text') or ''
            elif isinstance(item, str):
                text = item
            else:
                continue
            
            text = text.upper().strip()
            words = text.split()
            if words:
                last_word = ''.join(c for c in words[-1] if c.isalpha())
                if len(last_word) == 2 and last_word in VALID_COUNTRY_CODES:
                    self.features[f'{prefix}_address_country'] = last_word
                    self.features[f'{prefix}_is_domestic'] = (last_word == 'US')
                    self.features[f'{prefix}_is_international'] = (last_word != 'US')
                    return
    
    def _parse_intermediary_array(self, basic_payment: Dict):
        """Parse intermediary array for redundancy detection (9018)."""
        party_info = basic_payment.get('PartyInfo') or basic_payment.get('PartyInf') or {}
        intm_data = party_info.get('IntermediaryBankInf') or party_info.get('IntermediaryBankInfo')
        
        if not intm_data:
            self.features['intm_count'] = 0
            self.features['intm_has_multiple'] = False
            return
        
        if isinstance(intm_data, dict):
            intm_list = [intm_data]
        elif isinstance(intm_data, list):
            intm_list = intm_data
        else:
            self.features['intm_count'] = 0
            self.features['intm_has_multiple'] = False
            return
        
        self.features['intm_count'] = len(intm_list)
        self.features['intm_has_multiple'] = len(intm_list) > 1
        self.features['has_intermediary'] = len(intm_list) > 0
        
        if len(intm_list) < 2:
            return
        
        # Check for redundancy
        adr_bank_ids = []
        countries = []
        bic_prefixes = []
        
        for entry in intm_list:
            if not isinstance(entry, dict):
                continue
            
            basic = (
                entry.get('BasicPartyInf') or
                entry.get('BasicPartyInfo') or
                entry.get('BasicPartyBankInf') or
                entry.get('BasicPartyBankInfo') or
                entry
            )
            
            if not isinstance(basic, dict):
                continue
            
            # AdrBankID
            adr_id = basic.get('AdrBankID')
            if adr_id:
                if isinstance(adr_id, dict):
                    adr_id = adr_id.get('text') or adr_id.get('#text') or str(adr_id)
                adr_bank_ids.append(str(adr_id))
            
            # Country
            country = basic.get('Country')
            if country:
                countries.append(country.upper())
            
            # BIC prefix
            id_info = basic.get('ID', {})
            if isinstance(id_info, dict):
                bic = id_info.get('text') or id_info.get('#text') or ''
                if bic and len(bic) >= 6 and bic[:4].isalpha():
                    bic_prefixes.append(bic[:6].upper())
        
        # Check for shared values (redundancy signals)
        if len(adr_bank_ids) >= 2:
            self.features['intm_entries_share_adr_bank_id'] = len(set(adr_bank_ids)) < len(adr_bank_ids)
        
        if len(countries) >= 2:
            self.features['intm_entries_share_country'] = len(set(countries)) < len(countries)
        
        if len(bic_prefixes) >= 2:
            self.features['intm_entries_share_bic_prefix'] = len(set(bic_prefixes)) < len(bic_prefixes)
        
        # Combined redundancy signal
        redundancy_signals = sum([
            self.features.get('intm_entries_share_adr_bank_id', False),
            self.features.get('intm_entries_share_country', False),
            self.features.get('intm_entries_share_bic_prefix', False),
        ])
        self.features['intm_has_redundant_info'] = redundancy_signals >= 2
    
    def _compute_derived_features(self):
        """Compute derived/aggregate features."""
        # Any ID needs cleaning (for 9019)
        any_cleaning = False
        for prefix in PARTY_PREFIXES:
            if self.features.get(f'{prefix}_id_needs_cleaning'):
                any_cleaning = True
                break
        self.features['any_id_needs_cleaning'] = any_cleaning
        
        # Has duplicate BIC4 (for tree analysis)
        bic4s = []
        for prefix in PARTY_PREFIXES:
            bic = self.features.get(f'{prefix}_bic_country')
            if bic:
                bic4s.append(bic)
        self.features['has_duplicate_bic4'] = len(bic4s) != len(set(bic4s)) if bic4s else False
        
        # Has duplicate BIC
        bics = []
        for prefix in PARTY_PREFIXES:
            if self.features.get(f'{prefix}_has_bic'):
                bic_len = self.features.get(f'{prefix}_bic_length', 0)
                bics.append(bic_len)
        self.features['has_duplicate_bic'] = len(bics) != len(set(bics)) if len(bics) > 1 else False
        
        # All same country
        countries = []
        for prefix in PARTY_PREFIXES:
            country = (
                self.features.get(f'{prefix}_bic_country') or
                self.features.get(f'{prefix}_country') or
                self.features.get(f'{prefix}_address_country')
            )
            if country:
                countries.append(country.upper())
        self.features['all_same_country'] = len(set(countries)) <= 1 if countries else False
        
        # Is SEPA payment (all EU countries)
        sepa_countries = {'AT', 'BE', 'BG', 'HR', 'CY', 'CZ', 'DK', 'EE', 'FI', 'FR', 'DE', 'GR',
                         'HU', 'IE', 'IT', 'LV', 'LT', 'LU', 'MT', 'NL', 'PL', 'PT', 'RO', 'SK',
                         'SI', 'ES', 'SE', 'GB', 'IS', 'LI', 'NO', 'CH'}
        self.features['is_sepa_payment'] = all(c in sepa_countries for c in countries) if countries else False
    
    def _looks_like_bic(self, s: str) -> bool:
        if not s or len(s) not in (8, 11):
            return False
        return bool(re.match(r'^[A-Z]{4}[A-Z]{2}[A-Z0-9]{2,5}$', s.upper()))
    
    def _looks_like_iban(self, s: str) -> bool:
        if not s:
            return False
        s = s.upper().replace(' ', '').replace('-', '')
        if len(s) < 15 or len(s) > 34:
            return False
        return s[:2].isalpha() and s[2:4].isdigit()


# =============================================================================
# RESPONSE PARSER
# =============================================================================

def extract_codes_from_response(response: Dict) -> Set[str]:
    """Extract all codes from ACE response."""
    codes = set()
    
    audit = None
    paths = [
        ['AuditTrail'],
        ['IFML', 'File', 'Message', 'AuditTrail'],
        ['Response', 'IFML', 'File', 'Message', 'AuditTrail'],
    ]
    
    for path in paths:
        current = response
        for key in path:
            if isinstance(current, dict):
                current = current.get(key)
            else:
                current = None
                break
        if current:
            audit = current
            break
    
    if not audit:
        return codes
    
    msg_status = audit.get('MsgStatus', [])
    if isinstance(msg_status, dict):
        msg_status = [msg_status]
    
    for item in msg_status:
        code = item.get('Code', '')
        if code:
            codes.add(code)
    
    return codes


# =============================================================================
# DATA COLLECTOR
# =============================================================================

class DataCollector:
    """Collect features and codes from IFML files."""
    
    def __init__(self, debug: bool = False):
        self.extractor = FeatureExtractor(debug=debug)
        self.debug = debug
    
    def process_file(self, filepath: Path) -> List[Dict]:
        """Process a single file and return list of records."""
        records = []
        
        try:
            with open(filepath) as f:
                data = json.load(f)
        except Exception as e:
            if self.debug:
                print(f"Error reading {filepath}: {e}")
            return records
        
        if isinstance(data, dict):
            if 'Request' in data and 'Response' in data:
                record = self._process_payment(data, data.get('Response', {}))
                if record:
                    records.append(record)
            else:
                for key, val in data.items():
                    if isinstance(val, dict) and 'Request' in val and 'Response' in val:
                        record = self._process_payment(val, val.get('Response', {}))
                        if record:
                            records.append(record)
        
        return records
    
    def _process_payment(self, request: Dict, response: Dict) -> Optional[Dict]:
        """Process a single payment."""
        features = self.extractor.extract(request)
        codes = extract_codes_from_response(response)
        
        if not features:
            return None
        
        # Add codes as separate columns
        for code in codes:
            features[f'code_{code}'] = 1
        
        return features
    
    def collect_directory(self, dirpath: Path) -> List[Dict]:
        """Collect data from all JSON files in directory."""
        records = []
        
        json_files = list(dirpath.glob('*.json'))
        total = len(json_files)
        
        for i, filepath in enumerate(json_files):
            if self.debug and (i + 1) % 100 == 0:
                print(f"Processing {i+1}/{total}...")
            
            file_records = self.process_file(filepath)
            records.extend(file_records)
        
        return records


# =============================================================================
# DECISION TREE BUILDER
# =============================================================================

def filter_features_for_tree(df, exclude_spurious: bool = True):
    """
    Filter DataFrame columns to exclude spurious features.
    Returns feature columns suitable for decision tree analysis.
    """
    feature_cols = []
    
    for col in df.columns:
        # Skip target columns (code_*)
        if col.startswith('code_'):
            continue
        
        # Skip excluded features if requested
        if exclude_spurious:
            skip = False
            for excluded in EXCLUDED_FEATURES:
                if excluded in col.lower():
                    skip = True
                    break
            if skip:
                continue
        
        # Keep boolean and numeric columns
        if df[col].dtype in ['bool', 'int64', 'float64']:
            feature_cols.append(col)
        elif df[col].dtype == 'object':
            # Skip string columns (they need encoding)
            continue
    
    return feature_cols


def build_decision_tree(df, target_code: str, feature_cols: List[str], 
                        max_depth: int = 6, min_samples: int = 50) -> Optional[str]:
    """Build a decision tree for a specific code."""
    try:
        from sklearn.tree import DecisionTreeClassifier, export_text
    except ImportError:
        print("Error: sklearn not installed. Run: pip install scikit-learn")
        return None
    
    # Create target column
    target_col = f'code_{target_code}'
    if target_col not in df.columns:
        return None
    
    y = df[target_col].fillna(0).astype(int)
    
    # Check minimum samples
    positive_count = y.sum()
    if positive_count < min_samples:
        return None
    
    # Prepare features
    X = df[feature_cols].fillna(0)
    
    # Convert boolean columns to int
    for col in X.columns:
        if X[col].dtype == 'bool':
            X[col] = X[col].astype(int)
    
    # Build tree
    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=max(10, min_samples // 10),
        min_samples_split=max(20, min_samples // 5),
        class_weight='balanced'
    )
    
    clf.fit(X, y)
    
    # Export tree
    tree_text = export_text(clf, feature_names=list(X.columns))
    
    # Generate suggested rules
    rules = extract_rules_from_tree(clf, list(X.columns))
    
    return tree_text, rules, positive_count


def extract_rules_from_tree(clf, feature_names: List[str]) -> Dict:
    """Extract precondition rules from decision tree."""
    tree = clf.tree_
    
    # Find the path to class 1 (code fires)
    require_true = []
    require_false = []
    full_conditions = []
    
    # Simple heuristic: look at feature importances
    importances = clf.feature_importances_
    important_features = [
        (feature_names[i], importances[i])
        for i in range(len(feature_names))
        if importances[i] > 0.05
    ]
    important_features.sort(key=lambda x: -x[1])
    
    for feat, imp in important_features[:5]:
        # Determine if this is a require_true or require_false
        if 'valid' in feat.lower():
            require_false.append(feat.replace('_valid_format', '').replace('_valid_country', '').replace('_valid', ''))
        elif 'has_' in feat.lower():
            require_true.append(feat)
        
        full_conditions.append(f"{feat} (importance: {imp:.3f})")
    
    return {
        'require_true': list(set(require_true)),
        'require_false': list(set(require_false)),
        'important_features': full_conditions,
        'confidence': float(clf.score(clf.tree_.feature, clf.tree_.value.argmax(axis=2).flatten()[:len(clf.tree_.feature)])) if hasattr(clf, 'score') else 0.0
    }


def analyze_codes(csv_path: str, output_dir: str, prefix: str = '8', 
                  min_samples: int = 50, max_depth: int = 6):
    """Analyze codes and build decision trees."""
    try:
        import pandas as pd
    except ImportError:
        print("Error: pandas not installed")
        return
    
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} records")
    
    # Get feature columns (excluding spurious)
    feature_cols = filter_features_for_tree(df, exclude_spurious=True)
    print(f"Using {len(feature_cols)} features (spurious features excluded)")
    print(f"Excluded: {EXCLUDED_FEATURES}")
    
    # Find code columns
    code_cols = [c for c in df.columns if c.startswith(f'code_{prefix}')]
    print(f"Found {len(code_cols)} codes starting with {prefix}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Build trees
    results = []
    for code_col in sorted(code_cols):
        code = code_col.replace('code_', '')
        
        result = build_decision_tree(df, code, feature_cols, max_depth, min_samples)
        if result:
            tree_text, rules, count = result
            
            # Save tree
            tree_file = output_path / f"tree_{code}.txt"
            with open(tree_file, 'w') as f:
                f.write(f"Decision Tree for {code}\n")
                f.write("=" * 50 + "\n")
                f.write(f"Samples: {count}\n")
                f.write(f"Features used: {len(feature_cols)} (spurious excluded)\n\n")
                f.write(tree_text)
                f.write("\n\nSuggested Precondition Rules:\n")
                f.write("-" * 50 + "\n")
                f.write(json.dumps(rules, indent=2))
            
            print(f"  {code}: {count} samples -> {tree_file}")
            results.append({'code': code, 'samples': count, 'rules': rules})
        else:
            print(f"  {code}: skipped (< {min_samples} samples)")
    
    # Save summary
    summary_file = output_path / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            'total_records': len(df),
            'features_used': len(feature_cols),
            'excluded_features': list(EXCLUDED_FEATURES),
            'codes_analyzed': results
        }, f, indent=2)
    
    print(f"\nSummary saved to {summary_file}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='ACE Data Collector v2 - Excludes spurious features')
    subparsers = parser.add_subparsers(dest='command')
    
    # Collect command
    collect_parser = subparsers.add_parser('collect', help='Collect data from IFML files')
    collect_parser.add_argument('-i', '--input', required=True, help='Input directory')
    collect_parser.add_argument('-o', '--output', required=True, help='Output CSV file')
    collect_parser.add_argument('--debug', action='store_true')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Build decision trees')
    analyze_parser.add_argument('-i', '--input', required=True, help='Input CSV file')
    analyze_parser.add_argument('-o', '--output', required=True, help='Output directory')
    analyze_parser.add_argument('--prefix', default='8', help='Code prefix (8 or 9)')
    analyze_parser.add_argument('--min-samples', type=int, default=50)
    analyze_parser.add_argument('--max-depth', type=int, default=6)
    
    args = parser.parse_args()
    
    if args.command == 'collect':
        collector = DataCollector(debug=args.debug)
        records = collector.collect_directory(Path(args.input))
        
        if records:
            try:
                import pandas as pd
                df = pd.DataFrame(records)
                df.to_csv(args.output, index=False)
                print(f"Saved {len(records)} records to {args.output}")
            except ImportError:
                # Fallback to csv module
                if records:
                    keys = set()
                    for r in records:
                        keys.update(r.keys())
                    keys = sorted(keys)
                    
                    with open(args.output, 'w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=keys)
                        writer.writeheader()
                        writer.writerows(records)
                    print(f"Saved {len(records)} records to {args.output}")
    
    elif args.command == 'analyze':
        analyze_codes(args.input, args.output, args.prefix, 
                     args.min_samples, args.max_depth)
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
