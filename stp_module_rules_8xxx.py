#!/usr/bin/env python3
"""
================================================================================
ACE 8XXX CODE VALIDATOR
================================================================================

Validates rule-based predictions against actual ACE response codes.

This tool:
1. Parses IFML payments using ifml_parser_v3.py
2. Applies deterministic validation rules to predict 8XXX codes
3. Compares predictions with actual ACE response codes
4. Reports accuracy metrics (precision, recall, F1)

PREDICTABLE CODES (deterministic validation):
- 8001: Invalid BIC (format or country code)
- 8004: IBAN cannot be derived (needs IBAN but missing)
- 8005: Invalid BIC4 (first 4 chars)
- 8006: Invalid country code
- 8007: Amount decimal places exceeded
- 8022: IBAN/BIC country mismatch
- 8124: Invalid currency code
- 8894: Invalid IBAN (format or checksum)
- 8895: Invalid NCH/FEDABA (checksum)
- 8898: IBAN checksum failed

UNPREDICTABLE CODES (require directory/config):
- 8003, 8034-8036, 8464-8472, 8851-8853, 8905, 8906

Usage:
    # Validate single payment
    python ace_8xxx_validator.py single -r request.json -s response.json
    
    # Validate directory of payments
    python ace_8xxx_validator.py validate -i /path/to/data -o report.json
    
    # Show accuracy summary
    python ace_8xxx_validator.py accuracy -i /path/to/data

Author: ACE Pelican Team
Version: 1.0 (December 2025)
================================================================================
"""

import json
import argparse
import re
import sys
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set, Tuple, Any
from collections import defaultdict
import csv


# =============================================================================
# CONSTANTS
# =============================================================================

# Party suffixes in ACE responses
PARTY_SUFFIXES = ['ORGPTY', 'SNDBNK', 'DBTPTY', 'CDTPTY', 'INTBNK', 'BNFBNK', 'BNPPTY']

# Map ACE party suffix to parser prefix
SUFFIX_TO_PREFIX = {
    'ORGPTY': 'orig',
    'SNDBNK': 'send',
    'DBTPTY': 'dbt',
    'CDTPTY': 'cdt',
    'INTBNK': 'intm',
    'BNFBNK': 'bnf',
    'BNPPTY': 'bnp',
    'ACWI': 'acwi',
    'ORDI': 'ordi',
}

# Codes that are deterministically predictable
PREDICTABLE_CODES = {
    # 8XXX Validation Errors
    '8001',  # Invalid BIC
    '8004',  # IBAN cannot be derived
    '8005',  # Invalid BIC4
    '8006',  # Invalid country code
    '8007',  # Amount decimal places
    '8022',  # IBAN/BIC inconsistency
    '8124',  # Invalid currency
    '8852',  # Attribute length violation
    '8894',  # Invalid IBAN
    '8895',  # Invalid NCH
    '8898',  # IBAN checksum failed
    # 9XXX Repair Codes (structurally detectable)
    '9018',  # Duplicate party removed (multiple redundant intermediaries)
    '9019',  # Party identifier cleaned (non-alphanumeric chars)
}

# Codes that require directory lookup - can predict eligibility but not certainty
DIRECTORY_DEPENDENT = {
    '8003',  # File name derivation
    '8034',  # Forced debit
    '8035',  # FCDA validation
    '8036',  # FCDA name match
    '8464',  # Target channel
    '8465',  # Product code
    '8472',  # Fee code
    '8851',  # Field size (schema)
    '8853',  # Number format (schema)
    '8905',  # Hash mismatch
    '8906',  # Wrong flow
}

# Global codes (not party-specific)
GLOBAL_CODES = {'8007', '8022', '8023', '8024', '8025', '8027', '8028', '8029', '8033', '8124'}

# Bank party types - don't need IBANs
BANK_SUFFIXES = {'SNDBNK', 'INTBNK', 'BNFBNK'}

# Party account types - may need IBANs
PARTY_SUFFIXES_ACCOUNT = {'ORGPTY', 'DBTPTY', 'CDTPTY', 'BNPPTY'}

# Codes that only apply to party accounts, not banks
PARTY_ONLY_CODES = {'8004', '8030'}

# Valid ISO 3166-1 alpha-2 country codes
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

# Valid ISO 4217 currency codes (common ones)
VALID_CURRENCY_CODES = {
    'AED', 'AFN', 'ALL', 'AMD', 'ANG', 'AOA', 'ARS', 'AUD', 'AWG', 'AZN', 'BAM', 'BBD', 'BDT', 'BGN',
    'BHD', 'BIF', 'BMD', 'BND', 'BOB', 'BRL', 'BSD', 'BTN', 'BWP', 'BYN', 'BZD', 'CAD', 'CDF', 'CHF',
    'CLP', 'CNY', 'COP', 'CRC', 'CUP', 'CVE', 'CZK', 'DJF', 'DKK', 'DOP', 'DZD', 'EGP', 'ERN', 'ETB',
    'EUR', 'FJD', 'FKP', 'GBP', 'GEL', 'GHS', 'GIP', 'GMD', 'GNF', 'GTQ', 'GYD', 'HKD', 'HNL', 'HRK',
    'HTG', 'HUF', 'IDR', 'ILS', 'INR', 'IQD', 'IRR', 'ISK', 'JMD', 'JOD', 'JPY', 'KES', 'KGS', 'KHR',
    'KMF', 'KPW', 'KRW', 'KWD', 'KYD', 'KZT', 'LAK', 'LBP', 'LKR', 'LRD', 'LSL', 'LYD', 'MAD', 'MDL',
    'MGA', 'MKD', 'MMK', 'MNT', 'MOP', 'MRU', 'MUR', 'MVR', 'MWK', 'MXN', 'MYR', 'MZN', 'NAD', 'NGN',
    'NIO', 'NOK', 'NPR', 'NZD', 'OMR', 'PAB', 'PEN', 'PGK', 'PHP', 'PKR', 'PLN', 'PYG', 'QAR', 'RON',
    'RSD', 'RUB', 'RWF', 'SAR', 'SBD', 'SCR', 'SDG', 'SEK', 'SGD', 'SHP', 'SLL', 'SOS', 'SRD', 'SSP',
    'STN', 'SVC', 'SYP', 'SZL', 'THB', 'TJS', 'TMT', 'TND', 'TOP', 'TRY', 'TTD', 'TWD', 'TZS', 'UAH',
    'UGX', 'USD', 'UYU', 'UZS', 'VES', 'VND', 'VUV', 'WST', 'XAF', 'XCD', 'XOF', 'XPF', 'YER', 'ZAR',
    'ZMW', 'ZWL'
}

# IBAN lengths by country
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
    'UA': 29,
}

# IBAN-requiring countries
IBAN_COUNTRIES = {
    'DE', 'FR', 'GB', 'ES', 'IT', 'NL', 'BE', 'AT', 'CH', 'SE', 'NO', 'DK', 'FI',
    'PL', 'PT', 'IE', 'GR', 'CZ', 'HU', 'RO', 'SK', 'LU', 'HR', 'SI', 'BG', 'LT',
    'LV', 'EE', 'CY', 'MT', 'AE', 'SA', 'QA', 'KW', 'BH', 'IL', 'TR', 'EG', 'MA'
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ValidationResult:
    """Result for a single code check."""
    code: str
    party: Optional[str]
    fires: bool
    reason: str
    features_checked: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def code_with_party(self) -> str:
        if self.party:
            return f"{self.code}_{self.party}"
        return self.code


@dataclass
class PaymentResult:
    """Result of validating a single payment."""
    payment_id: str
    predicted_codes: Set[str]
    actual_codes: Set[str]
    correct: Set[str]
    missed: Set[str]  # False negatives
    false_positives: Set[str]
    details: List[ValidationResult]
    excluded_codes: Set[str]  # Codes excluded from comparison (directory-dependent)
    
    @property
    def passed(self) -> bool:
        return len(self.missed) == 0 and len(self.false_positives) == 0


@dataclass
class AccuracyReport:
    """Accuracy metrics across multiple payments."""
    total_payments: int
    payments_correct: int
    total_predictions: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    code_breakdown: Dict[str, Dict[str, int]]


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_bic(bic: str) -> Tuple[bool, bool]:
    """
    Validate BIC format and country code.
    Returns (format_valid, country_valid)
    """
    if not bic:
        return False, False
    
    bic = bic.upper().strip()
    
    # BIC must be 8 or 11 characters
    if len(bic) not in (8, 11):
        return False, False
    
    # Format: 4 letters (bank) + 2 letters (country) + 2 alphanum (location) + optional 3 alphanum (branch)
    pattern = r'^[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}([A-Z0-9]{3})?$'
    format_valid = bool(re.match(pattern, bic))
    
    # Country validation
    country_code = bic[4:6] if len(bic) >= 6 else ''
    country_valid = country_code in VALID_COUNTRY_CODES
    
    return format_valid, country_valid


def validate_bic4(bic: str) -> bool:
    """Validate first 4 characters of BIC (must be letters A-Z)."""
    if not bic or len(bic) < 4:
        return False
    return bic[:4].upper().isalpha()


def validate_iban(iban: str) -> Tuple[bool, bool]:
    """
    Validate IBAN format and checksum.
    Returns (format_valid, checksum_valid)
    """
    if not iban:
        return False, False
    
    # Clean IBAN
    iban = iban.upper().replace(' ', '').replace('-', '')
    
    if len(iban) < 5:
        return False, False
    
    # Check country code
    country_code = iban[:2]
    if not country_code.isalpha():
        return False, False
    
    # Check length for known countries
    expected_length = IBAN_LENGTHS.get(country_code)
    if expected_length:
        format_valid = len(iban) == expected_length
    else:
        # Generic IBAN length range
        format_valid = 15 <= len(iban) <= 34
    
    # Mod-97 checksum validation
    try:
        # Rearrange: move first 4 chars to end
        rearranged = iban[4:] + iban[:4]
        # Convert letters to numbers (A=10, B=11, etc.)
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
    """
    Validate US ABA routing number.
    Returns (format_valid, checksum_valid)
    """
    if not aba:
        return False, False
    
    aba = aba.strip()
    
    # Must be exactly 9 digits
    if len(aba) != 9 or not aba.isdigit():
        return False, False
    
    format_valid = True
    
    # Checksum: 3*d1 + 7*d2 + 1*d3 + 3*d4 + 7*d5 + 1*d6 + 3*d7 + 7*d8 + 1*d9 = 0 mod 10
    try:
        weights = [3, 7, 1, 3, 7, 1, 3, 7, 1]
        total = sum(int(d) * w for d, w in zip(aba, weights))
        checksum_valid = (total % 10 == 0)
    except (ValueError, TypeError):
        checksum_valid = False
    
    return format_valid, checksum_valid


def validate_currency(currency: str) -> bool:
    """Validate ISO 4217 currency code."""
    if not currency:
        return False
    return currency.upper().strip() in VALID_CURRENCY_CODES


def validate_country(country: str) -> bool:
    """Validate ISO 3166-1 alpha-2 country code."""
    if not country:
        return False
    return country.upper().strip() in VALID_COUNTRY_CODES


# =============================================================================
# FEATURE EXTRACTOR (Simplified from ifml_parser_v3)
# =============================================================================

class FeatureExtractor:
    """Extract validation-relevant features from IFML data."""
    
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
    
    def __init__(self, debug: bool = False):
        self.features = {}
        self.debug = debug
    
    def extract(self, data: Dict) -> Dict:
        """Extract all features from IFML data."""
        self.features = {}
        
        # Find Message node
        message = self._find_message(data)
        if not message:
            return self.features
        
        basic_payment = message.get('BasicPayment', {})
        if not basic_payment:
            return self.features
        
        # Parse basic transaction info
        self._parse_basic_info(basic_payment)
        
        # Parse monetary amounts
        self._parse_amounts(basic_payment)
        
        # Parse parties
        self._parse_parties(basic_payment)
        
        # Parse cross-party features
        self._parse_cross_party(basic_payment)
        
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
        self.features['transaction_id'] = basic_payment.get('TransactionID')
        self.features['transaction_uid'] = basic_payment.get('TransactionUID')
        self.features['source_code'] = basic_payment.get('SourceCode')
        self.features['incoming_msg_type'] = basic_payment.get('IncomingMsgType')
    
    def _parse_amounts(self, basic_payment: Dict):
        """Parse monetary amounts."""
        amounts = basic_payment.get('MonetaryAmount', [])
        if isinstance(amounts, dict):
            amounts = [amounts]
        
        for amt in amounts:
            amt_type = amt.get('@Type') or amt.get('Type') or 'Unknown'
            currency = amt.get('@Currency') or amt.get('Currency')
            amount_str = str(amt.get('Amount', '0'))
            
            if amt_type == 'Amount':
                self.features['primary_currency'] = currency
                self.features['has_currency'] = currency is not None
                self.features['currency_valid'] = validate_currency(currency) if currency else False
                
                # Check decimal places
                if '.' in amount_str:
                    decimals = len(amount_str.split('.')[1])
                    self.features['amount_decimals'] = decimals
                    self.features['amount_decimals_valid'] = decimals <= 2  # Most currencies
                else:
                    self.features['amount_decimals'] = 0
                    self.features['amount_decimals_valid'] = True
    
    def _parse_parties(self, basic_payment: Dict):
        """Parse all party information."""
        party_info = basic_payment.get('PartyInfo') or basic_payment.get('PartyInf') or {}
        
        if not isinstance(party_info, dict):
            return
        
        processed = set()
        for ifml_name, prefix in self.PARTY_MAPPING.items():
            party_data = party_info.get(ifml_name)
            if party_data:
                if isinstance(party_data, list):
                    party_data = party_data[0] if party_data else None
                if party_data and isinstance(party_data, dict):
                    if prefix not in processed:
                        self._extract_party(party_data, prefix)
                        processed.add(prefix)
    
    def _extract_party(self, party_data: Dict, prefix: str):
        """Extract features from a single party."""
        self.features[f'{prefix}_present'] = True
        
        # Get basic info section
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
        id_text = ''
        
        if id_field:
            self.features[f'{prefix}_has_id'] = True
            if isinstance(id_field, dict):
                id_type = (id_field.get('Type') or id_field.get('@Type') or '').upper()
                id_text = id_field.get('text') or id_field.get('#text') or ''
                
                if id_type in ('S', 'SWIFT', 'BIC'):
                    bic = id_text
                elif id_type == 'IBAN':
                    iban = id_text
                elif self._looks_like_bic(id_text):
                    bic = id_text
                elif self._looks_like_iban(id_text):
                    iban = id_text
            else:
                id_text = str(id_field)
                if self._looks_like_iban(id_text):
                    iban = id_text
                elif self._looks_like_bic(id_text):
                    bic = id_text
            
            # 9019: Check if ID needs cleaning (contains non-alphanumeric chars)
            if id_text:
                cleaned_id = re.sub(r'[^A-Za-z0-9]', '', id_text)
                self.features[f'{prefix}_id_needs_cleaning'] = (cleaned_id != id_text)
                if self.debug and cleaned_id != id_text:
                    print(f"[DEBUG] 9019: {prefix} ID needs cleaning: '{id_text}' -> '{cleaned_id}'")
        
        # Store BIC features
        if bic:
            self.features[f'{prefix}_has_bic'] = True
            self.features[f'{prefix}_bic'] = bic
            fmt_valid, ctry_valid = validate_bic(bic)
            self.features[f'{prefix}_bic_valid_format'] = fmt_valid
            self.features[f'{prefix}_bic_valid_country'] = ctry_valid
            self.features[f'{prefix}_bic4_valid'] = validate_bic4(bic)
            if len(bic) >= 6:
                self.features[f'{prefix}_bic_country'] = bic[4:6].upper()
        
        # Store IBAN features
        if iban:
            self.features[f'{prefix}_has_iban'] = True
            self.features[f'{prefix}_iban'] = iban
            fmt_valid, cksum_valid = validate_iban(iban)
            self.features[f'{prefix}_iban_valid_format'] = fmt_valid
            self.features[f'{prefix}_iban_checksum_valid'] = cksum_valid
            if len(iban) >= 2:
                self.features[f'{prefix}_iban_country'] = iban[:2].upper()
        
        # Parse account info
        acct_info = basic.get('AcctIDInfo') or basic.get('AcctIDInf')
        if acct_info and isinstance(acct_info, dict):
            self.features[f'{prefix}_has_account'] = True
            acct_id = acct_info.get('ID', {})
            if isinstance(acct_id, dict):
                acct_type = acct_id.get('@Type') or acct_id.get('Type')
                acct_value = acct_id.get('#text') or acct_id.get('text')
                self.features[f'{prefix}_account_type'] = acct_type
                
                if acct_type == 'IBAN' and acct_value:
                    self.features[f'{prefix}_has_iban'] = True
                    fmt_valid, cksum_valid = validate_iban(acct_value)
                    self.features[f'{prefix}_iban_valid_format'] = fmt_valid
                    self.features[f'{prefix}_iban_checksum_valid'] = cksum_valid
                    if len(acct_value) >= 2:
                        self.features[f'{prefix}_iban_country'] = acct_value[:2].upper()
        
        # Parse AdrBankID (NCH/routing number)
        adr_bank_id = basic.get('AdrBankID')
        if adr_bank_id:
            self.features[f'{prefix}_has_adr_bank_id'] = True
            if isinstance(adr_bank_id, dict):
                adr_type = adr_bank_id.get('@Type') or adr_bank_id.get('Type')
                adr_value = adr_bank_id.get('#text') or adr_bank_id.get('text')
            else:
                adr_type = None
                adr_value = str(adr_bank_id)
            
            if adr_type in ['FEDABA', 'FW', 'FEDWIRE'] or (adr_value and len(adr_value) == 9 and adr_value.isdigit()):
                self.features[f'{prefix}_has_nch'] = True
                self.features[f'{prefix}_nch_type'] = 'FEDABA'
                fmt_valid, cksum_valid = validate_fedaba(adr_value)
                self.features[f'{prefix}_nch_valid'] = fmt_valid
                self.features[f'{prefix}_fedaba_checksum_valid'] = cksum_valid
                # NCH validation is applicable for US domestic
                self.features[f'{prefix}_nch_validation_applicable'] = True
        
        # Parse country
        country = basic.get('Country') or party_data.get('Country')
        if country:
            self.features[f'{prefix}_has_country'] = True
            self.features[f'{prefix}_country'] = country
            self.features[f'{prefix}_country_valid'] = validate_country(country)
        
        # Check if IBAN is needed
        party_country = self.features.get(f'{prefix}_bic_country') or self.features.get(f'{prefix}_country')
        if party_country and party_country.upper() in IBAN_COUNTRIES:
            has_iban = self.features.get(f'{prefix}_has_iban', False)
            has_account = self.features.get(f'{prefix}_has_account', False)
            if not has_iban and not has_account:
                self.features[f'{prefix}_needs_iban'] = True
    
    def _parse_intermediary_array(self, basic_payment: Dict):
        """Parse intermediary array for 9018 (duplicate removal) detection."""
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
        
        if len(intm_list) < 2:
            return
        
        # Check for redundancy signals
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
            
            # BIC prefix (first 6 chars)
            id_info = basic.get('ID', {})
            if isinstance(id_info, dict):
                bic = id_info.get('text') or id_info.get('#text') or ''
                if bic and len(bic) >= 6 and bic[:4].isalpha():
                    bic_prefixes.append(bic[:6].upper())
        
        # Check for shared values (redundancy)
        self.features['intm_entries_share_adr_bank_id'] = (
            len(set(adr_bank_ids)) < len(adr_bank_ids) if len(adr_bank_ids) >= 2 else False
        )
        self.features['intm_entries_share_country'] = (
            len(set(countries)) < len(countries) if len(countries) >= 2 else False
        )
        self.features['intm_entries_share_bic_prefix'] = (
            len(set(bic_prefixes)) < len(bic_prefixes) if len(bic_prefixes) >= 2 else False
        )
        
        # Combined redundancy signal (2+ signals = redundant)
        redundancy_signals = sum([
            self.features.get('intm_entries_share_adr_bank_id', False),
            self.features.get('intm_entries_share_country', False),
            self.features.get('intm_entries_share_bic_prefix', False),
        ])
        self.features['intm_has_redundant_info'] = redundancy_signals >= 2
        
        if self.debug and self.features['intm_has_redundant_info']:
            print(f"[DEBUG] 9018: Multiple intermediaries with redundant info detected")
            print(f"  AdrBankIDs: {adr_bank_ids}")
            print(f"  Countries: {countries}")
            print(f"  BIC prefixes: {bic_prefixes}")
    
    def _parse_cross_party(self, basic_payment: Dict):
        """Parse cross-party validation features."""
        # First parse intermediary array for 9018
        self._parse_intermediary_array(basic_payment)
        
        # 8022: WireKey BIC vs BNP IBAN country match
        party_info = basic_payment.get('PartyInfo') or basic_payment.get('PartyInf') or {}
        
        cdt_data = party_info.get('CreditPartyInfo') or party_info.get('CreditPartyInf') or {}
        if isinstance(cdt_data, list):
            cdt_data = cdt_data[0] if cdt_data else {}
        
        # Extract WireKey BIC from credit party
        wirekey_bic_country = None
        if isinstance(cdt_data, dict):
            basic = (cdt_data.get('BasicPartyInfo') or cdt_data.get('BasicPartyInf') or
                    cdt_data.get('AccountPartyInfo') or cdt_data.get('AccountPartyInf') or cdt_data)
            if isinstance(basic, list):
                basic = basic[0] if basic else {}
            if isinstance(basic, dict):
                wirekey = basic.get('WireKey')
                if wirekey and isinstance(wirekey, dict):
                    wk_acct = wirekey.get('AcctIDInf') or wirekey.get('AcctIDInfo')
                    if wk_acct and isinstance(wk_acct, dict):
                        wk_id = wk_acct.get('ID')
                        if wk_id and isinstance(wk_id, dict):
                            wk_type = (wk_id.get('@Type') or wk_id.get('Type') or '').upper()
                            wk_text = wk_id.get('#text') or wk_id.get('text')
                            if wk_type in ['S', 'SWIFT', 'BIC'] and wk_text and len(wk_text) >= 6:
                                wirekey_bic_country = wk_text[4:6].upper()
        
        # Get BNP IBAN country
        bnp_iban_country = self.features.get('bnp_iban_country')
        
        # Compare
        if wirekey_bic_country and bnp_iban_country:
            self.features['wirekey_bic_bnp_iban_match'] = (wirekey_bic_country == bnp_iban_country)
            if self.debug and wirekey_bic_country != bnp_iban_country:
                print(f"[DEBUG] 8022: WireKey BIC country={wirekey_bic_country}, BNP IBAN country={bnp_iban_country}")
    
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
        if not s[:2].isalpha():
            return False
        country = s[:2]
        expected_len = IBAN_LENGTHS.get(country)
        if expected_len and len(s) == expected_len:
            return True
        if len(s) >= 15 and s[:2].isalpha() and s[2:4].isdigit():
            return True
        return False


# =============================================================================
# VALIDATION ENGINE
# =============================================================================

class ValidationEngine:
    """Apply validation rules to predict 8XXX codes."""
    
    PREFIX_TO_SUFFIX = {
        'orig': 'ORGPTY',
        'send': 'SNDBNK',
        'dbt': 'DBTPTY',
        'cdt': 'CDTPTY',
        'intm': 'INTBNK',
        'bnf': 'BNFBNK',
        'bnp': 'BNPPTY',
        'acwi': 'ACWI',
        'ordi': 'ORDI',
    }
    
    PARTY_PREFIXES = ['orig', 'send', 'dbt', 'cdt', 'intm', 'bnf', 'bnp', 'acwi', 'ordi']
    
    def __init__(self, debug: bool = False):
        self.debug = debug
    
    def validate(self, features: Dict) -> List[ValidationResult]:
        """Apply all validation rules and return results."""
        results = []
        
        # Check global codes
        results.extend(self._check_global_codes(features))
        
        # Check party-specific codes
        for prefix in self.PARTY_PREFIXES:
            if features.get(f'{prefix}_present'):
                suffix = self.PREFIX_TO_SUFFIX.get(prefix)
                results.extend(self._check_party_codes(features, prefix, suffix))
        
        return results
    
    def _check_global_codes(self, f: Dict) -> List[ValidationResult]:
        """Check codes that apply globally (not party-specific)."""
        results = []
        
        # 8007: Amount decimal places exceeded
        if f.get('amount_decimals_valid') is False:
            results.append(ValidationResult(
                code='8007',
                party=None,
                fires=True,
                reason=f"Amount has {f.get('amount_decimals', 0)} decimal places (max 2)",
                features_checked={'amount_decimals_valid': False}
            ))
        
        # 8022: IBAN/BIC country mismatch
        if f.get('wirekey_bic_bnp_iban_match') is False:
            results.append(ValidationResult(
                code='8022',
                party='BNFBNK',  # Always tagged to BNFBNK
                fires=True,
                reason="WireKey BIC country doesn't match BNP IBAN country",
                features_checked={'wirekey_bic_bnp_iban_match': False}
            ))
        
        # 8124: Invalid currency
        if f.get('has_currency') and f.get('currency_valid') is False:
            results.append(ValidationResult(
                code='8124',
                party=None,
                fires=True,
                reason=f"Invalid currency code: {f.get('primary_currency')}",
                features_checked={'currency_valid': False}
            ))
        
        # 9018: Duplicate party information removed (redundant intermediaries)
        if f.get('intm_has_multiple') and f.get('intm_has_redundant_info'):
            results.append(ValidationResult(
                code='9018',
                party=None,
                fires=True,
                reason=f"Multiple intermediaries ({f.get('intm_count', 0)}) with redundant info",
                features_checked={
                    'intm_has_multiple': True,
                    'intm_has_redundant_info': True,
                    'intm_entries_share_adr_bank_id': f.get('intm_entries_share_adr_bank_id'),
                    'intm_entries_share_country': f.get('intm_entries_share_country'),
                    'intm_entries_share_bic_prefix': f.get('intm_entries_share_bic_prefix'),
                }
            ))
        
        return results
    
    def _check_party_codes(self, f: Dict, prefix: str, suffix: str) -> List[ValidationResult]:
        """Check codes that apply to specific parties."""
        results = []
        
        # Helper to get prefixed feature
        def get(key: str, default=None):
            return f.get(f'{prefix}_{key}', default)
        
        # 8001: Invalid BIC
        if get('has_bic'):
            fmt_valid = get('bic_valid_format', True)
            ctry_valid = get('bic_valid_country', True)
            if not fmt_valid or not ctry_valid:
                results.append(ValidationResult(
                    code='8001',
                    party=suffix,
                    fires=True,
                    reason=f"BIC '{get('bic')}': format_valid={fmt_valid}, country_valid={ctry_valid}",
                    features_checked={
                        'has_bic': True,
                        'bic_valid_format': fmt_valid,
                        'bic_valid_country': ctry_valid
                    }
                ))
        
        # 8004: IBAN cannot be derived (only for party accounts, not banks)
        if suffix not in BANK_SUFFIXES:  # Banks don't need IBANs
            if get('needs_iban') and not get('has_iban'):
                results.append(ValidationResult(
                    code='8004',
                    party=suffix,
                    fires=True,
                    reason=f"Party in IBAN country but no IBAN present",
                    features_checked={
                        'needs_iban': True,
                        'has_iban': False
                    }
                ))
        
        # 8005: Invalid BIC4
        if get('has_bic') and get('bic4_valid') is False:
            results.append(ValidationResult(
                code='8005',
                party=suffix,
                fires=True,
                reason=f"BIC4 invalid (first 4 chars must be A-Z)",
                features_checked={
                    'has_bic': True,
                    'bic4_valid': False
                }
            ))
        
        # 8006: Invalid country code
        if get('has_country') and get('country_valid') is False:
            results.append(ValidationResult(
                code='8006',
                party=suffix,
                fires=True,
                reason=f"Invalid country code: {get('country')}",
                features_checked={
                    'has_country': True,
                    'country_valid': False
                }
            ))
        
        # 8894: Invalid IBAN
        if get('has_iban'):
            fmt_valid = get('iban_valid_format', True)
            cksum_valid = get('iban_checksum_valid', True)
            if not fmt_valid or not cksum_valid:
                results.append(ValidationResult(
                    code='8894',
                    party=suffix,
                    fires=True,
                    reason=f"IBAN invalid: format_valid={fmt_valid}, checksum_valid={cksum_valid}",
                    features_checked={
                        'has_iban': True,
                        'iban_valid_format': fmt_valid,
                        'iban_checksum_valid': cksum_valid
                    }
                ))
        
        # 8895: Invalid NCH (FEDABA checksum)
        if get('has_nch') and get('nch_validation_applicable'):
            nch_valid = get('nch_valid', True)
            fedaba_valid = get('fedaba_checksum_valid', True)
            if not nch_valid or not fedaba_valid:
                results.append(ValidationResult(
                    code='8895',
                    party=suffix,
                    fires=True,
                    reason=f"NCH invalid: format_valid={nch_valid}, checksum_valid={fedaba_valid}",
                    features_checked={
                        'has_nch': True,
                        'nch_valid': nch_valid,
                        'fedaba_checksum_valid': fedaba_valid
                    }
                ))
        
        # 8898: IBAN checksum failed (more specific than 8894)
        if get('has_iban') and get('iban_valid_format') and get('iban_checksum_valid') is False:
            results.append(ValidationResult(
                code='8898',
                party=suffix,
                fires=True,
                reason=f"IBAN checksum validation failed (mod-97)",
                features_checked={
                    'has_iban': True,
                    'iban_valid_format': True,
                    'iban_checksum_valid': False
                }
            ))
        
        # 9019: Party identifier cleaned of non-alphanumeric characters
        if get('id_needs_cleaning'):
            results.append(ValidationResult(
                code='9019',
                party=suffix,
                fires=True,
                reason=f"ID contains non-alphanumeric characters that will be cleaned",
                features_checked={
                    'id_needs_cleaning': True
                }
            ))
        
        return results


# =============================================================================
# RESPONSE PARSER
# =============================================================================

def extract_actual_codes(response: Dict, filter_prefix: str = None) -> Set[str]:
    """Extract actual codes from ACE response.
    
    Args:
        response: ACE response dictionary
        filter_prefix: If provided, only return codes starting with this prefix
                      (e.g., '8' for 8XXX, '9' for 9XXX, None for all)
    """
    codes = set()
    
    # Find AuditTrail
    audit = None
    paths = [
        ['AuditTrail'],
        ['IFML', 'File', 'Message', 'AuditTrail'],
        ['IFML', 'File', 'AuditTrail'],
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
    
    # Parse MsgStatus entries
    msg_status = audit.get('MsgStatus', [])
    if isinstance(msg_status, dict):
        msg_status = [msg_status]
    
    for item in msg_status:
        code = item.get('Code', '')
        party = item.get('Party', '')
        
        if code:
            # Filter by prefix if specified
            if filter_prefix and not code.startswith(filter_prefix):
                continue
            
            # Include party suffix if present
            if party:
                codes.add(f"{code}_{party}")
            else:
                codes.add(code)
    
    return codes


def get_transaction_id(data: Dict) -> str:
    """Extract transaction ID from IFML data."""
    paths = [
        ['Request', 'IFML', 'File', 'Message', 'BasicPayment', 'TransactionUID'],
        ['Request', 'IFML', 'File', 'Message', 'BasicPayment', 'TransactionID'],
        ['IFML', 'File', 'Message', 'BasicPayment', 'TransactionUID'],
        ['IFML', 'File', 'Message', 'BasicPayment', 'TransactionID'],
    ]
    
    for path in paths:
        current = data
        for key in path:
            if isinstance(current, dict):
                current = current.get(key)
            else:
                current = None
                break
        if isinstance(current, str):
            return current
    
    return "UNKNOWN"


# =============================================================================
# MAIN VALIDATOR
# =============================================================================

class ACEValidator:
    """Main validator class."""
    
    def __init__(self, debug: bool = False):
        self.extractor = FeatureExtractor(debug=debug)
        self.engine = ValidationEngine(debug=debug)
        self.debug = debug
    
    def validate_payment(self, request: Dict, response: Dict) -> PaymentResult:
        """Validate a single payment."""
        payment_id = get_transaction_id(request)
        
        # Extract features
        features = self.extractor.extract(request)
        
        # Get predictions
        results = self.engine.validate(features)
        predicted = {r.code_with_party for r in results if r.fires}
        
        # Get actual codes (both 8XXX and 9XXX that we can predict)
        actual_8xxx = extract_actual_codes(response, filter_prefix='8')
        actual_9xxx = extract_actual_codes(response, filter_prefix='9')
        actual_raw = actual_8xxx | actual_9xxx
        
        # Separate predictable vs directory-dependent
        actual_predictable = set()
        excluded = set()
        
        for code_str in actual_raw:
            base_code = code_str.split('_')[0]
            if base_code in DIRECTORY_DEPENDENT:
                excluded.add(code_str)
            elif base_code in PREDICTABLE_CODES:
                actual_predictable.add(code_str)
            else:
                # Code not in our prediction set - exclude
                excluded.add(code_str)
        
        # Compare (using predictable codes only)
        correct = predicted & actual_predictable
        missed = actual_predictable - predicted
        false_positives = predicted - actual_predictable
        
        return PaymentResult(
            payment_id=payment_id,
            predicted_codes=predicted,
            actual_codes=actual_predictable,
            correct=correct,
            missed=missed,
            false_positives=false_positives,
            details=results,
            excluded_codes=excluded
        )
    
    def validate_file(self, filepath: Path) -> List[PaymentResult]:
        """Validate all payments in a file."""
        results = []
        
        with open(filepath) as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            # Single payment or dict of payments
            if 'Request' in data and 'Response' in data:
                # Single payment with Request/Response
                result = self.validate_payment(data, data.get('Response', {}))
                results.append(result)
            else:
                # Dict keyed by transaction ID
                for key, val in data.items():
                    if isinstance(val, dict) and 'Request' in val and 'Response' in val:
                        result = self.validate_payment(val, val.get('Response', {}))
                        result.payment_id = key
                        results.append(result)
        
        return results
    
    def validate_directory(self, dirpath: Path) -> List[PaymentResult]:
        """Validate all JSON files in a directory."""
        results = []
        
        for filepath in sorted(dirpath.glob('*.json')):
            try:
                file_results = self.validate_file(filepath)
                results.extend(file_results)
            except Exception as e:
                if self.debug:
                    print(f"Error processing {filepath}: {e}")
        
        return results
    
    def compute_accuracy(self, results: List[PaymentResult]) -> AccuracyReport:
        """Compute accuracy metrics."""
        total = len(results)
        correct_payments = sum(1 for r in results if r.passed)
        
        tp = sum(len(r.correct) for r in results)
        fp = sum(len(r.false_positives) for r in results)
        fn = sum(len(r.missed) for r in results)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Code breakdown
        code_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
        
        for r in results:
            for code in r.correct:
                base = code.split('_')[0]
                code_stats[base]['tp'] += 1
            for code in r.false_positives:
                base = code.split('_')[0]
                code_stats[base]['fp'] += 1
            for code in r.missed:
                base = code.split('_')[0]
                code_stats[base]['fn'] += 1
        
        return AccuracyReport(
            total_payments=total,
            payments_correct=correct_payments,
            total_predictions=tp + fp,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            precision=precision,
            recall=recall,
            f1=f1,
            code_breakdown=dict(code_stats)
        )


# =============================================================================
# CLI
# =============================================================================

def print_result(result: PaymentResult, verbose: bool = False):
    """Print a single payment result."""
    status = "PASS" if result.passed else "FAIL"
    
    if result.passed and not verbose:
        print(f"{result.payment_id}: {status}")
        return
    
    print(f"\n{'='*60}")
    print(f"Payment: {result.payment_id}")
    print(f"Status: {status}")
    print(f"Predicted: {sorted(result.predicted_codes)}")
    print(f"Actual: {sorted(result.actual_codes)}")
    
    if result.correct:
        print(f"Correct (TP): {sorted(result.correct)}")
    if result.missed:
        print(f"Missed (FN): {sorted(result.missed)}")
    if result.false_positives:
        print(f"False Positive: {sorted(result.false_positives)}")
    if result.excluded_codes:
        print(f"Excluded (directory-dependent): {sorted(result.excluded_codes)}")
    
    if verbose:
        print("\nValidation Details:")
        for detail in result.details:
            if detail.fires:
                print(f"  {detail.code_with_party}: {detail.reason}")


def print_accuracy(report: AccuracyReport):
    """Print accuracy report."""
    print(f"\n{'='*60}")
    print("ACCURACY REPORT")
    print(f"{'='*60}")
    print(f"Total Payments: {report.total_payments}")
    print(f"Payments Correct: {report.payments_correct} ({100*report.payments_correct/report.total_payments:.1f}%)")
    print(f"\nCode-Level Metrics:")
    print(f"  True Positives: {report.true_positives}")
    print(f"  False Positives: {report.false_positives}")
    print(f"  False Negatives: {report.false_negatives}")
    print(f"\n  Precision: {report.precision:.3f}")
    print(f"  Recall: {report.recall:.3f}")
    print(f"  F1 Score: {report.f1:.3f}")
    
    if report.code_breakdown:
        print(f"\nBreakdown by Code:")
        for code in sorted(report.code_breakdown.keys()):
            stats = report.code_breakdown[code]
            total = stats['tp'] + stats['fp'] + stats['fn']
            if total > 0:
                p = stats['tp'] / (stats['tp'] + stats['fp']) if (stats['tp'] + stats['fp']) > 0 else 0
                r = stats['tp'] / (stats['tp'] + stats['fn']) if (stats['tp'] + stats['fn']) > 0 else 0
                print(f"  {code}: TP={stats['tp']}, FP={stats['fp']}, FN={stats['fn']} (P={p:.2f}, R={r:.2f})")


def main():
    parser = argparse.ArgumentParser(description='ACE 8XXX Code Validator')
    subparsers = parser.add_subparsers(dest='command', help='Command')
    
    # Single payment validation
    single_parser = subparsers.add_parser('single', help='Validate single payment')
    single_parser.add_argument('-r', '--request', required=True, help='Request JSON file')
    single_parser.add_argument('-s', '--response', required=True, help='Response JSON file')
    single_parser.add_argument('-v', '--verbose', action='store_true')
    single_parser.add_argument('--debug', action='store_true')
    
    # Directory validation
    validate_parser = subparsers.add_parser('validate', help='Validate directory')
    validate_parser.add_argument('-i', '--input', required=True, help='Input directory or file')
    validate_parser.add_argument('-o', '--output', help='Output report JSON')
    validate_parser.add_argument('-v', '--verbose', action='store_true')
    validate_parser.add_argument('--debug', action='store_true')
    
    # Accuracy summary
    accuracy_parser = subparsers.add_parser('accuracy', help='Show accuracy summary')
    accuracy_parser.add_argument('-i', '--input', required=True, help='Input directory or file')
    accuracy_parser.add_argument('--debug', action='store_true')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    debug = getattr(args, 'debug', False)
    validator = ACEValidator(debug=debug)
    
    if args.command == 'single':
        with open(args.request) as f:
            request = json.load(f)
        with open(args.response) as f:
            response = json.load(f)
        
        result = validator.validate_payment(request, response)
        print_result(result, args.verbose)
    
    elif args.command == 'validate':
        path = Path(args.input)
        if path.is_file():
            results = validator.validate_file(path)
        else:
            results = validator.validate_directory(path)
        
        for result in results:
            print_result(result, args.verbose)
        
        report = validator.compute_accuracy(results)
        print_accuracy(report)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump({
                    'accuracy': asdict(report),
                    'results': [
                        {
                            'payment_id': r.payment_id,
                            'passed': r.passed,
                            'predicted': list(r.predicted_codes),
                            'actual': list(r.actual_codes),
                            'missed': list(r.missed),
                            'false_positives': list(r.false_positives)
                        }
                        for r in results
                    ]
                }, f, indent=2)
            print(f"\nReport saved to {args.output}")
    
    elif args.command == 'accuracy':
        path = Path(args.input)
        if path.is_file():
            results = validator.validate_file(path)
        else:
            results = validator.validate_directory(path)
        
        report = validator.compute_accuracy(results)
        print_accuracy(report)


if __name__ == '__main__':
    main()
