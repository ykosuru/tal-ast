"""
IFML Parser - Extract structured features from ACE Pelican IFML payment messages.
Handles nested JSON structure and normalizes fields for ML consumption.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path


# Valid ISO 3166-1 alpha-2 country codes (subset of most common)
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

# IBAN lengths by country (ISO 13616)
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


def validate_bic(bic: str) -> Tuple[bool, bool]:
    """
    Validate BIC format and country code.
    Returns (format_valid, country_valid)
    
    BIC format: 4 letters (bank) + 2 letters (country) + 2 alphanum (location) + optional 3 alphanum (branch)
    """
    if not bic:
        return False, False
    
    bic = bic.upper().strip()
    
    # Check length (8 or 11)
    if len(bic) not in (8, 11):
        return False, False
    
    # Check format: AAAACCLL or AAAACCLLBBB
    # First 4: letters (bank code)
    # Next 2: letters (country code)
    # Next 2: alphanumeric (location)
    # Optional 3: alphanumeric (branch)
    pattern = r'^[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}([A-Z0-9]{3})?$'
    format_valid = bool(re.match(pattern, bic))
    
    # Check country code (positions 5-6, 0-indexed 4-5)
    country_code = bic[4:6] if len(bic) >= 6 else ''
    country_valid = country_code in VALID_COUNTRY_CODES
    
    return format_valid, country_valid


def validate_iban(iban: str) -> Tuple[bool, bool]:
    """
    Validate IBAN format and checksum.
    Returns (format_valid, checksum_valid)
    
    IBAN format: 2 letters (country) + 2 digits (check) + up to 30 alphanum (BBAN)
    Checksum: mod-97 validation
    """
    if not iban:
        return False, False
    
    iban = iban.upper().replace(' ', '').replace('-', '')
    
    # Check minimum length and country code
    if len(iban) < 5:
        return False, False
    
    country_code = iban[:2]
    if not country_code.isalpha():
        return False, False
    
    # Check if country is in our IBAN length table
    expected_length = IBAN_LENGTHS.get(country_code)
    if expected_length:
        format_valid = len(iban) == expected_length
    else:
        # Unknown country, just check basic format
        format_valid = 15 <= len(iban) <= 34
    
    # Validate checksum using mod-97
    try:
        # Move first 4 chars to end
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
        
        # Check if mod 97 == 1
        checksum_valid = int(numeric) % 97 == 1
    except (ValueError, OverflowError):
        checksum_valid = False
    
    return format_valid, checksum_valid


def detect_account_type(account: str, account_type: str = None) -> Dict[str, bool]:
    """
    Detect specific account identifier types.
    Returns dict with is_clabe, is_fedaba, is_chips_aba, is_chips_uid
    """
    result = {
        'is_clabe': False,
        'is_fedaba': False,
        'is_chips_aba': False,
        'is_chips_uid': False,
        'is_numeric': False,
        'length': 0
    }
    
    if not account:
        return result
    
    account = account.strip()
    result['length'] = len(account)
    result['is_numeric'] = account.isdigit()
    
    # CLABE: Mexican 18-digit bank account
    if len(account) == 18 and account.isdigit():
        result['is_clabe'] = True
    
    # FEDABA / Fedwire ABA: US 9-digit routing number
    if len(account) == 9 and account.isdigit():
        result['is_fedaba'] = True
    
    # CHIPS ABA: Also 9 digits but different validation
    # For now, same as FEDABA
    if len(account) == 9 and account.isdigit():
        result['is_chips_aba'] = True
    
    # CHIPS UID: 6 digits
    if len(account) == 6 and account.isdigit():
        result['is_chips_uid'] = True
    
    return result


@dataclass
class PartyInfo:
    """Normalized party information."""
    party_type: str
    has_id: bool = False
    id_value: Optional[str] = None
    id_type: Optional[str] = None
    has_bic: bool = False
    bic: Optional[str] = None
    bic_length: int = 0  # 4, 8, or 11 chars typically
    bic_country: Optional[str] = None  # Country code from BIC (chars 5-6)
    bic_valid_format: bool = False  # Is BIC format valid (8 or 11 alphanumeric)?
    bic_valid_country: bool = False  # Is BIC country code valid ISO?
    has_account: bool = False
    account_type: Optional[str] = None  # IBAN, BBAN, etc.
    account_value: Optional[str] = None
    account_length: int = 0
    account_numeric_only: bool = False
    iban_country: Optional[str] = None  # Country code from IBAN (first 2 chars)
    iban_valid_format: bool = False  # Is IBAN format valid?
    iban_checksum_valid: bool = False  # Does IBAN pass mod-97 check?
    bic_iban_country_match: Optional[bool] = None  # Do BIC and IBAN countries match?
    bic_party_country_match: Optional[bool] = None  # Does BIC country match party country?
    # Account identifier types detected
    is_clabe: bool = False  # Mexican 18-digit
    is_fedaba: bool = False  # US 9-digit routing
    is_chips_aba: bool = False
    is_chips_uid: bool = False
    country: Optional[str] = None
    mailing_country: Optional[str] = None
    residence_country: Optional[str] = None
    address_line_count: int = 0
    has_name: bool = False
    bank_flag: Optional[str] = None
    charge_flag: Optional[str] = None


@dataclass 
class MonetaryInfo:
    """Normalized monetary amount information."""
    amount_type: str
    currency: Optional[str] = None
    amount: float = 0.0


@dataclass
class IFMLFeatures:
    """Complete feature set extracted from an IFML request."""
    # Transaction identifiers
    transaction_id: Optional[str] = None
    transaction_uid: Optional[str] = None
    
    # Source/format info
    source_code: Optional[str] = None
    incoming_msg_type: Optional[str] = None
    incoming_format: Optional[str] = None
    bank_operation_code: Optional[str] = None
    
    # Location
    location_id_bank: Optional[str] = None
    
    # Monetary amounts
    amounts: List[MonetaryInfo] = field(default_factory=list)
    primary_currency: Optional[str] = None
    primary_amount: float = 0.0
    has_instructed_amount: bool = False
    instructed_amount: float = 0.0
    amount_mismatch: bool = False
    
    # Party information
    parties: Dict[str, PartyInfo] = field(default_factory=dict)
    
    # Derived cross-party features
    is_cross_border: bool = False
    has_intermediary: bool = False
    has_beneficiary_bank: bool = False
    originator_country: Optional[str] = None
    beneficiary_country: Optional[str] = None
    
    # Field presence flags
    has_bank_info: bool = False
    bank_info_count: int = 0
    
    # Raw data for debugging
    raw_json: Optional[dict] = None


class IFMLParser:
    """
    Parser for ACE Pelican IFML payment messages.
    Extracts structured features suitable for ML model training.
    """
    
    PARTY_TYPES = [
        'OriginatingPartyInfo',
        'SendingBankInfo', 
        'DebitPartyInfo',
        'CreditPartyInfo',
        'IntermediaryBankInfo',
        'BeneficiaryBankInfo',
        'AccountWithInstitution',
        'OrderingInstitution'
    ]
    
    # Alternate naming (without 'o' - Inf instead of Info)
    PARTY_TYPES_ALT = [
        'OriginatingPartyInf',
        'SendingBankInf',
        'DebitPartyInf',
        'CreditPartyInf',
        'IntermediaryBankInf',
        'BeneficiaryBankInf',
        'BeneficiaryPartyInf',
        'AccountWithInstitution',
        'OrderingInstitution'
    ]
    
    def __init__(self, keep_raw: bool = False):
        self.keep_raw = keep_raw
    
    def parse_file(self, filepath: str) -> IFMLFeatures:
        """Parse IFML from a file path."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return self.parse(data)
    
    def parse_json_string(self, json_str: str) -> IFMLFeatures:
        """Parse IFML from a JSON string."""
        data = json.loads(json_str)
        return self.parse(data)
    
    def parse(self, data: dict) -> IFMLFeatures:
        """
        Parse IFML data structure and extract features.
        Handles both request and response formats.
        """
        features = IFMLFeatures()
        
        if self.keep_raw:
            features.raw_json = data
        
        # Navigate to BasicPayment - handle different wrapper structures
        basic_payment = self._find_basic_payment(data)
        
        if not basic_payment:
            return features
        
        # Extract transaction info
        features.transaction_id = basic_payment.get('TransactionID')
        features.transaction_uid = basic_payment.get('TransactionUID')
        
        # Extract source/format info
        features.source_code = basic_payment.get('SourceCode')
        features.incoming_msg_type = basic_payment.get('IncomingMsgType')
        features.incoming_format = basic_payment.get('IncomingFormat')
        features.bank_operation_code = basic_payment.get('BankOperationCode')
        
        # Extract location info - handle both naming conventions
        location_info = basic_payment.get('LocationInfo') or basic_payment.get('LocationInf') or {}
        features.location_id_bank = location_info.get('IDBank')
        
        # Extract monetary amounts
        self._parse_monetary_amounts(basic_payment, features)
        
        # Extract party information
        self._parse_party_info(basic_payment, features)
        
        # Extract bank info
        self._parse_bank_info(basic_payment, features)
        
        # Derive cross-party features
        self._derive_cross_party_features(features)
        
        return features
    
    def _find_basic_payment(self, data: dict) -> Optional[dict]:
        """Navigate nested structure to find BasicPayment node."""
        # Try different wrapper patterns
        patterns = [
            # Pattern: {txn_id: {Request: {IFML: {File: {Message: {BasicPayment: ...}}}}}}
            lambda d: self._navigate_path(d, ['*', 'Request', 'IFML', 'File', 'Message', 'BasicPayment']),
            # Pattern: {Request: {IFML: {File: {Message: {BasicPayment: ...}}}}}
            lambda d: self._navigate_path(d, ['Request', 'IFML', 'File', 'Message', 'BasicPayment']),
            # Pattern: {IFML: {File: {Message: {BasicPayment: ...}}}}
            lambda d: self._navigate_path(d, ['IFML', 'File', 'Message', 'BasicPayment']),
            # Pattern: {BasicPayment: ...}
            lambda d: d.get('BasicPayment'),
        ]
        
        for pattern in patterns:
            result = pattern(data)
            if result:
                return result
        
        return None
    
    def _navigate_path(self, data: dict, path: List[str]) -> Optional[dict]:
        """Navigate a path through nested dicts, '*' matches any single key."""
        current = data
        
        for key in path:
            if not isinstance(current, dict):
                return None
            
            if key == '*':
                # Match any single key
                if len(current) == 1:
                    current = list(current.values())[0]
                else:
                    return None
            else:
                current = current.get(key)
                if current is None:
                    return None
        
        return current
    
    def _parse_monetary_amounts(self, basic_payment: dict, features: IFMLFeatures):
        """Extract and normalize monetary amount information."""
        amounts = basic_payment.get('MonetaryAmount', [])
        
        if isinstance(amounts, dict):
            amounts = [amounts]
        
        for amt in amounts:
            # Handle both @Type and Type naming conventions
            amt_type = amt.get('@Type') or amt.get('Type') or 'Unknown'
            currency = amt.get('@Currency') or amt.get('Currency')
            amount_val = self._parse_amount(amt.get('Amount', '0'))
            
            monetary_info = MonetaryInfo(
                amount_type=amt_type,
                currency=currency,
                amount=amount_val
            )
            features.amounts.append(monetary_info)
            
            # Track primary amount
            if amt_type == 'Amount':
                features.primary_currency = currency
                features.primary_amount = amount_val
            
            # Track instructed amount
            if amt_type == 'RelatedAmt' or amt.get('SubType') == 'INSTRUCTED AMOUNT':
                features.has_instructed_amount = True
                features.instructed_amount = amount_val
        
        # Check for amount mismatch
        if features.has_instructed_amount and features.primary_amount > 0:
            features.amount_mismatch = abs(features.primary_amount - features.instructed_amount) > 0.01
    
    def _parse_amount(self, amount_str: str) -> float:
        """Parse amount string to float."""
        try:
            return float(str(amount_str).replace(',', ''))
        except (ValueError, TypeError):
            return 0.0
    
    def _parse_party_info(self, basic_payment: dict, features: IFMLFeatures):
        """Extract party information for all party types."""
        # Handle both PartyInfo and PartyInf naming
        party_info = basic_payment.get('PartyInfo') or basic_payment.get('PartyInf') or {}
        
        # Try standard naming (Info)
        for party_type in self.PARTY_TYPES:
            party_data = party_info.get(party_type)
            if party_data:
                parsed_party = self._parse_single_party(party_type, party_data)
                features.parties[party_type] = parsed_party
        
        # Try alternate naming (Inf) and map to standard names
        alt_to_standard = {
            'OriginatingPartyInf': 'OriginatingPartyInfo',
            'SendingBankInf': 'SendingBankInfo',
            'DebitPartyInf': 'DebitPartyInfo',
            'CreditPartyInf': 'CreditPartyInfo',
            'IntermediaryBankInf': 'IntermediaryBankInfo',
            'BeneficiaryBankInf': 'BeneficiaryBankInfo',
            'BeneficiaryPartyInf': 'BeneficiaryBankInfo',  # Map to BeneficiaryBank
        }
        
        for alt_type, standard_type in alt_to_standard.items():
            if standard_type not in features.parties:  # Don't overwrite if already parsed
                party_data = party_info.get(alt_type)
                if party_data:
                    parsed_party = self._parse_single_party(standard_type, party_data)
                    features.parties[standard_type] = parsed_party
    
    def _parse_single_party(self, party_type: str, party_data: dict) -> PartyInfo:
        """Parse a single party's information."""
        party = PartyInfo(party_type=party_type)
        
        # Handle nested structures - multiple naming conventions
        basic_info = (
            party_data.get('BasicPartyInfo') or 
            party_data.get('BasicPartyInf') or
            party_data.get('BasicIDInfo') or 
            party_data.get('AccountPartyInfo') or
            party_data.get('AccountPartyInf') or
            party_data.get('BasicPartyBankInfo') or
            party_data.get('BasicPartyBankInf') or
            party_data
        )
        
        # Extract ID - handle multiple formats
        id_field = basic_info.get('ID')
        if id_field:
            if isinstance(id_field, dict):
                party.has_id = True
                # Handle both @Type/#text and Type/text formats
                party.id_type = id_field.get('@Type') or id_field.get('Type')
                party.id_value = id_field.get('#text') or id_field.get('text')
                if party.id_type == 'BIC':
                    party.has_bic = True
                    party.bic = party.id_value
            else:
                party.has_id = True
                party.id_value = str(id_field)
        
        # Check for BIC in ID type
        if isinstance(id_field, dict):
            id_type = id_field.get('@Type') or id_field.get('Type')
            if id_type == 'BIC':
                party.has_bic = True
                party.bic = id_field.get('#text') or id_field.get('text')
        
        # Extract account info - handle both AcctIDInfo and AcctIDInf
        acct_info = basic_info.get('AcctIDInfo') or basic_info.get('AcctIDInf')
        if acct_info:
            party.has_account = True
            if isinstance(acct_info, dict):
                acct_id = acct_info.get('ID', {})
                if isinstance(acct_id, dict):
                    party.account_type = acct_id.get('@Type') or acct_id.get('Type')
                    party.account_value = acct_id.get('#text') or acct_id.get('text')
        
        # Check AdrBankID for additional account info
        adr_bank_id = basic_info.get('AdrBankID')
        if adr_bank_id:
            party.has_account = True
            if isinstance(adr_bank_id, dict):
                party.account_type = party.account_type or adr_bank_id.get('@Type')
        
        # Extract country codes - check both basic_info and party_data levels
        party.country = (
            basic_info.get('Country') or 
            party_data.get('Country')
        )
        party.mailing_country = (
            basic_info.get('MailingCountry') or
            party_data.get('MailingCountry')
        )
        party.residence_country = (
            basic_info.get('ResidenceCountry') or
            party_data.get('ResidenceCountry')
        )
        
        # Extract address info - handle both AddressInfo and AddressInf
        address_info = (
            basic_info.get('AddressInfo') or 
            basic_info.get('AddressInf') or
            party_data.get('AddressInfo') or
            party_data.get('AddressInf')
        )
        if address_info:
            if isinstance(address_info, list):
                party.address_line_count = len(address_info)
                party.has_name = len(address_info) > 0
            elif isinstance(address_info, dict):
                party.address_line_count = 1
                party.has_name = True
        
        # Extract flags
        party.bank_flag = basic_info.get('BankFlag')
        party.charge_flag = basic_info.get('ChargeFlag')
        
        # Parse BIC structure and validate
        if party.bic:
            party.bic_length = len(party.bic)
            # BIC country is characters 5-6 (0-indexed: 4-5)
            if len(party.bic) >= 6:
                party.bic_country = party.bic[4:6].upper()
            # Validate BIC format and country
            party.bic_valid_format, party.bic_valid_country = validate_bic(party.bic)
        
        # Parse IBAN and validate
        if party.account_type == 'IBAN' and party.account_value:
            # IBAN country is first 2 characters
            if len(party.account_value) >= 2:
                party.iban_country = party.account_value[0:2].upper()
            # Validate IBAN format and checksum
            party.iban_valid_format, party.iban_checksum_valid = validate_iban(party.account_value)
        
        # Detect account type (CLABE, FEDABA, etc.) and get length
        if party.account_value:
            acct_info = detect_account_type(party.account_value, party.account_type)
            party.account_length = acct_info['length']
            party.account_numeric_only = acct_info['is_numeric']
            party.is_clabe = acct_info['is_clabe']
            party.is_fedaba = acct_info['is_fedaba']
            party.is_chips_aba = acct_info['is_chips_aba']
            party.is_chips_uid = acct_info['is_chips_uid']
        
        # Check if BIC and IBAN countries match
        if party.bic_country and party.iban_country:
            party.bic_iban_country_match = (party.bic_country == party.iban_country)
        
        # Check if BIC country matches party country
        if party.bic_country and party.country:
            party.bic_party_country_match = (party.bic_country == party.country.upper())
        
        return party
    
    def _parse_bank_info(self, basic_payment: dict, features: IFMLFeatures):
        """Extract BankInfo field details."""
        bank_info = basic_payment.get('BankInfo', [])
        
        if isinstance(bank_info, dict):
            bank_info = [bank_info]
        
        features.has_bank_info = len(bank_info) > 0
        features.bank_info_count = len(bank_info)
    
    def _derive_cross_party_features(self, features: IFMLFeatures):
        """Derive features that span multiple parties."""
        # Get originator country
        orig_party = features.parties.get('OriginatingPartyInfo')
        if orig_party:
            features.originator_country = orig_party.country
        
        # Get beneficiary country (from CreditParty or BeneficiaryBank)
        credit_party = features.parties.get('CreditPartyInfo')
        benef_bank = features.parties.get('BeneficiaryBankInfo')
        
        if credit_party:
            features.beneficiary_country = (
                credit_party.country or 
                credit_party.mailing_country
            )
        if not features.beneficiary_country and benef_bank:
            features.beneficiary_country = benef_bank.country
        
        # Determine if cross-border
        if features.originator_country and features.beneficiary_country:
            features.is_cross_border = (
                features.originator_country != features.beneficiary_country
            )
        
        # Check for intermediary
        features.has_intermediary = 'IntermediaryBankInfo' in features.parties
        features.has_beneficiary_bank = 'BeneficiaryBankInfo' in features.parties
    
    def to_dict(self, features: IFMLFeatures) -> dict:
        """Convert features to a flat dictionary for ML consumption."""
        result = {
            'transaction_id': features.transaction_id,
            'source_code': features.source_code,
            'incoming_msg_type': features.incoming_msg_type,
            'incoming_format': features.incoming_format,
            'bank_operation_code': features.bank_operation_code,
            'location_id_bank': features.location_id_bank,
            'primary_currency': features.primary_currency,
            'primary_amount': features.primary_amount,
            'has_instructed_amount': features.has_instructed_amount,
            'instructed_amount': features.instructed_amount,
            'amount_mismatch': features.amount_mismatch,
            'is_cross_border': features.is_cross_border,
            'has_intermediary': features.has_intermediary,
            'has_beneficiary_bank': features.has_beneficiary_bank,
            'originator_country': features.originator_country,
            'beneficiary_country': features.beneficiary_country,
            'has_bank_info': features.has_bank_info,
            'bank_info_count': features.bank_info_count,
            'amount_count': len(features.amounts),
        }
        
        # Add party-specific features
        for party_type in self.PARTY_TYPES:
            prefix = self._party_type_to_prefix(party_type)
            party = features.parties.get(party_type)
            
            if party:
                result[f'{prefix}_present'] = True
                result[f'{prefix}_has_id'] = party.has_id
                result[f'{prefix}_has_bic'] = party.has_bic
                result[f'{prefix}_bic_length'] = party.bic_length
                result[f'{prefix}_bic_country'] = party.bic_country
                result[f'{prefix}_bic_valid_format'] = party.bic_valid_format
                result[f'{prefix}_bic_valid_country'] = party.bic_valid_country
                result[f'{prefix}_has_account'] = party.has_account
                result[f'{prefix}_account_type'] = party.account_type
                result[f'{prefix}_account_length'] = party.account_length
                result[f'{prefix}_account_numeric'] = party.account_numeric_only
                result[f'{prefix}_iban_country'] = party.iban_country
                result[f'{prefix}_iban_valid_format'] = party.iban_valid_format
                result[f'{prefix}_iban_checksum_valid'] = party.iban_checksum_valid
                result[f'{prefix}_bic_iban_match'] = party.bic_iban_country_match
                result[f'{prefix}_bic_party_country_match'] = party.bic_party_country_match
                result[f'{prefix}_is_clabe'] = party.is_clabe
                result[f'{prefix}_is_fedaba'] = party.is_fedaba
                result[f'{prefix}_is_chips_aba'] = party.is_chips_aba
                result[f'{prefix}_is_chips_uid'] = party.is_chips_uid
                result[f'{prefix}_country'] = party.country
                result[f'{prefix}_mailing_country'] = party.mailing_country
                result[f'{prefix}_residence_country'] = party.residence_country
                result[f'{prefix}_address_lines'] = party.address_line_count
                result[f'{prefix}_has_name'] = party.has_name
            else:
                result[f'{prefix}_present'] = False
                result[f'{prefix}_has_id'] = False
                result[f'{prefix}_has_bic'] = False
                result[f'{prefix}_bic_length'] = 0
                result[f'{prefix}_bic_country'] = None
                result[f'{prefix}_bic_valid_format'] = False
                result[f'{prefix}_bic_valid_country'] = False
                result[f'{prefix}_has_account'] = False
                result[f'{prefix}_account_type'] = None
                result[f'{prefix}_account_length'] = 0
                result[f'{prefix}_account_numeric'] = False
                result[f'{prefix}_iban_country'] = None
                result[f'{prefix}_iban_valid_format'] = False
                result[f'{prefix}_iban_checksum_valid'] = False
                result[f'{prefix}_bic_iban_match'] = None
                result[f'{prefix}_bic_party_country_match'] = None
                result[f'{prefix}_is_clabe'] = False
                result[f'{prefix}_is_fedaba'] = False
                result[f'{prefix}_is_chips_aba'] = False
                result[f'{prefix}_is_chips_uid'] = False
                result[f'{prefix}_country'] = None
                result[f'{prefix}_mailing_country'] = None
                result[f'{prefix}_residence_country'] = None
                result[f'{prefix}_address_lines'] = 0
                result[f'{prefix}_has_name'] = False
        
        return result
    
    def _party_type_to_prefix(self, party_type: str) -> str:
        """Convert party type to short prefix for feature names."""
        prefixes = {
            'OriginatingPartyInfo': 'orig',
            'SendingBankInfo': 'send',
            'DebitPartyInfo': 'dbt',
            'CreditPartyInfo': 'cdt',
            'IntermediaryBankInfo': 'intm',
            'BeneficiaryBankInfo': 'bnf',
            'AccountWithInstitution': 'acwi',
            'OrderingInstitution': 'ordi'
        }
        return prefixes.get(party_type, party_type.lower()[:4])


class IFMLResponseParser:
    """Parser for ACE Pelican IFML response messages with error codes."""
    
    def parse_file(self, filepath: str) -> Tuple[str, List[dict]]:
        """Parse response file and extract transaction ID and message codes."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return self.parse(data)
    
    def parse(self, data: dict) -> Tuple[Optional[str], List[dict]]:
        """
        Parse response and extract error codes.
        Returns (transaction_uid, list of code dictionaries).
        """
        # Navigate to Message
        message = self._find_message(data)
        
        if not message:
            return None, []
        
        # Extract transaction UID
        basic_payment = message.get('BasicPayment', {})
        transaction_uid = (
            basic_payment.get('TransactionUID') or 
            message.get('TransactionUID')
        )
        
        # Extract audit trail codes
        audit_trail = message.get('AuditTrail') or {}
        
        # Handle case where AuditTrail is at File level
        if not audit_trail:
            file_node = self._find_file(data)
            if file_node:
                audit_trail = file_node.get('AuditTrail', {})
        
        msg_status = audit_trail.get('MsgStatus', [])
        
        if isinstance(msg_status, dict):
            msg_status = [msg_status]
        
        codes = []
        for status in msg_status:
            info_data = status.get('InformationalData', '')
            party_short, party_full = self._extract_party_hint(info_data)
            codes.append({
                'code': status.get('Code'),
                'severity': status.get('Severity'),
                'info': info_data,
                'party_hint': party_short,  # Short code like CDTPTY, BNFBNK
                'party_type': party_full,   # Full type like CreditPartyInfo
            })
        
        return transaction_uid, codes
    
    def _find_message(self, data: dict) -> Optional[dict]:
        """Find Message node in response structure."""
        def safe_get(d, *keys):
            """Safely navigate nested dicts."""
            current = d
            for key in keys:
                if not isinstance(current, dict):
                    return None
                current = current.get(key)
                if current is None:
                    return None
            return current if isinstance(current, dict) else None
        
        patterns = [
            lambda d: safe_get(d, 'Response', 'IFML', 'File', 'Message'),
            lambda d: safe_get(d, 'IFML', 'File', 'Message'),
            lambda d: safe_get(d, 'File', 'Message'),
            lambda d: safe_get(d, 'Message'),
        ]
        
        for pattern in patterns:
            result = pattern(data)
            if result:
                return result
        
        return None
    
    def _find_file(self, data: dict) -> Optional[dict]:
        """Find File node in response structure."""
        def safe_get(d, *keys):
            """Safely navigate nested dicts."""
            current = d
            for key in keys:
                if not isinstance(current, dict):
                    return None
                current = current.get(key)
                if current is None:
                    return None
            return current if isinstance(current, dict) else None
        
        patterns = [
            lambda d: safe_get(d, 'Response', 'IFML', 'File'),
            lambda d: safe_get(d, 'IFML', 'File'),
            lambda d: safe_get(d, 'File'),
        ]
        
        for pattern in patterns:
            result = pattern(data)
            if result:
                return result
        
        return None
    
    def _extract_party_hint(self, info: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract party identifier from informational data.
        Returns (short_code, full_party_type) tuple.
        """
        # Common patterns: "BNFBNK ...", "CDTPTY ...", "BNPPTY ..."
        party_patterns = {
            'BNFBNK': 'BeneficiaryBankInfo',
            'BNFPTY': 'BeneficiaryPartyInfo',
            'BNPPTY': 'BeneficiaryPartyInfo',
            'CDTPTY': 'CreditPartyInfo',
            'DBTPTY': 'DebitPartyInfo',
            'ORGPTY': 'OriginatingPartyInfo',
            'INTBNK': 'IntermediaryBankInfo',
            'SNDBNK': 'SendingBankInfo',
            'ACWBNK': 'AccountWithInstitution',
            'ORDBNK': 'OrderingInstitution',
        }
        
        for pattern, party_type in party_patterns.items():
            if info.startswith(pattern):
                return pattern, party_type
        
        return None, None


if __name__ == '__main__':
    # Test with sample data
    import sys
    
    parser = IFMLParser(keep_raw=True)
    resp_parser = IFMLResponseParser()
    
    # Test parsing
    sample_request = {
        "2025092900000192": {
            "Request": {
                "IFML": {
                    "File": {
                        "Message": {
                            "BasicPayment": {
                                "TransactionID": "2025092900000XXX",
                                "SourceCode": "SWF",
                                "IncomingMsgType": "103",
                                "BankOperationCode": "CRED",
                                "PartyInfo": {
                                    "CreditPartyInfo": {
                                        "BasicPartyInfo": {
                                            "MailingCountry": "JP"
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    features = parser.parse(sample_request)
    print("Parsed features:")
    feature_dict = parser.to_dict(features)
    for k, v in feature_dict.items():
        if v is not None and v != False and v != 0:
            print(f"  {k}: {v}")
