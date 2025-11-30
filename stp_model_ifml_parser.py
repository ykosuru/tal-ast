"""
IFML Parser - Updated with fixes from debugging sessions.

Key fixes:
1. BeneficiaryPartyInf now correctly handled (maps to bnf prefix, extracts IBAN)
2. Type="S" recognized as SWIFT/BIC identifier
3. ID cleaning detection for 9019 prediction (spaces, dashes, special chars in IDs)
4. Plain string IBANs detected by pattern (not just Type="IBAN")
5. NCH extraction from AdrBankID with value "121" style
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path


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


def validate_bic(bic: str) -> Tuple[bool, bool]:
    """Validate BIC format and country code. Returns (format_valid, country_valid)"""
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
    """Validate IBAN format and checksum. Returns (format_valid, checksum_valid)"""
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
    """Validate US ABA routing number. Returns (format_valid, checksum_valid)"""
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


def looks_like_bic(s: str) -> bool:
    """Quick check if string looks like a BIC."""
    if not s or len(s) not in (8, 11):
        return False
    s = s.upper()
    return bool(re.match(r'^[A-Z]{4}[A-Z]{2}[A-Z0-9]{2,5}$', s))


def looks_like_iban(s: str) -> bool:
    """Quick check if string looks like an IBAN (handles spaces)."""
    if not s:
        return False
    # Remove spaces and dashes for check
    cleaned = s.upper().replace(' ', '').replace('-', '')
    if len(cleaned) < 15 or len(cleaned) > 34:
        return False
    return bool(re.match(r'^[A-Z]{2}[0-9]{2}[A-Z0-9]+$', cleaned))


def id_needs_cleaning(id_value: str) -> bool:
    """Check if ID contains non-alphanumeric characters that ACE will clean (9019)."""
    if not id_value:
        return False
    cleaned = re.sub(r'[^A-Za-z0-9]', '', id_value)
    return cleaned != id_value


@dataclass
class PartyInfo:
    """Normalized party information with all features for ML."""
    party_type: str
    
    # Basic presence flags
    present: bool = True
    has_id: bool = False
    has_bic: bool = False
    has_account: bool = False
    has_iban: bool = False
    has_name: bool = False
    has_nch: bool = False
    
    # Raw values
    id_value: Optional[str] = None
    id_type: Optional[str] = None
    bic: Optional[str] = None
    iban: Optional[str] = None
    account_value: Optional[str] = None
    account_type: Optional[str] = None
    nch_value: Optional[str] = None
    nch_type: Optional[str] = None
    
    # Validation results
    bic_valid_format: bool = False
    bic_valid_country: bool = False
    bic_country: Optional[str] = None
    bic_length: int = 0
    iban_valid_format: bool = False
    iban_checksum_valid: bool = False
    iban_country: Optional[str] = None
    nch_valid: bool = False
    fedaba_checksum_valid: bool = False
    
    # Country fields
    country: Optional[str] = None
    mailing_country: Optional[str] = None
    residence_country: Optional[str] = None
    address_country: Optional[str] = None
    
    # Match flags
    bic_iban_country_match: Optional[bool] = None
    bic_party_country_match: Optional[bool] = None
    
    # Address
    address_line_count: int = 0
    is_domestic: bool = False
    is_international: bool = False
    
    # Account details
    account_length: int = 0
    account_numeric_only: bool = False
    is_clabe: bool = False
    is_fedaba: bool = False
    is_chips_aba: bool = False
    is_chips_uid: bool = False
    
    # Flags
    bank_flag: Optional[str] = None
    charge_flag: Optional[str] = None
    has_adr_bank_id: bool = False
    adr_bank_id_type: Optional[str] = None
    
    # IBAN requirement
    needs_iban: bool = False
    nch_sources: int = 0
    nch_validation_applicable: bool = False
    
    # === 9XXX REPAIR DETECTION FEATURES ===
    # 9019: ID cleaning detection - ANY identifier with non-alphanumeric chars
    id_needs_cleaning: bool = False
    iban_needs_cleaning: bool = False
    any_id_needs_cleaning: bool = False  # Aggregated: any ID in this party needs cleaning
    
    # Account cleaning
    account_has_dirty_chars: bool = False
    account_has_spaces: bool = False
    account_has_special_chars: bool = False
    name_has_dirty_chars: bool = False
    
    # Derivation features
    is_iban_derivable: bool = False
    has_bban_in_iban_country: bool = False
    is_bic_derivable: bool = False
    has_nch_no_bic: bool = False
    has_iban_no_bic: bool = False
    
    # NCH repair
    nch_has_dirty_chars: bool = False
    nch_needs_formatting: bool = False
    
    # Multiple/duplicate
    has_multiple_ids: bool = False
    has_duplicate_info: bool = False
    
    # Formatting
    account_needs_length_fix: bool = False
    iban_needs_formatting: bool = False


@dataclass
class IFMLFeatures:
    """Complete feature set extracted from an IFML request."""
    transaction_id: Optional[str] = None
    transaction_uid: Optional[str] = None
    source_code: Optional[str] = None
    incoming_msg_type: Optional[str] = None
    incoming_format: Optional[str] = None
    bank_operation_code: Optional[str] = None
    location_id_bank: Optional[str] = None
    
    primary_currency: Optional[str] = None
    primary_amount: float = 0.0
    has_instructed_amount: bool = False
    instructed_amount: float = 0.0
    amount_mismatch: bool = False
    
    parties: Dict[str, PartyInfo] = field(default_factory=dict)
    
    is_cross_border: bool = False
    has_intermediary: bool = False
    has_beneficiary_bank: bool = False
    originator_country: Optional[str] = None
    beneficiary_country: Optional[str] = None
    
    has_bank_info: bool = False
    bank_info_count: int = 0
    
    raw_json: Optional[dict] = None


class IFMLParser:
    """
    Parser for ACE Pelican IFML payment messages.
    Extracts structured features suitable for ML model training.
    """
    
    # Standard party types (Info suffix)
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
    
    # Alternate naming (Inf suffix) - maps to standard names
    PARTY_TYPE_ALT_MAP = {
        'OriginatingPartyInf': 'OriginatingPartyInfo',
        'SendingBankInf': 'SendingBankInfo',
        'DebitPartyInf': 'DebitPartyInfo',
        'CreditPartyInf': 'CreditPartyInfo',
        'IntermediaryBankInf': 'IntermediaryBankInfo',
        'BeneficiaryBankInf': 'BeneficiaryBankInfo',
        'BeneficiaryPartyInf': 'BeneficiaryBankInfo',  # Maps to same as BeneficiaryBank
    }
    
    # Party type to short prefix mapping
    PARTY_PREFIXES = {
        'OriginatingPartyInfo': 'orig',
        'SendingBankInfo': 'send',
        'DebitPartyInfo': 'dbt',
        'CreditPartyInfo': 'cdt',
        'IntermediaryBankInfo': 'intm',
        'BeneficiaryBankInfo': 'bnf',
        'AccountWithInstitution': 'acwi',
        'OrderingInstitution': 'ordi'
    }
    
    # Countries where IBAN is typically required
    IBAN_COUNTRIES = {
        'DE', 'FR', 'GB', 'ES', 'IT', 'NL', 'BE', 'AT', 'CH', 'SE', 'NO', 'DK', 'FI',
        'PL', 'PT', 'IE', 'GR', 'CZ', 'HU', 'RO', 'SK', 'LU', 'HR', 'SI', 'BG', 'LT',
        'LV', 'EE', 'CY', 'MT', 'AE', 'SA', 'QA', 'KW', 'BH', 'IL', 'TR', 'EG', 'MA'
    }
    
    def __init__(self, keep_raw: bool = False):
        self.keep_raw = keep_raw
    
    def parse_file(self, filepath: str) -> IFMLFeatures:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return self.parse(data)
    
    def parse(self, data: dict) -> IFMLFeatures:
        """Parse IFML data structure and extract features."""
        features = IFMLFeatures()
        
        if self.keep_raw:
            features.raw_json = data
        
        basic_payment = self._find_basic_payment(data)
        if not basic_payment:
            return features
        
        # Transaction info
        features.transaction_id = basic_payment.get('TransactionID')
        features.transaction_uid = basic_payment.get('TransactionUID')
        
        # Source/format info
        features.source_code = basic_payment.get('SourceCode')
        features.incoming_msg_type = basic_payment.get('IncomingMsgType')
        features.incoming_format = basic_payment.get('IncomingFormat')
        features.bank_operation_code = basic_payment.get('BankOperationCode')
        
        # Location
        location_info = basic_payment.get('LocationInfo') or basic_payment.get('LocationInf') or {}
        features.location_id_bank = location_info.get('IDBank')
        
        # Monetary amounts
        self._parse_monetary_amounts(basic_payment, features)
        
        # Party information
        self._parse_party_info(basic_payment, features)
        
        # Bank info
        self._parse_bank_info(basic_payment, features)
        
        # Cross-party features
        self._derive_cross_party_features(features)
        
        return features
    
    def _find_basic_payment(self, data: dict) -> Optional[dict]:
        """Navigate nested structure to find BasicPayment node."""
        patterns = [
            lambda d: self._navigate_path(d, ['*', 'Request', 'IFML', 'File', 'Message', 'BasicPayment']),
            lambda d: self._navigate_path(d, ['Request', 'IFML', 'File', 'Message', 'BasicPayment']),
            lambda d: self._navigate_path(d, ['IFML', 'File', 'Message', 'BasicPayment']),
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
        """Extract monetary amounts."""
        amounts = basic_payment.get('MonetaryAmount', [])
        if isinstance(amounts, dict):
            amounts = [amounts]
        
        for amt in amounts:
            amt_type = amt.get('@Type') or amt.get('Type') or 'Unknown'
            currency = amt.get('@Currency') or amt.get('Currency')
            amount_val = self._parse_amount(amt.get('Amount', '0'))
            
            if amt_type == 'Amount':
                features.primary_currency = currency
                features.primary_amount = amount_val
            
            if amt_type == 'RelatedAmt' or amt.get('SubType') == 'INSTRUCTED AMOUNT':
                features.has_instructed_amount = True
                features.instructed_amount = amount_val
        
        if features.has_instructed_amount and features.primary_amount > 0:
            features.amount_mismatch = abs(features.primary_amount - features.instructed_amount) > 0.01
    
    def _parse_amount(self, amount_str: str) -> float:
        try:
            return float(str(amount_str).replace(',', ''))
        except (ValueError, TypeError):
            return 0.0
    
    def _parse_party_info(self, basic_payment: dict, features: IFMLFeatures):
        """Extract party information for all party types."""
        party_info = basic_payment.get('PartyInfo') or basic_payment.get('PartyInf') or {}
        if not isinstance(party_info, dict):
            return
        
        # Process standard naming (Info)
        for party_type in self.PARTY_TYPES:
            party_data = party_info.get(party_type)
            if party_data:
                if isinstance(party_data, list):
                    party_data = party_data[0] if party_data else None
                if party_data and isinstance(party_data, dict):
                    parsed_party = self._parse_single_party(party_type, party_data)
                    features.parties[party_type] = parsed_party
        
        # Process alternate naming (Inf) - merge into existing if needed
        for alt_type, standard_type in self.PARTY_TYPE_ALT_MAP.items():
            party_data = party_info.get(alt_type)
            if party_data:
                if isinstance(party_data, list):
                    party_data = party_data[0] if party_data else None
                if party_data and isinstance(party_data, dict):
                    if standard_type in features.parties:
                        # Merge into existing party (e.g., BeneficiaryPartyInf into BeneficiaryBankInfo)
                        self._merge_party_data(features.parties[standard_type], party_data)
                    else:
                        parsed_party = self._parse_single_party(standard_type, party_data)
                        features.parties[standard_type] = parsed_party
    
    def _parse_single_party(self, party_type: str, party_data: dict) -> PartyInfo:
        """Parse a single party's information."""
        party = PartyInfo(party_type=party_type)
        
        if not isinstance(party_data, dict):
            return party
        
        # Get basic info - handle multiple naming conventions
        basic_info = (
            party_data.get('BasicPartyInfo') or 
            party_data.get('BasicPartyInf') or
            party_data.get('BasicIDInfo') or 
            party_data.get('AccountPartyInfo') or
            party_data.get('AccountPartyInf') or
            party_data
        )
        
        if isinstance(basic_info, list):
            basic_info = basic_info[0] if basic_info else {}
        if not isinstance(basic_info, dict):
            return party
        
        # Extract ID field - KEY FIX: handle Type="S" as BIC
        self._extract_id_field(party, basic_info)
        
        # Extract account info
        self._extract_account_info(party, basic_info)
        
        # Extract AdrBankID (NCH source)
        self._extract_adr_bank_id(party, basic_info)
        
        # Extract countries
        party.country = basic_info.get('Country') or party_data.get('Country')
        party.mailing_country = basic_info.get('MailingCountry') or party_data.get('MailingCountry')
        party.residence_country = basic_info.get('ResidenceCountry') or party_data.get('ResidenceCountry')
        
        # Extract address info
        self._extract_address_info(party, basic_info, party_data)
        
        # Flags
        party.bank_flag = basic_info.get('BankFlag')
        party.charge_flag = basic_info.get('ChargeFlag')
        
        # Validate BIC
        if party.bic:
            party.bic_length = len(party.bic)
            if len(party.bic) >= 6:
                party.bic_country = party.bic[4:6].upper()
            party.bic_valid_format, party.bic_valid_country = validate_bic(party.bic)
        
        # Validate IBAN
        if party.iban:
            if len(party.iban) >= 2:
                # Get country from cleaned IBAN
                cleaned_iban = party.iban.upper().replace(' ', '').replace('-', '')
                party.iban_country = cleaned_iban[:2]
            party.iban_valid_format, party.iban_checksum_valid = validate_iban(party.iban)
            party.has_iban = True
        
        # Check country matches
        if party.bic_country and party.iban_country:
            party.bic_iban_country_match = (party.bic_country == party.iban_country)
        if party.bic_country and party.country:
            party.bic_party_country_match = (party.bic_country == party.country.upper())
        
        # Determine if IBAN is needed
        effective_country = party.bic_country or party.iban_country or party.country or party.address_country
        if effective_country and effective_country.upper() in self.IBAN_COUNTRIES:
            party.needs_iban = True
        
        # NCH validation applicability (US domestic only)
        is_us_party = effective_country and effective_country.upper() == 'US'
        has_us_routing = party.is_fedaba or party.is_chips_aba or party.nch_type in ['FEDABA', 'CHIPS']
        party.nch_validation_applicable = is_us_party or (not effective_country and has_us_routing)
        
        # Count NCH sources
        nch_sources = 0
        if party.is_fedaba or party.is_chips_aba:
            nch_sources += 1
        if party.has_adr_bank_id and party.adr_bank_id_type in ['FEDABA', 'FW', 'FEDWIRE', 'CHIPS', 'CH']:
            nch_sources += 1
        party.nch_sources = nch_sources
        
        # === REPAIR DETECTION FEATURES ===
        self._detect_repair_features(party, basic_info, effective_country)
        
        return party
    
    def _extract_id_field(self, party: PartyInfo, basic_info: dict):
        """Extract ID field with proper type detection."""
        id_field = basic_info.get('ID')
        if not id_field:
            return
        
        party.has_id = True
        
        if isinstance(id_field, dict):
            # Handle typed ID: {Type: "S", text: "REVOGB2L"} or {Type: "IBAN", text: "GB..."}
            id_type = (id_field.get('Type') or id_field.get('@Type') or '').upper()
            id_text = id_field.get('text') or id_field.get('#text') or ''
            
            party.id_value = id_text
            party.id_type = id_type
            
            # KEY FIX: Type="S" means SWIFT/BIC
            if id_type in ('S', 'SWIFT', 'BIC'):
                party.has_bic = True
                party.bic = id_text
            elif id_type == 'IBAN':
                party.has_iban = True
                party.iban = id_text
            elif looks_like_bic(id_text):
                party.has_bic = True
                party.bic = id_text
            elif looks_like_iban(id_text):
                party.has_iban = True
                party.iban = id_text
        else:
            # Plain string ID
            id_text = str(id_field)
            party.id_value = id_text
            
            if looks_like_iban(id_text):
                party.has_iban = True
                party.iban = id_text
            elif looks_like_bic(id_text):
                party.has_bic = True
                party.bic = id_text
        
        # KEY FIX: Detect if ID needs cleaning (for 9019)
        if party.id_value:
            party.id_needs_cleaning = id_needs_cleaning(party.id_value)
        if party.iban:
            party.iban_needs_cleaning = id_needs_cleaning(party.iban)
    
    def _extract_account_info(self, party: PartyInfo, basic_info: dict):
        """Extract account information."""
        acct_info = basic_info.get('AcctIDInfo') or basic_info.get('AcctIDInf')
        if acct_info:
            party.has_account = True
            if isinstance(acct_info, dict):
                acct_id = acct_info.get('ID', {})
                if isinstance(acct_id, dict):
                    party.account_type = acct_id.get('@Type') or acct_id.get('Type')
                    party.account_value = acct_id.get('#text') or acct_id.get('text')
        
        if party.account_value:
            acct = party.account_value
            party.account_length = len(acct)
            party.account_numeric_only = acct.replace(' ', '').replace('-', '').isdigit()
            
            # Detect account types
            clean_acct = acct.replace(' ', '').replace('-', '')
            if len(clean_acct) == 18 and clean_acct.isdigit():
                party.is_clabe = True
            if len(clean_acct) == 9 and clean_acct.isdigit():
                party.is_fedaba = True
                party.is_chips_aba = True
            if len(clean_acct) == 6 and clean_acct.isdigit():
                party.is_chips_uid = True
            
            # Account cleaning detection
            party.account_has_dirty_chars = bool(re.search(r'[^a-zA-Z0-9]', acct))
            party.account_has_spaces = ' ' in acct
            party.account_has_special_chars = bool(re.search(r'[-/.\\]', acct))
    
    def _extract_adr_bank_id(self, party: PartyInfo, basic_info: dict):
        """Extract AdrBankID (NCH source)."""
        adr_bank_id = basic_info.get('AdrBankID')
        if not adr_bank_id:
            return
        
        party.has_adr_bank_id = True
        
        if isinstance(adr_bank_id, dict):
            party.adr_bank_id_type = adr_bank_id.get('@Type') or adr_bank_id.get('Type')
            adr_value = adr_bank_id.get('#text') or adr_bank_id.get('text')
        else:
            # Plain value like "121"
            adr_value = str(adr_bank_id)
            party.adr_bank_id_type = None
        
        if adr_value:
            # Detect NCH type from value or explicit type
            if party.adr_bank_id_type in ['FEDABA', 'FW', 'FEDWIRE']:
                party.has_nch = True
                party.nch_type = 'FEDABA'
                party.nch_value = adr_value
                fmt_valid, cksum_valid = validate_fedaba(adr_value)
                party.nch_valid = fmt_valid
                party.fedaba_checksum_valid = cksum_valid
            elif party.adr_bank_id_type in ['CHIPS', 'CH', 'CHIPSABA']:
                party.has_nch = True
                party.nch_type = 'CHIPS'
                party.nch_value = adr_value
                party.nch_valid = len(adr_value) == 6 and adr_value.isdigit()
            elif party.adr_bank_id_type in ['SORTCODE', 'SC', 'UKSC']:
                party.has_nch = True
                party.nch_type = 'SORTCODE'
                party.nch_value = adr_value
                party.nch_valid = len(adr_value.replace('-', '')) == 6
            elif adr_value.isdigit():
                # Auto-detect from length
                if len(adr_value) == 9:
                    party.has_nch = True
                    party.nch_type = 'FEDABA'
                    party.nch_value = adr_value
                    fmt_valid, cksum_valid = validate_fedaba(adr_value)
                    party.nch_valid = fmt_valid
                    party.fedaba_checksum_valid = cksum_valid
                elif len(adr_value) == 6:
                    party.has_nch = True
                    party.nch_type = 'CHIPS'
                    party.nch_value = adr_value
                    party.nch_valid = True
                elif len(adr_value) == 3:
                    # Could be a bank identifier like "121" - record but don't validate as NCH
                    party.nch_value = adr_value
    
    def _extract_address_info(self, party: PartyInfo, basic_info: dict, party_data: dict):
        """Extract address information."""
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
                
                # Extract country from address line 3 or last line
                party.address_country = self._extract_country_from_address(address_info)
                if party.address_country:
                    party.is_domestic = (party.address_country == 'US')
                    party.is_international = (party.address_country != 'US')
            elif isinstance(address_info, dict):
                party.address_line_count = 1
                party.has_name = True
    
    def _extract_country_from_address(self, address_info: list) -> Optional[str]:
        """Extract country code from address lines."""
        if not address_info:
            return None
        
        # Look for line 3 (Seq: "3")
        target_line = None
        for item in address_info:
            if isinstance(item, dict):
                seq = item.get('Seq') or item.get('seq')
                if seq == '3' or seq == 3:
                    target_line = item.get('text') or item.get('Text') or ''
                    break
        
        # If no line 3, use last line
        if not target_line and address_info:
            last_item = address_info[-1]
            if isinstance(last_item, dict):
                target_line = last_item.get('text') or last_item.get('Text') or ''
            elif isinstance(last_item, str):
                target_line = last_item
        
        if not target_line:
            return None
        
        target_line = target_line.upper().strip()
        
        # Look for country code
        parts = [p.strip() for p in target_line.replace(',', ' , ').split()]
        for i, part in enumerate(parts):
            clean_part = ''.join(c for c in part if c.isalpha())
            if len(clean_part) == 2 and clean_part in VALID_COUNTRY_CODES:
                return clean_part
        
        return None
    
    def _detect_repair_features(self, party: PartyInfo, basic_info: dict, effective_country: Optional[str]):
        """Detect features that indicate repair codes (9XXX) will fire."""
        
        # 9019: Aggregate ID cleaning flag
        party.any_id_needs_cleaning = party.id_needs_cleaning or party.iban_needs_cleaning
        
        # IBAN derivation (9004, 9007)
        if effective_country and party.has_account and not party.has_iban:
            if effective_country.upper() in self.IBAN_COUNTRIES:
                party.is_iban_derivable = True
                party.has_bban_in_iban_country = (party.account_type in ['BBAN', 'D', 'DDA', None])
        
        # BIC derivation (9005, 9008)
        if not party.has_bic:
            if party.has_nch or party.is_fedaba or party.is_chips_aba:
                party.is_bic_derivable = True
                party.has_nch_no_bic = True
            if party.has_iban:
                party.is_bic_derivable = True
                party.has_iban_no_bic = True
        
        # NCH repair (9000, 9021, 9028)
        if party.nch_value:
            party.nch_has_dirty_chars = bool(re.search(r'[^0-9]', party.nch_value))
            if party.nch_type == 'FEDABA' and len(party.nch_value.replace('-', '').replace(' ', '')) != 9:
                party.nch_needs_formatting = True
            elif party.nch_type == 'CHIPS' and len(party.nch_value.replace('-', '').replace(' ', '')) != 6:
                party.nch_needs_formatting = True
        
        # Multiple IDs (9017, 9018)
        id_info = basic_info.get('ID', {})
        if isinstance(id_info, list) and len(id_info) > 1:
            party.has_multiple_ids = True
            party.has_duplicate_info = True
        
        # Account length issues (9022)
        if party.account_value and party.account_type:
            acct_len = len(party.account_value.replace(' ', '').replace('-', ''))
            expected_lengths = {
                'IBAN': (15, 34),
                'CLABE': (18, 18),
                'FEDABA': (9, 9),
            }
            if party.account_type in expected_lengths:
                min_len, max_len = expected_lengths[party.account_type]
                if acct_len < min_len or acct_len > max_len:
                    party.account_needs_length_fix = True
        
        # IBAN formatting (9006, 9012)
        if party.has_iban and party.iban:
            iban = party.iban
            # Needs formatting if has spaces, lowercase, or wrong length
            if ' ' in iban or '-' in iban:
                party.iban_needs_formatting = True
            if iban != iban.upper():
                party.iban_needs_formatting = True
    
    def _merge_party_data(self, existing_party: PartyInfo, additional_data: dict):
        """Merge additional party data (e.g., BeneficiaryPartyInf into BeneficiaryBankInfo)."""
        basic_info = (
            additional_data.get('BasicPartyInfo') or 
            additional_data.get('BasicPartyInf') or
            additional_data
        )
        
        if isinstance(basic_info, list):
            basic_info = basic_info[0] if basic_info else {}
        if not isinstance(basic_info, dict):
            return
        
        # Extract ID if we don't already have an IBAN
        id_field = basic_info.get('ID')
        if id_field and not existing_party.has_iban:
            if isinstance(id_field, dict):
                id_text = id_field.get('text') or id_field.get('#text') or ''
            else:
                id_text = str(id_field)
            
            if looks_like_iban(id_text):
                existing_party.has_iban = True
                existing_party.iban = id_text
                existing_party.iban_needs_cleaning = id_needs_cleaning(id_text)
                
                # Validate
                if len(id_text) >= 2:
                    cleaned_iban = id_text.upper().replace(' ', '').replace('-', '')
                    existing_party.iban_country = cleaned_iban[:2]
                existing_party.iban_valid_format, existing_party.iban_checksum_valid = validate_iban(id_text)
                
                # Update match flags
                if existing_party.bic_country and existing_party.iban_country:
                    existing_party.bic_iban_country_match = (existing_party.bic_country == existing_party.iban_country)
            
            # Also check for ID cleaning
            if id_text and not existing_party.id_needs_cleaning:
                existing_party.id_needs_cleaning = id_needs_cleaning(id_text)
    
    def _parse_bank_info(self, basic_payment: dict, features: IFMLFeatures):
        """Extract BankInf information."""
        bank_info = basic_payment.get('BankInf') or basic_payment.get('BankInfo')
        if bank_info:
            features.has_bank_info = True
            if isinstance(bank_info, list):
                features.bank_info_count = len(bank_info)
            else:
                features.bank_info_count = 1
    
    def _derive_cross_party_features(self, features: IFMLFeatures):
        """Derive features that span multiple parties."""
        # Originator country
        if 'OriginatingPartyInfo' in features.parties:
            orig = features.parties['OriginatingPartyInfo']
            features.originator_country = orig.country or orig.address_country or orig.bic_country
        
        # Beneficiary country
        for party_type in ['BeneficiaryBankInfo', 'CreditPartyInfo']:
            if party_type in features.parties:
                party = features.parties[party_type]
                features.beneficiary_country = (
                    party.country or party.address_country or 
                    party.bic_country or party.iban_country
                )
                if features.beneficiary_country:
                    break
        
        # Cross-border detection
        if features.originator_country and features.beneficiary_country:
            features.is_cross_border = (features.originator_country != features.beneficiary_country)
        
        # Intermediary present
        features.has_intermediary = 'IntermediaryBankInfo' in features.parties
        
        # Beneficiary bank present
        features.has_beneficiary_bank = 'BeneficiaryBankInfo' in features.parties
    
    def to_dict(self, features: IFMLFeatures) -> dict:
        """Convert features to flat dictionary for ML."""
        result = {
            'transaction_id': features.transaction_id,
            'transaction_uid': features.transaction_uid,
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
        }
        
        # Flatten party features
        for party_type, party in features.parties.items():
            prefix = self.PARTY_PREFIXES.get(party_type, party_type.lower()[:4])
            party_dict = asdict(party)
            for key, value in party_dict.items():
                if key != 'party_type':
                    result[f'{prefix}_{key}'] = value
        
        # Add empty party placeholders
        for party_type in self.PARTY_TYPES:
            if party_type not in features.parties:
                prefix = self.PARTY_PREFIXES.get(party_type, party_type.lower()[:4])
                result[f'{prefix}_present'] = False
        
        return result


if __name__ == '__main__':
    # Test parsing
    parser = IFMLParser(keep_raw=True)


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
    # Run test
    parser = IFMLParser(keep_raw=True)
    
    # Test with sample that has BeneficiaryPartyInf with IBAN with spaces
    sample = {
        "Request": {
            "IFML": {
                "File": {
                    "Message": {
                        "BasicPayment": {
                            "TransactionUID": "TEST123",
                            "SourceCode": "SWF",
                            "PartyInf": {
                                "BeneficiaryBankInf": {
                                    "BasicPartyInf": {
                                        "AdrBankID": "121",
                                        "ID": {
                                            "Type": "S",
                                            "text": "REVOGB2L"
                                        }
                                    }
                                },
                                "BeneficiaryPartyInf": {
                                    "BasicPartyInf": {
                                        "ID": "GB14 REVO 0099 7044 3094 24"
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    features = parser.parse(sample)
    feature_dict = parser.to_dict(features)
    
    print("Parsed features:")
    for k, v in sorted(feature_dict.items()):
        if v is not None and v != False and v != 0 and v != '':
            print(f"  {k}: {v}")
    
    # Check key fixes
    print("\n=== Key Fixes Verification ===")
    print(f"bnf_has_bic: {feature_dict.get('bnf_has_bic')}")
    print(f"bnf_bic: {feature_dict.get('bnf_bic')}")
    print(f"bnf_has_iban: {feature_dict.get('bnf_has_iban')}")
    print(f"bnf_iban: {feature_dict.get('bnf_iban')}")
    print(f"bnf_id_needs_cleaning: {feature_dict.get('bnf_id_needs_cleaning')}")
    print(f"bnf_iban_needs_cleaning: {feature_dict.get('bnf_iban_needs_cleaning')}")
