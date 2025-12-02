"""
IFML Parser v3 - Enhanced with 8852 Attribute Length Validation
================================================================

Added features for 8852 (Incorrect length of attribute) detection:
- Field length tracking for all key attributes
- Maximum length validation based on ACE/IFML schema rules
- Detailed length violation reporting

Schema-based maximum lengths (from IFML/ISO 20022 standards):
- Name fields: 35 characters per line, 140 total
- Address lines: 35 characters each, max 4 lines
- BIC: 8 or 11 characters
- IBAN: Country-specific (15-34 chars)
- Currency: 3 characters (ISO 4217)
- Country: 2 characters (ISO 3166-1 alpha-2)
- Reference/ID: 35 characters typically
- Free text fields: 35-140 characters depending on field
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path


# =============================================================================
# CONSTANTS
# =============================================================================

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

# =============================================================================
# 8852: ATTRIBUTE LENGTH CONSTRAINTS (ACE/IFML Schema)
# =============================================================================

# Maximum lengths for IFML attributes based on ACE Pelican schema
# These are derived from ISO 20022 and SWIFT standards
ATTRIBUTE_MAX_LENGTHS = {
    # Party identification
    'ID': 35,                    # Party identifier
    'BIC': 11,                   # BIC/SWIFT code (8 or 11)
    'IBAN': 34,                  # IBAN (country-specific, max 34)
    'AccountNumber': 35,         # Generic account number
    'AdrBankID': 35,             # Address bank identifier
    
    # Name and address
    'Name': 140,                 # Party name (4 lines x 35)
    'AddressLine': 35,           # Single address line
    'AddressLineCount': 4,       # Max 4 address lines
    'City': 35,                  # City name
    'PostalCode': 16,            # Postal/ZIP code
    'Country': 2,                # ISO country code
    
    # Transaction info
    'TransactionID': 35,         # Transaction identifier
    'TransactionUID': 36,        # Transaction UID (UUID format)
    'EndToEndID': 35,            # End-to-end reference
    'InstructionID': 35,         # Instruction ID
    'RemittanceInfo': 140,       # Remittance information (4x35)
    'PaymentPurpose': 35,        # Payment purpose code
    
    # Monetary
    'Currency': 3,               # ISO 4217 currency code
    'Amount': 18,                # Amount (15 digits + decimal + sign)
    
    # Free text fields
    'InformationalData': 280,    # Informational data (8x35)
    'SenderReference': 16,       # Sender's reference (SWIFT)
    'RelatedReference': 16,      # Related reference
    'BankOperationCode': 4,      # Bank operation code
    'InstructionCode': 35,       # Instruction code
    
    # WireKey fields
    'WireKeyID': 35,             # WireKey identifier
    'WireKeyBIC': 11,            # WireKey BIC
}

# Fields that must be checked for 8852 at party level
PARTY_LENGTH_FIELDS = ['ID', 'BIC', 'AccountNumber', 'AdrBankID', 'Name']

# Fields with line-by-line length limits
LINE_LIMITED_FIELDS = {
    'AddressInfo': 35,           # Each line max 35 chars
    'Name': 35,                  # Name can span multiple lines, each max 35
    'RemittanceInfo': 35,        # Remittance info per line
}


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
    
    if len(bic) not in (8, 11):
        return False, False
    
    pattern = r'^[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}([A-Z0-9]{3})?$'
    format_valid = bool(re.match(pattern, bic))
    
    country_code = bic[4:6] if len(bic) >= 6 else ''
    country_valid = country_code in VALID_COUNTRY_CODES
    
    return format_valid, country_valid


def validate_iban(iban: str) -> Tuple[bool, bool]:
    """
    Validate IBAN format and checksum.
    Returns (format_valid, checksum_valid)
    """
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
    """Validate US ABA routing number."""
    if not aba:
        return False, False
    
    aba = aba.strip()
    
    if len(aba) != 9 or not aba.isdigit():
        return False, False
    
    format_valid = True
    
    try:
        weights = [3, 7, 1, 3, 7, 1, 3, 7, 1]
        total = sum(int(d) * w for d, w in zip(aba, weights))
        checksum_valid = (total % 10 == 0)
    except (ValueError, TypeError):
        checksum_valid = False
    
    return format_valid, checksum_valid


def extract_country_from_address(address_info: list) -> Tuple[Optional[str], bool, bool]:
    """Extract country code from AddressInf list."""
    if not address_info or not isinstance(address_info, list):
        return None, False, False
    
    target_line = None
    
    for item in address_info:
        if isinstance(item, dict):
            seq = item.get('Seq') or item.get('seq')
            if seq == '3' or seq == 3:
                target_line = item.get('text') or item.get('Text') or ''
                break
    
    if not target_line and address_info:
        last_item = address_info[-1]
        if isinstance(last_item, dict):
            target_line = last_item.get('text') or last_item.get('Text') or ''
        elif isinstance(last_item, str):
            target_line = last_item
    
    if not target_line:
        return None, False, False
    
    target_line = target_line.upper().strip()
    
    parts = [p.strip() for p in target_line.replace(',', ' , ').split()]
    
    found_country = None
    for i, part in enumerate(parts):
        clean_part = ''.join(c for c in part if c.isalpha())
        if len(clean_part) == 2 and clean_part in VALID_COUNTRY_CODES:
            found_country = clean_part
            if i > 0 and parts[i-1] == ',':
                break
    
    if not found_country:
        words = target_line.split()
        if words:
            last_word = ''.join(c for c in words[-1] if c.isalpha())
            if len(last_word) == 2 and last_word in VALID_COUNTRY_CODES:
                found_country = last_word
    
    if found_country:
        is_domestic = (found_country == 'US')
        is_international = (found_country != 'US')
        return found_country, is_domestic, is_international
    
    return None, False, False


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class LengthViolation:
    """Records a single attribute length violation for 8852."""
    field_name: str              # Name of the field
    actual_length: int           # Actual length of the value
    max_length: int              # Maximum allowed length
    value_preview: str           # First 20 chars of value for debugging
    party_type: Optional[str]    # Which party (if applicable)
    line_number: Optional[int]   # Line number for multi-line fields


@dataclass
class PartyInfo:
    """Normalized party information with length tracking."""
    party_type: str
    has_id: bool = False
    id_value: Optional[str] = None
    id_type: Optional[str] = None
    id_length: int = 0                    # NEW: Track ID length
    has_bic: bool = False
    bic: Optional[str] = None
    bic_value: Optional[str] = None
    bic_length: int = 0
    bic_country: Optional[str] = None
    bic_valid_format: bool = False
    bic_valid_country: bool = False
    has_account: bool = False
    account_type: Optional[str] = None
    account_value: Optional[str] = None
    account_length: int = 0
    account_numeric_only: bool = False
    iban_country: Optional[str] = None
    iban_valid_format: bool = False
    iban_checksum_valid: bool = False
    bic_iban_country_match: Optional[bool] = None
    bic_party_country_match: Optional[bool] = None
    is_clabe: bool = False
    is_fedaba: bool = False
    is_chips_aba: bool = False
    is_chips_uid: bool = False
    country: Optional[str] = None
    mailing_country: Optional[str] = None
    residence_country: Optional[str] = None
    address_line_count: int = 0
    address_lines: List[str] = None
    has_name: bool = False
    name_value: Optional[str] = None      # NEW: Track name value
    name_length: int = 0                  # NEW: Track name length
    bank_flag: Optional[str] = None
    charge_flag: Optional[str] = None
    address_country: Optional[str] = None
    is_domestic: bool = False
    is_international: bool = False
    has_nch: bool = False
    nch_type: Optional[str] = None
    nch_value: Optional[str] = None
    nch_valid: bool = False
    fedaba_checksum_valid: bool = False
    has_adr_bank_id: bool = False
    adr_bank_id_type: Optional[str] = None
    adr_bank_id_value: Optional[str] = None  # NEW: Track AdrBankID value
    adr_bank_id_length: int = 0              # NEW: Track AdrBankID length
    has_wirekey_bic: bool = False
    wirekey_bic: Optional[str] = None
    wirekey_bic_country: Optional[str] = None
    needs_iban: bool = False
    has_iban: bool = False
    nch_sources: int = 0
    nch_validation_applicable: bool = False
    
    # Repair detection features
    account_has_dirty_chars: bool = False
    name_has_dirty_chars: bool = False
    account_has_spaces: bool = False
    account_has_special_chars: bool = False
    is_iban_derivable: bool = False
    has_bban_in_iban_country: bool = False
    is_bic_derivable: bool = False
    has_nch_no_bic: bool = False
    has_iban_no_bic: bool = False
    nch_has_dirty_chars: bool = False
    nch_needs_formatting: bool = False
    has_multiple_ids: bool = False
    has_duplicate_info: bool = False
    account_needs_length_fix: bool = False
    iban_needs_formatting: bool = False
    
    # Compound ID features
    id_has_slash: bool = False
    id_is_compound: bool = False
    id_compound_parts: int = 0
    id_has_bic_pattern: bool = False
    id_has_nch_pattern: bool = False
    id_has_bic_and_nch: bool = False
    
    # === 8852: LENGTH VIOLATION FEATURES ===
    has_length_violation: bool = False           # Any length violation
    id_too_long: bool = False                    # ID exceeds max length
    bic_too_long: bool = False                   # BIC exceeds 11 chars
    account_too_long: bool = False               # Account exceeds max
    name_too_long: bool = False                  # Name exceeds max per line
    address_line_too_long: bool = False          # Any address line too long
    adr_bank_id_too_long: bool = False           # AdrBankID too long
    length_violations: List[LengthViolation] = None  # Detailed violations
    
    def __post_init__(self):
        if self.address_lines is None:
            self.address_lines = []
        if self.length_violations is None:
            self.length_violations = []
        if self.bic and not self.bic_value:
            self.bic_value = self.bic


@dataclass 
class MonetaryInfo:
    """Normalized monetary amount information."""
    amount_type: str
    currency: Optional[str] = None
    amount: float = 0.0
    currency_length: int = 0          # NEW: Track currency length
    amount_str_length: int = 0        # NEW: Track amount string length


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
    
    # Monetary
    amounts: List[MonetaryInfo] = field(default_factory=list)
    primary_currency: Optional[str] = None
    primary_amount: float = 0.0
    has_instructed_amount: bool = False
    instructed_amount: float = 0.0
    amount_mismatch: bool = False
    
    # Parties
    parties: Dict[str, PartyInfo] = field(default_factory=dict)
    
    # Cross-party features
    is_cross_border: bool = False
    has_intermediary: bool = False
    has_beneficiary_bank: bool = False
    originator_country: Optional[str] = None
    beneficiary_country: Optional[str] = None
    has_bank_info: bool = False
    bank_info_count: int = 0
    
    # Raw data
    raw_json: Optional[dict] = None
    
    # Intermediary features
    intm_count: int = 0
    intm_has_multiple: bool = False
    intm_entries_share_adr_bank_id: bool = False
    intm_entries_share_country: bool = False
    intm_entries_share_bic_prefix: bool = False
    intm_has_redundant_info: bool = False
    intm_types_present: List[str] = field(default_factory=list)
    intm_has_multiple_types: bool = False
    
    # === 8852: GLOBAL LENGTH VIOLATION FEATURES ===
    has_any_length_violation: bool = False       # Any violation in message
    length_violation_count: int = 0              # Total number of violations
    transaction_id_too_long: bool = False        # TransactionID > 35
    transaction_uid_too_long: bool = False       # TransactionUID > 36
    currency_too_long: bool = False              # Currency > 3
    remittance_info_too_long: bool = False       # RemittanceInfo line > 35
    all_length_violations: List[LengthViolation] = field(default_factory=list)


# =============================================================================
# PARSER
# =============================================================================

class IFMLParser:
    """
    Parser for ACE Pelican IFML payment messages.
    Enhanced with 8852 attribute length validation.
    """
    
    PARTY_TYPES = [
        'OriginatingPartyInfo', 'SendingBankInfo', 'DebitPartyInfo',
        'CreditPartyInfo', 'IntermediaryBankInfo', 'BeneficiaryBankInfo',
        'BeneficiaryPartyInfo', 'AccountWithInstitution', 'OrderingInstitution'
    ]
    
    PARTY_TYPES_ALT = [
        'OriginatingPartyInf', 'SendingBankInf', 'DebitPartyInf',
        'CreditPartyInf', 'IntermediaryBankInf', 'BeneficiaryBankInf',
        'BeneficiaryPartyInf', 'AccountWithInstitution', 'OrderingInstitution'
    ]
    
    def __init__(self, keep_raw: bool = False):
        self.keep_raw = keep_raw
    
    def _looks_like_bic(self, s: str) -> bool:
        """Quick check if string looks like a BIC."""
        if not s or len(s) not in (8, 11):
            return False
        s = s.upper()
        return bool(re.match(r'^[A-Z]{4}[A-Z]{2}[A-Z0-9]{2,5}$', s))
    
    def _looks_like_iban(self, s: str) -> bool:
        """Quick check if string looks like an IBAN."""
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
    
    def _check_length(self, value: str, field_name: str, max_length: int,
                      party_type: Optional[str] = None, 
                      line_number: Optional[int] = None) -> Optional[LengthViolation]:
        """Check if a field value exceeds maximum length."""
        if not value:
            return None
        
        actual_length = len(value)
        if actual_length > max_length:
            return LengthViolation(
                field_name=field_name,
                actual_length=actual_length,
                max_length=max_length,
                value_preview=value[:30] + ('...' if len(value) > 30 else ''),
                party_type=party_type,
                line_number=line_number
            )
        return None
    
    def _analyze_compound_id(self, id_value: str) -> Dict[str, Any]:
        """Analyze if ID contains multiple values (compound ID)."""
        result = {
            'has_slash': False,
            'is_compound': False,
            'compound_parts': 1,
            'has_bic_pattern': False,
            'has_nch_pattern': False,
        }
        
        if not id_value:
            return result
        
        result['has_slash'] = '/' in id_value
        
        if result['has_slash']:
            parts = id_value.split('/')
            result['compound_parts'] = len(parts)
            result['is_compound'] = len(parts) > 1
            
            for part in parts:
                part = part.strip()
                if self._looks_like_bic(part):
                    result['has_bic_pattern'] = True
                if part.isdigit() and len(part) in (6, 9):
                    result['has_nch_pattern'] = True
        
        return result
    
    def parse(self, data: dict) -> IFMLFeatures:
        """Parse IFML request and extract features with length validation."""
        features = IFMLFeatures()
        
        if self.keep_raw:
            features.raw_json = data
        
        message = self._find_message(data)
        if not message:
            return features
        
        basic_payment = message.get('BasicPayment', {})
        if not basic_payment:
            return features
        
        # Parse basic info
        self._parse_basic_info(basic_payment, features)
        
        # Parse monetary amounts with length tracking
        self._parse_monetary_amounts(basic_payment, features)
        
        # Parse party info with length validation
        self._parse_party_info(basic_payment, features)
        
        # Compute cross-party features
        self._compute_cross_party_features(features)
        
        # Aggregate length violations
        self._aggregate_length_violations(features)
        
        return features
    
    def _find_message(self, data: dict) -> Optional[dict]:
        """Find Message node in nested structure."""
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
            lambda d: safe_get(d, '*', 'Request', 'IFML', 'File', 'Message'),
            lambda d: safe_get(d, 'Request', 'IFML', 'File', 'Message'),
            lambda d: safe_get(d, 'IFML', 'File', 'Message'),
            lambda d: safe_get(d, 'File', 'Message'),
            lambda d: safe_get(d, 'Message'),
        ]
        
        for pattern in patterns:
            try:
                result = pattern(data)
                if result:
                    return result
            except:
                pass
        
        # Try with transaction ID key
        if isinstance(data, dict) and len(data) == 1:
            key = list(data.keys())[0]
            nested = data[key]
            for pattern in patterns:
                try:
                    result = pattern(nested)
                    if result:
                        return result
                except:
                    pass
        
        return None
    
    def _parse_basic_info(self, basic_payment: dict, features: IFMLFeatures):
        """Extract basic transaction information with length checks."""
        features.transaction_id = basic_payment.get('TransactionID')
        features.transaction_uid = basic_payment.get('TransactionUID')
        features.source_code = basic_payment.get('SourceCode')
        features.incoming_msg_type = basic_payment.get('IncomingMsgType')
        features.incoming_format = basic_payment.get('IncomingFormat')
        features.bank_operation_code = basic_payment.get('BankOperationCode')
        
        # 8852: Check transaction ID lengths
        if features.transaction_id:
            if len(features.transaction_id) > ATTRIBUTE_MAX_LENGTHS['TransactionID']:
                features.transaction_id_too_long = True
                features.all_length_violations.append(LengthViolation(
                    field_name='TransactionID',
                    actual_length=len(features.transaction_id),
                    max_length=ATTRIBUTE_MAX_LENGTHS['TransactionID'],
                    value_preview=features.transaction_id[:30],
                    party_type=None,
                    line_number=None
                ))
        
        if features.transaction_uid:
            if len(features.transaction_uid) > ATTRIBUTE_MAX_LENGTHS['TransactionUID']:
                features.transaction_uid_too_long = True
                features.all_length_violations.append(LengthViolation(
                    field_name='TransactionUID',
                    actual_length=len(features.transaction_uid),
                    max_length=ATTRIBUTE_MAX_LENGTHS['TransactionUID'],
                    value_preview=features.transaction_uid[:30],
                    party_type=None,
                    line_number=None
                ))
    
    def _parse_monetary_amounts(self, basic_payment: dict, features: IFMLFeatures):
        """Extract monetary amounts with length tracking."""
        amounts = basic_payment.get('MonetaryAmount', [])
        
        if isinstance(amounts, dict):
            amounts = [amounts]
        
        for amt in amounts:
            amt_type = amt.get('@Type') or amt.get('Type') or 'Unknown'
            currency = amt.get('@Currency') or amt.get('Currency')
            amount_str = str(amt.get('Amount', '0'))
            amount_val = self._parse_amount(amount_str)
            
            monetary_info = MonetaryInfo(
                amount_type=amt_type,
                currency=currency,
                amount=amount_val,
                currency_length=len(currency) if currency else 0,
                amount_str_length=len(amount_str)
            )
            features.amounts.append(monetary_info)
            
            # 8852: Check currency length
            if currency and len(currency) > ATTRIBUTE_MAX_LENGTHS['Currency']:
                features.currency_too_long = True
                features.all_length_violations.append(LengthViolation(
                    field_name='Currency',
                    actual_length=len(currency),
                    max_length=ATTRIBUTE_MAX_LENGTHS['Currency'],
                    value_preview=currency,
                    party_type=None,
                    line_number=None
                ))
            
            if amt_type == 'Amount':
                features.primary_currency = currency
                features.primary_amount = amount_val
            
            if amt_type == 'RelatedAmt' or amt.get('SubType') == 'INSTRUCTED AMOUNT':
                features.has_instructed_amount = True
                features.instructed_amount = amount_val
        
        if features.has_instructed_amount and features.primary_amount > 0:
            features.amount_mismatch = abs(features.primary_amount - features.instructed_amount) > 0.01
    
    def _parse_amount(self, amount_str: str) -> float:
        """Parse amount string to float."""
        try:
            return float(str(amount_str).replace(',', ''))
        except (ValueError, TypeError):
            return 0.0
    
    def _parse_party_info(self, basic_payment: dict, features: IFMLFeatures):
        """Extract party information with length validation."""
        party_info = basic_payment.get('PartyInfo') or basic_payment.get('PartyInf') or {}
        
        if not isinstance(party_info, dict):
            return
        
        # Standard naming
        for party_type in self.PARTY_TYPES:
            party_data = party_info.get(party_type)
            if party_data:
                if isinstance(party_data, list):
                    party_data = party_data[0] if party_data else None
                if party_data and isinstance(party_data, dict):
                    parsed_party = self._parse_single_party(party_type, party_data)
                    features.parties[party_type] = parsed_party
        
        # Alternate naming
        alt_to_standard = {
            'OriginatingPartyInf': 'OriginatingPartyInfo',
            'SendingBankInf': 'SendingBankInfo',
            'DebitPartyInf': 'DebitPartyInfo',
            'CreditPartyInf': 'CreditPartyInfo',
            'IntermediaryBankInf': 'IntermediaryBankInfo',
            'BeneficiaryBankInf': 'BeneficiaryBankInfo',
        }
        
        for alt_type, standard_type in alt_to_standard.items():
            if standard_type not in features.parties:
                party_data = party_info.get(alt_type)
                if party_data:
                    if isinstance(party_data, list):
                        party_data = party_data[0] if party_data else None
                    if party_data and isinstance(party_data, dict):
                        parsed_party = self._parse_single_party(standard_type, party_data)
                        features.parties[standard_type] = parsed_party
        
        # BeneficiaryPartyInf
        bnf_party_inf = party_info.get('BeneficiaryPartyInf')
        if bnf_party_inf:
            if isinstance(bnf_party_inf, list):
                bnf_party_inf = bnf_party_inf[0] if bnf_party_inf else None
            if bnf_party_inf and isinstance(bnf_party_inf, dict):
                parsed_party = self._parse_single_party('BeneficiaryPartyInfo', bnf_party_inf)
                features.parties['BeneficiaryPartyInfo'] = parsed_party
        
        # Parse intermediary array
        self._parse_intermediary_array(party_info, features)
    
    def _parse_single_party(self, party_type: str, party_data: dict) -> PartyInfo:
        """Parse a single party's information with length validation."""
        party = PartyInfo(party_type=party_type)
        
        if not isinstance(party_data, dict):
            return party
        
        # Get basic info section
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
        
        if isinstance(basic_info, list):
            basic_info = basic_info[0] if basic_info else {}
        
        if not isinstance(basic_info, dict):
            return party
        
        # === PARSE ID WITH LENGTH CHECK ===
        id_field = basic_info.get('ID')
        if id_field:
            if isinstance(id_field, dict):
                party.has_id = True
                party.id_type = id_field.get('@Type') or id_field.get('Type')
                party.id_value = id_field.get('#text') or id_field.get('text')
            else:
                party.has_id = True
                party.id_value = str(id_field)
            
            if party.id_value:
                party.id_length = len(party.id_value)
                
                # 8852: Check ID length
                max_id_len = ATTRIBUTE_MAX_LENGTHS['ID']
                if party.id_length > max_id_len:
                    party.id_too_long = True
                    party.has_length_violation = True
                    party.length_violations.append(LengthViolation(
                        field_name='ID',
                        actual_length=party.id_length,
                        max_length=max_id_len,
                        value_preview=party.id_value[:30],
                        party_type=party_type,
                        line_number=None
                    ))
                
                # Detect BIC/IBAN
                if party.id_type and party.id_type.upper() in ('BIC', 'S', 'SWIFT'):
                    if self._looks_like_bic(party.id_value):
                        party.has_bic = True
                        party.bic = party.id_value
                        party.bic_value = party.id_value
                elif party.id_type and party.id_type.upper() == 'IBAN':
                    party.has_iban = True
                    party.account_type = 'IBAN'
                    party.account_value = party.id_value
                elif self._looks_like_iban(party.id_value):
                    party.has_iban = True
                    party.account_type = 'IBAN'
                    party.account_value = party.id_value
                elif self._looks_like_bic(party.id_value):
                    party.has_bic = True
                    party.bic = party.id_value
                    party.bic_value = party.id_value
        
        # === PARSE BIC WITH LENGTH CHECK ===
        if party.bic:
            party.bic_length = len(party.bic)
            
            # 8852: BIC must be 8 or 11 chars - anything longer is invalid
            if party.bic_length > ATTRIBUTE_MAX_LENGTHS['BIC']:
                party.bic_too_long = True
                party.has_length_violation = True
                party.length_violations.append(LengthViolation(
                    field_name='BIC',
                    actual_length=party.bic_length,
                    max_length=ATTRIBUTE_MAX_LENGTHS['BIC'],
                    value_preview=party.bic[:20],
                    party_type=party_type,
                    line_number=None
                ))
            
            if party.bic_length >= 6:
                party.bic_country = party.bic[4:6].upper()
            party.bic_valid_format, party.bic_valid_country = validate_bic(party.bic)
        
        # === PARSE ACCOUNT WITH LENGTH CHECK ===
        acct_info = basic_info.get('AcctIDInfo') or basic_info.get('AcctIDInf')
        if acct_info:
            party.has_account = True
            if isinstance(acct_info, dict):
                acct_id = acct_info.get('ID', {})
                if isinstance(acct_id, dict):
                    party.account_type = acct_id.get('@Type') or acct_id.get('Type')
                    party.account_value = acct_id.get('#text') or acct_id.get('text')
        
        if party.account_value:
            party.account_length = len(party.account_value)
            
            # 8852: Check account length
            max_acct_len = ATTRIBUTE_MAX_LENGTHS['AccountNumber']
            if party.account_length > max_acct_len:
                party.account_too_long = True
                party.has_length_violation = True
                party.length_violations.append(LengthViolation(
                    field_name='AccountNumber',
                    actual_length=party.account_length,
                    max_length=max_acct_len,
                    value_preview=party.account_value[:30],
                    party_type=party_type,
                    line_number=None
                ))
        
        # === PARSE IBAN ===
        if party.account_type == 'IBAN' and party.account_value:
            if len(party.account_value) >= 2:
                party.iban_country = party.account_value[0:2].upper()
            party.iban_valid_format, party.iban_checksum_valid = validate_iban(party.account_value)
        
        # === PARSE ADR_BANK_ID WITH LENGTH CHECK ===
        adr_bank_id = basic_info.get('AdrBankID')
        if adr_bank_id:
            party.has_adr_bank_id = True
            if isinstance(adr_bank_id, dict):
                party.adr_bank_id_type = adr_bank_id.get('@Type') or adr_bank_id.get('Type')
                party.adr_bank_id_value = adr_bank_id.get('#text') or adr_bank_id.get('text')
            else:
                party.adr_bank_id_value = str(adr_bank_id)
            
            if party.adr_bank_id_value:
                party.adr_bank_id_length = len(party.adr_bank_id_value)
                
                # 8852: Check AdrBankID length
                max_adr_len = ATTRIBUTE_MAX_LENGTHS['AdrBankID']
                if party.adr_bank_id_length > max_adr_len:
                    party.adr_bank_id_too_long = True
                    party.has_length_violation = True
                    party.length_violations.append(LengthViolation(
                        field_name='AdrBankID',
                        actual_length=party.adr_bank_id_length,
                        max_length=max_adr_len,
                        value_preview=party.adr_bank_id_value[:30],
                        party_type=party_type,
                        line_number=None
                    ))
                
                # Parse NCH type
                if party.adr_bank_id_type in ['FEDABA', 'FW', 'FEDWIRE']:
                    party.has_nch = True
                    party.nch_type = 'FEDABA'
                    party.nch_value = party.adr_bank_id_value
                    fmt_valid, cksum_valid = validate_fedaba(party.adr_bank_id_value)
                    party.nch_valid = fmt_valid
                    party.fedaba_checksum_valid = cksum_valid
        
        # === PARSE NAME WITH LENGTH CHECK ===
        name_field = basic_info.get('Name')
        if name_field:
            party.has_name = True
            if isinstance(name_field, dict):
                party.name_value = name_field.get('#text') or name_field.get('text') or ''
            else:
                party.name_value = str(name_field)
            
            if party.name_value:
                party.name_length = len(party.name_value)
                
                # 8852: Check name length (total max 140, per line max 35)
                max_name_len = ATTRIBUTE_MAX_LENGTHS['Name']
                if party.name_length > max_name_len:
                    party.name_too_long = True
                    party.has_length_violation = True
                    party.length_violations.append(LengthViolation(
                        field_name='Name',
                        actual_length=party.name_length,
                        max_length=max_name_len,
                        value_preview=party.name_value[:30],
                        party_type=party_type,
                        line_number=None
                    ))
        
        # === PARSE ADDRESS WITH LENGTH CHECK ===
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
                party.address_lines = []
                
                for i, addr in enumerate(address_info):
                    if isinstance(addr, dict):
                        text = addr.get('text') or addr.get('#text') or ''
                    elif isinstance(addr, str):
                        text = addr
                    else:
                        text = ''
                    
                    if text:
                        party.address_lines.append(text)
                        
                        # 8852: Check each address line length
                        max_line_len = LINE_LIMITED_FIELDS['AddressInfo']
                        if len(text) > max_line_len:
                            party.address_line_too_long = True
                            party.has_length_violation = True
                            party.length_violations.append(LengthViolation(
                                field_name='AddressLine',
                                actual_length=len(text),
                                max_length=max_line_len,
                                value_preview=text[:30],
                                party_type=party_type,
                                line_number=i + 1
                            ))
                
                addr_country, is_dom, is_intl = extract_country_from_address(address_info)
                party.address_country = addr_country
                party.is_domestic = is_dom
                party.is_international = is_intl
        
        # === COUNTRY CODES ===
        party.country = basic_info.get('Country') or party_data.get('Country')
        party.mailing_country = basic_info.get('MailingCountry') or party_data.get('MailingCountry')
        party.residence_country = basic_info.get('ResidenceCountry') or party_data.get('ResidenceCountry')
        
        # === WIREKEY ===
        wirekey = basic_info.get('WireKey')
        if wirekey and isinstance(wirekey, dict):
            wk_acct = wirekey.get('AcctIDInf') or wirekey.get('AcctIDInfo')
            if wk_acct and isinstance(wk_acct, dict):
                wk_id = wk_acct.get('ID')
                if wk_id and isinstance(wk_id, dict):
                    wk_type = wk_id.get('@Type') or wk_id.get('Type')
                    wk_text = wk_id.get('#text') or wk_id.get('text')
                    if wk_type in ['S', 'SWIFT', 'BIC'] and wk_text:
                        if self._looks_like_bic(wk_text):
                            party.has_wirekey_bic = True
                            party.wirekey_bic = wk_text
                            if len(wk_text) >= 6:
                                party.wirekey_bic_country = wk_text[4:6].upper()
        
        # === IBAN REQUIREMENT ===
        party.has_iban = (party.account_type == 'IBAN')
        iban_countries = {
            'DE', 'FR', 'GB', 'ES', 'IT', 'NL', 'BE', 'AT', 'CH', 'SE', 'NO', 'DK', 'FI',
            'PL', 'PT', 'IE', 'GR', 'CZ', 'HU', 'RO', 'SK', 'LU', 'HR', 'SI', 'BG', 'LT',
            'LV', 'EE', 'CY', 'MT', 'AE', 'SA', 'QA', 'KW', 'BH', 'IL', 'TR', 'EG', 'MA'
        }
        
        party_country = (
            party.bic_country or party.iban_country or party.country or
            party.mailing_country or party.residence_country or party.address_country
        )
        if party_country and party_country.upper() in iban_countries:
            has_derivation_info = party.has_iban or party.has_account
            if not has_derivation_info:
                party.needs_iban = True
        
        # === COMPOUND ID ===
        if party.id_value:
            compound_info = self._analyze_compound_id(party.id_value)
            party.id_has_slash = compound_info['has_slash']
            party.id_is_compound = compound_info['is_compound']
            party.id_compound_parts = compound_info['compound_parts']
            party.id_has_bic_pattern = compound_info['has_bic_pattern']
            party.id_has_nch_pattern = compound_info['has_nch_pattern']
            party.id_has_bic_and_nch = compound_info['has_bic_pattern'] and compound_info['has_nch_pattern']
        
        # === REPAIR FEATURES ===
        if party.account_value:
            party.account_has_dirty_chars = bool(re.search(r'[^a-zA-Z0-9]', party.account_value))
            party.account_has_spaces = ' ' in party.account_value
            party.account_has_special_chars = bool(re.search(r'[-/.\\]', party.account_value))
        
        if party.name_value:
            party.name_has_dirty_chars = bool(re.search(r'[@#$%^*=+\[\]{}|\\<>]', party.name_value))
        
        return party
    
    def _parse_intermediary_array(self, party_info: dict, features: IFMLFeatures):
        """Parse IntermediaryBankInf array for redundancy detection."""
        intm_data = party_info.get('IntermediaryBankInf') or party_info.get('IntermediaryBankInfo')
        
        if not intm_data:
            return
        
        if isinstance(intm_data, dict):
            intm_list = [intm_data]
        elif isinstance(intm_data, list):
            intm_list = intm_data
        else:
            return
        
        features.intm_count = len(intm_list)
        features.intm_has_multiple = len(intm_list) > 1
        
        if len(intm_list) < 2:
            return
        
        adr_bank_ids = []
        countries = []
        bic_prefixes = []
        entry_types = []
        
        for entry in intm_list:
            if not isinstance(entry, dict):
                continue
            
            entry_type = entry.get('Type')
            if entry_type:
                entry_types.append(entry_type)
            
            basic = (
                entry.get('BasicPartyInf') or 
                entry.get('BasicPartyInfo') or
                entry.get('BasicPartyBankInf') or
                entry.get('BasicPartyBankInfo') or
                entry
            )
            
            if not isinstance(basic, dict):
                continue
            
            adr_id = basic.get('AdrBankID')
            if adr_id:
                if isinstance(adr_id, dict):
                    adr_id = adr_id.get('text') or adr_id.get('#text') or str(adr_id)
                adr_bank_ids.append(str(adr_id))
            
            country = basic.get('Country')
            if not country:
                addr_info = basic.get('AddressInf') or basic.get('AddressInfo') or []
                if addr_info:
                    country, _, _ = extract_country_from_address(addr_info)
            if country:
                countries.append(country.upper())
            
            id_info = basic.get('ID', {})
            if isinstance(id_info, dict):
                bic = id_info.get('text') or id_info.get('#text') or ''
                id_type = (id_info.get('Type') or id_info.get('@Type') or '').upper()
                if id_type in ('BIC', 'S', 'SWIFT', '') and bic:
                    if len(bic) >= 6 and bic[:4].isalpha():
                        bic_prefixes.append(bic[:6].upper())
        
        features.intm_types_present = list(set(entry_types))
        features.intm_has_multiple_types = len(set(entry_types)) > 1
        
        if len(adr_bank_ids) >= 2:
            features.intm_entries_share_adr_bank_id = len(set(adr_bank_ids)) < len(adr_bank_ids)
        
        if len(countries) >= 2:
            features.intm_entries_share_country = len(set(countries)) < len(countries)
        
        if len(bic_prefixes) >= 2:
            features.intm_entries_share_bic_prefix = len(set(bic_prefixes)) < len(bic_prefixes)
        
        redundancy_signals = sum([
            features.intm_entries_share_adr_bank_id,
            features.intm_entries_share_country,
            features.intm_entries_share_bic_prefix,
        ])
        features.intm_has_redundant_info = redundancy_signals >= 2
    
    def _compute_cross_party_features(self, features: IFMLFeatures):
        """Compute features that require cross-party analysis."""
        # Get key parties
        cdt = features.parties.get('CreditPartyInfo')
        bnf = features.parties.get('BeneficiaryBankInfo')
        bnp = features.parties.get('BeneficiaryPartyInfo')
        
        # 8022: WireKey BIC vs BNP IBAN country match
        if cdt and bnp:
            if cdt.has_wirekey_bic and cdt.wirekey_bic_country and bnp.has_iban and bnp.iban_country:
                features.parties['CreditPartyInfo'].bic_iban_country_match = (
                    cdt.wirekey_bic_country == bnp.iban_country
                )
    
    def _aggregate_length_violations(self, features: IFMLFeatures):
        """Aggregate all length violations from parties."""
        total_violations = len(features.all_length_violations)
        
        for party_type, party in features.parties.items():
            if party and party.length_violations:
                features.all_length_violations.extend(party.length_violations)
                total_violations += len(party.length_violations)
        
        features.length_violation_count = len(features.all_length_violations)
        features.has_any_length_violation = features.length_violation_count > 0
    
    def to_dict(self, features: IFMLFeatures) -> Dict[str, Any]:
        """Convert features to flat dictionary for ML/rules engine."""
        result = {}
        
        # Basic info
        result['transaction_id'] = features.transaction_id
        result['transaction_uid'] = features.transaction_uid
        result['source_code'] = features.source_code
        result['incoming_msg_type'] = features.incoming_msg_type
        result['bank_operation_code'] = features.bank_operation_code
        
        # Monetary
        result['has_amount'] = len(features.amounts) > 0
        result['primary_currency'] = features.primary_currency
        result['primary_amount'] = features.primary_amount
        result['amount_mismatch'] = features.amount_mismatch
        
        # Intermediary
        result['intm_count'] = features.intm_count
        result['intm_has_multiple'] = features.intm_has_multiple
        result['intm_has_redundant_info'] = features.intm_has_redundant_info
        result['has_intermediary'] = features.intm_count > 0
        
        # 8852: Global length violation features
        result['has_any_length_violation'] = features.has_any_length_violation
        result['length_violation_count'] = features.length_violation_count
        result['transaction_id_too_long'] = features.transaction_id_too_long
        result['transaction_uid_too_long'] = features.transaction_uid_too_long
        result['currency_too_long'] = features.currency_too_long
        
        # Party features
        party_prefixes = {
            'OriginatingPartyInfo': 'orig',
            'SendingBankInfo': 'send',
            'DebitPartyInfo': 'dbt',
            'CreditPartyInfo': 'cdt',
            'IntermediaryBankInfo': 'intm',
            'BeneficiaryBankInfo': 'bnf',
            'BeneficiaryPartyInfo': 'bnp',
            'AccountWithInstitution': 'acwi',
            'OrderingInstitution': 'ordi',
        }
        
        for party_type, prefix in party_prefixes.items():
            party = features.parties.get(party_type)
            
            if party:
                result[f'{prefix}_present'] = True
                result[f'{prefix}_has_id'] = party.has_id
                result[f'{prefix}_id_length'] = party.id_length
                result[f'{prefix}_has_bic'] = party.has_bic
                result[f'{prefix}_bic_length'] = party.bic_length
                result[f'{prefix}_bic_valid_format'] = party.bic_valid_format
                result[f'{prefix}_bic_valid_country'] = party.bic_valid_country
                result[f'{prefix}_bic_country'] = party.bic_country
                result[f'{prefix}_has_account'] = party.has_account
                result[f'{prefix}_account_length'] = party.account_length
                result[f'{prefix}_has_iban'] = party.has_iban
                result[f'{prefix}_iban_valid_format'] = party.iban_valid_format
                result[f'{prefix}_iban_checksum_valid'] = party.iban_checksum_valid
                result[f'{prefix}_has_nch'] = party.has_nch
                result[f'{prefix}_nch_valid'] = party.nch_valid
                result[f'{prefix}_fedaba_checksum_valid'] = party.fedaba_checksum_valid
                result[f'{prefix}_has_adr_bank_id'] = party.has_adr_bank_id
                result[f'{prefix}_country'] = party.country or party.bic_country
                result[f'{prefix}_is_domestic'] = party.is_domestic
                result[f'{prefix}_needs_iban'] = party.needs_iban
                result[f'{prefix}_has_name'] = party.has_name
                result[f'{prefix}_name_length'] = party.name_length
                result[f'{prefix}_address_line_count'] = party.address_line_count
                
                # 8852: Party-level length violation features
                result[f'{prefix}_has_length_violation'] = party.has_length_violation
                result[f'{prefix}_id_too_long'] = party.id_too_long
                result[f'{prefix}_bic_too_long'] = party.bic_too_long
                result[f'{prefix}_account_too_long'] = party.account_too_long
                result[f'{prefix}_name_too_long'] = party.name_too_long
                result[f'{prefix}_address_line_too_long'] = party.address_line_too_long
                result[f'{prefix}_adr_bank_id_too_long'] = party.adr_bank_id_too_long
            else:
                result[f'{prefix}_present'] = False
                result[f'{prefix}_has_length_violation'] = False
        
        # Cross-party features
        cdt = features.parties.get('CreditPartyInfo')
        bnp = features.parties.get('BeneficiaryPartyInfo')
        
        if cdt and bnp and cdt.has_wirekey_bic and bnp.has_iban:
            if cdt.wirekey_bic_country and bnp.iban_country:
                result['wirekey_bic_bnp_iban_match'] = (cdt.wirekey_bic_country == bnp.iban_country)
            else:
                result['wirekey_bic_bnp_iban_match'] = None
        else:
            result['wirekey_bic_bnp_iban_match'] = None
        
        return result


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    import sys
    
    parser = IFMLParser(keep_raw=True)
    
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        with open(filepath) as f:
            data = json.load(f)
        
        features = parser.parse(data)
        feature_dict = parser.to_dict(features)
        
        print("\n=== PARSED FEATURES ===")
        for k, v in sorted(feature_dict.items()):
            if v is not None and v != False and v != 0 and v != '':
                print(f"  {k}: {v}")
        
        print(f"\n=== 8852 LENGTH VIOLATIONS ({features.length_violation_count}) ===")
        for violation in features.all_length_violations:
            print(f"  {violation.field_name}: {violation.actual_length} > {violation.max_length}")
            print(f"    Party: {violation.party_type or 'Global'}")
            print(f"    Preview: {violation.value_preview}")
    else:
        print("Usage: python ifml_parser_v3.py <ifml_file.json>")
