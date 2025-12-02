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


def validate_fedaba(aba: str) -> Tuple[bool, bool]:
    """
    Validate US ABA routing number (FEDABA).
    
    ABA routing numbers are 9 digits with a specific checksum:
    - Digits weighted: 3, 7, 1, 3, 7, 1, 3, 7, 1
    - Sum of (digit * weight) mod 10 must equal 0
    
    Returns:
        Tuple of (format_valid, checksum_valid)
    """
    if not aba:
        return False, False
    
    aba = aba.strip()
    
    # Must be exactly 9 digits
    if len(aba) != 9 or not aba.isdigit():
        return False, False
    
    format_valid = True
    
    # ABA checksum algorithm
    try:
        weights = [3, 7, 1, 3, 7, 1, 3, 7, 1]
        total = sum(int(d) * w for d, w in zip(aba, weights))
        checksum_valid = (total % 10 == 0)
    except (ValueError, TypeError):
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


def extract_country_from_address(address_info: list) -> Tuple[Optional[str], bool, bool]:
    """
    Extract country code from AddressInf list.
    
    Looks at line 3 (Seq: 3) or last line for country code.
    Country code should be a valid ISO code, often surrounded by commas
    or at the end of the line.
    
    Examples:
        "NEW YORK, NY, US" -> 'US'
        "New York NY 10036 US" -> 'US'
        "LONDON, GB" -> 'GB'
        "FRANKFURT, DE 60311" -> 'DE'
    
    Args:
        address_info: List of address line dicts with 'Seq' and 'text' keys
        
    Returns:
        Tuple of (country_code, is_domestic, is_international)
        is_domestic = True if country is US
        is_international = True if country is non-US valid country
    """
    if not address_info or not isinstance(address_info, list):
        return None, False, False
    
    # Find line 3 (Seq: "3") or use last line
    target_line = None
    
    # Sort by Seq if available and find line 3
    for item in address_info:
        if isinstance(item, dict):
            seq = item.get('Seq') or item.get('seq')
            if seq == '3' or seq == 3:
                target_line = item.get('text') or item.get('Text') or ''
                break
    
    # If no line 3, use the last line
    if not target_line and address_info:
        last_item = address_info[-1]
        if isinstance(last_item, dict):
            target_line = last_item.get('text') or last_item.get('Text') or ''
        elif isinstance(last_item, str):
            target_line = last_item
    
    if not target_line:
        return None, False, False
    
    target_line = target_line.upper().strip()
    
    # Strategy 1: Look for country code with comma before it (", US" or ", GB")
    # Split by comma and check each part
    parts = [p.strip() for p in target_line.replace(',', ' , ').split()]
    
    # Check each word/part for valid country code
    found_country = None
    for i, part in enumerate(parts):
        # Clean the part - remove any punctuation
        clean_part = ''.join(c for c in part if c.isalpha())
        if len(clean_part) == 2 and clean_part in VALID_COUNTRY_CODES:
            found_country = clean_part
            # Prefer codes that come after a comma or at the end
            if i > 0 and parts[i-1] == ',':
                break  # Found country after comma, this is likely the right one
    
    # Strategy 2: Check last 2-3 characters if no country found
    if not found_country:
        # Check if last word is a country code
        words = target_line.split()
        if words:
            last_word = ''.join(c for c in words[-1] if c.isalpha())
            if len(last_word) == 2 and last_word in VALID_COUNTRY_CODES:
                found_country = last_word
    
    # Strategy 3: Check for country code embedded in zip (e.g., "10036 US")
    if not found_country:
        import re
        # Look for pattern: digits followed by space and 2-letter code
        match = re.search(r'\d+\s+([A-Z]{2})\b', target_line)
        if match and match.group(1) in VALID_COUNTRY_CODES:
            found_country = match.group(1)
    
    if found_country:
        is_domestic = (found_country == 'US')
        is_international = (found_country != 'US')
        return found_country, is_domestic, is_international
    
    return None, False, False


@dataclass
class PartyInfo:
    """Normalized party information."""
    party_type: str
    has_id: bool = False
    id_value: Optional[str] = None
    id_type: Optional[str] = None
    has_bic: bool = False
    bic: Optional[str] = None
    bic_value: Optional[str] = None  # Alias for bic for cross-party comparison
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
    address_lines: List[str] = None  # Actual address text for duplicate detection
    has_name: bool = False
    bank_flag: Optional[str] = None
    charge_flag: Optional[str] = None
    # Address parsing features
    address_country: Optional[str] = None  # Country code extracted from address line 3
    is_domestic: bool = False  # True if address country is US
    is_international: bool = False  # True if address country is non-US
    # NCH/Clearing code features (for 8026, 8895)
    has_nch: bool = False  # Has any national clearing house code
    nch_type: Optional[str] = None  # FEDABA, CHIPS, SORTCODE, etc.
    nch_value: Optional[str] = None  # The actual clearing code
    nch_valid: bool = False  # Is the NCH code valid format
    fedaba_checksum_valid: bool = False  # Does FEDABA pass ABA checksum
    has_adr_bank_id: bool = False  # Has AdrBankID field
    adr_bank_id_type: Optional[str] = None  # Type of AdrBankID (FEDABA, CHIPS, etc.)
    # WireKey BIC (Account With Institution) - for 8022 cross-party check
    has_wirekey_bic: bool = False
    wirekey_bic: Optional[str] = None
    wirekey_bic_country: Optional[str] = None
    # IBAN requirement indicator (for 8004)
    needs_iban: bool = False  # International payment that should have IBAN
    has_iban: bool = False  # Explicit IBAN present
    # Multi-source indicators (for 8026 - NCH inconsistency)
    nch_sources: int = 0  # Count of NCH sources (account, BIC, IBAN)
    # NCH validation applicability (for 8895)
    nch_validation_applicable: bool = False  # True only for US domestic payments
    
    # === REPAIR DETECTION FEATURES (for 9XXX codes) ===
    # 9002, 9009, 9014, 9015, 9019, 9020: Cleaning features
    account_has_dirty_chars: bool = False  # Non-alphanumeric in account number
    name_has_dirty_chars: bool = False  # Non-alphanumeric in name (beyond normal punctuation)
    account_has_spaces: bool = False  # Spaces in account (needs cleaning)
    account_has_special_chars: bool = False  # Special chars like -/. in account
    
    # 9004, 9007, 9018: IBAN derivation features
    is_iban_derivable: bool = False  # Has country + account, could derive IBAN
    has_bban_in_iban_country: bool = False  # Has BBAN but in IBAN-required country
    
    # 9005, 9008: BIC derivation features  
    is_bic_derivable: bool = False  # Has NCH or IBAN, could derive BIC
    has_nch_no_bic: bool = False  # Has NCH but no BIC (BIC derivable from NCH)
    has_iban_no_bic: bool = False  # Has IBAN but no BIC (BIC derivable from IBAN)
    
    # 9000, 9021, 9028: NCH repair features
    nch_has_dirty_chars: bool = False  # NCH needs cleaning
    nch_needs_formatting: bool = False  # NCH wrong length/format but fixable
    
    # 9017, 9018: Duplicate/multiple party features
    has_multiple_ids: bool = False  # Multiple ID values present
    has_duplicate_info: bool = False  # Duplicate information detected
    
    # 9022: Account formatting features
    account_needs_length_fix: bool = False  # Account too short/long for type
    
    # 9006, 9012: IBAN formatting features
    iban_needs_formatting: bool = False  # IBAN present but needs cleanup/formatting
    
    # === COMPOUND ID FEATURES (for 8026, 8022) ===
    id_has_slash: bool = False  # ID contains '/' separator
    id_is_compound: bool = False  # ID appears to contain multiple values
    id_compound_parts: int = 0  # Number of parts in compound ID
    id_has_bic_pattern: bool = False  # Compound contains BIC-like pattern
    id_has_nch_pattern: bool = False  # Compound contains NCH-like pattern
    id_has_bic_and_nch: bool = False  # Contains both - high risk for 8026
    
    def __post_init__(self):
        if self.address_lines is None:
            self.address_lines = []
        # Ensure bic_value is set from bic
        if self.bic and not self.bic_value:
            self.bic_value = self.bic


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
    
    # === INTERMEDIARY REDUNDANCY FEATURES (for 9018, 9024) ===
    intm_count: int = 0  # Number of IntermediaryBankInf entries
    intm_has_multiple: bool = False  # More than one intermediary entry
    intm_entries_share_adr_bank_id: bool = False  # Multiple entries have same AdrBankID
    intm_entries_share_country: bool = False  # Multiple entries have same country
    intm_entries_share_bic_prefix: bool = False  # Multiple entries have similar BIC
    intm_has_redundant_info: bool = False  # Composite: likely has duplicates to remove
    intm_types_present: List[str] = field(default_factory=list)  # e.g., ["IBK", "IB1"]
    intm_has_multiple_types: bool = False  # Different Type values present


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
        'BeneficiaryPartyInfo',  # Added for BNPPTY
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
    
    def _looks_like_bic(self, s: str) -> bool:
        """Quick check if string looks like a BIC."""
        if not s or len(s) not in (8, 11):
            return False
        s = s.upper()
        return bool(re.match(r'^[A-Z]{4}[A-Z]{2}[A-Z0-9]{2,5}$', s))
    
    def _looks_like_iban(self, s: str) -> bool:
        """
        Check if string looks like an IBAN by format and country-specific length.
        This detects IBAN-like strings by pattern, NOT by checksum validity.
        """
        if not s:
            return False
        cleaned = s.upper().replace(' ', '').replace('-', '')
        if len(cleaned) < 15 or len(cleaned) > 34:
            return False
        # Must start with 2 letters + 2 digits + alphanumeric
        if not re.match(r'^[A-Z]{2}[0-9]{2}[A-Z0-9]+$', cleaned):
            return False
        
        # Check country-specific IBAN length (common countries)
        # This helps avoid false positives like IT12345... (23 chars when IT needs 27)
        iban_lengths = {
            'AL': 28, 'AD': 24, 'AT': 20, 'AZ': 28, 'BH': 22, 'BY': 28, 'BE': 16,
            'BA': 20, 'BR': 29, 'BG': 22, 'CR': 22, 'HR': 21, 'CY': 28, 'CZ': 24,
            'DK': 18, 'DO': 28, 'TL': 23, 'EE': 20, 'FO': 18, 'FI': 18, 'FR': 27,
            'GE': 22, 'DE': 22, 'GI': 23, 'GR': 27, 'GL': 18, 'GT': 28, 'HU': 28,
            'IS': 26, 'IQ': 23, 'IE': 22, 'IL': 23, 'IT': 27, 'JO': 30, 'KZ': 20,
            'XK': 20, 'KW': 30, 'LV': 21, 'LB': 28, 'LI': 21, 'LT': 20, 'LU': 20,
            'MK': 19, 'MT': 31, 'MR': 27, 'MU': 30, 'MC': 27, 'MD': 24, 'ME': 22,
            'NL': 18, 'NO': 15, 'PK': 24, 'PS': 29, 'PL': 28, 'PT': 25, 'QA': 29,
            'RO': 24, 'LC': 32, 'SM': 27, 'ST': 25, 'SA': 24, 'RS': 22, 'SC': 31,
            'SK': 24, 'SI': 19, 'ES': 24, 'SE': 24, 'CH': 21, 'TN': 24, 'TR': 26,
            'UA': 29, 'AE': 23, 'GB': 22, 'VA': 22, 'VG': 24
        }
        
        country = cleaned[:2]
        expected_len = iban_lengths.get(country)
        
        if expected_len:
            # If we know the expected length, require exact match
            return len(cleaned) == expected_len
        else:
            # Unknown country - accept if reasonable length
            return True
    
    def _validate_iban_checksum(self, s: str) -> bool:
        """
        Validate IBAN mod-97 checksum.
        Returns True if checksum is valid, False otherwise.
        """
        if not s:
            return False
        cleaned = s.upper().replace(' ', '').replace('-', '')
        if len(cleaned) < 15 or len(cleaned) > 34:
            return False
        if not re.match(r'^[A-Z]{2}[0-9]{2}[A-Z0-9]+$', cleaned):
            return False
        # Move first 4 chars to end and convert letters to numbers (A=10, B=11, etc.)
        rearranged = cleaned[4:] + cleaned[:4]
        numeric = ''
        for char in rearranged:
            if char.isdigit():
                numeric += char
            else:
                numeric += str(ord(char) - ord('A') + 10)
        try:
            return int(numeric) % 97 == 1
        except (ValueError, OverflowError):
            return False
    
    def _analyze_compound_id(self, id_text: str) -> dict:
        """
        Analyze ID for compound/multi-value patterns.
        
        Compound IDs like "CUSCAU2S/AU805013" contain both a BIC and NCH,
        which can trigger 8026 (NCH inconsistency) if they don't agree,
        or 8022 (BIC/IBAN mismatch) if countries differ.
        
        Returns:
            dict with has_slash, is_compound, compound_parts, has_bic_pattern, has_nch_pattern
        """
        result = {
            'has_slash': False,
            'is_compound': False,
            'compound_parts': 0,
            'has_bic_pattern': False,
            'has_nch_pattern': False,
        }
        
        if not id_text:
            return result
        
        id_text = str(id_text).strip()
        
        # Check for slash separator
        result['has_slash'] = '/' in id_text
        
        if '/' in id_text:
            parts = [p.strip() for p in id_text.split('/') if p.strip()]
            result['is_compound'] = len(parts) > 1
            result['compound_parts'] = len(parts)
            
            for part in parts:
                # BIC pattern: 8 or 11 chars, first 4 are letters (bank code)
                # positions 5-6 are letters (country code)
                if len(part) in [8, 11]:
                    if part[:4].isalpha() and len(part) >= 6 and part[4:6].isalpha():
                        result['has_bic_pattern'] = True
                
                # NCH patterns:
                # - FEDABA: 9 digits
                # - CHIPS: 6 digits  
                # - AU BSB: 6 digits or XX + 6 digits (e.g., AU805013)
                # - Sort code: 6 digits with possible dashes
                clean_part = part.replace('-', '').replace(' ', '')
                
                if clean_part.isdigit() and len(clean_part) in [6, 9]:
                    result['has_nch_pattern'] = True
                
                # Australian BSB pattern: AU + 6 digits or 2 letters + 6 digits
                if len(part) >= 8:
                    prefix = part[:2]
                    suffix = part[2:].replace('-', '')
                    if prefix.isalpha() and suffix.isdigit() and len(suffix) >= 6:
                        result['has_nch_pattern'] = True
        
        return result
    
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
        
        # Ensure party_info is a dict
        if not isinstance(party_info, dict):
            return
        
        # Try standard naming (Info)
        for party_type in self.PARTY_TYPES:
            party_data = party_info.get(party_type)
            if party_data:
                # Handle list - take first element
                if isinstance(party_data, list):
                    party_data = party_data[0] if party_data else None
                if party_data and isinstance(party_data, dict):
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
        }
        
        for alt_type, standard_type in alt_to_standard.items():
            if standard_type not in features.parties:  # Don't overwrite if already parsed
                party_data = party_info.get(alt_type)
                if party_data:
                    # Handle list - take first element
                    if isinstance(party_data, list):
                        party_data = party_data[0] if party_data else None
                    if party_data and isinstance(party_data, dict):
                        parsed_party = self._parse_single_party(standard_type, party_data)
                        features.parties[standard_type] = parsed_party
        
        # FIX: Parse BeneficiaryPartyInf as a separate party (BeneficiaryPartyInfo -> bnp_)
        # This is distinct from BeneficiaryBankInfo and maps to BNPPTY in ACE responses
        bnf_party_inf = party_info.get('BeneficiaryPartyInf')
        if bnf_party_inf:
            if isinstance(bnf_party_inf, list):
                bnf_party_inf = bnf_party_inf[0] if bnf_party_inf else None
            if bnf_party_inf and isinstance(bnf_party_inf, dict):
                parsed_party = self._parse_single_party('BeneficiaryPartyInfo', bnf_party_inf)
                features.parties['BeneficiaryPartyInfo'] = parsed_party
        
        # Parse intermediary array for redundancy detection (9018, 9024)
        self._parse_intermediary_array(party_info, features)
    
    def _parse_intermediary_array(self, party_info: dict, features: IFMLFeatures):
        """
        Parse IntermediaryBankInf as an array to detect redundancy patterns.
        
        This handles the case where IntermediaryBankInf is a list of entries
        (e.g., IBK, IB1) and detects when entries share information that
        indicates duplicates that ACE will remove (9018) or consolidate (9024).
        """
        # Get intermediary data - handle both naming conventions
        intm_data = party_info.get('IntermediaryBankInf') or party_info.get('IntermediaryBankInfo')
        
        if not intm_data:
            return
        
        # Normalize to list
        if isinstance(intm_data, dict):
            intm_list = [intm_data]
        elif isinstance(intm_data, list):
            intm_list = intm_data
        else:
            return
        
        features.intm_count = len(intm_list)
        features.intm_has_multiple = len(intm_list) > 1
        
        if len(intm_list) < 2:
            # Single intermediary - no redundancy possible
            return
        
        # Collect identifying information from each entry
        adr_bank_ids = []
        countries = []
        bic_prefixes = []
        entry_types = []
        
        for entry in intm_list:
            if not isinstance(entry, dict):
                continue
            
            # Get entry type (IBK, IB1, etc.)
            entry_type = entry.get('Type')
            if entry_type:
                entry_types.append(entry_type)
            
            # Navigate to basic info - handle multiple naming conventions
            basic = (
                entry.get('BasicPartyInf') or 
                entry.get('BasicPartyInfo') or
                entry.get('BasicPartyBankInf') or
                entry.get('BasicPartyBankInfo') or
                entry
            )
            
            if not isinstance(basic, dict):
                continue
            
            # Extract AdrBankID
            adr_id = basic.get('AdrBankID')
            if adr_id:
                if isinstance(adr_id, dict):
                    adr_id = adr_id.get('text') or adr_id.get('#text') or str(adr_id)
                adr_bank_ids.append(str(adr_id))
            
            # Extract country
            country = basic.get('Country')
            if not country:
                # Try to get from address
                addr_info = basic.get('AddressInf') or basic.get('AddressInfo') or []
                if addr_info:
                    country, _, _ = extract_country_from_address(addr_info)
            if country:
                countries.append(country.upper())
            
            # Extract BIC prefix (first 6 chars = bank + country)
            id_info = basic.get('ID', {})
            if isinstance(id_info, dict):
                bic = id_info.get('text') or id_info.get('#text') or ''
                id_type = (id_info.get('Type') or id_info.get('@Type') or '').upper()
                
                # Only consider BIC-type IDs
                if id_type in ('BIC', 'S', 'SWIFT', '') and bic:
                    # Check if it looks like a BIC
                    if len(bic) >= 6 and bic[:4].isalpha():
                        bic_prefixes.append(bic[:6].upper())
            elif isinstance(id_info, str) and len(id_info) >= 6:
                if id_info[:4].isalpha():
                    bic_prefixes.append(id_info[:6].upper())
        
        # Store entry types
        features.intm_types_present = list(set(entry_types))
        features.intm_has_multiple_types = len(set(entry_types)) > 1
        
        # Detect redundancy patterns
        # Same AdrBankID across entries = strong redundancy signal
        if len(adr_bank_ids) >= 2:
            features.intm_entries_share_adr_bank_id = len(set(adr_bank_ids)) < len(adr_bank_ids)
        
        # Same country across entries = routing redundancy
        if len(countries) >= 2:
            features.intm_entries_share_country = len(set(countries)) < len(countries)
        
        # Same BIC prefix = same bank, likely duplicate
        if len(bic_prefixes) >= 2:
            features.intm_entries_share_bic_prefix = len(set(bic_prefixes)) < len(bic_prefixes)
        
        # Composite redundancy indicator
        # If 2+ signals match, very likely ACE will consolidate (9018/9024)
        redundancy_signals = sum([
            features.intm_entries_share_adr_bank_id,
            features.intm_entries_share_country,
            features.intm_entries_share_bic_prefix,
        ])
        features.intm_has_redundant_info = redundancy_signals >= 2
    
    def _parse_single_party(self, party_type: str, party_data: dict) -> PartyInfo:
        """Parse a single party's information."""
        party = PartyInfo(party_type=party_type)
        
        # Ensure party_data is a dict
        if not isinstance(party_data, dict):
            return party
        
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
        
        # Handle basic_info being a list - take first element
        if isinstance(basic_info, list):
            basic_info = basic_info[0] if basic_info else {}
        
        # Ensure basic_info is a dict
        if not isinstance(basic_info, dict):
            return party
        
        # Extract ID - handle multiple formats
        id_field = basic_info.get('ID')
        if id_field:
            if isinstance(id_field, dict):
                party.has_id = True
                # Handle both @Type/#text and Type/text formats
                party.id_type = id_field.get('@Type') or id_field.get('Type')
                party.id_value = id_field.get('#text') or id_field.get('text')
                # FIX: Type="S" means SWIFT/BIC, but only if value looks like a BIC
                if party.id_type and party.id_type.upper() in ('BIC', 'S', 'SWIFT'):
                    # Only treat as BIC if value actually looks like one (8-11 alphanumeric chars)
                    if party.id_value and self._looks_like_bic(party.id_value):
                        party.has_bic = True
                        party.bic = party.id_value
                elif party.id_type and party.id_type.upper() == 'IBAN':
                    party.has_iban = True
                    party.account_type = 'IBAN'
                    party.account_value = party.id_value
            else:
                party.has_id = True
                party.id_value = str(id_field)
                # For plain string IDs: detect IBAN by pattern so 8894/8898 can fire on invalid IBANs
                # Checksum validation is done separately - has_iban=True doesn't mean it's valid
                if self._looks_like_iban(party.id_value):
                    party.has_iban = True
                    party.account_type = 'IBAN'
                    party.account_value = party.id_value
                elif self._looks_like_bic(party.id_value):
                    party.has_bic = True
                    party.bic = party.id_value
                    party.bic_value = party.id_value
        
        # === COMPOUND ID ANALYSIS (for 8026, 8022) ===
        # Detect IDs like "CUSCAU2S/AU805013" that contain multiple values
        if party.id_value:
            compound_info = self._analyze_compound_id(party.id_value)
            party.id_has_slash = compound_info['has_slash']
            party.id_is_compound = compound_info['is_compound']
            party.id_compound_parts = compound_info['compound_parts']
            party.id_has_bic_pattern = compound_info['has_bic_pattern']
            party.id_has_nch_pattern = compound_info['has_nch_pattern']
            party.id_has_bic_and_nch = compound_info['has_bic_pattern'] and compound_info['has_nch_pattern']
        
        # Check for BIC in ID type (legacy check) - only if value looks like a BIC
        if isinstance(id_field, dict):
            id_type = id_field.get('@Type') or id_field.get('Type')
            id_val = id_field.get('#text') or id_field.get('text')
            if id_type and id_type.upper() in ('BIC', 'S', 'SWIFT'):
                if id_val and self._looks_like_bic(id_val):
                    party.has_bic = True
                    party.bic = id_val
                    party.bic_value = party.bic
        
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
                # Store address lines for duplicate detection
                party.address_lines = []
                for addr in address_info:
                    if isinstance(addr, dict):
                        text = addr.get('text') or addr.get('#text') or ''
                        if text:
                            party.address_lines.append(text)
                    elif isinstance(addr, str):
                        party.address_lines.append(addr)
                # Extract country from address line 3
                addr_country, is_dom, is_intl = extract_country_from_address(address_info)
                party.address_country = addr_country
                party.is_domestic = is_dom
                party.is_international = is_intl
            elif isinstance(address_info, dict):
                party.address_line_count = 1
                party.has_name = True
                text = address_info.get('text') or address_info.get('#text') or ''
                if text:
                    party.address_lines = [text]
        
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
        
        # === NCH/Clearing Code Features (for 8026, 8895) ===
        
        # Check AdrBankID for NCH codes
        adr_bank_id = basic_info.get('AdrBankID')
        if adr_bank_id:
            party.has_adr_bank_id = True
            if isinstance(adr_bank_id, dict):
                party.adr_bank_id_type = adr_bank_id.get('@Type') or adr_bank_id.get('Type')
                adr_value = adr_bank_id.get('#text') or adr_bank_id.get('text')
                
                # Check for NCH types
                if party.adr_bank_id_type in ['FEDABA', 'FW', 'FEDWIRE']:
                    party.has_nch = True
                    party.nch_type = 'FEDABA'
                    party.nch_value = adr_value
                    if adr_value:
                        fmt_valid, cksum_valid = validate_fedaba(adr_value)
                        party.nch_valid = fmt_valid
                        party.fedaba_checksum_valid = cksum_valid
                elif party.adr_bank_id_type in ['CHIPS', 'CH', 'CHIPSABA']:
                    party.has_nch = True
                    party.nch_type = 'CHIPS'
                    party.nch_value = adr_value
                    # CHIPS ABA is 6 digits
                    party.nch_valid = adr_value and len(adr_value) == 6 and adr_value.isdigit()
                elif party.adr_bank_id_type in ['SORTCODE', 'SC', 'UKSC']:
                    party.has_nch = True
                    party.nch_type = 'SORTCODE'
                    party.nch_value = adr_value
                    # UK Sort Code is 6 digits
                    party.nch_valid = adr_value and len(adr_value.replace('-', '')) == 6
        
        # Detect NCH from account number if FEDABA detected
        if party.is_fedaba and party.account_value:
            party.has_nch = True
            party.nch_type = party.nch_type or 'FEDABA'
            party.nch_value = party.nch_value or party.account_value
            fmt_valid, cksum_valid = validate_fedaba(party.account_value)
            party.nch_valid = party.nch_valid or fmt_valid
            party.fedaba_checksum_valid = party.fedaba_checksum_valid or cksum_valid
        
        if party.is_chips_aba and party.account_value:
            party.has_nch = True
            party.nch_type = party.nch_type or 'CHIPS'
        
        # === WireKey BIC Parsing (Account With Institution) ===
        # WireKey contains BIC for correspondent/intermediary routing
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
        
        # === IBAN Requirement Features (for 8004) ===
        # International payments to IBAN countries should have IBAN
        party.has_iban = (party.account_type == 'IBAN')
        
        # IBAN is typically needed for international payments to Europe, etc.
        iban_countries = {
            'DE', 'FR', 'GB', 'ES', 'IT', 'NL', 'BE', 'AT', 'CH', 'SE', 'NO', 'DK', 'FI',
            'PL', 'PT', 'IE', 'GR', 'CZ', 'HU', 'RO', 'SK', 'LU', 'HR', 'SI', 'BG', 'LT',
            'LV', 'EE', 'CY', 'MT', 'AE', 'SA', 'QA', 'KW', 'BH', 'IL', 'TR', 'EG', 'MA'
        }
        
        # Determine if this party needs IBAN based on country
        party_country = (
            party.bic_country or 
            party.iban_country or 
            party.country or 
            party.mailing_country or
            party.residence_country or
            party.address_country
        )
        if party_country and party_country.upper() in iban_countries:
            # Only set needs_iban if we don't already have IBAN AND
            # we don't have account info that could be used to derive IBAN
            # ACE can derive IBAN from: account + BIC, account + NCH, or just account in some countries
            has_derivation_info = (
                party.has_iban or  # Already has IBAN
                party.has_account  # Has account info (ACE may be able to derive)
            )
            if not has_derivation_info:
                party.needs_iban = True
        
        # Count NCH sources for inconsistency detection (8026)
        nch_sources = 0
        if party.is_fedaba or party.is_chips_aba:
            nch_sources += 1  # From account
        if party.has_adr_bank_id and party.adr_bank_id_type in ['FEDABA', 'FW', 'FEDWIRE', 'CHIPS', 'CH']:
            nch_sources += 1  # From AdrBankID
        if party.bic and party.bic_country == 'US':
            nch_sources += 1  # Could derive NCH from BIC
        party.nch_sources = nch_sources
        
        # === NCH Validation Applicability (for 8895) ===
        # NCH validation only applies to domestic US payments, NOT international
        # If payment is international (non-US beneficiary), NCH errors shouldn't fire
        party_country = (
            party.bic_country or 
            party.iban_country or 
            party.country or 
            party.address_country
        )
        
        # NCH validation is applicable only if:
        # 1. Party is in US (domestic), OR
        # 2. No country detected but has US-style routing (FEDABA/CHIPS)
        is_us_party = party_country and party_country.upper() == 'US'
        has_us_routing = party.is_fedaba or party.is_chips_aba or party.nch_type in ['FEDABA', 'CHIPS']
        party.nch_validation_applicable = is_us_party or (not party_country and has_us_routing)
        
        # === REPAIR DETECTION FEATURES (for 9XXX codes) ===
        
        # 9002, 9009, 9014, 9015: Account cleaning detection
        if party.account_value:
            acct = party.account_value
            # Check for dirty chars (non-alphanumeric)
            has_non_alnum = bool(re.search(r'[^a-zA-Z0-9]', acct))
            party.account_has_dirty_chars = has_non_alnum
            party.account_has_spaces = ' ' in acct
            party.account_has_special_chars = bool(re.search(r'[-/.\\]', acct))
        
        # Name cleaning detection (beyond normal punctuation)
        if hasattr(party, 'has_name') and party.has_name:
            # Name has dirty chars if it contains unusual characters
            # Note: names commonly have spaces, periods, commas - so we look for unusual ones
            name_val = basic_info.get('Name', '') or ''
            if isinstance(name_val, dict):
                name_val = name_val.get('#text', '') or name_val.get('text', '')
            if name_val:
                # Unusual chars in names: digits, @, #, $, %, etc.
                party.name_has_dirty_chars = bool(re.search(r'[@#$%^*=+\[\]{}|\\<>]', str(name_val)))
        
        # 9004, 9007: IBAN derivation detection
        # Can derive IBAN if: has country + has account number + in IBAN country + no IBAN yet
        if party_country and party.has_account and not party.has_iban:
            if party_country.upper() in iban_countries:
                party.is_iban_derivable = True
                party.has_bban_in_iban_country = (party.account_type in ['BBAN', 'D', 'DDA', None])
        
        # 9005, 9008: BIC derivation detection
        # Can derive BIC if: has NCH (US) or has IBAN, but no BIC
        if not party.has_bic:
            if party.has_nch or party.is_fedaba or party.is_chips_aba:
                party.is_bic_derivable = True
                party.has_nch_no_bic = True
            if party.has_iban:
                party.is_bic_derivable = True
                party.has_iban_no_bic = True
        
        # 9000, 9021, 9028: NCH repair detection
        if party.nch_value:
            nch = party.nch_value
            # NCH has dirty chars
            party.nch_has_dirty_chars = bool(re.search(r'[^0-9]', nch))
            # NCH needs formatting (wrong length for type)
            if party.nch_type == 'FEDABA' and len(nch.replace('-', '').replace(' ', '')) != 9:
                party.nch_needs_formatting = True
            elif party.nch_type == 'CHIPS' and len(nch.replace('-', '').replace(' ', '')) != 6:
                party.nch_needs_formatting = True
        
        # 9017, 9018: Multiple/duplicate info detection
        # Check if multiple IDs present
        id_info = basic_info.get('ID', {})
        if isinstance(id_info, list) and len(id_info) > 1:
            party.has_multiple_ids = True
            party.has_duplicate_info = True
        
        # 9022: Account length issues
        if party.account_value and party.account_type:
            acct_len = len(party.account_value.replace(' ', '').replace('-', ''))
            # Expected lengths by type
            expected_lengths = {
                'IBAN': (15, 34),  # Range
                'CLABE': (18, 18),
                'FEDABA': (9, 9),
            }
            if party.account_type in expected_lengths:
                min_len, max_len = expected_lengths[party.account_type]
                if acct_len < min_len or acct_len > max_len:
                    party.account_needs_length_fix = True
        
        # 9006, 9012: IBAN formatting detection
        if party.has_iban and party.account_value:
            iban = party.account_value
            # IBAN needs formatting if: has spaces, lowercase, or wrong structure
            if ' ' in iban or iban != iban.upper() or not iban[:2].isalpha():
                party.iban_needs_formatting = True
        
        return party
    
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
                id_type = (id_field.get('Type') or id_field.get('@Type') or '').upper()
                if id_type == 'IBAN' or self._looks_like_iban(id_text):
                    existing_party.has_iban = True
                    existing_party.account_type = 'IBAN'
                    existing_party.account_value = id_text
                    # Validate IBAN
                    if len(id_text) >= 2:
                        cleaned = id_text.upper().replace(' ', '').replace('-', '')
                        existing_party.iban_country = cleaned[:2]
                    existing_party.iban_valid_format, existing_party.iban_checksum_valid = validate_iban(id_text)
                    # Check if needs formatting
                    if ' ' in id_text or '-' in id_text:
                        existing_party.iban_needs_formatting = True
                    # Update country match
                    if existing_party.bic_country and existing_party.iban_country:
                        existing_party.bic_iban_country_match = (existing_party.bic_country == existing_party.iban_country)
            else:
                id_text = str(id_field)
                if self._looks_like_iban(id_text):
                    existing_party.has_iban = True
                    existing_party.account_type = 'IBAN'
                    existing_party.account_value = id_text
                    # Validate IBAN
                    if len(id_text) >= 2:
                        cleaned = id_text.upper().replace(' ', '').replace('-', '')
                        existing_party.iban_country = cleaned[:2]
                    existing_party.iban_valid_format, existing_party.iban_checksum_valid = validate_iban(id_text)
                    # Check if needs formatting
                    if ' ' in id_text or '-' in id_text:
                        existing_party.iban_needs_formatting = True
                    # Update country match
                    if existing_party.bic_country and existing_party.iban_country:
                        existing_party.bic_iban_country_match = (existing_party.bic_country == existing_party.iban_country)
    
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
        
        # ==================================================================
        # 9017/9018: Cross-party duplicate detection
        # ==================================================================
        self._detect_cross_party_duplicates(features)
    
    def _detect_cross_party_duplicates(self, features: IFMLFeatures):
        """
        Detect duplicate information across parties for 9017/9018.
        
        9017 = Multiple party information present
        9018 = Duplicate party information removed
        
        Key pattern: When IntermediaryBankInf and BeneficiaryBankInf both exist
        and share similar info (BIC, address), ACE removes duplicate from beneficiary.
        """
        intm = features.parties.get('IntermediaryBankInfo')
        bnf = features.parties.get('BeneficiaryBankInfo')
        
        # Pattern 1: Intermediary and Beneficiary Bank both present
        # This is the main trigger for 9018
        if intm and bnf:
            # Check for overlapping BICs
            if intm.bic_value and bnf.bic_value:
                if intm.bic_value == bnf.bic_value:
                    bnf.has_duplicate_info = True
                    intm.has_duplicate_info = True
            
            # Check for overlapping addresses (same country + similar address)
            if intm.country and bnf.country and intm.country == bnf.country:
                # Same country - likely duplicate routing info
                bnf.has_duplicate_info = True
            
            # Check for overlapping address text
            intm_addr = getattr(intm, 'address_lines', []) or []
            bnf_addr = getattr(bnf, 'address_lines', []) or []
            if intm_addr and bnf_addr:
                # Check if any address line matches
                intm_addr_set = set(str(a).upper().strip() for a in intm_addr if a)
                bnf_addr_set = set(str(a).upper().strip() for a in bnf_addr if a)
                if intm_addr_set & bnf_addr_set:  # Intersection
                    bnf.has_duplicate_info = True
            
            # If both have address info and intermediary exists, 
            # beneficiary bank info is often duplicate
            if intm.has_name and bnf.has_name:
                bnf.has_duplicate_info = True
        
        # Pattern 2: Check CreditParty vs BeneficiaryBank overlap
        cdt = features.parties.get('CreditPartyInfo')
        if cdt and bnf:
            if cdt.bic_value and bnf.bic_value and cdt.bic_value == bnf.bic_value:
                # Same BIC in both - one is duplicate
                bnf.has_duplicate_info = True
    
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
            # === INTERMEDIARY REDUNDANCY FEATURES (for 9018, 9024) ===
            'intm_count': features.intm_count,
            'intm_has_multiple': features.intm_has_multiple,
            'intm_entries_share_adr_bank_id': features.intm_entries_share_adr_bank_id,
            'intm_entries_share_country': features.intm_entries_share_country,
            'intm_entries_share_bic_prefix': features.intm_entries_share_bic_prefix,
            'intm_has_redundant_info': features.intm_has_redundant_info,
            'intm_has_multiple_types': features.intm_has_multiple_types,
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
                result[f'{prefix}_address_country'] = party.address_country
                result[f'{prefix}_is_domestic'] = party.is_domestic
                result[f'{prefix}_is_international'] = party.is_international
                # NCH/Clearing code features (for 8026, 8895)
                result[f'{prefix}_has_nch'] = party.has_nch
                result[f'{prefix}_nch_type'] = party.nch_type
                result[f'{prefix}_nch_valid'] = party.nch_valid
                result[f'{prefix}_fedaba_checksum_valid'] = party.fedaba_checksum_valid
                result[f'{prefix}_has_adr_bank_id'] = party.has_adr_bank_id
                result[f'{prefix}_adr_bank_id_type'] = party.adr_bank_id_type
                # IBAN requirement features (for 8004)
                result[f'{prefix}_has_iban'] = party.has_iban
                result[f'{prefix}_needs_iban'] = party.needs_iban
                # NCH source count (for 8026 - inconsistency)
                result[f'{prefix}_nch_sources'] = party.nch_sources
                # NCH validation applicability (for 8895)
                result[f'{prefix}_nch_validation_applicable'] = party.nch_validation_applicable
                # === REPAIR DETECTION FEATURES (for 9XXX) ===
                # Cleaning features (9002, 9009, 9014, 9015)
                result[f'{prefix}_account_has_dirty_chars'] = party.account_has_dirty_chars
                result[f'{prefix}_name_has_dirty_chars'] = party.name_has_dirty_chars
                result[f'{prefix}_account_has_spaces'] = party.account_has_spaces
                result[f'{prefix}_account_has_special_chars'] = party.account_has_special_chars
                # IBAN derivation (9004, 9007)
                result[f'{prefix}_is_iban_derivable'] = party.is_iban_derivable
                result[f'{prefix}_has_bban_in_iban_country'] = party.has_bban_in_iban_country
                # BIC derivation (9005, 9008)
                result[f'{prefix}_is_bic_derivable'] = party.is_bic_derivable
                result[f'{prefix}_has_nch_no_bic'] = party.has_nch_no_bic
                result[f'{prefix}_has_iban_no_bic'] = party.has_iban_no_bic
                # NCH repair (9000, 9021, 9028)
                result[f'{prefix}_nch_has_dirty_chars'] = party.nch_has_dirty_chars
                result[f'{prefix}_nch_needs_formatting'] = party.nch_needs_formatting
                # Multiple/duplicate (9017, 9018)
                result[f'{prefix}_has_multiple_ids'] = party.has_multiple_ids
                result[f'{prefix}_has_duplicate_info'] = party.has_duplicate_info
                # Account formatting (9022)
                result[f'{prefix}_account_needs_length_fix'] = party.account_needs_length_fix
                # IBAN formatting (9006, 9012)
                result[f'{prefix}_iban_needs_formatting'] = party.iban_needs_formatting
                # Compound ID features (for 8026, 8022)
                result[f'{prefix}_id_has_slash'] = party.id_has_slash
                result[f'{prefix}_id_is_compound'] = party.id_is_compound
                result[f'{prefix}_id_compound_parts'] = party.id_compound_parts
                result[f'{prefix}_id_has_bic_pattern'] = party.id_has_bic_pattern
                result[f'{prefix}_id_has_nch_pattern'] = party.id_has_nch_pattern
                result[f'{prefix}_id_has_bic_and_nch'] = party.id_has_bic_and_nch
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
                result[f'{prefix}_address_country'] = None
                result[f'{prefix}_is_domestic'] = False
                result[f'{prefix}_is_international'] = False
                # NCH/Clearing code features (for 8026, 8895)
                result[f'{prefix}_has_nch'] = False
                result[f'{prefix}_nch_type'] = None
                result[f'{prefix}_nch_valid'] = False
                result[f'{prefix}_fedaba_checksum_valid'] = False
                result[f'{prefix}_has_adr_bank_id'] = False
                result[f'{prefix}_adr_bank_id_type'] = None
                # IBAN requirement features (for 8004)
                result[f'{prefix}_has_iban'] = False
                result[f'{prefix}_needs_iban'] = False
                # NCH source count (for 8026 - inconsistency)
                result[f'{prefix}_nch_sources'] = 0
                # NCH validation applicability (for 8895)
                result[f'{prefix}_nch_validation_applicable'] = False
                # === REPAIR DETECTION FEATURES (for 9XXX) ===
                # Cleaning features (9002, 9009, 9014, 9015)
                result[f'{prefix}_account_has_dirty_chars'] = False
                result[f'{prefix}_name_has_dirty_chars'] = False
                result[f'{prefix}_account_has_spaces'] = False
                result[f'{prefix}_account_has_special_chars'] = False
                # IBAN derivation (9004, 9007)
                result[f'{prefix}_is_iban_derivable'] = False
                result[f'{prefix}_has_bban_in_iban_country'] = False
                # BIC derivation (9005, 9008)
                result[f'{prefix}_is_bic_derivable'] = False
                result[f'{prefix}_has_nch_no_bic'] = False
                result[f'{prefix}_has_iban_no_bic'] = False
                # NCH repair (9000, 9021, 9028)
                result[f'{prefix}_nch_has_dirty_chars'] = False
                result[f'{prefix}_nch_needs_formatting'] = False
                # Multiple/duplicate (9017, 9018)
                result[f'{prefix}_has_multiple_ids'] = False
                result[f'{prefix}_has_duplicate_info'] = False
                # Account formatting (9022)
                result[f'{prefix}_account_needs_length_fix'] = False
                # IBAN formatting (9006, 9012)
                result[f'{prefix}_iban_needs_formatting'] = False
                # Compound ID features (for 8026, 8022)
                result[f'{prefix}_id_has_slash'] = False
                result[f'{prefix}_id_is_compound'] = False
                result[f'{prefix}_id_compound_parts'] = 0
                result[f'{prefix}_id_has_bic_pattern'] = False
                result[f'{prefix}_id_has_nch_pattern'] = False
                result[f'{prefix}_id_has_bic_and_nch'] = False
        
        # === CROSS-PARTY CONSISTENCY FEATURES (for 8022-8033) ===
        # Collect values across all parties for consistency checks
        all_ibans = set()
        all_bbans = set()
        all_nchs = set()
        all_countries = set()
        all_bic4s = set()
        all_accounts = set()
        all_clabes = set()
        all_domestic_accounts = set()
        
        # Track if any party has these features (for global require_true)
        any_has_iban = False
        any_has_bban = False
        any_has_nch = False
        any_has_country = False
        any_has_bic = False
        any_has_account = False
        any_has_clabe = False
        
        for party_type, party in features.parties.items():
            if not party:
                continue
            
            # Collect IBANs
            if party.has_iban and party.account_value:
                any_has_iban = True
                iban = party.account_value.upper().replace(' ', '').replace('-', '')
                all_ibans.add(iban)
                # BBAN is IBAN without country code and check digits (chars 5+)
                if len(iban) > 4:
                    all_bbans.add(iban[4:])
                    any_has_bban = True
            
            # Collect NCH/routing codes
            if party.has_nch and party.nch_value:
                any_has_nch = True
                all_nchs.add(party.nch_value.replace('-', '').replace(' ', ''))
            
            # Collect countries
            if party.country:
                any_has_country = True
                all_countries.add(party.country.upper())
            if party.bic_country:
                any_has_country = True
                all_countries.add(party.bic_country.upper())
            if party.iban_country:
                any_has_country = True
                all_countries.add(party.iban_country.upper())
            
            # Collect BIC4 (first 4 chars of BIC)
            if party.has_bic and party.bic and len(party.bic) >= 4:
                any_has_bic = True
                all_bic4s.add(party.bic[:4].upper())
            
            # Collect account numbers
            if party.has_account and party.account_value:
                any_has_account = True
                all_accounts.add(party.account_value.replace(' ', '').replace('-', ''))
            
            # Collect CLABEs
            if party.is_clabe and party.account_value:
                any_has_clabe = True
                all_clabes.add(party.account_value)
            
            # Collect domestic accounts (non-IBAN)
            if party.has_account and not party.has_iban and party.account_value:
                all_domestic_accounts.add(party.account_value.replace(' ', '').replace('-', ''))
        
        # Global "has_*" features (True if ANY party has the feature)
        result['has_iban'] = any_has_iban
        result['has_bban'] = any_has_bban
        result['has_nch'] = any_has_nch
        result['has_country'] = any_has_country
        result['has_bic'] = any_has_bic
        result['has_account'] = any_has_account
        result['has_clabe'] = any_has_clabe
        
        # Consistency = True if 0 or 1 unique values, False if multiple different values
        result['ibans_consistent'] = len(all_ibans) <= 1
        result['bbans_consistent'] = len(all_bbans) <= 1
        result['nchs_consistent'] = len(all_nchs) <= 1
        result['countries_consistent'] = len(all_countries) <= 1
        result['bic4s_consistent'] = len(all_bic4s) <= 1
        result['accounts_consistent'] = len(all_accounts) <= 1
        result['clabes_consistent'] = len(all_clabes) <= 1
        result['domestic_accounts_consistent'] = len(all_domestic_accounts) <= 1
        
        # Also track if we have any of these for the consistency checks to be meaningful
        result['has_multiple_ibans'] = len(all_ibans) > 1
        result['has_multiple_nchs'] = len(all_nchs) > 1
        result['has_multiple_countries'] = len(all_countries) > 1
        result['has_multiple_bic4s'] = len(all_bic4s) > 1
        
        # === CROSS-PARTY RELATIONSHIP FEATURES (for 8022, 8026) ===
        # 8022: IBAN inconsistent with Account With Institution BIC
        # ACE message: "BNFBNK IBAN inconsistent with Account With Institution BIC"
        # This checks if WireKey BIC country matches BeneficiaryParty's IBAN country
        bnf = features.parties.get('BeneficiaryBankInfo')
        bnp = features.parties.get('BeneficiaryPartyInfo')
        cdt = features.parties.get('CreditPartyInfo')
        
        # For 8022: Check WireKey BIC (Account With Institution) against BNP's IBAN
        result['bnf_bic_bnp_iban_match'] = None  # Default
        
        if cdt and bnp:
            # CreditPartyInf's WireKey BIC vs BeneficiaryPartyInf's IBAN
            if cdt.has_wirekey_bic and cdt.wirekey_bic_country and bnp.has_iban and bnp.iban_country:
                result['wirekey_bic_bnp_iban_match'] = (cdt.wirekey_bic_country == bnp.iban_country)
            else:
                result['wirekey_bic_bnp_iban_match'] = None
        else:
            result['wirekey_bic_bnp_iban_match'] = None
            
        # Also check BNF's BIC against BNP's IBAN (alternate 8022 trigger)
        if bnf and bnp:
            if bnf.has_bic and bnf.bic_country and bnp.has_iban and bnp.iban_country:
                result['bnf_bic_bnp_iban_match'] = (bnf.bic_country == bnp.iban_country)
        
        # 8026: NCH inconsistency - check if AdrBankID conflicts with BIC country
        # If a party has AdrBankID (potential NCH) AND BIC from different country, 
        # ACE derives NCH from BIC which conflicts with AdrBankID
        for party_type, party in features.parties.items():
            if not party:
                continue
            prefix = self._party_type_to_prefix(party_type)
            
            # Check for AdrBankID + BIC country conflict
            # AdrBankID "121" with Finnish BIC suggests NCH inconsistency
            if party.has_adr_bank_id and party.has_bic and party.bic_country:
                # AdrBankID with BIC from non-US country = likely inconsistency
                # ACE derives NCH from BIC, which won't match the AdrBankID
                result[f'{prefix}_nch_consistent_with_bic'] = False  # False = inconsistent
                result[f'{prefix}_has_nch_inconsistency'] = True
            else:
                result[f'{prefix}_nch_consistent_with_bic'] = True  # No conflict
                result[f'{prefix}_has_nch_inconsistency'] = False
        
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
            'BeneficiaryPartyInfo': 'bnp',  # Added for BNPPTY
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
