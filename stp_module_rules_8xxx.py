#!/usr/bin/env python3
"""
================================================================================
8XXX VALIDATION RULES - Deterministic Error Code Checks
================================================================================

PURPOSE:
    Implements deterministic decision trees for 8XXX ACE validation errors.
    These codes indicate something is WRONG with the input data.
    All 8XXX codes are fully deterministic - no directory lookups required.

CODES IMPLEMENTED:
    8001 - Invalid BIC (format or country)
    8004 - IBAN cannot be derived
    8005 - Invalid BIC4 (first 4 chars)
    8006 - Invalid country code
    8007 - Fractional digits exceed maximum
    8022 - IBAN inconsistent with BIC (country mismatch)
    8023 - IBAN inconsistency in message (multiple different IBANs)
    8024 - BBAN inconsistency in message
    8025 - Domestic account inconsistency
    8026 - NCH inconsistency in message
    8027 - ISO country code inconsistency
    8028 - BIC4 inconsistency in message
    8029 - Account number inconsistency
    8030 - IBAN derivation not supported for country
    8033 - CLABE inconsistency (Mexico)
    8124 - Invalid currency
    8851 - Incorrect field size
    8852 - Incorrect length of attribute
    8894 - Invalid IBAN (format or checksum)
    8895 - Invalid NCH code (routing number)
    8898 - IBAN check digit failed

USAGE:
    from rules_8xxx import Rules8XXX
    
    rules = Rules8XXX()
    results = rules.check_all(features_dict, parties_dict)
    
    # Or check individual codes:
    result = rules.check_8001(party_features, party_name)

AUTHOR: ACE Pelican ML Team
VERSION: 1.0 (December 2025)
================================================================================
"""

import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field


# =============================================================================
# REFERENCE DATA
# =============================================================================

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

# Valid ISO 4217 currency codes
VALID_CURRENCY_CODES = {
    'AED', 'AFN', 'ALL', 'AMD', 'ANG', 'AOA', 'ARS', 'AUD', 'AWG', 'AZN',
    'BAM', 'BBD', 'BDT', 'BGN', 'BHD', 'BIF', 'BMD', 'BND', 'BOB', 'BRL',
    'BSD', 'BTN', 'BWP', 'BYN', 'BZD', 'CAD', 'CDF', 'CHF', 'CLP', 'CNY',
    'COP', 'CRC', 'CUC', 'CUP', 'CVE', 'CZK', 'DJF', 'DKK', 'DOP', 'DZD',
    'EGP', 'ERN', 'ETB', 'EUR', 'FJD', 'FKP', 'GBP', 'GEL', 'GHS', 'GIP',
    'GMD', 'GNF', 'GTQ', 'GYD', 'HKD', 'HNL', 'HRK', 'HTG', 'HUF', 'IDR',
    'ILS', 'INR', 'IQD', 'IRR', 'ISK', 'JMD', 'JOD', 'JPY', 'KES', 'KGS',
    'KHR', 'KMF', 'KPW', 'KRW', 'KWD', 'KYD', 'KZT', 'LAK', 'LBP', 'LKR',
    'LRD', 'LSL', 'LYD', 'MAD', 'MDL', 'MGA', 'MKD', 'MMK', 'MNT', 'MOP',
    'MRU', 'MUR', 'MVR', 'MWK', 'MXN', 'MYR', 'MZN', 'NAD', 'NGN', 'NIO',
    'NOK', 'NPR', 'NZD', 'OMR', 'PAB', 'PEN', 'PGK', 'PHP', 'PKR', 'PLN',
    'PYG', 'QAR', 'RON', 'RSD', 'RUB', 'RWF', 'SAR', 'SBD', 'SCR', 'SDG',
    'SEK', 'SGD', 'SHP', 'SLL', 'SOS', 'SRD', 'SSP', 'STN', 'SVC', 'SYP',
    'SZL', 'THB', 'TJS', 'TMT', 'TND', 'TOP', 'TRY', 'TTD', 'TWD', 'TZS',
    'UAH', 'UGX', 'USD', 'UYU', 'UZS', 'VES', 'VND', 'VUV', 'WST', 'XAF',
    'XCD', 'XOF', 'XPF', 'YER', 'ZAR', 'ZMW', 'ZWL'
}

# Currency decimal places (ISO 4217)
CURRENCY_DECIMALS = {
    # 0 decimal places
    'BIF': 0, 'CLP': 0, 'DJF': 0, 'GNF': 0, 'ISK': 0, 'JPY': 0, 'KMF': 0,
    'KRW': 0, 'PYG': 0, 'RWF': 0, 'UGX': 0, 'VND': 0, 'VUV': 0, 'XAF': 0,
    'XOF': 0, 'XPF': 0,
    # 3 decimal places
    'BHD': 3, 'IQD': 3, 'JOD': 3, 'KWD': 3, 'LYD': 3, 'OMR': 3, 'TND': 3,
    # Default is 2 decimal places
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

# Countries that require IBAN (SEPA + others)
IBAN_REQUIRED_COUNTRIES = set(IBAN_LENGTHS.keys())

# Countries that support IBAN derivation from domestic account
IBAN_DERIVATION_SUPPORTED = {
    'AT', 'BE', 'CH', 'CZ', 'DE', 'DK', 'EE', 'ES', 'FI', 'FR', 'GB', 'GR',
    'HR', 'HU', 'IE', 'IT', 'LT', 'LU', 'LV', 'NL', 'NO', 'PL', 'PT', 'RO',
    'SE', 'SI', 'SK'
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class CodeCheckResult:
    """
    Result of checking a single error code.
    
    Attributes:
        code: The error code (e.g., "8001")
        fires: True if code WILL fire (deterministic)
        eligible: True if eligible for directory lookup (9XXX only)
        cannot_fire: True if preconditions not met
        confidence: 1.0 for deterministic, 0.7-0.9 for directory-dependent
        decision_path: List of decision steps taken
        party: Which party triggered the code (if applicable)
        features_used: Feature values that influenced the decision
        description: Human-readable description of what the code means
    """
    code: str
    fires: bool = False
    eligible: bool = False
    cannot_fire: bool = False
    confidence: float = 1.0
    decision_path: List[str] = field(default_factory=list)
    party: Optional[str] = None
    features_used: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'code': self.code,
            'fires': self.fires,
            'eligible': self.eligible,
            'cannot_fire': self.cannot_fire,
            'confidence': self.confidence,
            'decision_path': self.decision_path,
            'party': self.party,
            'features_used': self.features_used,
            'description': self.description
        }


# =============================================================================
# 8XXX VALIDATION RULES CLASS
# =============================================================================

class Rules8XXX:
    """
    Deterministic rules for 8XXX validation error codes.
    
    All 8XXX codes are fully deterministic from input features.
    No directory lookups required.
    
    Usage:
        rules = Rules8XXX()
        
        # Check all codes for all parties
        results = rules.check_all(features_dict, parties_dict)
        
        # Check single code for single party
        result = rules.check_8001(party_features, "BNFBNK")
    """
    
    # Party type to code suffix mapping
    PARTY_SUFFIXES = {
        'OriginatingParty': 'ORGPTY',
        'SendingBank': 'SNDBNK',
        'DebitParty': 'DBTPTY',
        'CreditParty': 'CDTPTY',
        'IntermediaryBank': 'INTMBNK',
        'BeneficiaryBank': 'BNFBNK',
        'BeneficiaryParty': 'BNPPTY',
        'AccountWithInstitution': 'ACWI',
        'OrderingInstitution': 'ORDI',
    }
    
    # Feature prefix mapping (short prefixes used in flat feature dict)
    # NOTE: Using only the short form names to avoid duplicates
    PARTY_PREFIXES = {
        'OriginatingParty': 'orig',
        'SendingBank': 'send',
        'DebitParty': 'dbt',
        'CreditParty': 'cdt',
        'IntermediaryBank': 'intm',
        'BeneficiaryBank': 'bnf',
        'BeneficiaryParty': 'bnp',
        'AccountWithInstitution': 'acwi',
        'OrderingInstitution': 'ordi',
    }
    
    def __init__(self):
        """Initialize the 8XXX rules engine."""
        pass
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _get_party_features(self, features: Dict, party_type: str) -> Dict:
        """
        Extract features for a specific party from flat feature dict.
        
        Args:
            features: Flat feature dictionary with prefixed keys
            party_type: Party type name (e.g., "BeneficiaryBankInfo")
            
        Returns:
            Dict with party-specific features (prefix removed)
        """
        prefix = self.PARTY_PREFIXES.get(party_type, '')
        if not prefix:
            return {}
        
        prefix_with_underscore = f"{prefix}_"
        party_features = {}
        
        for key, value in features.items():
            if key.startswith(prefix_with_underscore):
                # Remove prefix to get base feature name
                base_key = key[len(prefix_with_underscore):]
                party_features[base_key] = value
            # Also include non-prefixed keys (global features)
            elif '_' not in key or not any(key.startswith(p + '_') for p in self.PARTY_PREFIXES.values()):
                party_features[key] = value
        
        return party_features
    
    def _get_suffix(self, party_type: str) -> str:
        """Get code suffix for party type."""
        return self.PARTY_SUFFIXES.get(party_type, 'UNKNOWN')
    
    def _validate_bic_format(self, bic: str) -> Tuple[bool, bool, str]:
        """
        Validate BIC format and country.
        
        Returns:
            Tuple of (format_valid, country_valid, country_code)
        """
        if not bic:
            return False, False, ''
        
        bic = bic.upper().strip()
        
        # Check length (8 or 11)
        if len(bic) not in (8, 11):
            return False, False, ''
        
        # Check format pattern
        pattern = r'^[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}([A-Z0-9]{3})?$'
        format_valid = bool(re.match(pattern, bic))
        
        # Extract and validate country
        country_code = bic[4:6] if len(bic) >= 6 else ''
        country_valid = country_code in VALID_COUNTRY_CODES
        
        return format_valid, country_valid, country_code
    
    def _validate_iban(self, iban: str) -> Tuple[bool, bool, str]:
        """
        Validate IBAN format and checksum.
        
        Returns:
            Tuple of (format_valid, checksum_valid, country_code)
        """
        if not iban:
            return False, False, ''
        
        iban = iban.upper().replace(' ', '').replace('-', '')
        
        if len(iban) < 5:
            return False, False, ''
        
        country_code = iban[:2]
        if not country_code.isalpha():
            return False, False, ''
        
        # Check length for country
        expected_length = IBAN_LENGTHS.get(country_code)
        if expected_length:
            format_valid = len(iban) == expected_length
        else:
            format_valid = 15 <= len(iban) <= 34
        
        # Validate checksum (mod-97)
        try:
            rearranged = iban[4:] + iban[:4]
            numeric = ''
            for char in rearranged:
                if char.isdigit():
                    numeric += char
                elif char.isalpha():
                    numeric += str(ord(char) - ord('A') + 10)
                else:
                    return format_valid, False, country_code
            
            checksum_valid = int(numeric) % 97 == 1
        except (ValueError, OverflowError):
            checksum_valid = False
        
        return format_valid, checksum_valid, country_code
    
    def _validate_fedaba(self, aba: str) -> Tuple[bool, bool]:
        """
        Validate US ABA routing number.
        
        Returns:
            Tuple of (format_valid, checksum_valid)
        """
        if not aba:
            return False, False
        
        aba = aba.strip().replace('-', '').replace(' ', '')
        
        # Must be 9 digits
        if len(aba) != 9 or not aba.isdigit():
            return False, False
        
        format_valid = True
        
        # ABA checksum: weights [3,7,1,3,7,1,3,7,1], sum mod 10 = 0
        try:
            weights = [3, 7, 1, 3, 7, 1, 3, 7, 1]
            total = sum(int(d) * w for d, w in zip(aba, weights))
            checksum_valid = (total % 10 == 0)
        except (ValueError, TypeError):
            checksum_valid = False
        
        return format_valid, checksum_valid
    
    # =========================================================================
    # INDIVIDUAL CODE CHECKS
    # =========================================================================
    
    def check_8001(self, features: Dict, party: str = None) -> CodeCheckResult:
        """
        Check 8001: Invalid BIC
        
        Decision Tree:
            has_bic = True?
            ├─ NO → Cannot fire
            └─ YES → bic_valid_format?
                      ├─ NO → 8001 FIRES
                      └─ YES → bic_valid_country?
                                ├─ NO → 8001 FIRES
                                └─ YES → Cannot fire
        """
        result = CodeCheckResult(
            code='8001',
            party=party,
            description='Invalid BIC - format or country code invalid'
        )
        
        has_bic = features.get('has_bic', False)
        bic_value = features.get('bic') or features.get('bic_value', '')
        
        result.features_used = {
            'has_bic': has_bic,
            'bic_value': bic_value
        }
        
        # Step 1: Check if BIC present
        if not has_bic or not bic_value:
            result.cannot_fire = True
            result.decision_path = ['has_bic=False', '→ Cannot fire (no BIC to validate)']
            return result
        
        result.decision_path.append(f'has_bic=True (bic={bic_value})')
        
        # Step 2: Validate BIC
        format_valid, country_valid, country_code = self._validate_bic_format(bic_value)
        
        result.features_used['bic_valid_format'] = format_valid
        result.features_used['bic_valid_country'] = country_valid
        result.features_used['bic_country'] = country_code
        
        if not format_valid:
            result.fires = True
            result.decision_path.append(f'bic_valid_format=False')
            result.decision_path.append('→ 8001 FIRES (invalid BIC format)')
            return result
        
        result.decision_path.append(f'bic_valid_format=True')
        
        if not country_valid:
            result.fires = True
            result.decision_path.append(f'bic_valid_country=False (country={country_code})')
            result.decision_path.append('→ 8001 FIRES (invalid country in BIC)')
            return result
        
        result.decision_path.append(f'bic_valid_country=True (country={country_code})')
        result.cannot_fire = True
        result.decision_path.append('→ Cannot fire (BIC is valid)')
        
        return result
    
    def check_8004(self, features: Dict, party: str = None) -> CodeCheckResult:
        """
        Check 8004: IBAN Cannot Be Derived
        
        ACTUAL ACE BEHAVIOR:
        8004 fires when:
        1. Destination country requires IBAN (is an IBAN country)
        2. No valid IBAN is provided
        3. ACE cannot derive IBAN from available account info
        
        Decision Tree:
            is_iban_country? (destination requires IBAN)
            ├─ NO → Cannot fire
            └─ YES → has_valid_iban?
                      ├─ YES → Cannot fire
                      └─ NO → can_derive_iban? (has account + BIC/NCH for lookup)
                               ├─ YES → ELIGIBLE (directory lookup needed)
                               └─ NO → 8004 FIRES
        
        NOTE: This is directory-dependent. We can predict eligibility but 
              not the final outcome since derivation may fail.
        """
        result = CodeCheckResult(
            code='8004',
            party=party,
            description='IBAN cannot be derived - required but not provided or derivable'
        )
        
        # Get country - check multiple sources
        country = (features.get('country') or 
                   features.get('address_country') or 
                   features.get('iban_country') or
                   features.get('bic_country') or '')
        
        has_iban = features.get('has_iban', False)
        iban_value = features.get('iban') or features.get('account_value', '')
        iban_valid = features.get('iban_valid_format', False) and features.get('iban_checksum_valid', False)
        
        has_account = features.get('has_account', False)
        account_value = features.get('account_value', '')
        
        has_bic = features.get('has_bic', False)
        has_nch = features.get('has_nch', False)
        
        # Check if IBAN is actually present and valid
        if has_iban and iban_value:
            # Validate IBAN format
            iban_clean = iban_value.upper().replace(' ', '').replace('-', '')
            if len(iban_clean) >= 15 and iban_clean[:2].isalpha():
                iban_valid = True
        
        # Check if country requires IBAN
        is_iban_country = country.upper() in IBAN_REQUIRED_COUNTRIES if country else False
        
        result.features_used = {
            'country': country,
            'is_iban_country': is_iban_country,
            'has_iban': has_iban,
            'iban_valid': iban_valid,
            'has_account': has_account,
            'has_bic': has_bic,
            'has_nch': has_nch
        }
        
        # Step 1: Check if destination requires IBAN
        if not is_iban_country:
            result.cannot_fire = True
            result.decision_path = [f'is_iban_country=False (country={country})', 
                                    '→ Cannot fire (IBAN not required for this country)']
            return result
        
        result.decision_path.append(f'is_iban_country=True (country={country})')
        
        # Step 2: Check if valid IBAN already present
        if has_iban and iban_valid:
            result.cannot_fire = True
            result.decision_path.append('has_valid_iban=True')
            result.decision_path.append('→ Cannot fire (valid IBAN present)')
            return result
        
        result.decision_path.append(f'has_valid_iban=False')
        
        # Step 3: Check if IBAN can be derived
        # Derivation requires: account number + (BIC or NCH) for directory lookup
        can_derive = has_account and (has_bic or has_nch)
        derivation_supported = country.upper() in IBAN_DERIVATION_SUPPORTED if country else False
        
        result.features_used['can_derive'] = can_derive
        result.features_used['derivation_supported'] = derivation_supported
        
        if can_derive and derivation_supported:
            # Has the ingredients for derivation - outcome depends on directory
            result.eligible = True
            result.confidence = 0.70  # ~70% of eligible cases result in 8004
            result.decision_path.append(f'can_derive=True, derivation_supported=True')
            result.decision_path.append('→ ELIGIBLE (directory lookup will determine outcome)')
            return result
        
        # Cannot derive - 8004 will fire
        result.fires = True
        if not has_account:
            result.decision_path.append('has_account=False')
            result.decision_path.append('→ 8004 FIRES (no account to derive IBAN from)')
        elif not derivation_supported:
            result.decision_path.append(f'derivation_supported=False (country={country})')
            result.decision_path.append('→ 8004 FIRES (IBAN derivation not supported for country)')
        else:
            result.decision_path.append('can_derive=False (missing BIC/NCH for lookup)')
            result.decision_path.append('→ 8004 FIRES (cannot perform directory lookup)')
        
        return result
    
    def check_8005(self, features: Dict, party: str = None) -> CodeCheckResult:
        """
        Check 8005: Invalid BIC4 (first 4 characters)
        
        Decision Tree:
            has_bic = True?
            ├─ NO → Cannot fire
            └─ YES → bic_length >= 4?
                      ├─ NO → 8005 FIRES
                      └─ YES → bic4_valid? (first 4 chars are letters)
                                ├─ NO → 8005 FIRES
                                └─ YES → Cannot fire
        """
        result = CodeCheckResult(
            code='8005',
            party=party,
            description='Invalid BIC4 - first 4 characters of BIC invalid'
        )
        
        has_bic = features.get('has_bic', False)
        bic_value = features.get('bic') or features.get('bic_value', '')
        
        result.features_used = {
            'has_bic': has_bic,
            'bic_value': bic_value
        }
        
        if not has_bic or not bic_value:
            result.cannot_fire = True
            result.decision_path = ['has_bic=False', '→ Cannot fire (no BIC)']
            return result
        
        result.decision_path.append(f'has_bic=True (bic={bic_value})')
        
        bic = bic_value.upper().strip()
        bic_length = len(bic)
        
        result.features_used['bic_length'] = bic_length
        
        if bic_length < 4:
            result.fires = True
            result.decision_path.append(f'bic_length={bic_length} < 4')
            result.decision_path.append('→ 8005 FIRES (BIC too short for BIC4)')
            return result
        
        result.decision_path.append(f'bic_length={bic_length} >= 4')
        
        # Check if first 4 chars are letters
        bic4 = bic[:4]
        bic4_valid = bic4.isalpha()
        
        result.features_used['bic4'] = bic4
        result.features_used['bic4_valid'] = bic4_valid
        
        if not bic4_valid:
            result.fires = True
            result.decision_path.append(f'bic4_valid=False (bic4={bic4})')
            result.decision_path.append('→ 8005 FIRES (BIC4 contains non-letters)')
            return result
        
        result.decision_path.append(f'bic4_valid=True (bic4={bic4})')
        result.cannot_fire = True
        result.decision_path.append('→ Cannot fire (BIC4 is valid)')
        
        return result
    
    def check_8006(self, features: Dict, party: str = None) -> CodeCheckResult:
        """
        Check 8006: Invalid Country Code
        
        Decision Tree:
            has_country? (any country field)
            ├─ NO → Cannot fire
            └─ YES → country in ISO 3166-1?
                      ├─ YES → Cannot fire
                      └─ NO → 8006 FIRES
        """
        result = CodeCheckResult(
            code='8006',
            party=party,
            description='Invalid country code - not a valid ISO 3166-1 alpha-2 code'
        )
        
        # Check all possible country sources
        country = features.get('country', '')
        bic_country = features.get('bic_country', '')
        iban_country = features.get('iban_country', '')
        address_country = features.get('address_country', '')
        
        result.features_used = {
            'country': country,
            'bic_country': bic_country,
            'iban_country': iban_country,
            'address_country': address_country
        }
        
        # Collect all country values
        countries = [c for c in [country, bic_country, iban_country, address_country] if c]
        
        if not countries:
            result.cannot_fire = True
            result.decision_path = ['has_country=False', '→ Cannot fire (no country to validate)']
            return result
        
        result.decision_path.append(f'has_country=True (countries={countries})')
        
        # Check each country for validity
        invalid_countries = [c for c in countries if c not in VALID_COUNTRY_CODES]
        
        if invalid_countries:
            result.fires = True
            result.decision_path.append(f'invalid_countries={invalid_countries}')
            result.decision_path.append('→ 8006 FIRES (invalid country code found)')
            result.features_used['invalid_countries'] = invalid_countries
            return result
        
        result.cannot_fire = True
        result.decision_path.append('all_countries_valid=True')
        result.decision_path.append('→ Cannot fire (all country codes valid)')
        
        return result
    
    def check_8007(self, features: Dict, party: str = None) -> CodeCheckResult:
        """
        Check 8007: Fractional Digits Exceed Maximum
        
        Decision Tree:
            has_amount = True?
            ├─ NO → Cannot fire
            └─ YES → Get allowed decimals for currency
                      └─ actual_decimals > allowed_decimals?
                                ├─ NO → Cannot fire
                                └─ YES → 8007 FIRES
        """
        result = CodeCheckResult(
            code='8007',
            party=party,
            description='Fractional digits exceed maximum allowed for currency'
        )
        
        has_amount = features.get('has_amount', False) or features.get('primary_amount', 0) > 0
        currency = features.get('primary_currency', '')
        amount = features.get('primary_amount', 0)
        
        result.features_used = {
            'has_amount': has_amount,
            'currency': currency,
            'amount': amount
        }
        
        if not has_amount or not amount:
            result.cannot_fire = True
            result.decision_path = ['has_amount=False', '→ Cannot fire (no amount)']
            return result
        
        result.decision_path.append(f'has_amount=True (amount={amount}, currency={currency})')
        
        if not currency:
            result.cannot_fire = True
            result.decision_path.append('currency=None')
            result.decision_path.append('→ Cannot fire (no currency to check decimals)')
            return result
        
        # Get allowed decimals for currency
        allowed_decimals = CURRENCY_DECIMALS.get(currency.upper(), 2)
        result.features_used['allowed_decimals'] = allowed_decimals
        
        # Count actual decimal places
        amount_str = str(amount)
        if '.' in amount_str:
            actual_decimals = len(amount_str.split('.')[1])
        else:
            actual_decimals = 0
        
        result.features_used['actual_decimals'] = actual_decimals
        result.decision_path.append(f'allowed_decimals={allowed_decimals}, actual_decimals={actual_decimals}')
        
        if actual_decimals > allowed_decimals:
            result.fires = True
            result.decision_path.append('→ 8007 FIRES (too many decimal places)')
            return result
        
        result.cannot_fire = True
        result.decision_path.append('→ Cannot fire (decimal places within limit)')
        
        return result
    
    def check_8022(self, features: Dict, party: str = None) -> CodeCheckResult:
        """
        Check 8022: IBAN Inconsistent with BIC
        
        Decision Tree:
            has_iban = True?
            ├─ NO → Cannot fire
            └─ YES → has_bic = True?
                      ├─ NO → Cannot fire
                      └─ YES → Extract countries:
                                iban_country = iban[0:2]
                                bic_country = bic[4:6]
                                └─ countries match?
                                     ├─ YES → Cannot fire
                                     └─ NO → 8022 FIRES
        """
        result = CodeCheckResult(
            code='8022',
            party=party,
            description='IBAN inconsistent with BIC - country codes do not match'
        )
        
        has_iban = features.get('has_iban', False)
        has_bic = features.get('has_bic', False)
        iban_value = features.get('iban') or features.get('account_value', '')
        bic_value = features.get('bic') or features.get('bic_value', '')
        
        result.features_used = {
            'has_iban': has_iban,
            'has_bic': has_bic,
            'iban_value': iban_value,
            'bic_value': bic_value
        }
        
        # Step 1: Check IBAN present
        if not has_iban:
            result.cannot_fire = True
            result.decision_path = ['has_iban=False', '→ Cannot fire (no IBAN)']
            return result
        
        result.decision_path.append('has_iban=True')
        
        # Step 2: Check BIC present
        if not has_bic:
            result.cannot_fire = True
            result.decision_path.append('has_bic=False')
            result.decision_path.append('→ Cannot fire (no BIC to compare)')
            return result
        
        result.decision_path.append('has_bic=True')
        
        # Step 3: Extract and compare countries
        iban_country = features.get('iban_country', '')
        bic_country = features.get('bic_country', '')
        
        # If not in features, extract from values
        if not iban_country and iban_value:
            iban_clean = iban_value.upper().replace(' ', '').replace('-', '')
            if len(iban_clean) >= 2:
                iban_country = iban_clean[:2]
        
        if not bic_country and bic_value:
            bic_clean = bic_value.upper().strip()
            if len(bic_clean) >= 6:
                bic_country = bic_clean[4:6]
        
        result.features_used['iban_country'] = iban_country
        result.features_used['bic_country'] = bic_country
        
        result.decision_path.append(f'iban_country={iban_country}, bic_country={bic_country}')
        
        # Check if both countries are valid before comparing
        if not iban_country or not bic_country:
            result.cannot_fire = True
            result.decision_path.append('→ Cannot fire (missing country code)')
            return result
        
        # Compare countries
        if iban_country == bic_country:
            result.cannot_fire = True
            result.decision_path.append('bic_iban_match=True')
            result.decision_path.append('→ Cannot fire (countries match)')
            return result
        
        result.fires = True
        result.decision_path.append('bic_iban_match=False')
        result.decision_path.append(f'→ 8022 FIRES ({iban_country} != {bic_country})')
        
        return result
    
    def check_8023(self, features: Dict, global_features: Dict = None) -> CodeCheckResult:
        """
        Check 8023: IBAN Inconsistency in Message
        
        Decision Tree:
            Count IBANs across all parties
            └─ iban_count > 1?
                ├─ NO → Cannot fire
                └─ YES → All IBAN values identical?
                          ├─ YES → Cannot fire
                          └─ NO → 8023 FIRES
        
        Note: This is a cross-party check requiring aggregated features.
        """
        result = CodeCheckResult(
            code='8023',
            party=None,  # Cross-party check
            description='IBAN inconsistency - multiple different IBANs in message'
        )
        
        gf = global_features or features
        
        iban_count = gf.get('iban_count', 0)
        ibans_consistent = gf.get('ibans_consistent', True)
        iban_values = gf.get('iban_values', [])
        
        result.features_used = {
            'iban_count': iban_count,
            'ibans_consistent': ibans_consistent,
            'iban_values': iban_values
        }
        
        if iban_count <= 1:
            result.cannot_fire = True
            result.decision_path = [f'iban_count={iban_count} <= 1', '→ Cannot fire (0 or 1 IBAN)']
            return result
        
        result.decision_path.append(f'iban_count={iban_count} > 1')
        
        if ibans_consistent:
            result.cannot_fire = True
            result.decision_path.append('ibans_consistent=True')
            result.decision_path.append('→ Cannot fire (all IBANs identical)')
            return result
        
        result.fires = True
        result.decision_path.append('ibans_consistent=False')
        result.decision_path.append('→ 8023 FIRES (different IBANs in message)')
        
        return result
    
    def check_8024(self, features: Dict, global_features: Dict = None) -> CodeCheckResult:
        """
        Check 8024: BBAN Inconsistency in Message
        
        Similar logic to 8023 but for BBAN values.
        """
        result = CodeCheckResult(
            code='8024',
            party=None,
            description='BBAN inconsistency - multiple different BBANs in message'
        )
        
        gf = global_features or features
        
        bban_count = gf.get('bban_count', 0)
        bbans_consistent = gf.get('bbans_consistent', True)
        
        result.features_used = {
            'bban_count': bban_count,
            'bbans_consistent': bbans_consistent
        }
        
        if bban_count <= 1:
            result.cannot_fire = True
            result.decision_path = [f'bban_count={bban_count} <= 1', '→ Cannot fire']
            return result
        
        result.decision_path.append(f'bban_count={bban_count} > 1')
        
        if bbans_consistent:
            result.cannot_fire = True
            result.decision_path.append('bbans_consistent=True')
            result.decision_path.append('→ Cannot fire')
            return result
        
        result.fires = True
        result.decision_path.append('bbans_consistent=False')
        result.decision_path.append('→ 8024 FIRES')
        
        return result
    
    def check_8025(self, features: Dict, global_features: Dict = None) -> CodeCheckResult:
        """
        Check 8025: Domestic Account Number Inconsistency
        """
        result = CodeCheckResult(
            code='8025',
            party=None,
            description='Domestic account inconsistency - multiple different accounts'
        )
        
        gf = global_features or features
        
        account_count = gf.get('domestic_account_count', 0)
        accounts_consistent = gf.get('domestic_accounts_consistent', True)
        
        result.features_used = {
            'domestic_account_count': account_count,
            'domestic_accounts_consistent': accounts_consistent
        }
        
        if account_count <= 1:
            result.cannot_fire = True
            result.decision_path = [f'domestic_account_count={account_count} <= 1', '→ Cannot fire']
            return result
        
        result.decision_path.append(f'domestic_account_count={account_count} > 1')
        
        if accounts_consistent:
            result.cannot_fire = True
            result.decision_path.append('→ Cannot fire (accounts consistent)')
            return result
        
        result.fires = True
        result.decision_path.append('→ 8025 FIRES')
        
        return result
    
    def check_8026(self, features: Dict, global_features: Dict = None) -> CodeCheckResult:
        """
        Check 8026: NCH Inconsistency in Message
        
        Decision Tree:
            has_nch = True?
            ├─ NO → Cannot fire
            └─ YES → Check inconsistency sources:
                      │
                      │  SOURCE 1: Multiple NCH values across parties
                      │  nch_count > 1 AND NOT nchs_consistent → FIRES
                      │
                      │  SOURCE 2: Compound ID with embedded NCH
                      │  id_has_bic_and_nch = True → HIGH RISK
                      │
                      │  SOURCE 3: Multiple NCH sources within party
                      │  nch_sources > 1 → ELEVATED RISK
        """
        result = CodeCheckResult(
            code='8026',
            party=None,
            description='NCH inconsistency - conflicting routing codes in message'
        )
        
        gf = global_features or features
        
        has_nch = gf.get('has_nch', False)
        nch_count = gf.get('nch_count', 0)
        nchs_consistent = gf.get('nchs_consistent', True)
        id_has_bic_and_nch = gf.get('id_has_bic_and_nch', False)
        nch_sources = gf.get('nch_sources', 0)
        id_is_compound = gf.get('id_is_compound', False)
        
        result.features_used = {
            'has_nch': has_nch,
            'nch_count': nch_count,
            'nchs_consistent': nchs_consistent,
            'id_has_bic_and_nch': id_has_bic_and_nch,
            'nch_sources': nch_sources,
            'id_is_compound': id_is_compound
        }
        
        # Check if any NCH present
        if not has_nch and nch_count == 0:
            result.cannot_fire = True
            result.decision_path = ['has_nch=False', '→ Cannot fire (no NCH)']
            return result
        
        result.decision_path.append('has_nch=True')
        
        # Source 1: Multiple NCH values not consistent
        if nch_count > 1 and not nchs_consistent:
            result.fires = True
            result.decision_path.append(f'nch_count={nch_count} > 1')
            result.decision_path.append('nchs_consistent=False')
            result.decision_path.append('→ 8026 FIRES (multiple inconsistent NCH values)')
            return result
        
        # Source 2: Compound ID with both BIC and NCH (high risk)
        if id_has_bic_and_nch:
            result.fires = True
            result.confidence = 0.9  # High confidence but not 100%
            result.decision_path.append('id_has_bic_and_nch=True')
            result.decision_path.append('→ 8026 FIRES (compound ID with BIC and NCH)')
            return result
        
        # Source 3: Multiple NCH sources within party
        if nch_sources > 1:
            result.fires = True
            result.confidence = 0.8  # Elevated risk
            result.decision_path.append(f'nch_sources={nch_sources} > 1')
            result.decision_path.append('→ 8026 FIRES (multiple NCH sources)')
            return result
        
        result.cannot_fire = True
        result.decision_path.append('No inconsistency detected')
        result.decision_path.append('→ Cannot fire')
        
        return result
    
    def check_8027(self, features: Dict, global_features: Dict = None) -> CodeCheckResult:
        """
        Check 8027: ISO Country Code Inconsistency
        """
        result = CodeCheckResult(
            code='8027',
            party=None,
            description='ISO country code inconsistency in message'
        )
        
        gf = global_features or features
        
        country_count = gf.get('country_count', 0)
        countries_consistent = gf.get('countries_consistent', True)
        
        result.features_used = {
            'country_count': country_count,
            'countries_consistent': countries_consistent
        }
        
        if country_count <= 1:
            result.cannot_fire = True
            result.decision_path = [f'country_count={country_count} <= 1', '→ Cannot fire']
            return result
        
        result.decision_path.append(f'country_count={country_count} > 1')
        
        if countries_consistent:
            result.cannot_fire = True
            result.decision_path.append('→ Cannot fire (countries consistent)')
            return result
        
        result.fires = True
        result.decision_path.append('→ 8027 FIRES')
        
        return result
    
    def check_8028(self, features: Dict, global_features: Dict = None) -> CodeCheckResult:
        """
        Check 8028: BIC4 Inconsistency in Message
        """
        result = CodeCheckResult(
            code='8028',
            party=None,
            description='BIC4 inconsistency - multiple different BIC4 values'
        )
        
        gf = global_features or features
        
        bic4_count = gf.get('bic4_count', 0)
        bic4s_consistent = gf.get('bic4s_consistent', True)
        
        result.features_used = {
            'bic4_count': bic4_count,
            'bic4s_consistent': bic4s_consistent
        }
        
        if bic4_count <= 1:
            result.cannot_fire = True
            result.decision_path = [f'bic4_count={bic4_count} <= 1', '→ Cannot fire']
            return result
        
        result.decision_path.append(f'bic4_count={bic4_count} > 1')
        
        if bic4s_consistent:
            result.cannot_fire = True
            result.decision_path.append('→ Cannot fire')
            return result
        
        result.fires = True
        result.decision_path.append('→ 8028 FIRES')
        
        return result
    
    def check_8029(self, features: Dict, global_features: Dict = None) -> CodeCheckResult:
        """
        Check 8029: Account Number Inconsistency
        """
        result = CodeCheckResult(
            code='8029',
            party=None,
            description='Account number inconsistency in message'
        )
        
        gf = global_features or features
        
        account_count = gf.get('account_count', 0)
        accounts_consistent = gf.get('accounts_consistent', True)
        
        result.features_used = {
            'account_count': account_count,
            'accounts_consistent': accounts_consistent
        }
        
        if account_count <= 1:
            result.cannot_fire = True
            result.decision_path = [f'account_count={account_count} <= 1', '→ Cannot fire']
            return result
        
        result.decision_path.append(f'account_count={account_count} > 1')
        
        if accounts_consistent:
            result.cannot_fire = True
            result.decision_path.append('→ Cannot fire')
            return result
        
        result.fires = True
        result.decision_path.append('→ 8029 FIRES')
        
        return result
    
    def check_8030(self, features: Dict, party: str = None) -> CodeCheckResult:
        """
        Check 8030: IBAN Derivation Not Supported for Country
        
        Decision Tree:
            needs_iban = True?
            ├─ NO → Cannot fire
            └─ YES → has_iban?
                      ├─ YES → Cannot fire
                      └─ NO → iban_derivation_supported?
                                ├─ YES → Cannot fire (8004 may fire if fails)
                                └─ NO → 8030 FIRES
        """
        result = CodeCheckResult(
            code='8030',
            party=party,
            description='IBAN derivation not supported for country'
        )
        
        needs_iban = features.get('needs_iban', False)
        has_iban = features.get('has_iban', False)
        country = features.get('country') or features.get('address_country', '')
        
        result.features_used = {
            'needs_iban': needs_iban,
            'has_iban': has_iban,
            'country': country
        }
        
        if not needs_iban:
            result.cannot_fire = True
            result.decision_path = ['needs_iban=False', '→ Cannot fire']
            return result
        
        result.decision_path.append('needs_iban=True')
        
        if has_iban:
            result.cannot_fire = True
            result.decision_path.append('has_iban=True')
            result.decision_path.append('→ Cannot fire (IBAN present)')
            return result
        
        result.decision_path.append('has_iban=False')
        
        derivation_supported = country in IBAN_DERIVATION_SUPPORTED if country else False
        result.features_used['iban_derivation_supported'] = derivation_supported
        
        if derivation_supported:
            result.cannot_fire = True
            result.decision_path.append(f'iban_derivation_supported=True (country={country})')
            result.decision_path.append('→ Cannot fire (8004 handles derivation failure)')
            return result
        
        result.fires = True
        result.decision_path.append(f'iban_derivation_supported=False (country={country})')
        result.decision_path.append('→ 8030 FIRES')
        
        return result
    
    def check_8033(self, features: Dict, global_features: Dict = None) -> CodeCheckResult:
        """
        Check 8033: CLABE Inconsistency (Mexico)
        """
        result = CodeCheckResult(
            code='8033',
            party=None,
            description='CLABE inconsistency - multiple different CLABE values (Mexico)'
        )
        
        gf = global_features or features
        
        clabe_count = gf.get('clabe_count', 0)
        clabes_consistent = gf.get('clabes_consistent', True)
        is_clabe = gf.get('is_clabe', False)
        
        result.features_used = {
            'clabe_count': clabe_count,
            'clabes_consistent': clabes_consistent,
            'is_clabe': is_clabe
        }
        
        if clabe_count <= 1 and not is_clabe:
            result.cannot_fire = True
            result.decision_path = ['clabe_count <= 1', '→ Cannot fire']
            return result
        
        if clabe_count <= 1:
            result.cannot_fire = True
            result.decision_path = [f'clabe_count={clabe_count} <= 1', '→ Cannot fire']
            return result
        
        result.decision_path.append(f'clabe_count={clabe_count} > 1')
        
        if clabes_consistent:
            result.cannot_fire = True
            result.decision_path.append('→ Cannot fire (CLABEs consistent)')
            return result
        
        result.fires = True
        result.decision_path.append('→ 8033 FIRES')
        
        return result
    
    def check_8124(self, features: Dict, party: str = None) -> CodeCheckResult:
        """
        Check 8124: Invalid Currency
        
        Decision Tree:
            has_currency = True?
            ├─ NO → Cannot fire
            └─ YES → currency in ISO 4217?
                      ├─ YES → Cannot fire
                      └─ NO → 8124 FIRES
        """
        result = CodeCheckResult(
            code='8124',
            party=party,
            description='Invalid currency code - not a valid ISO 4217 code'
        )
        
        currency = features.get('primary_currency') or features.get('currency', '')
        has_currency = bool(currency)
        
        result.features_used = {
            'has_currency': has_currency,
            'currency': currency
        }
        
        if not has_currency:
            result.cannot_fire = True
            result.decision_path = ['has_currency=False', '→ Cannot fire']
            return result
        
        result.decision_path.append(f'has_currency=True (currency={currency})')
        
        currency_upper = currency.upper().strip()
        currency_valid = currency_upper in VALID_CURRENCY_CODES
        
        result.features_used['currency_valid'] = currency_valid
        
        if currency_valid:
            result.cannot_fire = True
            result.decision_path.append('currency_valid=True')
            result.decision_path.append('→ Cannot fire')
            return result
        
        result.fires = True
        result.decision_path.append('currency_valid=False')
        result.decision_path.append('→ 8124 FIRES')
        
        return result
    
    def check_8852(self, features: Dict, party: str = None) -> CodeCheckResult:
        """
        Check 8852: Incorrect Length of Attribute
        
        Decision Tree:
            has_account = True?
            ├─ NO → Cannot fire
            └─ YES → Determine expected length by type
                      └─ account_length in expected range?
                           ├─ YES → Cannot fire
                           └─ NO → 8852 FIRES
        """
        result = CodeCheckResult(
            code='8852',
            party=party,
            description='Incorrect length of attribute'
        )
        
        has_account = features.get('has_account', False)
        account_value = features.get('account_value', '')
        account_type = features.get('account_type', '')
        is_clabe = features.get('is_clabe', False)
        is_fedaba = features.get('is_fedaba', False)
        has_iban = features.get('has_iban', False)
        
        result.features_used = {
            'has_account': has_account,
            'account_value': account_value,
            'account_type': account_type,
            'is_clabe': is_clabe,
            'is_fedaba': is_fedaba,
            'has_iban': has_iban
        }
        
        if not has_account and not account_value:
            result.cannot_fire = True
            result.decision_path = ['has_account=False', '→ Cannot fire']
            return result
        
        account_length = len(account_value) if account_value else 0
        result.features_used['account_length'] = account_length
        result.decision_path.append(f'has_account=True (length={account_length})')
        
        # Determine expected length
        length_valid = True
        expected_range = None
        
        if is_clabe:
            expected_range = (18, 18)
            length_valid = account_length == 18
        elif is_fedaba:
            expected_range = (9, 9)
            length_valid = account_length == 9
        elif has_iban or account_type == 'IBAN':
            expected_range = (15, 34)
            length_valid = 15 <= account_length <= 34
        else:
            # Generic account - wide range
            expected_range = (1, 50)
            length_valid = 1 <= account_length <= 50
        
        result.features_used['expected_range'] = expected_range
        result.features_used['length_valid'] = length_valid
        
        result.decision_path.append(f'expected_range={expected_range}')
        
        if length_valid:
            result.cannot_fire = True
            result.decision_path.append('length_valid=True')
            result.decision_path.append('→ Cannot fire')
            return result
        
        result.fires = True
        result.decision_path.append('length_valid=False')
        result.decision_path.append('→ 8852 FIRES')
        
        return result
    
    def check_8894(self, features: Dict, party: str = None) -> CodeCheckResult:
        """
        Check 8894: Invalid IBAN
        
        Decision Tree:
            has_iban = True?
            ├─ NO → Cannot fire
            └─ YES → iban_valid_format?
                      ├─ NO → 8894 FIRES
                      └─ YES → iban_checksum_valid?
                                ├─ NO → 8894 FIRES (or 8898)
                                └─ YES → Cannot fire
        """
        result = CodeCheckResult(
            code='8894',
            party=party,
            description='Invalid IBAN - format or checksum error'
        )
        
        has_iban = features.get('has_iban', False)
        iban_value = features.get('iban') or features.get('account_value', '')
        
        # Check if it looks like an IBAN
        if iban_value:
            iban_clean = iban_value.upper().replace(' ', '').replace('-', '')
            if len(iban_clean) >= 4 and iban_clean[:2].isalpha():
                has_iban = True
        
        result.features_used = {
            'has_iban': has_iban,
            'iban_value': iban_value
        }
        
        if not has_iban or not iban_value:
            result.cannot_fire = True
            result.decision_path = ['has_iban=False', '→ Cannot fire']
            return result
        
        result.decision_path.append(f'has_iban=True (iban={iban_value})')
        
        # Validate IBAN
        format_valid, checksum_valid, country = self._validate_iban(iban_value)
        
        result.features_used['iban_valid_format'] = format_valid
        result.features_used['iban_checksum_valid'] = checksum_valid
        result.features_used['iban_country'] = country
        
        if not format_valid:
            result.fires = True
            result.decision_path.append('iban_valid_format=False')
            result.decision_path.append('→ 8894 FIRES (invalid IBAN format)')
            return result
        
        result.decision_path.append('iban_valid_format=True')
        
        if not checksum_valid:
            result.fires = True
            result.decision_path.append('iban_checksum_valid=False')
            result.decision_path.append('→ 8894 FIRES (IBAN checksum failed)')
            return result
        
        result.decision_path.append('iban_checksum_valid=True')
        result.cannot_fire = True
        result.decision_path.append('→ Cannot fire (IBAN is valid)')
        
        return result
    
    def check_8895(self, features: Dict, party: str = None) -> CodeCheckResult:
        """
        Check 8895: Invalid NCH Code (Routing Number)
        
        Decision Tree:
            has_nch = True?
            ├─ NO → Cannot fire
            └─ YES → nch_validation_applicable?
                      ├─ NO → Cannot fire (international)
                      └─ YES → Validate by type:
                                FEDABA: 9 digits + checksum
                                CHIPS: 6 digits
                                etc.
        """
        result = CodeCheckResult(
            code='8895',
            party=party,
            description='Invalid NCH code - routing number format or checksum error'
        )
        
        has_nch = features.get('has_nch', False)
        nch_value = features.get('nch_value') or features.get('adr_bank_id', '')
        nch_type = features.get('nch_type', '')
        nch_validation_applicable = features.get('nch_validation_applicable', False)
        is_fedaba = features.get('is_fedaba', False)
        country = features.get('country') or features.get('address_country', '')
        
        # Determine if NCH validation applies
        if not nch_validation_applicable:
            # Check if it should apply (US domestic)
            if country == 'US' or is_fedaba:
                nch_validation_applicable = True
        
        result.features_used = {
            'has_nch': has_nch,
            'nch_value': nch_value,
            'nch_type': nch_type,
            'nch_validation_applicable': nch_validation_applicable,
            'is_fedaba': is_fedaba,
            'country': country
        }
        
        if not has_nch and not nch_value:
            result.cannot_fire = True
            result.decision_path = ['has_nch=False', '→ Cannot fire']
            return result
        
        result.decision_path.append(f'has_nch=True (nch={nch_value})')
        
        if not nch_validation_applicable:
            result.cannot_fire = True
            result.decision_path.append('nch_validation_applicable=False')
            result.decision_path.append('→ Cannot fire (international, no NCH validation)')
            return result
        
        result.decision_path.append('nch_validation_applicable=True')
        
        # Validate the NCH
        nch_clean = nch_value.strip().replace('-', '').replace(' ', '') if nch_value else ''
        
        # Check for FEDABA (9 digits)
        if is_fedaba or len(nch_clean) == 9:
            format_valid, checksum_valid = self._validate_fedaba(nch_clean)
            result.features_used['fedaba_format_valid'] = format_valid
            result.features_used['fedaba_checksum_valid'] = checksum_valid
            
            if not format_valid:
                result.fires = True
                result.decision_path.append('fedaba_format_valid=False')
                result.decision_path.append('→ 8895 FIRES (invalid FEDABA format)')
                return result
            
            result.decision_path.append('fedaba_format_valid=True')
            
            if not checksum_valid:
                result.fires = True
                result.decision_path.append('fedaba_checksum_valid=False')
                result.decision_path.append('→ 8895 FIRES (FEDABA checksum failed)')
                return result
            
            result.decision_path.append('fedaba_checksum_valid=True')
        
        # Check for CHIPS (6 digits)
        elif len(nch_clean) == 6:
            if not nch_clean.isdigit():
                result.fires = True
                result.decision_path.append('chips_valid=False (not all digits)')
                result.decision_path.append('→ 8895 FIRES')
                return result
            result.decision_path.append('chips_valid=True')
        
        result.cannot_fire = True
        result.decision_path.append('→ Cannot fire (NCH is valid)')
        
        return result
    
    def check_8898(self, features: Dict, party: str = None) -> CodeCheckResult:
        """
        Check 8898: IBAN Check Digit Failed
        
        More specific than 8894 - fires only when format is valid but checksum fails.
        
        Decision Tree:
            has_iban = True?
            ├─ NO → Cannot fire
            └─ YES → iban_valid_format?
                      ├─ NO → 8894 fires instead
                      └─ YES → iban_checksum_valid?
                                ├─ YES → Cannot fire
                                └─ NO → 8898 FIRES
        """
        result = CodeCheckResult(
            code='8898',
            party=party,
            description='IBAN check digit failed - format valid but checksum error'
        )
        
        has_iban = features.get('has_iban', False)
        iban_value = features.get('iban') or features.get('account_value', '')
        
        result.features_used = {
            'has_iban': has_iban,
            'iban_value': iban_value
        }
        
        if not has_iban or not iban_value:
            result.cannot_fire = True
            result.decision_path = ['has_iban=False', '→ Cannot fire']
            return result
        
        result.decision_path.append(f'has_iban=True')
        
        format_valid, checksum_valid, _ = self._validate_iban(iban_value)
        
        result.features_used['iban_valid_format'] = format_valid
        result.features_used['iban_checksum_valid'] = checksum_valid
        
        if not format_valid:
            result.cannot_fire = True
            result.decision_path.append('iban_valid_format=False')
            result.decision_path.append('→ Cannot fire (8894 fires for format errors)')
            return result
        
        result.decision_path.append('iban_valid_format=True')
        
        if checksum_valid:
            result.cannot_fire = True
            result.decision_path.append('iban_checksum_valid=True')
            result.decision_path.append('→ Cannot fire')
            return result
        
        result.fires = True
        result.decision_path.append('iban_checksum_valid=False')
        result.decision_path.append('→ 8898 FIRES (checksum specifically failed)')
        
        return result
    
    # =========================================================================
    # AGGREGATE CHECK METHOD
    # =========================================================================
    
    def check_all(self, features: Dict, parties: Dict = None, 
                  global_features: Dict = None) -> List[CodeCheckResult]:
        """
        Check all 8XXX codes for all parties.
        
        Args:
            features: Flat feature dictionary (prefixed keys)
            parties: Dict of party_type -> party_features (optional)
            global_features: Cross-party aggregated features (optional)
        
        Returns:
            List of CodeCheckResult for all codes checked
        """
        results = []
        
        # Global/cross-party checks (no specific party)
        global_checks = [
            (self.check_8023, 'IBAN inconsistency'),
            (self.check_8024, 'BBAN inconsistency'),
            (self.check_8025, 'Domestic account inconsistency'),
            (self.check_8026, 'NCH inconsistency'),
            (self.check_8027, 'Country code inconsistency'),
            (self.check_8028, 'BIC4 inconsistency'),
            (self.check_8029, 'Account inconsistency'),
            (self.check_8033, 'CLABE inconsistency'),
        ]
        
        gf = global_features or features
        for check_func, desc in global_checks:
            try:
                result = check_func(features, gf)
                results.append(result)
            except Exception as e:
                # Log error but continue
                results.append(CodeCheckResult(
                    code=check_func.__name__.replace('check_', ''),
                    cannot_fire=True,
                    decision_path=[f'Error: {str(e)}']
                ))
        
        # Currency check (global)
        results.append(self.check_8124(features))
        results.append(self.check_8007(features))
        
        # Per-party checks
        # NOTE: 8851/8852 are schema-level errors detected during XML parsing,
        # not from IFML features, so they are excluded from rules engine.
        party_checks = [
            self.check_8001,  # Invalid BIC
            self.check_8004,  # IBAN cannot be derived (common code!)
            self.check_8005,  # Invalid BIC4
            self.check_8006,  # Invalid country
            self.check_8022,  # IBAN/BIC mismatch
            # self.check_8030,  # IBAN derivation not supported - covered by 8004
            # self.check_8852,  # REMOVED - schema error, not detectable from features
            self.check_8894,  # Invalid IBAN
            self.check_8895,  # Invalid NCH
            self.check_8898,  # IBAN check digit failed
        ]
        
        # If parties dict provided, check each party
        if parties:
            for party_type, party_features in parties.items():
                suffix = self._get_suffix(party_type)
                for check_func in party_checks:
                    try:
                        result = check_func(party_features, suffix)
                        # Add party-specific code suffix
                        if result.fires or result.eligible:
                            result.code = f"{result.code}_{suffix}"
                        results.append(result)
                    except Exception as e:
                        results.append(CodeCheckResult(
                            code=f"{check_func.__name__.replace('check_', '')}_{suffix}",
                            cannot_fire=True,
                            decision_path=[f'Error: {str(e)}']
                        ))
        else:
            # Extract parties from flat features and check each
            for party_type, prefix in self.PARTY_PREFIXES.items():
                # Check if this party exists in features - check multiple indicators
                party_exists = (
                    features.get(f"{prefix}_present", False) or
                    features.get(f"{prefix}_has_bic", False) or
                    features.get(f"{prefix}_has_iban", False) or
                    features.get(f"{prefix}_has_account", False) or
                    features.get(f"{prefix}_has_nch", False) or
                    features.get(f"{prefix}_has_name", False) or
                    features.get(f"{prefix}_has_id", False) or
                    features.get(f"{prefix}_country", '') != '' or
                    features.get(f"{prefix}_bic", '') != ''
                )
                
                if party_exists:
                    party_features = self._get_party_features(features, party_type)
                    suffix = self._get_suffix(party_type)
                    
                    for check_func in party_checks:
                        try:
                            result = check_func(party_features, suffix)
                            if result.fires or result.eligible:
                                result.code = f"{result.code}_{suffix}"
                            results.append(result)
                        except Exception as e:
                            results.append(CodeCheckResult(
                                code=f"{check_func.__name__.replace('check_', '')}_{suffix}",
                                cannot_fire=True,
                                decision_path=[f'Error: {str(e)}']
                            ))
        
        return results
    
    def get_fired_codes(self, results: List[CodeCheckResult]) -> List[str]:
        """Extract codes that fired from results."""
        return [r.code for r in results if r.fires]
    
    def get_eligible_codes(self, results: List[CodeCheckResult]) -> List[str]:
        """Extract codes that are eligible (pending lookup) from results."""
        return [r.code for r in results if r.eligible]
    
    def get_cannot_fire_codes(self, results: List[CodeCheckResult]) -> List[str]:
        """Extract codes that cannot fire from results."""
        return [r.code for r in results if r.cannot_fire]


# =============================================================================
# STANDALONE TESTING
# =============================================================================

if __name__ == '__main__':
    # Simple test
    rules = Rules8XXX()
    
    # Test 8001 with invalid BIC
    test_features = {
        'has_bic': True,
        'bic': 'INVALID123',  # Invalid format
    }
    result = rules.check_8001(test_features, 'BNFBNK')
    print(f"8001 Test: fires={result.fires}")
    print(f"  Decision path: {result.decision_path}")
    
    # Test 8894 with invalid IBAN
    test_features = {
        'has_iban': True,
        'iban': 'DE99123456789012345678',  # Bad checksum
    }
    result = rules.check_8894(test_features, 'CDTPTY')
    print(f"\n8894 Test: fires={result.fires}")
    print(f"  Decision path: {result.decision_path}")
    
    # Test 8022 with country mismatch
    test_features = {
        'has_iban': True,
        'has_bic': True,
        'iban': 'DE89370400440532013000',  # German IBAN
        'bic': 'ABORFRPP',  # French BIC
    }
    result = rules.check_8022(test_features, 'BNFBNK')
    print(f"\n8022 Test: fires={result.fires}")
    print(f"  Decision path: {result.decision_path}")
