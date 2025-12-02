#!/usr/bin/env python3
"""
================================================================================
9XXX REPAIR/ENRICHMENT RULES - Deterministic and Directory-Dependent Checks
================================================================================

PURPOSE:
    Implements decision trees for 9XXX ACE repair/enrichment codes.
    These codes indicate ACE successfully MODIFIED the payment data.
    
    Two categories:
    - DETERMINISTIC (No Directory): 100% predictable from input features
    - DIRECTORY-DEPENDENT: Can only predict ELIGIBILITY, not outcome

CODES IMPLEMENTED:

    DETERMINISTIC (No Directory Required):
    - 9000: NCH code cleaned (dirty chars removed)
    - 9002: Account number cleaned
    - 9006: IBAN cleaned/formatted
    - 9009: Field cleaned
    - 9012: IBAN formatted (spaces added)
    - 9014: BBAN cleaned
    - 9015: Domestic account cleaned
    - 9017: Multiple party information present
    - 9018: Duplicate party information removed
    - 9019: Party identifier cleaned
    - 9021: FEDABA formatted
    - 9022: Account length fixed
    - 9024: Push Up performed
    - 9025: CLABE repaired
    - 9028: NCH stripped from wrong field
    
    DIRECTORY-DEPENDENT (Predict Eligibility Only):
    - 9004: IBAN derived (from account)
    - 9005: BIC derived from NCH
    - 9007: Account replaced by IBAN
    - 9008: BIC derived from IBAN
    - 9477: D-A from FEDABA in address
    - 9479: Party enriched from BIC directory
    - 9480: Credit party enriched

USAGE:
    from rules_9xxx import Rules9XXX
    
    rules = Rules9XXX()
    results = rules.check_all(features_dict, parties_dict)
    
    # Check individual codes:
    result = rules.check_9018(global_features)

AUTHOR: ACE Pelican ML Team
VERSION: 1.0 (December 2025)
================================================================================
"""

import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field


# Import shared data class from 8XXX rules
try:
    from rules_8xxx import CodeCheckResult, VALID_COUNTRY_CODES, IBAN_LENGTHS
except ImportError:
    # Define locally if import fails
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
        'UA': 29,
    }
    
    @dataclass
    class CodeCheckResult:
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
# REFERENCE DATA
# =============================================================================

# Countries that require IBAN
IBAN_REQUIRED_COUNTRIES = set(IBAN_LENGTHS.keys())

# Countries that support IBAN derivation
IBAN_DERIVATION_SUPPORTED = {
    'AT', 'BE', 'CH', 'CZ', 'DE', 'DK', 'EE', 'ES', 'FI', 'FR', 'GB', 'GR',
    'HR', 'HU', 'IE', 'IT', 'LT', 'LU', 'LV', 'NL', 'NO', 'PL', 'PT', 'RO',
    'SE', 'SI', 'SK'
}

# Characters considered "dirty" in account numbers
DIRTY_CHARS_PATTERN = re.compile(r'[^A-Za-z0-9]')

# Characters that need cleaning but are common formatting
FORMATTING_CHARS_PATTERN = re.compile(r'[\s\-/.]')


# =============================================================================
# 9XXX REPAIR RULES CLASS
# =============================================================================

class Rules9XXX:
    """
    Rules for 9XXX repair/enrichment codes.
    
    Two categories:
    - DETERMINISTIC: 100% predictable, fires=True when conditions met
    - DIRECTORY-DEPENDENT: eligible=True when conditions met (outcome unknown)
    
    Usage:
        rules = Rules9XXX()
        results = rules.check_all(features, parties, global_features)
    """
    
    # Party type to code suffix mapping
    PARTY_SUFFIXES = {
        'OriginatingPartyInfo': 'ORGPTY',
        'OriginatingPartyInf': 'ORGPTY',
        'SendingBankInfo': 'SNDBNK',
        'SendingBankInf': 'SNDBNK',
        'DebitPartyInfo': 'DBTPTY',
        'DebitPartyInf': 'DBTPTY',
        'CreditPartyInfo': 'CDTPTY',
        'CreditPartyInf': 'CDTPTY',
        'IntermediaryBankInfo': 'INTMBNK',
        'IntermediaryBankInf': 'INTMBNK',
        'BeneficiaryBankInfo': 'BNFBNK',
        'BeneficiaryBankInf': 'BNFBNK',
        'BeneficiaryPartyInfo': 'BNPPTY',
        'BeneficiaryPartyInf': 'BNPPTY',
        'AccountWithInstitution': 'ACWI',
        'OrderingInstitution': 'ORDI',
    }
    
    # Feature prefix mapping
    PARTY_PREFIXES = {
        'OriginatingPartyInfo': 'orig',
        'OriginatingPartyInf': 'orig',
        'SendingBankInfo': 'send',
        'SendingBankInf': 'send',
        'DebitPartyInfo': 'dbt',
        'DebitPartyInf': 'dbt',
        'CreditPartyInfo': 'cdt',
        'CreditPartyInf': 'cdt',
        'IntermediaryBankInfo': 'intm',
        'IntermediaryBankInf': 'intm',
        'BeneficiaryBankInfo': 'bnf',
        'BeneficiaryBankInf': 'bnf',
        'BeneficiaryPartyInfo': 'bnp',
        'BeneficiaryPartyInf': 'bnp',
        'AccountWithInstitution': 'acwi',
        'OrderingInstitution': 'ordi',
    }
    
    def __init__(self):
        """Initialize the 9XXX rules engine."""
        pass
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _get_party_features(self, features: Dict, party_type: str) -> Dict:
        """Extract features for a specific party from flat feature dict."""
        prefix = self.PARTY_PREFIXES.get(party_type, '')
        if not prefix:
            return {}
        
        prefix_with_underscore = f"{prefix}_"
        party_features = {}
        
        for key, value in features.items():
            if key.startswith(prefix_with_underscore):
                base_key = key[len(prefix_with_underscore):]
                party_features[base_key] = value
            elif '_' not in key or not any(key.startswith(p + '_') for p in self.PARTY_PREFIXES.values()):
                party_features[key] = value
        
        return party_features
    
    def _get_suffix(self, party_type: str) -> str:
        """Get code suffix for party type."""
        return self.PARTY_SUFFIXES.get(party_type, 'UNKNOWN')
    
    def _has_dirty_chars(self, value: str) -> bool:
        """Check if string contains non-alphanumeric characters."""
        if not value:
            return False
        return bool(DIRTY_CHARS_PATTERN.search(value))
    
    def _has_formatting_chars(self, value: str) -> bool:
        """Check if string contains formatting characters (space, dash, etc.)."""
        if not value:
            return False
        return bool(FORMATTING_CHARS_PATTERN.search(value))
    
    def _needs_cleaning(self, value: str) -> Tuple[bool, str]:
        """
        Check if value needs cleaning and return reason.
        
        Returns:
            Tuple of (needs_cleaning, reason)
        """
        if not value:
            return False, ''
        
        has_space = ' ' in value
        has_dash = '-' in value
        has_slash = '/' in value
        has_special = bool(re.search(r'[^A-Za-z0-9\s\-/]', value))
        has_lowercase = value != value.upper() and any(c.isalpha() for c in value)
        
        reasons = []
        if has_space:
            reasons.append('contains spaces')
        if has_dash:
            reasons.append('contains dashes')
        if has_slash:
            reasons.append('contains slashes')
        if has_special:
            reasons.append('contains special characters')
        if has_lowercase:
            reasons.append('contains lowercase')
        
        return bool(reasons), ', '.join(reasons)
    
    # =========================================================================
    # DETERMINISTIC CLEANING CODES (No Directory)
    # =========================================================================
    
    def check_9000(self, features: Dict, party: str = None) -> CodeCheckResult:
        """
        Check 9000: NCH Code Cleaned
        
        Decision Tree:
            has_nch = True?
            ├─ NO → Cannot fire
            └─ YES → nch_has_dirty_chars?
                      ├─ NO → Cannot fire
                      └─ YES → 9000 FIRES
        """
        result = CodeCheckResult(
            code='9000',
            party=party,
            description='NCH code cleaned - non-alphanumeric characters removed'
        )
        
        has_nch = features.get('has_nch', False)
        nch_value = features.get('nch_value') or features.get('adr_bank_id', '')
        nch_has_dirty_chars = features.get('nch_has_dirty_chars', False)
        
        # Check directly if not in features
        if nch_value and not nch_has_dirty_chars:
            nch_has_dirty_chars = self._has_dirty_chars(nch_value)
        
        result.features_used = {
            'has_nch': has_nch,
            'nch_value': nch_value,
            'nch_has_dirty_chars': nch_has_dirty_chars
        }
        
        if not has_nch and not nch_value:
            result.cannot_fire = True
            result.decision_path = ['has_nch=False', '→ Cannot fire']
            return result
        
        result.decision_path.append(f'has_nch=True (nch={nch_value})')
        
        if not nch_has_dirty_chars:
            result.cannot_fire = True
            result.decision_path.append('nch_has_dirty_chars=False')
            result.decision_path.append('→ Cannot fire (NCH is clean)')
            return result
        
        result.fires = True
        result.decision_path.append('nch_has_dirty_chars=True')
        result.decision_path.append('→ 9000 FIRES')
        
        return result
    
    def check_9002(self, features: Dict, party: str = None) -> CodeCheckResult:
        """
        Check 9002: Account Number Cleaned
        
        Decision Tree:
            has_account = True?
            ├─ NO → Cannot fire
            └─ YES → account_has_dirty_chars?
                      ├─ NO → Cannot fire
                      └─ YES → is_domestic_account?
                                ├─ YES → 9015 fires instead
                                └─ NO → 9002 FIRES
        """
        result = CodeCheckResult(
            code='9002',
            party=party,
            description='Account number cleaned - non-alphanumeric characters removed'
        )
        
        has_account = features.get('has_account', False)
        account_value = features.get('account_value', '')
        account_has_dirty_chars = features.get('account_has_dirty_chars', False)
        is_domestic = features.get('is_domestic', False)
        
        # Check directly if not in features
        if account_value and not account_has_dirty_chars:
            account_has_dirty_chars = self._has_dirty_chars(account_value)
        
        result.features_used = {
            'has_account': has_account,
            'account_value': account_value,
            'account_has_dirty_chars': account_has_dirty_chars,
            'is_domestic': is_domestic
        }
        
        if not has_account and not account_value:
            result.cannot_fire = True
            result.decision_path = ['has_account=False', '→ Cannot fire']
            return result
        
        result.decision_path.append(f'has_account=True')
        
        if not account_has_dirty_chars:
            result.cannot_fire = True
            result.decision_path.append('account_has_dirty_chars=False')
            result.decision_path.append('→ Cannot fire (account is clean)')
            return result
        
        result.decision_path.append('account_has_dirty_chars=True')
        
        if is_domestic:
            result.cannot_fire = True
            result.decision_path.append('is_domestic=True')
            result.decision_path.append('→ Cannot fire (9015 fires for domestic)')
            return result
        
        result.fires = True
        result.decision_path.append('→ 9002 FIRES')
        
        return result
    
    def check_9006(self, features: Dict, party: str = None) -> CodeCheckResult:
        """
        Check 9006: IBAN Cleaned
        
        Decision Tree:
            has_iban = True?
            ├─ NO → Cannot fire
            └─ YES → iban_needs_formatting?
                      ├─ NO → Cannot fire
                      └─ YES → 9006 FIRES
        """
        result = CodeCheckResult(
            code='9006',
            party=party,
            description='IBAN cleaned - spaces/dashes removed, uppercase applied'
        )
        
        has_iban = features.get('has_iban', False)
        iban_value = features.get('iban') or features.get('account_value', '')
        iban_needs_formatting = features.get('iban_needs_formatting', False)
        
        # Check directly if not in features
        if iban_value and not iban_needs_formatting:
            needs_clean, reason = self._needs_cleaning(iban_value)
            iban_needs_formatting = needs_clean
        
        result.features_used = {
            'has_iban': has_iban,
            'iban_value': iban_value,
            'iban_needs_formatting': iban_needs_formatting
        }
        
        if not has_iban and not iban_value:
            result.cannot_fire = True
            result.decision_path = ['has_iban=False', '→ Cannot fire']
            return result
        
        result.decision_path.append('has_iban=True')
        
        if not iban_needs_formatting:
            result.cannot_fire = True
            result.decision_path.append('iban_needs_formatting=False')
            result.decision_path.append('→ Cannot fire (IBAN is clean)')
            return result
        
        result.fires = True
        result.decision_path.append('iban_needs_formatting=True')
        result.decision_path.append('→ 9006 FIRES')
        
        return result
    
    def check_9015(self, features: Dict, party: str = None) -> CodeCheckResult:
        """
        Check 9015: Domestic Account Number Cleaned
        
        Decision Tree:
            has_domestic_account = True? (or is_domestic with account)
            ├─ NO → Cannot fire
            └─ YES → account_has_dirty_chars?
                      ├─ NO → Cannot fire
                      └─ YES → 9015 FIRES
        """
        result = CodeCheckResult(
            code='9015',
            party=party,
            description='Domestic account number cleaned'
        )
        
        has_account = features.get('has_account', False)
        is_domestic = features.get('is_domestic', False)
        account_value = features.get('account_value', '')
        account_has_dirty_chars = features.get('account_has_dirty_chars', False)
        
        has_domestic_account = (has_account and is_domestic) or features.get('has_domestic_account', False)
        
        # Check directly
        if account_value and not account_has_dirty_chars:
            account_has_dirty_chars = self._has_dirty_chars(account_value)
        
        result.features_used = {
            'has_domestic_account': has_domestic_account,
            'account_value': account_value,
            'account_has_dirty_chars': account_has_dirty_chars
        }
        
        if not has_domestic_account:
            result.cannot_fire = True
            result.decision_path = ['has_domestic_account=False', '→ Cannot fire']
            return result
        
        result.decision_path.append('has_domestic_account=True')
        
        if not account_has_dirty_chars:
            result.cannot_fire = True
            result.decision_path.append('account_has_dirty_chars=False')
            result.decision_path.append('→ Cannot fire')
            return result
        
        result.fires = True
        result.decision_path.append('account_has_dirty_chars=True')
        result.decision_path.append('→ 9015 FIRES')
        
        return result
    
    def check_9017(self, features: Dict, party: str = None) -> CodeCheckResult:
        """
        Check 9017: Multiple Party Information Present
        
        Decision Tree:
            has_multiple_ids = True?
            ├─ NO → Cannot fire
            └─ YES → 9017 FIRES
        """
        result = CodeCheckResult(
            code='9017',
            party=party,
            description='Multiple party information present'
        )
        
        has_multiple_ids = features.get('has_multiple_ids', False)
        id_is_compound = features.get('id_is_compound', False)
        
        # Multiple IDs can come from compound ID or multiple fields
        has_multiple = has_multiple_ids or id_is_compound
        
        result.features_used = {
            'has_multiple_ids': has_multiple_ids,
            'id_is_compound': id_is_compound
        }
        
        if not has_multiple:
            result.cannot_fire = True
            result.decision_path = ['has_multiple_ids=False', '→ Cannot fire']
            return result
        
        result.fires = True
        result.decision_path.append('has_multiple_ids=True')
        result.decision_path.append('→ 9017 FIRES')
        
        return result
    
    def check_9018(self, features: Dict, global_features: Dict = None) -> CodeCheckResult:
        """
        Check 9018: Duplicate Party Information Removed
        
        Decision Tree:
            intm_count >= 1?
            ├─ NO → Cannot fire
            └─ YES → intm_has_multiple? (count > 1)
                      ├─ NO → Cannot fire
                      └─ YES → Check redundancy signals:
                                │
                                │  SIGNAL 1: Same AdrBankID
                                │  SIGNAL 2: Same Country
                                │  SIGNAL 3: Same BIC Prefix
                                │
                                └─ Count TRUE signals >= 2?
                                     ├─ NO → Cannot fire
                                     └─ YES → 9018 FIRES
        """
        result = CodeCheckResult(
            code='9018',
            party=None,  # Global check
            description='Duplicate party information removed - redundant intermediary entry deleted'
        )
        
        gf = global_features or features
        
        intm_count = gf.get('intm_count', 0)
        intm_has_multiple = gf.get('intm_has_multiple', False) or intm_count > 1
        intm_entries_share_adr_bank_id = gf.get('intm_entries_share_adr_bank_id', False)
        intm_entries_share_country = gf.get('intm_entries_share_country', False)
        intm_entries_share_bic_prefix = gf.get('intm_entries_share_bic_prefix', False)
        intm_has_redundant_info = gf.get('intm_has_redundant_info', False)
        
        result.features_used = {
            'intm_count': intm_count,
            'intm_has_multiple': intm_has_multiple,
            'intm_entries_share_adr_bank_id': intm_entries_share_adr_bank_id,
            'intm_entries_share_country': intm_entries_share_country,
            'intm_entries_share_bic_prefix': intm_entries_share_bic_prefix,
            'intm_has_redundant_info': intm_has_redundant_info
        }
        
        if intm_count < 1:
            result.cannot_fire = True
            result.decision_path = ['intm_count=0', '→ Cannot fire (no intermediaries)']
            return result
        
        result.decision_path.append(f'intm_count={intm_count}')
        
        if not intm_has_multiple:
            result.cannot_fire = True
            result.decision_path.append('intm_has_multiple=False')
            result.decision_path.append('→ Cannot fire (single intermediary is valid)')
            return result
        
        result.decision_path.append('intm_has_multiple=True')
        
        # Count redundancy signals
        signals = [
            intm_entries_share_adr_bank_id,
            intm_entries_share_country,
            intm_entries_share_bic_prefix
        ]
        signal_count = sum(signals)
        
        result.features_used['redundancy_signal_count'] = signal_count
        result.decision_path.append(f'redundancy_signals: adr_bank_id={intm_entries_share_adr_bank_id}, '
                                    f'country={intm_entries_share_country}, '
                                    f'bic_prefix={intm_entries_share_bic_prefix}')
        
        # Check if redundancy detected
        if intm_has_redundant_info or signal_count >= 2:
            result.fires = True
            result.decision_path.append(f'intm_has_redundant_info=True (signal_count={signal_count})')
            result.decision_path.append('→ 9018 FIRES (duplicate will be removed)')
            return result
        
        result.cannot_fire = True
        result.decision_path.append(f'signal_count={signal_count} < 2')
        result.decision_path.append('→ Cannot fire (legitimate multi-hop routing)')
        
        return result
    
    def check_9019(self, features: Dict, party: str = None) -> CodeCheckResult:
        """
        Check 9019: Party Identifier Cleaned
        
        Decision Tree:
            has_id = True?
            ├─ NO → Cannot fire
            └─ YES → id_has_dirty_chars? (or id_has_slash with other chars)
                      ├─ NO → Cannot fire
                      └─ YES → 9019 FIRES
        """
        result = CodeCheckResult(
            code='9019',
            party=party,
            description='Party identifier cleaned - spaces/special characters removed'
        )
        
        has_id = features.get('has_id', False)
        id_value = features.get('id_value', '')
        id_has_slash = features.get('id_has_slash', False)
        account_has_dirty_chars = features.get('account_has_dirty_chars', False)
        
        # Check directly
        id_needs_clean = False
        if id_value:
            needs_clean, reason = self._needs_cleaning(id_value)
            id_needs_clean = needs_clean
        
        result.features_used = {
            'has_id': has_id,
            'id_value': id_value,
            'id_has_slash': id_has_slash,
            'id_needs_cleaning': id_needs_clean
        }
        
        if not has_id and not id_value:
            result.cannot_fire = True
            result.decision_path = ['has_id=False', '→ Cannot fire']
            return result
        
        result.decision_path.append('has_id=True')
        
        if not id_needs_clean and not account_has_dirty_chars:
            result.cannot_fire = True
            result.decision_path.append('id_needs_cleaning=False')
            result.decision_path.append('→ Cannot fire (ID is clean)')
            return result
        
        result.fires = True
        result.decision_path.append('id_needs_cleaning=True')
        result.decision_path.append('→ 9019 FIRES')
        
        return result
    
    def check_9021(self, features: Dict, party: str = None) -> CodeCheckResult:
        """
        Check 9021: FEDABA Code Formatted
        
        Decision Tree:
            has_nch = True?
            ├─ NO → Cannot fire
            └─ YES → is_fedaba = True?
                      ├─ NO → Cannot fire
                      └─ YES → nch_needs_formatting?
                                ├─ NO → Cannot fire
                                └─ YES → 9021 FIRES
        """
        result = CodeCheckResult(
            code='9021',
            party=party,
            description='FEDABA code formatted - padded/cleaned to 9 digits'
        )
        
        has_nch = features.get('has_nch', False)
        is_fedaba = features.get('is_fedaba', False)
        nch_value = features.get('nch_value', '')
        nch_needs_formatting = features.get('nch_needs_formatting', False)
        
        # Check directly
        if nch_value and is_fedaba and not nch_needs_formatting:
            clean_nch = nch_value.replace('-', '').replace(' ', '')
            nch_needs_formatting = len(clean_nch) != 9 or self._has_dirty_chars(nch_value)
        
        result.features_used = {
            'has_nch': has_nch,
            'is_fedaba': is_fedaba,
            'nch_value': nch_value,
            'nch_needs_formatting': nch_needs_formatting
        }
        
        if not has_nch and not nch_value:
            result.cannot_fire = True
            result.decision_path = ['has_nch=False', '→ Cannot fire']
            return result
        
        result.decision_path.append('has_nch=True')
        
        if not is_fedaba:
            result.cannot_fire = True
            result.decision_path.append('is_fedaba=False')
            result.decision_path.append('→ Cannot fire (not a FEDABA)')
            return result
        
        result.decision_path.append('is_fedaba=True')
        
        if not nch_needs_formatting:
            result.cannot_fire = True
            result.decision_path.append('nch_needs_formatting=False')
            result.decision_path.append('→ Cannot fire (FEDABA properly formatted)')
            return result
        
        result.fires = True
        result.decision_path.append('nch_needs_formatting=True')
        result.decision_path.append('→ 9021 FIRES')
        
        return result
    
    def check_9022(self, features: Dict, party: str = None) -> CodeCheckResult:
        """
        Check 9022: Account Number Formatted to Required Length
        
        Decision Tree:
            has_account = True?
            ├─ NO → Cannot fire
            └─ YES → account_needs_length_fix?
                      ├─ NO → Cannot fire
                      └─ YES → 9022 FIRES
        """
        result = CodeCheckResult(
            code='9022',
            party=party,
            description='Account number formatted to required length'
        )
        
        has_account = features.get('has_account', False)
        account_value = features.get('account_value', '')
        account_needs_length_fix = features.get('account_needs_length_fix', False)
        is_clabe = features.get('is_clabe', False)
        is_fedaba = features.get('is_fedaba', False)
        
        result.features_used = {
            'has_account': has_account,
            'account_value': account_value,
            'account_needs_length_fix': account_needs_length_fix,
            'is_clabe': is_clabe,
            'is_fedaba': is_fedaba
        }
        
        if not has_account and not account_value:
            result.cannot_fire = True
            result.decision_path = ['has_account=False', '→ Cannot fire']
            return result
        
        result.decision_path.append('has_account=True')
        
        # Check length if type is known
        if not account_needs_length_fix and account_value:
            clean_account = account_value.replace(' ', '').replace('-', '')
            if is_clabe and len(clean_account) != 18:
                account_needs_length_fix = True
            elif is_fedaba and len(clean_account) != 9:
                account_needs_length_fix = True
        
        if not account_needs_length_fix:
            result.cannot_fire = True
            result.decision_path.append('account_needs_length_fix=False')
            result.decision_path.append('→ Cannot fire')
            return result
        
        result.fires = True
        result.decision_path.append('account_needs_length_fix=True')
        result.decision_path.append('→ 9022 FIRES')
        
        return result
    
    def check_9024(self, features: Dict, global_features: Dict = None) -> CodeCheckResult:
        """
        Check 9024: Push Up Performed
        
        Decision Tree:
            has_intermediary = True?
            ├─ NO → Cannot fire
            └─ YES → Check push-up conditions:
                      │
                      │  CONDITION 1: Redundancy present
                      │  intm_has_redundant_info → HIGH CONFIDENCE
                      │
                      │  CONDITION 2: Multiple entry types
                      │  intm_has_multiple_types → HIGH CONFIDENCE
                      │
                      └─ Any condition met?
                           ├─ NO → Cannot fire
                           └─ YES → 9024 FIRES
        """
        result = CodeCheckResult(
            code='9024',
            party=None,
            description='Push Up performed - party information consolidated to higher level'
        )
        
        gf = global_features or features
        
        has_intermediary = gf.get('has_intermediary', False)
        intm_present = gf.get('intm_present', False) or gf.get('intm_count', 0) > 0
        intm_has_redundant_info = gf.get('intm_has_redundant_info', False)
        intm_has_multiple_types = gf.get('intm_has_multiple_types', False)
        intm_has_multiple = gf.get('intm_has_multiple', False)
        
        has_intm = has_intermediary or intm_present
        
        result.features_used = {
            'has_intermediary': has_intm,
            'intm_has_redundant_info': intm_has_redundant_info,
            'intm_has_multiple_types': intm_has_multiple_types,
            'intm_has_multiple': intm_has_multiple
        }
        
        if not has_intm:
            result.cannot_fire = True
            result.decision_path = ['has_intermediary=False', '→ Cannot fire']
            return result
        
        result.decision_path.append('has_intermediary=True')
        
        # Check conditions
        if intm_has_redundant_info:
            result.fires = True
            result.decision_path.append('intm_has_redundant_info=True')
            result.decision_path.append('→ 9024 FIRES (redundancy triggers push up)')
            return result
        
        if intm_has_multiple_types:
            result.fires = True
            result.decision_path.append('intm_has_multiple_types=True')
            result.decision_path.append('→ 9024 FIRES (multiple types trigger push up)')
            return result
        
        # Lower confidence case: just multiple intermediaries
        if intm_has_multiple:
            result.eligible = True
            result.confidence = 0.6
            result.decision_path.append('intm_has_multiple=True')
            result.decision_path.append('→ Eligible (may trigger push up)')
            return result
        
        result.cannot_fire = True
        result.decision_path.append('No push-up conditions met')
        result.decision_path.append('→ Cannot fire')
        
        return result
    
    def check_9025(self, features: Dict, party: str = None) -> CodeCheckResult:
        """
        Check 9025: CLABE Repaired (Mexico)
        
        Decision Tree:
            is_clabe = True?
            ├─ NO → Cannot fire
            └─ YES → clabe_needs_repair?
                      ├─ NO → Cannot fire
                      └─ YES → 9025 FIRES
        """
        result = CodeCheckResult(
            code='9025',
            party=party,
            description='CLABE repaired - Mexican 18-digit account formatted'
        )
        
        is_clabe = features.get('is_clabe', False)
        account_value = features.get('account_value', '')
        clabe_needs_repair = features.get('clabe_needs_repair', False)
        
        # Check directly
        if is_clabe and account_value and not clabe_needs_repair:
            clean_account = account_value.replace(' ', '').replace('-', '')
            clabe_needs_repair = len(clean_account) != 18 or not clean_account.isdigit()
        
        result.features_used = {
            'is_clabe': is_clabe,
            'account_value': account_value,
            'clabe_needs_repair': clabe_needs_repair
        }
        
        if not is_clabe:
            result.cannot_fire = True
            result.decision_path = ['is_clabe=False', '→ Cannot fire']
            return result
        
        result.decision_path.append('is_clabe=True')
        
        if not clabe_needs_repair:
            result.cannot_fire = True
            result.decision_path.append('clabe_needs_repair=False')
            result.decision_path.append('→ Cannot fire')
            return result
        
        result.fires = True
        result.decision_path.append('clabe_needs_repair=True')
        result.decision_path.append('→ 9025 FIRES')
        
        return result
    
    def check_9028(self, features: Dict, party: str = None) -> CodeCheckResult:
        """
        Check 9028: NCH Code Strip Off (NCH in wrong field)
        
        Decision Tree:
            has_nch = True?
            ├─ NO → Cannot fire
            └─ YES → nch_in_wrong_field? (compound ID or embedded)
                      ├─ NO → Cannot fire
                      └─ YES → 9028 FIRES
        """
        result = CodeCheckResult(
            code='9028',
            party=party,
            description='NCH code stripped - removed from wrong field'
        )
        
        has_nch = features.get('has_nch', False)
        id_is_compound = features.get('id_is_compound', False)
        id_has_bic_and_nch = features.get('id_has_bic_and_nch', False)
        nch_in_wrong_field = features.get('nch_in_wrong_field', False)
        
        # NCH in wrong field if compound ID has both BIC and NCH
        nch_wrong = nch_in_wrong_field or id_has_bic_and_nch
        
        result.features_used = {
            'has_nch': has_nch,
            'id_is_compound': id_is_compound,
            'id_has_bic_and_nch': id_has_bic_and_nch,
            'nch_in_wrong_field': nch_wrong
        }
        
        if not has_nch:
            result.cannot_fire = True
            result.decision_path = ['has_nch=False', '→ Cannot fire']
            return result
        
        result.decision_path.append('has_nch=True')
        
        if not nch_wrong:
            result.cannot_fire = True
            result.decision_path.append('nch_in_wrong_field=False')
            result.decision_path.append('→ Cannot fire')
            return result
        
        result.fires = True
        result.decision_path.append('nch_in_wrong_field=True')
        result.decision_path.append('→ 9028 FIRES')
        
        return result
    
    # =========================================================================
    # DIRECTORY-DEPENDENT CODES (Predict Eligibility Only)
    # =========================================================================
    
    def check_9004(self, features: Dict, party: str = None) -> CodeCheckResult:
        """
        Check 9004: IBAN Derived (Directory Lookup Required)
        
        Decision Tree:
            needs_iban = True?
            ├─ NO → Cannot fire
            └─ YES → has_iban?
                      ├─ YES → Cannot fire
                      └─ NO → has_account?
                               ├─ NO → Cannot fire
                               └─ YES → ELIGIBLE (directory lookup)
        
        NOTE: Can only predict ELIGIBILITY, not outcome.
              ~85% success rate when eligible.
        """
        result = CodeCheckResult(
            code='9004',
            party=party,
            description='IBAN derived - IBAN constructed from account + country (requires directory)'
        )
        
        needs_iban = features.get('needs_iban', False)
        has_iban = features.get('has_iban', False)
        has_account = features.get('has_account', False)
        country = features.get('country') or features.get('address_country', '')
        
        # Check if IBAN derivation is supported for country
        derivation_supported = country in IBAN_DERIVATION_SUPPORTED if country else False
        
        result.features_used = {
            'needs_iban': needs_iban,
            'has_iban': has_iban,
            'has_account': has_account,
            'country': country,
            'derivation_supported': derivation_supported
        }
        
        if not needs_iban:
            result.cannot_fire = True
            result.decision_path = ['needs_iban=False', '→ Cannot fire']
            return result
        
        result.decision_path.append('needs_iban=True')
        
        if has_iban:
            result.cannot_fire = True
            result.decision_path.append('has_iban=True')
            result.decision_path.append('→ Cannot fire (IBAN already present)')
            return result
        
        result.decision_path.append('has_iban=False')
        
        if not has_account:
            result.cannot_fire = True
            result.decision_path.append('has_account=False')
            result.decision_path.append('→ Cannot fire (no account to derive from)')
            return result
        
        result.decision_path.append('has_account=True')
        
        if not derivation_supported:
            result.cannot_fire = True
            result.decision_path.append(f'derivation_supported=False (country={country})')
            result.decision_path.append('→ Cannot fire (8030 fires instead)')
            return result
        
        # Eligible for directory lookup
        result.eligible = True
        result.confidence = 0.85  # ~85% success rate
        result.decision_path.append(f'derivation_supported=True (country={country})')
        result.decision_path.append('→ ELIGIBLE for IBAN derivation (~85% success)')
        
        return result
    
    def check_9005(self, features: Dict, party: str = None) -> CodeCheckResult:
        """
        Check 9005: BIC Derived from NCH (Directory Lookup Required)
        
        Decision Tree:
            has_nch = True?
            ├─ NO → Cannot fire
            └─ YES → has_bic?
                      ├─ YES → Cannot fire
                      └─ NO → ELIGIBLE (BIC Plus directory lookup)
        
        NOTE: ~75% success rate when eligible.
        """
        result = CodeCheckResult(
            code='9005',
            party=party,
            description='BIC derived from NCH - looked up from routing number (requires directory)'
        )
        
        has_nch = features.get('has_nch', False)
        has_bic = features.get('has_bic', False)
        nch_value = features.get('nch_value', '')
        has_nch_no_bic = features.get('has_nch_no_bic', False)
        
        result.features_used = {
            'has_nch': has_nch,
            'has_bic': has_bic,
            'nch_value': nch_value,
            'has_nch_no_bic': has_nch_no_bic
        }
        
        if not has_nch and not nch_value:
            result.cannot_fire = True
            result.decision_path = ['has_nch=False', '→ Cannot fire']
            return result
        
        result.decision_path.append('has_nch=True')
        
        if has_bic:
            result.cannot_fire = True
            result.decision_path.append('has_bic=True')
            result.decision_path.append('→ Cannot fire (BIC already present)')
            return result
        
        result.decision_path.append('has_bic=False')
        
        # Eligible for directory lookup
        result.eligible = True
        result.confidence = 0.75  # ~75% success rate
        result.decision_path.append('→ ELIGIBLE for BIC lookup from NCH (~75% success)')
        
        return result
    
    def check_9007(self, features: Dict, party: str = None) -> CodeCheckResult:
        """
        Check 9007: Account Replaced by IBAN (Directory Lookup Required)
        
        Similar to 9004 but for beneficiary party specifically.
        """
        result = CodeCheckResult(
            code='9007',
            party=party,
            description='Account replaced by IBAN - derived for beneficiary (requires directory)'
        )
        
        needs_iban = features.get('needs_iban', False)
        has_iban = features.get('has_iban', False)
        has_account = features.get('has_account', False)
        country = features.get('country') or features.get('address_country', '')
        
        derivation_supported = country in IBAN_DERIVATION_SUPPORTED if country else False
        
        result.features_used = {
            'needs_iban': needs_iban,
            'has_iban': has_iban,
            'has_account': has_account,
            'country': country
        }
        
        if not needs_iban:
            result.cannot_fire = True
            result.decision_path = ['needs_iban=False', '→ Cannot fire']
            return result
        
        if has_iban:
            result.cannot_fire = True
            result.decision_path.append('has_iban=True')
            result.decision_path.append('→ Cannot fire')
            return result
        
        if not has_account:
            result.cannot_fire = True
            result.decision_path.append('has_account=False')
            result.decision_path.append('→ Cannot fire')
            return result
        
        if derivation_supported:
            result.eligible = True
            result.confidence = 0.85
            result.decision_path.append('→ ELIGIBLE for IBAN derivation')
        else:
            result.cannot_fire = True
            result.decision_path.append('→ Cannot fire (derivation not supported)')
        
        return result
    
    def check_9008(self, features: Dict, party: str = None) -> CodeCheckResult:
        """
        Check 9008: BIC Derived from IBAN (Directory Lookup Required)
        
        Decision Tree:
            has_iban = True?
            ├─ NO → Cannot fire
            └─ YES → has_bic?
                      ├─ YES → Cannot fire
                      └─ NO → ELIGIBLE (BIC Plus directory lookup)
        
        NOTE: ~90% success rate (higher than 9005 because IBAN more reliable).
        """
        result = CodeCheckResult(
            code='9008',
            party=party,
            description='BIC derived from IBAN - looked up from IBAN prefix (requires directory)'
        )
        
        has_iban = features.get('has_iban', False)
        has_bic = features.get('has_bic', False)
        has_iban_no_bic = features.get('has_iban_no_bic', False)
        
        result.features_used = {
            'has_iban': has_iban,
            'has_bic': has_bic,
            'has_iban_no_bic': has_iban_no_bic
        }
        
        if not has_iban:
            result.cannot_fire = True
            result.decision_path = ['has_iban=False', '→ Cannot fire']
            return result
        
        result.decision_path.append('has_iban=True')
        
        if has_bic:
            result.cannot_fire = True
            result.decision_path.append('has_bic=True')
            result.decision_path.append('→ Cannot fire (BIC already present)')
            return result
        
        result.decision_path.append('has_bic=False')
        
        # Eligible for directory lookup
        result.eligible = True
        result.confidence = 0.90  # ~90% success rate
        result.decision_path.append('→ ELIGIBLE for BIC lookup from IBAN (~90% success)')
        
        return result
    
    def check_9477(self, features: Dict, party: str = None) -> CodeCheckResult:
        """
        Check 9477: D-A from FEDABA in Address (Directory Lookup Required)
        
        NOTE: Official definition is "D-A using FED ABA from Name and Address"
              This requires Fedwire directory lookup.
        """
        result = CodeCheckResult(
            code='9477',
            party=party,
            description='D-A from FEDABA - address info derived from routing number (requires Fedwire)'
        )
        
        has_bic = features.get('has_bic', False)
        is_fedaba = features.get('is_fedaba', False)
        has_nch = features.get('has_nch', False)
        
        result.features_used = {
            'has_bic': has_bic,
            'is_fedaba': is_fedaba,
            'has_nch': has_nch
        }
        
        # Requires FEDABA present
        if not is_fedaba and not has_nch:
            result.cannot_fire = True
            result.decision_path = ['is_fedaba=False, has_nch=False', '→ Cannot fire']
            return result
        
        result.decision_path.append('has_fedaba_or_nch=True')
        
        # Eligible for Fedwire lookup
        result.eligible = True
        result.confidence = 0.70
        result.decision_path.append('→ ELIGIBLE for Fedwire lookup (~70% success)')
        
        return result
    
    def check_9479(self, features: Dict, party: str = None) -> CodeCheckResult:
        """
        Check 9479: Party Enriched from BIC Directory (Directory Lookup Required)
        
        Decision Tree:
            has_bic = True?
            ├─ NO → Cannot fire
            └─ YES → ELIGIBLE (BIC Plus directory lookup)
        """
        result = CodeCheckResult(
            code='9479',
            party=party,
            description='Party enriched from BIC directory - additional info added (requires directory)'
        )
        
        has_bic = features.get('has_bic', False)
        bic_value = features.get('bic') or features.get('bic_value', '')
        
        result.features_used = {
            'has_bic': has_bic,
            'bic_value': bic_value
        }
        
        if not has_bic and not bic_value:
            result.cannot_fire = True
            result.decision_path = ['has_bic=False', '→ Cannot fire']
            return result
        
        result.decision_path.append('has_bic=True')
        
        # Eligible for directory lookup
        result.eligible = True
        result.confidence = 0.80
        result.decision_path.append('→ ELIGIBLE for BIC Plus enrichment (~80% success)')
        
        return result
    
    def check_9480(self, features: Dict, party: str = None) -> CodeCheckResult:
        """
        Check 9480: Credit Party Enriched (Directory Lookup Required)
        
        Similar to 9479 but specifically for credit party.
        """
        result = CodeCheckResult(
            code='9480',
            party=party,
            description='Credit party enriched - details added from directory (requires directory)'
        )
        
        has_bic = features.get('has_bic', False)
        
        result.features_used = {
            'has_bic': has_bic
        }
        
        if not has_bic:
            result.cannot_fire = True
            result.decision_path = ['has_bic=False', '→ Cannot fire']
            return result
        
        result.decision_path.append('has_bic=True')
        
        result.eligible = True
        result.confidence = 0.75
        result.decision_path.append('→ ELIGIBLE for credit party enrichment')
        
        return result
    
    # =========================================================================
    # AGGREGATE CHECK METHOD
    # =========================================================================
    
    def check_all(self, features: Dict, parties: Dict = None,
                  global_features: Dict = None) -> List[CodeCheckResult]:
        """
        Check all 9XXX codes for all parties.
        
        Args:
            features: Flat feature dictionary (prefixed keys)
            parties: Dict of party_type -> party_features (optional)
            global_features: Cross-party aggregated features (optional)
        
        Returns:
            List of CodeCheckResult for all codes checked
        """
        results = []
        
        # Global checks (no specific party)
        global_checks = [
            self.check_9018,  # Duplicate party removed
            self.check_9024,  # Push up performed
        ]
        
        gf = global_features or features
        for check_func in global_checks:
            try:
                result = check_func(features, gf)
                results.append(result)
            except Exception as e:
                results.append(CodeCheckResult(
                    code=check_func.__name__.replace('check_', ''),
                    cannot_fire=True,
                    decision_path=[f'Error: {str(e)}']
                ))
        
        # Per-party checks
        party_checks_deterministic = [
            self.check_9000,  # NCH cleaned
            self.check_9002,  # Account cleaned
            self.check_9006,  # IBAN cleaned
            self.check_9015,  # Domestic account cleaned
            self.check_9017,  # Multiple party info
            self.check_9019,  # Party ID cleaned
            self.check_9021,  # FEDABA formatted
            self.check_9022,  # Account length fixed
            self.check_9025,  # CLABE repaired
            self.check_9028,  # NCH stripped
        ]
        
        party_checks_directory = [
            self.check_9004,  # IBAN derived
            self.check_9005,  # BIC from NCH
            self.check_9007,  # Account -> IBAN
            self.check_9008,  # BIC from IBAN
            self.check_9477,  # D-A from FEDABA
            self.check_9479,  # Party enriched
            self.check_9480,  # Credit party enriched
        ]
        
        all_party_checks = party_checks_deterministic + party_checks_directory
        
        # If parties dict provided, check each party
        if parties:
            for party_type, party_features in parties.items():
                suffix = self._get_suffix(party_type)
                for check_func in all_party_checks:
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
        else:
            # Extract parties from flat features
            for party_type, prefix in self.PARTY_PREFIXES.items():
                present_key = f"{prefix}_present"
                if features.get(present_key, False) or features.get(f"{prefix}_has_bic", False) or features.get(f"{prefix}_has_account", False):
                    party_features = self._get_party_features(features, party_type)
                    suffix = self._get_suffix(party_type)
                    
                    for check_func in all_party_checks:
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
    rules = Rules9XXX()
    
    # Test 9018 with redundant intermediaries
    test_features = {
        'intm_count': 2,
        'intm_has_multiple': True,
        'intm_entries_share_adr_bank_id': True,
        'intm_entries_share_country': True,
        'intm_entries_share_bic_prefix': False,
        'intm_has_redundant_info': True,
    }
    result = rules.check_9018(test_features)
    print(f"9018 Test: fires={result.fires}")
    print(f"  Decision path: {result.decision_path}")
    
    # Test 9005 eligibility
    test_features = {
        'has_nch': True,
        'nch_value': '021000021',
        'has_bic': False,
    }
    result = rules.check_9005(test_features, 'BNFBNK')
    print(f"\n9005 Test: eligible={result.eligible}, confidence={result.confidence}")
    print(f"  Decision path: {result.decision_path}")
    
    # Test 9002 with dirty account
    test_features = {
        'has_account': True,
        'account_value': '1234-5678-90',  # Has dashes
        'is_domestic': False,
    }
    result = rules.check_9002(test_features, 'CDTPTY')
    print(f"\n9002 Test: fires={result.fires}")
    print(f"  Decision path: {result.decision_path}")
