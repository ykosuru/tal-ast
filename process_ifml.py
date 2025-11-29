#!/usr/bin/env python3
"""
ACE Pelican IFML Processor and Rule Verification

This script processes IFML payment data, predicts 8XXX/9XXX codes based on
deterministic rules, and compares predictions against actual ACE response codes.

Input format:
{
  "payment_id": {
    "Request": { "IFML": { "File": { "Message": { ... } } } },
    "Response": { ... }  // Contains actual ACE codes
  }
}

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
    
    IBAN_COUNTRIES = {
        'AD', 'AE', 'AL', 'AT', 'AZ', 'BA', 'BE', 'BG', 'BH', 'BR', 'BY', 
        'CH', 'CR', 'CY', 'CZ', 'DE', 'DK', 'DO', 'EE', 'EG', 'ES', 'FI', 
        'FO', 'FR', 'GB', 'GE', 'GI', 'GL', 'GR', 'GT', 'HR', 'HU', 'IE', 
        'IL', 'IQ', 'IS', 'IT', 'JO', 'KW', 'KZ', 'LB', 'LC', 'LI', 'LT', 
        'LU', 'LV', 'MC', 'MD', 'ME', 'MK', 'MR', 'MT', 'MU', 'NL', 'NO', 
        'PK', 'PL', 'PS', 'PT', 'QA', 'RO', 'RS', 'SA', 'SC', 'SE', 'SI', 
        'SK', 'SM', 'ST', 'SV', 'TL', 'TN', 'TR', 'UA', 'VA', 'VG', 'XK'
    }
    
    def __init__(self):
        self.features = {}
    
    def extract(self, ifml: Dict) -> Dict:
        """Extract all features from IFML message"""
        self.features = {}
        
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
        
        return self.features
    
    def _get_message(self, ifml: Dict) -> Optional[Dict]:
        """Navigate to Message level"""
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
        """Extract basic payment info"""
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
                    break
    
    def _extract_party_features(self, party_data: Dict, prefix: str):
        """Extract features for a single party"""
        self.features[f'{prefix}_present'] = True
        
        basic_party = party_data.get('BasicPartyInf', {})
        account_party = party_data.get('AccountPartyInf', {})
        
        if not basic_party and account_party:
            basic_party = account_party.get('BasicPartyInf', account_party)
        
        self._extract_id_info(basic_party, account_party, prefix)
        self._extract_address_info(basic_party, prefix)
        self._extract_address_info(account_party, prefix)
        self._extract_country(party_data, basic_party, prefix)
        
        country = self.features.get(f'{prefix}_country', '')
        self.features[f'{prefix}_needs_iban'] = country in self.IBAN_COUNTRIES
        
        address_inf = basic_party.get('AddressInf', [])
        if address_inf:
            self.features[f'{prefix}_has_name'] = True
            self.features[f'{prefix}_address_lines'] = len(address_inf)
        else:
            self.features[f'{prefix}_has_name'] = False
            self.features[f'{prefix}_address_lines'] = 0
    
    def _extract_id_info(self, basic_party: Dict, account_party: Dict, prefix: str):
        """Extract ID information (BIC, IBAN, NCH, Account)"""
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
        """Classify identifiers into BIC, IBAN, NCH, Account"""
        self.features[f'{prefix}_has_bic'] = False
        self.features[f'{prefix}_has_iban'] = False
        self.features[f'{prefix}_has_nch'] = False
        self.features[f'{prefix}_has_account'] = False
        
        bic = None
        iban = None
        nch = None
        account = None
        
        if id_type == 'S':
            bic = id_text
        elif id_type in ('P', 'D'):
            account = id_text
        
        if acct_type == 'S':
            bic = acct_text
        elif acct_type in ('P', 'D', '') and acct_text:
            account = acct_text
        
        for text in [id_text, acct_text]:
            if text:
                parsed = self._parse_composite_id(text)
                if parsed.get('bic'):
                    bic = parsed['bic']
                if parsed.get('account'):
                    account = parsed['account']
        
        if adr_bank_id and self._looks_like_bic(adr_bank_id):
            bic = adr_bank_id
        
        if account and self._looks_like_iban(account):
            iban = account
            account = None
        
        if account and self._looks_like_nch(account):
            nch = account
            account = None
        
        if bic:
            self.features[f'{prefix}_has_bic'] = True
            self.features[f'{prefix}_bic'] = bic
            self._validate_bic(prefix, bic)
        
        if iban:
            self.features[f'{prefix}_has_iban'] = True
            self.features[f'{prefix}_iban'] = iban
            self._validate_iban(prefix, iban)
        
        if nch:
            self.features[f'{prefix}_has_nch'] = True
            self.features[f'{prefix}_nch'] = nch
            self._validate_nch(prefix, nch)
        
        if account:
            self.features[f'{prefix}_has_account'] = True
            self.features[f'{prefix}_account'] = account
            self._validate_account(prefix, account)
    
    def _parse_composite_id(self, text: str) -> Dict:
        """Parse composite ID like 'BNPATHBK/(AC1016850017)'"""
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
        text = text.strip()
        if re.match(r'^\d{9}$', text):
            return True
        if re.match(r'^\d{6}$', text):
            return True
        return False
    
    def _validate_bic(self, prefix: str, bic: str):
        bic = bic.strip().upper()
        self.features[f'{prefix}_bic_length'] = len(bic)
        valid_format = bool(re.match(r'^[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}([A-Z0-9]{3})?$', bic))
        self.features[f'{prefix}_bic_valid_format'] = valid_format
        
        if len(bic) >= 6:
            country = bic[4:6]
            self.features[f'{prefix}_bic_country'] = country
            valid_country = bool(re.match(r'^[A-Z]{2}$', country))
            self.features[f'{prefix}_bic_valid_country'] = valid_country
        else:
            self.features[f'{prefix}_bic_valid_country'] = False
    
    def _validate_iban(self, prefix: str, iban: str):
        original = iban
        iban = iban.replace(' ', '').replace('-', '').upper()
        
        valid_format = bool(re.match(r'^[A-Z]{2}[0-9]{2}[A-Z0-9]+$', iban))
        self.features[f'{prefix}_iban_valid_format'] = valid_format
        
        if len(iban) >= 2:
            self.features[f'{prefix}_iban_country'] = iban[:2]
        
        checksum_valid = self._validate_iban_checksum(iban)
        self.features[f'{prefix}_iban_checksum_valid'] = checksum_valid
        
        self.features[f'{prefix}_iban_needs_formatting'] = (
            ' ' in original or '-' in original or original != original.upper()
        )
        
        bic_country = self.features.get(f'{prefix}_bic_country', '')
        iban_country = self.features.get(f'{prefix}_iban_country', '')
        if bic_country and iban_country:
            self.features[f'{prefix}_bic_iban_match'] = (bic_country == iban_country)
        else:
            self.features[f'{prefix}_bic_iban_match'] = True
    
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
        
        if len(nch_clean) == 9:
            self.features[f'{prefix}_nch_type'] = 'FEDABA'
            self.features[f'{prefix}_nch_valid'] = True
            self.features[f'{prefix}_fedaba_checksum_valid'] = self._validate_aba_checksum(nch_clean)
        elif len(nch_clean) == 6:
            self.features[f'{prefix}_nch_type'] = 'CHIPS_OR_SORT'
            self.features[f'{prefix}_nch_valid'] = True
            self.features[f'{prefix}_fedaba_checksum_valid'] = True
        else:
            self.features[f'{prefix}_nch_type'] = 'OTHER'
            self.features[f'{prefix}_nch_valid'] = False
            self.features[f'{prefix}_fedaba_checksum_valid'] = True
        
        self.features[f'{prefix}_nch_sources'] = 1
        self.features[f'{prefix}_nch_consistent'] = True
        self.features[f'{prefix}_nch_needs_formatting'] = False
        self.features[f'{prefix}_nch_validation_applicable'] = (
            self.features.get(f'{prefix}_country', '') == 'US' or
            self.features.get('is_domestic', False)
        )
    
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
        self.features[f'{prefix}_is_clabe'] = (len(clean) == 18 and clean.isdigit())
    
    def _extract_address_info(self, party_data: Dict, prefix: str):
        address_inf = party_data.get('AddressInf', [])
        if address_inf and isinstance(address_inf, list):
            address_text = ' '.join(
                line.get('text', '') for line in address_inf if isinstance(line, dict)
            )
            self.features[f'{prefix}_address_text'] = address_text
    
    def _extract_country(self, party_data: Dict, basic_party: Dict, prefix: str):
        country = (
            party_data.get('Country') or
            party_data.get('ResidenceCountry') or
            party_data.get('MailingCountry') or
            basic_party.get('Country') or
            ''
        )
        self.features[f'{prefix}_country'] = country
        self.features[f'{prefix}_is_domestic'] = (country == 'US')


# =============================================================================
# CODE PREDICTOR - Emits predicted 8XXX and 9XXX codes
# =============================================================================

PARTY_PREFIXES = ['orig', 'ordi', 'send', 'dbt', 'cdt', 'bnf', 'bnf_bank', 'intm', 'acwi']


class CodePredictor:
    """Predicts 8XXX and 9XXX codes based on extracted features."""
    
    def __init__(self, features: Dict):
        self.features = features
        self.predictions = []
    
    def predict_all(self) -> List[Tuple[str, str, str]]:
        """Run all prediction rules and return predicted codes"""
        self.predictions = []
        
        for party in PARTY_PREFIXES:
            if not self._get(party, 'present', False):
                continue
            
            # 8XXX Validation Errors
            self._check_8001(party)
            self._check_8004(party)
            self._check_8022(party)
            self._check_8026(party)
            self._check_8894(party)
            self._check_8895(party)
            self._check_8898(party)
            self._check_8892(party)
            
            # 9XXX No-Directory Repairs
            self._check_9000(party)
            self._check_9002(party)
            self._check_9006(party)
            self._check_9017(party)
            self._check_9018(party)
            self._check_9021(party)
            self._check_9022(party)
            
            # 9XXX Directory-Dependent
            self._check_9004(party)
            self._check_9005(party)
            self._check_9008(party)
            self._check_9477(party)
        
        return self.predictions
    
    def _get(self, party: str, key: str, default=None):
        return self.features.get(f'{party}_{key}', default)
    
    def _emit(self, code: str, party: str, reason: str):
        self.predictions.append((code, party, reason))
    
    # 8XXX VALIDATION RULES
    
    def _check_8001(self, party: str):
        if not self._get(party, 'has_bic', False):
            return
        if not self._get(party, 'bic_valid_format', True):
            self._emit('8001', party, f"BIC format invalid: {self._get(party, 'bic', '')}")
        elif not self._get(party, 'bic_valid_country', True):
            self._emit('8001', party, f"BIC country invalid: {self._get(party, 'bic_country', '')}")
    
    def _check_8004(self, party: str):
        if self._get(party, 'needs_iban', False) and not self._get(party, 'has_iban', False):
            self._emit('8004', party, f"IBAN required for {self._get(party, 'country', '')} but missing")
    
    def _check_8022(self, party: str):
        if self._get(party, 'has_iban', False) and self._get(party, 'has_bic', False):
            if not self._get(party, 'bic_iban_match', True):
                self._emit('8022', party, f"BIC({self._get(party, 'bic_country', '')}) != IBAN({self._get(party, 'iban_country', '')})")
    
    def _check_8026(self, party: str):
        if self._get(party, 'has_nch', False) and self._get(party, 'nch_sources', 0) > 1:
            if not self._get(party, 'nch_consistent', True):
                self._emit('8026', party, "NCH values inconsistent")
    
    def _check_8894(self, party: str):
        if not self._get(party, 'has_iban', False):
            return
        if not self._get(party, 'iban_valid_format', True):
            self._emit('8894', party, f"IBAN format invalid: {self._get(party, 'iban', '')}")
        elif not self._get(party, 'iban_checksum_valid', True):
            self._emit('8894', party, f"IBAN checksum invalid: {self._get(party, 'iban', '')}")
    
    def _check_8895(self, party: str):
        if not self._get(party, 'has_nch', False) or not self._get(party, 'nch_validation_applicable', False):
            return
        if not self._get(party, 'nch_valid', True):
            self._emit('8895', party, f"NCH format invalid: {self._get(party, 'nch', '')}")
        elif not self._get(party, 'fedaba_checksum_valid', True):
            self._emit('8895', party, f"NCH checksum invalid: {self._get(party, 'nch', '')}")
    
    def _check_8898(self, party: str):
        if self._get(party, 'has_iban', False) and self._get(party, 'iban_valid_format', True):
            if not self._get(party, 'iban_checksum_valid', True):
                self._emit('8898', party, f"IBAN mod-97 failed: {self._get(party, 'iban', '')}")
    
    def _check_8892(self, party: str):
        if self._get(party, 'has_account', False) and not self._get(party, 'account_valid_format', True):
            self._emit('8892', party, f"Account format invalid: {self._get(party, 'account', '')}")
    
    # 9XXX NO-DIRECTORY RULES
    
    def _check_9000(self, party: str):
        if self._get(party, 'has_nch', False) and self._get(party, 'nch_has_dirty_chars', False):
            self._emit('9000', party, f"NCH cleaned: {self._get(party, 'nch', '')}")
    
    def _check_9002(self, party: str):
        if self._get(party, 'has_account', False) and self._get(party, 'account_has_dirty_chars', False):
            self._emit('9002', party, f"Account cleaned: {self._get(party, 'account', '')}")
    
    def _check_9006(self, party: str):
        if self._get(party, 'has_iban', False) and self._get(party, 'iban_needs_formatting', False):
            self._emit('9006', party, f"IBAN formatted: {self._get(party, 'iban', '')}")
    
    def _check_9017(self, party: str):
        if self._get(party, 'has_multiple_ids', False):
            self._emit('9017', party, "Multiple IDs split")
    
    def _check_9018(self, party: str):
        if self._get(party, 'has_duplicate_info', False):
            self._emit('9018', party, "Duplicate removed")
    
    def _check_9021(self, party: str):
        if self._get(party, 'has_nch', False) and self._get(party, 'nch_needs_formatting', False):
            self._emit('9021', party, f"NCH formatted: {self._get(party, 'nch', '')}")
    
    def _check_9022(self, party: str):
        if self._get(party, 'has_account', False) and self._get(party, 'account_needs_length_fix', False):
            self._emit('9022', party, f"Account length fixed: {self._get(party, 'account', '')}")
    
    # 9XXX DIRECTORY-DEPENDENT
    
    def _check_9004(self, party: str):
        if party not in ('cdt', 'bnf'):
            return
        if self._get(party, 'needs_iban', False) and not self._get(party, 'has_iban', False):
            if self._get(party, 'has_account', False):
                self._emit('9004', party, "IBAN derivation eligible (directory dependent)")
    
    def _check_9005(self, party: str):
        if self._get(party, 'has_nch', False) and not self._get(party, 'has_bic', False):
            self._emit('9005', party, "BIC from NCH eligible (directory dependent)")
    
    def _check_9008(self, party: str):
        if self._get(party, 'has_iban', False) and not self._get(party, 'has_bic', False):
            self._emit('9008', party, "BIC from IBAN eligible (directory dependent)")
    
    def _check_9477(self, party: str):
        if self._get(party, 'has_bic', False) and self._get(party, 'bic_length', 0) == 8:
            self._emit('9477', party, "BIC 8→11 expansion eligible (directory dependent)")


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
                       'ErrorCode', 'Code', 'code']:
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
    """Result for a single payment"""
    payment_id: str
    predicted_codes: List[Tuple[str, str, str]]
    actual_codes: Set[str]
    true_positives: List[str] = field(default_factory=list)
    false_positives: List[str] = field(default_factory=list)
    false_negatives: List[str] = field(default_factory=list)
    success: bool = False
    
    def compute_match(self):
        predicted_base = {code for code, _, _ in self.predicted_codes}
        self.true_positives = list(predicted_base & self.actual_codes)
        self.false_positives = list(predicted_base - self.actual_codes)
        self.false_negatives = list(self.actual_codes - predicted_base)
        
        # Filter to 8xxx/9xxx only
        relevant_actual = {c for c in self.actual_codes if c.startswith(('8', '9'))}
        relevant_predicted = {c for c in predicted_base if c.startswith(('8', '9'))}
        self.success = (relevant_predicted == relevant_actual)
    
    def to_dict(self) -> Dict:
        return {
            'payment_id': self.payment_id,
            'predicted_codes': [{'code': c, 'party': p, 'reason': r} for c, p, r in self.predicted_codes],
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
        print(f"\nPredicted ({len(predicted_codes)}):")
        for code, party, reason in predicted_codes:
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
        stats['f1'] = 2 * stats['precision'] * stats['recall'] / (stats['precision'] + stats['recall']) if (stats['precision'] + stats['recall']) > 0 else 0
    
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
    print("ACE PELICAN RULE VERIFICATION REPORT")
    print("=" * 70)
    print(f"\n📊 SUMMARY")
    print(f"   Total: {s['total_payments']} | Success: {s['successful']} | Failed: {s['failed']}")
    print(f"   Success Rate: {s['success_rate']:.2%}")
    
    for label, key in [("8XXX VALIDATION", 'code_stats_8xxx'), ("9XXX REPAIR", 'code_stats_9xxx')]:
        stats = report.get(key, {})
        if stats:
            print(f"\n📋 {label}")
            print(f"   {'Code':<8} {'TP':<6} {'FP':<6} {'FN':<6} {'Prec':<10} {'Recall':<10}")
            print(f"   {'-'*50}")
            for code in sorted(stats.keys()):
                st = stats[code]
                print(f"   {code:<8} {st['tp']:<6} {st['fp']:<6} {st['fn']:<6} {st['precision']:<10.2%} {st['recall']:<10.2%}")
    
    failures = report.get('failures', [])
    if failures:
        print(f"\n⚠️  FAILURES ({len(failures)} shown):")
        for f in failures[:5]:
            print(f"\n   {f['payment_id']}")
            print(f"   Predicted: {[p['code'] for p in f['predicted_codes']]}")
            print(f"   Actual: {f['actual_codes']}")
            print(f"   FP: {f['false_positives']} | FN: {f['false_negatives']}")


def main():
    parser = argparse.ArgumentParser(description='Process IFML, predict 8XXX/9XXX codes, compare with actual')
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
