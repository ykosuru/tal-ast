#!/usr/bin/env python3
"""
ACE Pelican IFML Verification Tool.
Uses enhanced parser that handles all ID type variations.
"""

import json
import sys
import re
import argparse
from pathlib import Path
from typing import Dict, Set, List, Tuple, Optional, Any
from dataclasses import dataclass, field

PARTY_PREFIXES = ['orig', 'send', 'dbt', 'cdt', 'intm', 'bnf', 'acwi', 'ordi']

PARTY_MAPPING = {
    'OriginatingPartyInf': 'orig', 'OriginatingPartyInfo': 'orig',
    'SendingBankInf': 'send', 'SendingBankInfo': 'send',
    'DebitPartyInf': 'dbt', 'DebitPartyInfo': 'dbt',
    'CreditPartyInf': 'cdt', 'CreditPartyInfo': 'cdt',
    'IntermediaryBankInf': 'intm', 'IntermediaryBankInfo': 'intm',
    'BeneficiaryBankInf': 'bnf', 'BeneficiaryBankInfo': 'bnf',
    'BeneficiaryPartyInf': 'bnf',  # Both map to bnf
    'AccountWithInstitution': 'acwi',
    'OrderingInstitution': 'ordi',
}

DIRECTORY_DEPENDENT_CODES = {
    '8004', '8036', '8464', '8465', '8472',
    '9004', '9005', '9007', '9008', '9013', '9024', '9476', '9477', '9479', '9480', '9961', '9970', '9985',
}

VALID_COUNTRIES = {'US', 'GB', 'DE', 'FR', 'FI', 'SE', 'NO', 'DK', 'NL', 'BE', 'AT', 'CH', 'IT', 'ES', 'PT', 'IE', 'AU', 'NZ', 'CA', 'JP', 'CN', 'SG', 'HK'}

INCLUDE_DIRECTORY = False

@dataclass
class VerificationResult:
    trn_id: str
    predicted: Set[str]
    actual: Set[str]
    passed: bool = field(init=False)
    directory_codes: Set[str] = field(default_factory=set)
    directory_eligible: List[Tuple[str, str, str]] = field(default_factory=list)
    features: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        self.passed = self.predicted == self.actual

# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def is_valid_bic(s: str) -> Tuple[bool, bool]:
    """Check if string is valid BIC. Returns (format_valid, country_valid)."""
    if not s or len(s) not in (8, 11):
        return False, False
    s = s.upper()
    if not re.match(r'^[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}([A-Z0-9]{3})?$', s):
        return False, False
    country = s[4:6]
    return True, country in VALID_COUNTRIES

def is_valid_iban(s: str) -> Tuple[bool, bool]:
    """Check if string is valid IBAN. Returns (format_valid, checksum_valid)."""
    if not s or len(s) < 5:
        return False, False
    s = s.upper().replace(' ', '').replace('-', '')
    if not re.match(r'^[A-Z]{2}[0-9]{2}[A-Z0-9]+$', s):
        return False, False
    if len(s) < 15 or len(s) > 34:
        return False, False
    # Checksum
    try:
        rearranged = s[4:] + s[:4]
        numeric = ''.join(str(ord(c) - ord('A') + 10) if c.isalpha() else c for c in rearranged)
        return True, int(numeric) % 97 == 1
    except:
        return True, False

def looks_like_bic(s: str) -> bool:
    """Quick check if string looks like a BIC."""
    if not s or len(s) not in (8, 11):
        return False
    return bool(re.match(r'^[A-Z]{4}[A-Z]{2}[A-Z0-9]{2,5}$', s.upper()))

def looks_like_iban(s: str) -> bool:
    """Quick check if string looks like an IBAN."""
    if not s or len(s) < 15:
        return False
    s = s.upper().replace(' ', '')
    return bool(re.match(r'^[A-Z]{2}[0-9]{2}[A-Z0-9]+$', s))

def looks_like_nch(s: str) -> str:
    """Check what type of NCH this looks like. Returns type or empty string."""
    if not s:
        return ''
    s = s.strip()
    if len(s) == 9 and s.isdigit():
        return 'FEDABA'
    if len(s) == 6 and s.isdigit():
        return 'CHIPS_OR_SORT'
    if len(s) == 3 and s.isdigit():
        return 'BSB'  # Australian BSB or Finnish bank code
    return ''

# =============================================================================
# FEATURE EXTRACTOR
# =============================================================================

class FeatureExtractor:
    def __init__(self, debug: bool = False):
        self.features = {}
        self.debug = debug
    
    def extract(self, data: Dict) -> Dict:
        self.features = {}
        
        # Navigate to Message
        message = self._find_message(data)
        if not message:
            if self.debug:
                print("  [EXTRACT] No message found!")
            return self.features
        
        # Get PartyInf - check both Message level and BasicPayment level
        party_inf = message.get('PartyInf') or message.get('PartyInfo') or {}
        if not party_inf:
            bp = message.get('BasicPayment', {})
            party_inf = bp.get('PartyInf') or bp.get('PartyInfo') or {}
        
        if self.debug:
            print(f"  [EXTRACT] PartyInf keys: {list(party_inf.keys()) if party_inf else 'EMPTY'}")
        
        # Process each party
        processed = set()
        for ifml_name, prefix in PARTY_MAPPING.items():
            party_data = party_inf.get(ifml_name)
            if party_data:
                if self.debug:
                    print(f"  [EXTRACT] Processing {ifml_name} -> {prefix}")
                if prefix in processed:
                    self._merge_party(party_data, prefix)
                else:
                    self._extract_party(party_data, prefix)
                    processed.add(prefix)
        
        # Set defaults for missing parties
        for prefix in PARTY_PREFIXES:
            if f'{prefix}_present' not in self.features:
                self.features[f'{prefix}_present'] = False
        
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
        return data if isinstance(data, dict) and 'PartyInf' in data else None
    
    def _extract_party(self, party_data: Dict, prefix: str):
        """Extract features from a party."""
        if isinstance(party_data, list):
            party_data = party_data[0] if party_data else {}
        
        self.features[f'{prefix}_present'] = True
        
        # Get BasicPartyInf or AccountPartyInf
        basic = party_data.get('BasicPartyInf') or party_data.get('BasicPartyInfo') or {}
        account = party_data.get('AccountPartyInf') or party_data.get('AccountPartyInfo') or {}
        if not basic and account:
            basic = account.get('BasicPartyInf', account)
        
        if isinstance(basic, list):
            basic = basic[0] if basic else {}
        
        # Extract ID
        id_field = basic.get('ID')
        bic, iban, nch, account_num = None, None, None, None
        
        if id_field:
            self.features[f'{prefix}_has_id'] = True
            if isinstance(id_field, dict):
                id_type = id_field.get('Type') or id_field.get('@Type') or ''
                id_text = id_field.get('text') or id_field.get('#text') or ''
                
                # Type-based detection
                if id_type.upper() in ('S', 'SWIFT', 'BIC'):
                    bic = id_text
                elif id_type.upper() == 'IBAN':
                    iban = id_text
                else:
                    # Auto-detect
                    if looks_like_bic(id_text):
                        bic = id_text
                    elif looks_like_iban(id_text):
                        iban = id_text
                    else:
                        account_num = id_text
            else:
                # Plain string - auto-detect
                id_text = str(id_field)
                if looks_like_iban(id_text):
                    iban = id_text
                elif looks_like_bic(id_text):
                    bic = id_text
                else:
                    account_num = id_text
        
        # Extract AdrBankID (NCH)
        adr_bank_id = basic.get('AdrBankID')
        if adr_bank_id:
            if isinstance(adr_bank_id, dict):
                nch = adr_bank_id.get('text') or adr_bank_id.get('#text') or ''
            else:
                nch = str(adr_bank_id)
            self.features[f'{prefix}_has_adr_bank_id'] = True
        
        # Also check AcctIDInf for IDs
        acct_id_inf = basic.get('AcctIDInf') or account.get('AcctIDInf') or {}
        if acct_id_inf:
            acct_id = acct_id_inf.get('ID')
            if acct_id:
                if isinstance(acct_id, dict):
                    acct_type = acct_id.get('Type') or ''
                    acct_text = acct_id.get('text') or ''
                    if acct_type.upper() == 'IBAN' or looks_like_iban(acct_text):
                        iban = iban or acct_text
                    else:
                        account_num = account_num or acct_text
                elif looks_like_iban(str(acct_id)):
                    iban = iban or str(acct_id)
                else:
                    account_num = account_num or str(acct_id)
        
        # Store extracted values
        if bic:
            self.features[f'{prefix}_has_bic'] = True
            self.features[f'{prefix}_bic'] = bic
            fmt, ctry = is_valid_bic(bic)
            self.features[f'{prefix}_bic_valid_format'] = fmt
            self.features[f'{prefix}_bic_valid_country'] = ctry
            if len(bic) >= 6:
                self.features[f'{prefix}_bic_country'] = bic[4:6].upper()
        
        if iban:
            self.features[f'{prefix}_has_iban'] = True
            self.features[f'{prefix}_iban'] = iban
            fmt, cksum = is_valid_iban(iban)
            self.features[f'{prefix}_iban_valid_format'] = fmt
            self.features[f'{prefix}_iban_checksum_valid'] = cksum
            if len(iban) >= 2:
                self.features[f'{prefix}_iban_country'] = iban[:2].upper()
        
        if nch:
            self.features[f'{prefix}_has_nch'] = True
            self.features[f'{prefix}_nch'] = nch
            nch_type = looks_like_nch(nch)
            self.features[f'{prefix}_nch_type'] = nch_type
        
        if account_num:
            self.features[f'{prefix}_has_account'] = True
            self.features[f'{prefix}_account'] = account_num
        
        # BIC/IBAN country match
        bic_country = self.features.get(f'{prefix}_bic_country')
        iban_country = self.features.get(f'{prefix}_iban_country')
        if bic_country and iban_country:
            self.features[f'{prefix}_bic_iban_match'] = (bic_country == iban_country)
        
        # Country
        country = party_data.get('MailingCountry') or party_data.get('Country') or basic.get('Country') or ''
        if country:
            self.features[f'{prefix}_country'] = country
    
    def _merge_party(self, party_data: Dict, prefix: str):
        """Merge additional party data (e.g., BeneficiaryPartyInf into bnf)."""
        if isinstance(party_data, list):
            party_data = party_data[0] if party_data else {}
        
        basic = party_data.get('BasicPartyInf') or party_data.get('BasicPartyInfo') or {}
        if isinstance(basic, list):
            basic = basic[0] if basic else {}
        
        # Extract ID and merge if not already present
        id_field = basic.get('ID')
        if id_field and not self.features.get(f'{prefix}_has_iban'):
            if isinstance(id_field, dict):
                id_text = id_field.get('text') or id_field.get('#text') or ''
            else:
                id_text = str(id_field)
            
            if looks_like_iban(id_text):
                self.features[f'{prefix}_has_iban'] = True
                self.features[f'{prefix}_iban'] = id_text
                fmt, cksum = is_valid_iban(id_text)
                self.features[f'{prefix}_iban_valid_format'] = fmt
                self.features[f'{prefix}_iban_checksum_valid'] = cksum
                if len(id_text) >= 2:
                    self.features[f'{prefix}_iban_country'] = id_text[:2].upper()
                
                # Update BIC/IBAN match
                bic_country = self.features.get(f'{prefix}_bic_country')
                iban_country = self.features.get(f'{prefix}_iban_country')
                if bic_country and iban_country:
                    self.features[f'{prefix}_bic_iban_match'] = (bic_country == iban_country)

# =============================================================================
# RULE ENGINE
# =============================================================================

class RuleEngine:
    def __init__(self, features: Dict, debug: bool = False):
        self.f = features
        self.codes = set()
        self.directory_eligible = []
        self.debug = debug
    
    def predict(self) -> Set[str]:
        self.codes = set()
        for prefix in PARTY_PREFIXES:
            if self.f.get(f'{prefix}_present'):
                self._check_8xxx(prefix)
                self._check_9xxx(prefix)
                if INCLUDE_DIRECTORY:
                    self._check_directory(prefix)
        return self.codes
    
    def _get(self, prefix: str, key: str, default=None):
        return self.f.get(f'{prefix}_{key}', default)
    
    def _emit(self, code: str):
        self.codes.add(code)
    
    def _check_8xxx(self, p: str):
        # 8001: Invalid BIC
        if self._get(p, 'has_bic'):
            if not self._get(p, 'bic_valid_format', True):
                if self.debug: print(f"  [8001 {p}] bic_valid_format=False")
                self._emit('8001')
            elif not self._get(p, 'bic_valid_country', True):
                if self.debug: print(f"  [8001 {p}] bic_valid_country=False")
                self._emit('8001')
        
        # 8022: BIC/IBAN country mismatch
        if self._get(p, 'has_bic') and self._get(p, 'has_iban'):
            if self._get(p, 'bic_iban_match') == False:
                if self.debug: print(f"  [8022 {p}] bic_iban_match=False")
                self._emit('8022')
        
        # 8022/8026: BIC vs NCH type mismatch
        if self._get(p, 'has_bic') and self._get(p, 'has_nch'):
            bic_country = self._get(p, 'bic_country', '')
            nch_type = self._get(p, 'nch_type', '')
            if self.debug: print(f"  [8022/8026 {p}] bic_country={bic_country}, nch_type={nch_type}")
            
            if nch_type == 'FEDABA' and bic_country and bic_country != 'US':
                if self.debug: print(f"  [8022 {p}] FEDABA with non-US BIC")
                self._emit('8022')
                self._emit('8026')
            elif nch_type == 'BSB' and bic_country and bic_country != 'AU':
                if self.debug: print(f"  [8022 {p}] BSB with non-AU BIC ({bic_country})")
                self._emit('8022')
                self._emit('8026')
        
        # 8894: Invalid IBAN
        if self._get(p, 'has_iban'):
            if not self._get(p, 'iban_valid_format', True):
                if self.debug: print(f"  [8894 {p}] iban_valid_format=False")
                self._emit('8894')
            elif not self._get(p, 'iban_checksum_valid', True):
                if self.debug: print(f"  [8894 {p}] iban_checksum_valid=False")
                self._emit('8894')
            else:
                # Check IBAN bank code vs NCH mismatch
                iban = self._get(p, 'iban', '')
                nch = self._get(p, 'nch', '')
                if iban and nch and len(iban) >= 7 and len(nch) == 3:
                    iban_bank = iban[4:7]
                    if iban_bank != nch:
                        if self.debug: print(f"  [8894 {p}] IBAN bank code {iban_bank} != NCH {nch}")
                        self._emit('8894')
    
    def _check_9xxx(self, p: str):
        if self._get(p, 'account_has_dirty_chars'): self._emit('9002')
        if self._get(p, 'iban_needs_formatting'): self._emit('9006')
        if self._get(p, 'has_multiple_ids'): self._emit('9017')
        if self._get(p, 'has_duplicate_info'): self._emit('9018')
    
    def _check_directory(self, p: str):
        if self._get(p, 'has_bic'):
            self.directory_eligible.append(('8022', p, 'BIC→NCH derivation → IF INCONSISTENT: 8022'))
            self.directory_eligible.append(('8026', p, 'NCH derivation → IF INCONSISTENT: 8026'))
        if self._get(p, 'has_iban'):
            self.directory_eligible.append(('8894', p, 'IBAN→NCH derivation → IF INVALID: 8894'))
    
    def get_directory_eligible(self):
        return self.directory_eligible

# =============================================================================
# RESPONSE PARSING
# =============================================================================

def extract_actual_codes(response: Dict, include_dir: bool = False) -> Set[str]:
    codes = set()
    
    # Find AuditTrail
    audit = None
    for path in [
        ['AuditTrail'],
        ['IFML', 'File', 'Message', 'AuditTrail'],
        ['IFML', 'File', 'AuditTrail'],
    ]:
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
        if code and (code.startswith('8') or code.startswith('9')):
            if include_dir or code not in DIRECTORY_DEPENDENT_CODES:
                codes.add(code)
    
    return codes

def get_transaction_id(data: Dict) -> str:
    for path in [
        ['Request', 'IFML', 'File', 'Message', 'BasicPayment', 'TransactionUID'],
        ['Request', 'IFML', 'File', 'Message', 'BasicPayment', 'TransactionID'],
    ]:
        current = data
        for key in path:
            if isinstance(current, dict):
                current = current.get(key)
            else:
                break
        if isinstance(current, str):
            return current
    return "UNKNOWN"

# =============================================================================
# MAIN
# =============================================================================

def process_payment(pid: str, request: Dict, response: Dict, debug: bool = False) -> VerificationResult:
    if debug:
        print(f"\n=== DEBUG: {pid} ===")
    
    extractor = FeatureExtractor(debug=debug)
    features = extractor.extract(request)
    
    if debug:
        print("  Features:")
        for k, v in sorted(features.items()):
            if v is not None and v != False and v != 0 and v != '':
                print(f"    {k}: {v}")
    
    engine = RuleEngine(features, debug=debug)
    predicted = engine.predict()
    
    actual = extract_actual_codes(response, INCLUDE_DIRECTORY)
    all_actual = extract_actual_codes(response, True)
    
    result = VerificationResult(trn_id=pid, predicted=predicted, actual=actual)
    result.directory_codes = all_actual - actual
    result.directory_eligible = engine.get_directory_eligible()
    result.features = features
    
    if debug:
        print(f"\n  Predicted: {sorted(predicted)}")
        print(f"  Actual: {sorted(actual)}")
        print(f"  FN: {sorted(actual - predicted)}")
        print(f"  FP: {sorted(predicted - actual)}")
    
    return result

def process_file(filepath: Path, verbose: bool = False, debug: bool = False) -> List[VerificationResult]:
    results = []
    with open(filepath) as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        if 'Request' in data and 'Response' in data:
            pid = get_transaction_id(data)
            results.append(process_payment(pid, data, data.get('Response', {}), debug))
        else:
            for key, val in data.items():
                if isinstance(val, dict) and 'Request' in val and 'Response' in val:
                    results.append(process_payment(key, val, val.get('Response', {}), debug))
    return results

def print_results(results: List[VerificationResult]):
    passed = failed = 0
    for r in results:
        if r.passed:
            passed += 1
            print(f"{r.trn_id} PASS" + (f" [Dir: {sorted(r.directory_codes)}]" if r.directory_codes else ""))
        else:
            failed += 1
            print(f"\n{r.trn_id} FAIL")
            print(f"  PREDICTED: {sorted(r.predicted)}")
            print(f"  ACTUAL: {sorted(r.actual)}")
            if r.actual - r.predicted:
                print(f"  FN: {sorted(r.actual - r.predicted)}")
            if r.predicted - r.actual:
                print(f"  FP: {sorted(r.predicted - r.actual)}")
    
    print(f"\n{'='*60}")
    print(f"SUMMARY: {passed}/{len(results)} passed ({100*passed/len(results):.1f}%), {failed} failed")
    print(f"{'='*60}")

def main():
    global INCLUDE_DIRECTORY
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--include-directory', action='store_true')
    args = parser.parse_args()
    
    INCLUDE_DIRECTORY = args.include_directory
    path = Path(args.input)
    
    if path.is_file():
        results = process_file(path, args.verbose, args.debug)
    else:
        results = []
        for f in sorted(path.glob('*.json')):
            results.extend(process_file(f, args.verbose, args.debug))
    
    if results:
        print_results(results)
    else:
        print("No payments found")

if __name__ == '__main__':
    main()
