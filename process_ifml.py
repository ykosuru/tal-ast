#!/usr/bin/env python3
"""
ACE Pelican IFML Verification Tool - Debug Version
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
    'BeneficiaryPartyInf': 'bnf',
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
    features: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        self.passed = self.predicted == self.actual

def is_valid_bic(s: str) -> Tuple[bool, bool]:
    if not s or len(s) not in (8, 11):
        return False, False
    s = s.upper()
    if not re.match(r'^[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}([A-Z0-9]{3})?$', s):
        return False, False
    country = s[4:6]
    return True, country in VALID_COUNTRIES

def is_valid_iban(s: str) -> Tuple[bool, bool]:
    if not s or len(s) < 5:
        return False, False
    s = s.upper().replace(' ', '').replace('-', '')
    if not re.match(r'^[A-Z]{2}[0-9]{2}[A-Z0-9]+$', s):
        return False, False
    if len(s) < 15 or len(s) > 34:
        return False, False
    try:
        rearranged = s[4:] + s[:4]
        numeric = ''.join(str(ord(c) - ord('A') + 10) if c.isalpha() else c for c in rearranged)
        return True, int(numeric) % 97 == 1
    except:
        return True, False

def looks_like_bic(s: str) -> bool:
    if not s or len(s) not in (8, 11):
        return False
    return bool(re.match(r'^[A-Z]{4}[A-Z]{2}[A-Z0-9]{2,5}$', s.upper()))

def looks_like_iban(s: str) -> bool:
    if not s or len(s) < 15:
        return False
    s = s.upper().replace(' ', '')
    return bool(re.match(r'^[A-Z]{2}[0-9]{2}[A-Z0-9]+$', s))

def looks_like_nch(s: str) -> str:
    if not s:
        return ''
    s = s.strip()
    if len(s) == 9 and s.isdigit():
        return 'FEDABA'
    if len(s) == 6 and s.isdigit():
        return 'CHIPS_OR_SORT'
    if len(s) == 3 and s.isdigit():
        return 'BSB'
    return ''

class FeatureExtractor:
    def __init__(self, debug: bool = False):
        self.features = {}
        self.debug = debug
    
    def extract(self, data: Dict) -> Dict:
        self.features = {}
        
        if self.debug:
            print(f"\n[DEBUG] === STARTING EXTRACTION ===")
            print(f"[DEBUG] Top-level keys: {list(data.keys())[:10]}")
        
        # Navigate to Message
        message = self._find_message(data)
        if not message:
            if self.debug:
                print("[DEBUG] ERROR: No message found!")
            return self.features
        
        if self.debug:
            print(f"[DEBUG] Message keys: {list(message.keys())[:10]}")
        
        # Get PartyInf - check multiple locations
        party_inf = None
        
        # Location 1: Message.PartyInf
        party_inf = message.get('PartyInf') or message.get('PartyInfo')
        if self.debug:
            print(f"[DEBUG] Message.PartyInf: {type(party_inf).__name__} -> {list(party_inf.keys()) if isinstance(party_inf, dict) else 'NOT FOUND'}")
        
        # Location 2: Message.BasicPayment.PartyInf
        if not party_inf:
            bp = message.get('BasicPayment', {})
            if self.debug:
                print(f"[DEBUG] BasicPayment keys: {list(bp.keys()) if isinstance(bp, dict) else 'NOT DICT'}")
            party_inf = bp.get('PartyInf') or bp.get('PartyInfo')
            if self.debug:
                print(f"[DEBUG] BasicPayment.PartyInf: {type(party_inf).__name__} -> {list(party_inf.keys()) if isinstance(party_inf, dict) else 'NOT FOUND'}")
        
        if not party_inf or not isinstance(party_inf, dict):
            if self.debug:
                print("[DEBUG] ERROR: No PartyInf found!")
            return self.features
        
        if self.debug:
            print(f"[DEBUG] PartyInf keys: {list(party_inf.keys())}")
        
        # Process each party
        processed = set()
        for ifml_name, prefix in PARTY_MAPPING.items():
            party_data = party_inf.get(ifml_name)
            if party_data:
                if self.debug:
                    print(f"\n[DEBUG] Found {ifml_name} -> {prefix}")
                    print(f"[DEBUG]   Type: {type(party_data).__name__}")
                    if isinstance(party_data, dict):
                        print(f"[DEBUG]   Keys: {list(party_data.keys())}")
                
                if prefix in processed:
                    if self.debug:
                        print(f"[DEBUG]   Merging into existing {prefix}")
                    self._merge_party(party_data, prefix)
                else:
                    self._extract_party(party_data, prefix)
                    processed.add(prefix)
        
        # Set defaults
        for prefix in PARTY_PREFIXES:
            if f'{prefix}_present' not in self.features:
                self.features[f'{prefix}_present'] = False
        
        return self.features
    
    def _find_message(self, data: Dict) -> Optional[Dict]:
        paths = [
            ['Request', 'IFML', 'File', 'Message'],
            ['IFML', 'File', 'Message'],
            ['File', 'Message'],
            ['Message'],
        ]
        for path in paths:
            if self.debug:
                print(f"[DEBUG] Trying path: {path}")
            current = data
            for key in path:
                if isinstance(current, dict):
                    current = current.get(key)
                    if self.debug and current:
                        print(f"[DEBUG]   Found '{key}': {type(current).__name__}")
                else:
                    current = None
                    break
            if current and isinstance(current, dict):
                if self.debug:
                    print(f"[DEBUG] SUCCESS: Found Message via {path}")
                return current
        
        # Fallback: check if data itself has PartyInf
        if isinstance(data, dict) and ('PartyInf' in data or 'PartyInfo' in data):
            if self.debug:
                print("[DEBUG] Using data itself as Message (has PartyInf)")
            return data
        
        return None
    
    def _extract_party(self, party_data: Dict, prefix: str):
        if isinstance(party_data, list):
            if self.debug:
                print(f"[DEBUG]   Party is LIST with {len(party_data)} items, using first")
            party_data = party_data[0] if party_data else {}
        
        self.features[f'{prefix}_present'] = True
        
        # Get BasicPartyInf
        basic = party_data.get('BasicPartyInf') or party_data.get('BasicPartyInfo') or {}
        account = party_data.get('AccountPartyInf') or party_data.get('AccountPartyInfo') or {}
        
        if self.debug:
            print(f"[DEBUG]   BasicPartyInf: {type(basic).__name__} -> {list(basic.keys()) if isinstance(basic, dict) else 'NOT DICT'}")
        
        if not basic and account:
            basic = account.get('BasicPartyInf', account)
        
        if isinstance(basic, list):
            if self.debug:
                print(f"[DEBUG]   BasicPartyInf is LIST, using first")
            basic = basic[0] if basic else {}
        
        # === Extract ID ===
        id_field = basic.get('ID')
        bic, iban, nch, account_num = None, None, None, None
        
        if self.debug:
            print(f"[DEBUG]   ID field: {type(id_field).__name__} = {id_field}")
        
        if id_field:
            self.features[f'{prefix}_has_id'] = True
            if isinstance(id_field, dict):
                id_type = id_field.get('Type') or id_field.get('@Type') or ''
                id_text = id_field.get('text') or id_field.get('#text') or ''
                
                if self.debug:
                    print(f"[DEBUG]   ID Type: '{id_type}', Text: '{id_text}'")
                
                if id_type.upper() in ('S', 'SWIFT', 'BIC'):
                    bic = id_text
                    if self.debug:
                        print(f"[DEBUG]   -> Detected as BIC: {bic}")
                elif id_type.upper() == 'IBAN':
                    iban = id_text
                    if self.debug:
                        print(f"[DEBUG]   -> Detected as IBAN: {iban}")
                else:
                    if looks_like_bic(id_text):
                        bic = id_text
                        if self.debug:
                            print(f"[DEBUG]   -> Auto-detected as BIC: {bic}")
                    elif looks_like_iban(id_text):
                        iban = id_text
                        if self.debug:
                            print(f"[DEBUG]   -> Auto-detected as IBAN: {iban}")
                    else:
                        account_num = id_text
                        if self.debug:
                            print(f"[DEBUG]   -> Stored as account: {account_num}")
            else:
                id_text = str(id_field)
                if self.debug:
                    print(f"[DEBUG]   ID is plain string: '{id_text}'")
                
                if looks_like_iban(id_text):
                    iban = id_text
                    if self.debug:
                        print(f"[DEBUG]   -> Auto-detected as IBAN: {iban}")
                elif looks_like_bic(id_text):
                    bic = id_text
                    if self.debug:
                        print(f"[DEBUG]   -> Auto-detected as BIC: {bic}")
                else:
                    account_num = id_text
                    if self.debug:
                        print(f"[DEBUG]   -> Stored as account: {account_num}")
        
        # === Extract AdrBankID (NCH) ===
        adr_bank_id = basic.get('AdrBankID')
        if self.debug:
            print(f"[DEBUG]   AdrBankID: {type(adr_bank_id).__name__} = {adr_bank_id}")
        
        if adr_bank_id:
            if isinstance(adr_bank_id, dict):
                nch = adr_bank_id.get('text') or adr_bank_id.get('#text') or ''
            else:
                nch = str(adr_bank_id)
            self.features[f'{prefix}_has_adr_bank_id'] = True
            if self.debug:
                print(f"[DEBUG]   -> NCH value: {nch}")
        
        # === Store extracted values ===
        if bic:
            self.features[f'{prefix}_has_bic'] = True
            self.features[f'{prefix}_bic'] = bic
            fmt, ctry = is_valid_bic(bic)
            self.features[f'{prefix}_bic_valid_format'] = fmt
            self.features[f'{prefix}_bic_valid_country'] = ctry
            if len(bic) >= 6:
                self.features[f'{prefix}_bic_country'] = bic[4:6].upper()
            if self.debug:
                print(f"[DEBUG]   BIC stored: {bic}, country={self.features.get(f'{prefix}_bic_country')}")
        
        if iban:
            self.features[f'{prefix}_has_iban'] = True
            self.features[f'{prefix}_iban'] = iban
            fmt, cksum = is_valid_iban(iban)
            self.features[f'{prefix}_iban_valid_format'] = fmt
            self.features[f'{prefix}_iban_checksum_valid'] = cksum
            if len(iban) >= 2:
                self.features[f'{prefix}_iban_country'] = iban[:2].upper()
            if self.debug:
                print(f"[DEBUG]   IBAN stored: {iban}, country={self.features.get(f'{prefix}_iban_country')}")
        
        if nch:
            self.features[f'{prefix}_has_nch'] = True
            self.features[f'{prefix}_nch'] = nch
            nch_type = looks_like_nch(nch)
            self.features[f'{prefix}_nch_type'] = nch_type
            if self.debug:
                print(f"[DEBUG]   NCH stored: {nch}, type={nch_type}")
        
        if account_num:
            self.features[f'{prefix}_has_account'] = True
            self.features[f'{prefix}_account'] = account_num
        
        # BIC/IBAN match
        bic_country = self.features.get(f'{prefix}_bic_country')
        iban_country = self.features.get(f'{prefix}_iban_country')
        if bic_country and iban_country:
            self.features[f'{prefix}_bic_iban_match'] = (bic_country == iban_country)
            if self.debug:
                print(f"[DEBUG]   BIC/IBAN match: {bic_country} == {iban_country} -> {self.features[f'{prefix}_bic_iban_match']}")
        
        # Country
        country = party_data.get('MailingCountry') or party_data.get('Country') or basic.get('Country') or ''
        if country:
            self.features[f'{prefix}_country'] = country
    
    def _merge_party(self, party_data: Dict, prefix: str):
        if isinstance(party_data, list):
            party_data = party_data[0] if party_data else {}
        
        basic = party_data.get('BasicPartyInf') or party_data.get('BasicPartyInfo') or {}
        if isinstance(basic, list):
            basic = basic[0] if basic else {}
        
        id_field = basic.get('ID')
        if self.debug:
            print(f"[DEBUG]   Merge - ID field: {type(id_field).__name__} = {id_field}")
        
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
                
                if self.debug:
                    print(f"[DEBUG]   Merged IBAN: {id_text}")
                
                # Update match
                bic_country = self.features.get(f'{prefix}_bic_country')
                iban_country = self.features.get(f'{prefix}_iban_country')
                if bic_country and iban_country:
                    self.features[f'{prefix}_bic_iban_match'] = (bic_country == iban_country)

class RuleEngine:
    def __init__(self, features: Dict, debug: bool = False):
        self.f = features
        self.codes = set()
        self.debug = debug
    
    def predict(self) -> Set[str]:
        self.codes = set()
        for prefix in PARTY_PREFIXES:
            if self.f.get(f'{prefix}_present'):
                self._check_8xxx(prefix)
        return self.codes
    
    def _get(self, prefix: str, key: str, default=None):
        return self.f.get(f'{prefix}_{key}', default)
    
    def _emit(self, code: str):
        self.codes.add(code)
    
    def _check_8xxx(self, p: str):
        if self.debug:
            print(f"\n[RULES] Checking {p}:")
            print(f"[RULES]   has_bic={self._get(p, 'has_bic')}, has_iban={self._get(p, 'has_iban')}, has_nch={self._get(p, 'has_nch')}")
        
        # 8001: Invalid BIC
        if self._get(p, 'has_bic'):
            if not self._get(p, 'bic_valid_format', True):
                if self.debug: print(f"[RULES]   8001: bic_valid_format=False")
                self._emit('8001')
            elif not self._get(p, 'bic_valid_country', True):
                if self.debug: print(f"[RULES]   8001: bic_valid_country=False")
                self._emit('8001')
        
        # 8022: BIC/IBAN mismatch
        if self._get(p, 'has_bic') and self._get(p, 'has_iban'):
            match = self._get(p, 'bic_iban_match')
            if self.debug: print(f"[RULES]   bic_iban_match={match}")
            if match == False:
                if self.debug: print(f"[RULES]   -> EMIT 8022")
                self._emit('8022')
        
        # 8022/8026: BIC vs NCH type
        if self._get(p, 'has_bic') and self._get(p, 'has_nch'):
            bic_country = self._get(p, 'bic_country', '')
            nch_type = self._get(p, 'nch_type', '')
            if self.debug: print(f"[RULES]   bic_country={bic_country}, nch_type={nch_type}")
            
            if nch_type == 'FEDABA' and bic_country and bic_country != 'US':
                if self.debug: print(f"[RULES]   -> EMIT 8022, 8026 (FEDABA + non-US BIC)")
                self._emit('8022')
                self._emit('8026')
            elif nch_type == 'BSB' and bic_country and bic_country != 'AU':
                if self.debug: print(f"[RULES]   -> EMIT 8022, 8026 (BSB + non-AU BIC: {bic_country})")
                self._emit('8022')
                self._emit('8026')
        
        # 8894: Invalid IBAN
        if self._get(p, 'has_iban'):
            if self.debug:
                print(f"[RULES]   iban_valid_format={self._get(p, 'iban_valid_format', True)}")
                print(f"[RULES]   iban_checksum_valid={self._get(p, 'iban_checksum_valid', True)}")
            
            if not self._get(p, 'iban_valid_format', True):
                if self.debug: print(f"[RULES]   -> EMIT 8894 (format)")
                self._emit('8894')
            elif not self._get(p, 'iban_checksum_valid', True):
                if self.debug: print(f"[RULES]   -> EMIT 8894 (checksum)")
                self._emit('8894')
            else:
                # IBAN bank code vs NCH
                iban = self._get(p, 'iban', '')
                nch = self._get(p, 'nch', '')
                if self.debug: print(f"[RULES]   iban={iban}, nch={nch}")
                
                if iban and nch and len(iban) >= 7 and len(nch) == 3:
                    iban_bank = iban[4:7]
                    if self.debug: print(f"[RULES]   iban_bank_code={iban_bank}, nch={nch}")
                    if iban_bank != nch:
                        if self.debug: print(f"[RULES]   -> EMIT 8894 (IBAN bank != NCH)")
                        self._emit('8894')

def extract_actual_codes(response: Dict, include_dir: bool = False) -> Set[str]:
    codes = set()
    
    audit = None
    for path in [['AuditTrail'], ['IFML', 'File', 'Message', 'AuditTrail'], ['IFML', 'File', 'AuditTrail']]:
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

def process_payment(pid: str, request: Dict, response: Dict, debug: bool = False) -> VerificationResult:
    extractor = FeatureExtractor(debug=debug)
    features = extractor.extract(request)
    
    if debug:
        print(f"\n[DEBUG] === FINAL FEATURES ===")
        for k, v in sorted(features.items()):
            if v is not None and v != False and v != 0 and v != '':
                print(f"  {k}: {v}")
    
    engine = RuleEngine(features, debug=debug)
    predicted = engine.predict()
    
    actual = extract_actual_codes(response, INCLUDE_DIRECTORY)
    all_actual = extract_actual_codes(response, True)
    
    result = VerificationResult(trn_id=pid, predicted=predicted, actual=actual)
    result.directory_codes = all_actual - actual
    result.features = features
    
    if debug:
        print(f"\n[RESULT] Predicted: {sorted(predicted)}")
        print(f"[RESULT] Actual: {sorted(actual)}")
        print(f"[RESULT] FN (missing): {sorted(actual - predicted)}")
        print(f"[RESULT] FP (extra): {sorted(predicted - actual)}")
    
    return result

def process_file(filepath: Path, verbose: bool = False, debug: bool = False) -> List[VerificationResult]:
    results = []
    with open(filepath) as f:
        data = json.load(f)
    
    if debug:
        print(f"\n[FILE] Loading {filepath}")
        print(f"[FILE] Top-level type: {type(data).__name__}")
        if isinstance(data, dict):
            print(f"[FILE] Top-level keys: {list(data.keys())}")
    
    if isinstance(data, dict):
        if 'Request' in data and 'Response' in data:
            pid = get_transaction_id(data)
            if debug:
                print(f"[FILE] Single payment: {pid}")
            results.append(process_payment(pid, data, data.get('Response', {}), debug))
        else:
            for key, val in data.items():
                if isinstance(val, dict) and 'Request' in val and 'Response' in val:
                    if debug:
                        print(f"[FILE] Payment by ID: {key}")
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
                print(f"  FN (missing): {sorted(r.actual - r.predicted)}")
            if r.predicted - r.actual:
                print(f"  FP (extra): {sorted(r.predicted - r.actual)}")
    
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
