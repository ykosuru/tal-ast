#!/usr/bin/env python3
"""
ACE Pelican IFML Verification Tool
Conservative rules - avoids false positives by only checking static validation.
Directory-dependent codes (derived value checks) are excluded from comparison.
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

# Codes we CANNOT predict without directory/reference data
# These are excluded from pass/fail comparison
DIRECTORY_DEPENDENT_CODES = {
    # 8xxx validation that requires derived values or ACE-specific rules
    '8001',   # BIC validation - ACE uses different rules than standard
    '8004',   # IBAN required (needs country rules)
    '8022',   # NCH inconsistency (needs derived NCH from BIC/IBAN)
    '8026',   # NCH inconsistency (needs derived NCH)
    '8036',   # Directory lookup
    '8464', '8465', '8472',  # Directory-based
    '8852',   # ACE-specific validation
    '8894',   # Often fired on derived NCH mismatch, not just IBAN format
    '8896',   # ACE-specific validation
    # 9xxx repair codes (9019 removed - now predicted)
    '9004', '9005', '9007', '9008', '9013', '9017', '9018', '9024',
    '9476', '9477', '9479', '9480', '9961', '9970', '9985', '9999',
}

# Full ISO 3166-1 alpha-2 country codes
VALID_COUNTRIES = {
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

# =============================================================================
# VALIDATION
# =============================================================================

def is_valid_bic(s: str) -> Tuple[bool, bool]:
    """Returns (format_valid, country_valid)"""
    if not s or len(s) not in (8, 11):
        return False, False
    s = s.upper()
    if not re.match(r'^[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}([A-Z0-9]{3})?$', s):
        return False, False
    return True, s[4:6] in VALID_COUNTRIES

def is_valid_iban(s: str) -> Tuple[bool, bool]:
    """Returns (format_valid, checksum_valid)"""
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
    return bool(re.match(r'^[A-Z]{2}[0-9]{2}[A-Z0-9]+$', s.upper().replace(' ', '')))

# =============================================================================
# FEATURE EXTRACTOR
# =============================================================================

class FeatureExtractor:
    def __init__(self, debug: bool = False):
        self.features = {}
        self.debug = debug
    
    def extract(self, data: Dict) -> Dict:
        self.features = {}
        
        # Find Message
        message = self._find_message(data)
        if not message:
            return self.features
        
        # Find PartyInf (check both Message level and BasicPayment level)
        party_inf = message.get('PartyInf') or message.get('PartyInfo') or {}
        if not party_inf:
            bp = message.get('BasicPayment', {})
            party_inf = bp.get('PartyInf') or bp.get('PartyInfo') or {}
        
        if self.debug and party_inf:
            print(f"[DEBUG] PartyInf keys: {list(party_inf.keys())}")
        
        # Process parties
        processed = set()
        for ifml_name, prefix in PARTY_MAPPING.items():
            party_data = party_inf.get(ifml_name)
            if party_data:
                if prefix in processed:
                    self._merge_party(party_data, prefix)
                else:
                    self._extract_party(party_data, prefix)
                    processed.add(prefix)
        
        for prefix in PARTY_PREFIXES:
            if f'{prefix}_present' not in self.features:
                self.features[f'{prefix}_present'] = False
        
        return self.features
    
    def _find_message(self, data: Dict) -> Optional[Dict]:
        for path in [
            ['Request', 'IFML', 'File', 'Message'],
            ['IFML', 'File', 'Message'],
            ['File', 'Message'],
            ['Message'],
        ]:
            current = data
            for key in path:
                if isinstance(current, dict):
                    current = current.get(key)
                else:
                    current = None
                    break
            if current and isinstance(current, dict):
                return current
        return data if isinstance(data, dict) and ('PartyInf' in data or 'BasicPayment' in data) else None
    
    def _extract_party(self, party_data: Dict, prefix: str):
        if isinstance(party_data, list):
            party_data = party_data[0] if party_data else {}
        
        self.features[f'{prefix}_present'] = True
        
        basic = party_data.get('BasicPartyInf') or party_data.get('BasicPartyInfo') or {}
        if isinstance(basic, list):
            basic = basic[0] if basic else {}
        
        # Extract ID
        id_field = basic.get('ID')
        bic, iban = None, None
        
        if id_field:
            self.features[f'{prefix}_has_id'] = True
            if isinstance(id_field, dict):
                id_type = (id_field.get('Type') or id_field.get('@Type') or '').upper()
                id_text = id_field.get('text') or id_field.get('#text') or ''
                
                if id_type in ('S', 'SWIFT', 'BIC'):
                    bic = id_text
                elif id_type == 'IBAN':
                    iban = id_text
                elif looks_like_bic(id_text):
                    bic = id_text
                elif looks_like_iban(id_text):
                    iban = id_text
            else:
                id_text = str(id_field)
                if looks_like_iban(id_text):
                    iban = id_text
                elif looks_like_bic(id_text):
                    bic = id_text
        
        # Store BIC
        if bic:
            self.features[f'{prefix}_has_bic'] = True
            self.features[f'{prefix}_bic'] = bic
            fmt, ctry = is_valid_bic(bic)
            self.features[f'{prefix}_bic_valid_format'] = fmt
            self.features[f'{prefix}_bic_valid_country'] = ctry
            if len(bic) >= 6:
                self.features[f'{prefix}_bic_country'] = bic[4:6].upper()
        
        # Store IBAN
        if iban:
            self.features[f'{prefix}_has_iban'] = True
            self.features[f'{prefix}_iban'] = iban
            self.features[f'{prefix}_iban_raw'] = iban  # Keep raw for cleaning detection
            # Check if IBAN needs cleaning (has spaces, dashes, etc.)
            cleaned = re.sub(r'[^A-Za-z0-9]', '', iban)
            if cleaned != iban:
                self.features[f'{prefix}_iban_needs_cleaning'] = True
                if self.debug:
                    print(f"[DEBUG] {prefix}: IBAN needs cleaning - raw: '{iban}' -> cleaned: '{cleaned}'")
            fmt, cksum = is_valid_iban(iban)
            self.features[f'{prefix}_iban_valid_format'] = fmt
            self.features[f'{prefix}_iban_checksum_valid'] = cksum
            if len(iban) >= 2:
                self.features[f'{prefix}_iban_country'] = iban[:2].upper()
        
        # BIC/IBAN country match
        bic_country = self.features.get(f'{prefix}_bic_country')
        iban_country = self.features.get(f'{prefix}_iban_country')
        if bic_country and iban_country:
            self.features[f'{prefix}_bic_iban_match'] = (bic_country == iban_country)
    
    def _merge_party(self, party_data: Dict, prefix: str):
        if isinstance(party_data, list):
            party_data = party_data[0] if party_data else {}
        
        basic = party_data.get('BasicPartyInf') or party_data.get('BasicPartyInfo') or {}
        if isinstance(basic, list):
            basic = basic[0] if basic else {}
        
        id_field = basic.get('ID')
        if id_field and not self.features.get(f'{prefix}_has_iban'):
            id_text = id_field.get('text') or id_field.get('#text') or str(id_field) if isinstance(id_field, dict) else str(id_field)
            
            if looks_like_iban(id_text):
                self.features[f'{prefix}_has_iban'] = True
                self.features[f'{prefix}_iban'] = id_text
                self.features[f'{prefix}_iban_raw'] = id_text  # Keep raw for cleaning detection
                # Check if IBAN needs cleaning
                cleaned = re.sub(r'[^A-Za-z0-9]', '', id_text)
                if cleaned != id_text:
                    self.features[f'{prefix}_iban_needs_cleaning'] = True
                    if self.debug:
                        print(f"[DEBUG] {prefix} (merge): IBAN needs cleaning - raw: '{id_text}' -> cleaned: '{cleaned}'")
                fmt, cksum = is_valid_iban(id_text)
                self.features[f'{prefix}_iban_valid_format'] = fmt
                self.features[f'{prefix}_iban_checksum_valid'] = cksum
                if len(id_text) >= 2:
                    self.features[f'{prefix}_iban_country'] = id_text[:2].upper()
                
                bic_country = self.features.get(f'{prefix}_bic_country')
                iban_country = self.features.get(f'{prefix}_iban_country')
                if bic_country and iban_country:
                    self.features[f'{prefix}_bic_iban_match'] = (bic_country == iban_country)

# =============================================================================
# RULE ENGINE - Conservative, static validation only
# =============================================================================

class RuleEngine:
    def __init__(self, features: Dict, debug: bool = False):
        self.f = features
        self.codes = set()
        self.debug = debug
    
    def predict(self) -> Set[str]:
        self.codes = set()
        for prefix in PARTY_PREFIXES:
            if self.f.get(f'{prefix}_present'):
                self._check_static_8xxx(prefix)
                self._check_static_9xxx(prefix)
        return self.codes
    
    def _get(self, prefix: str, key: str, default=None):
        return self.f.get(f'{prefix}_{key}', default)
    
    def _emit(self, code: str):
        self.codes.add(code)
    
    def _check_static_8xxx(self, p: str):
        """Only check statically verifiable conditions - no directory lookups."""
        # Currently limited static checks - most codes require directory data
        pass
    
    def _check_static_9xxx(self, p: str):
        """Check for 9xxx repair codes that are statically detectable."""
        
        # 9019: Party Identifier will be cleaned of non-alphanumeric characters
        # Fires when ID contains spaces, dashes, or special chars
        if self._get(p, 'iban_needs_cleaning'):
            if self.debug:
                print(f"[RULES] {p}: 9019 - IBAN needs cleaning: {self._get(p, 'iban_raw')}")
            self._emit('9019')

# =============================================================================
# RESPONSE PARSING
# =============================================================================

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

# =============================================================================
# MAIN
# =============================================================================

def process_payment(pid: str, request: Dict, response: Dict, debug: bool = False) -> VerificationResult:
    extractor = FeatureExtractor(debug=debug)
    features = extractor.extract(request)
    
    engine = RuleEngine(features, debug=debug)
    predicted = engine.predict()
    
    actual = extract_actual_codes(response, INCLUDE_DIRECTORY)
    all_actual = extract_actual_codes(response, True)
    
    result = VerificationResult(trn_id=pid, predicted=predicted, actual=actual)
    result.directory_codes = all_actual - actual
    result.features = features
    
    if debug:
        print(f"[RESULT] Predicted: {sorted(predicted)}")
        print(f"[RESULT] Actual: {sorted(actual)}")
        print(f"[RESULT] Directory excluded: {sorted(result.directory_codes)}")
    
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

def process_directory(dirpath: Path, verbose: bool = False, debug: bool = False) -> List[VerificationResult]:
    results = []
    for f in sorted(dirpath.glob('*.json')):
        try:
            results.extend(process_file(f, verbose, debug))
        except Exception as e:
            if verbose:
                print(f"Error: {f}: {e}")
    return results

def print_results(results: List[VerificationResult], verbose: bool = False):
    passed = failed = 0
    failure_codes = {}
    
    for r in results:
        if r.passed:
            passed += 1
            if verbose or r.directory_codes:
                dir_info = f" [Dir: {sorted(r.directory_codes)}]" if r.directory_codes else ""
                print(f"{r.trn_id} PASS{dir_info}")
        else:
            failed += 1
            fp = r.predicted - r.actual
            fn = r.actual - r.predicted
            
            # Track failure codes
            for code in fp:
                failure_codes[code] = failure_codes.get(code, {'fp': 0, 'fn': 0})
                failure_codes[code]['fp'] += 1
            for code in fn:
                failure_codes[code] = failure_codes.get(code, {'fp': 0, 'fn': 0})
                failure_codes[code]['fn'] += 1
            
            print(f"\n{r.trn_id} FAIL")
            print(f"  PREDICTED: {sorted(r.predicted)}")
            print(f"  ACTUAL: {sorted(r.actual)}")
            if fn:
                print(f"  FN: {sorted(fn)}")
            if fp:
                print(f"  FP: {sorted(fp)}")
            if r.directory_codes:
                print(f"  [Dir excluded: {sorted(r.directory_codes)}]")
            
            # Extra debug for 9015 and 9019 failures
            if '9015' in fp or '9015' in fn or '9019' in fp or '9019' in fn:
                print(f"  --- DEBUG 9015/9019 ---")
                for prefix in PARTY_PREFIXES:
                    has_iban = r.features.get(f'{prefix}_has_iban')
                    has_bic = r.features.get(f'{prefix}_has_bic')
                    has_nch = r.features.get(f'{prefix}_has_nch')
                    has_id = r.features.get(f'{prefix}_has_id')
                    
                    if has_iban or has_bic or has_nch or has_id:
                        parts = []
                        if has_id:
                            parts.append(f"ID=yes")
                        if has_iban:
                            iban_raw = r.features.get(f'{prefix}_iban_raw', r.features.get(f'{prefix}_iban', '?'))
                            needs_clean = r.features.get(f'{prefix}_iban_needs_cleaning', False)
                            parts.append(f"IBAN='{iban_raw}' clean={needs_clean}")
                        if has_bic:
                            bic = r.features.get(f'{prefix}_bic', '?')
                            parts.append(f"BIC='{bic}'")
                        if has_nch:
                            nch = r.features.get(f'{prefix}_nch', '?')
                            nch_type = r.features.get(f'{prefix}_nch_type', '?')
                            parts.append(f"NCH='{nch}'({nch_type})")
                        print(f"  {prefix}: {', '.join(parts)}")
    
    total = len(results)
    print(f"\n{'='*60}")
    print(f"SUMMARY: {passed}/{total} passed ({100*passed/total:.1f}%), {failed} failed")
    print(f"(Directory-dependent codes excluded: {sorted(DIRECTORY_DEPENDENT_CODES)})")
    
    if failure_codes:
        print(f"\nFAILURE BREAKDOWN:")
        for code in sorted(failure_codes.keys()):
            stats = failure_codes[code]
            print(f"  {code}: FP={stats['fp']}, FN={stats['fn']}")
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
        results = process_directory(path, args.verbose, args.debug)
    
    if results:
        print_results(results, args.verbose)
    else:
        print("No payments found")

if __name__ == '__main__':
    main()
