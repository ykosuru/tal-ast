#!/usr/bin/env python3
"""
BBAN (Basic Bank Account Number) Validation by Country

BBAN is the country-specific part of an IBAN after the country code and check digits.
Example: DE89 3704 0044 0532 0130 00
         ^^   ^^^^^^^^^^^^^^^^^^^
         CC   BBAN (country-specific format)

Each country has different BBAN formats with varying:
- Length
- Structure (bank code, branch code, account number positions)
- Internal check digits (some countries have them)

This module validates BBAN structure and check digits where applicable.
"""

import re
from typing import Tuple, Optional, Dict

# BBAN specifications by country
# Format: (total_length, structure_regex, has_check_digit, check_digit_func)
# Structure uses: b=bank, s=branch/sort, c=check, n=account number, x=national check
BBAN_SPECS: Dict[str, dict] = {
    # Western Europe
    'DE': {  # Germany: 8-digit bank code (Bankleitzahl) + 10-digit account
        'length': 18,
        'structure': r'^(\d{8})(\d{10})$',
        'parts': ['bank_code', 'account'],
        'has_check': False,
    },
    'FR': {  # France: 5 bank + 5 branch + 11 account + 2 check (RIB key)
        'length': 23,
        'structure': r'^(\d{5})(\d{5})([A-Z0-9]{11})(\d{2})$',
        'parts': ['bank_code', 'branch', 'account', 'check'],
        'has_check': True,
        'check_func': 'french_rib_key',
    },
    'ES': {  # Spain: 4 bank + 4 branch + 2 check + 10 account
        'length': 20,
        'structure': r'^(\d{4})(\d{4})(\d{2})(\d{10})$',
        'parts': ['bank_code', 'branch', 'check', 'account'],
        'has_check': True,
        'check_func': 'spanish_check',
    },
    'IT': {  # Italy: 1 check (CIN) + 5 bank (ABI) + 5 branch (CAB) + 12 account
        'length': 23,
        'structure': r'^([A-Z])(\d{5})(\d{5})([A-Z0-9]{12})$',
        'parts': ['check', 'bank_code', 'branch', 'account'],
        'has_check': True,
        'check_func': 'italian_cin',
    },
    'NL': {  # Netherlands: 4 bank + 10 account
        'length': 14,
        'structure': r'^([A-Z]{4})(\d{10})$',
        'parts': ['bank_code', 'account'],
        'has_check': False,
    },
    'BE': {  # Belgium: 3 bank + 7 account + 2 check (mod 97)
        'length': 12,
        'structure': r'^(\d{3})(\d{7})(\d{2})$',
        'parts': ['bank_code', 'account', 'check'],
        'has_check': True,
        'check_func': 'belgian_check',
    },
    'AT': {  # Austria: 5 bank + 11 account
        'length': 16,
        'structure': r'^(\d{5})(\d{11})$',
        'parts': ['bank_code', 'account'],
        'has_check': False,
    },
    'CH': {  # Switzerland: 5 bank + 12 account
        'length': 17,
        'structure': r'^(\d{5})([A-Z0-9]{12})$',
        'parts': ['bank_code', 'account'],
        'has_check': False,
    },
    'LU': {  # Luxembourg: 3 bank + 13 account
        'length': 16,
        'structure': r'^(\d{3})([A-Z0-9]{13})$',
        'parts': ['bank_code', 'account'],
        'has_check': False,
    },
    
    # UK & Ireland
    'GB': {  # UK: 4 bank (BIC) + 6 sort code + 8 account
        'length': 18,
        'structure': r'^([A-Z]{4})(\d{6})(\d{8})$',
        'parts': ['bank_code', 'sort_code', 'account'],
        'has_check': False,  # UK has modulus checking but complex
    },
    'IE': {  # Ireland: 4 bank + 6 branch + 8 account
        'length': 18,
        'structure': r'^([A-Z]{4})(\d{6})(\d{8})$',
        'parts': ['bank_code', 'branch', 'account'],
        'has_check': False,
    },
    
    # Nordic
    'FI': {  # Finland: 3 bank + 11 account (includes check)
        'length': 14,
        'structure': r'^(\d{3})(\d{11})$',
        'parts': ['bank_code', 'account'],
        'has_check': True,
        'check_func': 'finnish_check',
    },
    'SE': {  # Sweden: 3 bank + 17 account
        'length': 20,
        'structure': r'^(\d{3})(\d{17})$',
        'parts': ['bank_code', 'account'],
        'has_check': False,
    },
    'NO': {  # Norway: 4 bank + 6 account + 1 check
        'length': 11,
        'structure': r'^(\d{4})(\d{6})(\d{1})$',
        'parts': ['bank_code', 'account', 'check'],
        'has_check': True,
        'check_func': 'norwegian_check',
    },
    'DK': {  # Denmark: 4 bank + 10 account
        'length': 14,
        'structure': r'^(\d{4})(\d{10})$',
        'parts': ['bank_code', 'account'],
        'has_check': False,
    },
    
    # Eastern Europe
    'PL': {  # Poland: 8 bank/branch + 16 account (with check in position 9)
        'length': 24,
        'structure': r'^(\d{8})(\d{16})$',
        'parts': ['bank_branch', 'account'],
        'has_check': True,
        'check_func': 'polish_check',
    },
    'CZ': {  # Czech: 4 bank + 6 prefix + 10 account
        'length': 20,
        'structure': r'^(\d{4})(\d{6})(\d{10})$',
        'parts': ['bank_code', 'prefix', 'account'],
        'has_check': True,
        'check_func': 'czech_check',
    },
    'SK': {  # Slovakia: Same as Czech
        'length': 20,
        'structure': r'^(\d{4})(\d{6})(\d{10})$',
        'parts': ['bank_code', 'prefix', 'account'],
        'has_check': True,
        'check_func': 'czech_check',
    },
    'HU': {  # Hungary: 3 bank + 4 branch + 1 check + 15 account + 1 check
        'length': 24,
        'structure': r'^(\d{3})(\d{4})(\d{1})(\d{15})(\d{1})$',
        'parts': ['bank_code', 'branch', 'check1', 'account', 'check2'],
        'has_check': True,
        'check_func': 'hungarian_check',
    },
    'RO': {  # Romania: 4 bank + 16 account
        'length': 20,
        'structure': r'^([A-Z]{4})([A-Z0-9]{16})$',
        'parts': ['bank_code', 'account'],
        'has_check': False,
    },
    'BG': {  # Bulgaria: 4 bank + 4 branch + 2 account type + 8 account
        'length': 18,
        'structure': r'^([A-Z]{4})(\d{4})(\d{2})([A-Z0-9]{8})$',
        'parts': ['bank_code', 'branch', 'account_type', 'account'],
        'has_check': False,
    },
    'HR': {  # Croatia: 7 bank + 10 account
        'length': 17,
        'structure': r'^(\d{7})(\d{10})$',
        'parts': ['bank_code', 'account'],
        'has_check': False,
    },
    'SI': {  # Slovenia: 5 bank + 8 account + 2 check
        'length': 15,
        'structure': r'^(\d{5})(\d{8})(\d{2})$',
        'parts': ['bank_code', 'account', 'check'],
        'has_check': True,
        'check_func': 'slovenian_check',
    },
    'RS': {  # Serbia: 3 bank + 13 account + 2 check
        'length': 18,
        'structure': r'^(\d{3})(\d{13})(\d{2})$',
        'parts': ['bank_code', 'account', 'check'],
        'has_check': True,
        'check_func': 'mod97_check',
    },
    
    # Baltic
    'EE': {  # Estonia: 2 bank + 14 account (with check)
        'length': 16,
        'structure': r'^(\d{2})(\d{14})$',
        'parts': ['bank_code', 'account'],
        'has_check': True,
        'check_func': 'estonian_check',
    },
    'LV': {  # Latvia: 4 bank + 13 account
        'length': 17,
        'structure': r'^([A-Z]{4})([A-Z0-9]{13})$',
        'parts': ['bank_code', 'account'],
        'has_check': False,
    },
    'LT': {  # Lithuania: 5 bank + 11 account
        'length': 16,
        'structure': r'^(\d{5})(\d{11})$',
        'parts': ['bank_code', 'account'],
        'has_check': False,
    },
    
    # Southern Europe
    'PT': {  # Portugal: 4 bank + 4 branch + 11 account + 2 check
        'length': 21,
        'structure': r'^(\d{4})(\d{4})(\d{11})(\d{2})$',
        'parts': ['bank_code', 'branch', 'account', 'check'],
        'has_check': True,
        'check_func': 'portuguese_check',
    },
    'GR': {  # Greece: 3 bank + 4 branch + 16 account
        'length': 23,
        'structure': r'^(\d{3})(\d{4})([A-Z0-9]{16})$',
        'parts': ['bank_code', 'branch', 'account'],
        'has_check': False,
    },
    'CY': {  # Cyprus: 3 bank + 5 branch + 16 account
        'length': 24,
        'structure': r'^(\d{3})(\d{5})([A-Z0-9]{16})$',
        'parts': ['bank_code', 'branch', 'account'],
        'has_check': False,
    },
    'MT': {  # Malta: 4 bank + 5 branch + 18 account
        'length': 27,
        'structure': r'^([A-Z]{4})(\d{5})([A-Z0-9]{18})$',
        'parts': ['bank_code', 'branch', 'account'],
        'has_check': False,
    },
    
    # Middle East
    'SA': {  # Saudi Arabia: 2 bank + 18 account
        'length': 20,
        'structure': r'^(\d{2})([A-Z0-9]{18})$',
        'parts': ['bank_code', 'account'],
        'has_check': False,
    },
    'AE': {  # UAE: 3 bank + 16 account
        'length': 19,
        'structure': r'^(\d{3})(\d{16})$',
        'parts': ['bank_code', 'account'],
        'has_check': False,
    },
    'IL': {  # Israel: 3 bank + 3 branch + 13 account
        'length': 19,
        'structure': r'^(\d{3})(\d{3})(\d{13})$',
        'parts': ['bank_code', 'branch', 'account'],
        'has_check': False,
    },
    'TR': {  # Turkey: 5 bank + 1 reserved + 16 account
        'length': 22,
        'structure': r'^(\d{5})(0)([A-Z0-9]{16})$',
        'parts': ['bank_code', 'reserved', 'account'],
        'has_check': False,
    },
    'QA': {  # Qatar: 4 bank + 21 account
        'length': 25,
        'structure': r'^([A-Z]{4})([A-Z0-9]{21})$',
        'parts': ['bank_code', 'account'],
        'has_check': False,
    },
    'KW': {  # Kuwait: 4 bank + 22 account
        'length': 26,
        'structure': r'^([A-Z]{4})([A-Z0-9]{22})$',
        'parts': ['bank_code', 'account'],
        'has_check': False,
    },
    'BH': {  # Bahrain: 4 bank + 14 account
        'length': 18,
        'structure': r'^([A-Z]{4})([A-Z0-9]{14})$',
        'parts': ['bank_code', 'account'],
        'has_check': False,
    },
    
    # Other
    'MC': {  # Monaco: Same as France
        'length': 23,
        'structure': r'^(\d{5})(\d{5})([A-Z0-9]{11})(\d{2})$',
        'parts': ['bank_code', 'branch', 'account', 'check'],
        'has_check': True,
        'check_func': 'french_rib_key',
    },
    'SM': {  # San Marino: Same as Italy
        'length': 23,
        'structure': r'^([A-Z])(\d{5})(\d{5})([A-Z0-9]{12})$',
        'parts': ['check', 'bank_code', 'branch', 'account'],
        'has_check': True,
        'check_func': 'italian_cin',
    },
    'LI': {  # Liechtenstein: Same as Switzerland
        'length': 17,
        'structure': r'^(\d{5})([A-Z0-9]{12})$',
        'parts': ['bank_code', 'account'],
        'has_check': False,
    },
}


# =============================================================================
# CHECK DIGIT CALCULATION FUNCTIONS
# =============================================================================

def _letter_to_number(letter: str) -> int:
    """Convert letter to number for check calculations (A=10, B=11, etc.)"""
    if letter.isdigit():
        return int(letter)
    return ord(letter.upper()) - ord('A') + 10


def french_rib_key(bank: str, branch: str, account: str, stated_check: str) -> bool:
    """
    Validate French RIB key (clé RIB).
    Key = 97 - ((bank * 89 + branch * 15 + account * 3) mod 97)
    Account letters are converted: A-I→1-9, J-R→1-9, S-Z→2-9
    """
    try:
        # Convert letters in account to numbers
        def convert_account(acc: str) -> str:
            result = ''
            for c in acc.upper():
                if c.isdigit():
                    result += c
                elif 'A' <= c <= 'I':
                    result += str(ord(c) - ord('A') + 1)
                elif 'J' <= c <= 'R':
                    result += str(ord(c) - ord('J') + 1)
                elif 'S' <= c <= 'Z':
                    result += str(ord(c) - ord('S') + 2)
                else:
                    result += '0'
            return result
        
        account_num = convert_account(account)
        bank_num = int(bank)
        branch_num = int(branch)
        acc_num = int(account_num)
        
        calculated = 97 - ((bank_num * 89 + branch_num * 15 + acc_num * 3) % 97)
        return int(stated_check) == calculated
    except (ValueError, TypeError):
        return False


def spanish_check(bank: str, branch: str, stated_check: str, account: str) -> bool:
    """
    Validate Spanish BBAN check digits.
    First digit validates bank+branch, second validates account.
    Uses weights [1,2,4,8,5,10,9,7,3,6] and mod 11.
    """
    try:
        weights = [1, 2, 4, 8, 5, 10, 9, 7, 3, 6]
        
        # First check digit: validates "00" + bank + branch
        digits1 = "00" + bank + branch
        sum1 = sum(int(d) * w for d, w in zip(digits1, weights))
        check1 = 11 - (sum1 % 11)
        if check1 == 11:
            check1 = 0
        elif check1 == 10:
            check1 = 1
        
        # Second check digit: validates account
        sum2 = sum(int(d) * w for d, w in zip(account, weights))
        check2 = 11 - (sum2 % 11)
        if check2 == 11:
            check2 = 0
        elif check2 == 10:
            check2 = 1
        
        expected = f"{check1}{check2}"
        return stated_check == expected
    except (ValueError, TypeError):
        return False


def italian_cin(stated_cin: str, bank: str, branch: str, account: str) -> bool:
    """
    Validate Italian CIN (Control Internal Number).
    Complex algorithm with different values for odd/even positions.
    """
    try:
        # Odd position values (1st, 3rd, 5th, etc. - 1-indexed)
        odd_values = {
            '0': 1, '1': 0, '2': 5, '3': 7, '4': 9, '5': 13, '6': 15, '7': 17, '8': 19, '9': 21,
            'A': 1, 'B': 0, 'C': 5, 'D': 7, 'E': 9, 'F': 13, 'G': 15, 'H': 17, 'I': 19, 'J': 21,
            'K': 2, 'L': 4, 'M': 18, 'N': 20, 'O': 11, 'P': 3, 'Q': 6, 'R': 8, 'S': 12, 'T': 14,
            'U': 16, 'V': 10, 'W': 22, 'X': 25, 'Y': 24, 'Z': 23,
        }
        # Even position values (2nd, 4th, 6th, etc.)
        even_values = {
            '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
            'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9,
            'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19,
            'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25,
        }
        cin_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        
        # Combine bank + branch + account
        code = (bank + branch + account).upper()
        
        total = 0
        for i, char in enumerate(code):
            if (i + 1) % 2 == 1:  # Odd position (1-indexed)
                total += odd_values.get(char, 0)
            else:  # Even position
                total += even_values.get(char, 0)
        
        calculated_cin = cin_letters[total % 26]
        return stated_cin.upper() == calculated_cin
    except (ValueError, TypeError, KeyError):
        return False


def belgian_check(bank: str, account: str, stated_check: str) -> bool:
    """
    Validate Belgian BBAN check digits (mod 97).
    Check = bank + account mod 97, with 97 if result is 0.
    """
    try:
        num = int(bank + account)
        calculated = num % 97
        if calculated == 0:
            calculated = 97
        return int(stated_check) == calculated
    except (ValueError, TypeError):
        return False


def finnish_check(bank: str, account: str) -> bool:
    """
    Validate Finnish account number check digit (last digit of account).
    Uses Luhn-like algorithm with weights [2,1,2,1,2,1,2,1...] from right.
    """
    try:
        # Full number is bank + account, check digit is last
        full = bank + account
        if len(full) < 2:
            return False
        
        digits = full[:-1]
        stated_check = int(full[-1])
        
        # Finnish uses weights 2,1,2,1... from right to left
        weights = [2, 1] * ((len(digits) + 1) // 2)
        weights = weights[:len(digits)]
        weights.reverse()
        
        total = 0
        for d, w in zip(digits, weights):
            product = int(d) * w
            total += product if product < 10 else product - 9
        
        calculated = (10 - (total % 10)) % 10
        return stated_check == calculated
    except (ValueError, TypeError):
        return False


def norwegian_check(bank: str, account: str, stated_check: str) -> bool:
    """
    Validate Norwegian account check digit.
    Uses MOD 11 with weights [5,4,3,2,7,6,5,4,3,2] on bank+account.
    """
    try:
        digits = bank + account
        weights = [5, 4, 3, 2, 7, 6, 5, 4, 3, 2]
        
        total = sum(int(d) * w for d, w in zip(digits, weights))
        remainder = total % 11
        
        if remainder == 0:
            calculated = 0
        elif remainder == 1:
            return False  # Invalid account number
        else:
            calculated = 11 - remainder
        
        return int(stated_check) == calculated
    except (ValueError, TypeError):
        return False


def polish_check(bank_branch: str, account: str) -> bool:
    """
    Validate Polish BBAN.
    First 8 digits (bank+branch) include check digit at position 8.
    Uses weights [3,9,7,1,3,9,7] on first 7 digits, check = 10 - (sum mod 10).
    """
    try:
        if len(bank_branch) != 8:
            return False
        
        digits = bank_branch[:7]
        stated_check = int(bank_branch[7])
        weights = [3, 9, 7, 1, 3, 9, 7]
        
        total = sum(int(d) * w for d, w in zip(digits, weights))
        calculated = (10 - (total % 10)) % 10
        
        return stated_check == calculated
    except (ValueError, TypeError):
        return False


def czech_check(bank: str, prefix: str, account: str) -> bool:
    """
    Validate Czech/Slovak BBAN check digits.
    Prefix and account each have their own check digit (last digit).
    Uses weights [6,3,7,9,10,5,8,4,2,1].
    """
    try:
        weights = [6, 3, 7, 9, 10, 5, 8, 4, 2, 1]
        
        # Check prefix (6 digits)
        if len(prefix) == 6:
            prefix_weights = weights[4:]  # Last 6 weights
            total = sum(int(d) * w for d, w in zip(prefix, prefix_weights))
            if total % 11 != 0:
                return False
        
        # Check account (10 digits)
        if len(account) == 10:
            total = sum(int(d) * w for d, w in zip(account, weights))
            if total % 11 != 0:
                return False
        
        return True
    except (ValueError, TypeError):
        return False


def hungarian_check(bank: str, branch: str, check1: str, account: str, check2: str) -> bool:
    """
    Validate Hungarian BBAN check digits.
    Uses weights [9,7,3,1] repeated.
    """
    try:
        weights = [9, 7, 3, 1]
        
        # First part: bank + branch (7 digits)
        part1 = bank + branch
        total1 = sum(int(d) * weights[i % 4] for i, d in enumerate(part1))
        calc1 = (10 - (total1 % 10)) % 10
        if int(check1) != calc1:
            return False
        
        # Second part: account (15 digits)
        total2 = sum(int(d) * weights[i % 4] for i, d in enumerate(account))
        calc2 = (10 - (total2 % 10)) % 10
        if int(check2) != calc2:
            return False
        
        return True
    except (ValueError, TypeError):
        return False


def slovenian_check(bank: str, account: str, stated_check: str) -> bool:
    """Validate Slovenian BBAN (mod 97)."""
    try:
        num = int(bank + account)
        calculated = 98 - (num * 100) % 97
        return int(stated_check) == calculated
    except (ValueError, TypeError):
        return False


def estonian_check(bank: str, account: str) -> bool:
    """
    Validate Estonian account check digit (last digit).
    Uses weights [7,3,1] repeated.
    """
    try:
        full = bank + account
        digits = full[:-1]
        stated_check = int(full[-1])
        
        weights = [7, 3, 1]
        total = sum(int(d) * weights[i % 3] for i, d in enumerate(digits))
        calculated = (10 - (total % 10)) % 10
        
        return stated_check == calculated
    except (ValueError, TypeError):
        return False


def portuguese_check(bank: str, branch: str, account: str, stated_check: str) -> bool:
    """Validate Portuguese BBAN (mod 97)."""
    try:
        num = int(bank + branch + account)
        calculated = 98 - (num * 100) % 97
        return int(stated_check) == calculated
    except (ValueError, TypeError):
        return False


def mod97_check(bank: str, account: str, stated_check: str) -> bool:
    """Generic mod 97 check (used by multiple countries)."""
    try:
        num = int(bank + account)
        calculated = 98 - (num * 100) % 97
        return int(stated_check) == calculated
    except (ValueError, TypeError):
        return False


# =============================================================================
# MAIN VALIDATION FUNCTION
# =============================================================================

def extract_bban(iban: str) -> Tuple[str, str]:
    """
    Extract country code and BBAN from an IBAN.
    Returns (country_code, bban)
    """
    if not iban:
        return '', ''
    
    cleaned = iban.upper().replace(' ', '').replace('-', '')
    if len(cleaned) < 5:
        return '', ''
    
    country = cleaned[:2]
    bban = cleaned[4:]  # Skip country (2) + check digits (2)
    
    return country, bban


def validate_bban(iban: str) -> Tuple[bool, bool, Optional[str]]:
    """
    Validate the BBAN portion of an IBAN.
    
    Args:
        iban: Full IBAN string
        
    Returns:
        Tuple of:
        - structure_valid: Does BBAN match expected structure for country?
        - check_valid: Do internal check digits pass? (True if no check required)
        - error_detail: Description of error if invalid, None if valid
    """
    country, bban = extract_bban(iban)
    
    if not country or not bban:
        return False, False, "Could not extract BBAN from IBAN"
    
    # Get spec for this country
    spec = BBAN_SPECS.get(country)
    if not spec:
        # Unknown country - can't validate BBAN structure
        return True, True, None  # Assume valid if unknown
    
    # Check length
    if len(bban) != spec['length']:
        return False, False, f"BBAN length {len(bban)} != expected {spec['length']} for {country}"
    
    # Check structure
    match = re.match(spec['structure'], bban)
    if not match:
        return False, False, f"BBAN structure doesn't match {country} format"
    
    # Structure is valid
    structure_valid = True
    
    # Check internal check digits if applicable
    if not spec.get('has_check', False):
        return True, True, None  # No check digit to validate
    
    # Get check function
    check_func_name = spec.get('check_func')
    if not check_func_name:
        return True, True, None
    
    # Extract parts from match
    parts = list(match.groups())
    part_names = spec['parts']
    
    # Call appropriate check function
    check_valid = False
    try:
        if check_func_name == 'french_rib_key':
            check_valid = french_rib_key(*parts)
        elif check_func_name == 'spanish_check':
            check_valid = spanish_check(*parts)
        elif check_func_name == 'italian_cin':
            check_valid = italian_cin(*parts)
        elif check_func_name == 'belgian_check':
            check_valid = belgian_check(*parts)
        elif check_func_name == 'finnish_check':
            check_valid = finnish_check(*parts)
        elif check_func_name == 'norwegian_check':
            check_valid = norwegian_check(*parts)
        elif check_func_name == 'polish_check':
            check_valid = polish_check(*parts)
        elif check_func_name == 'czech_check':
            check_valid = czech_check(*parts)
        elif check_func_name == 'hungarian_check':
            check_valid = hungarian_check(*parts)
        elif check_func_name == 'slovenian_check':
            check_valid = slovenian_check(*parts)
        elif check_func_name == 'estonian_check':
            check_valid = estonian_check(*parts)
        elif check_func_name == 'portuguese_check':
            check_valid = portuguese_check(*parts)
        elif check_func_name == 'mod97_check':
            check_valid = mod97_check(*parts)
        else:
            # Unknown check function, assume valid
            check_valid = True
    except Exception as e:
        return True, False, f"Check digit calculation error: {e}"
    
    if not check_valid:
        return True, False, f"BBAN internal check digit invalid for {country}"
    
    return True, True, None


def validate_iban_complete(iban: str) -> Tuple[bool, bool, bool, Optional[str]]:
    """
    Complete IBAN validation including BBAN structure and check digits.
    
    Returns:
        Tuple of:
        - iban_format_valid: Basic IBAN format OK (length, country)
        - iban_checksum_valid: IBAN mod-97 check passes
        - bban_valid: BBAN structure and internal check digits OK
        - error_detail: Description of first error found, or None
    """
    if not iban:
        return False, False, False, "Empty IBAN"
    
    cleaned = iban.upper().replace(' ', '').replace('-', '')
    
    # Basic format check
    if len(cleaned) < 5:
        return False, False, False, "IBAN too short"
    
    country = cleaned[:2]
    if not country.isalpha():
        return False, False, False, "Invalid country code"
    
    # Check IBAN length for country
    from ifml_parser import IBAN_LENGTHS
    expected_len = IBAN_LENGTHS.get(country)
    if expected_len and len(cleaned) != expected_len:
        return False, False, False, f"IBAN length {len(cleaned)} != expected {expected_len}"
    
    format_valid = True
    
    # IBAN mod-97 checksum
    try:
        rearranged = cleaned[4:] + cleaned[:4]
        numeric = ''
        for char in rearranged:
            if char.isdigit():
                numeric += char
            elif char.isalpha():
                numeric += str(ord(char) - ord('A') + 10)
            else:
                return True, False, False, f"Invalid character in IBAN: {char}"
        
        checksum_valid = int(numeric) % 97 == 1
    except (ValueError, OverflowError):
        checksum_valid = False
    
    if not checksum_valid:
        return True, False, False, "IBAN checksum (mod-97) invalid"
    
    # BBAN validation
    bban_struct_valid, bban_check_valid, bban_error = validate_bban(iban)
    
    if not bban_struct_valid:
        return True, True, False, bban_error
    
    if not bban_check_valid:
        return True, True, False, bban_error
    
    return True, True, True, None


# =============================================================================
# TEST
# =============================================================================

if __name__ == '__main__':
    # Test cases
    test_ibans = [
        # Valid IBANs
        ("DE89370400440532013000", "Germany - valid"),
        ("FR7630006000011234567890189", "France - valid"),
        ("ES9121000418450200051332", "Spain - valid"),
        ("GB82WEST12345698765432", "UK - valid"),
        ("NL91ABNA0417164300", "Netherlands - valid"),
        ("BE68539007547034", "Belgium - valid"),
        ("PL61109010140000071219812874", "Poland - valid"),
        
        # Invalid - wrong checksum
        ("DE89370400440532013001", "Germany - wrong IBAN checksum"),
        
        # Invalid - wrong BBAN structure
        ("DE8937040044053201300", "Germany - BBAN too short"),
        
        # Invalid - wrong BBAN check digit (if we can detect)
        ("ES9121000418450200051333", "Spain - wrong BBAN check"),
    ]
    
    print("BBAN Validation Tests")
    print("=" * 70)
    
    for iban, desc in test_ibans:
        fmt_ok, cksum_ok, bban_ok, error = validate_iban_complete(iban)
        status = "✅" if (fmt_ok and cksum_ok and bban_ok) else "❌"
        print(f"{status} {desc}")
        print(f"   IBAN: {iban}")
        print(f"   Format: {fmt_ok}, Checksum: {cksum_ok}, BBAN: {bban_ok}")
        if error:
            print(f"   Error: {error}")
        print()
