#!/usr/bin/env python3
"""
Directory Validator for ACE Error Prediction

Validates BIC, FEDABA, and NCH codes against actual directories.
This improves precision by checking if codes actually exist, not just format.

Usage:
    validator = DirectoryValidator()
    validator.load_fedaba('fedach_directory.csv')
    validator.load_bic('swift_codes.csv')
    
    exists, info = validator.validate_fedaba('021000021')
    exists, info = validator.validate_bic('DEUTDEFF')
"""

import csv
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Set
from dataclasses import dataclass


@dataclass
class BICEntry:
    bic: str
    bank_name: str
    country: str
    city: str = ""
    status: str = "active"


@dataclass
class FedABAEntry:
    routing_number: str
    bank_name: str
    city: str
    state: str
    status: str = "active"


class DirectoryValidator:
    """
    Validates bank codes against actual directories.
    
    Supports:
    - BIC/SWIFT codes
    - US FEDABA routing numbers
    - UK Sort Codes
    - German BLZ
    - Australian BSB
    """
    
    def __init__(self):
        # BIC directory: key = 8-char BIC (without branch)
        self.bic_directory: Dict[str, BICEntry] = {}
        
        # FEDABA directory: key = 9-digit routing number
        self.fedaba_directory: Dict[str, FedABAEntry] = {}
        
        # UK Sort Codes: key = 6-digit sort code
        self.uk_sort_codes: Set[str] = set()
        
        # German BLZ: key = 8-digit code
        self.german_blz: Set[str] = set()
        
        # Australian BSB: key = 6-digit code
        self.australian_bsb: Set[str] = set()
        
        # Track what's loaded
        self.loaded = {
            'bic': False,
            'fedaba': False,
            'uk_sort': False,
            'german_blz': False,
            'australian_bsb': False
        }
    
    # =========================================================================
    # LOADING METHODS
    # =========================================================================
    
    def load_fedaba(self, filepath: str, delimiter: str = ',') -> int:
        """
        Load FedACH directory (US routing numbers).
        
        Expected CSV format:
        routing_number,bank_name,city,state,status
        021000021,JPMORGAN CHASE,NEW YORK,NY,active
        
        The Federal Reserve provides this file at:
        https://www.frbservices.org/EPaymentsDirectory/download.html
        
        Returns number of entries loaded.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"FEDABA file not found: {filepath}")
        
        count = 0
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                routing = row.get('routing_number', '').strip()
                if len(routing) == 9 and routing.isdigit():
                    self.fedaba_directory[routing] = FedABAEntry(
                        routing_number=routing,
                        bank_name=row.get('bank_name', ''),
                        city=row.get('city', ''),
                        state=row.get('state', ''),
                        status=row.get('status', 'active')
                    )
                    count += 1
        
        self.loaded['fedaba'] = True
        print(f"Loaded {count} FEDABA routing numbers")
        return count
    
    def load_bic(self, filepath: str, delimiter: str = ',') -> int:
        """
        Load BIC/SWIFT directory from generic CSV.
        
        Expected CSV format:
        bic,bank_name,country,city,status
        DEUTDEFF,DEUTSCHE BANK AG,DE,FRANKFURT,active
        
        Note: BIC should be 8 or 11 chars. We store 8-char version as key.
        
        Returns number of entries loaded.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"BIC file not found: {filepath}")
        
        count = 0
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                bic = row.get('bic', '').upper().strip()
                if len(bic) >= 8:
                    base_bic = bic[:8]  # Store without branch code
                    self.bic_directory[base_bic] = BICEntry(
                        bic=base_bic,
                        bank_name=row.get('bank_name', ''),
                        country=row.get('country', bic[4:6]),
                        city=row.get('city', ''),
                        status=row.get('status', 'active')
                    )
                    count += 1
        
        self.loaded['bic'] = True
        print(f"Loaded {count} BIC codes")
        return count
    
    def load_bic_from_opensanctions(self, filepath: str) -> int:
        """
        Load BIC directory from OpenSanctions dataset.
        
        Download from: https://www.opensanctions.org/datasets/iso9362_bic/
        
        OpenSanctions JSON format (one object per line - JSON Lines):
        {"id": "iso9362-bic-DEUTDEFF", "schema": "Company", "properties": {
            "swiftBic": ["DEUTDEFF"], "name": ["DEUTSCHE BANK AG"], 
            "country": ["de"], "address": ["FRANKFURT"]
        }}
        
        Returns number of entries loaded.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"OpenSanctions file not found: {filepath}")
        
        count = 0
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    props = data.get('properties', {})
                    
                    # Get BIC from swiftBic field (it's a list)
                    bic_list = props.get('swiftBic', [])
                    if not bic_list:
                        continue
                    
                    bic = bic_list[0].upper().strip()
                    if len(bic) >= 8:
                        base_bic = bic[:8]
                        
                        # Get other fields
                        names = props.get('name', [''])
                        countries = props.get('country', [''])
                        addresses = props.get('address', [''])
                        
                        self.bic_directory[base_bic] = BICEntry(
                            bic=base_bic,
                            bank_name=names[0] if names else '',
                            country=countries[0].upper() if countries else bic[4:6],
                            city=addresses[0] if addresses else '',
                            status='active'
                        )
                        count += 1
                except json.JSONDecodeError:
                    continue
        
        self.loaded['bic'] = True
        print(f"Loaded {count} BIC codes from OpenSanctions")
        return count
    
    def load_bic_from_github_swiftcodes(self, directory: str) -> int:
        """
        Load BIC directory from GitHub PeterNotenboom/SwiftCodes repo.
        
        Clone from: https://github.com/PeterNotenboom/SwiftCodes
        
        Structure: One JSON file per country in the directory
        Each file contains array of bank objects:
        [
            {
                "bank": "DEUTSCHE BANK AG",
                "city": "FRANKFURT",
                "branch": "",
                "swift_code": "DEUTDEFFXXX"
            }
        ]
        
        Args:
            directory: Path to the cloned repo's data directory
            
        Returns number of entries loaded.
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            raise FileNotFoundError(f"SwiftCodes directory not found: {directory}")
        
        count = 0
        # Process all JSON files in directory
        for json_file in dir_path.glob('*.json'):
            # Country code from filename (e.g., "DE.json" -> "DE")
            country = json_file.stem.upper()
            
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    banks = json.load(f)
                
                for bank in banks:
                    bic = bank.get('swift_code', '').upper().strip()
                    if len(bic) >= 8:
                        base_bic = bic[:8]
                        
                        self.bic_directory[base_bic] = BICEntry(
                            bic=base_bic,
                            bank_name=bank.get('bank', ''),
                            country=country,
                            city=bank.get('city', ''),
                            status='active'
                        )
                        count += 1
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load {json_file}: {e}")
                continue
        
        self.loaded['bic'] = True
        print(f"Loaded {count} BIC codes from GitHub SwiftCodes")
        return count
    
    def load_bic_from_gleif(self, filepath: str) -> int:
        """
        Load BIC directory from GLEIF BIC-to-LEI mapping file.
        
        Download from: https://www.gleif.org/en/lei-data/lei-mapping/download-bic-to-lei-relationship-files
        
        GLEIF CSV format:
        LEI,BIC
        5493001KJTIIGC8Y1R12,DEUTDEFF
        
        Note: This only contains BICs that have LEI mapping (~60k)
        
        Returns number of entries loaded.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"GLEIF file not found: {filepath}")
        
        count = 0
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                bic = row.get('BIC', '').upper().strip()
                if len(bic) >= 8:
                    base_bic = bic[:8]
                    
                    # GLEIF only has LEI and BIC, limited info
                    self.bic_directory[base_bic] = BICEntry(
                        bic=base_bic,
                        bank_name='',  # Not in GLEIF file
                        country=bic[4:6] if len(bic) >= 6 else '',
                        city='',
                        status='active'
                    )
                    count += 1
        
        self.loaded['bic'] = True
        print(f"Loaded {count} BIC codes from GLEIF")
        return count
    
    def load_bic_from_csv_simple(self, filepath: str) -> int:
        """
        Load BIC directory from simple CSV (just BIC codes, one per line).
        
        Format: Just BIC codes, optionally with header
        DEUTDEFF
        BNPAFRPP
        CHASUS33
        
        Returns number of entries loaded.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"BIC file not found: {filepath}")
        
        count = 0
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                bic = line.strip().upper()
                # Skip header or invalid lines
                if len(bic) >= 8 and bic[:4].isalpha():
                    base_bic = bic[:8]
                    
                    self.bic_directory[base_bic] = BICEntry(
                        bic=base_bic,
                        bank_name='',
                        country=bic[4:6] if len(bic) >= 6 else '',
                        city='',
                        status='active'
                    )
                    count += 1
        
        self.loaded['bic'] = True
        print(f"Loaded {count} BIC codes from simple CSV")
        return count
    
    def load_uk_sort_codes(self, filepath: str) -> int:
        """Load UK Sort Codes (6 digits)."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"UK Sort Code file not found: {filepath}")
        
        count = 0
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                code = line.strip().replace('-', '')
                if len(code) == 6 and code.isdigit():
                    self.uk_sort_codes.add(code)
                    count += 1
        
        self.loaded['uk_sort'] = True
        print(f"Loaded {count} UK Sort Codes")
        return count
    
    def load_german_blz(self, filepath: str) -> int:
        """Load German BLZ codes (8 digits)."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"German BLZ file not found: {filepath}")
        
        count = 0
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                code = line.strip()
                if len(code) == 8 and code.isdigit():
                    self.german_blz.add(code)
                    count += 1
        
        self.loaded['german_blz'] = True
        print(f"Loaded {count} German BLZ codes")
        return count
    
    def load_australian_bsb(self, filepath: str) -> int:
        """Load Australian BSB codes (6 digits)."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Australian BSB file not found: {filepath}")
        
        count = 0
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                code = line.strip().replace('-', '')
                if len(code) == 6 and code.isdigit():
                    self.australian_bsb.add(code)
                    count += 1
        
        self.loaded['australian_bsb'] = True
        print(f"Loaded {count} Australian BSB codes")
        return count
    
    # =========================================================================
    # VALIDATION METHODS
    # =========================================================================
    
    def validate_fedaba(self, routing: str) -> Tuple[bool, Optional[str], Optional[FedABAEntry]]:
        """
        Validate US FEDABA routing number.
        
        Returns:
            (exists, error_message, entry)
            - exists: True if routing number is in directory
            - error_message: None if exists, error description otherwise
            - entry: FedABAEntry if exists, None otherwise
        """
        if not routing:
            return False, "Empty routing number", None
        
        routing = routing.strip()
        
        # Format check first
        if len(routing) != 9 or not routing.isdigit():
            return False, "Invalid format: must be 9 digits", None
        
        # Checksum validation
        weights = [3, 7, 1, 3, 7, 1, 3, 7, 1]
        total = sum(int(d) * w for d, w in zip(routing, weights))
        if total % 10 != 0:
            return False, "Checksum validation failed", None
        
        # Directory lookup
        if not self.loaded['fedaba']:
            return True, "Directory not loaded (format valid)", None
        
        if routing in self.fedaba_directory:
            entry = self.fedaba_directory[routing]
            if entry.status.lower() != 'active':
                return False, f"Routing number inactive: {entry.bank_name}", entry
            return True, None, entry
        else:
            return False, "Routing number not in FedACH directory", None
    
    def validate_bic(self, bic: str) -> Tuple[bool, Optional[str], Optional[BICEntry]]:
        """
        Validate BIC/SWIFT code.
        
        Returns:
            (exists, error_message, entry)
            - exists: True if BIC is in directory
            - error_message: None if exists, error description otherwise
            - entry: BICEntry if exists, None otherwise
        """
        if not bic:
            return False, "Empty BIC", None
        
        bic = bic.upper().strip()
        
        # Format check
        if len(bic) not in (8, 11):
            return False, "Invalid length: must be 8 or 11 characters", None
        
        import re
        pattern = r'^[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}([A-Z0-9]{3})?$'
        if not re.match(pattern, bic):
            return False, "Invalid format: must match BIC pattern", None
        
        # Country code validation (basic)
        country = bic[4:6]
        # Could add VALID_COUNTRY_CODES check here
        
        # Directory lookup
        if not self.loaded['bic']:
            return True, "Directory not loaded (format valid)", None
        
        base_bic = bic[:8]
        if base_bic in self.bic_directory:
            entry = self.bic_directory[base_bic]
            if entry.status.lower() != 'active':
                return False, f"BIC inactive: {entry.bank_name}", entry
            return True, None, entry
        else:
            return False, "BIC not in SWIFT directory", None
    
    def validate_nch(self, code: str, country: str) -> Tuple[bool, Optional[str]]:
        """
        Validate National Clearing House code based on country.
        
        Returns:
            (exists, error_message)
        """
        if not code:
            return False, "Empty NCH code"
        
        code = code.strip().replace('-', '').replace(' ', '')
        country = country.upper() if country else ''
        
        if country == 'US':
            exists, msg, _ = self.validate_fedaba(code)
            return exists, msg
        
        elif country == 'GB':
            if not self.loaded['uk_sort']:
                # Format check only
                if len(code) == 6 and code.isdigit():
                    return True, "Directory not loaded (format valid)"
                return False, "Invalid UK Sort Code format (must be 6 digits)"
            
            if code in self.uk_sort_codes:
                return True, None
            return False, "Sort Code not in UK directory"
        
        elif country == 'DE':
            if not self.loaded['german_blz']:
                if len(code) == 8 and code.isdigit():
                    return True, "Directory not loaded (format valid)"
                return False, "Invalid German BLZ format (must be 8 digits)"
            
            if code in self.german_blz:
                return True, None
            return False, "BLZ not in German directory"
        
        elif country == 'AU':
            if not self.loaded['australian_bsb']:
                if len(code) == 6 and code.isdigit():
                    return True, "Directory not loaded (format valid)"
                return False, "Invalid Australian BSB format (must be 6 digits)"
            
            if code in self.australian_bsb:
                return True, None
            return False, "BSB not in Australian directory"
        
        else:
            # Unknown country - format check only
            return True, f"No directory for country {country}"
    
    # =========================================================================
    # BATCH VALIDATION
    # =========================================================================
    
    def validate_payment_features(self, features: Dict) -> Dict[str, Tuple[bool, str]]:
        """
        Validate all BIC/NCH codes in payment features.
        
        Returns dict of validation results:
        {
            'bnf_bic': (exists, message),
            'cdt_bic': (exists, message),
            ...
        }
        """
        results = {}
        
        for prefix in ['bnf_', 'cdt_', 'dbt_', 'orig_', 'intm_', 'send_']:
            # BIC validation
            bic = features.get(f'{prefix}bic_code')
            if bic:
                exists, msg, _ = self.validate_bic(bic)
                results[f'{prefix}bic'] = (exists, msg)
            
            # Account/NCH validation (need country for context)
            account = features.get(f'{prefix}account_value')
            country = features.get(f'{prefix}country') or features.get(f'{prefix}iban_country')
            
            if account and country:
                # Check if it looks like a routing number
                account_clean = account.strip()
                if len(account_clean) == 9 and account_clean.isdigit() and country == 'US':
                    exists, msg = self.validate_nch(account_clean, 'US')
                    results[f'{prefix}fedaba'] = (exists, msg)
        
        return results
    
    # =========================================================================
    # STATS
    # =========================================================================
    
    def get_stats(self) -> Dict:
        """Get directory statistics."""
        return {
            'bic_count': len(self.bic_directory),
            'fedaba_count': len(self.fedaba_directory),
            'uk_sort_count': len(self.uk_sort_codes),
            'german_blz_count': len(self.german_blz),
            'australian_bsb_count': len(self.australian_bsb),
            'loaded': self.loaded
        }


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == '__main__':
    validator = DirectoryValidator()
    
    print("="*70)
    print("DIRECTORY VALIDATOR - EXAMPLE USAGE")
    print("="*70)
    
    # =========================================================================
    # Loading Directories (uncomment the one you have)
    # =========================================================================
    
    print("\n--- Loading Examples (uncomment to use) ---\n")
    
    # Option 1: OpenSanctions (JSON Lines format)
    # Download from: https://www.opensanctions.org/datasets/iso9362_bic/
    # validator.load_bic_from_opensanctions('iso9362_bic.json')
    
    # Option 2: GitHub SwiftCodes (directory of JSON files)
    # Clone from: https://github.com/PeterNotenboom/SwiftCodes
    # validator.load_bic_from_github_swiftcodes('./SwiftCodes/data/')
    
    # Option 3: GLEIF BIC-LEI mapping (CSV)
    # Download from: https://www.gleif.org/en/lei-data/lei-mapping/download-bic-to-lei-relationship-files
    # validator.load_bic_from_gleif('bic_lei_mapping.csv')
    
    # Option 4: Simple CSV (one BIC per line)
    # validator.load_bic_from_csv_simple('bic_codes.txt')
    
    # Option 5: Generic CSV with headers
    # validator.load_bic('swift_codes.csv')
    
    # FEDABA (US Routing Numbers)
    # Download from: https://www.frbservices.org/EPaymentsDirectory/download.html
    # validator.load_fedaba('fedach_directory.csv')
    
    # =========================================================================
    # Format Validation (works without directory files)
    # =========================================================================
    
    print("\n--- Format Validation (no directory needed) ---\n")
    
    # Valid BIC format
    exists, msg, entry = validator.validate_bic('DEUTDEFF')
    print(f"DEUTDEFF: exists={exists}, msg={msg}")
    
    # Valid BIC with branch code
    exists, msg, entry = validator.validate_bic('DEUTDEFFXXX')
    print(f"DEUTDEFFXXX: exists={exists}, msg={msg}")
    
    # Invalid BIC format (too short)
    exists, msg, entry = validator.validate_bic('INVALID')
    print(f"INVALID: exists={exists}, msg={msg}")
    
    # Invalid BIC format (wrong pattern)
    exists, msg, entry = validator.validate_bic('12345678')
    print(f"12345678: exists={exists}, msg={msg}")
    
    print()
    
    # Valid FEDABA format (checksum passes)
    exists, msg, entry = validator.validate_fedaba('021000021')
    print(f"021000021 (JPMorgan): exists={exists}, msg={msg}")
    
    # Valid FEDABA format
    exists, msg, entry = validator.validate_fedaba('011000015')
    print(f"011000015 (Fed Reserve): exists={exists}, msg={msg}")
    
    # Invalid FEDABA (wrong checksum)
    exists, msg, entry = validator.validate_fedaba('021000022')
    print(f"021000022 (bad checksum): exists={exists}, msg={msg}")
    
    # Invalid FEDABA (wrong length)
    exists, msg, entry = validator.validate_fedaba('12345')
    print(f"12345 (wrong length): exists={exists}, msg={msg}")
    
    # =========================================================================
    # NCH Validation by Country
    # =========================================================================
    
    print("\n--- NCH Validation by Country ---\n")
    
    # US routing number
    exists, msg = validator.validate_nch('021000021', 'US')
    print(f"US 021000021: exists={exists}, msg={msg}")
    
    # UK sort code (format only without directory)
    exists, msg = validator.validate_nch('123456', 'GB')
    print(f"UK 123456: exists={exists}, msg={msg}")
    
    # German BLZ (format only without directory)
    exists, msg = validator.validate_nch('12345678', 'DE')
    print(f"DE 12345678: exists={exists}, msg={msg}")
    
    # Australian BSB (format only without directory)
    exists, msg = validator.validate_nch('123456', 'AU')
    print(f"AU 123456: exists={exists}, msg={msg}")
    
    # =========================================================================
    # Stats
    # =========================================================================
    
    print("\n--- Directory Stats ---\n")
    print(f"Stats: {validator.get_stats()}")
    
    print("\n" + "="*70)
    print("To enable full directory lookups, load the appropriate files.")
    print("="*70)
