#!/usr/bin/env python3
"""
Build a "known valid" bank code directory from production data.

Logic:
- If ACE did NOT fire 8894 on a transaction, the IBAN was valid
- Extract bank codes from all "clean" transactions  
- Build lookup set of known-valid (country, bank_code) pairs
- Use this to predict 8894 on new transactions with unknown bank codes

Usage:
    # Build directory from production data
    python build_bank_directory.py --data-dir ./prd_emts --output bank_directory.json
    
    # Test with the built directory
    python build_bank_directory.py --data-dir ./prd_emts --directory bank_directory.json --test
"""

import argparse
import json
import re
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, Set, Tuple, Optional, List

# BBAN bank code positions by country
# Format: (start_index, end_index) within BBAN (after country+check digits)
BANK_CODE_POSITIONS = {
    # Western Europe
    'DE': (0, 8),   # 8-digit BLZ
    'FR': (0, 5),   # 5-digit bank code
    'ES': (0, 4),   # 4-digit bank code
    'IT': (1, 6),   # 5-digit ABI (after CIN check letter)
    'NL': (0, 4),   # 4-letter bank code
    'BE': (0, 3),   # 3-digit bank code
    'AT': (0, 5),   # 5-digit bank code
    'CH': (0, 5),   # 5-digit bank code
    'LU': (0, 3),   # 3-digit bank code
    # UK & Ireland
    'GB': (0, 4),   # 4-letter bank code
    'IE': (0, 4),   # 4-letter bank code
    # Nordic
    'FI': (0, 3),   # 3-digit bank code
    'SE': (0, 3),   # 3-digit clearing code
    'NO': (0, 4),   # 4-digit bank code
    'DK': (0, 4),   # 4-digit bank code
    # Eastern Europe
    'PL': (0, 8),   # 8-digit bank/branch code
    'CZ': (0, 4),   # 4-digit bank code
    'SK': (0, 4),   # 4-digit bank code
    'HU': (0, 3),   # 3-digit bank code
    'RO': (0, 4),   # 4-letter bank code
    'BG': (0, 4),   # 4-letter bank code
    'HR': (0, 7),   # 7-digit bank code
    'SI': (0, 5),   # 5-digit bank code
    # Baltic
    'EE': (0, 2),   # 2-digit bank code
    'LV': (0, 4),   # 4-letter bank code
    'LT': (0, 5),   # 5-digit bank code
    # Southern Europe
    'PT': (0, 4),   # 4-digit bank code
    'GR': (0, 3),   # 3-digit bank code
    'CY': (0, 3),   # 3-digit bank code
    'MT': (0, 4),   # 4-letter bank code
    # Middle East
    'SA': (0, 2),   # 2-digit bank code
    'AE': (0, 3),   # 3-digit bank code
    'TR': (0, 5),   # 5-digit bank code
    'IL': (0, 3),   # 3-digit bank code
    'QA': (0, 4),   # 4-letter bank code
    'KW': (0, 4),   # 4-letter bank code
    'BH': (0, 4),   # 4-letter bank code
}


def looks_like_iban(s: str) -> bool:
    """Check if string looks like an IBAN."""
    if not s or not isinstance(s, str):
        return False
    cleaned = s.upper().replace(' ', '').replace('-', '')
    if len(cleaned) < 15 or len(cleaned) > 34:
        return False
    return bool(re.match(r'^[A-Z]{2}[0-9]{2}[A-Z0-9]+$', cleaned))


def extract_bank_code(iban: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract country and bank code from IBAN.
    
    Returns:
        (country_code, bank_code) or (None, None) if can't extract
    """
    if not iban:
        return None, None
    
    # Clean IBAN
    cleaned = iban.upper().replace(' ', '').replace('-', '')
    
    if len(cleaned) < 5:
        return None, None
    
    country = cleaned[:2]
    bban = cleaned[4:]  # Skip country (2) + check digits (2)
    
    # Get bank code position for this country
    positions = BANK_CODE_POSITIONS.get(country)
    
    if not positions:
        # Unknown country - use first 4 chars as default
        if len(bban) >= 4:
            return country, bban[:4]
        return country, None
    
    start, end = positions
    if len(bban) >= end:
        bank_code = bban[start:end]
        return country, bank_code
    
    return country, None


def extract_ibans_from_obj(obj) -> List[str]:
    """
    Recursively extract all IBAN values from a JSON object.
    """
    ibans = []
    
    if isinstance(obj, dict):
        # Check if this is an ID field with IBAN
        if 'ID' in obj:
            id_field = obj['ID']
            if isinstance(id_field, dict):
                id_type = (id_field.get('Type') or id_field.get('@Type') or '').upper()
                id_text = id_field.get('text') or id_field.get('#text') or ''
                
                if id_type == 'IBAN' or looks_like_iban(str(id_text)):
                    ibans.append(str(id_text))
            elif isinstance(id_field, str) and looks_like_iban(id_field):
                ibans.append(id_field)
        
        # Also check for AcctIDInfo / AcctIDInf
        for acct_key in ['AcctIDInfo', 'AcctIDInf']:
            if acct_key in obj:
                acct = obj[acct_key]
                if isinstance(acct, dict):
                    acct_id = acct.get('ID', {})
                    if isinstance(acct_id, dict):
                        acct_type = (acct_id.get('Type') or acct_id.get('@Type') or '').upper()
                        acct_text = acct_id.get('text') or acct_id.get('#text') or ''
                        if acct_type == 'IBAN' or looks_like_iban(str(acct_text)):
                            ibans.append(str(acct_text))
        
        # Recurse into all values
        for value in obj.values():
            ibans.extend(extract_ibans_from_obj(value))
    
    elif isinstance(obj, list):
        for item in obj:
            ibans.extend(extract_ibans_from_obj(item))
    
    return ibans


def extract_error_codes(obj) -> List[str]:
    """
    Extract error codes from response object.
    """
    codes = []
    
    if isinstance(obj, dict):
        # Check for MsgStatus
        if 'MsgStatus' in obj:
            status_list = obj['MsgStatus']
            if isinstance(status_list, dict):
                status_list = [status_list]
            for status in status_list:
                if isinstance(status, dict):
                    code = status.get('Code')
                    if code:
                        codes.append(str(code))
        
        # Check for AuditTrail
        if 'AuditTrail' in obj:
            codes.extend(extract_error_codes(obj['AuditTrail']))
        
        # Recurse
        for value in obj.values():
            codes.extend(extract_error_codes(value))
    
    elif isinstance(obj, list):
        for item in obj:
            codes.extend(extract_error_codes(item))
    
    return codes


def process_json_file(filepath: Path) -> List[Tuple[str, List[str], List[str]]]:
    """
    Process a JSON file containing transactions.
    
    Returns list of (transaction_id, ibans, error_codes) tuples.
    """
    results = []
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"  Warning: Could not read {filepath}: {e}")
        return results
    
    # Handle different file structures
    if isinstance(data, dict):
        # Check if top-level keys are transaction IDs
        for key, value in data.items():
            if isinstance(value, dict):
                # This might be a transaction
                request = value.get('Request', {})
                response = value.get('Response', {})
                
                # Extract IBANs from request
                ibans = extract_ibans_from_obj(request)
                
                # Extract error codes from response
                error_codes = extract_error_codes(response)
                
                if ibans or error_codes:
                    results.append((key, ibans, error_codes))
    
    return results


def build_directory(data_dir: str, output_file: str, limit: int = 0):
    """
    Build bank code directory from production data.
    """
    print(f"Scanning {data_dir} for JSON files...")
    
    data_path = Path(data_dir)
    json_files = list(data_path.glob("*.json"))
    
    print(f"Found {len(json_files)} JSON files")
    
    # Track valid bank codes
    valid_bank_codes = defaultdict(set)  # country -> set of bank codes
    codes_with_8894 = defaultdict(set)   # country -> set of bank codes seen in 8894 cases
    
    # Statistics
    stats = {
        'files_processed': 0,
        'transactions_processed': 0,
        'transactions_with_iban': 0,
        'transactions_with_8894': 0,
        'ibans_extracted': 0,
    }
    
    for i, json_file in enumerate(json_files):
        if limit > 0 and stats['transactions_processed'] >= limit:
            break
        
        transactions = process_json_file(json_file)
        stats['files_processed'] += 1
        
        for txn_id, ibans, error_codes in transactions:
            if limit > 0 and stats['transactions_processed'] >= limit:
                break
            
            stats['transactions_processed'] += 1
            
            has_8894 = any('8894' in str(c) for c in error_codes)
            
            if has_8894:
                stats['transactions_with_8894'] += 1
            
            if ibans:
                stats['transactions_with_iban'] += 1
                
                for iban in ibans:
                    country, bank_code = extract_bank_code(iban)
                    
                    if country and bank_code:
                        stats['ibans_extracted'] += 1
                        
                        if has_8894:
                            codes_with_8894[country].add(bank_code)
                        else:
                            # No 8894 = this bank code is valid
                            valid_bank_codes[country].add(bank_code)
        
        if (i + 1) % 1 == 0:
            print(f"  Processed {i+1}/{len(json_files)} files, {stats['transactions_processed']:,} transactions...")
    
    # Bank codes that ONLY appeared with 8894 (never without) are potentially invalid
    potentially_invalid = {}
    for country, codes in codes_with_8894.items():
        invalid_only = codes - valid_bank_codes.get(country, set())
        if invalid_only:
            potentially_invalid[country] = sorted(list(invalid_only))
    
    # Prepare output
    directory = {
        'metadata': {
            'source': str(data_dir),
            'files_processed': stats['files_processed'],
            'transactions_processed': stats['transactions_processed'],
            'transactions_with_iban': stats['transactions_with_iban'],
            'transactions_with_8894': stats['transactions_with_8894'],
            'ibans_extracted': stats['ibans_extracted'],
            'unique_bank_codes': sum(len(v) for v in valid_bank_codes.values()),
        },
        'valid_bank_codes': {
            country: sorted(list(codes)) 
            for country, codes in sorted(valid_bank_codes.items())
        },
        'potentially_invalid': potentially_invalid,
        'country_stats': {
            country: len(codes)
            for country, codes in sorted(valid_bank_codes.items())
        }
    }
    
    # Save
    with open(output_file, 'w') as f:
        json.dump(directory, f, indent=2)
    
    print(f"\n{'='*60}")
    print("BANK DIRECTORY BUILD COMPLETE")
    print('='*60)
    print(f"Files processed: {stats['files_processed']}")
    print(f"Transactions processed: {stats['transactions_processed']:,}")
    print(f"Transactions with IBAN: {stats['transactions_with_iban']:,}")
    print(f"Transactions with 8894: {stats['transactions_with_8894']:,}")
    print(f"IBANs extracted: {stats['ibans_extracted']:,}")
    print(f"Unique valid bank codes: {directory['metadata']['unique_bank_codes']:,}")
    print(f"\nCountry breakdown:")
    for country, count in sorted(directory['country_stats'].items(), key=lambda x: -x[1])[:15]:
        print(f"  {country}: {count:,} bank codes")
    if len(directory['country_stats']) > 15:
        print(f"  ... and {len(directory['country_stats']) - 15} more countries")
    
    if potentially_invalid:
        print(f"\nPotentially invalid bank codes (only seen with 8894):")
        for country, codes in potentially_invalid.items():
            print(f"  {country}: {codes[:5]}{'...' if len(codes) > 5 else ''}")
    
    print(f"\nSaved to: {output_file}")
    
    return directory


def test_with_directory(data_dir: str, directory_file: str, limit: int = 0):
    """
    Test 8894 prediction using the built bank directory.
    """
    print(f"Loading bank directory from {directory_file}...")
    with open(directory_file, 'r') as f:
        directory = json.load(f)
    
    valid_codes = directory['valid_bank_codes']
    
    # Convert to sets for fast lookup
    valid_lookup = {
        country: set(codes) 
        for country, codes in valid_codes.items()
    }
    
    total_codes = sum(len(v) for v in valid_lookup.values())
    print(f"Loaded {total_codes:,} valid bank codes from {len(valid_lookup)} countries")
    
    print(f"\nScanning {data_dir} for test data...")
    
    data_path = Path(data_dir)
    json_files = list(data_path.glob("*.json"))
    
    # Test prediction
    tp, tn, fp, fn = 0, 0, 0, 0
    fn_examples = []
    fp_examples = []
    processed = 0
    
    for json_file in json_files:
        transactions = process_json_file(json_file)
        
        for txn_id, ibans, error_codes in transactions:
            if limit > 0 and processed >= limit:
                break
            
            processed += 1
            
            # Get actual result
            has_actual_8894 = any('8894' in str(c) for c in error_codes)
            
            # Predict based on bank code lookup
            predicted_8894 = False
            unknown_codes = []
            
            for iban in ibans:
                country, bank_code = extract_bank_code(iban)
                
                if country and bank_code:
                    country_codes = valid_lookup.get(country, set())
                    
                    if country not in valid_lookup:
                        # Unknown country - can't validate
                        pass
                    elif bank_code not in country_codes:
                        # Unknown bank code - predict 8894
                        predicted_8894 = True
                        unknown_codes.append(f"{country}:{bank_code}")
            
            # Classify
            if predicted_8894 and has_actual_8894:
                tp += 1
            elif not predicted_8894 and not has_actual_8894:
                tn += 1
            elif predicted_8894 and not has_actual_8894:
                fp += 1
                if len(fp_examples) < 5:
                    fp_examples.append((txn_id, unknown_codes))
            else:
                fn += 1
                if len(fn_examples) < 5:
                    fn_examples.append((txn_id, ibans[:2]))
        
        if limit > 0 and processed >= limit:
            break
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n{'='*60}")
    print("8894 PREDICTION RESULTS (Bank Directory Lookup)")
    print('='*60)
    print(f"""
    Transactions tested: {processed:,}
    
    Confusion Matrix:
                         Actual 8894    Not 8894
    Predicted 8894         TP={tp:<6}     FP={fp:<6}
    Not Predicted          FN={fn:<6}     TN={tn:<6}
    
    Metrics:
      Precision: {precision*100:.1f}%  (when we predict 8894, how often correct)
      Recall:    {recall*100:.1f}%  (what % of actual 8894 do we catch)
      F1 Score:  {f1*100:.1f}%
    """)
    
    if fn_examples:
        print("False Negatives (8894 occurred but we didn't predict):")
        for txn, ibans in fn_examples:
            print(f"  {txn}: {ibans}")
    
    if fp_examples:
        print("\nFalse Positives (predicted but didn't occur):")
        for txn, unknown in fp_examples:
            print(f"  {txn}: unknown codes = {unknown}")


def main():
    parser = argparse.ArgumentParser(description='Build/test bank code directory from production data')
    parser.add_argument('--data-dir', required=True, help='IFML data directory')
    parser.add_argument('--output', default='bank_directory.json', help='Output file for directory')
    parser.add_argument('--directory', help='Existing directory file for testing')
    parser.add_argument('--test', action='store_true', help='Test prediction with directory')
    parser.add_argument('--limit', type=int, default=0, help='Max transactions (0=all)')
    
    args = parser.parse_args()
    
    if args.test:
        if not args.directory:
            parser.error("--directory required when using --test")
        test_with_directory(args.data_dir, args.directory, args.limit)
    else:
        build_directory(args.data_dir, args.output, args.limit)


if __name__ == "__main__":
    main()
