"""
Analyze ACE code co-occurrence to discover mutually exclusive pairs.

Mutually exclusive codes are those that:
1. Have high individual frequency but rarely co-occur
2. Target the same party/field with conflicting validations

Usage:
    python analyze_exclusions.py --data-dir ./parsed_data --output exclusions.json
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple
import itertools
import re


def load_codes_from_directory(data_dir: str) -> List[Set[str]]:
    """Load all response codes from parsed IFML files."""
    data_path = Path(data_dir)
    all_code_sets = []
    
    for json_file in sorted(data_path.glob('*.json')):
        try:
            with open(json_file) as f:
                data = json.load(f)
            
            for txn_id, payment in data.items():
                if not isinstance(payment, dict):
                    continue
                
                response = payment.get('Response', {})
                codes = extract_codes_from_response(response)
                if codes:
                    all_code_sets.append(codes)
                    
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    return all_code_sets


def extract_codes_from_response(response: dict) -> Set[str]:
    """Extract all codes from a response."""
    codes = set()
    
    def find_codes(obj, path=""):
        if isinstance(obj, dict):
            # Check for Code field
            if 'Code' in obj:
                code = obj['Code']
                if code:
                    # Also capture party hint from InformationalData
                    info = obj.get('InformationalData', '')
                    party = extract_party_hint(info)
                    if party:
                        codes.add(f"{code}_{party}")
                    codes.add(str(code))
            
            # Check for MsgStatus which may be a list
            if 'MsgStatus' in obj:
                status = obj['MsgStatus']
                if isinstance(status, list):
                    for item in status:
                        find_codes(item)
                else:
                    find_codes(status)
            
            # Recurse
            for k, v in obj.items():
                find_codes(v, f"{path}.{k}")
                
        elif isinstance(obj, list):
            for item in obj:
                find_codes(item, path)
    
    find_codes(response)
    return codes


def extract_party_hint(info: str) -> str:
    """Extract party hint from informational data."""
    if not info:
        return None
    
    party_patterns = ['BNFBNK', 'BNFPTY', 'BNPPTY', 'CDTPTY', 'DBTPTY', 
                      'ORGPTY', 'INTBNK', 'SNDBNK', 'ACWBNK', 'ORDBNK']
    
    for pattern in party_patterns:
        if info.startswith(pattern):
            return pattern
    
    return None


def build_cooccurrence_matrix(code_sets: List[Set[str]]) -> Dict[Tuple[str, str], int]:
    """Build matrix of how often code pairs co-occur."""
    cooccurrence = defaultdict(int)
    
    for codes in code_sets:
        # Get all pairs in this transaction
        code_list = sorted(codes)
        for c1, c2 in itertools.combinations(code_list, 2):
            pair = (c1, c2) if c1 < c2 else (c2, c1)
            cooccurrence[pair] += 1
    
    return cooccurrence


def find_mutual_exclusions(code_sets: List[Set[str]], 
                           min_individual_freq: int = 10,
                           max_cooccurrence_ratio: float = 0.05) -> List[Dict]:
    """
    Find mutually exclusive code pairs.
    
    Args:
        code_sets: List of code sets per transaction
        min_individual_freq: Minimum times each code must appear individually
        max_cooccurrence_ratio: Max ratio of co-occurrence to min individual freq
        
    Returns:
        List of exclusion rules
    """
    # Count individual code frequencies
    code_freq = Counter()
    for codes in code_sets:
        code_freq.update(codes)
    
    # Build co-occurrence matrix
    cooccurrence = build_cooccurrence_matrix(code_sets)
    
    # Find candidate exclusions
    exclusions = []
    
    # Get codes that meet minimum frequency
    frequent_codes = {c for c, f in code_freq.items() if f >= min_individual_freq}
    
    # Check all pairs
    for c1, c2 in itertools.combinations(sorted(frequent_codes), 2):
        pair = (c1, c2)
        co_count = cooccurrence.get(pair, 0)
        
        min_freq = min(code_freq[c1], code_freq[c2])
        ratio = co_count / min_freq if min_freq > 0 else 0
        
        if ratio <= max_cooccurrence_ratio:
            # Check if codes are related (same base code or same party)
            base1 = c1.split('_')[0]
            base2 = c2.split('_')[0]
            party1 = c1.split('_')[1] if '_' in c1 else None
            party2 = c2.split('_')[1] if '_' in c2 else None
            
            # Determine relationship type
            relationship = None
            if party1 and party2 and party1 == party2:
                relationship = f"same_party_{party1}"
            elif base1 == base2:
                relationship = "same_base_code"
            elif abs(int(base1) - int(base2)) <= 10:
                relationship = "nearby_codes"
            
            exclusions.append({
                'code1': c1,
                'code2': c2,
                'code1_freq': code_freq[c1],
                'code2_freq': code_freq[c2],
                'cooccurrence': co_count,
                'ratio': round(ratio, 4),
                'relationship': relationship,
                'confidence': 'high' if ratio == 0 else 'medium' if ratio < 0.02 else 'low'
            })
    
    # Sort by confidence then ratio
    exclusions.sort(key=lambda x: (x['confidence'] != 'high', x['ratio']))
    
    return exclusions


def find_same_field_conflicts(code_sets: List[Set[str]]) -> List[Dict]:
    """
    Find codes that validate the same field but with different checks.
    These are likely mutually exclusive.
    
    Examples:
    - 8894 (Invalid IBAN) vs 8896 (Invalid Domestic Account) - same account field
    - 8001 (Invalid BIC) vs 8005 (Invalid BIC4) - same BIC field
    """
    # Define field groupings based on code descriptions
    field_groups = {
        'account': ['8894', '8896', '8004', '8033'],  # IBAN, domestic, BBAN, CLABE
        'bic': ['8001', '8005', '8006'],  # BIC validations
        'country': ['8006', '8027', '8022'],  # Country code validations
        'amount': ['8090', '8091'],  # Amount validations
    }
    
    conflicts = []
    
    # Count co-occurrences for codes in same field group
    code_freq = Counter()
    for codes in code_sets:
        code_freq.update(codes)
    
    cooccurrence = build_cooccurrence_matrix(code_sets)
    
    for field, field_codes in field_groups.items():
        # Find all codes that match these base codes
        matching_codes = set()
        for codes in code_sets:
            for c in codes:
                base = c.split('_')[0]
                if base in field_codes:
                    matching_codes.add(c)
        
        # Check pairs within this field
        for c1, c2 in itertools.combinations(sorted(matching_codes), 2):
            if code_freq[c1] < 5 or code_freq[c2] < 5:
                continue
            
            pair = (c1, c2) if c1 < c2 else (c2, c1)
            co_count = cooccurrence.get(pair, 0)
            min_freq = min(code_freq[c1], code_freq[c2])
            ratio = co_count / min_freq if min_freq > 0 else 0
            
            if ratio < 0.1:  # Less than 10% co-occurrence
                conflicts.append({
                    'code1': c1,
                    'code2': c2,
                    'field': field,
                    'code1_freq': code_freq[c1],
                    'code2_freq': code_freq[c2],
                    'cooccurrence': co_count,
                    'ratio': round(ratio, 4)
                })
    
    return conflicts


def generate_exclusion_rules(exclusions: List[Dict], 
                             same_field: List[Dict]) -> Dict:
    """Generate exclusion rules for predictor."""
    rules = {
        'mutual_exclusions': [],
        'same_field_conflicts': [],
        'resolution_hints': {}
    }
    
    # High confidence exclusions
    for exc in exclusions:
        if exc['confidence'] == 'high':
            rules['mutual_exclusions'].append({
                'codes': [exc['code1'], exc['code2']],
                'evidence': f"Never co-occur (0/{exc['code1_freq']}+{exc['code2_freq']} transactions)"
            })
    
    # Same field conflicts with resolution hints
    for conf in same_field:
        base1 = conf['code1'].split('_')[0]
        base2 = conf['code2'].split('_')[0]
        
        # Add resolution hint based on code semantics
        hint = None
        if '8894' in [base1, base2] and '8896' in [base1, base2]:
            hint = "Use is_international feature: True->8894, False->8896"
        elif '8001' in [base1, base2] and '8005' in [base1, base2]:
            hint = "Use bic_length feature: <=4->8005, >4->8001"
        
        rules['same_field_conflicts'].append({
            'codes': [conf['code1'], conf['code2']],
            'field': conf['field'],
            'ratio': conf['ratio'],
            'resolution': hint
        })
        
        if hint:
            key = tuple(sorted([conf['code1'], conf['code2']]))
            rules['resolution_hints'][str(key)] = hint
    
    return rules


def print_analysis(code_sets: List[Set[str]], 
                   exclusions: List[Dict],
                   same_field: List[Dict]):
    """Print analysis summary."""
    print("=" * 70)
    print("ACE Code Co-occurrence Analysis")
    print("=" * 70)
    
    # Basic stats
    code_freq = Counter()
    for codes in code_sets:
        code_freq.update(codes)
    
    print(f"\nTotal transactions: {len(code_sets)}")
    print(f"Unique codes: {len(code_freq)}")
    print(f"Top 10 codes:")
    for code, freq in code_freq.most_common(10):
        print(f"  {code}: {freq}")
    
    # Exclusions
    print(f"\n{'='*70}")
    print("POTENTIAL MUTUAL EXCLUSIONS (rarely co-occur)")
    print("=" * 70)
    
    high_conf = [e for e in exclusions if e['confidence'] == 'high'][:20]
    if high_conf:
        print(f"\nHigh confidence (never co-occur): {len(high_conf)} pairs")
        for exc in high_conf[:10]:
            print(f"  {exc['code1']} vs {exc['code2']}")
            print(f"    Frequencies: {exc['code1_freq']} / {exc['code2_freq']}, Co-occur: {exc['cooccurrence']}")
    
    # Same field conflicts
    print(f"\n{'='*70}")
    print("SAME-FIELD CONFLICTS (validate same data differently)")
    print("=" * 70)
    
    for conf in same_field[:10]:
        print(f"\n  {conf['code1']} vs {conf['code2']} (field: {conf['field']})")
        print(f"    Frequencies: {conf['code1_freq']} / {conf['code2_freq']}")
        print(f"    Co-occurrence: {conf['cooccurrence']} ({conf['ratio']*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='Analyze ACE code exclusions')
    parser.add_argument('--data-dir', required=True, help='Directory with parsed IFML files')
    parser.add_argument('--output', default='exclusion_rules.json', help='Output rules file')
    parser.add_argument('--min-freq', type=int, default=10, help='Minimum code frequency')
    parser.add_argument('--max-ratio', type=float, default=0.05, help='Max co-occurrence ratio')
    
    args = parser.parse_args()
    
    print(f"Loading data from {args.data_dir}...")
    code_sets = load_codes_from_directory(args.data_dir)
    print(f"Loaded {len(code_sets)} transactions with codes")
    
    if not code_sets:
        print("No data found!")
        return
    
    print("\nAnalyzing co-occurrences...")
    exclusions = find_mutual_exclusions(
        code_sets, 
        min_individual_freq=args.min_freq,
        max_cooccurrence_ratio=args.max_ratio
    )
    
    print("Finding same-field conflicts...")
    same_field = find_same_field_conflicts(code_sets)
    
    # Print analysis
    print_analysis(code_sets, exclusions, same_field)
    
    # Generate and save rules
    rules = generate_exclusion_rules(exclusions, same_field)
    
    with open(args.output, 'w') as f:
        json.dump(rules, f, indent=2)
    print(f"\nRules saved to {args.output}")
    
    # Also save raw analysis
    analysis_file = args.output.replace('.json', '_analysis.json')
    with open(analysis_file, 'w') as f:
        json.dump({
            'exclusions': exclusions[:100],
            'same_field_conflicts': same_field,
            'total_transactions': len(code_sets)
        }, f, indent=2)
    print(f"Full analysis saved to {analysis_file}")


if __name__ == '__main__':
    main()
