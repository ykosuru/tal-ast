#!/usr/bin/env python3
"""
Test rules for 8004 against actual IFMLs.
8004 = IBAN Cannot Be Derived

EXACT SAME LOGIC as validate_8xxx_all.py (which has 100% Recall, 33% Precision)

Usage:
    python 8004.py --data-dir /path/to/ifml/prod/data 
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

from data_pipeline import IFMLDataPipeline

TARGET_CODE = '8004'


def check_8004_rules(features: Dict) -> Tuple[bool, List[str]]:
    """
    8004 = IBAN Cannot Be Derived
    
    EXACT COPY from validate_8xxx_all.py
    Expected: 100% Recall, 33% Precision
    """
    reasons = []
    
    def get(feat, default=None):
        return features.get(feat, default)
    
    # Rule 1: needs_iban but no IBAN
    for prefix in ['bnf_', 'cdt_', 'dbt_', 'orig_']:
        needs_iban = get(f'{prefix}needs_iban', False)
        has_iban = get(f'{prefix}has_iban', False)
        if needs_iban and not has_iban:
            party = prefix.rstrip('_').upper()
            reasons.append(f"{party}: needs_iban but no IBAN")
    
    # Rule 2: has_account but no IBAN (IBAN derivation needed)
    for prefix in ['bnf_', 'cdt_', 'dbt_']:
        has_account = get(f'{prefix}has_account', False)
        has_iban = get(f'{prefix}has_iban', False)
        if has_account and not has_iban:
            party = prefix.rstrip('_').upper()
            reasons.append(f"{party}: has_account but no IBAN")
    
    # Rule 3: missing_required_iban
    if get('missing_required_iban', False):
        reasons.append("missing_required_iban=True")
    
    return len(reasons) > 0, reasons


def main():
    parser = argparse.ArgumentParser(description=f'Test {TARGET_CODE} rules')
    parser.add_argument('--data-dir', required=True, help='IFML JSON directory')
    parser.add_argument('--limit', type=int, default=10000, help='Max records')
    parser.add_argument('--show-fp', type=int, default=10, help='FP to show')
    parser.add_argument('--show-fn', type=int, default=10, help='FN to show')
    
    args = parser.parse_args()
    
    print(f"Loading IFML data for {TARGET_CODE} validation...")
    pipeline = IFMLDataPipeline()
    data_path = Path(args.data_dir)
    
    if data_path.is_file():
        pipeline.load_single_file(str(data_path))
    else:
        pipeline.load_directory(str(data_path), "*.json")
    
    print(f"Pipeline has {len(pipeline.records)} records")
    
    # Classification
    tp, tn, fp, fn = 0, 0, 0, 0
    fp_list = []
    fn_list = []
    
    for i, record in enumerate(pipeline.records):
        if i >= args.limit:
            break
        
        txn_id = record.transaction_id
        features = record.request_features
        codes = record.error_codes_only + (record.composite_codes or [])
        
        has_actual = any(TARGET_CODE in str(c) for c in codes)
        predicted, reasons = check_8004_rules(features)
        
        if predicted and has_actual:
            tp += 1
        elif not predicted and not has_actual:
            tn += 1
        elif predicted and not has_actual:
            fp += 1
            if len(fp_list) < 100:
                fp_list.append((txn_id, reasons, codes))
        else:
            fn += 1
            if len(fn_list) < 100:
                fn_list.append((txn_id, codes, features))
    
    # Metrics
    total = tp + tn + fp + fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Report
    print("\n" + "="*70)
    print(f"TEST RESULTS: {TARGET_CODE} - IBAN Cannot Be Derived")
    print("="*70)
    
    print(f"""
CONFUSION MATRIX:
                    Actual
                 {TARGET_CODE}      Not {TARGET_CODE}
              ┌──────────┬──────────┐
Predicted     │          │          │
   {TARGET_CODE}       │ TP={tp:<5} │ FP={fp:<5} │
              ├──────────┼──────────┤
Not {TARGET_CODE}      │ FN={fn:<5} │ TN={tn:<5} │
              └──────────┴──────────┘

METRICS:
  Precision: {precision*100:>6.2f}%  (TP / (TP + FP))
  Recall:    {recall*100:>6.2f}%  (TP / (TP + FN))
  F1 Score:  {f1*100:>6.2f}%

RULES (same as validate_8xxx_all.py):
  1. needs_iban=True AND has_iban=False
  2. has_account=True AND has_iban=False
  3. missing_required_iban=True
    """)
    
    # False Positives Analysis
    if fp_list and args.show_fp > 0:
        print("\n" + "="*70)
        print(f"FALSE POSITIVES ({fp} total, showing {min(fp, args.show_fp)}):")
        print("="*70)
        
        # Analyze which rules trigger FPs
        fp_triggers = defaultdict(int)
        for _, reasons, _ in fp_list:
            for r in reasons:
                trigger = r.split(':')[0] if ':' in r else r
                fp_triggers[trigger] += 1
        
        print("\nFP Trigger Analysis:")
        for trigger, count in sorted(fp_triggers.items(), key=lambda x: -x[1]):
            print(f"  {trigger}: {count} ({count/fp*100:.1f}%)")
        
        print("\nSample FPs:")
        for i, (txn, reasons, codes) in enumerate(fp_list[:args.show_fp], 1):
            print(f"  {i}. {txn}")
            print(f"     Trigger: {reasons[0] if reasons else '?'}")
            print(f"     Actual codes: {codes}")
    
    # False Negatives Analysis  
    if fn_list and args.show_fn > 0:
        print("\n" + "="*70)
        print(f"FALSE NEGATIVES ({fn} total, showing {min(fn, args.show_fn)}):")
        print("="*70)
        
        print("\nSample FNs:")
        for i, (txn, codes, features) in enumerate(fn_list[:args.show_fn], 1):
            print(f"  {i}. {txn}")
            print(f"     Codes: {codes}")
            print(f"     bnf_needs_iban={features.get('bnf_needs_iban')}, bnf_has_iban={features.get('bnf_has_iban')}")
            print(f"     bnf_has_account={features.get('bnf_has_account')}, missing_required_iban={features.get('missing_required_iban')}")


if __name__ == "__main__":
    main()
