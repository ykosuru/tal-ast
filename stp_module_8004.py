#!/usr/bin/env python3
"""
Test rules for 8004 - IBAN Cannot Be Derived (V4 - Tighter Rules)

V4 CHANGES:
- REMOVED Rule 2 (has_account but no IBAN) - caused 6021 FPs instead of 8004
- Keep only: needs_iban + missing_required_iban

8004 = "IBAN Cannot Be Derived"
ACE attempts IBAN derivation and FAILS (bank not in directory)

Usage:
    python 8004.py --data-dir /path/to/ifml/data
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

sys.path.insert(0, '/mnt/project')
from data_pipeline import IFMLDataPipeline

TARGET_CODE = '8004'

def check_8004_rules(features: Dict) -> Tuple[bool, List[str]]:
    """
    8004 = IBAN Cannot Be Derived
    
    V4: Tighter rules - removed has_account rule that caused 6021 FPs
    
    Only predict when:
    1. Party NEEDS IBAN (in IBAN country) but doesn't have one
    2. IBAN is missing
    
    REMOVED: has_account but no IBAN (triggers 6021, not 8004)
    """
    reasons = []
    
    def get(feat, default=None):
        return features.get(feat, default)
    
    # -------------------------------------------------------------------------
    # Rule 1: Party needs IBAN but doesn't have one
    # This is the PRIMARY trigger for IBAN derivation
    # -------------------------------------------------------------------------
    for prefix in ['bnf_', 'cdt_', 'dbt_', 'orig_']:
        if get(f'{prefix}needs_iban', False) and not get(f'{prefix}has_iban', False):
            party = prefix.rstrip('_').upper()
            reasons.append(f"{party}: needs_iban but no IBAN")
    
    # -------------------------------------------------------------------------
    # Rule 2: System flag for missing required IBAN
    # -------------------------------------------------------------------------
    if get('missing_required_iban', False):
        reasons.append("missing_required_iban=True")
    
    # -------------------------------------------------------------------------
    # REMOVED: has_account but no IBAN
    # This rule was causing FPs where 6021 fired instead of 8004
    # 6021 = different validation (not IBAN derivation failure)
    # -------------------------------------------------------------------------
    
    return len(reasons) > 0, reasons


def get_debug_features(features: Dict) -> Dict:
    """Extract key features for debugging."""
    return {
        'bnf_needs_iban': features.get('bnf_needs_iban'),
        'bnf_has_iban': features.get('bnf_has_iban'),
        'bnf_has_account': features.get('bnf_has_account'),
        'cdt_needs_iban': features.get('cdt_needs_iban'),
        'cdt_has_iban': features.get('cdt_has_iban'),
        'cdt_has_account': features.get('cdt_has_account'),
        'dbt_needs_iban': features.get('dbt_needs_iban'),
        'dbt_has_iban': features.get('dbt_has_iban'),
        'missing_required_iban': features.get('missing_required_iban'),
    }


def main():
    parser = argparse.ArgumentParser(description=f'Test {TARGET_CODE} rules V4')
    parser.add_argument('--data-dir', required=True, help='IFML JSON directory')
    parser.add_argument('--limit', type=int, default=0, help='Max records (0=all)')
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
    trigger_counts = defaultdict(int)
    
    limit = args.limit if args.limit > 0 else len(pipeline.records)
    
    for i, record in enumerate(pipeline.records):
        if i >= limit:
            break
        
        txn_id = record.transaction_id
        features = record.request_features
        codes = record.error_codes_only + (record.composite_codes or [])
        
        has_actual = any(TARGET_CODE in str(c) for c in codes)
        predicted, reasons = check_8004_rules(features)
        
        # Track triggers
        for r in reasons:
            trigger = r.split(':')[0] if ':' in r else r
            trigger_counts[trigger] += 1
        
        if predicted and has_actual:
            tp += 1
        elif not predicted and not has_actual:
            tn += 1
        elif predicted and not has_actual:
            fp += 1
            if len(fp_list) < 100:
                fp_list.append((txn_id, reasons, codes[:8], get_debug_features(features)))
        else:
            fn += 1
            if len(fn_list) < 100:
                fn_list.append((txn_id, codes, get_debug_features(features)))
    
    # Metrics
    total = tp + tn + fp + fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Report
    print("\n" + "="*70)
    print(f"TEST RESULTS: {TARGET_CODE} - IBAN Cannot Be Derived (V4 - Tighter)")
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

RULES (V4 - TIGHTER):
  1. needs_iban=True AND has_iban=False
  2. missing_required_iban=True
  
  REMOVED: has_account AND NOT has_iban (caused 6021 FPs)
    """)
    
    # Trigger analysis
    if trigger_counts:
        print("\nPREDICTION TRIGGERS:")
        for trigger, count in sorted(trigger_counts.items(), key=lambda x: -x[1]):
            print(f"  {trigger}: {count}")
    
    # False Positives Analysis
    if fp_list and args.show_fp > 0:
        print("\n" + "="*70)
        print(f"FALSE POSITIVES ({fp} total, showing {min(fp, args.show_fp)}):")
        print("="*70)
        
        # Analyze what codes actually fired
        fp_actual_codes = defaultdict(int)
        for _, _, codes, _ in fp_list:
            for c in codes:
                code_base = str(c).split('_')[0] if '_' in str(c) else str(c)
                fp_actual_codes[code_base] += 1
        
        print("\nWhat codes ACTUALLY fired (instead of 8004):")
        for code, count in sorted(fp_actual_codes.items(), key=lambda x: -x[1])[:10]:
            print(f"  {code}: {count}")
        
        # Analyze which rules triggered FPs
        fp_triggers = defaultdict(int)
        for _, reasons, _, _ in fp_list:
            for r in reasons:
                trigger = r.split(':')[0] if ':' in r else r
                fp_triggers[trigger] += 1
        
        print("\nFP Trigger Analysis:")
        for trigger, count in sorted(fp_triggers.items(), key=lambda x: -x[1]):
            print(f"  {trigger}: {count} ({count/fp*100:.1f}%)")
        
        print("\nSample FPs:")
        for i, (txn, reasons, codes, debug) in enumerate(fp_list[:args.show_fp], 1):
            print(f"  {i}. {txn}")
            print(f"     Trigger: {reasons[0] if reasons else '?'}")
            print(f"     Actual codes: {codes}")
    
    # False Negatives Analysis  
    if fn_list and args.show_fn > 0:
        print("\n" + "="*70)
        print(f"FALSE NEGATIVES ({fn} total, showing {min(fn, args.show_fn)}):")
        print("="*70)
        
        # Analyze FN patterns
        fn_patterns = defaultdict(int)
        for _, _, debug in fn_list:
            for k, v in debug.items():
                if v is True:
                    fn_patterns[f"{k}=True"] += 1
                elif v is False:
                    fn_patterns[f"{k}=False"] += 1
        
        if fn_patterns:
            print("\nPattern Analysis (features in FN cases):")
            for p, c in sorted(fn_patterns.items(), key=lambda x: -x[1])[:10]:
                pct = c / fn * 100 if fn > 0 else 0
                print(f"  {p}: {c} ({pct:.1f}%)")
        
        print("\nSample FNs:")
        for i, (txn, codes, debug) in enumerate(fn_list[:args.show_fn], 1):
            print(f"  {i}. {txn}")
            print(f"     Codes: {[c for c in codes if TARGET_CODE in str(c)]}")
            print(f"     needs_iban: bnf={debug.get('bnf_needs_iban')}, cdt={debug.get('cdt_needs_iban')}, dbt={debug.get('dbt_needs_iban')}")
            print(f"     has_iban: bnf={debug.get('bnf_has_iban')}, cdt={debug.get('cdt_has_iban')}, dbt={debug.get('dbt_has_iban')}")
            print(f"     missing_required_iban={debug.get('missing_required_iban')}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    recall_icon = "✅" if recall >= 0.95 else "⚠️" if recall >= 0.50 else "❌"
    precision_icon = "✅" if precision >= 0.30 else "⚠️" if precision >= 0.10 else "❌"
    
    print(f"""
    Recall:    {recall*100:>6.1f}%  {recall_icon}
    Precision: {precision*100:>6.1f}%  {precision_icon}
    F1 Score:  {f1*100:>6.1f}%
    
    V4 Changes:
    - REMOVED: has_account but no IBAN (was causing 6021 FPs)
    - KEPT: needs_iban but no IBAN
    - KEPT: missing_required_iban flag
    """)


if __name__ == "__main__":
    main()
