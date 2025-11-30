"""
Test ACE model predictions against actual responses.

Usage:
    python test_model.py --model-dir ./models_8x --test-dir ./test_data
    python test_model.py --model-dir ./models_8x --test-dir ./test_data --series 8
"""

import json
import argparse
from pathlib import Path
from predictor import ACEPredictor


def extract_codes_from_response(response: dict, composite: bool = True) -> set:
    """Extract codes from response.
    
    Args:
        response: ACE response dict
        composite: If True, extract composite codes (8895_BNFBNK), else base codes (8895)
    """
    codes = set()
    
    def find_codes(obj):
        if isinstance(obj, dict):
            if 'Code' in obj:
                code = obj['Code']
                if code:
                    code_str = str(code).split('_')[0]  # Base code
                    
                    if composite:
                        # Extract party from InformationalData (e.g., "BNFBNK NCH code...")
                        info_data = obj.get('InformationalData', '')
                        if info_data:
                            # Party is usually first word
                            parts = str(info_data).split()
                            if parts:
                                party = parts[0].upper()
                                # Only add party suffix if it looks like a party code
                                if party in ['BNFBNK', 'CDTPTY', 'BNPPTY', 'DBTPTY', 'INTMBNK', 
                                            'ORGPTY', 'SNDBNK', 'CDTBNK', 'DBTBNK']:
                                    codes.add(f"{code_str}_{party}")
                                else:
                                    codes.add(code_str)  # No party found, use base
                            else:
                                codes.add(code_str)
                        else:
                            codes.add(code_str)
                    else:
                        codes.add(code_str)
            
            if 'MsgStatus' in obj:
                status = obj['MsgStatus']
                if isinstance(status, list):
                    for item in status:
                        find_codes(item)
                else:
                    find_codes(status)
            
            for v in obj.values():
                find_codes(v)
                
        elif isinstance(obj, list):
            for item in obj:
                find_codes(item)
    
    find_codes(response)
    return codes


def main():
    parser = argparse.ArgumentParser(description='Test ACE model')
    parser.add_argument('--model-dir', required=True, help='Model directory')
    parser.add_argument('--test-dir', required=True, help='Test data directory')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold')
    parser.add_argument('--series', default=None, help='Code series (e.g., 8 for 8XXX)')
    parser.add_argument('--composite', action='store_true', help='Compare composite codes (8895_BNFBNK) instead of base codes')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model_dir}...")
    predictor = ACEPredictor(args.model_dir)
    
    # Find test files
    test_path = Path(args.test_dir)
    test_files = sorted(test_path.glob('*.json'))
    print(f"Found {len(test_files)} test files")
    print(f"Composite mode: {args.composite}\n")
    
    success_count = 0
    fail_count = 0
    total_count = 0
    fail_examples = []  # Store first few failures for review
    
    for filepath in test_files:
        with open(filepath) as f:
            data = json.load(f)
        
        for txn_id, payment in data.items():
            if not isinstance(payment, dict):
                continue
            
            request = payment.get('Request')
            response = payment.get('Response')
            
            if not request:
                continue
            
            # Get actual codes
            actual_codes = set()
            if response:
                actual_codes = extract_codes_from_response(response, composite=args.composite)
                if args.series:
                    actual_codes = {c for c in actual_codes if c.split('_')[0].startswith(args.series)}
            
            # Predict
            try:
                result = predictor.predict(request, threshold=args.threshold, 
                                          include_explanation=False,
                                          composite_only=args.composite)
                
                if args.composite:
                    # Keep composite codes as-is
                    predicted_codes = set(result.predicted_codes)
                else:
                    # Convert to base codes for comparison
                    predicted_codes = {c.split('_')[0] for c in result.predicted_codes}
                
                if args.series:
                    predicted_codes = {c for c in predicted_codes if c.split('_')[0].startswith(args.series)}
            except Exception as e:
                fail_count += 1
                total_count += 1
                if len(fail_examples) < 10:
                    fail_examples.append(f"{txn_id}: ERROR - {e}")
                continue
            
            # Compare
            total_count += 1
            if predicted_codes == actual_codes:
                success_count += 1
            else:
                fail_count += 1
                if len(fail_examples) < 10:
                    fail_examples.append(f"{txn_id}: predicted {sorted(predicted_codes)}, actual {sorted(actual_codes)}")
            
            # Progress every 1000
            if total_count % 1000 == 0:
                pct = success_count / total_count * 100 if total_count > 0 else 0
                print(f"Progress: {total_count} processed, {success_count} success, {fail_count} fail ({pct:.1f}%)")
    
    # Summary
    print("\n" + "=" * 50)
    print(f"SUCCESS: {success_count}")
    print(f"FAIL: {fail_count}")
    print(f"Total: {total_count}")
    if total_count > 0:
        print(f"Accuracy: {success_count/total_count*100:.1f}%")
    
    if fail_examples:
        print("\nFirst failures:")
        for ex in fail_examples:
            print(f"  {ex}")


if __name__ == '__main__':
    main()
