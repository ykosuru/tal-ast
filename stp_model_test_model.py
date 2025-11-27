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


def extract_codes_from_response(response: dict) -> set:
    """Extract base codes from response."""
    codes = set()
    
    def find_codes(obj):
        if isinstance(obj, dict):
            if 'Code' in obj:
                code = obj['Code']
                if code:
                    codes.add(str(code).split('_')[0])
            
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
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model_dir}...")
    predictor = ACEPredictor(args.model_dir)
    
    # Find test files
    test_path = Path(args.test_dir)
    test_files = sorted(test_path.glob('*.json'))
    print(f"Found {len(test_files)} test files\n")
    
    success_count = 0
    fail_count = 0
    
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
                actual_codes = extract_codes_from_response(response)
                if args.series:
                    actual_codes = {c for c in actual_codes if c.startswith(args.series)}
            
            # Predict
            try:
                result = predictor.predict(request, threshold=args.threshold, include_explanation=False)
                predicted_codes = {c.split('_')[0] for c in result.predicted_codes}
                if args.series:
                    predicted_codes = {c for c in predicted_codes if c.startswith(args.series)}
            except Exception as e:
                print(f"{filepath.name}, {txn_id}, FAIL (error: {e})")
                fail_count += 1
                continue
            
            # Compare
            if predicted_codes == actual_codes:
                print(f"{txn_id}, SUCCESS")
                success_count += 1
            else:
                print(f"{filepath.name}, {txn_id}, FAIL")
                print(f"  Predicted: {sorted(predicted_codes)}")
                print(f"  Actual:    {sorted(actual_codes)}")
                fail_count += 1
    
    # Summary
    print("\n" + "=" * 50)
    print(f"SUCCESS: {success_count}")
    print(f"FAIL: {fail_count}")
    print(f"Total: {success_count + fail_count}")
    if success_count + fail_count > 0:
        print(f"Accuracy: {success_count/(success_count+fail_count)*100:.1f}%")


if __name__ == '__main__':
    main()
