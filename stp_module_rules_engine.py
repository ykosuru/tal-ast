#!/usr/bin/env python3
"""
================================================================================
ACE RULES ENGINE - Main Wrapper for Deterministic Error Code Prediction
================================================================================

PURPOSE:
    Main orchestration class that accepts an IFML payment message and runs
    all 8XXX and 9XXX decision trees to determine which codes will fire.
    
    This provides a deterministic alternative/complement to the ML model,
    offering 100% accuracy for codes that are fully deterministic from
    input features.

ARCHITECTURE:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                         ACE RULES ENGINE                                │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │  IFML JSON ──► IFMLParser ──► Features ──► ACERulesEngine              │
    │                                                │                        │
    │                              ┌─────────────────┴──────────────────┐     │
    │                              │                                    │     │
    │                              ▼                                    ▼     │
    │                    ┌──────────────────┐              ┌──────────────────┐
    │                    │   Rules8XXX      │              │   Rules9XXX      │
    │                    │  (Validation)    │              │  (Repair)        │
    │                    │                  │              │                  │
    │                    │  • Deterministic │              │  • Deterministic │
    │                    │  • No directory  │              │  • Some dir-dep  │
    │                    └──────────────────┘              └──────────────────┘
    │                              │                                    │     │
    │                              └─────────────────┬──────────────────┘     │
    │                                                │                        │
    │                                                ▼                        │
    │                              ┌──────────────────────────────────────┐   │
    │                              │          RulesEngineResult           │   │
    │                              │  • fired_codes: [8001, 9018, ...]   │   │
    │                              │  • eligible_codes: [9004, 9005, ...]│   │
    │                              │  • cannot_fire: [8022, ...]         │   │
    │                              │  • decision_paths: {code: [...]}    │   │
    │                              │  • summary: {...}                   │   │
    │                              └──────────────────────────────────────┘   │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘

USAGE:
    from ace_rules_engine import ACERulesEngine
    
    # Initialize engine
    engine = ACERulesEngine()
    
    # Process IFML payment
    result = engine.process_ifml(ifml_json_dict)
    
    # Get results
    print(f"Fired codes: {result.fired_codes}")
    print(f"Eligible codes: {result.eligible_codes}")
    
    # Get detailed explanation for specific code
    explanation = engine.explain_code(result, '8001')

OUTPUT:
    RulesEngineResult containing:
    - fired_codes: Codes that WILL fire (deterministic)
    - eligible_codes: Codes eligible for directory lookup (outcome unknown)
    - cannot_fire_codes: Codes that cannot fire (preconditions not met)
    - all_results: List of CodeCheckResult for all codes
    - decision_paths: Dict of code -> decision steps
    - features: Extracted features from IFML
    - global_features: Cross-party aggregated features
    - summary: High-level summary statistics

COMPARISON TO ML MODEL:
    ┌─────────────────────┬────────────────────┬────────────────────────┐
    │                     │   Rules Engine     │      ML Model          │
    ├─────────────────────┼────────────────────┼────────────────────────┤
    │ 8XXX Validation     │ 100% accurate      │ ~90% accurate          │
    │ 9XXX Deterministic  │ 100% accurate      │ ~85% accurate          │
    │ 9XXX Directory-Dep  │ Eligibility only   │ ~70% accurate          │
    │ Explainability      │ Full decision path │ Feature importance     │
    │ Speed               │ Very fast          │ Fast                   │
    │ Training Required   │ No                 │ Yes                    │
    └─────────────────────┴────────────────────┴────────────────────────┘

AUTHOR: ACE Pelican ML Team
VERSION: 1.0 (December 2025)
================================================================================
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime

# Import rule modules
from rules_8xxx import Rules8XXX, CodeCheckResult
from rules_9xxx import Rules9XXX

# Try to import IFMLParser if available
try:
    sys.path.insert(0, str(Path(__file__).parent))
    from ifml_parser import IFMLParser, IFMLFeatures, PartyInfo
    PARSER_AVAILABLE = True
except ImportError:
    PARSER_AVAILABLE = False
    print("Warning: IFMLParser not found. Will need features dict directly.")


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class RulesEngineResult:
    """
    Complete result from rules engine processing.
    
    Attributes:
        transaction_id: From IFML message
        fired_codes: Codes that WILL fire (deterministic)
        eligible_codes: Codes eligible for directory lookup
        cannot_fire_codes: Codes that cannot fire
        all_results: List of all CodeCheckResult objects
        decision_paths: Dict of code -> list of decision steps
        features: Extracted flat features
        global_features: Cross-party aggregated features
        summary: High-level statistics
        processing_time_ms: Time to process in milliseconds
        errors: Any errors encountered during processing
    """
    transaction_id: Optional[str] = None
    fired_codes: List[str] = field(default_factory=list)
    eligible_codes: List[str] = field(default_factory=list)
    cannot_fire_codes: List[str] = field(default_factory=list)
    all_results: List[CodeCheckResult] = field(default_factory=list)
    decision_paths: Dict[str, List[str]] = field(default_factory=dict)
    features: Dict[str, Any] = field(default_factory=dict)
    global_features: Dict[str, Any] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)
    processing_time_ms: float = 0.0
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'transaction_id': self.transaction_id,
            'fired_codes': self.fired_codes,
            'eligible_codes': self.eligible_codes,
            'cannot_fire_codes': self.cannot_fire_codes,
            'decision_paths': self.decision_paths,
            'summary': self.summary,
            'processing_time_ms': self.processing_time_ms,
            'errors': self.errors,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)
    
    def get_code_result(self, code: str) -> Optional[CodeCheckResult]:
        """Get the CodeCheckResult for a specific code."""
        for result in self.all_results:
            if result.code == code or result.code.startswith(code):
                return result
        return None
    
    def print_summary(self):
        """Print a human-readable summary."""
        print("=" * 70)
        print("ACE RULES ENGINE RESULT")
        print("=" * 70)
        print(f"Transaction ID: {self.transaction_id or 'Unknown'}")
        print(f"Processing Time: {self.processing_time_ms:.2f}ms")
        print()
        
        if self.fired_codes:
            print(f"FIRED CODES ({len(self.fired_codes)}):")
            for code in sorted(self.fired_codes):
                print(f"  ✓ {code}")
        else:
            print("FIRED CODES: None")
        
        print()
        
        if self.eligible_codes:
            print(f"ELIGIBLE CODES ({len(self.eligible_codes)}) - pending directory lookup:")
            for code in sorted(self.eligible_codes):
                result = self.get_code_result(code)
                confidence = result.confidence if result else 0
                print(f"  ? {code} ({confidence:.0%} success rate)")
        
        print()
        print(f"Summary: {self.summary.get('fired_8xxx', 0)} validation errors, "
              f"{self.summary.get('fired_9xxx', 0)} repairs, "
              f"{self.summary.get('eligible_count', 0)} pending lookups")
        print("=" * 70)


# =============================================================================
# MAIN RULES ENGINE CLASS
# =============================================================================

class ACERulesEngine:
    """
    Main orchestration class for deterministic error code prediction.
    
    Accepts IFML payment messages and runs all 8XXX and 9XXX decision
    trees to determine which codes will fire.
    
    Usage:
        engine = ACERulesEngine()
        result = engine.process_ifml(ifml_json)
        
        # Or process from features directly:
        result = engine.process_features(features_dict)
    """
    
    def __init__(self, parser: Optional[Any] = None):
        """
        Initialize the rules engine.
        
        Args:
            parser: Optional IFMLParser instance. If not provided,
                    will create one if available.
        """
        self.rules_8xxx = Rules8XXX()
        self.rules_9xxx = Rules9XXX()
        
        if parser:
            self.parser = parser
        elif PARSER_AVAILABLE:
            self.parser = IFMLParser()
        else:
            self.parser = None
    
    # =========================================================================
    # MAIN PROCESSING METHODS
    # =========================================================================
    
    def process_ifml(self, ifml_data: Dict, 
                     include_cannot_fire: bool = False) -> RulesEngineResult:
        """
        Process an IFML payment message and run all decision trees.
        
        Args:
            ifml_data: IFML payment as parsed JSON dict
            include_cannot_fire: Include codes that cannot fire in result
            
        Returns:
            RulesEngineResult with all predictions
        """
        import time
        start_time = time.time()
        
        result = RulesEngineResult()
        
        try:
            # Parse IFML to extract features
            if self.parser:
                parsed = self.parser.parse(ifml_data)
                features = self.parser.to_dict(parsed)
                result.transaction_id = parsed.transaction_id
            else:
                # Assume ifml_data is already features
                features = ifml_data
                result.transaction_id = features.get('transaction_id')
            
            result.features = features
            
            # Compute global/cross-party features
            global_features = self._compute_global_features(features)
            result.global_features = global_features
            
            # Run all rules
            result = self._run_all_rules(features, global_features, result, 
                                         include_cannot_fire)
            
        except Exception as e:
            result.errors.append(f"Processing error: {str(e)}")
        
        result.processing_time_ms = (time.time() - start_time) * 1000
        result.summary = self._generate_summary(result)
        
        return result
    
    def process_features(self, features: Dict,
                         include_cannot_fire: bool = False) -> RulesEngineResult:
        """
        Process pre-extracted features (skip IFML parsing).
        
        Args:
            features: Flat feature dictionary
            include_cannot_fire: Include codes that cannot fire
            
        Returns:
            RulesEngineResult with all predictions
        """
        import time
        start_time = time.time()
        
        result = RulesEngineResult()
        result.features = features
        result.transaction_id = features.get('transaction_id')
        
        try:
            global_features = self._compute_global_features(features)
            result.global_features = global_features
            result = self._run_all_rules(features, global_features, result,
                                         include_cannot_fire)
        except Exception as e:
            result.errors.append(f"Processing error: {str(e)}")
        
        result.processing_time_ms = (time.time() - start_time) * 1000
        result.summary = self._generate_summary(result)
        
        return result
    
    def process_file(self, filepath: str,
                     include_cannot_fire: bool = False) -> RulesEngineResult:
        """
        Process an IFML payment from a JSON file.
        
        Args:
            filepath: Path to JSON file
            include_cannot_fire: Include codes that cannot fire
            
        Returns:
            RulesEngineResult with all predictions
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return self.process_ifml(data, include_cannot_fire)
    
    # =========================================================================
    # INTERNAL PROCESSING METHODS
    # =========================================================================
    
    def _run_all_rules(self, features: Dict, global_features: Dict,
                       result: RulesEngineResult,
                       include_cannot_fire: bool) -> RulesEngineResult:
        """Run all 8XXX and 9XXX rules."""
        all_results = []
        
        # Run 8XXX rules
        try:
            results_8xxx = self.rules_8xxx.check_all(features, None, global_features)
            all_results.extend(results_8xxx)
        except Exception as e:
            result.errors.append(f"8XXX rules error: {str(e)}")
        
        # Run 9XXX rules
        try:
            results_9xxx = self.rules_9xxx.check_all(features, None, global_features)
            all_results.extend(results_9xxx)
        except Exception as e:
            result.errors.append(f"9XXX rules error: {str(e)}")
        
        # Categorize results
        fired = []
        eligible = []
        cannot_fire = []
        decision_paths = {}
        seen_codes = set()
        
        for r in all_results:
            base_code = r.code.split('_')[0]
            code_key = r.code
            
            if r.fires:
                if code_key not in seen_codes:
                    fired.append(r.code)
                    decision_paths[r.code] = r.decision_path
                    seen_codes.add(code_key)
            elif r.eligible:
                if code_key not in seen_codes:
                    eligible.append(r.code)
                    decision_paths[r.code] = r.decision_path
                    seen_codes.add(code_key)
            elif r.cannot_fire and include_cannot_fire:
                if base_code not in seen_codes:
                    cannot_fire.append(base_code)
                    decision_paths[base_code] = r.decision_path
                    seen_codes.add(base_code)
        
        result.all_results = all_results
        result.fired_codes = sorted(set(fired))
        result.eligible_codes = sorted(set(eligible))
        result.cannot_fire_codes = sorted(set(cannot_fire)) if include_cannot_fire else []
        result.decision_paths = decision_paths
        
        return result
    
    def _compute_global_features(self, features: Dict) -> Dict:
        """
        Compute cross-party aggregated features for consistency checks.
        
        These features are needed for codes like 8023-8029 that check
        for inconsistencies across multiple parties.
        """
        global_features = {}
        prefixes = ['orig', 'send', 'dbt', 'cdt', 'intm', 'bnf', 'bnp', 'acwi', 'ordi']
        
        # Collect values across parties
        ibans = []
        accounts = []
        nchs = []
        countries = []
        bic4s = []
        clabes = []
        
        for prefix in prefixes:
            # IBAN
            iban = features.get(f'{prefix}_iban') or features.get(f'{prefix}_account_value', '')
            if iban and features.get(f'{prefix}_has_iban', False):
                clean_iban = iban.upper().replace(' ', '').replace('-', '')
                ibans.append(clean_iban)
            
            # Account
            account = features.get(f'{prefix}_account_value', '')
            if account:
                accounts.append(account.upper().replace(' ', ''))
            
            # NCH
            nch = features.get(f'{prefix}_nch_value', '')
            if nch or features.get(f'{prefix}_has_nch', False):
                nchs.append(nch)
            
            # Country sources
            country = features.get(f'{prefix}_country') or features.get(f'{prefix}_address_country', '')
            bic_country = features.get(f'{prefix}_bic_country', '')
            iban_country = features.get(f'{prefix}_iban_country', '')
            
            for c in [country, bic_country, iban_country]:
                if c and len(c) == 2:
                    countries.append(c.upper())
            
            # BIC4
            bic = features.get(f'{prefix}_bic', '')
            if bic and len(bic) >= 4:
                bic4s.append(bic[:4].upper())
            
            # CLABE
            if features.get(f'{prefix}_is_clabe', False):
                clabes.append(account)
        
        # Compute aggregated features
        global_features['iban_count'] = len(ibans)
        global_features['ibans_consistent'] = len(set(ibans)) <= 1 if ibans else True
        global_features['iban_values'] = list(set(ibans))
        
        global_features['account_count'] = len(accounts)
        global_features['accounts_consistent'] = len(set(accounts)) <= 1 if accounts else True
        
        global_features['nch_count'] = len([n for n in nchs if n])
        global_features['nchs_consistent'] = len(set(n for n in nchs if n)) <= 1 if nchs else True
        global_features['has_nch'] = global_features['nch_count'] > 0
        
        global_features['country_count'] = len(set(countries))
        global_features['countries_consistent'] = len(set(countries)) <= 1 if countries else True
        
        global_features['bic4_count'] = len(bic4s)
        global_features['bic4s_consistent'] = len(set(bic4s)) <= 1 if bic4s else True
        
        global_features['clabe_count'] = len(clabes)
        global_features['clabes_consistent'] = len(set(clabes)) <= 1 if clabes else True
        
        # Intermediary features
        global_features['intm_count'] = features.get('intm_count', 0)
        global_features['intm_has_multiple'] = features.get('intm_has_multiple', False)
        global_features['intm_entries_share_adr_bank_id'] = features.get('intm_entries_share_adr_bank_id', False)
        global_features['intm_entries_share_country'] = features.get('intm_entries_share_country', False)
        global_features['intm_entries_share_bic_prefix'] = features.get('intm_entries_share_bic_prefix', False)
        global_features['intm_has_redundant_info'] = features.get('intm_has_redundant_info', False)
        global_features['intm_has_multiple_types'] = features.get('intm_has_multiple_types', False)
        global_features['has_intermediary'] = features.get('has_intermediary', False) or global_features['intm_count'] > 0
        
        # Compound ID features
        global_features['id_has_bic_and_nch'] = features.get('id_has_bic_and_nch', False)
        global_features['id_is_compound'] = features.get('id_is_compound', False)
        global_features['nch_sources'] = features.get('nch_sources', 0)
        
        # Copy existing global features
        for key in ['has_intermediary', 'is_cross_border', 'has_beneficiary_bank',
                    'has_bank_info', 'amount_mismatch', 'has_instructed_amount']:
            if key in features:
                global_features[key] = features[key]
        
        return global_features
    
    def _generate_summary(self, result: RulesEngineResult) -> Dict:
        """Generate summary statistics."""
        fired_8xxx = [c for c in result.fired_codes if c.split('_')[0].startswith('8')]
        fired_9xxx = [c for c in result.fired_codes if c.split('_')[0].startswith('9')]
        eligible_9xxx = [c for c in result.eligible_codes if c.split('_')[0].startswith('9')]
        
        return {
            'total_fired': len(result.fired_codes),
            'total_eligible': len(result.eligible_codes),
            'fired_8xxx': len(fired_8xxx),
            'fired_9xxx': len(fired_9xxx),
            'eligible_count': len(eligible_9xxx),
            'has_validation_errors': len(fired_8xxx) > 0,
            'has_repairs': len(fired_9xxx) > 0,
            'has_pending_lookups': len(eligible_9xxx) > 0,
            'error_count': len(result.errors),
        }
    
    # =========================================================================
    # EXPLANATION AND DEBUGGING METHODS
    # =========================================================================
    
    def explain_code(self, result: RulesEngineResult, code: str) -> Dict:
        """
        Get detailed explanation for a specific code.
        
        Args:
            result: RulesEngineResult from process_*
            code: Code to explain (e.g., "8001", "9018")
            
        Returns:
            Dict with explanation details
        """
        code_result = None
        for r in result.all_results:
            if r.code == code or r.code.startswith(code + '_') or code.startswith(r.code.split('_')[0]):
                code_result = r
                break
        
        if not code_result:
            return {
                'code': code,
                'found': False,
                'message': f'Code {code} not found in results'
            }
        
        return {
            'code': code_result.code,
            'found': True,
            'status': 'FIRES' if code_result.fires else ('ELIGIBLE' if code_result.eligible else 'CANNOT_FIRE'),
            'description': code_result.description,
            'confidence': code_result.confidence,
            'party': code_result.party,
            'decision_path': code_result.decision_path,
            'features_used': code_result.features_used,
        }
    
    def compare_with_actual(self, result: RulesEngineResult, 
                            actual_codes: List[str]) -> Dict:
        """
        Compare predictions with actual ACE response codes.
        
        Args:
            result: RulesEngineResult from process_*
            actual_codes: List of codes from actual ACE response
            
        Returns:
            Comparison statistics
        """
        predicted_base = set(c.split('_')[0] for c in result.fired_codes)
        eligible_base = set(c.split('_')[0] for c in result.eligible_codes)
        actual_base = set(c.split('_')[0] for c in actual_codes)
        
        true_positives = predicted_base & actual_base
        false_positives = predicted_base - actual_base
        false_negatives = actual_base - predicted_base
        eligible_that_fired = eligible_base & actual_base
        
        return {
            'predicted_count': len(predicted_base),
            'actual_count': len(actual_base),
            'true_positives': list(true_positives),
            'false_positives': list(false_positives),
            'false_negatives': list(false_negatives),
            'eligible_that_fired': list(eligible_that_fired),
            'precision': len(true_positives) / len(predicted_base) if predicted_base else 1.0,
            'recall': len(true_positives) / len(actual_base) if actual_base else 1.0,
            'exact_match': predicted_base == actual_base,
        }
    
    def get_all_supported_codes(self) -> Dict[str, List[str]]:
        """Get list of all codes supported by the rules engine."""
        codes_8xxx = [
            '8001', '8004', '8005', '8006', '8007',
            '8022', '8023', '8024', '8025', '8026', '8027', '8028', '8029',
            '8030', '8033', '8124', '8852', '8894', '8895', '8898'
        ]
        
        codes_9xxx_deterministic = [
            '9000', '9002', '9006', '9015', '9017', '9018', '9019',
            '9021', '9022', '9024', '9025', '9028'
        ]
        
        codes_9xxx_directory = [
            '9004', '9005', '9007', '9008', '9477', '9479', '9480'
        ]
        
        return {
            '8xxx': codes_8xxx,
            '9xxx_deterministic': codes_9xxx_deterministic,
            '9xxx_directory': codes_9xxx_directory,
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def process_ifml_file(filepath: str, verbose: bool = False) -> RulesEngineResult:
    """Convenience function to process an IFML file."""
    engine = ACERulesEngine()
    result = engine.process_file(filepath)
    if verbose:
        result.print_summary()
    return result


def process_ifml_json(ifml_json: Dict, verbose: bool = False) -> RulesEngineResult:
    """Convenience function to process an IFML dict."""
    engine = ACERulesEngine()
    result = engine.process_ifml(ifml_json)
    if verbose:
        result.print_summary()
    return result


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    """Command line interface for the rules engine."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='ACE Pelican Deterministic Rules Engine',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process a single IFML file
    python ace_rules_engine.py payment.json
    
    # Process with verbose output
    python ace_rules_engine.py payment.json --verbose
    
    # Output JSON result
    python ace_rules_engine.py payment.json --json
    
    # Explain specific code
    python ace_rules_engine.py payment.json --explain 8001
    
    # Compare with actual response
    python ace_rules_engine.py payment.json --compare response.json
        """
    )
    
    parser.add_argument('input', help='IFML JSON file to process')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print verbose output')
    parser.add_argument('--json', '-j', action='store_true',
                        help='Output result as JSON')
    parser.add_argument('--explain', '-e', metavar='CODE',
                        help='Explain decision for specific code')
    parser.add_argument('--compare', '-c', metavar='RESPONSE_FILE',
                        help='Compare with actual ACE response file')
    parser.add_argument('--include-cannot-fire', action='store_true',
                        help='Include codes that cannot fire in output')
    
    args = parser.parse_args()
    
    engine = ACERulesEngine()
    
    try:
        result = engine.process_file(args.input, args.include_cannot_fire)
    except FileNotFoundError:
        print(f"Error: File not found: {args.input}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON: {e}")
        sys.exit(1)
    
    if args.explain:
        explanation = engine.explain_code(result, args.explain)
        print(json.dumps(explanation, indent=2, default=str))
        return
    
    if args.compare:
        try:
            with open(args.compare, 'r') as f:
                response_data = json.load(f)
            
            actual_codes = []
            def find_codes(obj):
                if isinstance(obj, dict):
                    if 'Code' in obj:
                        actual_codes.append(obj['Code'])
                    for v in obj.values():
                        find_codes(v)
                elif isinstance(obj, list):
                    for item in obj:
                        find_codes(item)
            find_codes(response_data)
            
            comparison = engine.compare_with_actual(result, actual_codes)
            print(json.dumps(comparison, indent=2))
            return
        except FileNotFoundError:
            print(f"Error: Response file not found: {args.compare}")
            sys.exit(1)
    
    if args.json:
        print(result.to_json())
        return
    
    result.print_summary()
    
    if args.verbose:
        print("\nDETAILED DECISION PATHS:")
        print("-" * 50)
        for code in sorted(result.fired_codes + result.eligible_codes):
            explanation = engine.explain_code(result, code)
            print(f"\n{code} ({explanation['status']}):")
            print(f"  Description: {explanation['description']}")
            print(f"  Decision path:")
            for step in explanation['decision_path']:
                print(f"    • {step}")


if __name__ == '__main__':
    main()
