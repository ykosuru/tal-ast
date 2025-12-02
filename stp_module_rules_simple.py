"""
ACE Pelican Simple Rules Engine

This rules engine directly applies the precondition_rules.json logic to determine
which codes MUST fire, MIGHT fire, or CANNOT fire based on the features.

The logic matches exactly what predictor.py does in _apply_precondition_filter:
- require_true: ALL features must be True for the code to be possible
- require_false: At least ONE feature must be False for the error to actually fire

For validation errors (8XXX):
- If require_true is met AND at least one require_false is False -> CODE FIRES
- If require_true is NOT met -> CODE CANNOT FIRE
- If require_true is met but ALL require_false are True -> CODE CANNOT FIRE

For repair codes (9XXX):
- Similar logic but some are directory-dependent (can only predict eligibility)
"""

import json
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path


# Party suffix to prefix mapping (matches predictor.py)
SUFFIX_TO_PREFIX = {
    'ORGPTY': 'orig',
    'SNDBNK': 'send', 
    'DBTPTY': 'dbt',
    'CDTPTY': 'cdt',
    'INTBNK': 'intm',
    'INTMBNK': 'intm',
    'BNFBNK': 'bnf',
    'BNPPTY': 'bnp',
    'ACWI': 'acwi',
    'ORDI': 'ordi',
}

ALL_PREFIXES = ['orig', 'send', 'dbt', 'cdt', 'intm', 'bnf', 'bnp', 'ordi', 'acwi']

# Codes that are directory-dependent (we can only predict eligibility)
DIRECTORY_DEPENDENT_CODES = {
    '9004', '9005', '9007', '9008', '9013', '9023', '9032',
    '9475', '9476', '9477', '9478', '9479', '9480', '9481',
    '9485', '9486', '9961', '9985', '9986'
}

# Codes to always suppress
SUPPRESSED_CODES = {'9999', '9490'}


@dataclass
class RuleResult:
    """Result of applying rules to a single code."""
    code: str
    status: str  # 'fires', 'eligible', 'cannot_fire', 'suppressed'
    reason: str
    party: Optional[str] = None
    features_checked: Optional[Dict[str, Any]] = None


@dataclass 
class RulesEngineResult:
    """Complete result from rules engine."""
    fired_codes: List[str]
    eligible_codes: List[str]
    cannot_fire_codes: List[str]
    all_results: List[RuleResult]
    feature_dict: Dict[str, Any]


class ACERulesEngine:
    """
    Simple rules engine that applies precondition_rules.json logic directly.
    """
    
    def __init__(self, precondition_rules_path: str = None):
        """
        Initialize with precondition rules.
        
        Args:
            precondition_rules_path: Path to precondition_rules.json
        """
        self.precondition_rules = {}
        
        if precondition_rules_path:
            self.load_rules(precondition_rules_path)
    
    def load_rules(self, path: str):
        """Load precondition rules from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Filter out comment keys
        self.precondition_rules = {
            k: v for k, v in data.items() 
            if not k.startswith('_') and isinstance(v, dict)
        }
    
    def check_code(self, 
                   code: str, 
                   feature_dict: Dict[str, Any],
                   party_suffix: str = None) -> RuleResult:
        """
        Check if a specific code fires based on features.
        
        Args:
            code: The ACE code (e.g., '8001', '8894')
            feature_dict: Flat feature dictionary from ifml_parser.to_dict()
            party_suffix: Optional party suffix (e.g., 'BNFBNK')
        
        Returns:
            RuleResult indicating status
        """
        # Check suppressed codes
        if code in SUPPRESSED_CODES:
            return RuleResult(
                code=code,
                status='suppressed',
                reason='Code is always suppressed (unpredictable)',
                party=party_suffix
            )
        
        # Get precondition rules
        rules = self.precondition_rules.get(code, {})
        require_true = rules.get('require_true', [])
        require_false = rules.get('require_false', [])
        is_global = rules.get('global_feature', False)
        
        if not require_true and not require_false:
            # No rules defined - assume eligible
            return RuleResult(
                code=code,
                status='eligible' if code in DIRECTORY_DEPENDENT_CODES else 'cannot_fire',
                reason='No precondition rules defined',
                party=party_suffix
            )
        
        # Determine prefixes to check
        if is_global:
            # Global features (like intm_*) don't need party prefix
            prefixes_to_check = [None]
        elif party_suffix:
            prefix = SUFFIX_TO_PREFIX.get(party_suffix)
            prefixes_to_check = [prefix] if prefix else ALL_PREFIXES
        else:
            prefixes_to_check = ALL_PREFIXES
        
        # Check conditions for each applicable party
        last_result = None
        for prefix in prefixes_to_check:
            result = self._check_conditions(
                code, feature_dict, require_true, require_false, prefix
            )
            last_result = result
            
            # If fires or eligible, return immediately
            if result.status == 'fires' or result.status == 'eligible':
                return result
            
            # If cannot_fire but require_true was met, that's a definitive result
            # (means all validity checks passed - don't need to check other parties)
            if result.status == 'cannot_fire' and result.features_checked:
                # Check if require_true was actually met
                require_true_met = True
                for feat_base in require_true:
                    feat_name = f'{prefix}_{feat_base}' if prefix else feat_base
                    if feat_name not in result.features_checked:
                        feat_name = feat_base
                    val = result.features_checked.get(feat_name)
                    if not (isinstance(val, bool) and val is True):
                        require_true_met = False
                        break
                
                if require_true_met:
                    # require_true was met but require_false all passed
                    return result
        
        # No party met the conditions - return the last result or generic message
        if last_result and last_result.status == 'cannot_fire':
            return last_result
        
        return RuleResult(
            code=code,
            status='cannot_fire',
            reason=f'No party meets require_true conditions: {require_true}',
            party=party_suffix
        )
    
    def _check_conditions(self,
                          code: str,
                          feature_dict: Dict[str, Any],
                          require_true: List[str],
                          require_false: List[str],
                          prefix: Optional[str]) -> RuleResult:
        """
        Check conditions for a specific party prefix.
        
        Returns RuleResult with appropriate status.
        """
        features_checked = {}
        
        # === Check require_true: ALL must be True ===
        for feat_base in require_true:
            # Try prefixed version first, then base
            if prefix:
                feat_name = f'{prefix}_{feat_base}'
                value = feature_dict.get(feat_name)
                if value is None:
                    value = feature_dict.get(feat_base)
                    feat_name = feat_base
            else:
                feat_name = feat_base
                value = feature_dict.get(feat_base)
            
            features_checked[feat_name] = value
            
            # Must be explicitly True
            if not (isinstance(value, bool) and value is True):
                return RuleResult(
                    code=code,
                    status='cannot_fire',
                    reason=f'require_true not met: {feat_name}={value}',
                    party=prefix,
                    features_checked=features_checked
                )
        
        # require_true is met - now check require_false
        
        # === Check require_false: At least ONE must be False ===
        if not require_false:
            # No validity checks required - if require_true met, code is eligible
            if code in DIRECTORY_DEPENDENT_CODES:
                return RuleResult(
                    code=code,
                    status='eligible',
                    reason='require_true met, no require_false, directory-dependent',
                    party=prefix,
                    features_checked=features_checked
                )
            else:
                # For cleaning codes, if the dirty feature is true, it fires
                # But we need explicit dirty char detection
                return RuleResult(
                    code=code,
                    status='eligible',
                    reason='require_true met, no require_false defined',
                    party=prefix,
                    features_checked=features_checked
                )
        
        # Check if at least one validity check failed
        any_false = False
        all_true_fields = []
        missing_fields = []
        
        for feat_base in require_false:
            if prefix:
                feat_name = f'{prefix}_{feat_base}'
                value = feature_dict.get(feat_name)
                if value is None:
                    value = feature_dict.get(feat_base)
                    feat_name = feat_base
            else:
                feat_name = feat_base
                value = feature_dict.get(feat_base)
            
            features_checked[feat_name] = value
            
            if isinstance(value, bool):
                if value is False:
                    any_false = True
                    break  # Found one False, that's enough
                else:
                    all_true_fields.append(feat_name)
            elif value is None:
                # Feature not present - don't treat as False, treat as unknown
                # This prevents false positives when features aren't computed
                missing_fields.append(feat_name)
        
        if any_false:
            # At least one validity check EXPLICITLY failed - code FIRES
            if code in DIRECTORY_DEPENDENT_CODES:
                return RuleResult(
                    code=code,
                    status='eligible',
                    reason=f'Conditions met, but directory lookup required',
                    party=prefix,
                    features_checked=features_checked
                )
            else:
                return RuleResult(
                    code=code,
                    status='fires',
                    reason=f'require_true met AND at least one require_false is False',
                    party=prefix,
                    features_checked=features_checked
                )
        elif missing_fields and not all_true_fields:
            # All require_false features are missing - cannot determine
            return RuleResult(
                code=code,
                status='eligible',
                reason=f'Cannot determine - missing features: {missing_fields}',
                party=prefix,
                features_checked=features_checked
            )
        else:
            # All validity checks passed (True) - code CANNOT fire
            return RuleResult(
                code=code,
                status='cannot_fire',
                reason=f'All validity checks passed: {all_true_fields}',
                party=prefix,
                features_checked=features_checked
            )
    
    def check_all_codes(self, 
                        feature_dict: Dict[str, Any],
                        codes_to_check: List[str] = None) -> RulesEngineResult:
        """
        Check all codes for a payment.
        
        Args:
            feature_dict: Flat feature dictionary from ifml_parser.to_dict()
            codes_to_check: Optional list of codes to check (default: all defined codes)
        
        Returns:
            RulesEngineResult with categorized codes
        """
        if codes_to_check is None:
            codes_to_check = list(self.precondition_rules.keys())
        
        all_results = []
        fired_codes = []
        eligible_codes = []
        cannot_fire_codes = []
        
        # Determine which parties are present
        present_parties = []
        for prefix in ALL_PREFIXES:
            if feature_dict.get(f'{prefix}_present', False):
                # Map prefix back to suffix
                for suffix, pfx in SUFFIX_TO_PREFIX.items():
                    if pfx == prefix:
                        present_parties.append(suffix)
                        break
        
        # Check each code
        for code in codes_to_check:
            rules = self.precondition_rules.get(code, {})
            is_global = rules.get('global_feature', False)
            
            if is_global:
                # Check once without party suffix
                result = self.check_code(code, feature_dict, None)
                all_results.append(result)
                
                if result.status == 'fires':
                    fired_codes.append(code)
                elif result.status == 'eligible':
                    eligible_codes.append(code)
                else:
                    cannot_fire_codes.append(code)
            else:
                # Check for each present party
                code_fired = False
                code_eligible = False
                
                for party_suffix in present_parties:
                    result = self.check_code(code, feature_dict, party_suffix)
                    
                    # Create composite code
                    composite_code = f'{code}_{party_suffix}'
                    result.code = composite_code
                    all_results.append(result)
                    
                    if result.status == 'fires':
                        fired_codes.append(composite_code)
                        code_fired = True
                    elif result.status == 'eligible':
                        eligible_codes.append(composite_code)
                        code_eligible = True
                    else:
                        cannot_fire_codes.append(composite_code)
                
                # If no parties present, check with all prefixes
                if not present_parties:
                    result = self.check_code(code, feature_dict, None)
                    all_results.append(result)
                    
                    if result.status == 'fires':
                        fired_codes.append(code)
                    elif result.status == 'eligible':
                        eligible_codes.append(code)
                    else:
                        cannot_fire_codes.append(code)
        
        return RulesEngineResult(
            fired_codes=fired_codes,
            eligible_codes=eligible_codes,
            cannot_fire_codes=cannot_fire_codes,
            all_results=all_results,
            feature_dict=feature_dict
        )
    
    def filter_ml_predictions(self,
                              ml_predictions: List[str],
                              feature_dict: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """
        Filter ML predictions using rules - same logic as predictor._apply_precondition_filter.
        
        This is the KEY function - it takes ML predictions and removes impossible ones.
        
        Args:
            ml_predictions: List of codes predicted by ML model
            feature_dict: Feature dictionary
        
        Returns:
            Tuple of (filtered_predictions, removed_predictions)
        """
        filtered = []
        removed = []
        
        for code in ml_predictions:
            # Parse composite code
            parts = code.split('_')
            base_code = parts[0]
            party_suffix = parts[1] if len(parts) > 1 else None
            
            # Check with rules
            result = self.check_code(base_code, feature_dict, party_suffix)
            
            if result.status in ('fires', 'eligible'):
                filtered.append(code)
            else:
                removed.append(f"{code} ({result.reason})")
        
        return filtered, removed
    
    def explain_code(self, code: str, feature_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Explain why a code would or would not fire.
        
        Args:
            code: The ACE code (can be composite like '8001_BNFBNK')
            feature_dict: Feature dictionary
        
        Returns:
            Explanation dictionary
        """
        parts = code.split('_')
        base_code = parts[0]
        party_suffix = parts[1] if len(parts) > 1 else None
        
        rules = self.precondition_rules.get(base_code, {})
        result = self.check_code(base_code, feature_dict, party_suffix)
        
        return {
            'code': code,
            'base_code': base_code,
            'party': party_suffix,
            'status': result.status,
            'reason': result.reason,
            'rules': {
                'require_true': rules.get('require_true', []),
                'require_false': rules.get('require_false', []),
                'description': rules.get('description', 'No description')
            },
            'features_checked': result.features_checked
        }


# =============================================================================
# STANDALONE TESTING
# =============================================================================

if __name__ == '__main__':
    import sys
    
    # Load rules
    engine = ACERulesEngine()
    
    # Try to find precondition_rules.json
    possible_paths = [
        '/mnt/user-data/uploads/precondition_rules.json',
        'precondition_rules.json',
        '../precondition_rules.json'
    ]
    
    for path in possible_paths:
        if Path(path).exists():
            engine.load_rules(path)
            print(f"Loaded rules from {path}")
            break
    else:
        print("WARNING: Could not find precondition_rules.json")
        sys.exit(1)
    
    print(f"Loaded {len(engine.precondition_rules)} code rules")
    print()
    
    # Test with sample features
    test_features = {
        # Beneficiary Bank
        'bnf_present': True,
        'bnf_has_bic': True,
        'bnf_bic_valid_format': True,
        'bnf_bic_valid_country': True,
        'bnf_has_iban': True,
        'bnf_iban_valid_format': True,
        'bnf_iban_checksum_valid': False,  # Invalid checksum!
        'bnf_bic_iban_match': True,
        'bnf_has_nch': True,
        'bnf_nch_valid': True,
        'bnf_fedaba_checksum_valid': True,
        
        # Credit Party
        'cdt_present': True,
        'cdt_has_bic': False,
        'cdt_has_iban': True,
        'cdt_iban_valid_format': True,
        'cdt_iban_checksum_valid': True,
        'cdt_needs_iban': True,
        
        # No intermediaries
        'intm_has_multiple': False,
        'intm_has_redundant_info': False,
    }
    
    print("Testing with sample features...")
    print("=" * 60)
    
    # Check specific codes
    codes_to_test = ['8001', '8004', '8022', '8894', '8895', '8898', '9004', '9005', '9008', '9018']
    
    for code in codes_to_test:
        # Test for BNFBNK
        result = engine.check_code(code, test_features, 'BNFBNK')
        print(f"{code}_BNFBNK: {result.status}")
        print(f"  Reason: {result.reason}")
        
        # Test for CDTPTY
        result = engine.check_code(code, test_features, 'CDTPTY')
        print(f"{code}_CDTPTY: {result.status}")
        print(f"  Reason: {result.reason}")
        print()
    
    print("=" * 60)
    print("Testing check_all_codes...")
    
    result = engine.check_all_codes(test_features)
    print(f"Fired: {result.fired_codes}")
    print(f"Eligible: {result.eligible_codes}")
    print(f"Cannot fire: {len(result.cannot_fire_codes)} codes")
