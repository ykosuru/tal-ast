#!/usr/bin/env python3
"""
================================================================================
ACE RULES ENGINE v2.0
================================================================================

PURPOSE:
    Applies precondition rules to filter ML predictions for ACE Pelican error
    codes. This engine mirrors the filtering logic in predictor.py to determine
    which codes can/cannot fire based on payment features.

ARCHITECTURE:
    The ML model achieves ~90% accuracy by:
    1. Neural network predicts probabilities for all codes
    2. Precondition filter removes impossible predictions
    3. This engine implements step 2

RULE LOGIC:
    - require_true: ALL features must be True for code to be POSSIBLE
    - require_false: At least ONE feature must be False for code to FIRE
    
    Example for 8001 (Invalid BIC):
        require_true: [has_bic]        -> Party must have a BIC
        require_false: [bic_valid_format, bic_valid_country]
                                       -> At least one must be False (invalid)

USAGE:
    from ace_rules_v2 import ACERulesEngine
    from ifml_parser_v2 import IFMLParser
    
    parser = IFMLParser()
    engine = ACERulesEngine('precondition_rules_v2.json')
    
    # From IFML dict
    result = engine.process_ifml(ifml_dict)
    
    # From pre-extracted features
    result = engine.check_all(feature_dict)
    
    # Filter ML predictions
    filtered = engine.filter_predictions(['8001', '8004', '9004'], feature_dict)

AUTHOR: ACE Pelican ML Team
VERSION: 2.0 (December 2025)
================================================================================
"""

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Tuple


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class CodeResult:
    """Result of checking a single code."""
    code: str
    status: str  # 'fires', 'eligible', 'cannot_fire', 'suppressed'
    reason: str
    party: Optional[str] = None
    features_checked: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def code_with_party(self) -> str:
        """Return code with party suffix if applicable."""
        if self.party:
            return f"{self.code}_{self.party}"
        return self.code


@dataclass
class EngineResult:
    """Result of checking all codes for a payment."""
    fired: List[str]
    eligible: List[str]
    cannot_fire: List[str]
    details: List[CodeResult]
    features: Dict[str, Any]
    
    def summary(self) -> str:
        """Return a summary string."""
        return (f"Fired: {len(self.fired)}, "
                f"Eligible: {len(self.eligible)}, "
                f"Cannot Fire: {len(self.cannot_fire)}")


@dataclass 
class ComparisonResult:
    """Result of comparing predictions with actual ACE response."""
    correct: List[str]
    missed: List[str]
    false_positives: List[str]
    precision: float
    recall: float
    f1: float


# =============================================================================
# CONSTANTS
# =============================================================================

# Party suffixes used in ACE responses
PARTY_SUFFIXES = ['ORGPTY', 'SNDBNK', 'DBTPTY', 'CDTPTY', 'INTBNK', 'BNFBNK', 'BNPPTY']

# Map party suffix to feature prefix
SUFFIX_TO_PREFIX = {
    'ORGPTY': 'orig',
    'SNDBNK': 'send', 
    'DBTPTY': 'dbt',
    'CDTPTY': 'cdt',
    'INTBNK': 'intm',
    'BNFBNK': 'bnf',
    'BNPPTY': 'bnp',
    'ACWI': 'acwi',
    'ORDI': 'ordi',
}

# Codes that require directory lookup - can only predict eligibility
DIRECTORY_DEPENDENT = {
    '9004', '9005', '9007', '9008', '9013', '9023', '9032',
    '9475', '9476', '9477', '9478', '9479', '9480', '9481',
    '9485', '9486', '9961', '9985', '9986'
}

# Codes that are unpredictable
SUPPRESSED = {'9999', '9490'}

# Global codes (not party-specific)
GLOBAL_CODES = {'8007', '8023', '8024', '8025', '8026', '8027', '8028', 
                '8029', '8033', '8124', '9018', '9024'}


# =============================================================================
# RULES ENGINE
# =============================================================================

class ACERulesEngine:
    """
    Engine for applying precondition rules to ACE Pelican payments.
    """
    
    def __init__(self, rules_path: str = 'precondition_rules_v2.json'):
        """
        Initialize the rules engine.
        
        Args:
            rules_path: Path to precondition rules JSON file
        """
        self.rules = self._load_rules(rules_path)
        self.codes = [k for k in self.rules.keys() if not k.startswith('_')]
        self._parser = None
    
    def _load_rules(self, path: str) -> Dict:
        """Load rules from JSON file."""
        rules_file = Path(path)
        if not rules_file.exists():
            # Try relative to this file
            rules_file = Path(__file__).parent / path
        if not rules_file.exists():
            raise FileNotFoundError(f"Rules file not found: {path}")
        
        with open(rules_file) as f:
            return json.load(f)
    
    @property
    def parser(self):
        """Lazy load IFML parser."""
        if self._parser is None:
            from ifml_parser_v2 import IFMLParser
            self._parser = IFMLParser()
        return self._parser
    
    # -------------------------------------------------------------------------
    # CORE CHECKING LOGIC
    # -------------------------------------------------------------------------
    
    def check_code(self, code: str, features: Dict[str, Any], 
                   party_suffix: Optional[str] = None) -> CodeResult:
        """
        Check if a single code fires for given features.
        
        Args:
            code: ACE code (e.g., '8001', '9004')
            features: Feature dictionary from IFML parser
            party_suffix: Party suffix (e.g., 'BNFBNK') or None for global codes
            
        Returns:
            CodeResult with status and explanation
        """
        # Handle suppressed codes
        if code in SUPPRESSED:
            return CodeResult(
                code=code,
                status='suppressed',
                reason=f"{code} is unpredictable (generic/directory-only)",
                party=party_suffix
            )
        
        # Get rule definition
        rule = self.rules.get(code)
        if not rule:
            return CodeResult(
                code=code,
                status='unknown',
                reason=f"No rule defined for {code}",
                party=party_suffix
            )
        
        # Determine feature prefix
        is_global = rule.get('global_feature', False) or code in GLOBAL_CODES
        prefix = '' if is_global else SUFFIX_TO_PREFIX.get(party_suffix, '')
        
        # Check preconditions
        require_true = rule.get('require_true', [])
        require_false = rule.get('require_false', [])
        
        status, reason, checked = self._evaluate_conditions(
            features, prefix, require_true, require_false, code
        )
        
        # Directory-dependent codes can only be 'eligible', not 'fires'
        if status == 'fires' and code in DIRECTORY_DEPENDENT:
            status = 'eligible'
            reason = f"{code} requires directory lookup - eligible but not deterministic"
        
        return CodeResult(
            code=code,
            status=status,
            reason=reason,
            party=party_suffix if not is_global else None,
            features_checked=checked
        )
    
    def _evaluate_conditions(self, features: Dict, prefix: str,
                            require_true: List[str], require_false: List[str],
                            code: str) -> Tuple[str, str, Dict]:
        """
        Evaluate require_true and require_false conditions.
        
        Returns:
            Tuple of (status, reason, features_checked)
        """
        checked = {}
        
        # Step 1: Check require_true (ALL must be True)
        for feat_base in require_true:
            feat_name = f"{prefix}_{feat_base}" if prefix else feat_base
            value = features.get(feat_name)
            checked[feat_name] = value
            
            if not (isinstance(value, bool) and value is True):
                return (
                    'cannot_fire',
                    f"{code} cannot fire: {feat_name}={value} (must be True)",
                    checked
                )
        
        # Step 2: Check require_false (at least ONE must be explicitly False)
        if not require_false:
            # No validity checks - code is eligible if require_true passed
            return ('eligible', f"{code} eligible: no validity checks defined", checked)
        
        any_false = False
        all_none = True
        
        for feat_base in require_false:
            feat_name = f"{prefix}_{feat_base}" if prefix else feat_base
            value = features.get(feat_name)
            checked[feat_name] = value
            
            if value is not None:
                all_none = False
            
            if isinstance(value, bool) and value is False:
                any_false = True
                break
        
        if any_false:
            return ('fires', f"{code} fires: validation failed", checked)
        elif all_none:
            return ('eligible', f"{code} eligible: validity features missing", checked)
        else:
            return ('cannot_fire', f"{code} cannot fire: all validations passed", checked)
    
    # -------------------------------------------------------------------------
    # BATCH CHECKING
    # -------------------------------------------------------------------------
    
    def check_all(self, features: Dict[str, Any]) -> EngineResult:
        """
        Check all codes for a payment.
        
        Args:
            features: Feature dictionary from IFML parser
            
        Returns:
            EngineResult with all codes categorized
        """
        fired = []
        eligible = []
        cannot_fire = []
        details = []
        
        for code in self.codes:
            rule = self.rules.get(code, {})
            is_global = rule.get('global_feature', False) or code in GLOBAL_CODES
            
            if is_global:
                # Check once without party suffix
                result = self.check_code(code, features, None)
                details.append(result)
                self._categorize(result, fired, eligible, cannot_fire)
            else:
                # Check for each party
                for suffix in PARTY_SUFFIXES:
                    prefix = SUFFIX_TO_PREFIX.get(suffix)
                    if not prefix:
                        continue
                    
                    # Skip if party not present
                    if not features.get(f"{prefix}_present", False):
                        continue
                    
                    result = self.check_code(code, features, suffix)
                    details.append(result)
                    self._categorize(result, fired, eligible, cannot_fire)
        
        return EngineResult(
            fired=fired,
            eligible=eligible,
            cannot_fire=cannot_fire,
            details=details,
            features=features
        )
    
    def _categorize(self, result: CodeResult, 
                   fired: List, eligible: List, cannot_fire: List):
        """Add result to appropriate category list."""
        code_str = result.code_with_party
        
        if result.status == 'fires':
            fired.append(code_str)
        elif result.status == 'eligible':
            eligible.append(code_str)
        elif result.status in ('cannot_fire', 'suppressed'):
            cannot_fire.append(code_str)
    
    # -------------------------------------------------------------------------
    # ML PREDICTION FILTERING
    # -------------------------------------------------------------------------
    
    def filter_predictions(self, predictions: List[str], 
                          features: Dict[str, Any]) -> List[str]:
        """
        Filter ML predictions using precondition rules.
        
        This is the key function for integrating with the ML model.
        It removes predictions that are logically impossible.
        
        Args:
            predictions: List of predicted codes (e.g., ['8001_BNFBNK', '9004_CDTPTY'])
            features: Feature dictionary from IFML parser
            
        Returns:
            Filtered list of predictions (impossible ones removed)
        """
        filtered = []
        
        for pred in predictions:
            # Parse code and party
            if '_' in pred:
                code, party = pred.rsplit('_', 1)
            else:
                code, party = pred, None
            
            # Check if code can fire
            result = self.check_code(code, features, party)
            
            # Keep only if fires or eligible
            if result.status in ('fires', 'eligible'):
                filtered.append(pred)
        
        return filtered
    
    # -------------------------------------------------------------------------
    # IFML PROCESSING
    # -------------------------------------------------------------------------
    
    def process_ifml(self, ifml: Dict) -> EngineResult:
        """
        Process an IFML payment and check all rules.
        
        Args:
            ifml: IFML payment dictionary
            
        Returns:
            EngineResult with all codes categorized
        """
        parsed = self.parser.parse(ifml)
        features = self.parser.to_dict(parsed)
        return self.check_all(features)
    
    def process_file(self, path: str) -> EngineResult:
        """
        Process an IFML file and check all rules.
        
        Args:
            path: Path to IFML JSON file
            
        Returns:
            EngineResult with all codes categorized
        """
        with open(path) as f:
            ifml = json.load(f)
        return self.process_ifml(ifml)
    
    # -------------------------------------------------------------------------
    # COMPARISON WITH ACTUAL RESULTS
    # -------------------------------------------------------------------------
    
    def compare(self, ifml: Dict, actual_codes: List[str]) -> ComparisonResult:
        """
        Compare engine predictions with actual ACE response codes.
        
        Args:
            ifml: IFML payment dictionary
            actual_codes: List of actual codes from ACE response
            
        Returns:
            ComparisonResult with metrics
        """
        result = self.process_ifml(ifml)
        
        # Predicted = fired + eligible
        predicted = set(result.fired + result.eligible)
        actual = set(actual_codes)
        
        correct = sorted(predicted & actual)
        missed = sorted(actual - predicted)
        false_positives = sorted(predicted - actual)
        
        # Calculate metrics
        tp = len(correct)
        fp = len(false_positives)
        fn = len(missed)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return ComparisonResult(
            correct=correct,
            missed=missed,
            false_positives=false_positives,
            precision=precision,
            recall=recall,
            f1=f1
        )
    
    # -------------------------------------------------------------------------
    # EXPLAINABILITY
    # -------------------------------------------------------------------------
    
    def explain(self, code: str, features: Dict[str, Any],
                party_suffix: Optional[str] = None) -> str:
        """
        Generate a human-readable explanation for a code.
        
        Args:
            code: ACE code to explain
            features: Feature dictionary
            party_suffix: Party suffix (optional)
            
        Returns:
            Explanation string
        """
        rule = self.rules.get(code, {})
        result = self.check_code(code, features, party_suffix)
        
        lines = [
            f"Code: {code}",
            f"Description: {rule.get('description', 'N/A')}",
            f"Status: {result.status.upper()}",
            f"Reason: {result.reason}",
            "",
            "Features Checked:"
        ]
        
        for feat, val in result.features_checked.items():
            lines.append(f"  {feat}: {val}")
        
        return "\n".join(lines)
    
    def list_codes(self, category: str = 'all') -> List[str]:
        """
        List available codes.
        
        Args:
            category: 'all', '8xxx', '9xxx', 'global', or 'party'
            
        Returns:
            List of code strings
        """
        if category == '8xxx':
            return sorted([c for c in self.codes if c.startswith('8')])
        elif category == '9xxx':
            return sorted([c for c in self.codes if c.startswith('9')])
        elif category == 'global':
            return sorted([c for c in self.codes if c in GLOBAL_CODES])
        elif category == 'party':
            return sorted([c for c in self.codes if c not in GLOBAL_CODES])
        else:
            return sorted(self.codes)


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """Command-line interface for the rules engine."""
    import argparse
    
    parser = argparse.ArgumentParser(description='ACE Rules Engine v2')
    parser.add_argument('ifml_file', nargs='?', help='IFML JSON file to process')
    parser.add_argument('--rules', default='precondition_rules_v2.json',
                       help='Path to precondition rules JSON')
    parser.add_argument('--list', choices=['all', '8xxx', '9xxx', 'global', 'party'],
                       help='List available codes')
    parser.add_argument('--explain', help='Explain a specific code')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show detailed output')
    
    args = parser.parse_args()
    
    engine = ACERulesEngine(args.rules)
    
    if args.list:
        codes = engine.list_codes(args.list)
        print(f"\n{args.list.upper()} Codes ({len(codes)}):")
        for code in codes:
            desc = engine.rules.get(code, {}).get('description', '')[:60]
            print(f"  {code}: {desc}...")
        return
    
    if args.ifml_file:
        result = engine.process_file(args.ifml_file)
        
        print(f"\n{result.summary()}")
        print(f"\nFIRED: {sorted(result.fired)}")
        print(f"ELIGIBLE: {sorted(result.eligible)[:10]}..." if len(result.eligible) > 10 
              else f"ELIGIBLE: {sorted(result.eligible)}")
        
        if args.verbose:
            print("\nDETAILS:")
            for detail in result.details:
                if detail.status == 'fires':
                    print(f"  {detail.code_with_party}: {detail.reason}")
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
