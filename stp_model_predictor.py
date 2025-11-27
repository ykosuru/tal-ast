"""
Inference Pipeline for ACE Error Code Prediction.
Production-ready predictor with explanation capabilities.
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import pickle

from ifml_parser import IFMLParser
from feature_engineering import IFMLFeatureEngineer
from model_training import ACEErrorCodeModel, RareCodeDetector


@dataclass
class PredictionResult:
    """Result of a single prediction."""
    transaction_id: Optional[str]
    predicted_codes: List[str]
    probabilities: Dict[str, float]
    confidence: float
    explanations: List[Dict[str, Any]]
    warnings: List[str]


class ACEPredictor:
    """
    Production predictor for ACE error codes.
    Loads trained models and provides prediction with explanations.
    """
    
    def __init__(self, model_dir: str):
        """
        Initialize predictor by loading models from directory.
        
        Args:
            model_dir: Directory containing saved models and encoders
        """
        self.model_dir = Path(model_dir)
        self.parser = IFMLParser()
        self.feature_engineer = IFMLFeatureEngineer()
        self.model: Optional[ACEErrorCodeModel] = None
        self.rare_detectors: Dict[str, RareCodeDetector] = {}
        self.label_mapping: Dict[int, str] = {}
        self.threshold: float = 0.5
        self.loaded = False
        self.exclusion_rules: Dict = {}  # Loaded from exclusion_rules.json
        self.prediction_config: Dict = {  # Defaults, can be overridden by prediction_config.json
            'negative_margin': 0.1,  # Code must beat __NO_XXXX__ by this margin
            'suppressed_codes': [],  # Codes to always suppress
            'high_threshold_codes': {},  # code -> threshold for codes needing higher confidence
        }
        
        self._load_models()
    
    def _load_models(self):
        """Load all model components."""
        # Load feature engineer
        fe_path = self.model_dir / 'feature_engineer.pkl'
        if fe_path.exists():
            self.feature_engineer.load(str(fe_path))
        else:
            raise FileNotFoundError(f"Feature engineer not found at {fe_path}")
        
        # Load main model
        model_path = self.model_dir / 'model.pkl'
        if model_path.exists():
            self.model = ACEErrorCodeModel()
            self.model.load(str(model_path))
        else:
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        # Load label mapping
        label_path = self.model_dir / 'label_encoder.pkl'
        if label_path.exists():
            with open(label_path, 'rb') as f:
                label_data = pickle.load(f)
                self.label_mapping = label_data.get('idx_to_code', {})
        
        # Load rare code detectors if present
        rare_dir = self.model_dir / 'rare_detectors'
        if rare_dir.exists():
            for detector_file in rare_dir.glob('*.pkl'):
                code = detector_file.stem
                with open(detector_file, 'rb') as f:
                    self.rare_detectors[code] = pickle.load(f)
        
        # Load exclusion rules if present
        rules_path = self.model_dir / 'exclusion_rules.json'
        if rules_path.exists():
            with open(rules_path, 'r') as f:
                self.exclusion_rules = json.load(f)
        
        # Load prediction config if present (allows tuning without code changes)
        config_path = self.model_dir / 'prediction_config.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
                self.prediction_config.update(loaded_config)
        
        self.loaded = True
    
    def predict(self, ifml_json: dict, 
                threshold: Optional[float] = None,
                include_explanation: bool = True) -> PredictionResult:
        """
        Predict error codes for a single IFML request.
        
        Args:
            ifml_json: IFML request as JSON dict
            threshold: Probability threshold for positive prediction (default 0.5)
            include_explanation: Whether to include feature explanations
        
        Returns:
            PredictionResult with predictions and explanations
        """
        if not self.loaded:
            raise RuntimeError("Models not loaded")
        
        threshold = threshold or self.threshold
        warnings = []
        
        # Parse IFML
        try:
            features = self.parser.parse(ifml_json)
            feature_dict = self.parser.to_dict(features)
        except Exception as e:
            return PredictionResult(
                transaction_id=None,
                predicted_codes=[],
                probabilities={},
                confidence=0.0,
                explanations=[],
                warnings=[f"Failed to parse IFML: {str(e)}"]
            )
        
        # Extract transaction ID
        txn_id = feature_dict.get('transaction_id')
        
        # Transform features
        X_raw = pd.DataFrame([feature_dict])
        try:
            X_transformed = self.feature_engineer.transform(X_raw)
            X = X_transformed.values
        except Exception as e:
            warnings.append(f"Feature transformation warning: {str(e)}")
            # Try with default values
            X = np.zeros((1, len(self.model.feature_names)))
        
        # Get predictions
        probabilities = self.model.predict_proba(X)[0]
        
        # Convert to code probabilities
        code_probs = {}
        predicted_codes = []
        no_series_probs = {}  # Track __NO_XXXX__ probabilities
        
        if isinstance(probabilities, np.ndarray):
            for i, prob in enumerate(probabilities):
                if i < len(self.model.class_names):
                    code = self.model.class_names[i]
                    code_probs[code] = float(prob)
                    
                    # Track __NO_XXXX__ class probabilities (e.g., __NO_8XXX__)
                    if code.startswith('__NO_') and 'XXX__' in code:
                        # Extract series (e.g., '8' from '__NO_8XXX__')
                        series = code.replace('__NO_', '').replace('XXX__', '').replace('_', '')
                        no_series_probs[series] = float(prob)
                        continue
                    
                    # Skip other internal classes
                    if code.startswith('__'):
                        continue
                    
                    if prob >= threshold:
                        predicted_codes.append(code)
        
        # Apply prediction config filters
        suppressed_codes = set(self.prediction_config.get('suppressed_codes', []))
        high_threshold_codes = self.prediction_config.get('high_threshold_codes', {})
        
        # Filter by suppressed codes list
        if suppressed_codes:
            before = len(predicted_codes)
            predicted_codes = [c for c in predicted_codes 
                              if c.split('_')[0] not in suppressed_codes]
            if len(predicted_codes) < before:
                warnings.append(f"Suppressed {before - len(predicted_codes)} codes from suppressed_codes list")
        
        # Filter by high threshold codes (require higher confidence for certain codes)
        if high_threshold_codes:
            filtered = []
            for code in predicted_codes:
                base_code = code.split('_')[0]
                required_threshold = high_threshold_codes.get(base_code)
                if required_threshold and code_probs.get(code, 0) < required_threshold:
                    warnings.append(f"Suppressed {code} (prob {code_probs.get(code, 0):.2f} < required {required_threshold})")
                else:
                    filtered.append(code)
            predicted_codes = filtered
        
        # Strict suppression: code must beat __NO_XXXX__ by margin to be predicted
        # This eliminates false positives when model is uncertain
        if no_series_probs:
            margin = self.prediction_config.get('negative_margin', 0.1)
            filtered_codes = []
            
            for code in predicted_codes:
                base_code = code.split('_')[0]
                # Find which series this code belongs to
                code_series = base_code[0] if base_code and base_code[0].isdigit() else None
                
                if code_series and code_series in no_series_probs:
                    no_error_prob = no_series_probs[code_series]
                    code_prob = code_probs.get(code, 0)
                    
                    # Only keep if code probability beats no-error by margin
                    if code_prob > no_error_prob + margin:
                        filtered_codes.append(code)
                    else:
                        warnings.append(f"Suppressed {code} (prob {code_prob:.2f} <= __NO_{code_series}XXX__ {no_error_prob:.2f} + {margin})")
                else:
                    # No negative class for this series, keep the prediction
                    filtered_codes.append(code)
            
            predicted_codes = filtered_codes
        
        # Check rare code detectors
        for code, detector in self.rare_detectors.items():
            try:
                rare_pred = detector.predict(X)
                if rare_pred[0] == 1:
                    rare_prob = detector.predict_proba(X)[0, 1]
                    if code not in code_probs or rare_prob > code_probs[code]:
                        code_probs[code] = float(rare_prob)
                        if rare_prob >= threshold and code not in predicted_codes:
                            predicted_codes.append(code)
                            warnings.append(f"Rare code {code} detected by anomaly model")
            except Exception as e:
                warnings.append(f"Rare detector for {code} failed: {str(e)}")
        
        # Calculate overall confidence
        if predicted_codes:
            confidence = np.mean([code_probs.get(c, 0) for c in predicted_codes])
        else:
            confidence = code_probs.get('__NO_ERROR__', 1.0)
        
        # Post-filter: Apply mutual exclusivity and semantic rules
        predicted_codes, semantic_warnings = self._apply_mutual_exclusivity(
            predicted_codes, code_probs, feature_dict
        )
        warnings.extend(semantic_warnings)
        
        # Generate explanations
        explanations = []
        if include_explanation and predicted_codes:
            try:
                exp = self.model.explain_prediction(X, sample_idx=0)
                explanations = exp.get('rules', [])
                
                # Add feature values explanation
                if exp.get('top_features'):
                    for feat in exp['top_features'][:3]:
                        explanations.append({
                            'type': 'feature',
                            'feature': feat['feature'],
                            'value': feat['value'],
                            'impact': feat.get('shap_value', 'N/A')
                        })
            except Exception as e:
                warnings.append(f"Explanation generation failed: {str(e)}")
        
        # Sort predicted codes by probability
        predicted_codes = sorted(predicted_codes, 
                                key=lambda c: code_probs.get(c, 0), 
                                reverse=True)
        
        # Filter probabilities to only codes above threshold (exclude internal classes)
        filtered_probs = {k: v for k, v in code_probs.items() 
                        if v >= threshold and not k.startswith('__')}
        
        return PredictionResult(
            transaction_id=txn_id,
            predicted_codes=predicted_codes,
            probabilities=filtered_probs,
            confidence=float(confidence),
            explanations=explanations,
            warnings=warnings
        )
    
    def _apply_mutual_exclusivity(self, 
                                   predicted_codes: List[str], 
                                   code_probs: Dict[str, float],
                                   feature_dict: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """
        Apply mutual exclusivity rules to filter conflicting predictions.
        
        Uses rules from:
        1. Loaded exclusion_rules.json (data-driven)
        2. Built-in rules (semantic knowledge)
        
        Returns:
            Tuple of (filtered_codes, warnings)
        """
        if not predicted_codes:
            return predicted_codes, []
        
        codes_to_remove = set()
        
        # === Apply loaded rules from exclusion_rules.json ===
        if self.exclusion_rules:
            # Apply mutual exclusions (codes that never co-occur)
            for rule in self.exclusion_rules.get('mutual_exclusions', []):
                codes = rule.get('codes', [])
                if len(codes) == 2:
                    c1, c2 = codes
                    if c1 in predicted_codes and c2 in predicted_codes:
                        # Keep the one with higher probability
                        if code_probs.get(c1, 0) >= code_probs.get(c2, 0):
                            codes_to_remove.add(c2)
                        else:
                            codes_to_remove.add(c1)
            
            # Apply same-field conflicts with resolution hints
            for conflict in self.exclusion_rules.get('same_field_conflicts', []):
                codes = conflict.get('codes', [])
                resolution = conflict.get('resolution')
                if len(codes) == 2:
                    c1, c2 = codes
                    if c1 in predicted_codes and c2 in predicted_codes:
                        # Use resolution hint if available
                        resolved = self._resolve_conflict(c1, c2, resolution, 
                                                         feature_dict, code_probs)
                        if resolved:
                            codes_to_remove.add(resolved)
        
        # === Built-in rules for known conflicts ===
        # 8894 (Invalid IBAN) vs 8896 (Invalid Domestic Account)
        builtin_rules = [
            # (code1, code2, feature, keep_code1_if_feature_true)
            ('8894', '8896', 'cdt_is_international', '8894'),
            ('8894_CDTPTY', '8896_CDTPTY', 'cdt_is_international', '8894_CDTPTY'),
            ('8894_BNFPTY', '8896_BNFPTY', 'bnf_is_international', '8894_BNFPTY'),
            ('8894_BNPPTY', '8896_BNPPTY', 'bnf_is_international', '8894_BNPPTY'),
            ('8894_DBTPTY', '8896_DBTPTY', 'dbt_is_international', '8894_DBTPTY'),
            ('8894_ORGPTY', '8896_ORGPTY', 'orig_is_international', '8894_ORGPTY'),
        ]
        
        for code1, code2, feature, keep_if_true in builtin_rules:
            has_code1 = code1 in predicted_codes
            has_code2 = code2 in predicted_codes
            
            if has_code1 and has_code2:
                feature_val = feature_dict.get(feature, False)
                if feature_val:
                    codes_to_remove.add(code2)
                else:
                    codes_to_remove.add(code1)
        
        # Handle base codes without party suffix
        base_8894 = any(c.startswith('8894') for c in predicted_codes)
        base_8896 = any(c.startswith('8896') for c in predicted_codes)
        
        if base_8894 and base_8896:
            is_intl = any(feature_dict.get(f'{p}_is_international', False) 
                         for p in ['cdt', 'bnf', 'dbt', 'orig'])
            
            if is_intl:
                codes_to_remove.update(c for c in predicted_codes if c.startswith('8896'))
            else:
                codes_to_remove.update(c for c in predicted_codes if c.startswith('8894'))
        
        # Filter out codes to remove
        filtered_codes = [c for c in predicted_codes if c not in codes_to_remove]
        
        # Apply semantic validation filters
        filtered_codes, semantic_warnings = self._apply_semantic_filters(filtered_codes, feature_dict)
        
        return filtered_codes, semantic_warnings
    
    def _apply_semantic_filters(self, predicted_codes: List[str], 
                                 feature_dict: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """
        Filter predictions based on semantic validation rules.
        
        Each code has specific conditions under which it makes sense.
        If conditions aren't met, the prediction is likely a false positive.
        
        Returns:
            Tuple of (filtered_codes, warnings)
        """
        if not predicted_codes:
            return predicted_codes, []
        
        warnings = []
        
        # Determine payment characteristics from all parties
        parties = ['cdt', 'bnf', 'dbt', 'orig', 'send', 'intm']
        
        is_domestic = any(feature_dict.get(f'{p}_is_domestic', False) for p in parties)
        is_international = any(feature_dict.get(f'{p}_is_international', False) for p in parties)
        
        # If no address country detected, we can't filter semantically
        has_address_info = is_domestic or is_international
        
        # Check if IBAN exists anywhere
        has_iban = any(
            feature_dict.get(f'{p}_account_type') == 'IBAN' or
            feature_dict.get(f'{p}_iban_country') is not None
            for p in parties
        )
        
        # Check if BIC exists anywhere
        has_bic = any(feature_dict.get(f'{p}_has_bic', False) for p in parties)
        
        # Define semantic rules
        # IBAN-related codes should NOT fire for purely domestic (US-only) payments
        iban_codes = {'8004', '8022', '8894'}  # IBAN cannot be derived, IBAN/BIC mismatch, Invalid IBAN
        
        # Domestic-related codes should NOT fire for international payments
        domestic_codes = {'8895', '8896'}  # Invalid NCH, Invalid Domestic Account
        
        # BIC-related codes need BIC to be present
        bic_codes = {'8001', '8005', '8006'}
        
        filtered_codes = []
        for code in predicted_codes:
            base_code = code.split('_')[0]
            keep = True
            reason = None
            
            # Check IBAN codes - suppress if domestic-only and no IBAN present
            if base_code in iban_codes:
                if has_address_info and is_domestic and not is_international and not has_iban:
                    keep = False
                    reason = f"domestic payment without IBAN"
            
            # Check domestic codes - suppress if international
            elif base_code in domestic_codes:
                if has_address_info and is_international and not is_domestic:
                    keep = False
                    reason = f"international payment"
            
            # Check BIC codes - suppress if no BIC present
            elif base_code in bic_codes:
                if not has_bic:
                    keep = False
                    reason = f"no BIC present"
            
            if keep:
                filtered_codes.append(code)
            else:
                warnings.append(f"Semantic filter: {code} suppressed ({reason})")
        
        return filtered_codes, warnings
    
    def _resolve_conflict(self, code1: str, code2: str, resolution: str,
                          feature_dict: Dict, code_probs: Dict) -> Optional[str]:
        """
        Resolve a conflict between two codes using resolution hint.
        Returns the code to REMOVE, or None if can't resolve.
        """
        if not resolution:
            # No hint - keep higher probability
            if code_probs.get(code1, 0) >= code_probs.get(code2, 0):
                return code2
            return code1
        
        # Parse resolution hint
        # Example: "Use is_international feature: True->8894, False->8896"
        if 'is_international' in resolution:
            # Check any party's is_international
            is_intl = any(feature_dict.get(f'{p}_is_international', False) 
                         for p in ['cdt', 'bnf', 'dbt', 'orig'])
            base1 = code1.split('_')[0]
            base2 = code2.split('_')[0]
            
            if is_intl:
                # International - prefer IBAN errors (8894)
                if '8896' in base2:
                    return code2
                elif '8896' in base1:
                    return code1
            else:
                # Domestic - prefer domestic errors (8896)
                if '8894' in base2:
                    return code2
                elif '8894' in base1:
                    return code1
        
        elif 'bic_length' in resolution:
            # Check BIC length
            bic_len = max(
                feature_dict.get(f'{p}_bic_length', 0) 
                for p in ['cdt', 'bnf', 'dbt', 'orig']
            )
            base1 = code1.split('_')[0]
            base2 = code2.split('_')[0]
            
            if bic_len <= 4:
                # Short BIC - prefer 8005
                if '8001' in base2:
                    return code2
                elif '8001' in base1:
                    return code1
            else:
                # Full BIC - prefer 8001
                if '8005' in base2:
                    return code2
                elif '8005' in base1:
                    return code1
        
        return None
    
    def predict_batch(self, ifml_jsons: List[dict],
                      threshold: Optional[float] = None) -> List[PredictionResult]:
        """
        Predict error codes for multiple IFML requests.
        
        Args:
            ifml_jsons: List of IFML request dicts
            threshold: Probability threshold
        
        Returns:
            List of PredictionResults
        """
        return [
            self.predict(ifml, threshold=threshold, include_explanation=False)
            for ifml in ifml_jsons
        ]
    
    def get_top_risk_codes(self, ifml_json: dict, top_n: int = 5) -> List[Tuple[str, float]]:
        """
        Get top N most likely error codes for a request.
        
        Returns:
            List of (code, probability) tuples sorted by probability
        """
        result = self.predict(ifml_json, threshold=0.0, include_explanation=False)
        
        sorted_probs = sorted(
            result.probabilities.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Filter out internal meta-codes
        filtered = [
            (code, prob) for code, prob in sorted_probs
            if not code.startswith('__')
        ]
        
        return filtered[:top_n]
    
    def analyze_request(self, ifml_json: dict) -> Dict[str, Any]:
        """
        Comprehensive analysis of a payment request.
        Returns detailed breakdown of features and predictions.
        """
        # Parse and extract features
        features = self.parser.parse(ifml_json)
        feature_dict = self.parser.to_dict(features)
        
        # Get prediction
        prediction = self.predict(ifml_json)
        
        # Identify potential issues
        issues = []
        
        # Check for common problem patterns
        if not feature_dict.get('cdt_has_account'):
            issues.append({
                'severity': 'high',
                'issue': 'Credit party missing account information',
                'related_codes': ['8004']
            })
        
        if feature_dict.get('cdt_account_type') == 'BBAN' and feature_dict.get('is_cross_border'):
            issues.append({
                'severity': 'medium',
                'issue': 'Cross-border payment with BBAN (IBAN may be required)',
                'related_codes': ['8004', '7452']
            })
        
        if not feature_dict.get('bnf_has_bic'):
            issues.append({
                'severity': 'medium',
                'issue': 'Beneficiary bank missing BIC',
                'related_codes': ['6007']
            })
        
        if feature_dict.get('amount_mismatch'):
            issues.append({
                'severity': 'medium',
                'issue': 'Amount mismatch between instructed and settlement amounts',
                'related_codes': []
            })
        
        analysis = {
            'transaction_id': feature_dict.get('transaction_id'),
            'summary': {
                'source': feature_dict.get('source_code'),
                'message_type': feature_dict.get('incoming_msg_type'),
                'operation': feature_dict.get('bank_operation_code'),
                'currency': feature_dict.get('primary_currency'),
                'amount': feature_dict.get('primary_amount'),
                'is_cross_border': feature_dict.get('is_cross_border'),
            },
            'parties': {
                'originator': {
                    'present': feature_dict.get('orig_present'),
                    'country': feature_dict.get('originator_country'),
                },
                'credit_party': {
                    'present': feature_dict.get('cdt_present'),
                    'has_account': feature_dict.get('cdt_has_account'),
                    'account_type': feature_dict.get('cdt_account_type'),
                    'country': feature_dict.get('cdt_country'),
                },
                'beneficiary_bank': {
                    'present': feature_dict.get('bnf_present'),
                    'has_bic': feature_dict.get('bnf_has_bic'),
                    'country': feature_dict.get('beneficiary_country'),
                },
                'intermediary': {
                    'present': feature_dict.get('has_intermediary'),
                }
            },
            'predicted_codes': prediction.predicted_codes,
            'all_probabilities': prediction.probabilities,
            'confidence': prediction.confidence,
            'identified_issues': issues,
            'explanations': prediction.explanations,
            'warnings': prediction.warnings
        }
        
        return analysis


class PredictorFactory:
    """Factory for creating and managing predictors."""
    
    _instances: Dict[str, ACEPredictor] = {}
    
    @classmethod
    def get_predictor(cls, model_dir: str) -> ACEPredictor:
        """Get or create a predictor instance."""
        if model_dir not in cls._instances:
            cls._instances[model_dir] = ACEPredictor(model_dir)
        return cls._instances[model_dir]
    
    @classmethod
    def reload_predictor(cls, model_dir: str) -> ACEPredictor:
        """Force reload a predictor."""
        if model_dir in cls._instances:
            del cls._instances[model_dir]
        return cls.get_predictor(model_dir)


def predict_from_file(model_dir: str, ifml_file: str) -> PredictionResult:
    """
    Convenience function to predict from a file.
    
    Args:
        model_dir: Directory containing trained models
        ifml_file: Path to IFML JSON file
    
    Returns:
        PredictionResult
    """
    predictor = PredictorFactory.get_predictor(model_dir)
    
    with open(ifml_file, 'r') as f:
        ifml_json = json.load(f)
    
    return predictor.predict(ifml_json)


def predict_from_string(model_dir: str, ifml_string: str) -> PredictionResult:
    """
    Convenience function to predict from a JSON string.
    
    Args:
        model_dir: Directory containing trained models
        ifml_string: IFML as JSON string
    
    Returns:
        PredictionResult
    """
    predictor = PredictorFactory.get_predictor(model_dir)
    ifml_json = json.loads(ifml_string)
    return predictor.predict(ifml_json)


if __name__ == '__main__':
    print("ACE Error Code Predictor")
    print("=" * 40)
    print("\nUsage:")
    print("  from predictor import ACEPredictor")
    print("  predictor = ACEPredictor('/path/to/models')")
    print("  result = predictor.predict(ifml_json)")
    print("\nOr use convenience functions:")
    print("  from predictor import predict_from_file")
    print("  result = predict_from_file('/path/to/models', 'request.json')")
