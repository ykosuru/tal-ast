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
        
        if isinstance(probabilities, np.ndarray):
            for i, prob in enumerate(probabilities):
                if i < len(self.model.class_names):
                    code = self.model.class_names[i]
                    code_probs[code] = float(prob)
                    
                    if prob >= threshold and code not in ['__NO_ERROR__', '__RARE__']:
                        predicted_codes.append(code)
        
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
        
        # Filter probabilities to only codes above threshold
        filtered_probs = {k: v for k, v in code_probs.items() 
                        if v >= threshold and k not in ['__NO_ERROR__', '__RARE__']}
        
        return PredictionResult(
            transaction_id=txn_id,
            predicted_codes=predicted_codes,
            probabilities=filtered_probs,
            confidence=float(confidence),
            explanations=explanations,
            warnings=warnings
        )
    
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
        
        # Filter out meta-codes
        filtered = [
            (code, prob) for code, prob in sorted_probs
            if code not in ['__NO_ERROR__', '__RARE__']
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
