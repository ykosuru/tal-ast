"""
Model Training for IFML Error Code Prediction.
Handles class imbalance, supports multiple model types, and provides explainability.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from pathlib import Path
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score
)
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, f1_score, accuracy_score
)
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    IsolationForest
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier

# Try to import optional libraries
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("LightGBM not available. Using sklearn alternatives.")

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("SHAP not available. Explainability features limited.")


@dataclass
class ModelConfig:
    """Configuration for model training."""
    model_type: str = 'lightgbm'  # 'lightgbm', 'xgboost', 'random_forest', 'gradient_boost', 'decision_tree'
    task_type: str = 'multiclass'  # 'binary', 'multiclass', 'multilabel'
    handle_imbalance: str = 'class_weight'  # 'class_weight', 'smote', 'undersample', 'none'
    n_estimators: int = 500
    max_depth: int = 6
    learning_rate: float = 0.05
    min_samples_leaf: int = 10
    random_state: int = 42
    cv_folds: int = 5


class RareCodeDetector:
    """
    One-class classifier for detecting rare error codes.
    Used when a code has too few samples for standard classification.
    """
    
    def __init__(self, code: str, contamination: float = 0.05):
        self.code = code
        self.contamination = contamination
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.fitted = False
    
    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None):
        """
        Fit on samples that triggered this code.
        X should contain only positive examples (samples with this code).
        """
        self.feature_names = feature_names or [f'f{i}' for i in range(X.shape[1])]
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict if samples are likely to trigger this code."""
        if not self.fitted:
            raise ValueError("Model not fitted")
        X_scaled = self.scaler.transform(X)
        # -1 for outliers (likely this code), 1 for normal
        predictions = self.model.predict(X_scaled)
        return (predictions == -1).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get anomaly scores as pseudo-probabilities."""
        if not self.fitted:
            raise ValueError("Model not fitted")
        X_scaled = self.scaler.transform(X)
        scores = -self.model.score_samples(X_scaled)  # Higher = more anomalous
        # Normalize to [0, 1]
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
        return np.column_stack([1 - scores, scores])


class RuleExtractor:
    """
    Extract interpretable rules for error codes using decision trees.
    Useful for understanding what triggers each code.
    """
    
    def __init__(self, max_depth: int = 5, min_samples_leaf: int = 10):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.trees: Dict[str, DecisionTreeClassifier] = {}
        self.rules: Dict[str, str] = {}
        self.feature_names: List[str] = []
    
    def fit(self, X: np.ndarray, y_multilabel: np.ndarray,
            code_names: List[str], feature_names: List[str]):
        """
        Fit one decision tree per error code.
        
        Args:
            X: Feature matrix
            y_multilabel: Multi-label target matrix (n_samples, n_codes)
            code_names: Names of codes corresponding to y columns
            feature_names: Names of features
        """
        self.feature_names = feature_names
        
        for i, code in enumerate(code_names):
            y_code = y_multilabel[:, i]
            
            # Skip if no positive samples
            if y_code.sum() == 0:
                continue
            
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                class_weight='balanced',
                random_state=42
            )
            
            tree.fit(X, y_code)
            self.trees[code] = tree
            self.rules[code] = export_text(tree, feature_names=feature_names)
        
        return self
    
    def get_rules(self, code: str) -> str:
        """Get decision rules for a specific code."""
        if code not in self.rules:
            return f"No rules extracted for code {code}"
        return self.rules[code]
    
    def get_top_features(self, code: str, top_n: int = 5) -> List[Tuple[str, float]]:
        """Get most important features for a code."""
        if code not in self.trees:
            return []
        
        tree = self.trees[code]
        importances = tree.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        return [(self.feature_names[i], importances[i]) for i in indices]
    
    def predict_with_explanation(self, X: np.ndarray, code: str) -> Tuple[np.ndarray, List[str]]:
        """Predict and explain using decision path."""
        if code not in self.trees:
            return np.zeros(len(X)), ["No model available"]
        
        tree = self.trees[code]
        predictions = tree.predict(X)
        
        # Get decision paths
        node_indicator = tree.decision_path(X)
        feature = tree.tree_.feature
        threshold = tree.tree_.threshold
        
        explanations = []
        for i in range(len(X)):
            node_indices = node_indicator[i].indices
            path_features = []
            
            for node_id in node_indices:
                if feature[node_id] >= 0:  # Not a leaf
                    feat_name = self.feature_names[feature[node_id]]
                    thresh = threshold[node_id]
                    value = X[i, feature[node_id]]
                    direction = "<=" if value <= thresh else ">"
                    path_features.append(f"{feat_name} {direction} {thresh:.2f}")
            
            explanations.append(" AND ".join(path_features[-3:]))  # Last 3 conditions
        
        return predictions, explanations


class ACEErrorCodeModel:
    """
    Main model class for ACE error code prediction.
    Supports multiple model types and handles class imbalance.
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.class_names: List[str] = []
        self.class_weights: Dict[int, float] = {}
        self.rare_code_detectors: Dict[str, RareCodeDetector] = {}
        self.rule_extractor: Optional[RuleExtractor] = None
        self.fitted = False
        self.metrics: Dict[str, Any] = {}
    
    def fit(self, X: np.ndarray, y: np.ndarray,
            feature_names: Optional[List[str]] = None,
            class_names: Optional[List[str]] = None,
            class_weights: Optional[Dict[int, float]] = None,
            validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        """
        Fit the model.
        
        Args:
            X: Feature matrix
            y: Target (1D for single-label, 2D for multi-label)
            feature_names: Names of features
            class_names: Names of classes
            class_weights: Pre-computed class weights (optional)
            validation_data: Optional (X_val, y_val) for early stopping
        """
        self.feature_names = feature_names or [f'f{i}' for i in range(X.shape[1])]
        self.class_names = class_names or [f'c{i}' for i in range(len(np.unique(y)) if y.ndim == 1 else y.shape[1])]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Compute class weights if not provided
        if class_weights is None and self.config.handle_imbalance == 'class_weight':
            self.class_weights = self._compute_class_weights(y)
        else:
            self.class_weights = class_weights or {}
        
        # Select and train model
        if self.config.task_type == 'multilabel':
            self._fit_multilabel(X_scaled, y, validation_data)
        else:
            self._fit_single_label(X_scaled, y, validation_data)
        
        # Extract rules for interpretability
        if self.config.task_type != 'binary':
            self.rule_extractor = RuleExtractor()
            y_ml = y if y.ndim == 2 else np.eye(len(self.class_names))[y]
            self.rule_extractor.fit(X_scaled, y_ml, self.class_names, self.feature_names)
        
        self.fitted = True
        return self
    
    def _compute_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """Compute balanced class weights."""
        if y.ndim == 2:
            # Multi-label: weight by inverse frequency
            freqs = y.sum(axis=0) / len(y)
            weights = {i: 1.0 / (freq + 0.01) for i, freq in enumerate(freqs)}
            # Normalize
            total = sum(weights.values())
            weights = {k: v / total * len(weights) for k, v in weights.items()}
        else:
            # Single label
            from sklearn.utils.class_weight import compute_class_weight
            classes = np.unique(y)
            weights_array = compute_class_weight('balanced', classes=classes, y=y)
            weights = {c: w for c, w in zip(classes, weights_array)}
        
        return weights
    
    def _fit_single_label(self, X: np.ndarray, y: np.ndarray,
                          validation_data: Optional[Tuple] = None):
        """Fit single-label classification model."""
        
        if self.config.model_type == 'lightgbm' and HAS_LIGHTGBM:
            self.model = lgb.LGBMClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                min_child_samples=self.config.min_samples_leaf,
                class_weight='balanced' if self.config.handle_imbalance == 'class_weight' else None,
                random_state=self.config.random_state,
                verbose=-1
            )
        
        elif self.config.model_type == 'xgboost' and HAS_XGBOOST:
            # Convert weights to sample weights for XGBoost
            sample_weights = np.array([self.class_weights.get(yi, 1.0) for yi in y])
            self.model = xgb.XGBClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                random_state=self.config.random_state,
                use_label_encoder=False,
                eval_metric='mlogloss'
            )
        
        elif self.config.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                min_samples_leaf=self.config.min_samples_leaf,
                class_weight='balanced' if self.config.handle_imbalance == 'class_weight' else None,
                random_state=self.config.random_state,
                n_jobs=-1
            )
        
        elif self.config.model_type == 'gradient_boost':
            self.model = GradientBoostingClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                min_samples_leaf=self.config.min_samples_leaf,
                random_state=self.config.random_state
            )
        
        else:  # decision_tree or fallback
            self.model = DecisionTreeClassifier(
                max_depth=self.config.max_depth,
                min_samples_leaf=self.config.min_samples_leaf,
                class_weight='balanced' if self.config.handle_imbalance == 'class_weight' else None,
                random_state=self.config.random_state
            )
        
        # Fit with optional early stopping
        if validation_data and hasattr(self.model, 'fit') and self.config.model_type in ['lightgbm', 'xgboost']:
            X_val, y_val = validation_data
            X_val_scaled = self.scaler.transform(X_val)
            
            if self.config.model_type == 'lightgbm' and HAS_LIGHTGBM:
                self.model.fit(
                    X, y,
                    eval_set=[(X_val_scaled, y_val)],
                    callbacks=[lgb.early_stopping(50, verbose=False)]
                )
            else:
                self.model.fit(X, y)
        else:
            self.model.fit(X, y)
    
    def _fit_multilabel(self, X: np.ndarray, y: np.ndarray,
                        validation_data: Optional[Tuple] = None):
        """Fit multi-label classification model."""
        
        # Use MultiOutputClassifier wrapper
        if self.config.model_type == 'lightgbm' and HAS_LIGHTGBM:
            base_model = lgb.LGBMClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                min_child_samples=self.config.min_samples_leaf,
                class_weight='balanced',
                random_state=self.config.random_state,
                verbose=-1
            )
        elif self.config.model_type == 'random_forest':
            base_model = RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                min_samples_leaf=self.config.min_samples_leaf,
                class_weight='balanced',
                random_state=self.config.random_state,
                n_jobs=-1
            )
        else:
            base_model = DecisionTreeClassifier(
                max_depth=self.config.max_depth,
                min_samples_leaf=self.config.min_samples_leaf,
                class_weight='balanced',
                random_state=self.config.random_state
            )
        
        self.model = MultiOutputClassifier(base_model, n_jobs=-1)
        self.model.fit(X, y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if not self.fitted:
            raise ValueError("Model not fitted")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.fitted:
            raise ValueError("Model not fitted")
        
        X_scaled = self.scaler.transform(X)
        
        if hasattr(self.model, 'predict_proba'):
            probs = self.model.predict_proba(X_scaled)
            
            # Handle MultiOutputClassifier output
            if isinstance(probs, list):
                # List of arrays, one per class
                probs = np.column_stack([p[:, 1] if p.shape[1] > 1 else p[:, 0] for p in probs])
            
            return probs
        else:
            # Fall back to hard predictions
            return self.predict(X_scaled)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate model performance.
        
        Returns dictionary with various metrics.
        """
        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)
        
        metrics = {
            'accuracy': accuracy_score(y.flatten(), y_pred.flatten()) if y.ndim == 2 else accuracy_score(y, y_pred)
        }
        
        if self.config.task_type == 'multilabel':
            # Per-class metrics with full breakdown
            from sklearn.metrics import precision_score, recall_score
            
            per_class_report = {}
            for i, class_name in enumerate(self.class_names):
                support = int(y[:, i].sum())
                if support > 0:  # Skip empty classes
                    precision = precision_score(y[:, i], y_pred[:, i], zero_division=0)
                    recall = recall_score(y[:, i], y_pred[:, i], zero_division=0)
                    f1 = f1_score(y[:, i], y_pred[:, i], zero_division=0)
                    
                    metrics[f'{class_name}_f1'] = f1
                    per_class_report[class_name] = {
                        'precision': precision,
                        'recall': recall,
                        'f1-score': f1,
                        'support': support
                    }
            
            metrics['macro_f1'] = f1_score(y, y_pred, average='macro', zero_division=0)
            metrics['micro_f1'] = f1_score(y, y_pred, average='micro', zero_division=0)
            metrics['classification_report'] = per_class_report
        else:
            metrics['macro_f1'] = f1_score(y, y_pred, average='macro', zero_division=0)
            metrics['weighted_f1'] = f1_score(y, y_pred, average='weighted', zero_division=0)
            
            # Per-class report
            report = classification_report(
                y, y_pred, 
                target_names=self.class_names[:len(np.unique(y))],
                output_dict=True,
                zero_division=0
            )
            metrics['classification_report'] = report
        
        self.metrics = metrics
        return metrics
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance from the model."""
        if not self.fitted:
            raise ValueError("Model not fitted")
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'estimators_'):
            # MultiOutputClassifier - average across estimators
            importances = np.mean([
                est.feature_importances_ for est in self.model.estimators_
                if hasattr(est, 'feature_importances_')
            ], axis=0)
        else:
            return pd.DataFrame({'feature': self.feature_names, 'importance': [0] * len(self.feature_names)})
        
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return df.head(top_n)
    
    def explain_prediction(self, X: np.ndarray, 
                          sample_idx: int = 0) -> Dict[str, Any]:
        """
        Explain a single prediction.
        Uses SHAP if available, otherwise uses rule extractor.
        """
        if not self.fitted:
            raise ValueError("Model not fitted")
        
        explanation = {
            'prediction': None,
            'probabilities': {},
            'top_features': [],
            'rules': []
        }
        
        X_sample = X[sample_idx:sample_idx+1]
        X_scaled = self.scaler.transform(X_sample)
        
        # Get prediction
        pred = self.predict(X_sample)[0]
        explanation['prediction'] = pred if isinstance(pred, (int, np.integer)) else pred.tolist()
        
        # Get probabilities
        proba = self.predict_proba(X_sample)[0]
        if isinstance(proba, np.ndarray):
            for i, p in enumerate(proba):
                if i < len(self.class_names):
                    explanation['probabilities'][self.class_names[i]] = float(p)
        
        # Use SHAP if available
        if HAS_SHAP and hasattr(self.model, 'predict'):
            try:
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(X_scaled)
                
                if isinstance(shap_values, list):
                    # Multi-class: take the predicted class
                    if isinstance(pred, (int, np.integer)):
                        shap_vals = shap_values[pred][0]
                    else:
                        shap_vals = np.mean([sv[0] for sv in shap_values], axis=0)
                else:
                    shap_vals = shap_values[0]
                
                # Top contributing features
                indices = np.argsort(np.abs(shap_vals))[::-1][:5]
                explanation['top_features'] = [
                    {
                        'feature': self.feature_names[i],
                        'value': float(X_sample[0, i]),
                        'shap_value': float(shap_vals[i])
                    }
                    for i in indices
                ]
            except Exception as e:
                pass  # Fall through to rule-based explanation
        
        # Add rule-based explanation
        if self.rule_extractor:
            for code in self.class_names[:5]:  # Top 5 codes
                if code in self.rule_extractor.trees:
                    _, explanations = self.rule_extractor.predict_with_explanation(
                        X_scaled, code
                    )
                    if explanations[0]:
                        explanation['rules'].append({
                            'code': code,
                            'rule': explanations[0]
                        })
        
        return explanation
    
    def save(self, filepath: str):
        """Save model and all components."""
        state = {
            'config': self.config,
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'class_names': self.class_names,
            'class_weights': self.class_weights,
            'metrics': self.metrics,
            'fitted': self.fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
    
    def load(self, filepath: str):
        """Load model and components."""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.config = state['config']
        self.model = state['model']
        self.scaler = state['scaler']
        self.feature_names = state['feature_names']
        self.class_names = state['class_names']
        self.class_weights = state['class_weights']
        self.metrics = state['metrics']
        self.fitted = state['fitted']
        
        return self


def train_with_cross_validation(X: np.ndarray, y: np.ndarray,
                                config: ModelConfig,
                                feature_names: List[str],
                                class_names: List[str]) -> Tuple[ACEErrorCodeModel, Dict]:
    """
    Train model with cross-validation and return best model.
    
    Returns:
        Tuple of (trained_model, cv_results)
    """
    n_samples = X.shape[0]
    
    # Adjust CV folds for small datasets
    if n_samples < config.cv_folds:
        print(f"Warning: Only {n_samples} samples. Skipping cross-validation.")
        # Train directly without CV
        model = ACEErrorCodeModel(config)
        model.fit(X, y, feature_names, class_names)
        return model, {'mean_f1': 0, 'std_f1': 0, 'fold_metrics': [], 'note': 'Too few samples for CV'}
    
    # Use fewer folds if needed
    actual_folds = min(config.cv_folds, n_samples)
    if actual_folds < config.cv_folds:
        print(f"Warning: Reducing CV folds from {config.cv_folds} to {actual_folds}")
    
    cv = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=42)
    
    cv_scores = []
    fold_metrics = []
    
    # For single-label, use y directly; for multi-label, use argmax
    y_stratify = y if y.ndim == 1 else y.sum(axis=1).astype(int)  # Use code count for stratification
    
    # Check minimum samples per class for stratification
    unique, counts = np.unique(y_stratify, return_counts=True)
    min_count = counts.min()
    if min_count < actual_folds:
        print(f"Warning: Some classes have only {min_count} samples. Using regular KFold.")
        from sklearn.model_selection import KFold
        cv = KFold(n_splits=actual_folds, shuffle=True, random_state=42)
        y_stratify = None  # Don't stratify
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y_stratify) if y_stratify is not None else cv.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model = ACEErrorCodeModel(config)
        model.fit(X_train, y_train, feature_names, class_names,
                 validation_data=(X_val, y_val))
        
        metrics = model.evaluate(X_val, y_val)
        cv_scores.append(metrics.get('macro_f1', metrics.get('accuracy', 0)))
        fold_metrics.append(metrics)
        
        print(f"Fold {fold + 1}: F1={cv_scores[-1]:.4f}")
    
    cv_results = {
        'mean_f1': np.mean(cv_scores),
        'std_f1': np.std(cv_scores),
        'fold_metrics': fold_metrics
    }
    
    print(f"\nCV Results: F1 = {cv_results['mean_f1']:.4f} (+/- {cv_results['std_f1']:.4f})")
    
    # Train final model on all data
    final_model = ACEErrorCodeModel(config)
    final_model.fit(X, y, feature_names, class_names)
    
    return final_model, cv_results


if __name__ == '__main__':
    # Test model training
    from data_pipeline import create_synthetic_training_data, IFMLDataPipeline, PaymentRecord
    from feature_engineering import IFMLFeatureEngineer, ErrorCodeLabelEncoder
    
    print("Testing model training with synthetic data...")
    
    # Generate synthetic data
    features_df, code_lists = create_synthetic_training_data(n_samples=1000)
    
    # Create pipeline
    pipeline = IFMLDataPipeline()
    
    for i, (_, row) in enumerate(features_df.iterrows()):
        record = PaymentRecord(
            transaction_id=f"TXN{i:06d}",
            request_features=row.to_dict(),
            response_codes=[{'code': c, 'severity': 'E' if c.startswith('8') else 'W' if c.startswith('7') else 'I', 'info': ''} for c in code_lists[i]],
            error_codes_only=code_lists[i],
            severity_map={c: 'E' if c.startswith('8') else 'W' if c.startswith('7') else 'I' for c in code_lists[i]}
        )
        pipeline.records.append(record)
    
    # Create dataset
    X_raw, X_transformed, y_multilabel = pipeline.create_dataset(min_code_samples=20)
    
    # Convert to arrays
    X = X_transformed.values
    y = y_multilabel.values
    feature_names = X_transformed.columns.tolist()
    class_names = y_multilabel.columns.tolist()
    
    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features, {y.shape[1]} classes")
    
    # Train model
    config = ModelConfig(
        model_type='random_forest',  # Use sklearn for testing
        task_type='multilabel',
        n_estimators=100,
        max_depth=6
    )
    
    model, cv_results = train_with_cross_validation(
        X, y, config, feature_names, class_names
    )
    
    # Evaluate
    print("\n" + "=" * 40)
    print("Feature Importance:")
    print(model.get_feature_importance(10))
    
    # Explain a prediction
    print("\n" + "=" * 40)
    print("Sample Prediction Explanation:")
    explanation = model.explain_prediction(X, sample_idx=0)
    print(json.dumps(explanation, indent=2, default=str))
