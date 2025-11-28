# ACE Pelican ML Prediction System - Design Document

## Table of Contents

1. [Overview](#overview)
2. [Why Two Models](#why-two-models)
3. [System Architecture](#system-architecture)
4. [Model Training Pipeline](#model-training-pipeline)
5. [Prediction Pipeline](#prediction-pipeline)
6. [Precondition Rules System](#precondition-rules-system)
7. [Configuration Reference](#configuration-reference)
8. [File Structure](#file-structure)

---

## Overview

The ACE Pelican ML Prediction System predicts error codes and repair actions for payment messages before they are processed by the actual ACE Pelican system. This enables:

- **Pre-validation**: Catch errors before submission
- **Repair prediction**: Anticipate what enrichments ACE will apply
- **Cost reduction**: Reduce ACE processing calls by fixing issues upfront

### Key Components

| Component | Purpose |
|-----------|---------|
| `ifml_parser.py` | Parse IFML JSON into structured features |
| `feature_engineering.py` | Transform parsed data into ML features |
| `model_training.py` | Train RandomForest multi-label classifiers |
| `predictor.py` | Production inference with filtering |
| `prediction_utils.py` | Standalone filter and explanation utilities |
| `precondition_rules.json` | ACE business rules for filtering |

---

## Why Two Models

### The Problem

ACE Pelican returns two types of codes:

| Series | Type | Meaning | Example |
|--------|------|---------|---------|
| **8XXX** | Validation Errors | Something is wrong with the input | 8894 = Invalid IBAN |
| **9XXX** | Repair/Enrichment | ACE fixed or enriched something | 9004 = IBAN derived |

These have fundamentally different characteristics:

### 8XXX Validation Errors

```
Characteristics:
- Binary: Either the data is valid or it's not
- Deterministic: Same input → same validation result
- Rule-based: Follows strict validation logic
- Sparse: Most payments have 0-2 errors

Examples:
- 8894: IBAN checksum failed
- 8895: Invalid routing number
- 8022: IBAN country doesn't match BIC country
```

### 9XXX Repair/Enrichment Codes

```
Characteristics:
- Conditional: Depends on what ACE can look up in directories
- External dependencies: BIC directory, IBAN derivation rules
- Dense: Payments often have 3-10 repairs
- Chained: One repair can trigger another

Examples:
- 9004: IBAN derived from account + country
- 9008: BIC derived from IBAN
- 9477: BIC expanded from 8 to 11 characters
```

### Why Separate Models Work Better

| Aspect | Combined Model | Separate Models |
|--------|----------------|-----------------|
| Class balance | 8XXX drowns in 9XXX frequency | Each optimized for its distribution |
| Feature importance | Conflicting signals | Clean feature relevance |
| Threshold tuning | One threshold doesn't fit all | Per-series optimization |
| Accuracy | ~85% | 8XXX: 93.9%, 9XXX: 79.4% |

### Decision Boundary Differences

```
8XXX Model learns:
  "If has_iban=True AND iban_checksum_valid=False → predict 8894"
  
9XXX Model learns:
  "If needs_iban=True AND has_iban=False AND has_account=True → predict 9004"
```

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         TRAINING PHASE                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Raw Data (JSON)                                                     │
│       │                                                              │
│       ▼                                                              │
│  ┌─────────────┐    ┌──────────────────┐    ┌─────────────────┐     │
│  │ IFML Parser │───▶│ Feature Engineer │───▶│ Model Training  │     │
│  └─────────────┘    └──────────────────┘    └─────────────────┘     │
│                                                    │                 │
│                              ┌─────────────────────┴──────────────┐  │
│                              ▼                                    ▼  │
│                     ┌──────────────┐                    ┌──────────────┐
│                     │ 8XXX Model   │                    │ 9XXX Model   │
│                     │ (models_8x/) │                    │ (models_9x/) │
│                     └──────────────┘                    └──────────────┘
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                        INFERENCE PHASE                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Payment JSON                                                        │
│       │                                                              │
│       ▼                                                              │
│  ┌─────────────┐    ┌──────────────────┐                            │
│  │ IFML Parser │───▶│ Feature Engineer │                            │
│  └─────────────┘    └──────────────────┘                            │
│                              │                                       │
│                              ▼                                       │
│                     ┌──────────────────┐                            │
│                     │  Model.predict() │                            │
│                     └──────────────────┘                            │
│                              │                                       │
│                              ▼                                       │
│              ┌───────────────────────────────┐                      │
│              │      FILTERING PIPELINE       │                      │
│              │                               │                      │
│              │  1. Precondition Filter       │◀── precondition_rules.json
│              │  2. Semantic Filter           │                      │
│              │  3. Mutual Exclusivity Filter │◀── exclusion_rules.json
│              │                               │                      │
│              └───────────────────────────────┘                      │
│                              │                                       │
│                              ▼                                       │
│                     ┌──────────────────┐                            │
│                     │ Filtered Codes   │                            │
│                     └──────────────────┘                            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Model Training Pipeline

### 1. Data Loading

```python
from data_pipeline import IFMLDataPipeline

pipeline = IFMLDataPipeline()
pipeline.load_combined_files('./raw_data', '*.json')
# Loads IFML requests paired with ACE responses
```

### 2. Feature Extraction

The `IFMLParser` extracts ~200 features per payment:

```python
# Party-level features (for each party: orig, cdt, bnf, intm, etc.)
{
    'cdt_present': True,
    'cdt_has_iban': True,
    'cdt_iban_valid_format': True,
    'cdt_iban_checksum_valid': True,
    'cdt_has_bic': True,
    'cdt_bic_valid_format': True,
    'cdt_bic_iban_match': True,
    'cdt_has_nch': False,
    'cdt_has_account': True,
    'cdt_needs_iban': True,
    'cdt_is_domestic': False,
    'cdt_is_international': True,
    ...
}

# Payment-level features
{
    'primary_currency': 'EUR',
    'is_cross_border': True,
    'incoming_msg_type': 'pacs.008',
    ...
}
```

### 3. Label Encoding

Codes are extracted and formatted as composite codes:

```python
# ACE returns: "8894" on party "CdtPty"
# We encode as: "8894_CDTPTY"

# This preserves party context for predictions
```

### 4. Model Training

```python
from model_training import ACEErrorCodeModel

# Train 8XXX model
model_8x = ACEErrorCodeModel()
model_8x.fit(X_train, y_train_8xxx)
model_8x.save('./models_8x/')

# Train 9XXX model  
model_9x = ACEErrorCodeModel()
model_9x.fit(X_train, y_train_9xxx)
model_9x.save('./models_9x/')
```

### Model Type: Multi-Label Classification

Each payment can have multiple error codes, so we use:

```
RandomForestClassifier with MultiOutputClassifier wrapper
- Each code is a binary classification target
- Predictions are independent per code
- Threshold (default 0.5) determines positive prediction
```

---

## Prediction Pipeline

### Step 1: Parse and Transform

```python
from predictor import ACEPredictor

predictor = ACEPredictor('./models_8x')
result = predictor.predict(ifml_json)
```

Internally:

```python
# 1. Parse IFML
features = self.parser.parse(ifml_json)
feature_dict = self.parser.to_dict(features)

# 2. Transform to model input
X = self.feature_engineer.transform(pd.DataFrame([feature_dict]))

# 3. Get probabilities
probabilities = self.model.predict_proba(X)
```

### Step 2: Apply Threshold

```python
predicted_codes = []
for idx, prob in enumerate(probabilities):
    code = self.class_names[idx]
    
    # Check per-code threshold from config
    threshold = self.prediction_config.get('high_threshold_codes', {}).get(code, 0.5)
    
    if prob >= threshold:
        predicted_codes.append(code)
```

### Step 3: Apply Filters

```python
# Filter 1: Precondition filter (from precondition_rules.json)
if self.precondition_rules:
    predicted_codes, warnings = self._apply_precondition_filter(
        predicted_codes, feature_dict)

# Filter 2: Semantic filter (built-in rules)
predicted_codes, warnings = self._apply_semantic_filters(
    predicted_codes, feature_dict)

# Filter 3: Mutual exclusivity filter (from exclusion_rules.json)
if self.exclusion_rules:
    predicted_codes, warnings = self._apply_mutual_exclusivity(
        predicted_codes, code_probs, feature_dict)
```

---

## Precondition Rules System

### Design Philosophy

The ML model learns correlations from training data, but some predictions are **logically impossible**:

```
Example: Model predicts 8894 (Invalid IBAN)
But:     has_iban=True AND iban_valid_format=True AND iban_checksum_valid=True

This is a FALSE POSITIVE - IBAN exists and is valid, so 8894 cannot fire.
```

Rather than retraining the model, we apply **post-prediction filtering** based on ACE business rules.

### Rule Structure

Each code has two condition types:

```json
{
  "8894": {
    "require_true": ["has_iban"],
    "require_false": ["iban_valid_format", "iban_checksum_valid"],
    "description": "Invalid IBAN - format or checksum invalid"
  }
}
```

| Condition | Logic | Example |
|-----------|-------|---------|
| `require_true` | ALL must be True | `has_iban=True` (IBAN must exist) |
| `require_false` | At least ONE must be False | `iban_valid=False` (validation must fail) |

### Filtering Logic

```python
def filter_predictions(predicted_codes, features):
    for code in predicted_codes:
        trigger_def = CODE_TRIGGERS.get(base_code)
        
        # Check require_true: ALL must be True
        for feat in require_true:
            if not features.get(f'{party}_{feat}'):
                FILTER_OUT(code, "required field missing or False")
        
        # Check require_false: At least ONE must be False
        all_valid = True
        for feat in require_false:
            if not features.get(f'{party}_{feat}'):
                all_valid = False  # Found an invalid check
                break
        
        if all_valid and require_false:
            FILTER_OUT(code, "all validity checks passed")
```

### Why Party Presence Matters for 9XXX

For repair codes, the party must exist to be repaired:

```json
{
  "9004": {
    "require_true": ["present", "needs_iban"],
    "require_false": ["has_iban"],
    "description": "IBAN derived for credit party"
  }
}
```

Logic:
- `present=True`: Credit party exists in the message
- `needs_iban=True`: Party's country requires IBAN
- `has_iban=False`: Party doesn't already have IBAN

If any condition fails, 9004 cannot fire.

### Rules Applied at Runtime

The filter checks are applied **after** model prediction but **before** returning results:

```
Model Output: [8894_CDTPTY, 8022_CDTPTY, 8004_CDTPTY]
                    │
                    ▼
            Precondition Filter
                    │
    ┌───────────────┼───────────────┐
    │               │               │
    ▼               ▼               ▼
8894_CDTPTY    8022_CDTPTY    8004_CDTPTY
has_iban=T     has_iban=T     needs_iban=T
iban_valid=T   bic_iban_match=T  has_iban=F
    │               │               │
    ▼               ▼               ▼
 FILTERED       FILTERED         KEPT
 (valid)        (match)      (no IBAN)
                    │
                    ▼
            Final Output: [8004_CDTPTY]
```

---

## Configuration Reference

### prediction_config.json

```json
{
  "negative_margin": 0.1,
  "suppressed_codes": [],
  
  "high_threshold_codes": {
    "8022": 0.70,
    "8026": 0.70,
    "8894": 0.70,
    "8895": 0.70,
    "9004": 0.70,
    "9479": 0.80,
    "9999": 0.80
  },
  
  "apply_precondition_filter": true,
  "apply_semantic_filter": true,
  "apply_exclusion_filter": true
}
```

| Setting | Purpose |
|---------|---------|
| `high_threshold_codes` | Per-code probability thresholds (higher = fewer false positives) |
| `apply_precondition_filter` | Enable/disable precondition rules |
| `apply_semantic_filter` | Enable/disable built-in semantic rules |
| `apply_exclusion_filter` | Enable/disable mutual exclusivity rules |
| `suppressed_codes` | Codes to never predict |

### precondition_rules.json

```json
{
  "8894": {
    "require_true": ["has_iban"],
    "require_false": ["iban_valid_format", "iban_checksum_valid"],
    "description": "Invalid IBAN"
  },
  "9004": {
    "require_true": ["present", "needs_iban"],
    "require_false": ["has_iban"],
    "description": "IBAN derived for credit party"
  }
}
```

### exclusion_rules.json (Optional)

```json
{
  "mutual_exclusions": [
    {"codes": ["8894", "8898"], "reason": "Both are IBAN errors, pick one"}
  ],
  "same_field_conflicts": [
    {"codes": ["8894", "8896"], "resolution": "Use is_international feature"}
  ]
}
```

---

## File Structure

```
ace_ml/
├── README.md                    # Quick start guide
├── DESIGN.md                    # This document
│
├── Core Modules
│   ├── ifml_parser.py           # Parse IFML JSON → structured features
│   ├── feature_engineering.py   # Transform features → ML input
│   ├── data_pipeline.py         # Load and prepare training data
│   ├── model_training.py        # Train multi-label classifiers
│   └── predictor.py             # Production inference with filtering
│
├── Utilities
│   ├── prediction_utils.py      # Standalone filter + explainer
│   ├── ace_codes.py             # ACE code definitions
│   └── test_model.py            # Accuracy testing
│
├── Config Files
│   ├── precondition_rules.json  # ACE business rules for filtering
│   ├── prediction_config.json   # Thresholds and settings
│   └── exclusion_rules.json     # Mutual exclusivity rules
│
└── Model Directories
    ├── models_8x/               # 8XXX validation error model
    │   ├── model.pkl
    │   ├── feature_engineer.pkl
    │   ├── training_info.json
    │   ├── precondition_rules.json
    │   └── prediction_config.json
    │
    └── models_9x/               # 9XXX repair/enrichment model
        ├── model.pkl
        ├── feature_engineer.pkl
        ├── training_info.json
        ├── precondition_rules.json
        └── prediction_config.json
```

---

## Usage Examples

### Training

```bash
# Train 8XXX model
python main.py train --data-dir ./raw_data --output-dir ./models_8x --series 8

# Train 9XXX model
python main.py train --data-dir ./raw_data --output-dir ./models_9x --series 9
```

### Testing

```bash
# Test with precondition filtering
python test_model.py --model-dir ./models_8x --data-dir ./raw_data --series 8
```

### Prediction

```python
from predictor import ACEPredictor

# Load model with filters
predictor = ACEPredictor('./models_8x')

# Predict
result = predictor.predict(payment_json)

print(result.predicted_codes)  # ['8004_CDTPTY']
print(result.warnings)         # Any filtering warnings
```

### Explanation

```bash
# Generate prediction explanations
python prediction_utils.py --model-dir ./models_8x --payment ./test.json
```

Output:
```
PREDICTION EXPLANATION
============================================================
Raw predicted: ['8894_BNFBNK', '8004_BNPPTY']
After filter: ['8004_BNPPTY']

Filtered out:
  - 8894_BNFBNK: all validity checks passed: ['bnf_iban_valid_format']

--- 8004_BNPPTY ---
ACE Definition: IBAN cannot be derived - IBAN required but missing

Field                                    Value      Explanation
----------------------------------------------------------------------
bnf_needs_iban                           True       Beneficiary Bank: Party is in IBAN-required country
bnf_has_iban                             False      Beneficiary Bank: IBAN is provided
```

---

## Performance Summary

| Model | Accuracy | Notes |
|-------|----------|-------|
| 8XXX (no filter) | 92.7% | Raw model predictions |
| 8XXX (with filter) | 93.9% | +1.2% from precondition filtering |
| 9XXX | 79.4% | More complex due to external dependencies |

### Remaining Error Sources

| Issue | Cause | Potential Fix |
|-------|-------|---------------|
| False positives on valid data | Model learned correlation, not causation | More precondition rules |
| Missing predictions | Rare codes with few training examples | Rare code detectors |
| 9XXX accuracy gap | External directory lookups not modeled | Add derived features |

---

## Appendix: ACE Code Reference

### 8XXX Validation Errors

| Code | Description | Precondition |
|------|-------------|--------------|
| 8001 | Invalid BIC | has_bic ∧ ¬bic_valid |
| 8004 | IBAN cannot be derived | needs_iban ∧ ¬has_iban |
| 8022 | IBAN/BIC country mismatch | has_iban ∧ has_bic ∧ ¬bic_iban_match |
| 8026 | NCH inconsistency | has_nch |
| 8894 | Invalid IBAN | has_iban ∧ ¬iban_valid |
| 8895 | Invalid NCH | has_nch ∧ ¬nch_valid |
| 8896 | Invalid domestic account | has_account ∧ ¬account_valid |
| 8898 | IBAN checksum failed | has_iban ∧ ¬iban_checksum_valid |

### 9XXX Repair Codes

| Code | Description | Precondition |
|------|-------------|--------------|
| 9002 | Account cleaned | present ∧ has_account |
| 9004 | IBAN derived (credit party) | present ∧ needs_iban ∧ ¬has_iban |
| 9005 | BIC derived from NCH | present ∧ has_nch ∧ ¬has_bic |
| 9007 | IBAN derived (beneficiary) | present ∧ needs_iban ∧ ¬has_iban |
| 9008 | BIC derived from IBAN | present ∧ has_iban ∧ ¬has_bic |
| 9477 | BIC enriched 8→11 | present ∧ has_bic |
| 9479 | Party enriched from BIC dir | present ∧ has_bic |
