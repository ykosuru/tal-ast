# ACE Pelican ML Prediction System - Design Document

## Table of Contents

1. [Overview](#overview)
2. [Why Two Models](#why-two-models)
3. [System Architecture](#system-architecture)
4. [Code Coverage](#code-coverage)
5. [Model Training Pipeline](#model-training-pipeline)
6. [Prediction Pipeline](#prediction-pipeline)
7. [Precondition Rules System](#precondition-rules-system)
8. [9XXX Trainability Classification](#9xxx-trainability-classification)
9. [Configuration Reference](#configuration-reference)
10. [CLI Reference](#cli-reference)
11. [File Structure](#file-structure)

---

## Overview

The ACE Pelican ML Prediction System predicts error codes and repair actions for payment messages before they are processed by the actual ACE Pelican system. This enables:

- **Pre-validation**: Catch errors before submission
- **Repair prediction**: Anticipate what enrichments ACE will apply
- **Cost reduction**: Reduce ACE processing calls by fixing issues upfront

### Code Coverage

| Series | Total Codes | Description |
|--------|-------------|-------------|
| **8XXX** | 34 | Validation errors |
| **9XXX** | 87 | Repair/enrichment codes |
| | 39 | → Trainable (predict from message) |
| | 46 | → Directory-dependent (need lookup) |
| | 2 | → Non-trainable (too generic) |

### Key Components

| Component | Purpose |
|-----------|---------|
| `ifml_parser.py` | Parse IFML JSON into 443 structured features |
| `feature_engineering.py` | Transform parsed data into ML features |
| `model_training.py` | Train RandomForest multi-label classifiers |
| `predictor.py` | Production inference with filtering |
| `prediction_utils.py` | Code definitions, filtering, explanations |
| `main.py` | CLI entry point |

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
│  │ (443 feat)  │    │                  │    │ (RandomForest)  │     │
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
│              ┌───────────────────────────────────┐                  │
│              │    PRECONDITION FILTER            │                  │
│              │    (filter_predictions)           │                  │
│              │                                   │                  │
│              │  • Check require_true conditions  │                  │
│              │  • Check require_false conditions │                  │
│              │  • Remove false positives         │                  │
│              └───────────────────────────────────┘                  │
│                              │                                       │
│                              ▼                                       │
│              ┌───────────────────────────────────┐                  │
│              │    OUTPUT FORMATTING              │                  │
│              │                                   │                  │
│              │  8XXX: Add trigger explanations   │                  │
│              │  9XXX: Add directory lookup info  │                  │
│              └───────────────────────────────────┘                  │
│                              │                                       │
│                              ▼                                       │
│                     ┌──────────────────┐                            │
│                     │ Final Predictions│                            │
│                     └──────────────────┘                            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Code Coverage

### 8XXX Validation Errors (34 codes)

| Code | Description | Key Triggers |
|------|-------------|--------------|
| 8001 | Invalid BIC | has_bic, !bic_valid |
| 8004 | IBAN cannot be derived | needs_iban, !has_iban |
| 8005 | Invalid BIC4 | has_bic |
| 8006 | Invalid country code | !country_valid |
| 8007 | Amount fractional digits exceed max | has_amount |
| 8022 | IBAN inconsistent with BIC | has_iban, has_bic, !bic_iban_match |
| 8023 | IBAN inconsistency in message | has_iban |
| 8024 | BBAN inconsistency in message | has_account |
| 8025 | Domestic Account inconsistency | has_account, is_domestic |
| 8026 | NCH inconsistency in message | has_nch |
| 8027 | ISO Country Code inconsistency | present |
| 8028 | BIC4 inconsistency | has_bic |
| 8029 | Account Number inconsistency | has_account |
| 8030 | IBAN derivation not supported | needs_iban, !has_iban |
| 8033 | CLABE inconsistency | has_account |
| 8034 | Forced Debit not allowed | - |
| 8035 | FCDA account validation failed | has_account (directory) |
| 8036 | FCDA account name matching failed | has_account (directory) |
| 8124 | Invalid currency | has_currency, !currency_valid |
| 8464 | Target Channel not derived | (directory) |
| 8465 | Product code not found | (directory) |
| 8472 | Fee code not derived | (directory) |
| 8851 | Incorrect field size | present |
| 8852 | Incorrect length of attribute | has_account |
| 8853 | Incorrect number format | present |
| 8892 | Invalid Account number | has_account, !account_valid |
| 8894 | Invalid IBAN | has_iban, !iban_checksum_valid |
| 8895 | Invalid NCH code | has_nch, !nch_valid |
| 8896 | Invalid Domestic Account Number | has_account, is_domestic |
| 8897 | Invalid BBAN | has_account, !bban_valid |
| 8898 | IBAN Check Digit validation failed | has_iban, !iban_checksum_valid |
| 8905 | Hash code Mismatch | - |
| 8906 | Message in wrong flow | - |

### 9XXX Repair Codes (87 codes)

#### Trainable (39 codes) - Can Predict From Message Alone

| Code | Description | Detectable From |
|------|-------------|-----------------|
| 9000 | NCH code cleaned | nch_has_dirty_chars |
| 9001 | D Field deleted | present |
| 9002 | Account cleaned of non-alphanumeric | account_has_dirty_chars |
| 9006 | IBAN Cleaned | iban_needs_formatting |
| 9009 | Field Cleaned | present |
| 9010 | Field 23E generated | - |
| 9012 | IBAN Formatted | iban_needs_formatting |
| 9013 | Name and Address repair | has_name |
| 9014 | BBAN cleaned | has_account |
| 9015 | Domestic account cleaned | account_has_dirty_chars |
| 9017 | Multiple party info present | present |
| 9018 | Duplicate party info removed | present |
| 9019 | Party ID cleaned (spaces/dashes) | id_has_dirty_chars, iban_needs_formatting |
| 9020 | Party ID cleaned | id_has_dirty_chars |
| 9021 | FEDABA correctly formatted | nch_needs_formatting |
| 9022 | Account formatted to length | has_account |
| 9026 | Purpose Info found in Field 72 | - |
| 9028 | NCH Code strip off | has_nch |
| 9479 | Account Number Cleaned | account_has_dirty_chars |
| 9483 | Receiver's duplicate info removed | present |
| 9487 | Receiver's BIC info removed | has_bic |
| 9492 | Charge Field updated with value B | - |
| 9493 | Debit Charge Field update with value N | - |
| 9494 | Field updated with GTYOUR | - |
| 9495 | Field updated with FCY | - |
| 9496 | Field updated with FCY OUR CHARGES | - |
| 9497 | Field updated with FCY GTYOUR | - |
| 9498 | PRESAM SecWir updated with Y | - |
| 9918 | Attribute type auto corrected | - |
| 9938 | Redundant phrases deleted | - |
| 9964 | Advice instructions modified | - |
| 9965 | Charge flag modified | - |
| 9966 | Secondary wire flag modified | - |
| 9967 | Cleared PO Box number | has_address |
| 9968 | Parameter values added | - |
| 9969 | Related Amounts added | - |
| 9971 | Negative ref cancels pos ref | - |
| 9987 | Code Consistent | - |
| 9990 | Duplicate CDT info removed | present |

#### Directory-Dependent (46 codes) - Require External Lookup

| Code | Description | Directory |
|------|-------------|-----------|
| 9004 | IBAN repaired in Account line | IBAN_DERIVATION |
| 9005 | BIC replaced by derived BIC | NCH_TO_BIC |
| 9007 | Account replaced by IBAN | IBAN_DERIVATION |
| 9008 | IBAN to BIC repair | IBAN_TO_BIC |
| 9023 | Fund to Account repair | ACCOUNT_DIRECTORY |
| 9024 | Push Up performed | ROUTING |
| 9025 | CLABE repaired | CLABE_DIRECTORY |
| 9027 | FCDA account reformatted | FCDA_DIRECTORY |
| 9029 | GL account insertion for FED | GL_DIRECTORY |
| 9030 | DDA to GL attribute type conversion | GL_DIRECTORY |
| 9031 | Trust account to GL number insertion | GL_DIRECTORY |
| 9032 | 8 char BIC insertion | BIC_DIRECTORY |
| 9475 | D-A using Account No. from Name/Address | NAME_ADDRESS_PARSE |
| 9476 | D-A using CHIPS ABA from Name/Address | NAME_ADDRESS_PARSE |
| 9477 | D-A using FED ABA from Name/Address | NAME_ADDRESS_PARSE |
| 9478 | D-A using CHIPS UID from Name/Address | NAME_ADDRESS_PARSE |
| 9480 | Push Down type Push 1 | ROUTING |
| 9481 | Push Down type Push 2 | ROUTING |
| 9482 | Partial Push 2 | ROUTING |
| 9484 | Repaired from Ban to Bank Info | BANK_DIRECTORY |
| 9485 | D-A using NCH from Name/Address | NAME_ADDRESS_PARSE |
| 9486 | A to D performed | ROUTING |
| 9488 | Partial push down | ROUTING |
| 9490 | Fee Code Updated | FEE_DIRECTORY |
| 9491 | Tariff Fee Updated | FEE_DIRECTORY |
| 9901 | D-A using BIC from field | BIC_DIRECTORY |
| 9910 | BIC repaired from field 72 | BIC_DIRECTORY |
| 9917 | BIC repaired from BIC Plus | BIC_PLUS |
| 9932 | Account line repaired with Account | ACCOUNT_DIRECTORY |
| 9935 | Account line repaired by NCH | NCH_DIRECTORY |
| 9936 | D-A using BIC from Name/Address | NAME_TO_BIC |
| 9961 | BIC Derived from Name/Address | NAME_TO_BIC |
| 9962 | Account number updated | ACCOUNT_DIRECTORY |
| 9963 | Account number added | ACCOUNT_DIRECTORY |
| 9970 | D-A using BIC from Name/Address | NAME_TO_BIC |
| 9978-9984 | Various BIC derivations | BIC_DIRECTORY |
| 9985 | BIC from CHIPS ABA Repaired | CHIPS_ABA_TO_BIC |
| 9986 | Head Office BIC Repaired | BIC_DIRECTORY |
| 9991 | Account pushed to BBI from BBK | ROUTING |
| 9992 | NCH translocated to BBI | ROUTING |

#### Non-Trainable (2 codes)

| Code | Description | Reason |
|------|-------------|--------|
| 9439 | No Pattern Found | Too generic |
| 9999 | Field Repaired | Catch-all |

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

The `IFMLParser` extracts 443 features per payment:

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
    'cdt_iban_needs_formatting': False,  # For 9019 detection
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

Codes are encoded as composite codes to preserve party context:

```python
# ACE returns: "8894" on party "CdtPty"
# We encode as: "8894_CDTPTY"

# This enables party-specific predictions
```

### 4. Model Training

```bash
# Train 8XXX model (all 8XXX codes are trainable)
python main.py train --data-file payments.json --output-dir ./models_8x \
    --code-series 8 --composite

# Train 9XXX model (trainable codes only)
python main.py train --data-file payments.json --output-dir ./models_9x \
    --code-series 9 --composite --trainable-only
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

### Step 2: Apply Precondition Filter

The `filter_predictions()` function removes false positives:

```python
from prediction_utils import filter_predictions

raw_predictions = model.predict(X)
filtered, removed = filter_predictions(raw_predictions, features)
```

### Step 3: Format Output

For 8XXX codes, add trigger explanations:
```json
{
  "code": "8895_BNFBNK",
  "description": "Invalid NCH code",
  "probability": 0.85,
  "triggers": [
    {"feature": "bnf_has_nch", "value": true, "condition": "must be True"},
    {"feature": "bnf_nch_valid", "value": false, "condition": "must be False"}
  ]
}
```

For 9XXX directory-dependent codes, add lookup conditions:
```json
{
  "code": "9008_BNFBNK",
  "description": "BIC derived from IBAN",
  "category": "DIRECTORY_DEPENDENT",
  "needs_directory": true,
  "directory": "IBAN_TO_BIC",
  "lookup_condition": "LOOKUP(IBAN_TO_BIC, iban=GB82WEST12345698765432) -> IF found EMIT 9008 ELSE skip"
}
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

```python
CODE_TRIGGERS = {
    '8894': {
        'require_true': ['has_iban'],
        'require_false': ['iban_valid_format', 'iban_checksum_valid'],
        'description': 'Invalid IBAN - format or checksum invalid'
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

### Filter Applied at Runtime

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

## 9XXX Trainability Classification

### Why Classify 9XXX Codes?

Not all 9XXX codes can be predicted from message features alone:

| Category | Can Predict? | Example |
|----------|--------------|---------|
| **Trainable** | Yes | 9019: ID has spaces → will be cleaned |
| **Directory-Dependent** | Partial | 9008: Has IBAN, no BIC → *might* derive BIC (depends on lookup) |
| **Non-Trainable** | No | 9999: Generic catch-all |

### Using `--trainable-only` Flag

```bash
# Train only on codes we can actually predict
python main.py train --data-file payments.json --output-dir ./models_9x \
    --code-series 9 --composite --trainable-only
```

This filters training data to only include the 39 trainable 9XXX codes.

### Directory-Dependent Output Format

For codes that need directory lookup, the prediction output includes:

```json
{
  "code": "9004_CDTPTY",
  "category": "DIRECTORY_DEPENDENT",
  "needs_directory": true,
  "directory": "IBAN_DERIVATION",
  "lookup_condition": "LOOKUP(IBAN_DERIVATION, country=DE, account=123456789) -> IF found EMIT 9004 ELSE EMIT 8004",
  "note": "Prediction conditional on directory lookup result"
}
```

This tells the caller: "The model thinks this *might* fire, but you need to check the directory to be sure."

---

## CLI Reference

### Training

```bash
# Train 8XXX model
python main.py train --data-file payments.json --output-dir ./models_8x \
    --code-series 8 --composite

# Train 9XXX model (trainable only)
python main.py train --data-file payments.json --output-dir ./models_9x \
    --code-series 9 --composite --trainable-only

# Train with specific model type
python main.py train --data-file payments.json --output-dir ./models \
    --model-type random_forest --composite
```

**Training Options:**

| Option | Description |
|--------|-------------|
| `--data-file` | Single JSON file with payments |
| `--data-dir` | Directory of JSON files |
| `--output-dir` | Where to save model |
| `--code-series` | Filter to series (e.g., `8 9`) |
| `--composite` | Use party-composite codes (e.g., `8894_CDTPTY`) |
| `--trainable-only` | For 9XXX: only train on trainable codes |
| `--model-type` | `random_forest`, `gradient_boost`, `decision_tree` |
| `--min-samples` | Min samples for a code to be a class (default: 5) |

### Prediction

```bash
python main.py predict --model-dir ./models_8x --input request.json --threshold 0.5
```

**Output:**

```json
{
  "transaction_id": "TXN123",
  "predictions": [
    {
      "code": "8895_BNFBNK",
      "description": "Invalid NCH code",
      "probability": 0.85,
      "triggers": [...]
    }
  ],
  "filtered_out": [
    {"code": "8894_BNFBNK", "reason": "iban_checksum_valid=True"}
  ],
  "raw_prediction_count": 2,
  "filtered_prediction_count": 1
}
```

### Analysis

```bash
python main.py analyze --model-dir ./models_8x --input request.json
python main.py analyze --model-dir ./models_8x --input request.json --code 8895
```

### Feature Importance

```bash
python main.py importance --model-dir ./models_8x --top 30
python main.py importance --model-dir ./models_8x --code 8895_BNFBNK
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
  
  "apply_precondition_filter": true
}
```

| Setting | Purpose |
|---------|---------|
| `high_threshold_codes` | Per-code probability thresholds |
| `apply_precondition_filter` | Enable/disable precondition filtering |
| `suppressed_codes` | Codes to never predict |

---

## File Structure

```
ace_ml/
├── README_ML_MODEL.md           # Quick start guide
├── DESIGN.md                    # This document
│
├── Core Modules
│   ├── main.py                  # CLI entry point
│   ├── ifml_parser.py           # Parse IFML JSON → 443 features
│   ├── feature_engineering.py   # Transform features → ML input
│   ├── data_pipeline.py         # Load and prepare training data
│   ├── model_training.py        # Train multi-label classifiers
│   └── predictor.py             # Production inference
│
├── Utilities
│   ├── prediction_utils.py      # CODE_TRIGGERS, filter, explain
│   └── test_model.py            # Accuracy testing
│
├── Config Files
│   ├── precondition_rules.json  # ACE business rules
│   └── prediction_config.json   # Thresholds and settings
│
└── Model Directories
    ├── models_8x/               # 8XXX validation error model
    │   ├── model.pkl
    │   ├── feature_engineer.pkl
    │   └── training_info.json
    │
    └── models_9x/               # 9XXX repair model (trainable only)
        ├── model.pkl
        ├── feature_engineer.pkl
        └── training_info.json
```

---

## Key Insights

1. **Composite codes work better** - `8895_BNFBNK` learns party-specific patterns better than just `8895`

2. **9019 fires for ANY ID with special chars** - not just IBANs
   - Examples: `IE93-00-67`, `066196221 FFC X29968153`, `383174835:HALLERAPPRAISALSERVICEIN`

3. **Directory-dependent codes can't be fully predicted** - they depend on ACE's reference data
   - Train only on trainable codes with `--trainable-only`
   - Output includes lookup conditions for runtime evaluation

4. **Precondition filtering is essential** - removes false positives where ML predicts error but data is actually valid

5. **Feature count must match exactly** - parser outputs 443 features for all party types (even when absent) to ensure consistency between training and prediction

6. **Parser fixes applied:**
   - `Type="S"` recognized as BIC identifier
   - `BeneficiaryPartyInf` merged with `BeneficiaryBankInfo`
   - Plain string IBAN/BIC detection by pattern
   - `iban_needs_formatting` flag for 9019 detection
