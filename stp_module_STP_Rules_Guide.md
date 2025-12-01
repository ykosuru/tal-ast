# ACE Pelican Error Code Prediction - Complete Reference

## Cross-Referenced with Official Definitions | November 2025

---

## Table of Contents

1. [Overview](#1-overview)
2. [Code Series Summary](#2-code-series-summary)
3. [8XXX Validation Errors](#3-8xxx-validation-errors)
4. [9XXX Repair/Enrichment Codes](#4-9xxx-repairenrichment-codes)
5. [Directory Lookups](#5-directory-lookups)
6. [Complete Decision Trees](#6-complete-decision-trees)
7. [Feature Reference](#7-feature-reference)
8. [ML Model Strategy](#8-ml-model-strategy)
9. [Suppressed Codes](#9-suppressed-codes)
10. [Quick Reference Card](#10-quick-reference-card)

---

## 1. Overview

ACE Pelican processes payment messages (IFML format) and emits codes indicating:
- **Validation errors** (8XXX) - Something is wrong with the input
- **Repairs/Enrichments** (9XXX) - ACE fixed or enhanced the data
- **Warnings** (7XXX) - Potential issues that don't block processing
- **Information** (6XXX) - Informational messages about what was extracted

### Processing Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ACE PELICAN PROCESSING FLOW                          │
└─────────────────────────────────────────────────────────────────────────────┘

  Payment Message (IFML)
          │
          ▼
  ┌───────────────────┐
  │  PARSE & EXTRACT  │
  │  - Party info     │
  │  - Amounts        │
  │  - Routing codes  │
  └───────────────────┘
          │
          ▼
  ┌───────────────────┐
  │    VALIDATION     │──────► 8XXX Errors (if invalid)
  │  - Format checks  │
  │  - Checksum       │
  │  - Consistency    │
  └───────────────────┘
          │
          │ (if valid or repairable)
          ▼
  ┌───────────────────┐
  │  ENRICHMENT &     │──────► 9XXX Repairs (if changed)
  │  REPAIR           │──────► 9999 (if lookup failed)
  │  - Directory      │
  │    lookups        │
  │  - Field cleaning │
  └───────────────────┘
          │
          ▼
  ┌───────────────────┐
  │     OUTPUT        │
  │  - Enriched msg   │
  │  - All codes      │
  └───────────────────┘
```

---

## 2. Code Series Summary

| Series | Type | Count | Directory Required | Predictability |
|--------|------|-------|-------------------|----------------|
| **6XXX** | Information | ~100 | No | High |
| **7XXX** | Warnings | ~50 | No | High |
| **8XXX** | Validation Errors | ~30 | No | **Very High** |
| **9XXX** | Repairs | ~50 | Mixed | Medium |

### Prediction Categories

| Category | Codes | Method |
|----------|-------|--------|
| **Deterministic** | 8001-8030, 8892-8898, 9000-9028 (cleaning) | Rule-based decision trees |
| **Model-Assisted** | Complex pattern codes | ML model probability |
| **Directory-Dependent** | 9004-9008, 9475-9486, 9901-9992 | Predict eligibility only |
| **Suppressed** | 9999, 9490 | Do not predict |

---

## 3. 8XXX Validation Errors

All 8XXX codes are **fully deterministic** from input features - no directory lookups required.

### 8XXX Code Reference Table

| Code | Official Description | Precondition | Deterministic? |
|------|---------------------|--------------|----------------|
| **8001** | Invalid BIC | `has_bic = True` | ✅ YES |
| **8003** | File name derivation failed | System-level | ❌ NO (system) |
| **8004** | IBAN cannot be derived | `needs_iban = True` AND `has_account = True` | ⚠️ PARTIAL |
| **8005** | Invalid BIC4 | `has_bic = True` | ✅ YES |
| **8006** | Invalid country code | Country code present | ✅ YES |
| **8007** | Fractional digits exceed max | Amount present | ✅ YES |
| **8022** | IBAN inconsistent with BIC | `has_iban = True` AND `has_bic = True` | ✅ YES |
| **8023** | IBAN inconsistency in message | `iban_count > 1` | ✅ YES |
| **8024** | BBAN inconsistency in message | `bban_count > 1` | ✅ YES |
| **8025** | Domestic Account inconsistency | `domestic_account_count > 1` | ✅ YES |
| **8026** | NCH inconsistency in message | `nch_count > 1` OR `id_has_bic_and_nch` | ✅ YES |
| **8027** | ISO Country Code inconsistency | `country_count > 1` | ✅ YES |
| **8028** | BIC4 inconsistency in message | `bic4_count > 1` | ✅ YES |
| **8029** | Account Number inconsistency | `account_count > 1` | ✅ YES |
| **8030** | IBAN derivation not supported | `needs_iban = True` | ✅ YES |
| **8033** | CLABE inconsistency | `clabe_count > 1` | ✅ YES |
| **8034** | Forced Debit not allowed | Debit indicator | ❌ NO (business) |
| **8035** | FCDA validation failed | FCDA account | ⚠️ PARTIAL |
| **8036** | FCDA name matching failed | FCDA account | ❌ NO (directory) |
| **8124** | Invalid currency | Currency present | ✅ YES |
| **8464** | Target Channel not derived | Routing required | ❌ NO (business) |
| **8465** | Product code not found | Product code required | ❌ NO (directory) |
| **8472** | Fee code not derived | Fee required | ❌ NO (business) |
| **8851** | Incorrect field size | Field present | ✅ YES |
| **8852** | Incorrect length of attribute | Attribute present | ✅ YES |
| **8853** | Incorrect number format | Numeric field | ✅ YES |
| **8892** | Invalid Account number | `has_account = True` | ✅ YES |
| **8894** | Invalid IBAN | `has_iban = True` | ✅ YES |
| **8895** | Invalid NCH code | `has_nch = True` AND `nch_validation_applicable` | ✅ YES |
| **8896** | Invalid Domestic Account | `has_domestic_account = True` | ✅ YES |
| **8897** | Invalid BBAN | `has_bban = True` | ✅ YES |
| **8898** | IBAN Check Digit failed | `has_iban = True` AND `iban_valid_format` | ✅ YES |
| **8905** | Hash code Mismatch | Hash present | ❌ NO (system) |
| **8906** | Message in wrong flow | Flow indicator | ❌ NO (system) |

### 8XXX Decision Trees - Complete Reference

> **⚠️ Implementation Note**: Decision trees below describe the *logical rules* for each code. 
> However, our ML model may not have all features implemented. See Section 7 for actual feature availability.
> 
> **Fully Implemented**: 8001, 8004, 8022, 8026, 8030, 8894, 8895, 8898
> **Partially Implemented**: 8005, 8006, 8007, 8852, 8892, 8896, 8897
> **Not Implemented (ML only)**: 8023, 8024, 8025, 8027, 8028, 8029 (no cross-party aggregation)

---

#### 8001: Invalid BIC

```
has_bic = True?
│
├─ NO → Cannot fire
│
└─ YES → bic_valid_format? (8 or 11 chars, [A-Z]{4}[A-Z]{2}[A-Z0-9]{2,5})
          │
          ├─ NO → 8001 FIRES
          │
          └─ YES → bic_valid_country? (chars 5-6 in ISO 3166)
                    │
                    ├─ NO → 8001 FIRES
                    │
                    └─ YES → Cannot fire
```

**Features**: `has_bic`, `bic_valid_format`, `bic_valid_country`, `bic_length`

---

#### 8004: IBAN Cannot Be Derived

```
needs_iban = True?
│
├─ NO → Cannot fire
│
└─ YES → has_iban?
          │
          ├─ YES → Cannot fire
          │
          └─ NO → has_account?
                   │
                   ├─ NO → 8004 FIRES (nothing to derive from)
                   │
                   └─ YES → iban_derivation_supported?
                             │
                             ├─ NO → 8030 FIRES instead
                             │
                             └─ YES → Directory lookup attempted
                                      │
                                      ├─ Success → 9004 FIRES
                                      │
                                      └─ Fail → 8004 FIRES + 9999
```

**Features**: `needs_iban`, `has_iban`, `has_account`, `iban_derivation_supported`

**Note**: 8004 may fire AFTER a 9004 directory lookup fails.

---

#### 8005: Invalid BIC4

```
has_bic = True?
│
├─ NO → Cannot fire
│
└─ YES → bic_length >= 4?
          │
          ├─ NO → 8005 FIRES
          │
          └─ YES → bic4_valid? (first 4 chars = [A-Z]{4})
                    │
                    ├─ NO → 8005 FIRES
                    │
                    └─ YES → Cannot fire
```

**Features**: `has_bic`, `bic_length`, `bic4_valid`

---

#### 8006: Invalid Country Code

```
has_country? (any country field present)
│
├─ NO → Cannot fire
│
└─ YES → country_valid? (in ISO 3166-1 alpha-2 list)
          │
          ├─ NO → 8006 FIRES
          │
          └─ YES → Cannot fire
```

**Features**: `country`, `bic_country`, `iban_country`, `address_country`, `country_valid`

---

#### 8007: Fractional Digits Exceed Maximum

```
has_amount = True?
│
├─ NO → Cannot fire
│
└─ YES → Get currency decimal places (ISO 4217)
          │
          └─ actual_decimals > allowed_decimals?
                    │
                    ├─ NO → Cannot fire
                    │
                    └─ YES → 8007 FIRES
```

**Features**: `primary_currency`, `primary_amount`, `amount_decimal_places`

**Currency Rules**:
- JPY, KRW: 0 decimals
- BHD, KWD, OMR: 3 decimals
- Most others: 2 decimals

---

#### 8022: IBAN Inconsistent with BIC

```
has_iban = True?
│
├─ NO → Cannot fire
│
└─ YES → has_bic = True?
          │
          ├─ NO → Cannot fire
          │
          └─ YES → Extract countries:
                    │  iban_country = iban[0:2]
                    │  bic_country = bic[4:6]
                    │
                    └─ bic_iban_match? (countries equal)
                         │
                         ├─ YES → Cannot fire
                         │
                         └─ NO → 8022 FIRES

ELEVATED RISK CHECK:
─────────────────────
id_has_bic_and_nch = True?
│
└─ YES → Compound ID may have conflicting country info
         → 8022 HIGH RISK
```

**Features**: `has_iban`, `has_bic`, `bic_iban_match`, `bic_country`, `iban_country`, `id_has_bic_and_nch`

---

#### 8023: IBAN Inconsistency in Message

```
Count IBANs across all parties
│
└─ iban_count > 1?
    │
    ├─ NO → Cannot fire
    │
    └─ YES → All IBAN values identical?
              │
              ├─ YES → Cannot fire
              │
              └─ NO → 8023 FIRES
```

**Features**: `iban_count`, `ibans_consistent`

---

#### 8024: BBAN Inconsistency in Message

```
Count BBANs across all parties
│
└─ bban_count > 1?
    │
    ├─ NO → Cannot fire
    │
    └─ YES → All BBAN values identical?
              │
              ├─ YES → Cannot fire
              │
              └─ NO → 8024 FIRES
```

**Features**: `bban_count`, `bbans_consistent`

---

#### 8025: Domestic Account Number Inconsistency

```
Count domestic accounts across all parties
│
└─ domestic_account_count > 1?
    │
    ├─ NO → Cannot fire
    │
    └─ YES → All domestic account values identical?
              │
              ├─ YES → Cannot fire
              │
              └─ NO → 8025 FIRES
```

**Features**: `domestic_account_count`, `domestic_accounts_consistent`

---

#### 8026: NCH Inconsistency in Message

```
has_nch = True? (any party)
│
├─ NO → Cannot fire
│
└─ YES → Check inconsistency sources:
          │
          │  SOURCE 1: Multiple NCH values across parties
          │  ─────────────────────────────────────────────
          │  nch_count > 1?
          │  └─ YES → All NCH values identical?
          │            ├─ YES → (continue checks)
          │            └─ NO → 8026 FIRES
          │
          │  SOURCE 2: Compound ID with embedded NCH
          │  ─────────────────────────────────────────────
          │  id_has_bic_and_nch = True?
          │  └─ YES → 8026 HIGH RISK
          │           (e.g., "CUSCAU2S/AU805013" contains
          │            BIC and NCH that may conflict)
          │
          │  SOURCE 3: Multiple NCH sources within party
          │  ─────────────────────────────────────────────
          │  nch_sources > 1?
          │  └─ YES → 8026 ELEVATED RISK
          │           (NCH from AdrBankID vs account vs BIC)
          │
          └─ No inconsistency detected → Cannot fire
```

**Features**: `has_nch`, `nch_count`, `nchs_consistent`, `nch_sources`, `id_has_bic_and_nch`, `id_is_compound`

---

#### 8027: ISO Country Code Inconsistency

```
Count distinct country sources across message:
│  - party.country
│  - bic_country (from BIC chars 5-6)
│  - iban_country (from IBAN chars 1-2)
│  - address_country (from address line 3)
│
└─ country_count > 1?
    │
    ├─ NO → Cannot fire
    │
    └─ YES → All country values consistent?
              │
              ├─ YES → Cannot fire
              │
              └─ NO → 8027 FIRES
```

**Features**: `country_count`, `countries_consistent`, `country`, `bic_country`, `iban_country`, `address_country`

---

#### 8028: BIC4 Inconsistency in Message

```
Extract BIC4 (first 4 chars) from all BICs
│
└─ bic4_count > 1?
    │
    ├─ NO → Cannot fire
    │
    └─ YES → All BIC4 values identical?
              │
              ├─ YES → Cannot fire
              │
              └─ NO → 8028 FIRES
```

**Features**: `bic4_count`, `bic4s_consistent`

---

#### 8029: Account Number Inconsistency

```
Count account numbers across all parties
│
└─ account_count > 1?
    │
    ├─ NO → Cannot fire
    │
    └─ YES → All account values identical?
              │
              ├─ YES → Cannot fire
              │
              └─ NO → 8029 FIRES
```

**Features**: `account_count`, `accounts_consistent`

---

#### 8030: IBAN Derivation Not Supported for Country

```
needs_iban = True?
│
├─ NO → Cannot fire
│
└─ YES → has_iban?
          │
          ├─ YES → Cannot fire
          │
          └─ NO → iban_derivation_supported? (country in derivation list)
                    │
                    ├─ YES → Cannot fire (8004 may fire if derivation fails)
                    │
                    └─ NO → 8030 FIRES
```

**Features**: `needs_iban`, `has_iban`, `iban_derivation_supported`, `country`

**IBAN Derivation Supported Countries**: AT, BE, CH, DE, ES, FI, FR, GB, IE, IT, LU, NL, PT, SE, and others with published derivation rules.

---

#### 8033: CLABE Inconsistency in Message

```
is_clabe = True? (any party, Mexico)
│
├─ NO → Cannot fire
│
└─ YES → clabe_count > 1?
          │
          ├─ NO → Cannot fire
          │
          └─ YES → All CLABE values identical?
                    │
                    ├─ YES → Cannot fire
                    │
                    └─ NO → 8033 FIRES
```

**Features**: `is_clabe`, `clabe_count`, `clabes_consistent`, `country`

---

#### 8124: Invalid Currency

```
has_currency = True?
│
├─ NO → Cannot fire
│
└─ YES → currency in ISO 4217 list?
          │
          ├─ YES → Cannot fire
          │
          └─ NO → 8124 FIRES
```

**Features**: `primary_currency`, `currency_valid`

---

#### 8851: Incorrect Field Size

```
For each field with size constraints:
│
└─ field_length within allowed range?
    │
    ├─ YES → Cannot fire
    │
    └─ NO → 8851 FIRES
```

**Features**: Field-specific length checks

---

#### 8852: Incorrect Length of Attribute

```
has_account = True?
│
├─ NO → Cannot fire
│
└─ YES → Determine expected length by type:
          │
          │  IBAN: 15-34 chars
          │  CLABE: 18 chars
          │  FEDABA: 9 digits
          │  CHIPS ABA: 6 digits
          │  CHIPS UID: 6 digits
          │  Sort Code: 6 digits
          │
          └─ account_length in expected range?
               │
               ├─ YES → Cannot fire
               │
               └─ NO → 8852 FIRES
```

**Features**: `has_account`, `account_length`, `account_type`, `account_needs_length_fix`

---

#### 8853: Incorrect Number Format

```
For each numeric field:
│
└─ value matches expected numeric format?
    │
    ├─ YES → Cannot fire
    │
    └─ NO → 8853 FIRES
```

**Features**: Field-specific format validation

---

#### 8892: Invalid Account Number

```
has_account = True?
│
├─ NO → Cannot fire
│
└─ YES → account_type known?
          │
          ├─ NO → Generic format check
          │        └─ Contains only valid chars?
          │             ├─ YES → Cannot fire
          │             └─ NO → 8892 FIRES
          │
          └─ YES → Apply type-specific validation:
                    │
                    ├─ IBAN → Use 8894 rules
                    ├─ CLABE → 18 digits, checksum
                    ├─ FEDABA → 9 digits, ABA checksum
                    └─ Other → Format rules for type
                         │
                         └─ Valid?
                              ├─ YES → Cannot fire
                              └─ NO → 8892 FIRES
```

**Features**: `has_account`, `account_type`, `account_valid_format`

---

#### 8894: Invalid IBAN

```
has_iban = True?
│
├─ NO → Cannot fire
│
└─ YES → iban_valid_format?
          │
          │  Format: ^[A-Z]{2}[0-9]{2}[A-Z0-9]{11,30}$
          │  - First 2 chars: country code (letters)
          │  - Next 2 chars: check digits (numbers)
          │  - Remaining: BBAN (alphanumeric)
          │
          ├─ NO → 8894 FIRES
          │
          └─ YES → iban_checksum_valid?
                    │
                    │  Algorithm:
                    │  1. Move first 4 chars to end
                    │  2. Convert letters: A=10, B=11, ..., Z=35
                    │  3. Calculate: number mod 97
                    │  4. Valid if result = 1
                    │
                    ├─ NO → 8894 FIRES (or 8898 for checksum-specific)
                    │
                    └─ YES → Cannot fire
```

**Features**: `has_iban`, `iban_valid_format`, `iban_checksum_valid`, `iban_country`

---

#### 8895: Invalid NCH Code

```
has_nch = True?
│
├─ NO → Cannot fire
│
└─ YES → nch_validation_applicable?
          │
          │  Applicable when:
          │  - Party country is US, OR
          │  - No country but has US-style routing (FEDABA/CHIPS)
          │
          ├─ NO → Cannot fire (international, no validation)
          │
          └─ YES → Determine NCH type and validate:
                    │
                    ├─ FEDABA (9 digits)?
                    │    │
                    │    └─ nch_valid? (format)
                    │         │
                    │         ├─ NO → 8895 FIRES
                    │         │
                    │         └─ YES → fedaba_checksum_valid?
                    │                   │
                    │                   │  ABA Checksum Algorithm:
                    │                   │  weights = [3,7,1,3,7,1,3,7,1]
                    │                   │  sum = Σ(digit[i] × weight[i])
                    │                   │  valid if sum mod 10 = 0
                    │                   │
                    │                   ├─ NO → 8895 FIRES
                    │                   │
                    │                   └─ YES → Cannot fire
                    │
                    ├─ CHIPS ABA (6 digits)?
                    │    └─ All digits?
                    │         ├─ NO → 8895 FIRES
                    │         └─ YES → Cannot fire
                    │
                    ├─ CHIPS UID (6 chars)?
                    │    └─ Valid format?
                    │         ├─ NO → 8895 FIRES
                    │         └─ YES → Cannot fire
                    │
                    └─ Sort Code (6 digits)?
                         └─ Valid format?
                              ├─ NO → 8895 FIRES
                              └─ YES → Cannot fire
```

**Features**: `has_nch`, `nch_type`, `nch_valid`, `nch_validation_applicable`, `is_fedaba`, `fedaba_checksum_valid`, `is_chips_aba`, `is_chips_uid`

---

#### 8896: Invalid Domestic Account Number

```
has_domestic_account = True?
│
├─ NO → Cannot fire
│
└─ YES → Get country for validation rules
          │
          └─ domestic_account_valid?
               │
               │  Country-specific rules:
               │  - US: numeric, length varies
               │  - UK: 8 digits (after sort code)
               │  - DE: up to 10 digits
               │  - etc.
               │
               ├─ YES → Cannot fire
               │
               └─ NO → 8896 FIRES
```

**Features**: `has_domestic_account`, `domestic_account_valid`, `country`

---

#### 8897: Invalid BBAN

```
has_bban = True?
│
├─ NO → Cannot fire
│
└─ YES → Get country for BBAN rules
          │
          └─ bban_valid? (country-specific format)
               │
               │  BBAN formats by country:
               │  - DE: 18 chars (8 bank + 10 account)
               │  - FR: 23 chars (5 bank + 5 branch + 11 account + 2 key)
               │  - GB: 14 chars (4 bank + 6 sort + 8 account)
               │  - etc.
               │
               ├─ YES → Cannot fire
               │
               └─ NO → 8897 FIRES
```

**Features**: `has_bban`, `bban_valid`, `country`

---

#### 8898: IBAN Check Digit Failed

```
has_iban = True?
│
├─ NO → Cannot fire
│
└─ YES → iban_valid_format?
          │
          ├─ NO → 8894 FIRES instead (format error)
          │
          └─ YES → iban_checksum_valid?
                    │
                    │  Mod-97 Algorithm:
                    │  1. Rearrange: BBAN + country + check digits
                    │  2. Convert letters to numbers (A=10...Z=35)
                    │  3. Calculate mod 97
                    │  4. Must equal 1
                    │
                    ├─ YES → Cannot fire
                    │
                    └─ NO → 8898 FIRES
```

**Features**: `has_iban`, `iban_valid_format`, `iban_checksum_valid`

**Note**: 8898 is more specific than 8894 - it fires only when format is valid but checksum fails.

---

## 4. 9XXX Repair/Enrichment Codes

9XXX codes indicate ACE **successfully modified** the payment.

### Category A: No Directory (Always Deterministic)

| Code | Description | Precondition | Emit When |
|------|-------------|--------------|-----------|
| **9000** | NCH code cleaned | `has_nch` | `nch_has_dirty_chars` |
| **9001** | D Field deleted | D field present | Field should be removed |
| **9002** | Account cleaned | `has_account` | `account_has_dirty_chars` |
| **9006** | IBAN cleaned | `has_iban` | `iban_needs_formatting` |
| **9009** | Field cleaned | Field present | Has invalid characters |
| **9012** | IBAN formatted | `has_iban` | Needs space formatting |
| **9014** | BBAN cleaned | `has_bban` | `bban_has_dirty_chars` |
| **9015** | Domestic account cleaned | `has_domestic_account` | `account_has_dirty_chars` |
| **9017** | Multiple party info | Party present | `has_multiple_ids` |
| **9018** | Duplicate party removed | `intm_has_multiple` | `intm_has_redundant_info` |
| **9019** | Party identifier cleaned | `has_id` | ID has dirty chars |
| **9020** | Party identifier cleaned | `has_id` | ID has dirty chars |
| **9021** | FEDABA formatted | `is_fedaba` | `nch_needs_formatting` |
| **9022** | Account length fixed | `has_account` | `account_needs_length_fix` |
| **9024** | Push Up performed | `has_intermediary` | Redundancy detected |
| **9025** | CLABE repaired | `is_clabe` | CLABE needs repair |
| **9028** | NCH code stripped | `has_nch` | `nch_in_wrong_field` |
| **9483** | Receiver duplicate removed | Receiver info | Duplicate detected |
| **9487** | Receiver BIC removed | Receiver BIC | Redundant BIC |
| **9918** | Attribute type corrected | Attribute present | Wrong type |
| **9938** | Redundant phrases deleted | Field 72 | Has redundant text |

### Category B: Directory Lookup Required

| Code | Description | Precondition | Directory | Success → | Fail → |
|------|-------------|--------------|-----------|-----------|--------|
| **9004** | IBAN derived | `needs_iban AND NOT has_iban AND has_account` | IBAN Rules | 9004 | 9999 |
| **9005** | BIC from NCH | `has_nch AND NOT has_bic` | BIC Plus | 9005 | 9999 |
| **9007** | Account → IBAN | `needs_iban AND NOT has_iban` | IBAN Rules | 9007 | 9999 |
| **9008** | BIC from IBAN | `has_iban AND NOT has_bic` | BIC Plus | 9008 | 9999 |
| **9013** | Name/Address repair | `has_name AND NOT has_bic` | BIC Plus | 9013 | 9999 |
| **9023** | Fund to Account | Fund ID present | Fund Dir | 9023 | 9999 |
| **9032** | 8-char BIC inserted | BIC needed | BIC Plus | 9032 | 9999 |
| **9475** | D-A from Account | Account in address | CIF | 9475 | 9999 |
| **9476** | D-A from CHIPS ABA | CHIPS in address | CHIPS Dir | 9476 | 9999 |
| **9477** | D-A from FED ABA | FEDABA in address | Fedwire | 9477 | 9999 |
| **9478** | D-A from CHIPS UID | CHIPS UID in address | CHIPS Dir | 9478 | 9999 |
| **9479** | Account cleaned | `has_bic` | BIC Plus | 9479 | 9999 |
| **9480** | Push Down Type 1 | `has_bic` | BIC Plus | 9480 | 9999 |
| **9481** | Push Down Type 2 | `has_bic` | BIC Plus | 9481 | 9999 |
| **9485** | D-A from NCH | NCH in address | BIC Plus | 9485 | 9999 |
| **9486** | A to D performed | `has_bic` | BIC Plus | 9486 | 9999 |
| **9961** | BIC from Name/Addr | `has_name AND NOT has_bic` | BIC Plus | 9961 | 9999 |
| **9985** | BIC from CHIPS ABA | `has_chips_aba` | CHIPS Dir | 9985 | 9999 |
| **9986** | Head Office BIC | `has_bic` (branch) | BIC Plus | 9986 | 9999 |

### ⚠️ Important Cross-Reference Correction

> **9477**: Our precondition_rules.json has `require_true: ["present", "has_bic"]`
> 
> **Official Definition**: "D-A using FED ABA from Name and Address" (requires Fedwire lookup)
> 
> **Current Implementation**: We predict 9477 as BIC enrichment (has_bic → enriched)
> This is a simplification - actual 9477 extracts FEDABA from address and looks it up.
> 
> **Impact**: Our 9477 predictions are for BIC-based enrichment, not FEDABA extraction.

---

### 9XXX Decision Trees - Complete Reference (No Directory)

> **⚠️ Implementation Note**: Decision trees below describe *logical rules*. 
> See Section 7 for actual feature availability.
>
> **Fully Implemented**: 9002, 9006, 9017, 9018, 9019, 9021, 9022, 9024
> **Partially Implemented**: 9000, 9015, 9025
> **Directory Dependent** (predict eligibility only): 9004, 9005, 9007, 9008, 9477, 9479, 9480
> **Not Implemented**: 9001, 9009, 9012, 9014, 9026, 9027, 9028, 9483, 9487, 9488, 9918, 9938, 9987, 9990-9992

---

#### 9000: NCH Code Cleaned

```
has_nch = True?
│
├─ NO → Cannot fire
│
└─ YES → nch_has_dirty_chars?
          │
          │  Dirty chars: anything non-numeric
          │  (spaces, dashes, letters in routing number)
          │
          ├─ NO → Cannot fire
          │
          └─ YES → 9000 FIRES
```

**Features**: `has_nch`, `nch_has_dirty_chars`, `nch_value`

---

#### 9001: D Field Deleted

```
has_d_field = True?
│
├─ NO → Cannot fire
│
└─ YES → d_field_should_delete?
          │
          │  Delete when:
          │  - Redundant information
          │  - Conflicts with other fields
          │
          ├─ NO → Cannot fire
          │
          └─ YES → 9001 FIRES
```

**Features**: `has_d_field`, `d_field_redundant`

---

#### 9002: Account Number Cleaned

```
has_account = True?
│
├─ NO → Cannot fire
│
└─ YES → account_has_dirty_chars?
          │
          │  Dirty chars: non-alphanumeric
          │  (spaces, dashes, slashes, special chars)
          │
          ├─ NO → Cannot fire
          │
          └─ YES → is_domestic_account?
                    │
                    ├─ YES → 9015 FIRES instead
                    │
                    └─ NO → 9002 FIRES
```

**Features**: `has_account`, `account_has_dirty_chars`, `account_has_spaces`, `account_has_special_chars`, `is_domestic_account`

---

#### 9006: IBAN Cleaned

```
has_iban = True?
│
├─ NO → Cannot fire
│
└─ YES → iban_needs_formatting?
          │
          │  Needs formatting when:
          │  - Contains spaces
          │  - Contains lowercase letters
          │  - Contains dashes
          │
          ├─ NO → Cannot fire
          │
          └─ YES → 9006 FIRES
```

**Features**: `has_iban`, `iban_needs_formatting`

---

#### 9009: Field Cleaned

```
For each text field:
│
└─ field_has_invalid_chars?
    │
    │  Invalid chars: control characters,
    │  non-printable, invalid encoding
    │
    ├─ NO → Cannot fire
    │
    └─ YES → 9009 FIRES
```

**Features**: Field-specific dirty char detection

---

#### 9012: IBAN Formatted

```
has_iban = True?
│
├─ NO → Cannot fire
│
└─ YES → iban_needs_space_formatting?
          │
          │  Standard IBAN format:
          │  Groups of 4 chars with spaces
          │  e.g., "DE89 3704 0044 0532 0130 00"
          │
          ├─ NO → Cannot fire
          │
          └─ YES → 9012 FIRES
```

**Features**: `has_iban`, `iban_format_type`

**Note**: 9006 cleans (removes chars), 9012 formats (adds spaces)

---

#### 9014: BBAN Cleaned

```
has_bban = True?
│
├─ NO → Cannot fire
│
└─ YES → bban_has_dirty_chars?
          │
          │  Dirty chars: spaces, dashes,
          │  non-alphanumeric
          │
          ├─ NO → Cannot fire
          │
          └─ YES → 9014 FIRES
```

**Features**: `has_bban`, `bban_has_dirty_chars`

---

#### 9015: Domestic Account Number Cleaned

```
has_domestic_account = True?
│
├─ NO → Cannot fire
│
└─ YES → account_has_dirty_chars?
          │
          │  Dirty chars: spaces, dashes,
          │  non-numeric (for numeric accounts)
          │
          ├─ NO → Cannot fire
          │
          └─ YES → 9015 FIRES
```

**Features**: `has_domestic_account`, `account_has_dirty_chars`

---

#### 9017: Multiple Party Information Present

```
For party:
│
└─ has_multiple_ids = True?
    │
    │  Multiple IDs when:
    │  - More than one ID field populated
    │  - ID field contains multiple values
    │  - Multiple identifiers detected in parsing
    │
    ├─ NO → Cannot fire
    │
    └─ YES → 9017 FIRES
```

**Features**: `has_multiple_ids`

---

#### 9018: Duplicate Party Information Removed

```
intm_count >= 1?
│
├─ NO → Cannot fire
│
└─ YES → intm_has_multiple? (count > 1)
          │
          ├─ NO → Cannot fire (single intermediary is valid)
          │
          └─ YES → Check redundancy signals:
                    │
                    │  SIGNAL 1: Same AdrBankID
                    │  ─────────────────────────
                    │  intm_entries_share_adr_bank_id?
                    │  (e.g., both have AdrBankID="121")
                    │
                    │  SIGNAL 2: Same Country
                    │  ─────────────────────────
                    │  intm_entries_share_country?
                    │  (e.g., both have country="AU")
                    │
                    │  SIGNAL 3: Same BIC Prefix
                    │  ─────────────────────────
                    │  intm_entries_share_bic_prefix?
                    │  (e.g., both start with "WPACAU")
                    │
                    └─ Count TRUE signals >= 2?
                         │
                         │  (intm_has_redundant_info)
                         │
                         ├─ NO → Cannot fire
                         │       (legitimate multi-hop routing)
                         │
                         └─ YES → 9018 FIRES
                                  (duplicate will be removed)
```

**Features**: 
- `intm_count` - Number of IntermediaryBankInf entries
- `intm_has_multiple` - More than one entry
- `intm_entries_share_adr_bank_id` - Same AdrBankID
- `intm_entries_share_country` - Same country
- `intm_entries_share_bic_prefix` - Same BIC (first 6 chars)
- `intm_has_redundant_info` - Composite: 2+ signals true

**Key Insight**: Multiple intermediaries are valid for complex routing. 9018 fires only when entries contain **redundant** information (same bank referenced multiple times).

---

#### 9019/9020: Party Identifier Cleaned

```
has_id = True? (party identifier)
│
├─ NO → Cannot fire
│
└─ YES → id_has_dirty_chars?
          │
          │  Dirty chars in party ID:
          │  - Spaces
          │  - Special characters
          │  - Non-alphanumeric
          │
          ├─ NO → Cannot fire
          │
          └─ YES → 9019 FIRES (or 9020 for alternate)
```

**Features**: `has_id`, `id_has_slash`, `account_has_dirty_chars`

---

#### 9021: FEDABA Code Formatted

```
has_nch = True?
│
├─ NO → Cannot fire
│
└─ YES → is_fedaba = True?
          │
          ├─ NO → Cannot fire
          │
          └─ YES → nch_needs_formatting?
                    │
                    │  FEDABA formatting issues:
                    │  - Wrong length (not 9 digits)
                    │  - Contains dashes or spaces
                    │  - Leading zeros missing
                    │
                    ├─ NO → Cannot fire
                    │
                    └─ YES → 9021 FIRES
```

**Features**: `has_nch`, `is_fedaba`, `nch_needs_formatting`, `nch_type`

---

#### 9022: Account Number Formatted to Required Length

```
has_account = True?
│
├─ NO → Cannot fire
│
└─ YES → account_needs_length_fix?
          │
          │  Length issues:
          │  - Too short (needs padding)
          │  - Too long (needs truncation)
          │  - Expected lengths by type:
          │    IBAN: 15-34, CLABE: 18, FEDABA: 9
          │
          ├─ NO → Cannot fire
          │
          └─ YES → 9022 FIRES
```

**Features**: `has_account`, `account_needs_length_fix`, `account_length`, `account_type`

---

#### 9024: Push Up Performed

```
has_intermediary = True?
│
├─ NO → Cannot fire
│
└─ YES → Check push-up conditions:
          │
          │  CONDITION 1: Redundancy present
          │  ────────────────────────────────
          │  intm_has_redundant_info?
          │  └─ YES → 9024 HIGH CONFIDENCE
          │
          │  CONDITION 2: Multiple entry types
          │  ────────────────────────────────
          │  intm_has_multiple_types?
          │  (e.g., IBK and IB1 both present)
          │  └─ YES → 9024 HIGH CONFIDENCE
          │
          │  CONDITION 3: Info in wrong level
          │  ────────────────────────────────
          │  beneficiary_info_should_move_up?
          │  └─ YES → 9024 LIKELY
          │
          └─ Any condition met?
               │
               ├─ NO → Cannot fire
               │
               └─ YES → 9024 FIRES
```

**Features**: `has_intermediary`, `intm_has_redundant_info`, `intm_has_multiple`, `intm_has_multiple_types`

---

#### 9025: CLABE Repaired

```
is_clabe = True?
│
├─ NO → Cannot fire
│
└─ YES → clabe_needs_repair?
          │
          │  CLABE repair needed when:
          │  - Length not 18 digits
          │  - Contains non-digits
          │  - Checksum incorrect
          │
          ├─ NO → Cannot fire
          │
          └─ YES → 9025 FIRES
```

**Features**: `is_clabe`, `clabe_needs_repair`, `account_length`

---

#### 9026: Purpose Information Found in Field 72

```
has_field_72 = True?
│
├─ NO → Cannot fire
│
└─ YES → field_72_has_purpose_code?
          │
          │  Purpose codes: /PURP/, /BENM/, etc.
          │
          ├─ NO → Cannot fire
          │
          └─ YES → 9026 FIRES
```

**Features**: `has_field_72`, `field_72_content`

---

#### 9027: FCDA Account Reformatted

```
is_fcda_account = True?
│
├─ NO → Cannot fire
│
└─ YES → fcda_needs_formatting?
          │
          ├─ NO → Cannot fire
          │
          └─ YES → 9027 FIRES
```

**Features**: `is_fcda_account`, `fcda_needs_formatting`

---

#### 9028: NCH Code Strip Off

```
has_nch = True?
│
├─ NO → Cannot fire
│
└─ YES → nch_in_wrong_field?
          │
          │  NCH in wrong field when:
          │  - NCH embedded in account number field
          │  - NCH embedded in name/address
          │  - NCH embedded in ID field with other data
          │
          ├─ NO → Cannot fire
          │
          └─ YES → 9028 FIRES
```

**Features**: `has_nch`, `nch_in_wrong_field`, `id_is_compound`

---

#### 9483: Receiver's Duplicate Information Removed

```
has_receiver_info = True?
│
├─ NO → Cannot fire
│
└─ YES → receiver_has_duplicate?
          │
          │  Duplicate when receiver info
          │  matches another party's info
          │
          ├─ NO → Cannot fire
          │
          └─ YES → 9483 FIRES
```

**Features**: `has_receiver_info`, `receiver_duplicate_detected`

---

#### 9487: Receiver's BIC Information Removed

```
has_receiver_bic = True?
│
├─ NO → Cannot fire
│
└─ YES → receiver_bic_redundant?
          │
          │  Redundant when same BIC
          │  appears elsewhere in message
          │
          ├─ NO → Cannot fire
          │
          └─ YES → 9487 FIRES
```

**Features**: `has_receiver_bic`, `receiver_bic_redundant`

---

#### 9488: Partial Push Down Performed

```
push_down_needed = True?
│
├─ NO → Cannot fire
│
└─ YES → full_push_down_possible?
          │
          ├─ YES → 9480/9481 FIRES instead
          │
          └─ NO → partial_push_down_possible?
                   │
                   ├─ NO → Cannot fire
                   │
                   └─ YES → 9488 FIRES
```

**Features**: `push_down_needed`, `push_down_space_available`

---

#### 9918: Attribute Type Auto Corrected

```
has_attribute_type = True?
│
├─ NO → Cannot fire
│
└─ YES → attribute_type_wrong?
          │
          │  Wrong type when:
          │  - IBAN marked as BBAN
          │  - Domestic marked as IBAN
          │  - Type doesn't match format
          │
          ├─ NO → Cannot fire
          │
          └─ YES → 9918 FIRES
```

**Features**: `account_type`, `detected_account_type`

---

#### 9938: Redundant Phrases Deleted from Field 72

```
has_field_72 = True?
│
├─ NO → Cannot fire
│
└─ YES → field_72_has_redundant_phrases?
          │
          │  Redundant phrases:
          │  - Duplicate codewords
          │  - Repeated information
          │  - Standard filler text
          │
          ├─ NO → Cannot fire
          │
          └─ YES → 9938 FIRES
```

**Features**: `has_field_72`, `field_72_redundant_phrases`

---

#### 9987: Code Consistent

```
consistency_check_performed = True?
│
├─ NO → Cannot fire
│
└─ YES → all_codes_consistent?
          │
          ├─ NO → Various 802X codes fire
          │
          └─ YES → 9987 FIRES (confirmation)
```

**Features**: Consistency validation flags

---

#### 9990: Duplicate CDT Info Removed from BBI

```
has_cdt_in_bbi = True?
│
├─ NO → Cannot fire
│
└─ YES → cdt_bbi_is_duplicate?
          │
          │  Duplicate when CDT info in BBI
          │  matches CDT party info
          │
          ├─ NO → Cannot fire
          │
          └─ YES → 9990 FIRES
```

**Features**: `has_cdt_in_bbi`, `cdt_bbi_duplicate`

---

#### 9991: Account Number Pushed Down to BBI from BBK

```
has_account_in_bbk = True?
│
├─ NO → Cannot fire
│
└─ YES → account_should_move_to_bbi?
          │
          ├─ NO → Cannot fire
          │
          └─ YES → 9991 FIRES
```

**Features**: `has_account_in_bbk`, `account_move_needed`

---

#### 9992: NCH Code Translocated to BBI

```
has_nch = True?
│
├─ NO → Cannot fire
│
└─ YES → nch_in_wrong_party?
          │
          │  NCH should move to BBI
          │  (Bank-to-Bank Information)
          │
          ├─ NO → Cannot fire
          │
          └─ YES → 9992 FIRES
```

**Features**: `has_nch`, `nch_location`, `nch_should_move`

---

## 5. Directory Lookups

### Directory Reference

| Directory | Purpose | Lookup Key | Used By |
|-----------|---------|------------|---------|
| **BIC Plus** | Bank identifier cross-reference | BIC, NCH, IBAN prefix, Name+Address | 9005, 9008, 9013, 9032, 9479-9486, 9961, 9985, 9986 |
| **IBAN Derivation** | Country-specific IBAN construction | Country + Account + Bank code | 9004, 9007 |
| **Fedwire** | US bank routing | FEDABA (9-digit) | 9029, 9477 |
| **CHIPS** | CHIPS participant lookup | CHIPS ABA/UID | 9476, 9478, 9985 |
| **CIF** | Customer information | Account, Name | 9475 |
| **Fund** | Fund information | Fund ID | 9023 |

### Estimated Success Rates

| Lookup Type | Success Rate | Notes |
|-------------|--------------|-------|
| IBAN Derivation | ~85% | Fails for non-standard accounts |
| BIC from NCH | ~75% | Some NCH codes not in directory |
| BIC from IBAN | ~90% | High success for major banks |
| BIC from Name/Address | ~50% | Fuzzy matching, lower confidence |
| FEDABA from Address | ~60% | Pattern extraction + lookup |
| Party enrichment | ~80% | Depends on BIC coverage |

---

## 6. Complete Decision Trees

### Master 9XXX Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      9XXX REPAIR DECISION TREE                               │
└─────────────────────────────────────────────────────────────────────────────┘

══════════════════════════════════════════════════════════════════════════════
PHASE 1: CLEANING (No Directory - Always Deterministic)
══════════════════════════════════════════════════════════════════════════════

For each party:

    ACCOUNT CLEANING
    ────────────────
    has_account?
    └─ YES → account_has_dirty_chars?
              └─ YES → is_domestic? → 9015 : 9002
              
    IBAN CLEANING
    ─────────────
    has_iban?
    └─ YES → iban_needs_formatting?
              └─ YES → 9006
              
    NCH CLEANING
    ────────────
    has_nch?
    └─ YES → nch_has_dirty_chars? → 9000
             nch_in_wrong_field? → 9028
             is_fedaba AND nch_needs_formatting? → 9021
             
    DUPLICATE HANDLING
    ──────────────────
    has_multiple_ids? → 9017
    has_duplicate_info? → 9018 (see detailed tree above)
    
    LENGTH FIXING
    ─────────────
    account_needs_length_fix? → 9022


══════════════════════════════════════════════════════════════════════════════
PHASE 2: DERIVATION (Directory Required)
══════════════════════════════════════════════════════════════════════════════

    IBAN DERIVATION
    ───────────────
    needs_iban AND NOT has_iban AND has_account?
    └─ YES → iban_derivation_supported?
              ├─ NO → 8030 (validation)
              └─ YES → LOOKUP → Success: 9004/9007 | Fail: 9999
              
    BIC DERIVATION
    ──────────────
    NOT has_bic?
    └─ YES → has_nch? → LOOKUP → 9005 | 9999
             has_iban? → LOOKUP → 9008 | 9999
             has_name? → LOOKUP → 9961 | 9999


══════════════════════════════════════════════════════════════════════════════
PHASE 3: ENRICHMENT (Directory Required)
══════════════════════════════════════════════════════════════════════════════

    PARTY ENRICHMENT
    ────────────────
    has_bic?
    └─ YES → LOOKUP BIC Plus → 9479/9480/9486 | 9999
    
    D-A EXTRACTION (from Name/Address)
    ──────────────────────────────────
    has_fedaba_in_address? → LOOKUP Fedwire → 9477 | 9999
    has_chips_in_address? → LOOKUP CHIPS → 9476 | 9999
```

---

## 7. Feature Reference

### ⚠️ Implementation Status

This section documents features **actually implemented** in our ML model. Some features in decision trees above are theoretical - check this section for what's available.

### Global Features (Transaction Level) - IMPLEMENTED ✅

| Feature | Type | Description | Used By | Status |
|---------|------|-------------|---------|--------|
| `has_intermediary` | bool | Intermediary exists | 9024 | ✅ |
| `has_beneficiary_bank` | bool | Beneficiary bank exists | Multiple | ✅ |
| `has_bank_info` | bool | Bank info block exists | Multiple | ✅ |
| `is_cross_border` | bool | International payment | Multiple | ✅ |
| `has_instructed_amount` | bool | Instructed amount present | Amount validation | ✅ |
| `amount_mismatch` | bool | Amount values don't match | Amount validation | ✅ |
| `intm_count` | int | Number of intermediary entries | 9018, 9024 | ✅ |
| `intm_has_multiple` | bool | More than one intermediary | 9018 | ✅ |
| `intm_has_redundant_info` | bool | 2+ redundancy signals | 9018, 9024 | ✅ |
| `intm_entries_share_adr_bank_id` | bool | Same AdrBankID | 9018 | ✅ |
| `intm_entries_share_country` | bool | Same country | 9018 | ✅ |
| `intm_entries_share_bic_prefix` | bool | Same BIC prefix | 9018 | ✅ |
| `intm_has_multiple_types` | bool | Different Type values | 9024 | ✅ |

### Global Features - NOT IMPLEMENTED ⚠️

| Feature | Description | Why Not Implemented |
|---------|-------------|-------------------|
| `iban_count` | IBANs across parties | Cross-party aggregation not in parser |
| `bban_count` | BBANs across parties | Cross-party aggregation not in parser |
| `nch_count` | NCH values across parties | Cross-party aggregation not in parser |
| `bic4_count` | BIC4 values across parties | Cross-party aggregation not in parser |
| `account_count` | Accounts across parties | Cross-party aggregation not in parser |
| `*_consistent` | Consistency flags | Requires cross-party comparison |

**Impact**: 8023, 8024, 8025, 8027, 8028, 8029 consistency codes rely on ML model pattern learning rather than deterministic rules.

### Party-Level Features - IMPLEMENTED ✅

| Feature | Type | Description | Used By |
|---------|------|-------------|---------|
| `{party}_present` | bool | Party exists | All |
| `{party}_has_id` | bool | Party has identifier | 9017, 9019 |
| `{party}_has_bic` | bool | BIC provided | 8001, 8005, 8022 |
| `{party}_has_account` | bool | Account provided | 8892, 9002 |
| `{party}_has_name` | bool | Name provided | 9015 |
| `{party}_has_iban` | bool | IBAN provided | 8022, 8894, 9006 |
| `{party}_has_nch` | bool | NCH/routing provided | 8895, 9000, 9005 |
| `{party}_has_adr_bank_id` | bool | AdrBankID present | 9018 |
| `{party}_bic_valid_format` | bool | BIC format valid | 8001 |
| `{party}_bic_valid_country` | bool | BIC country valid | 8001 |
| `{party}_bic_iban_match` | bool | BIC/IBAN countries match | 8022 |
| `{party}_bic_party_country_match` | bool | BIC/party countries match | 8027 |
| `{party}_iban_valid_format` | bool | IBAN format valid | 8894 |
| `{party}_iban_checksum_valid` | bool | IBAN mod-97 passes | 8894, 8898 |
| `{party}_needs_iban` | bool | Country requires IBAN | 8004, 9004 |
| `{party}_nch_valid` | bool | NCH format valid | 8895 |
| `{party}_fedaba_checksum_valid` | bool | ABA checksum valid | 8895 |
| `{party}_nch_validation_applicable` | bool | US domestic | 8895 |
| `{party}_is_clabe` | bool | Mexican CLABE | 8033 |
| `{party}_is_fedaba` | bool | US FEDABA routing | 9021 |
| `{party}_is_chips_aba` | bool | CHIPS ABA | 9476 |
| `{party}_is_chips_uid` | bool | CHIPS UID | 9478 |
| `{party}_account_numeric` | bool | Account all digits | 8892 |
| `{party}_is_domestic` | bool | Domestic party | Multiple |
| `{party}_is_international` | bool | International party | Multiple |
| `{party}_account_has_dirty_chars` | bool | Needs cleaning | 9002 |
| `{party}_account_has_spaces` | bool | Has spaces | 9002 |
| `{party}_account_has_special_chars` | bool | Has special chars | 9002 |
| `{party}_name_has_dirty_chars` | bool | Name needs cleaning | 9015 |
| `{party}_account_needs_length_fix` | bool | Wrong length | 9022 |
| `{party}_iban_needs_formatting` | bool | IBAN needs cleanup | 9006 |
| `{party}_nch_has_dirty_chars` | bool | NCH needs cleaning | 9000 |
| `{party}_nch_needs_formatting` | bool | NCH needs format fix | 9021 |
| `{party}_is_iban_derivable` | bool | Can derive IBAN | 9004 |
| `{party}_has_bban_in_iban_country` | bool | BBAN in IBAN country | 9004 |
| `{party}_is_bic_derivable` | bool | Can derive BIC | 9005, 9008 |
| `{party}_has_nch_no_bic` | bool | Has NCH but no BIC | 9005 |
| `{party}_has_iban_no_bic` | bool | Has IBAN but no BIC | 9008 |
| `{party}_has_multiple_ids` | bool | Multiple IDs present | 9017 |
| `{party}_has_duplicate_info` | bool | Duplicate detected | 9018 |
| `{party}_id_needs_cleaning` | bool | ID needs cleaning | 9019 |
| `{party}_iban_needs_cleaning` | bool | IBAN needs cleaning | 9019 |
| `{party}_any_id_needs_cleaning` | bool | Any ID needs cleaning | 9019 |

### Compound ID Features - IMPLEMENTED ✅ (November 2025)

| Feature | Type | Description | Used By |
|---------|------|-------------|---------|
| `{party}_id_has_slash` | bool | ID contains '/' | 8026 |
| `{party}_id_is_compound` | bool | ID has multiple parts | 8026 |
| `{party}_id_has_bic_pattern` | bool | Part matches BIC format | 8026 |
| `{party}_id_has_nch_pattern` | bool | Part matches NCH format | 8026 |
| `{party}_id_has_bic_and_nch` | bool | Compound BIC+NCH | 8022, 8026 |
| `{party}_id_compound_parts` | int | Number of parts | 8026 |

### Numeric Features - IMPLEMENTED ✅

| Feature | Type | Description |
|---------|------|-------------|
| `primary_amount` | float | Primary transaction amount |
| `instructed_amount` | float | Instructed amount |
| `bank_info_count` | int | Number of bank info blocks |
| `amount_count` | int | Number of amount fields |
| `{party}_address_lines` | int | Number of address lines |
| `{party}_bic_length` | int | BIC length (4/8/11) |
| `{party}_account_length` | int | Account length |
| `{party}_nch_sources` | int | NCH source count |
| `intm_count` | int | Intermediary entry count |
| `{party}_id_compound_parts` | int | Compound ID parts |

### Party Prefixes

| Prefix | Party Type | IFML Path |
|--------|------------|-----------|
| `orig` | Originating Party | OriginatingPartyInf |
| `send` | Sending Bank | SendingBankInf |
| `dbt` | Debit Party | DebitPartyInf |
| `cdt` | Credit Party | CreditPartyInf |
| `intm` | Intermediary Bank | IntermediaryBankInf |
| `bnf` | Beneficiary Bank | BeneficiaryBankInf |
| `ordi` | Ordering Institution | OrderingBankInf |
| `acwi` | Account With Institution | AccountWithInf |

---

## 8. ML Model Strategy

### Actual Model Performance

| Model | Test Accuracy | Notes |
|-------|---------------|-------|
| **8XXX Validation** | **~90%** | High accuracy - most codes are structurally deterministic |
| **9XXX Repair** | **~70%** | Lower accuracy due to directory-dependent codes |

### Why 9XXX Accuracy is Lower

The 70% accuracy for 9XXX codes is expected because:

1. **Directory-Dependent Codes**: 9004, 9005, 9007, 9008, 9477, 9479, 9480, 9485, 9961 require external lookups
   - We can predict **eligibility** (preconditions met) but not **outcome** (lookup success/failure)
   - These codes fire ~70-90% of the time when eligible, explaining the gap

2. **9999 Suppressed**: The generic "Field Repaired" code is unpredictable and excluded

3. **Cross-Party Consistency Not Implemented**: 
   - 8023, 8024, 8025, 8027, 8028, 8029 rely on ML pattern learning
   - We don't have `iban_count`, `nch_count`, etc. aggregation features

### Prediction Strategy Summary

| Category | Codes | Method | Expected Accuracy |
|----------|-------|--------|-------------------|
| **8XXX Validation** | 8001, 8004, 8005, 8022, 8026, 8894, 8895, 8898 | Precondition + ML | **~90%** |
| **8XXX Consistency** | 8023, 8024, 8025, 8027, 8028, 8029 | ML only (no cross-party features) | **~75-85%** |
| **9XXX Cleaning** | 9000, 9002, 9006, 9015, 9019, 9021, 9022 | Precondition + ML | **~85%** |
| **9XXX Duplicate** | 9017, 9018, 9024 | Precondition + ML | **~80%** |
| **9XXX Derivation** | 9004, 9005, 9007, 9008, 9477, 9479 | Predict eligibility | **~60-70%** |
| **Suppressed** | 9999, 9490 | Do not predict | N/A |

### Recommended Improvements for 9XXX Accuracy

1. **Add Cross-Party Aggregation Features**:
   - `iban_count`, `nch_count`, `bic4_count`
   - `ibans_consistent`, `nchs_consistent`
   - Would improve 8023-8029 predictions

2. **Directory Lookup Success Modeling**:
   - Train separate model to predict lookup success probability
   - Could improve 9004, 9005, 9007, 9008 predictions

3. **Historical Success Rate Features**:
   - Add features for bank/country lookup success rates
   - e.g., "IBAN derivation success rate for country X"

### Current Precondition Rules (from precondition_rules.json)

| Code | require_true | require_false | Global? |
|------|--------------|---------------|---------|
| 8001 | `has_bic` | `bic_valid_format`, `bic_valid_country` | No |
| 8004 | `needs_iban` | `has_iban` | No |
| 8022 | `has_iban`, `has_bic` | `bic_iban_match` | No |
| 8026 | `has_nch` | - | No |
| 8894 | `has_iban` | `iban_valid_format`, `iban_checksum_valid` | No |
| 8895 | `has_nch` | `nch_valid`, `fedaba_checksum_valid` | No |
| 8898 | `has_iban` | `iban_checksum_valid` | No |
| 9004 | `present`, `needs_iban` | `has_iban` | No |
| 9005 | `present`, `has_nch` | `has_bic` | No |
| 9006 | `present`, `has_iban` | - | No |
| 9008 | `present`, `has_iban` | `has_bic` | No |
| 9018 | `intm_has_multiple`, `intm_has_redundant_info` | - | **Yes** |
| 9019 | `present`, `any_id_needs_cleaning` | - | No |
| 9024 | `intm_present` | - | **Yes** |

### Threshold Configuration (from prediction_config.json)

| Code | Threshold | Reason |
|------|-----------|--------|
| 8022 | 0.70 | Country mismatch must be certain |
| 8026 | 0.70 | NCH inconsistency needs confidence |
| 8894 | 0.70 | IBAN validation - avoid false positives |
| 8895 | 0.70 | NCH validation - avoid false positives |
| 9004 | 0.70 | Directory dependent |
| 9005 | 0.70 | Directory dependent |
| 9007 | 0.70 | Directory dependent |
| 9008 | 0.70 | Directory dependent |
| 9018 | 0.65 | Redundancy detection |
| 9024 | 0.70 | Push-up complexity |
| 9477 | 0.70 | BIC enrichment |
| 9479 | 0.80 | Party enrichment |
| 9480 | 0.75 | Credit party enrichment |

---

## 9. Suppressed Codes

| Code | Description | Reason |
|------|-------------|--------|
| **9999** | Field Repaired | Generic catch-all; fires when any directory lookup fails |
| **9490** | Fee Code Updated | Requires fee directory; high false positive rate |

**Configuration**:
```json
{
  "suppressed_codes": ["9999", "9490"]
}
```

---

## 10. Quick Reference Card

### 8XXX - When to Emit (✅ = Implemented, ⚠️ = Partial, ❌ = ML Only)

| Code | Condition | Status |
|------|-----------|--------|
| 8001 | `has_bic AND (NOT bic_valid_format OR NOT bic_valid_country)` | ✅ |
| 8004 | `needs_iban AND NOT has_iban` | ✅ |
| 8005 | `has_bic AND NOT bic4_valid` | ⚠️ |
| 8006 | `has_country AND NOT country_valid` | ⚠️ |
| 8022 | `has_iban AND has_bic AND NOT bic_iban_match` | ✅ |
| 8023 | `iban_count > 1 AND NOT ibans_consistent` | ❌ |
| 8026 | `has_nch AND (nch_sources > 1 OR id_has_bic_and_nch)` | ✅ |
| 8027 | `country_count > 1 AND NOT countries_consistent` | ❌ |
| 8028 | `bic4_count > 1 AND NOT bic4s_consistent` | ❌ |
| 8029 | `account_count > 1 AND NOT accounts_consistent` | ❌ |
| 8030 | `needs_iban AND NOT has_iban AND NOT iban_derivation_supported` | ✅ |
| 8894 | `has_iban AND (NOT iban_valid_format OR NOT iban_checksum_valid)` | ✅ |
| 8895 | `has_nch AND nch_validation_applicable AND (NOT nch_valid OR NOT fedaba_checksum_valid)` | ✅ |
| 8898 | `has_iban AND iban_valid_format AND NOT iban_checksum_valid` | ✅ |

### 9XXX No Directory - When to Emit

| Code | Condition | Status |
|------|-----------|--------|
| 9000 | `has_nch AND nch_has_dirty_chars` | ⚠️ |
| 9002 | `has_account AND account_has_dirty_chars` | ✅ |
| 9006 | `has_iban AND iban_needs_formatting` | ✅ |
| 9015 | `has_name AND name_has_dirty_chars` | ✅ |
| 9017 | `has_multiple_ids` | ✅ |
| 9018 | `intm_has_multiple AND intm_has_redundant_info` | ✅ |
| 9019 | `any_id_needs_cleaning` | ✅ |
| 9021 | `is_fedaba AND nch_needs_formatting` | ✅ |
| 9022 | `has_account AND account_needs_length_fix` | ✅ |
| 9024 | `intm_present` (confidence ↑ with `intm_has_redundant_info`) | ✅ |

### 9XXX Directory - When Eligible (Predict Eligibility Only)

| Code | Eligible When | Status |
|------|---------------|--------|
| 9004 | `needs_iban AND NOT has_iban AND has_account` | ✅ |
| 9005 | `has_nch AND NOT has_bic` | ✅ |
| 9007 | `needs_iban AND NOT has_iban` | ✅ |
| 9008 | `has_iban AND NOT has_bic` | ✅ |
| 9477 | `has_bic` (for enrichment) | ✅ |
| 9479 | `has_bic` (party enrichment) | ✅ |
| 9480 | `has_bic` (credit party enrichment) | ✅ |
| 9961 | `has_name AND NOT has_bic` | ❌ |

### Validation Algorithms

**IBAN Mod-97** (Implemented ✅):
```
Rearrange: move first 4 chars to end
Convert letters: A=10, B=11, ... Z=35
Calculate: number mod 97 must equal 1
```

**FEDABA Checksum** (Implemented ✅):
```
Weights: [3, 7, 1, 3, 7, 1, 3, 7, 1]
Sum: Σ(digit × weight)
Valid: sum mod 10 = 0
```

### Suppressed Codes

| Code | Description | Reason |
|------|-------------|--------|
| **9999** | Field Repaired | Generic catch-all, unpredictable |
| **9490** | Fee Code Updated | Requires fee directory |

---

## Summary: Model Accuracy Explanation

| Accuracy | Reason |
|----------|--------|
| **8XXX ~90%** | Most validation rules are structurally deterministic with implemented features |
| **9XXX ~70%** | Directory-dependent codes (9004-9008, 9477-9480) can only predict eligibility, not outcome |

### To Improve 9XXX Accuracy

1. Add cross-party aggregation features (`iban_count`, `nch_count`, etc.)
2. Model directory lookup success probability separately
3. Use historical success rates by bank/country as features

---

*ACE Pelican Error Code Prediction System*
*Cross-referenced with official ACE definitions*
*Model Accuracy: 8XXX ~90%, 9XXX ~70%*
*Last updated: November 2025*
