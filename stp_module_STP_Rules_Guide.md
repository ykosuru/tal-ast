# ACE Pelican Error Code Rules - Complete Reference (Cross-Referenced)

## Table of Contents

1. [Overview](#overview)
2. [Code Series Summary](#code-series-summary)
3. [8XXX Validation Errors](#8xxx-validation-errors)
4. [9XXX Repair/Enrichment Codes](#9xxx-repairenrichment-codes)
5. [Directory Lookups](#directory-lookups)
6. [Complete Decision Trees](#complete-decision-trees)
7. [Feature Reference](#feature-reference)
8. [ML Model Strategy](#ml-model-strategy)
9. [Cross-Reference Notes](#cross-reference-notes)

---

## Overview

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

## Code Series Summary

| Series | Type | Count | Directory Required | Predictability |
|--------|------|-------|-------------------|----------------|
| **6XXX** | Information | ~100 | No | High |
| **7XXX** | Warnings | ~50 | No | High |
| **8XXX** | Validation Errors | ~30 | No | **Very High** |
| **9XXX** | Repairs | ~50 | Mixed | Medium |

---

## 8XXX Validation Errors

All 8XXX codes are **fully deterministic** from input features - no directory lookups required.

### 8XXX Code Reference Table (Cross-Referenced with Official Definitions)

| Code | Official Description | Precondition | Validation Rule | Deterministic? |
|------|---------------------|--------------|-----------------|----------------|
| **8001** | Invalid BIC | `has_bic = True` | BIC format: 8 or 11 chars, `[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}([A-Z0-9]{3})?` | ✅ YES |
| **8003** | File name derivation failed | System-level | File naming rules | ❌ NO (system) |
| **8004** | IBAN cannot be derived | `needs_iban = True` AND `has_account = True` | IBAN derivation attempted and **failed** | ⚠️ PARTIAL - see note |
| **8005** | Invalid BIC4 | `has_bic = True` | BIC4 (first 4 chars) must be `[A-Z]{4}` | ✅ YES |
| **8006** | Invalid country code | Country code present | ISO 3166-1 alpha-2 validation | ✅ YES |
| **8007** | Fractional digits exceed max | Amount present | Currency decimal rules | ✅ YES |
| **8022** | IBAN inconsistent with Account With Institution BIC | `has_iban = True` AND `has_bic = True` | Country codes must match | ✅ YES |
| **8023** | IBAN inconsistency found in message | `iban_count > 1` | Multiple IBANs across parties must match | ✅ YES |
| **8024** | BBAN inconsistency found in message | `bban_count > 1` | Multiple BBANs must match | ✅ YES |
| **8025** | Domestic Account Number inconsistency | `domestic_account_count > 1` | Domestic accounts must match | ✅ YES |
| **8026** | NCH inconsistency found in message | `nch_count > 1` | Multiple NCH values must match | ✅ YES |
| **8027** | ISO Country Code inconsistency | `country_count > 1` | Country codes must be consistent | ✅ YES |
| **8028** | BIC4 inconsistency found in message | `bic4_count > 1` | BIC4 values must match | ✅ YES |
| **8029** | Account Number inconsistency | `account_count > 1` | Account numbers must match | ✅ YES |
| **8030** | IBAN derivation not supported for country | `needs_iban = True` | Country not in IBAN derivation rules | ✅ YES |
| **8033** | CLABE inconsistency found in message | `clabe_count > 1` | CLABEs must match (Mexico) | ✅ YES |
| **8034** | Forced Debit not allowed | Debit indicator | Business rule | ❌ NO (business) |
| **8035** | FCDA account validation failed | FCDA account | FCDA format rules | ⚠️ PARTIAL |
| **8036** | FCDA account name matching failed | FCDA account | Requires CIF lookup | ❌ NO (directory) |
| **8124** | Invalid currency | Currency present | ISO 4217 currency code | ✅ YES |
| **8464** | Target Channel not derived | Routing required | Routing logic | ❌ NO (business) |
| **8465** | Product code not found | Product code required | Product lookup | ❌ NO (directory) |
| **8472** | Fee code not derived | Fee required | Fee logic | ❌ NO (business) |
| **8851** | Incorrect field size | Field present | Field size rules | ✅ YES |
| **8852** | Incorrect length of attribute | Attribute present | Length rules by type | ✅ YES |
| **8853** | Incorrect number format | Numeric field | Number format rules | ✅ YES |
| **8892** | Invalid Account number | `has_account = True` | Account format rules | ✅ YES |
| **8894** | Invalid IBAN | `has_iban = True` | IBAN format + mod-97 | ✅ YES |
| **8895** | Invalid NCH code | `has_nch = True` AND `nch_validation_applicable = True` | NCH format + checksum | ✅ YES |
| **8896** | Invalid Domestic Account Number | `has_domestic_account = True` | Domestic format rules | ✅ YES |
| **8897** | Invalid BBAN | `has_bban = True` | BBAN format for country | ✅ YES |
| **8898** | IBAN Check Digit calculation/validation failed | `has_iban = True` AND `iban_valid_format = True` | mod-97 algorithm only | ✅ YES |
| **8905** | Hash code Mismatch | Hash present | Hash verification | ❌ NO (system) |
| **8906** | Message in wrong flow | Flow indicator | Flow rules | ❌ NO (system) |

### 8004 Special Note

> **Cross-Reference Finding**: The official definition is "IBAN cannot be derived" - this implies:
> 1. IBAN was **required** (country uses IBAN)
> 2. IBAN was **not provided**
> 3. IBAN derivation was **attempted** but **failed**
>
> This means 8004 may be emitted **after** a 9004 directory lookup fails. For ML prediction:
> - Predict 8004 when `needs_iban AND NOT has_iban AND NOT iban_derivation_supported`
> - For countries with derivation support, 8004 depends on directory lookup outcome

### 8XXX Decision Tree

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         8XXX VALIDATION DECISION TREE                        │
└─────────────────────────────────────────────────────────────────────────────┘

For each party in message (ORIG, SEND, DBT, CDT, INTM, BNF, ORDI, ACWI):

    ┌─────────────────────────────────────┐
    │         BIC VALIDATION              │
    └─────────────────────────────────────┘
                    │
                    ▼
            has_bic = True?
                    │
            ┌───────┴───────┐
            │ YES           │ NO
            ▼               ▼
    Validate BIC format     (skip)
            │
            ▼
    ┌─────────────────┐
    │ BIC4 valid?     │
    │ (first 4 chars  │
    │  [A-Z]{4})      │
    └─────────────────┘
            │
    ┌───────┴───────┐
    │ NO            │ YES
    ▼               ▼
  EMIT 8005     Full format valid?
                (8 or 11 chars)
                    │
            ┌───────┴───────┐
            │ NO            │ YES
            ▼               ▼
          EMIT 8001     Country valid?
                        (chars 5-6 in ISO)
                            │
                    ┌───────┴───────┐
                    │ NO            │ YES
                    ▼               ▼
                  EMIT 8001     (BIC OK)


    ┌─────────────────────────────────────┐
    │         IBAN VALIDATION             │
    └─────────────────────────────────────┘
                    │
                    ▼
            has_iban = True?
                    │
            ┌───────┴───────┐
            │ YES           │ NO
            ▼               ▼
    Validate IBAN       needs_iban = True?
            │                   │
            ▼           ┌───────┴───────┐
    ┌─────────────┐     │ YES           │ NO
    │Format valid?│     ▼               ▼
    │(^[A-Z]{2}   │   iban_derivation   (skip)
    │ [0-9]{2}... │   _supported?
    └─────────────┘         │
            │         ┌─────┴─────┐
    ┌───────┴───────┐ │ NO        │ YES
    │ NO            │ ▼           ▼
    ▼               │ EMIT 8030   (eligible for
  EMIT 8894         │             9004 derivation)
                    │
            ┌───────┴───────┐
            │ YES           │
            ▼               
    Checksum valid?         
    (mod-97 = 1)            
            │               
    ┌───────┴───────┐       
    │ NO            │ YES   
    ▼               ▼       
  EMIT 8898     has_bic = True?
                        │
                ┌───────┴───────┐
                │ YES           │ NO
                ▼               ▼
        Countries match?      (IBAN OK)
        (iban[0:2] = bic[4:6])
                │
        ┌───────┴───────┐
        │ NO            │ YES
        ▼               ▼
      EMIT 8022       (IBAN OK)


    ┌─────────────────────────────────────┐
    │         NCH VALIDATION              │
    └─────────────────────────────────────┘
                    │
                    ▼
            has_nch = True?
                    │
            ┌───────┴───────┐
            │ YES           │ NO
            ▼               ▼
    nch_validation_applicable?  (skip)
    (country = 'US')
                    │
            ┌───────┴───────┐
            │ YES           │ NO
            ▼               ▼
    Validate NCH format   (skip - international)
            │
            ▼
    ┌─────────────────┐
    │ NCH type?       │
    └─────────────────┘
            │
    ┌───────┼───────┬───────────┐
    │       │       │           │
    ▼       ▼       ▼           ▼
  FEDABA  CHIPS   SORTCODE    OTHER
  (9 dig) (6 dig) (6 dig)       │
    │       │       │           ▼
    ▼       ▼       ▼       EMIT 8895
  Valid?  Valid?  Valid?
    │       │       │
┌───┴───┐ ┌─┴─┐   ┌─┴─┐
N     Y   N   Y   N   Y
│     │   │   │   │   │
▼     ▼   ▼   ▼   ▼   ▼
8895 Chk  8895 OK 8895 OK
     sum?
      │
  ┌───┴───┐
  N       Y
  │       │
  ▼       ▼
8895    (OK)


    ┌─────────────────────────────────────┐
    │   CROSS-PARTY CONSISTENCY CHECKS    │
    │   (Message-level, not party-level)  │
    └─────────────────────────────────────┘
                    │
                    ▼
            iban_count > 1?
                    │
            ┌───────┴───────┐
            │ YES           │ NO
            ▼               ▼
        IBANs match?        (skip)
                │
        ┌───────┴───────┐
        │ NO            │ YES
        ▼               ▼
      EMIT 8023        (OK)

            bban_count > 1?
                    │
            ┌───────┴───────┐
            │ YES           │ NO
            ▼               ▼
        BBANs match?        (skip)
                │
        ┌───────┴───────┐
        │ NO            │ YES
        ▼               ▼
      EMIT 8024        (OK)

            nch_count > 1?
                    │
            ┌───────┴───────┐
            │ YES           │ NO
            ▼               ▼
        NCHs match?         (skip)
                │
        ┌───────┴───────┐
        │ NO            │ YES
        ▼               ▼
      EMIT 8026        (OK)

            bic4_count > 1?
                    │
            ┌───────┴───────┐
            │ YES           │ NO
            ▼               ▼
        BIC4s match?        (skip)
                │
        ┌───────┴───────┐
        │ NO            │ YES
        ▼               ▼
      EMIT 8028        (OK)

            account_count > 1?
                    │
            ┌───────┴───────┐
            │ YES           │ NO
            ▼               ▼
        Accounts match?     (skip)
                │
        ┌───────┴───────┐
        │ NO            │ YES
        ▼               ▼
      EMIT 8029        (OK)


    ┌─────────────────────────────────────┐
    │      ACCOUNT VALIDATION             │
    └─────────────────────────────────────┘
                    │
                    ▼
            has_account = True?
                    │
            ┌───────┴───────┐
            │ YES           │ NO
            ▼               ▼
        account_type?       (skip)
                │
    ┌───────┬───┴───┬───────┬───────┐
    │       │       │       │       │
    ▼       ▼       ▼       ▼       ▼
  IBAN    BBAN   CLABE  DOMESTIC  OTHER
    │       │       │       │       │
    ▼       ▼       ▼       ▼       ▼
  (see    Valid?  Valid?  Valid?  Valid?
  above)    │       │       │       │
        ┌───┴─┐ ┌───┴─┐ ┌───┴─┐ ┌───┴─┐
        N    Y  N    Y  N    Y  N    Y
        │    │  │    │  │    │  │    │
        ▼    ▼  ▼    ▼  ▼    ▼  ▼    ▼
      8897  OK 8033  OK 8896 OK 8892 OK
```

---

## 9XXX Repair/Enrichment Codes

9XXX codes indicate ACE **successfully modified** the payment. These fall into two categories:

### Category A: No Directory Lookup (Always Succeed When Precondition Met)

| Code | Official Description | Precondition | Deterministic? |
|------|---------------------|--------------|----------------|
| **9000** | NCH code cleaned | `has_nch AND nch_has_dirty_chars` | ✅ YES |
| **9001** | D Field deleted | D field present and should delete | ✅ YES |
| **9002** | Account number cleaned of non-alphanumeric | `has_account AND account_has_dirty_chars` | ✅ YES |
| **9006** | IBAN Cleaned | `has_iban AND iban_needs_formatting` | ✅ YES |
| **9009** | Field Cleaned | Field has invalid characters | ✅ YES |
| **9010** | Field 23E generated | SWIFT-specific | ⚠️ PARTIAL |
| **9012** | IBAN Formatted in field | `has_iban AND needs_space_formatting` | ✅ YES |
| **9014** | BBAN cleaned | `has_bban AND bban_has_dirty_chars` | ✅ YES |
| **9015** | Domestic account number cleaned | `has_domestic_account AND account_has_dirty_chars` | ✅ YES |
| **9017** | Multiple party information present | `has_multiple_ids` | ✅ YES |
| **9018** | Duplicate party information removed | `has_duplicate_info` | ✅ YES |
| **9019** | Party Identifier cleaned | `has_party_id AND party_id_has_dirty_chars` | ✅ YES |
| **9020** | Party Identifier cleaned (alt) | Same as 9019 | ✅ YES |
| **9021** | FEDABA Code correctly formatted | `has_nch AND is_fedaba AND nch_needs_formatting` | ✅ YES |
| **9022** | Account number formatted to required length | `has_account AND account_needs_length_fix` | ✅ YES |
| **9024** | Push Up performed | Info in wrong party field | ✅ YES |
| **9025** | CLABE repaired | `is_clabe AND clabe_needs_repair` | ✅ YES |
| **9026** | Purpose Information found in Field 72 | Purpose code in field 72 | ✅ YES |
| **9027** | FCDA account reformatted | FCDA account present | ✅ YES |
| **9028** | NCH Code strip off | `has_nch AND nch_in_wrong_field` | ✅ YES |
| **9029** | GL account insertion for FED | FED payment needs GL | ❌ NO (directory) |
| **9030** | DDA to GL attribute type conversion | DDA account conversion | ⚠️ PARTIAL |
| **9031** | Trust account to GL number insertion | Trust account conversion | ⚠️ PARTIAL |
| **9032** | 8 char BIC insertion | BIC derivation needed | ❌ NO (directory) |
| **9479** | Account Number Cleaned | `has_bic` (directory enrichment) | ❌ NO (directory) |
| **9483** | Receiver's duplicate information removed | Receiver has duplicate | ✅ YES |
| **9487** | Receiver's BIC information removed | Redundant receiver BIC | ✅ YES |
| **9488** | Partial push down performed | Partial info move needed | ✅ YES |
| **9490-9498** | Fee/Charge field updates | Fee-related | ⚠️ PARTIAL (business) |
| **9918** | Attribute type auto corrected | Wrong attribute type | ✅ YES |
| **9938** | Deleted redundant phrases from Sender to Receiver | Field 72 cleanup | ✅ YES |
| **9962-9969** | Various field updates | Field modifications | ✅ YES |
| **9971** | Negative ref cancels pos ref and benededuct | Reference logic | ⚠️ PARTIAL |
| **9987** | Code Consistent | Consistency validation | ✅ YES |
| **9990-9992** | Duplicate/push operations | Field restructuring | ✅ YES |

### Category B: Directory Lookup Required

| Code | Official Description | Precondition | Directory | Success → | Fail → |
|------|---------------------|--------------|-----------|-----------|--------|
| **9004** | IBAN repaired in Account Number line | `needs_iban AND NOT has_iban AND has_account` | IBAN Derivation | 9004 | 9999 |
| **9005** | BIC replaced by derived BIC | `has_nch AND NOT has_bic` | BIC Plus | 9005 | 9999 |
| **9007** | Account number replaced by IBAN | `needs_iban AND NOT has_iban AND has_account` | IBAN Derivation | 9007 | 9999 |
| **9008** | IBAN to BIC repair done | `has_iban AND NOT has_bic` | BIC Plus | 9008 | 9999 |
| **9013** | Name and Address repair done | `has_name AND NOT has_bic` | BIC Plus (fuzzy) | 9013 | 9999 |
| **9023** | Fund to Account repair done | Fund ID present | Fund Directory | 9023 | 9999 |
| **9475** | D-A using Account No. from Name and Address | Account in name/address | CIF Database | 9475 | 9999 |
| **9476** | D-A using CHIPS ABA from Name and Address | CHIPS ABA in name/address | CHIPS Directory | 9476 | 9999 |
| **9477** | D-A using FED ABA from Name and Address | FEDABA in name/address | Fedwire Directory | 9477 | 9999 |
| **9478** | D-A using CHIPS UID from Name and Address | CHIPS UID in name/address | CHIPS Directory | 9478 | 9999 |
| **9480** | Push Down of type Push 1 performed | `has_bic` | BIC Plus | 9480 | 9999 |
| **9481** | Push Down of type Push 2 performed | `has_bic` | BIC Plus | 9481 | 9999 |
| **9482** | Push Down of type partial Push 2 | `has_bic` | BIC Plus | 9482 | 9999 |
| **9484** | Repaired using info from Ban to Bank Info | B2B info present | BIC Plus | 9484 | 9999 |
| **9485** | D-A using NCH from Name and Address | NCH in name/address | BIC Plus | 9485 | 9999 |
| **9486** | A to D performed | `has_bic` | BIC Plus | 9486 | 9999 |
| **9901** | D-A using BIC retrieved from field | BIC in non-standard field | BIC Plus | 9901 | 9999 |
| **9910** | BIC repaired from BIC found in field 72 | BIC in field 72 | BIC Plus | 9910 | 9999 |
| **9917** | BIC repaired from BIC Plus using field 72 | BIC derivable from 72 | BIC Plus | 9917 | 9999 |
| **9932** | Account number line repaired with Account number | Account repair needed | BIC Plus | 9932 | 9999 |
| **9935** | Account number line repaired by NCH code | NCH available | BIC Plus | 9935 | 9999 |
| **9936** | D-A using BIC from Name and Address | BIC in name/address | BIC Plus | 9936 | 9999 |
| **9961** | BIC Derived from Name and Address | `has_name AND NOT has_bic` | BIC Plus (fuzzy) | 9961 | 9999 |
| **9970** | D-A using BIC from Name and Address (alt) | BIC in name/address | BIC Plus | 9970 | 9999 |
| **9978** | Numeric BIC derived using field 58 | Field 58 has numeric BIC | BIC Plus | 9978 | 9999 |
| **9979** | Numeric BIC derived using field 57 | Field 57 has numeric BIC | BIC Plus | 9979 | 9999 |
| **9980** | Alpha BIC repaired with numeric BIC | BIC repair needed | BIC Plus | 9980 | 9999 |
| **9981** | Merged BIC derived using field 58 | Field 58 merge | BIC Plus | 9981 | 9999 |
| **9982** | Merged BIC derived using field 57 | Field 57 merge | BIC Plus | 9982 | 9999 |
| **9983** | Exception BIC derived using field 58 | Exception handling | BIC Plus | 9983 | 9999 |
| **9984** | Exception BIC derived using field 57 | Exception handling | BIC Plus | 9984 | 9999 |
| **9985** | BIC from CHIPS ABA Repaired | `has_chips_aba` | CHIPS Directory | 9985 | 9999 |
| **9986** | Head Office BIC Repaired | `has_bic` (branch) | BIC Plus | 9986 | 9999 |

### ⚠️ Critical Cross-Reference Finding: 9477

> **IMPORTANT**: The original documentation incorrectly described 9477 as "BIC expanded 8→11".
> 
> **Official Definition**: "D-A using FED ABA from Name and Address"
> 
> This means 9477:
> - Requires Fedwire Directory lookup
> - Extracts FEDABA from name/address fields
> - Is **NOT** a deterministic BIC expansion rule
> - Should be moved to Category B (directory-dependent)

### 9999 - Manual Repair Fallback

| Code | Description | When Emitted |
|------|-------------|--------------|
| **9999** | Field Repaired | Any Category B directory lookup **fails** |

---

## Directory Lookups

### Directory Reference

| Directory | Purpose | Lookup Key | Returns | Used By |
|-----------|---------|------------|---------|---------|
| **BIC Plus Directory** | Bank identifier cross-reference | BIC, NCH, IBAN prefix, Name+Address | Full BIC (11-char), Bank name, Address, Branch info | 9005, 9008, 9013, 9032, 9479-9486, 9901, 9910, 9917, 9932, 9935, 9936, 9961, 9970, 9978-9984, 9986 |
| **IBAN Derivation Rules** | Country-specific IBAN construction | Country + Account + Bank code | Derived IBAN | 9004, 9007 |
| **Fedwire Participant Directory** | US bank routing | FEDABA (9-digit) | Bank info, GL account | 9029, 9477, 8895 (validation) |
| **CHIPS Directory** | CHIPS participant lookup | CHIPS ABA (6-digit), CHIPS UID | BIC, Bank info | 9476, 9478, 9985 |
| **CIF Database** | Customer information | Account number, Name | Customer details | 9475 |
| **Fund Directory** | Fund information | Fund ID | Account number, Fund name | 9023 |
| **SSI Table** | Standard Settlement Instructions | Payment details | Settlement routing | 6058, 6059, 6413 |

### Directory Lookup Success Rates (Estimated)

| Lookup Type | Estimated Success Rate | Notes |
|-------------|----------------------|-------|
| IBAN Derivation | ~85% | Fails for non-standard accounts |
| BIC from NCH | ~75% | Some NCH codes not in directory |
| BIC from IBAN | ~90% | High success for major banks |
| BIC from Name/Address | ~50% | Fuzzy matching, lower confidence |
| FEDABA from Name/Address (9477) | ~60% | Pattern extraction + lookup |
| Party enrichment | ~80% | Depends on BIC coverage |

---

## Complete Decision Trees

### 9XXX Master Decision Tree

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      9XXX REPAIR DECISION TREE                               │
└─────────────────────────────────────────────────────────────────────────────┘

For each party in message:

══════════════════════════════════════════════════════════════════════════════
PHASE 1: CLEANING (No Directory - Always Succeed)
══════════════════════════════════════════════════════════════════════════════

    ┌─────────────────────────────────────┐
    │      ACCOUNT CLEANING               │
    └─────────────────────────────────────┘
                    │
                    ▼
        has_account = True?
                    │
            ┌───────┴───────┐
            │ YES           │ NO
            ▼               ▼
    account_has_dirty_chars?  (skip)
                    │
            ┌───────┴───────┐
            │ YES           │ NO
            ▼               ▼
    is_domestic_account?  account_needs_length_fix?
            │                     │
    ┌───────┴───────┐     ┌───────┴───────┐
    │ YES           │ NO  │ YES           │ NO
    ▼               ▼     ▼               ▼
  EMIT 9015       EMIT  EMIT 9022       (skip)
  (domestic)      9002  (fix length)
                  (general)


    ┌─────────────────────────────────────┐
    │      IBAN CLEANING                  │
    └─────────────────────────────────────┘
                    │
                    ▼
            has_iban = True?
                    │
            ┌───────┴───────┐
            │ YES           │ NO
            ▼               ▼
    iban_needs_formatting?    (skip)
    (spaces, lowercase, dashes)
                    │
            ┌───────┴───────┐
            │ YES           │ NO
            ▼               ▼
        EMIT 9006           (skip)
        (clean IBAN)


    ┌─────────────────────────────────────┐
    │      BBAN CLEANING                  │
    └─────────────────────────────────────┘
                    │
                    ▼
            has_bban = True?
                    │
            ┌───────┴───────┐
            │ YES           │ NO
            ▼               ▼
    bban_has_dirty_chars?     (skip)
                    │
            ┌───────┴───────┐
            │ YES           │ NO
            ▼               ▼
        EMIT 9014           (skip)
        (clean BBAN)


    ┌─────────────────────────────────────┐
    │      NCH CLEANING                   │
    └─────────────────────────────────────┘
                    │
                    ▼
            has_nch = True?
                    │
            ┌───────┴───────┐
            │ YES           │ NO
            ▼               ▼
    nch_has_dirty_chars?      (skip)
                    │
            ┌───────┴───────┐
            │ YES           │ NO
            ▼               ▼
        EMIT 9000         nch_in_wrong_field?
        (clean NCH)               │
                          ┌───────┴───────┐
                          │ YES           │ NO
                          ▼               ▼
                      EMIT 9028       is_fedaba AND
                      (strip off)     nch_needs_formatting?
                                              │
                                      ┌───────┴───────┐
                                      │ YES           │ NO
                                      ▼               ▼
                                  EMIT 9021         (skip)
                                  (format FEDABA)


    ┌─────────────────────────────────────┐
    │      DUPLICATE HANDLING             │
    └─────────────────────────────────────┘
                    │
                    ▼
        has_multiple_ids = True?
                    │
            ┌───────┴───────┐
            │ YES           │ NO
            ▼               ▼
        EMIT 9017       has_duplicate_info?
        (split parties)         │
                        ┌───────┴───────┐
                        │ YES           │ NO
                        ▼               ▼
                    EMIT 9018         (skip)
                    (remove dup)


══════════════════════════════════════════════════════════════════════════════
PHASE 2: IBAN DERIVATION (Directory Required)
══════════════════════════════════════════════════════════════════════════════

    ┌─────────────────────────────────────┐
    │      IBAN DERIVATION                │
    └─────────────────────────────────────┘
                    │
                    ▼
            needs_iban = True?
                    │
            ┌───────┴───────┐
            │ YES           │ NO
            ▼               ▼
        has_iban = False?     (skip - already has IBAN)
                    │
            ┌───────┴───────┐
            │ YES           │ NO
            ▼               ▼
        has_account = True?   (skip - already has IBAN)
                    │
            ┌───────┴───────┐
            │ YES           │ NO
            ▼               ▼
    iban_derivation_supported?  (emit 8004 in validation -
                    │            cannot derive, no account)
            ┌───────┴───────┐
            │ YES           │ NO
            ▼               ▼
    ┌─────────────────┐   EMIT 8030
    │ LOOKUP:         │   (derivation not
    │ IBAN Derivation │    supported for country)
    │ Rules           │
    └─────────────────┘
            │
    ┌───────┴───────┐
    │ FOUND         │ NOT FOUND
    ▼               ▼
  EMIT 9004       EMIT 9999
  or 9007         (+ 8004 in validation)


══════════════════════════════════════════════════════════════════════════════
PHASE 3: BIC DERIVATION (Directory Required)
══════════════════════════════════════════════════════════════════════════════

    ┌─────────────────────────────────────┐
    │      BIC DERIVATION                 │
    └─────────────────────────────────────┘
                    │
                    ▼
            has_bic = False?
                    │
            ┌───────┴───────┐
            │ YES           │ NO
            ▼               ▼
        Can derive BIC?     (skip to enrichment)
                │
    ┌───────────┼───────────┐
    │           │           │
    ▼           ▼           ▼
  has_nch?   has_iban?   has_name?
    │           │           │
    ▼           ▼           ▼
┌────────┐  ┌────────┐  ┌────────┐
│ LOOKUP │  │ LOOKUP │  │ LOOKUP │
│ BIC+   │  │ BIC+   │  │ BIC+   │
│ by NCH │  │ by IBAN│  │ fuzzy  │
└────────┘  └────────┘  └────────┘
    │           │           │
┌───┴───┐   ┌───┴───┐   ┌───┴───┐
FND  NF     FND  NF     FND  NF
 │    │      │    │      │    │
 ▼    ▼      ▼    ▼      ▼    ▼
9005 9999   9008 9999   9961 9999


══════════════════════════════════════════════════════════════════════════════
PHASE 4: EMBEDDED IDENTIFIER EXTRACTION (Directory Required)
══════════════════════════════════════════════════════════════════════════════

    ┌─────────────────────────────────────┐
    │  EXTRACT FROM NAME/ADDRESS          │
    └─────────────────────────────────────┘
                    │
                    ▼
        has_fedaba_in_address?
                    │
            ┌───────┴───────┐
            │ YES           │ NO
            ▼               ▼
    ┌─────────────────┐   has_chips_aba_in_address?
    │ LOOKUP:         │           │
    │ Fedwire Dir     │   ┌───────┴───────┐
    │ (9477)          │   │ YES           │ NO
    └─────────────────┘   ▼               ▼
            │         ┌────────┐    has_chips_uid_in_address?
    ┌───────┴───────┐ │ LOOKUP │          │
    FND          NF   │ CHIPS  │  ┌───────┴───────┐
     │            │   │ (9476) │  │ YES           │ NO
     ▼            ▼   └────────┘  ▼               ▼
   9477         9999      │   ┌────────┐        (skip)
                      ┌───┴───┐│ LOOKUP │
                      FND   NF │ CHIPS  │
                       │     │ │ (9478) │
                       ▼     ▼ └────────┘
                     9476  9999     │
                               ┌───┴───┐
                               FND   NF
                                │     │
                                ▼     ▼
                              9478  9999


══════════════════════════════════════════════════════════════════════════════
PHASE 5: PARTY ENRICHMENT (Directory Required)
══════════════════════════════════════════════════════════════════════════════

    ┌─────────────────────────────────────┐
    │      PARTY ENRICHMENT               │
    └─────────────────────────────────────┘
                    │
                    ▼
            has_bic = True?
                    │
            ┌───────┴───────┐
            │ YES           │ NO
            ▼               ▼
    ┌─────────────────┐     (skip)
    │ LOOKUP:         │
    │ BIC Plus Dir    │
    │ (enrich party)  │
    └─────────────────┘
            │
    ┌───────┴───────┐
    │ FOUND         │ NOT FOUND
    ▼               ▼
  EMIT 9479       EMIT 9999
  9480/9481       (or skip if
  9486            optional)
```

---

## Feature Reference

### Input Features by Category

#### Party Presence
| Feature | Type | Description |
|---------|------|-------------|
| `{party}_present` | bool | Party exists in message |
| `{party}_has_id` | bool | Party has identifier |
| `{party}_has_name` | bool | Party has name |
| `{party}_address_lines` | int | Number of address lines |

#### IBAN Features
| Feature | Type | Description |
|---------|------|-------------|
| `{party}_has_iban` | bool | IBAN is provided |
| `{party}_needs_iban` | bool | Party's country requires IBAN |
| `{party}_iban_country` | str | Country from IBAN (first 2 chars) |
| `{party}_iban_valid_format` | bool | IBAN format valid for country |
| `{party}_iban_checksum_valid` | bool | IBAN passes mod-97 |
| `{party}_iban_needs_formatting` | bool | IBAN needs cleanup (spaces, case) |
| `{party}_iban_derivation_supported` | bool | Country supports IBAN derivation |

#### BBAN Features
| Feature | Type | Description |
|---------|------|-------------|
| `{party}_has_bban` | bool | BBAN is provided |
| `{party}_bban_from_iban` | str | BBAN extracted from IBAN |
| `{party}_bban_has_dirty_chars` | bool | BBAN needs cleaning |

#### BIC Features
| Feature | Type | Description |
|---------|------|-------------|
| `{party}_has_bic` | bool | BIC is provided |
| `{party}_bic` | str | BIC value |
| `{party}_bic_length` | int | BIC length (4, 8, or 11) |
| `{party}_bic4` | str | First 4 chars of BIC |
| `{party}_bic4_valid` | bool | BIC4 format valid ([A-Z]{4}) |
| `{party}_bic_country` | str | Country from BIC (chars 5-6) |
| `{party}_bic_valid_format` | bool | BIC format valid |
| `{party}_bic_valid_country` | bool | BIC country valid ISO |
| `{party}_bic_iban_match` | bool | BIC and IBAN countries match |

#### NCH/Routing Features
| Feature | Type | Description |
|---------|------|-------------|
| `{party}_has_nch` | bool | NCH/routing number provided |
| `{party}_nch_type` | str | FEDABA, CHIPS, SORTCODE, etc. |
| `{party}_nch` | str | The routing number |
| `{party}_nch_valid` | bool | NCH format valid |
| `{party}_is_fedaba` | bool | NCH is 9-digit FEDABA |
| `{party}_fedaba_checksum_valid` | bool | ABA checksum valid |
| `{party}_nch_validation_applicable` | bool | US domestic (validate NCH) |
| `{party}_nch_has_dirty_chars` | bool | NCH needs cleaning |
| `{party}_nch_needs_formatting` | bool | NCH needs reformatting |
| `{party}_nch_in_wrong_field` | bool | NCH embedded in wrong field |

#### Embedded Identifier Features (for D-A codes)
| Feature | Type | Description |
|---------|------|-------------|
| `{party}_has_fedaba_in_address` | bool | FEDABA pattern in name/address |
| `{party}_has_chips_in_address` | bool | CHIPS ABA in name/address |
| `{party}_has_bic_in_address` | bool | BIC pattern in name/address |

#### Account Features
| Feature | Type | Description |
|---------|------|-------------|
| `{party}_has_account` | bool | Account number provided |
| `{party}_account_type` | str | IBAN, BBAN, CLABE, etc. |
| `{party}_has_domestic_account` | bool | Domestic account type |
| `{party}_domestic_account_valid` | bool | Domestic format valid |
| `{party}_is_clabe` | bool | Mexican 18-digit |
| `{party}_is_chips_aba` | bool | CHIPS ABA |
| `{party}_is_chips_uid` | bool | CHIPS UID (6-digit) |
| `{party}_account_has_dirty_chars` | bool | Has non-alphanumeric |
| `{party}_account_has_spaces` | bool | Has spaces |
| `{party}_account_has_special_chars` | bool | Has -/. etc. |
| `{party}_account_needs_length_fix` | bool | Wrong length for type |
| `{party}_account_valid_format` | bool | Format is valid |

#### Duplicate/Multiple Detection
| Feature | Type | Description |
|---------|------|-------------|
| `{party}_has_multiple_ids` | bool | Multiple IDs present |
| `{party}_has_duplicate_info` | bool | Duplicate info detected |

#### Cross-Party Consistency Features (Message-level)
| Feature | Type | Description |
|---------|------|-------------|
| `iban_count` | int | Number of IBANs across all parties |
| `ibans_consistent` | bool | All IBANs match |
| `bban_count` | int | Number of BBANs |
| `bbans_consistent` | bool | All BBANs match |
| `nch_count` | int | Number of NCH values |
| `nchs_consistent` | bool | All NCHs match |
| `bic4_count` | int | Number of BIC4 values |
| `bic4s_consistent` | bool | All BIC4s match |
| `account_count` | int | Number of accounts |
| `accounts_consistent` | bool | All accounts match |
| `country_count` | int | Number of country values |
| `countries_consistent` | bool | Countries consistent |

#### Country/Payment Type
| Feature | Type | Description |
|---------|------|-------------|
| `{party}_country` | str | Party's country code |
| `{party}_country_valid` | bool | Country is valid ISO code |
| `{party}_is_domestic` | bool | US domestic |

### Party Prefixes

| Prefix | Party Type | IFML Path |
|--------|------------|-----------|
| `orig` | Originating Party | OriginatingPartyInf |
| `ordi` | Ordering Institution | OrderingBankInf |
| `send` | Sending Bank | SendingBankInf |
| `dbt` | Debit Party | DebitPartyInf |
| `cdt` | Credit Party | CreditPartyInf |
| `bnf_bank` | Beneficiary Bank | BeneficiaryBankInf |
| `bnf` | Beneficiary Party | BeneficiaryPartyInf |
| `intm` | Intermediary Bank | IntermediaryBankInf |
| `acwi` | Account With Institution | AccountWithInf |

---

## ML Model Strategy

### Summary Table (Cross-Referenced)

| Category | Codes | Directory | Predictability | Strategy |
|----------|-------|-----------|----------------|----------|
| **8XXX Validation** | All deterministic 8XXX | No | **100%** | Direct rule prediction |
| **9XXX Cleaning** | 9000, 9002, 9006, 9014, 9015, 9017, 9018, 9021, 9022, 9028 | No | **100%** | Direct rule prediction |
| **9XXX Derivation** | 9004, 9005, 9007, 9008, 9013, 9032, 9961 | Yes | **~70-85%** | Predict eligibility |
| **9XXX D-A Extraction** | 9475, 9476, 9477, 9478, 9485 | Yes | **~50-70%** | Predict eligibility |
| **9XXX Enrichment** | 9479, 9480, 9481, 9486, 9985, 9986 | Yes | **~80-95%** | Predict eligibility |
| **9999 Fallback** | 9999 | N/A | **Inverse** | Predict when directory lookup likely fails |

### Recommended Approach

```python
# Pseudo-code for prediction strategy (Cross-Referenced Version)

def predict_codes(features):
    codes = []
    
    # ═══════════════════════════════════════════════════════════════════
    # CATEGORY 1: 8XXX - 100% predictable from features
    # ═══════════════════════════════════════════════════════════════════
    
    # === MESSAGE-LEVEL CONSISTENCY CHECKS ===
    
    # 8023: IBAN inconsistency
    if features.get('iban_count', 0) > 1:
        if not features.get('ibans_consistent', True):
            codes.append('8023')
    
    # 8024: BBAN inconsistency
    if features.get('bban_count', 0) > 1:
        if not features.get('bbans_consistent', True):
            codes.append('8024')
    
    # 8026: NCH inconsistency
    if features.get('nch_count', 0) > 1:
        if not features.get('nchs_consistent', True):
            codes.append('8026')
    
    # 8028: BIC4 inconsistency
    if features.get('bic4_count', 0) > 1:
        if not features.get('bic4s_consistent', True):
            codes.append('8028')
    
    # 8029: Account inconsistency
    if features.get('account_count', 0) > 1:
        if not features.get('accounts_consistent', True):
            codes.append('8029')
    
    # === PARTY-LEVEL VALIDATION ===
    
    for party in PARTIES:
        if not features.get(f'{party}_present'):
            continue
        
        # 8001: Invalid BIC
        if features.get(f'{party}_has_bic'):
            if not features.get(f'{party}_bic_valid_format', True):
                codes.append(f'8001_{party}')
            elif not features.get(f'{party}_bic_valid_country', True):
                codes.append(f'8001_{party}')
        
        # 8005: Invalid BIC4
        if features.get(f'{party}_has_bic'):
            if not features.get(f'{party}_bic4_valid', True):
                codes.append(f'8005_{party}')
        
        # 8006: Invalid country code
        country = features.get(f'{party}_country')
        if country and not features.get(f'{party}_country_valid', True):
            codes.append(f'8006_{party}')
        
        # 8022: IBAN/BIC country mismatch
        if features.get(f'{party}_has_iban') and features.get(f'{party}_has_bic'):
            if not features.get(f'{party}_bic_iban_match', True):
                codes.append(f'8022_{party}')
        
        # 8030: IBAN derivation not supported
        if features.get(f'{party}_needs_iban') and not features.get(f'{party}_has_iban'):
            if features.get(f'{party}_has_account'):
                if not features.get(f'{party}_iban_derivation_supported', True):
                    codes.append(f'8030_{party}')
        
        # 8892: Invalid account number
        if features.get(f'{party}_has_account'):
            if not features.get(f'{party}_account_valid_format', True):
                codes.append(f'8892_{party}')
        
        # 8894: Invalid IBAN
        if features.get(f'{party}_has_iban'):
            if not features.get(f'{party}_iban_valid_format', True):
                codes.append(f'8894_{party}')
            elif not features.get(f'{party}_iban_checksum_valid', True):
                codes.append(f'8894_{party}')
        
        # 8895: Invalid NCH code (US domestic only)
        if features.get(f'{party}_has_nch'):
            if features.get(f'{party}_nch_validation_applicable'):
                if not features.get(f'{party}_nch_valid', True):
                    codes.append(f'8895_{party}')
                elif not features.get(f'{party}_fedaba_checksum_valid', True):
                    codes.append(f'8895_{party}')
        
        # 8896: Invalid domestic account
        if features.get(f'{party}_has_domestic_account'):
            if not features.get(f'{party}_domestic_account_valid', True):
                codes.append(f'8896_{party}')
        
        # 8898: IBAN checksum failed (more specific than 8894)
        if features.get(f'{party}_has_iban'):
            if features.get(f'{party}_iban_valid_format', True):
                if not features.get(f'{party}_iban_checksum_valid', True):
                    codes.append(f'8898_{party}')
    
    
    # ═══════════════════════════════════════════════════════════════════
    # CATEGORY 2: 9XXX Cleaning - 100% predictable (No Directory)
    # ═══════════════════════════════════════════════════════════════════
    
    for party in PARTIES:
        if not features.get(f'{party}_present'):
            continue
        
        # 9000: NCH code cleaned
        if features.get(f'{party}_has_nch'):
            if features.get(f'{party}_nch_has_dirty_chars'):
                codes.append(f'9000_{party}')
        
        # 9002: Account number cleaned
        if features.get(f'{party}_has_account'):
            if features.get(f'{party}_account_has_dirty_chars'):
                if not features.get(f'{party}_has_domestic_account'):
                    codes.append(f'9002_{party}')
        
        # 9006: IBAN cleaned
        if features.get(f'{party}_has_iban'):
            if features.get(f'{party}_iban_needs_formatting'):
                codes.append(f'9006_{party}')
        
        # 9014: BBAN cleaned
        if features.get(f'{party}_has_bban'):
            bban = features.get(f'{party}_bban_from_iban', '')
            if bban and re.search(r'[^A-Za-z0-9]', bban):
                codes.append(f'9014_{party}')
        
        # 9015: Domestic account cleaned
        if features.get(f'{party}_has_domestic_account'):
            if features.get(f'{party}_account_has_dirty_chars'):
                codes.append(f'9015_{party}')
        
        # 9017: Multiple party info
        if features.get(f'{party}_has_multiple_ids'):
            codes.append(f'9017_{party}')
        
        # 9018: Duplicate removed
        if features.get(f'{party}_has_duplicate_info'):
            codes.append(f'9018_{party}')
        
        # 9021: FEDABA formatted
        if features.get(f'{party}_has_nch'):
            if features.get(f'{party}_is_fedaba'):
                if features.get(f'{party}_nch_needs_formatting'):
                    codes.append(f'9021_{party}')
        
        # 9022: Account length fixed
        if features.get(f'{party}_has_account'):
            if features.get(f'{party}_account_needs_length_fix'):
                codes.append(f'9022_{party}')
        
        # 9028: NCH code strip off
        if features.get(f'{party}_has_nch'):
            if features.get(f'{party}_nch_in_wrong_field'):
                codes.append(f'9028_{party}')
    
    
    # ═══════════════════════════════════════════════════════════════════
    # CATEGORY 3: 9XXX Directory-Dependent - Predict eligibility only
    # ═══════════════════════════════════════════════════════════════════
    
    for party in PARTIES:
        if not features.get(f'{party}_present'):
            continue
        
        # --- IBAN Derivation Eligibility (9004/9007) ---
        if (features.get(f'{party}_needs_iban') and 
            not features.get(f'{party}_has_iban') and
            features.get(f'{party}_has_account') and
            features.get(f'{party}_iban_derivation_supported', True)):
            
            codes.append(f'9004_{party}')  # Mark as eligible
        
        # --- BIC Derivation Eligibility ---
        if not features.get(f'{party}_has_bic'):
            
            # 9005: BIC from NCH
            if features.get(f'{party}_has_nch'):
                codes.append(f'9005_{party}')  # Eligible
            
            # 9008: BIC from IBAN
            elif features.get(f'{party}_has_iban'):
                codes.append(f'9008_{party}')  # Eligible
            
            # 9961: BIC from Name/Address (fuzzy)
            elif (features.get(f'{party}_has_name') and 
                  features.get(f'{party}_address_lines', 0) >= 2):
                codes.append(f'9961_{party}')  # Eligible
        
        # --- D-A Extraction Eligibility (from Name/Address) ---
        
        # 9477: FED ABA from Name/Address (NOT BIC expansion!)
        if features.get(f'{party}_has_fedaba_in_address'):
            codes.append(f'9477_{party}')  # Eligible for Fedwire lookup
        
        # 9476: CHIPS ABA from Name/Address
        if features.get(f'{party}_has_chips_in_address'):
            codes.append(f'9476_{party}')  # Eligible for CHIPS lookup
    
    return codes
```

---

## Cross-Reference Notes

### Key Corrections from Official ACE Code Definitions

| Issue | Original | Corrected |
|-------|----------|-----------|
| **9477** | "BIC expanded 8→11" | "D-A using FED ABA from Name and Address" - requires Fedwire Directory |
| **8004** | "IBAN required but missing" | "IBAN cannot be derived" - implies derivation was attempted |
| **8023-8029** | Not implemented | Added message-level cross-party consistency checks |
| **8005** | Not implemented | Added BIC4 validation |
| **8006** | Not implemented | Added country code validation |
| **8030** | Not implemented | Added IBAN derivation not supported check |
| **8896** | Not implemented | Added domestic account validation |
| **9014** | Not implemented | Added BBAN cleaning |
| **9015** | Not implemented | Added domestic account cleaning |
| **9028** | Not implemented | Added NCH strip off |

### Codes NOT Predictable from IFML Features

| Code | Reason |
|------|--------|
| 8003 | System-level (file naming) |
| 8034 | Business logic (forced debit rules) |
| 8036 | Requires CIF directory |
| 8464 | Business logic (routing) |
| 8465 | Requires product directory |
| 8472 | Business logic (fee derivation) |
| 8905 | System-level (hash verification) |
| 8906 | System-level (flow routing) |

### Validation Checksum Algorithms

**IBAN Mod-97**:
```python
def validate_iban_checksum(iban):
    rearranged = iban[4:] + iban[:4]
    numeric = ''.join(str(ord(c) - ord('A') + 10) if c.isalpha() else c for c in rearranged)
    return int(numeric) % 97 == 1
```

**ABA/FEDABA Checksum**:
```python
def validate_aba_checksum(aba):
    if len(aba) != 9 or not aba.isdigit():
        return False
    weights = [3, 7, 1, 3, 7, 1, 3, 7, 1]
    total = sum(int(d) * w for d, w in zip(aba, weights))
    return total % 10 == 0
```

---

## Quick Reference Card

### 8XXX - When to Emit (Deterministic)

| Code | Emit When |
|------|-----------|
| 8001 | `has_bic AND (NOT bic_valid_format OR NOT bic_valid_country)` |
| 8005 | `has_bic AND NOT bic4_valid` |
| 8006 | `has_country AND NOT country_valid` |
| 8022 | `has_iban AND has_bic AND NOT bic_iban_match` |
| 8023 | `iban_count > 1 AND NOT ibans_consistent` |
| 8024 | `bban_count > 1 AND NOT bbans_consistent` |
| 8026 | `nch_count > 1 AND NOT nchs_consistent` |
| 8028 | `bic4_count > 1 AND NOT bic4s_consistent` |
| 8029 | `account_count > 1 AND NOT accounts_consistent` |
| 8030 | `needs_iban AND NOT has_iban AND NOT iban_derivation_supported` |
| 8892 | `has_account AND NOT account_valid_format` |
| 8894 | `has_iban AND (NOT iban_valid_format OR NOT iban_checksum_valid)` |
| 8895 | `has_nch AND nch_validation_applicable AND (NOT nch_valid OR NOT fedaba_checksum_valid)` |
| 8896 | `has_domestic_account AND NOT domestic_account_valid` |
| 8898 | `has_iban AND iban_valid_format AND NOT iban_checksum_valid` |

### 9XXX No Directory - When to Emit (Deterministic)

| Code | Emit When |
|------|-----------|
| 9000 | `has_nch AND nch_has_dirty_chars` |
| 9002 | `has_account AND account_has_dirty_chars AND NOT has_domestic_account` |
| 9006 | `has_iban AND iban_needs_formatting` |
| 9014 | `has_bban AND bban_has_dirty_chars` |
| 9015 | `has_domestic_account AND account_has_dirty_chars` |
| 9017 | `has_multiple_ids` |
| 9018 | `has_duplicate_info` |
| 9021 | `has_nch AND is_fedaba AND nch_needs_formatting` |
| 9022 | `has_account AND account_needs_length_fix` |
| 9028 | `has_nch AND nch_in_wrong_field` |

### 9XXX With Directory - When Eligible (Predict Success/Fail)

| Code | Eligible When | Success → | Fail → |
|------|---------------|-----------|--------|
| 9004 | `needs_iban AND NOT has_iban AND has_account AND iban_derivation_supported` | 9004 | 9999 |
| 9005 | `has_nch AND NOT has_bic` | 9005 | 9999 |
| 9008 | `has_iban AND NOT has_bic` | 9008 | 9999 |
| 9013 | `has_name AND NOT has_bic` | 9013 | 9999 |
| 9476 | `has_chips_in_address` | 9476 | 9999 |
| 9477 | `has_fedaba_in_address` | 9477 | 9999 |
| 9961 | `has_name AND NOT has_bic AND address_lines >= 2` | 9961 | 9999 |
| 9985 | `has_chips_aba` | 9985 | 9999 |

---

*Documentation generated for ACE Pelican ML Error Code Prediction System*
*Cross-referenced against official ACE code definitions*
*Last updated: 2025*
