# ACE Pelican Error Code Rules - Complete Reference

## Table of Contents

1. [Overview](#overview)
2. [Code Series Summary](#code-series-summary)
3. [8XXX Validation Errors](#8xxx-validation-errors)
4. [9XXX Repair/Enrichment Codes](#9xxx-repairenrichment-codes)
5. [Directory Lookups](#directory-lookups)
6. [Complete Decision Trees](#complete-decision-trees)
7. [Feature Reference](#feature-reference)
8. [ML Model Strategy](#ml-model-strategy)

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

### 8XXX Code Reference Table

| Code | Description | Precondition | Validation Rule | Emits When |
|------|-------------|--------------|-----------------|------------|
| **8001** | Invalid BIC | `has_bic = True` | BIC format: 8 or 11 chars, `[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}([A-Z0-9]{3})?` | `bic_valid_format = False` OR `bic_valid_country = False` |
| **8004** | IBAN cannot be derived | `needs_iban = True` | IBAN required for country | `has_iban = False` AND derivation not possible |
| **8005** | Invalid BIC4 | `has_bic = True` AND `bic_length ≤ 4` | BIC4 format check | Format invalid |
| **8006** | Invalid country code | Country code present | ISO 3166-1 alpha-2 | Country not in valid list |
| **8007** | Fractional digits exceed max | Amount present | Currency decimal rules | Too many decimal places |
| **8022** | IBAN/BIC country mismatch | `has_iban = True` AND `has_bic = True` | Country codes match | `iban_country ≠ bic_country` |
| **8023** | IBAN inconsistency | `has_iban = True` | Multiple IBAN sources consistent | IBANs from different fields conflict |
| **8024** | BBAN inconsistency | `has_account = True` | Multiple BBAN sources consistent | BBANs conflict |
| **8025** | Domestic account inconsistency | `has_account = True` | Account numbers consistent | Accounts conflict |
| **8026** | NCH inconsistency | `has_nch = True` | Multiple NCH sources consistent | `nch_sources > 1` AND values differ |
| **8027** | ISO country inconsistency | Multiple country sources | Countries consistent | Countries conflict |
| **8028** | BIC4 inconsistency | `has_bic = True` | BIC4 values consistent | BIC4s conflict |
| **8029** | Account number inconsistency | `has_account = True` | Accounts consistent | Accounts conflict |
| **8030** | IBAN derivation not supported | `needs_iban = True` | Country in derivation list | Country not supported |
| **8033** | CLABE inconsistency | `is_clabe = True` | CLABE values consistent | CLABEs conflict |
| **8034** | Forced debit not allowed | Debit indicator | Debit allowed for msg type | Forced debit on restricted type |
| **8035** | FCDA validation failed | FCDA account | FCDA format rules | Format invalid |
| **8036** | FCDA name mismatch | FCDA account | Name matches account | Name doesn't match |
| **8124** | Invalid currency | Currency present | ISO 4217 currency code | Currency not valid |
| **8464** | Target channel not derived | Routing required | Channel derivable | Cannot determine channel |
| **8465** | Product code not found | Product code required | Product exists | Product unknown |
| **8472** | Fee code not derived | Fee required | Fee derivable | Cannot determine fee |
| **8851** | Incorrect field size | Field present | Field size rules | Size out of range |
| **8852** | Incorrect attribute length | Attribute present | Length rules by type | Length mismatch |
| **8853** | Incorrect number format | Numeric field | Number format rules | Format invalid |
| **8892** | Invalid account number | `has_account = True` | Account format rules | Format invalid |
| **8894** | Invalid IBAN | `has_iban = True` | IBAN format + mod-97 | `iban_valid_format = False` OR `iban_checksum_valid = False` |
| **8895** | Invalid NCH code | `has_nch = True` AND `nch_validation_applicable = True` | NCH format + checksum | `nch_valid = False` OR `fedaba_checksum_valid = False` |
| **8896** | Invalid domestic account | `has_account = True` AND `is_domestic = True` | Domestic format rules | Format invalid |
| **8897** | Invalid BBAN | `account_type = BBAN` | BBAN format for country | Format invalid |
| **8898** | IBAN checksum failed | `has_iban = True` | mod-97 algorithm | `iban_checksum_valid = False` |
| **8905** | Hash code mismatch | Hash present | Hash verification | Hash doesn't match |
| **8906** | Message in wrong flow | Flow indicator | Flow rules | Wrong flow |

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
    │ Format valid?   │
    │ (8 or 11 chars, │
    │  correct struct)│
    └─────────────────┘
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
    │(correct len │   EMIT 8004       (skip)
    │ for country)│
    └─────────────┘
            │
    ┌───────┴───────┐
    │ NO            │ YES
    ▼               ▼
  EMIT 8894     Checksum valid?
                (mod-97 = 1)
                    │
            ┌───────┴───────┐
            │ NO            │ YES
            ▼               ▼
    EMIT 8894 or 8898   has_bic = True?
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
    (domestic US payment)
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
  (9 dig) (6 dig) (6 dig)
    │       │       │
    ▼       ▼       ▼
  Format  Format  Format
  valid?  valid?  valid?
    │       │       │
    ▼       ▼       ▼
  ┌─┴─┐   ┌─┴─┐   ┌─┴─┐
  N   Y   N   Y   N   Y
  │   │   │   │   │   │
  ▼   ▼   ▼   ▼   ▼   ▼
8895 Chk 8895 OK 8895 OK
     sum
      │
  ┌───┴───┐
  │ NO    │ YES
  ▼       ▼
EMIT    (NCH OK)
8895


    ┌─────────────────────────────────────┐
    │      CONSISTENCY CHECKS             │
    └─────────────────────────────────────┘
                    │
                    ▼
            nch_sources > 1?
                    │
            ┌───────┴───────┐
            │ YES           │ NO
            ▼               ▼
        NCH values match?   (skip)
                │
        ┌───────┴───────┐
        │ NO            │ YES
        ▼               ▼
      EMIT 8026        (OK)

            Multiple IBANs?
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
    ┌───────┬───┴───┬───────┐
    │       │       │       │
    ▼       ▼       ▼       ▼
  IBAN    BBAN   CLABE   OTHER
    │       │       │       │
    ▼       ▼       ▼       ▼
  (see    Valid   Valid   Valid
  above)  format? format? format?
            │       │       │
        ┌───┴─┐ ┌───┴─┐ ┌───┴─┐
        N    Y  N    Y  N    Y
        │    │  │    │  │    │
        ▼    ▼  ▼    ▼  ▼    ▼
      8897  OK 8033  OK 8892 OK
```

---

## 9XXX Repair/Enrichment Codes

9XXX codes indicate ACE **successfully modified** the payment. These fall into two categories:

### Category A: No Directory Lookup (Always Succeed)

| Code | Description | Precondition | Action | Always Emits |
|------|-------------|--------------|--------|--------------|
| **9000** | NCH code cleaned | `has_nch = True` AND `nch_has_dirty_chars = True` | Remove non-numeric characters | ✓ Yes |
| **9002** | Account cleaned | `present = True` AND `has_account = True` AND `account_has_dirty_chars = True` | Remove non-alphanumeric | ✓ Yes |
| **9006** | IBAN cleaned | `present = True` AND `has_iban = True` AND `iban_needs_formatting = True` | Uppercase, remove spaces/dashes | ✓ Yes |
| **9009** | Field cleaned | Field has invalid characters | Remove invalid characters | ✓ Yes |
| **9012** | IBAN formatted | `present = True` AND `has_iban = True` AND needs formatting | Insert spaces per country rules | ✓ Yes |
| **9014** | BBAN cleaned | `account_type = BBAN` AND has dirty chars | Clean BBAN | ✓ Yes |
| **9015** | Domestic account cleaned | `present = True` AND `has_account = True` AND `account_has_dirty_chars = True` | Clean account | ✓ Yes |
| **9017** | Multiple party info split | `present = True` AND `has_multiple_ids = True` | Split into separate parties | ✓ Yes |
| **9018** | Duplicate info removed | `present = True` AND `has_duplicate_info = True` | Remove duplicate | ✓ Yes |
| **9019** | Party ID cleaned | `present = True` AND `has_id = True` AND dirty chars | Clean party identifier | ✓ Yes |
| **9020** | Party ID cleaned (alt) | `present = True` AND `has_id = True` AND dirty chars | Clean party identifier | ✓ Yes |
| **9021** | FEDABA formatted | `present = True` AND `has_nch = True` AND `nch_needs_formatting = True` | Format to 9 digits | ✓ Yes |
| **9022** | Account length fixed | `present = True` AND `has_account = True` AND `account_needs_length_fix = True` | Pad/trim to expected length | ✓ Yes |
| **9024** | Push up performed | Info in wrong party field | Move to correct field | ✓ Yes |
| **9025** | CLABE repaired | `is_clabe = True` AND needs repair | Fix CLABE format | ✓ Yes |
| **9026** | Purpose info found | Purpose in field 72 | Extract purpose code | ✓ Yes |
| **9027** | FCDA reformatted | FCDA account present | Reformat FCDA | ✓ Yes |
| **9028** | NCH code stripped | NCH embedded in wrong field | Extract and move NCH | ✓ Yes |
| **9030** | DDA to GL conversion | DDA account | Convert to GL format | ✓ Yes |
| **9031** | Trust account to GL | Trust account | Insert GL number | ✓ Yes |
| **9483** | Receiver duplicate removed | Receiver info duplicated | Remove duplicate | ✓ Yes |
| **9487** | Receiver BIC removed | Redundant receiver BIC | Remove BIC | ✓ Yes |
| **9488** | Partial push down | Partial info move needed | Move partial info | ✓ Yes |
| **9490** | Fee code updated | Fee derivation | Update fee code | ✓ Yes |
| **9491** | Tariff fee updated | Tariff calculation | Update tariff | ✓ Yes |
| **9492-9498** | Charge field updates | Charge flags | Update charge fields | ✓ Yes |
| **9918** | Attribute type corrected | Wrong attribute type | Auto-correct type | ✓ Yes |
| **9938** | Redundant phrases deleted | Field 72 has redundant text | Remove phrases | ✓ Yes |
| **9962** | Account number updated | Account needs update | Update account | ✓ Yes |
| **9963** | Account number added | Account missing | Add account | ✓ Yes |
| **9964** | Advice instructions modified | Advice needs change | Modify advice | ✓ Yes |
| **9965** | Charge flag modified | Charge flag incorrect | Fix charge flag | ✓ Yes |
| **9966** | Secondary wire flag modified | Wire flag incorrect | Fix wire flag | ✓ Yes |
| **9967** | PO Box cleared | PO Box present | Clear PO Box | ✓ Yes |
| **9968** | Parameter values added | Parameters missing | Add parameters | ✓ Yes |
| **9969** | Related amounts added | Related amounts missing | Add amounts | ✓ Yes |
| **9971** | Negative ref cancels positive | Conflicting refs | Resolve conflict | ✓ Yes |
| **9987** | Code consistent | Code validation | Mark as consistent | ✓ Yes |
| **9990** | CDT duplicate removed | CDT duplicated in BBI | Remove from BBI | ✓ Yes |
| **9991** | Account pushed to BBI | Account in BBK | Move to BBI | ✓ Yes |
| **9992** | NCH translocated to BBI | NCH in wrong location | Move to BBI | ✓ Yes |

### Category B: Directory Lookup Required

| Code | Description | Precondition | Directory | If Found | If Not Found |
|------|-------------|--------------|-----------|----------|--------------|
| **9004** | IBAN derived (credit) | `cdt_present` AND `cdt_needs_iban` AND NOT `cdt_has_iban` AND `cdt_has_account` | IBAN Derivation Rules | **9004** | **9999** |
| **9005** | BIC from NCH | `present` AND `has_nch` AND NOT `has_bic` | BIC Plus Directory | **9005** | **9999** |
| **9007** | IBAN derived (beneficiary) | `bnf_present` AND `bnf_needs_iban` AND NOT `bnf_has_iban` AND `bnf_has_account` | IBAN Derivation Rules | **9007** | **9999** |
| **9008** | BIC from IBAN | `present` AND `has_iban` AND NOT `has_bic` | BIC Plus Directory | **9008** | **9999** |
| **9013** | Name/Address repair | `present` AND `has_name` AND NOT `has_bic` | BIC Plus Directory | **9013** | **9999** |
| **9023** | Fund to Account repair | Fund ID present | Fund Directory | **9023** | **9999** |
| **9029** | GL account for FED | FED payment | Fedwire Directory | **9029** | **9999** |
| **9032** | 8-char BIC inserted | `present` AND NOT `has_bic` AND derivable | BIC Plus Directory | **9032** | **9999** |
| **9475** | D-A using account from name | Account in name/address | CIF Database | **9475** | **9999** |
| **9476** | D-A using CHIPS ABA from name | CHIPS ABA in name/address | CHIPS Database | **9476** | **9999** |
| **9477** | BIC expanded 8→11 | `present` AND `has_bic` AND `bic_length = 8` | BIC Plus Directory | **9477** | **9999** |
| **9478** | D-A using CHIPS UID from name | CHIPS UID in name/address | CHIPS Database | **9478** | **9999** |
| **9479** | Account cleaned (from directory) | `present` AND `has_bic` | BIC Plus Directory | **9479** | **9999** |
| **9480** | Push down type 1 | `present` AND `has_bic` | BIC Plus Directory | **9480** | **9999** |
| **9481** | Push down type 2 | `present` AND `has_bic` | BIC Plus Directory | **9481** | **9999** |
| **9482** | Partial push down type 2 | `present` AND `has_bic` | BIC Plus Directory | **9482** | **9999** |
| **9484** | Repaired from Bank to Bank info | B2B info present | BIC Plus Directory | **9484** | **9999** |
| **9485** | D-A using NCH from name | NCH in name/address | BIC Plus Directory | **9485** | **9999** |
| **9486** | A to D performed | `present` AND `has_bic` | BIC Plus Directory | **9486** | **9999** |
| **9901** | D-A using BIC from field | BIC in non-standard field | BIC Plus Directory | **9901** | **9999** |
| **9910** | BIC repaired from field 72 | BIC in field 72 | BIC Plus Directory | **9910** | **9999** |
| **9917** | BIC from BIC Plus using field 72 | BIC derivable from 72 | BIC Plus Directory | **9917** | **9999** |
| **9932** | Account line repaired | Account needs repair | BIC Plus Directory | **9932** | **9999** |
| **9935** | Account repaired by NCH | NCH available | BIC Plus Directory | **9935** | **9999** |
| **9936** | D-A using BIC from name | BIC in name/address | BIC Plus Directory | **9936** | **9999** |
| **9961** | BIC from name/address | `present` AND `has_name` AND NOT `has_bic` | BIC Plus Directory (fuzzy) | **9961** | **9999** |
| **9970** | D-A using BIC from name (alt) | BIC in name/address | BIC Plus Directory | **9970** | **9999** |
| **9978** | Numeric BIC from field 58 | Field 58 has numeric BIC | BIC Plus Directory | **9978** | **9999** |
| **9979** | Numeric BIC from field 57 | Field 57 has numeric BIC | BIC Plus Directory | **9979** | **9999** |
| **9980** | Alpha BIC repaired with numeric | BIC repair needed | BIC Plus Directory | **9980** | **9999** |
| **9981** | Merged BIC from field 58 | Field 58 merge | BIC Plus Directory | **9981** | **9999** |
| **9982** | Merged BIC from field 57 | Field 57 merge | BIC Plus Directory | **9982** | **9999** |
| **9983** | Exception BIC from field 58 | Exception handling | BIC Plus Directory | **9983** | **9999** |
| **9984** | Exception BIC from field 57 | Exception handling | BIC Plus Directory | **9984** | **9999** |
| **9985** | BIC from CHIPS ABA | `present` AND `is_chips_aba` | CHIPS Directory | **9985** | **9999** |
| **9986** | Head office BIC repaired | `present` AND `has_bic` (branch) | BIC Plus Directory | **9986** | **9999** |

### 9999 - Manual Repair Fallback

| Code | Description | When Emitted |
|------|-------------|--------------|
| **9999** | Field repaired (manual queue) | Any Category B directory lookup **fails** |

---

## Directory Lookups

### Directory Reference

| Directory | Purpose | Lookup Key | Returns | Used By |
|-----------|---------|------------|---------|---------|
| **BIC Plus Directory** | Bank identifier cross-reference | BIC, NCH, IBAN prefix, Name+Address | Full BIC (11-char), Bank name, Address, Branch info | 9005, 9008, 9013, 9032, 9477, 9479-9486, 9901, 9910, 9917, 9932, 9935, 9936, 9961, 9970, 9978-9984, 9986 |
| **IBAN Derivation Rules** | Country-specific IBAN construction | Country + Account + Bank code | Derived IBAN | 9004, 9007 |
| **Fedwire Participant Directory** | US bank routing | FEDABA (9-digit) | Bank info, GL account | 9029, 8895 (validation) |
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
| BIC 8→11 expansion | ~95% | Most 8-char BICs have 11-char |
| BIC from Name/Address | ~50% | Fuzzy matching, lower confidence |
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
        EMIT 9002         account_has_spaces?
        (clean account)           │
                          ┌───────┴───────┐
                          │ YES           │ NO
                          ▼               ▼
                      EMIT 9002       account_needs_length_fix?
                                              │
                                      ┌───────┴───────┐
                                      │ YES           │ NO
                                      ▼               ▼
                                  EMIT 9022         (skip)
                                  (fix length)


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
                    │
            ┌───────┴───────┐
            │ YES           │ NO
            ▼               ▼
        EMIT 9006           (skip)
        (clean IBAN)


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
        EMIT 9000         nch_needs_formatting?
        (clean NCH)               │
                          ┌───────┴───────┐
                          │ YES           │ NO
                          ▼               ▼
                      EMIT 9021         (skip)
                      (format NCH)


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
    ┌─────────────────┐   (cannot derive - emit 8004 in validation)
    │ LOOKUP:         │
    │ IBAN Derivation │
    │ Rules           │
    │                 │
    │ Key: country +  │
    │      account +  │
    │      bank code  │
    └─────────────────┘
            │
    ┌───────┴───────┐
    │ FOUND         │ NOT FOUND
    ▼               ▼
  Party type?     EMIT 9999
        │         (manual repair)
    ┌───┴───┐
    │       │
    ▼       ▼
  CDT     BNF
    │       │
    ▼       ▼
EMIT    EMIT
9004    9007


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
        Can derive BIC?     (skip to BIC enrichment)
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
PHASE 4: BIC ENRICHMENT (Directory Required)
══════════════════════════════════════════════════════════════════════════════

    ┌─────────────────────────────────────┐
    │      BIC ENRICHMENT                 │
    └─────────────────────────────────────┘
                    │
                    ▼
            has_bic = True?
                    │
            ┌───────┴───────┐
            │ YES           │ NO
            ▼               ▼
        bic_length = 8?     (skip)
                │
        ┌───────┴───────┐
        │ YES           │ NO
        ▼               ▼
    ┌────────────┐    Needs enrichment?
    │ LOOKUP:    │          │
    │ BIC Plus   │    ┌─────┴─────┐
    │ 8→11 char  │    │ YES       │ NO
    └────────────┘    ▼           ▼
        │         ┌────────┐    (skip)
    ┌───┴───┐     │ LOOKUP │
    FND   NF      │ BIC+   │
     │     │      │ enrich │
     ▼     ▼      └────────┘
   9477  9999         │
                  ┌───┴───┐
                  FND   NF
                   │     │
                   ▼     ▼
                 9479  9999
                 9480


══════════════════════════════════════════════════════════════════════════════
PHASE 5: SPECIAL REPAIRS (Directory Required)
══════════════════════════════════════════════════════════════════════════════

    ┌─────────────────────────────────────┐
    │      CHIPS/FEDWIRE SPECIFIC         │
    └─────────────────────────────────────┘
                    │
                    ▼
            is_chips_aba = True?
                    │
            ┌───────┴───────┐
            │ YES           │ NO
            ▼               ▼
    ┌─────────────────┐   is_fedaba = True?
    │ LOOKUP:         │         │
    │ CHIPS Directory │   ┌─────┴─────┐
    └─────────────────┘   │ YES       │ NO
            │             ▼           ▼
    ┌───────┴───────┐  ┌────────┐   (skip)
    FND          NF    │ LOOKUP │
     │            │    │ Fedwire│
     ▼            ▼    └────────┘
   9985         9999       │
                       ┌───┴───┐
                       FND   NF
                        │     │
                        ▼     ▼
                      9029  9999
```

### Combined 8XXX + 9XXX Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    COMPLETE ACE PROCESSING FLOW                              │
└─────────────────────────────────────────────────────────────────────────────┘

                    ┌──────────────────┐
                    │  PAYMENT INPUT   │
                    │     (IFML)       │
                    └────────┬─────────┘
                             │
                             ▼
        ┌────────────────────────────────────────┐
        │           PHASE 1: PARSING             │
        │                                        │
        │  Extract: Parties, Amounts, Routing    │
        │  Emit: 6XXX informational codes        │
        └────────────────────┬───────────────────┘
                             │
                             ▼
        ┌────────────────────────────────────────┐
        │        PHASE 2: VALIDATION             │
        │                                        │
        │  For each party:                       │
        │    • BIC validation    → 8001          │
        │    • IBAN validation   → 8894, 8898    │
        │    • NCH validation    → 8895          │
        │    • Consistency       → 8022-8029     │
        │    • IBAN required?    → 8004, 8030    │
        │                                        │
        │  Emit: 8XXX errors, 7XXX warnings      │
        └────────────────────┬───────────────────┘
                             │
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
     ┌────────────────┐            ┌────────────────┐
     │ FATAL ERRORS   │            │  REPAIRABLE    │
     │                │            │                │
     │ • 8001 Invalid │            │ Continue to    │
     │   BIC format   │            │ repair phase   │
     │ • 8894 Invalid │            │                │
     │   IBAN         │            │                │
     │ • etc.         │            │                │
     └───────┬────────┘            └───────┬────────┘
             │                             │
             ▼                             ▼
     ┌────────────────┐     ┌────────────────────────────────┐
     │ REJECT or      │     │      PHASE 3: CLEANING         │
     │ MANUAL QUEUE   │     │      (No Directory)            │
     └────────────────┘     │                                │
                            │  • Account cleaning  → 9002    │
                            │  • IBAN formatting   → 9006    │
                            │  • NCH formatting    → 9021    │
                            │  • Length fixes      → 9022    │
                            │  • Duplicate removal → 9017-18 │
                            │                                │
                            │  Always succeed if precondition│
                            │  met                           │
                            └───────────────┬────────────────┘
                                            │
                                            ▼
                            ┌────────────────────────────────┐
                            │    PHASE 4: IBAN DERIVATION    │
                            │    (Directory Required)        │
                            │                                │
                            │  needs_iban AND NOT has_iban?  │
                            │         │                      │
                            │    ┌────┴────┐                 │
                            │    ▼         ▼                 │
                            │  LOOKUP    SKIP                │
                            │    │                           │
                            │  ┌─┴─┐                         │
                            │  ▼   ▼                         │
                            │ 9004 9999                      │
                            │ 9007                           │
                            └───────────────┬────────────────┘
                                            │
                                            ▼
                            ┌────────────────────────────────┐
                            │     PHASE 5: BIC DERIVATION    │
                            │     (Directory Required)       │
                            │                                │
                            │  NOT has_bic AND derivable?    │
                            │         │                      │
                            │    ┌────┴────┐                 │
                            │    ▼         ▼                 │
                            │  LOOKUP    SKIP                │
                            │    │                           │
                            │  ┌─┴─┐                         │
                            │  ▼   ▼                         │
                            │ 9005 9999                      │
                            │ 9008                           │
                            │ 9961                           │
                            └───────────────┬────────────────┘
                                            │
                                            ▼
                            ┌────────────────────────────────┐
                            │     PHASE 6: BIC ENRICHMENT    │
                            │     (Directory Required)       │
                            │                                │
                            │  has_bic AND needs enrichment? │
                            │         │                      │
                            │    ┌────┴────┐                 │
                            │    ▼         ▼                 │
                            │  LOOKUP    SKIP                │
                            │    │                           │
                            │  ┌─┴─┐                         │
                            │  ▼   ▼                         │
                            │ 9477 9999                      │
                            │ 9479                           │
                            │ 9480                           │
                            └───────────────┬────────────────┘
                                            │
                                            ▼
                            ┌────────────────────────────────┐
                            │         PHASE 7: OUTPUT        │
                            │                                │
                            │  • Enriched payment message    │
                            │  • All emitted codes           │
                            │  • Routing determined          │
                            └────────────────────────────────┘
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
| `{party}_iban_needs_formatting` | bool | IBAN needs cleanup |
| `{party}_is_iban_derivable` | bool | Has data to derive IBAN |
| `{party}_has_bban_in_iban_country` | bool | Has BBAN in IBAN country |

#### BIC Features
| Feature | Type | Description |
|---------|------|-------------|
| `{party}_has_bic` | bool | BIC is provided |
| `{party}_bic` | str | BIC value |
| `{party}_bic_length` | int | BIC length (4, 8, or 11) |
| `{party}_bic_country` | str | Country from BIC (chars 5-6) |
| `{party}_bic_valid_format` | bool | BIC format valid |
| `{party}_bic_valid_country` | bool | BIC country valid ISO |
| `{party}_bic_iban_match` | bool | BIC and IBAN countries match |
| `{party}_is_bic_derivable` | bool | Has data to derive BIC |
| `{party}_has_nch_no_bic` | bool | Has NCH but no BIC |
| `{party}_has_iban_no_bic` | bool | Has IBAN but no BIC |

#### NCH/Routing Features
| Feature | Type | Description |
|---------|------|-------------|
| `{party}_has_nch` | bool | NCH/routing number provided |
| `{party}_nch_type` | str | FEDABA, CHIPS, SORTCODE, etc. |
| `{party}_nch_value` | str | The routing number |
| `{party}_nch_valid` | bool | NCH format valid |
| `{party}_fedaba_checksum_valid` | bool | ABA checksum valid |
| `{party}_nch_sources` | int | Count of NCH sources |
| `{party}_nch_validation_applicable` | bool | US domestic (validate NCH) |
| `{party}_nch_has_dirty_chars` | bool | NCH needs cleaning |
| `{party}_nch_needs_formatting` | bool | NCH needs reformatting |

#### Account Features
| Feature | Type | Description |
|---------|------|-------------|
| `{party}_has_account` | bool | Account number provided |
| `{party}_account_type` | str | IBAN, BBAN, CLABE, etc. |
| `{party}_account_length` | int | Account number length |
| `{party}_account_numeric` | bool | All digits |
| `{party}_is_clabe` | bool | Mexican 18-digit |
| `{party}_is_fedaba` | bool | US 9-digit routing |
| `{party}_is_chips_aba` | bool | CHIPS ABA |
| `{party}_is_chips_uid` | bool | CHIPS UID (6-digit) |
| `{party}_account_has_dirty_chars` | bool | Has non-alphanumeric |
| `{party}_account_has_spaces` | bool | Has spaces |
| `{party}_account_has_special_chars` | bool | Has -/. etc. |
| `{party}_account_needs_length_fix` | bool | Wrong length for type |

#### Duplicate/Multiple Detection
| Feature | Type | Description |
|---------|------|-------------|
| `{party}_has_multiple_ids` | bool | Multiple IDs present |
| `{party}_has_duplicate_info` | bool | Duplicate info detected |

#### Payment Type
| Feature | Type | Description |
|---------|------|-------------|
| `{party}_is_domestic` | bool | US domestic |
| `{party}_is_international` | bool | Non-US |
| `is_cross_border` | bool | Originator ≠ Beneficiary country |

### Party Prefixes

| Prefix | Party Type | Common Suffix |
|--------|------------|---------------|
| `orig` | Originating Party | ORGPTY |
| `send` | Sending Bank | SNDBNK |
| `dbt` | Debit Party | DBTPTY |
| `cdt` | Credit Party | CDTPTY |
| `intm` | Intermediary Bank | INTBNK |
| `bnf` | Beneficiary Bank | BNFBNK, BNPPTY |
| `ordi` | Ordering Institution | ORDBNK |
| `acwi` | Account With Institution | ACWBNK |

---

## ML Model Strategy

### Summary Table

| Category | Codes | Directory | Predictability | Strategy |
|----------|-------|-----------|----------------|----------|
| **8XXX Validation** | All 8XXX | No | **100%** | Direct rule prediction |
| **9XXX Cleaning** | 9000, 9002, 9006, 9014, 9015, 9017, 9018, 9021, 9022 | No | **100%** | Direct rule prediction |
| **9XXX Derivation** | 9004, 9005, 9007, 9008, 9032, 9961 | Yes | **~70-85%** | Predict eligibility |
| **9XXX Enrichment** | 9477, 9479, 9480, 9486 | Yes | **~80-95%** | Predict eligibility |
| **9999 Fallback** | 9999 | N/A | **Inverse** | Predict when directory lookup likely fails |

### Recommended Approach

```python
# Pseudo-code for prediction strategy

def predict_codes(features):
    codes = []
    
    # ═══════════════════════════════════════════════════════════════════
    # CATEGORY 1: 8XXX - 100% predictable from features
    # ═══════════════════════════════════════════════════════════════════
    
    for party in PARTIES:
        # BIC validation
        if features[f'{party}_has_bic']:
            if not features[f'{party}_bic_valid_format'] or \
               not features[f'{party}_bic_valid_country']:
                codes.append(f'8001_{party_suffix}')
        
        # IBAN validation
        if features[f'{party}_has_iban']:
            if not features[f'{party}_iban_valid_format']:
                codes.append(f'8894_{party_suffix}')
            elif not features[f'{party}_iban_checksum_valid']:
                codes.append(f'8898_{party_suffix}')
            
            if features[f'{party}_has_bic']:
                if not features[f'{party}_bic_iban_match']:
                    codes.append(f'8022_{party_suffix}')
        
        # IBAN required but missing
        if features[f'{party}_needs_iban'] and not features[f'{party}_has_iban']:
            codes.append(f'8004_{party_suffix}')
        
        # NCH validation (domestic only)
        if features[f'{party}_has_nch'] and features[f'{party}_nch_validation_applicable']:
            if not features[f'{party}_nch_valid'] or \
               not features[f'{party}_fedaba_checksum_valid']:
                codes.append(f'8895_{party_suffix}')
        
        # Consistency checks
        if features[f'{party}_nch_sources'] > 1:
            codes.append(f'8026_{party_suffix}')
    
    # ═══════════════════════════════════════════════════════════════════
    # CATEGORY 2: 9XXX Cleaning - 100% predictable
    # ═══════════════════════════════════════════════════════════════════
    
    for party in PARTIES:
        if features[f'{party}_present']:
            
            # Account cleaning
            if features[f'{party}_has_account']:
                if features[f'{party}_account_has_dirty_chars']:
                    codes.append(f'9002_{party_suffix}')
                if features[f'{party}_account_needs_length_fix']:
                    codes.append(f'9022_{party_suffix}')
            
            # IBAN formatting
            if features[f'{party}_has_iban']:
                if features[f'{party}_iban_needs_formatting']:
                    codes.append(f'9006_{party_suffix}')
            
            # NCH cleaning
            if features[f'{party}_has_nch']:
                if features[f'{party}_nch_has_dirty_chars']:
                    codes.append(f'9000_{party_suffix}')
                if features[f'{party}_nch_needs_formatting']:
                    codes.append(f'9021_{party_suffix}')
            
            # Duplicate handling
            if features[f'{party}_has_multiple_ids']:
                codes.append(f'9017_{party_suffix}')
            if features[f'{party}_has_duplicate_info']:
                codes.append(f'9018_{party_suffix}')
    
    # ═══════════════════════════════════════════════════════════════════
    # CATEGORY 3: 9XXX Directory-Dependent - Predict eligibility
    # ═══════════════════════════════════════════════════════════════════
    
    for party in PARTIES:
        if not features[f'{party}_present']:
            continue
        
        # IBAN derivation eligibility
        if features[f'{party}_needs_iban'] and \
           not features[f'{party}_has_iban'] and \
           features[f'{party}_has_account']:
            
            # Use ML model probability or heuristic
            if likely_iban_derivation_success(features, party):
                if party == 'cdt':
                    codes.append(f'9004_{party_suffix}')
                elif party == 'bnf':
                    codes.append(f'9007_{party_suffix}')
            else:
                codes.append(f'9999_{party_suffix}')
        
        # BIC derivation eligibility
        if not features[f'{party}_has_bic']:
            
            if features[f'{party}_has_nch']:
                if likely_bic_from_nch_success(features, party):
                    codes.append(f'9005_{party_suffix}')
                else:
                    codes.append(f'9999_{party_suffix}')
            
            elif features[f'{party}_has_iban']:
                if likely_bic_from_iban_success(features, party):
                    codes.append(f'9008_{party_suffix}')
                else:
                    codes.append(f'9999_{party_suffix}')
            
            elif features[f'{party}_has_name']:
                # Fuzzy match - lower confidence
                if likely_bic_from_name_success(features, party):
                    codes.append(f'9961_{party_suffix}')
                else:
                    codes.append(f'9999_{party_suffix}')
        
        # BIC enrichment eligibility
        if features[f'{party}_has_bic']:
            
            if features[f'{party}_bic_length'] == 8:
                if likely_bic_expansion_success(features, party):
                    codes.append(f'9477_{party_suffix}')
                else:
                    codes.append(f'9999_{party_suffix}')
            
            # Party enrichment
            if likely_party_enrichment_success(features, party):
                codes.append(f'9479_{party_suffix}')
    
    return codes


def likely_iban_derivation_success(features, party):
    """
    Heuristic for IBAN derivation success.
    Could be replaced with ML model.
    """
    # Major IBAN countries with good derivation support
    well_supported = {'DE', 'FR', 'GB', 'ES', 'IT', 'NL', 'BE', 'AT', 'CH'}
    
    country = features.get(f'{party}_country') or \
              features.get(f'{party}_bic_country') or \
              features.get(f'{party}_address_country')
    
    if country in well_supported:
        return 0.90  # 90% success rate
    elif country:
        return 0.70  # 70% for other IBAN countries
    else:
        return 0.50  # Unknown country


def likely_bic_from_nch_success(features, party):
    """Heuristic for BIC from NCH lookup success."""
    nch_type = features.get(f'{party}_nch_type')
    
    if nch_type == 'FEDABA' and features.get(f'{party}_fedaba_checksum_valid'):
        return 0.85  # Valid FEDABA - good chance
    elif nch_type == 'CHIPS':
        return 0.80
    elif nch_type == 'SORTCODE':
        return 0.75
    else:
        return 0.50


def likely_bic_from_iban_success(features, party):
    """Heuristic for BIC from IBAN lookup success."""
    if features.get(f'{party}_iban_valid_format') and \
       features.get(f'{party}_iban_checksum_valid'):
        return 0.90  # Valid IBAN - high success
    else:
        return 0.60


def likely_bic_expansion_success(features, party):
    """Heuristic for BIC 8→11 expansion success."""
    if features.get(f'{party}_bic_valid_format'):
        return 0.95  # Most valid 8-char BICs can expand
    else:
        return 0.50


def likely_party_enrichment_success(features, party):
    """Heuristic for party enrichment success."""
    if features.get(f'{party}_bic_valid_format') and \
       features.get(f'{party}_bic_valid_country'):
        return 0.80
    else:
        return 0.40
```

---

## Quick Reference Card

### 8XXX - When to Emit

| Code | Emit When |
|------|-----------|
| 8001 | `has_bic AND (NOT bic_valid_format OR NOT bic_valid_country)` |
| 8004 | `needs_iban AND NOT has_iban` |
| 8022 | `has_iban AND has_bic AND NOT bic_iban_match` |
| 8026 | `has_nch AND nch_sources > 1` |
| 8894 | `has_iban AND (NOT iban_valid_format OR NOT iban_checksum_valid)` |
| 8895 | `has_nch AND nch_validation_applicable AND (NOT nch_valid OR NOT fedaba_checksum_valid)` |
| 8898 | `has_iban AND NOT iban_checksum_valid` |

### 9XXX No Directory - When to Emit

| Code | Emit When |
|------|-----------|
| 9000 | `has_nch AND nch_has_dirty_chars` |
| 9002 | `present AND has_account AND account_has_dirty_chars` |
| 9006 | `present AND has_iban AND iban_needs_formatting` |
| 9017 | `present AND has_multiple_ids` |
| 9018 | `present AND has_duplicate_info` |
| 9021 | `present AND has_nch AND nch_needs_formatting` |
| 9022 | `present AND has_account AND account_needs_length_fix` |

### 9XXX With Directory - When Eligible

| Code | Eligible When | Success → | Fail → |
|------|---------------|-----------|--------|
| 9004 | `cdt_present AND cdt_needs_iban AND NOT cdt_has_iban AND cdt_has_account` | 9004 | 9999 |
| 9005 | `present AND has_nch AND NOT has_bic` | 9005 | 9999 |
| 9007 | `bnf_present AND bnf_needs_iban AND NOT bnf_has_iban AND bnf_has_account` | 9007 | 9999 |
| 9008 | `present AND has_iban AND NOT has_bic` | 9008 | 9999 |
| 9477 | `present AND has_bic AND bic_length = 8` | 9477 | 9999 |
| 9479 | `present AND has_bic` | 9479 | 9999 |
| 9961 | `present AND has_name AND NOT has_bic` | 9961 | 9999 |

---

*Documentation generated for ACE Pelican ML Error Code Prediction System*
*Last updated: 2025*
