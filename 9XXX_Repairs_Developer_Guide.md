# 9XXX Repair Codes - Developer Guide

## Overview

9XXX codes indicate ACE Pelican **repaired/transformed** the payment message. Unlike 8XXX (validation errors), 9XXX means "we fixed this for you."

**Key Limitation:** Many 9XXX repairs depend on ACE's internal BIC/IBAN directories. Without directory access, prediction accuracy ceiling is ~75-80%.

---

## Code Reference

| Code | Party | Meaning | Trigger Condition | Requires Directory? |
|------|-------|---------|-------------------|---------------------|
| **9002** | BNPPTY | Account cleaned | Account contains spaces, special chars (-/.\) | No |
| **9004** | BNPPTY | IBAN derived | Has BBAN + country, no IBAN, ACE can derive | **Yes** |
| **9005** | BNFBNK | BIC derived from NCH | Has NCH (FEDABA/CHIPS), no BIC | **Yes** |
| **9006** | BNPPTY | IBAN formatted | IBAN has spaces, lowercase, wrong structure | No |
| **9007** | BNPPTY | IBAN derived (alt) | Same as 9004, different party | **Yes** |
| **9008** | BNFBNK | BIC derived from IBAN | Has IBAN, no BIC | **Yes** |
| **9015** | BNPPTY | Name cleaned | Name contains special chars (@#$%^) | No |
| **9017** | CDTPTY | Duplicate info removed | Multiple IDs for same party | No |
| **9018** | BNFBNK/INTBNK | Duplicate info merged | Duplicate party info detected | No |
| **9019** | BNFBNK/BNPPTY | Party info cleaned | Party data normalized | No |
| **9021** | Various | NCH formatted | NCH wrong length/format but fixable | No |
| **9024** | INTBNK | Intermediary added | Cross-border + US benef + no intermediary | **Yes** |
| **9025** | BNPPTY | Account formatted | Account wrong length for type | No |
| **9477** | Various | BIC enriched | BIC expanded from 8 to 11 chars | **Yes** |
| **9479** | BNFBNK/BNPPTY | Party enriched | Missing party info added from directory | **Yes** |
| **9480** | CDTPTY/INTBNK | Address enriched | Address added from BIC directory | **Yes** |
| **9938** | - | Message transformed | Format transformation applied | No |
| **9970** | BNFBNK/CDTPTY | Regulatory info added | Compliance data added | **Yes** |
| **9999** | Various | Multiple repairs | Catch-all for multiple repairs | Varies |

---

## Detection Logic (Without Directory)

```python
def can_detect_without_directory(code: str) -> bool:
    """Codes we can reliably predict without BIC/IBAN directory."""
    no_directory_needed = ['9002', '9006', '9015', '9017', '9018', '9021', '9025']
    base_code = code.split('_')[0]
    return base_code in no_directory_needed


def detect_9002_account_cleaning(features: dict, prefix: str) -> bool:
    """Account has characters that will be cleaned."""
    return (
        features.get(f'{prefix}_account_has_dirty_chars', False) or
        features.get(f'{prefix}_account_has_spaces', False) or
        features.get(f'{prefix}_account_has_special_chars', False)
    )


def detect_9006_iban_formatting(features: dict, prefix: str) -> bool:
    """IBAN present but needs reformatting."""
    return features.get(f'{prefix}_iban_needs_formatting', False)


def detect_9015_name_cleaning(features: dict, prefix: str) -> bool:
    """Name has characters that will be cleaned."""
    return features.get(f'{prefix}_name_has_dirty_chars', False)


def detect_9017_9018_duplicates(features: dict, prefix: str) -> bool:
    """Duplicate party info detected."""
    return (
        features.get(f'{prefix}_has_multiple_ids', False) or
        features.get(f'{prefix}_has_duplicate_info', False)
    )


def detect_9021_nch_formatting(features: dict, prefix: str) -> bool:
    """NCH needs reformatting."""
    return features.get(f'{prefix}_nch_needs_formatting', False)
```

---

## Directory Lookup Specifications

### 9004/9007: IBAN Derivation

**Purpose:** Derive IBAN from domestic account number + country

**Lookup Parties (in order of priority):**
1. `CreditPartyInfo` (cdt) - Primary beneficiary account
2. `BeneficiaryBankInfo` (bnf) - Beneficiary bank account
3. `DebitPartyInfo` (dbt) - For return payments

**Lookup Input:**
```python
{
    'country_code': str,       # 2-char ISO (e.g., 'DE', 'FR', 'GB')
    'bank_code': str,          # National bank code (e.g., SORT code, BLZ)
    'account_number': str,     # Domestic account number (BBAN)
}
```

**Lookup Logic:**
```python
def detect_9004_9007_iban_derivation(features: dict, prefix: str, iban_directory) -> tuple:
    """
    IBAN can be derived from account + country.
    
    Returns: (can_derive: bool, derived_iban: str or None, party_code: str)
    """
    # Skip if already has IBAN
    if features.get(f'{prefix}_has_iban', False):
        return False, None, None
    
    # Skip if no account
    if not features.get(f'{prefix}_has_account', False):
        return False, None, None
    
    # Get country - check multiple sources
    country = (
        features.get(f'{prefix}_iban_country') or      # From partial IBAN
        features.get(f'{prefix}_bic_country') or       # From BIC
        features.get(f'{prefix}_country') or           # From party info
        features.get(f'{prefix}_address_country')      # From address
    )
    
    if not country:
        return False, None, None
    
    # Only IBAN countries (SEPA + others)
    iban_countries = {
        'AD', 'AE', 'AL', 'AT', 'AZ', 'BA', 'BE', 'BG', 'BH', 'BR',
        'BY', 'CH', 'CR', 'CY', 'CZ', 'DE', 'DK', 'DO', 'EE', 'EG',
        'ES', 'FI', 'FO', 'FR', 'GB', 'GE', 'GI', 'GL', 'GR', 'GT',
        'HR', 'HU', 'IE', 'IL', 'IQ', 'IS', 'IT', 'JO', 'KW', 'KZ',
        'LB', 'LC', 'LI', 'LT', 'LU', 'LV', 'MC', 'MD', 'ME', 'MK',
        'MR', 'MT', 'MU', 'NL', 'NO', 'PK', 'PL', 'PS', 'PT', 'QA',
        'RO', 'RS', 'SA', 'SC', 'SE', 'SI', 'SK', 'SM', 'ST', 'SV',
        'TL', 'TN', 'TR', 'UA', 'VA', 'VG', 'XK'
    }
    
    if country.upper() not in iban_countries:
        return False, None, None
    
    account = features.get(f'{prefix}_account_value', '')
    bank_code = features.get(f'{prefix}_nch_value', '')  # Sort code, BLZ, etc.
    
    # === DIRECTORY LOOKUP ===
    result = iban_directory.lookup(
        country=country.upper(),
        bank_code=bank_code,
        account_number=account.replace(' ', '').replace('-', '')
    )
    
    # === HANDLE LOOKUP RESULT ===
    if result is None or result.get('status') == 'NOT_FOUND':
        # Directory has no mapping for this bank/account
        # DO NOT predict 9004/9007 - ACE won't derive either
        return False, None, None
    
    if result.get('status') == 'INVALID_INPUT':
        # Account/bank code format invalid for this country
        # May result in 8XXX validation error instead
        return False, None, None
    
    if result.get('status') == 'FOUND':
        derived_iban = result.get('iban')
        party_code = _prefix_to_party(prefix)
        return True, derived_iban, party_code
    
    return False, None, None
```

**Directory Response Handling:**

| Response Status | Meaning | Action |
|-----------------|---------|--------|
| `FOUND` | IBAN derivable | Predict 9004/9007 |
| `NOT_FOUND` | Bank/account not in directory | Do NOT predict 9004/9007 |
| `INVALID_INPUT` | Input format wrong | Do NOT predict (may be 8XXX error) |
| `COUNTRY_NOT_SUPPORTED` | Country not in IBAN system | Do NOT predict |

---

### 9005: BIC Derivation from NCH (Routing Number)

**Purpose:** Derive BIC from US routing number (FEDABA/CHIPS)

**Lookup Parties (in order of priority):**
1. `BeneficiaryBankInfo` (bnf) - Most common
2. `IntermediaryBankInfo` (intm) - For correspondent banks
3. `CreditPartyInfo` (cdt) - If bank info in credit party

**Lookup Input:**
```python
{
    'nch_type': str,           # 'FEDABA', 'CHIPS', 'FEDWIRE'
    'nch_value': str,          # 9-digit ABA or 6-digit CHIPS
}
```

**Lookup Logic:**
```python
def detect_9005_bic_from_nch(features: dict, prefix: str, bic_directory) -> tuple:
    """
    BIC can be derived from NCH (US routing number).
    
    Returns: (can_derive: bool, derived_bic: str or None, party_code: str)
    """
    # Skip if already has BIC
    if features.get(f'{prefix}_has_bic', False):
        return False, None, None
    
    # Check for NCH presence
    has_nch = (
        features.get(f'{prefix}_has_nch', False) or
        features.get(f'{prefix}_is_fedaba', False) or
        features.get(f'{prefix}_is_chips_aba', False)
    )
    
    if not has_nch:
        return False, None, None
    
    nch_type = features.get(f'{prefix}_nch_type', '')
    nch_value = features.get(f'{prefix}_nch_value', '')
    
    # Clean NCH value
    nch_clean = nch_value.replace(' ', '').replace('-', '')
    
    # Validate NCH format before lookup
    if nch_type in ['FEDABA', 'FW', 'FEDWIRE']:
        if len(nch_clean) != 9 or not nch_clean.isdigit():
            return False, None, None  # Invalid FEDABA
    elif nch_type in ['CHIPS', 'CH']:
        if len(nch_clean) != 6 or not nch_clean.isdigit():
            return False, None, None  # Invalid CHIPS
    
    # === DIRECTORY LOOKUP ===
    result = bic_directory.lookup_by_nch(
        nch_type=nch_type,
        nch_value=nch_clean
    )
    
    # === HANDLE LOOKUP RESULT ===
    if result is None or result.get('status') == 'NOT_FOUND':
        # This routing number not in BIC directory
        # Could be: small bank, credit union, or obsolete routing number
        # DO NOT predict 9005 - ACE won't derive either
        return False, None, None
    
    if result.get('status') == 'MULTIPLE_MATCHES':
        # Routing number maps to multiple BICs (rare but possible)
        # ACE may use additional info to disambiguate
        # Conservative: DO NOT predict
        return False, None, None
    
    if result.get('status') == 'FOUND':
        derived_bic = result.get('bic')
        party_code = _prefix_to_party(prefix)
        return True, derived_bic, party_code
    
    return False, None, None
```

**Common NCH Types:**

| Type | Format | Example | Country |
|------|--------|---------|---------|
| FEDABA | 9 digits | 021000021 | US |
| CHIPS | 6 digits | 000123 | US |
| SORTCODE | 6 digits | 20-00-00 | GB |
| BLZ | 8 digits | 37040044 | DE |
| BSB | 6 digits | 012-003 | AU |

---

### 9008: BIC Derivation from IBAN

**Purpose:** Derive BIC from IBAN (extract bank code from IBAN, lookup BIC)

**Lookup Parties (in order of priority):**
1. `BeneficiaryBankInfo` (bnf)
2. `CreditPartyInfo` (cdt)
3. `IntermediaryBankInfo` (intm)

**Lookup Input:**
```python
{
    'iban': str,               # Full IBAN (e.g., 'DE89370400440532013000')
}
# OR extracted components:
{
    'country_code': str,       # First 2 chars of IBAN
    'bank_code': str,          # Chars 5-12 (varies by country)
}
```

**Lookup Logic:**
```python
def detect_9008_bic_from_iban(features: dict, prefix: str, bic_directory) -> tuple:
    """
    BIC can be derived from IBAN.
    
    Returns: (can_derive: bool, derived_bic: str or None, party_code: str)
    """
    # Skip if already has BIC
    if features.get(f'{prefix}_has_bic', False):
        return False, None, None
    
    # Need IBAN
    if not features.get(f'{prefix}_has_iban', False):
        return False, None, None
    
    iban = features.get(f'{prefix}_account_value', '')
    iban_clean = iban.replace(' ', '').upper()
    
    # Validate IBAN format
    if len(iban_clean) < 15 or not iban_clean[:2].isalpha():
        return False, None, None
    
    country_code = iban_clean[:2]
    
    # Extract bank code (position varies by country)
    bank_code = _extract_bank_code_from_iban(iban_clean, country_code)
    
    if not bank_code:
        return False, None, None
    
    # === DIRECTORY LOOKUP ===
    result = bic_directory.lookup_by_iban(
        iban=iban_clean
    )
    # OR lookup by extracted bank code:
    # result = bic_directory.lookup_by_bank_code(
    #     country=country_code,
    #     bank_code=bank_code
    # )
    
    # === HANDLE LOOKUP RESULT ===
    if result is None or result.get('status') == 'NOT_FOUND':
        # Bank not in BIC directory
        # Possible for small banks or new IBANs
        return False, None, None
    
    if result.get('status') == 'FOUND':
        derived_bic = result.get('bic')
        party_code = _prefix_to_party(prefix)
        return True, derived_bic, party_code
    
    return False, None, None


def _extract_bank_code_from_iban(iban: str, country: str) -> str:
    """Extract bank identifier from IBAN based on country rules."""
    # Bank code position varies by country
    bank_code_positions = {
        'DE': (4, 12),   # Germany: 8-digit BLZ at position 4-12
        'GB': (4, 8),    # UK: 4-char bank code at position 4-8
        'FR': (4, 9),    # France: 5-digit bank code at position 4-9
        'ES': (4, 8),    # Spain: 4-digit bank code at position 4-8
        'IT': (5, 10),   # Italy: 5-char ABI code at position 5-10
        'NL': (4, 8),    # Netherlands: 4-char bank code at position 4-8
        'BE': (4, 7),    # Belgium: 3-digit bank code at position 4-7
        'AT': (4, 9),    # Austria: 5-digit BLZ at position 4-9
        'CH': (4, 9),    # Switzerland: 5-digit BC at position 4-9
        # Add more countries as needed
    }
    
    positions = bank_code_positions.get(country)
    if positions and len(iban) >= positions[1]:
        return iban[positions[0]:positions[1]]
    
    # Default: chars 4-8
    return iban[4:8] if len(iban) >= 8 else None
```

---

### 9024: Intermediary Bank Added

**Purpose:** Add correspondent/intermediary bank for cross-border USD payments

**Lookup Parties:**
1. `BeneficiaryBankInfo` (bnf) - Check if needs correspondent

**Lookup Input:**
```python
{
    'beneficiary_bic': str,    # Beneficiary bank BIC
    'currency': str,           # Payment currency
    'originator_country': str, # Sender country
}
```

**Lookup Logic:**
```python
def detect_9024_intermediary_added(features: dict, routing_directory) -> tuple:
    """
    Intermediary bank will be added for cross-border payment.
    
    Returns: (will_add: bool, intermediary_bic: str or None)
    """
    # Skip if already has intermediary
    if features.get('has_intermediary', False):
        return False, None
    
    # Must be cross-border
    if not features.get('is_cross_border', False):
        return False, None
    
    currency = features.get('primary_currency', '')
    bnf_bic = features.get('bnf_bic', '')
    bnf_country = features.get('bnf_bic_country', '') or features.get('bnf_country', '')
    
    # Most common: USD payments to non-US banks need US correspondent
    if currency == 'USD' and bnf_country != 'US':
        # === DIRECTORY LOOKUP ===
        result = routing_directory.get_correspondent(
            beneficiary_bic=bnf_bic,
            currency=currency
        )
        
        if result and result.get('status') == 'FOUND':
            intermediary_bic = result.get('correspondent_bic')
            return True, intermediary_bic
    
    # EUR payments may need TARGET2 intermediary
    if currency == 'EUR':
        # Similar lookup logic
        pass
    
    return False, None
```

**When ACE Adds Intermediary:**

| Currency | Beneficiary Location | Intermediary Added? |
|----------|---------------------|---------------------|
| USD | Non-US bank | Yes - US correspondent |
| EUR | Non-SEPA bank | Maybe - depends on routing |
| GBP | Non-UK bank | Maybe - depends on bank |

---

### 9477: BIC Enrichment (8 → 11 chars)

**Purpose:** Expand short BIC (8 chars) to full BIC (11 chars with branch code)

**Lookup Input:**
```python
{
    'bic_8': str,              # 8-character BIC (e.g., 'DEUTDEFF')
}
```

**Lookup Logic:**
```python
def detect_9477_bic_enrichment(features: dict, prefix: str, bic_directory) -> tuple:
    """
    BIC will be expanded from 8 to 11 characters.
    
    Returns: (will_expand: bool, expanded_bic: str or None, party_code: str)
    """
    bic = features.get(f'{prefix}_bic', '')
    bic_length = features.get(f'{prefix}_bic_length', 0)
    
    # Only applies to 8-char BICs
    if bic_length != 8:
        return False, None, None
    
    # === DIRECTORY LOOKUP ===
    result = bic_directory.expand_bic(bic_8=bic)
    
    if result and result.get('status') == 'FOUND':
        # Default branch code is usually 'XXX' but may be specific
        expanded_bic = result.get('bic_11', bic + 'XXX')
        party_code = _prefix_to_party(prefix)
        return True, expanded_bic, party_code
    
    # If not found, ACE typically appends 'XXX'
    # But may not trigger 9477 if just default expansion
    return False, None, None
```

---

### 9479: Party Enrichment

**Purpose:** Add missing party information from BIC directory

**Lookup Input:**
```python
{
    'bic': str,                # Party's BIC
}
```

**What Gets Enriched:**
- Bank name
- Bank address
- Country
- Branch information

**Lookup Logic:**
```python
def detect_9479_party_enrichment(features: dict, prefix: str, bic_directory) -> tuple:
    """
    Missing party info will be added from BIC directory.
    
    Returns: (will_enrich: bool, enrichment_data: dict, party_code: str)
    """
    # Need BIC for lookup
    if not features.get(f'{prefix}_has_bic', False):
        return False, None, None
    
    bic = features.get(f'{prefix}_bic', '')
    has_name = features.get(f'{prefix}_has_name', False)
    address_lines = features.get(f'{prefix}_address_lines', 0)
    
    # Check what's missing
    missing_name = not has_name
    missing_address = address_lines == 0
    
    if not missing_name and not missing_address:
        return False, None, None  # Nothing to enrich
    
    # === DIRECTORY LOOKUP ===
    result = bic_directory.lookup_by_bic(bic=bic)
    
    if result is None or result.get('status') == 'NOT_FOUND':
        return False, None, None
    
    if result.get('status') == 'FOUND':
        enrichment = {}
        if missing_name and result.get('bank_name'):
            enrichment['name'] = result.get('bank_name')
        if missing_address and result.get('address'):
            enrichment['address'] = result.get('address')
        
        if enrichment:
            party_code = _prefix_to_party(prefix)
            return True, enrichment, party_code
    
    return False, None, None
```

---

## Directory API Specifications

**Recommended Interface:**

```python
class BICDirectory:
    def lookup_by_bic(self, bic: str) -> dict:
        """
        Lookup bank info by BIC.
        
        Returns: {
            'status': 'FOUND' | 'NOT_FOUND' | 'INVALID_BIC',
            'bic_11': str,           # Full 11-char BIC
            'bank_name': str,
            'address': {
                'line1': str,
                'line2': str,
                'city': str,
                'country': str,
            },
            'branch_info': str,
        }
        """
        pass
    
    def lookup_by_nch(self, nch_type: str, nch_value: str) -> dict:
        """
        Lookup BIC by national clearing code.
        
        Returns: {
            'status': 'FOUND' | 'NOT_FOUND' | 'MULTIPLE_MATCHES',
            'bic': str,
            'bank_name': str,
        }
        """
        pass
    
    def lookup_by_iban(self, iban: str) -> dict:
        """
        Lookup BIC from IBAN.
        
        Returns: {
            'status': 'FOUND' | 'NOT_FOUND',
            'bic': str,
            'bank_name': str,
        }
        """
        pass
    
    def expand_bic(self, bic_8: str) -> dict:
        """
        Expand 8-char BIC to 11-char.
        
        Returns: {
            'status': 'FOUND' | 'NOT_FOUND',
            'bic_11': str,
        }
        """
        pass


class IBANDirectory:
    def lookup(self, country: str, bank_code: str, account_number: str) -> dict:
        """
        Check if IBAN can be derived.
        
        Returns: {
            'status': 'FOUND' | 'NOT_FOUND' | 'INVALID_INPUT' | 'COUNTRY_NOT_SUPPORTED',
            'iban': str,             # Derived IBAN if found
            'check_digits': str,     # IBAN check digits
        }
        """
        pass
    
    def validate(self, iban: str) -> dict:
        """
        Validate IBAN format and checksum.
        
        Returns: {
            'valid': bool,
            'country': str,
            'bank_code': str,
            'account_number': str,
        }
        """
        pass


class RoutingDirectory:
    def get_correspondent(self, beneficiary_bic: str, currency: str) -> dict:
        """
        Get correspondent/intermediary bank for cross-border payment.
        
        Returns: {
            'status': 'FOUND' | 'NOT_FOUND' | 'DIRECT_POSSIBLE',
            'correspondent_bic': str,
            'correspondent_name': str,
        }
        """
        pass
    
    def requires_intermediary(self, beneficiary_bic: str, currency: str) -> bool:
        """Check if payment requires intermediary."""
        pass
```

---

## Summary: Decision Tree for 9XXX Prediction

```
For each party (bnf, cdt, intm, ordi, acwi):

1. HAS ACCOUNT BUT NO IBAN?
   ├── Is IBAN country? 
   │   ├── YES → Lookup IBAN directory
   │   │   ├── FOUND → Predict 9004/9007
   │   │   └── NOT_FOUND → No prediction
   │   └── NO → No prediction

2. HAS NCH BUT NO BIC?
   ├── Is valid NCH format?
   │   ├── YES → Lookup BIC directory by NCH
   │   │   ├── FOUND → Predict 9005
   │   │   └── NOT_FOUND → No prediction
   │   └── NO → No prediction

3. HAS IBAN BUT NO BIC?
   ├── Lookup BIC directory by IBAN
   │   ├── FOUND → Predict 9008
   │   └── NOT_FOUND → No prediction

4. HAS 8-CHAR BIC?
   ├── Lookup BIC expansion
   │   ├── FOUND specific branch → Predict 9477
   │   └── DEFAULT XXX → Maybe no 9477

5. HAS BIC BUT MISSING NAME/ADDRESS?
   ├── Lookup BIC directory
   │   ├── FOUND → Predict 9479
   │   └── NOT_FOUND → No prediction

6. CROSS-BORDER USD WITHOUT INTERMEDIARY?
   ├── Lookup routing directory
   │   ├── NEEDS CORRESPONDENT → Predict 9024
   │   └── DIRECT POSSIBLE → No prediction

7. NO DIRECTORY NEEDED:
   ├── Account has dirty chars → Predict 9002
   ├── Name has dirty chars → Predict 9015
   ├── IBAN needs formatting → Predict 9006
   ├── NCH needs formatting → Predict 9021
   └── Duplicate info → Predict 9017/9018
```

---

## Party Code Reference

| Prefix | Party Code | Full Name | Common 9XXX Codes |
|--------|------------|-----------|-------------------|
| orig | ORGPTY | Originating Party | 9002, 9015 |
| send | SNDBNK | Sending Bank | 9005, 9477 |
| dbt | DBTPTY | Debit Party | 9002, 9015 |
| cdt | CDTPTY | Credit Party | 9004, 9017, 9480 |
| intm | INTBNK | Intermediary Bank | 9005, 9018, 9024 |
| bnf | BNFBNK | Beneficiary Bank | 9005, 9008, 9479 |
| ordi | ORGPTY | Originating Party (alt) | 9002, 9015 |
| acwi | CDTPTY | Credit Party (alt) | 9004, 9007 |

---

## Helper Function

```python
def _prefix_to_party(prefix: str) -> str:
    """Convert feature prefix to party code."""
    mapping = {
        'orig': 'ORGPTY',
        'send': 'SNDBNK', 
        'dbt': 'DBTPTY',
        'cdt': 'CDTPTY',
        'intm': 'INTBNK',
        'bnf': 'BNFBNK',
        'ordi': 'ORGPTY',
        'acwi': 'CDTPTY',
    }
    return mapping.get(prefix, 'BNPPTY')
```

---

## ML Model Usage (Without Directory)

When directory lookup is not available, use the ML model with threshold configuration:

```python
from predictor import ACEPredictor

# Load 9XXX model
predictor = ACEPredictor('./models_9x_v3')

# Predict
result = predictor.predict(payment_request, threshold=0.5)

# 9XXX codes in result.predicted_codes
nine_xxx_codes = [c for c in result.predicted_codes if c.startswith('9')]
```

**Expected Accuracy:** ~79% with threshold tuning (without directory lookup)

**Threshold Configuration (prediction_config.json):**
```json
{
  "high_threshold_codes": {
    "9479": 0.80,
    "9999": 0.80,
    "9018": 0.75,
    "9024": 0.75,
    "9002": 0.70,
    "9015": 0.70,
    "9019": 0.70
  }
}
```
