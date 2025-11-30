#!/usr/bin/env python3
"""
ACE Pelican Prediction Utilities

Combined module for:
1. Filtering false positive predictions based on ACE preconditions
2. Explaining predictions with human-readable trigger conditions

Usage as filter:
    from prediction_utils import filter_predictions
    filtered, removed = filter_predictions(predicted_codes, features)

Usage as explainer:
    python prediction_utils.py --model-dir ./models_8x --data-dir ./raw_data --output explanations.md

Usage for single payment:
    python prediction_utils.py --model-dir ./models_8x --payment ./test.json
"""

import argparse
import json
import pickle
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


# =============================================================================
# ACE PELICAN CODE DEFINITIONS
# =============================================================================

# Feature to human-readable explanation mapping
FEATURE_EXPLANATIONS = {
    # IBAN related
    'needs_iban': 'Party is in IBAN-required country',
    'has_iban': 'IBAN is provided',
    'iban_valid_format': 'IBAN format is valid',
    'iban_checksum_valid': 'IBAN checksum passes',
    'missing_required_iban': 'Required IBAN is missing',
    'iban_needs_formatting': 'IBAN contains spaces/dashes that need cleaning',
    
    # BIC related
    'has_bic': 'BIC is provided',
    'bic_valid_format': 'BIC format is valid (8 or 11 chars)',
    'bic_valid_country': 'BIC country code is valid',
    'bic_iban_match': 'BIC and IBAN countries match',
    'bic_party_country_match': 'BIC matches party country',
    
    # NCH/Routing related
    'has_nch': 'NCH/routing number is provided',
    'nch_valid': 'NCH format is valid',
    'fedaba_checksum_valid': 'FEDABA checksum passes',
    'nch_validation_applicable': 'NCH validation applies to this payment',
    'has_adr_bank_id': 'Bank ID in address block',
    
    # Account related
    'has_account': 'Account number is provided',
    'account_numeric': 'Account is numeric only',
    'is_iban': 'Account is IBAN format',
    'is_clabe': 'Account is CLABE format (Mexico)',
    'is_fedaba': 'Account is FEDABA format',
    'account_has_dirty_chars': 'Account contains non-alphanumeric characters',
    'account_has_spaces': 'Account contains spaces',
    'account_has_special_chars': 'Account contains special characters (-/.)',
    
    # Party presence
    'present': 'Party is present in message',
    'has_id': 'Party has ID',
    'has_name': 'Party has name',
    
    # Payment type
    'is_domestic': 'Domestic payment',
    'is_international': 'International payment',
    'is_cross_border': 'Cross-border payment',
}

# ACE Pelican code trigger definitions
# require_true: features that MUST be True for the code to be valid
# require_false: features that MUST be False for the code to be valid
CODE_TRIGGERS = {
    # ==========================================================================
    # 8XXX VALIDATION ERRORS - All are trainable from message features
    # ==========================================================================
    
    # --- BIC Validation ---
    '8001': {
        'require_true': ['has_bic'],
        'require_false': ['bic_valid_format', 'bic_valid_country'],
        'description': 'Invalid BIC',
        'needs_directory': False
    },
    '8005': {
        'require_true': ['has_bic'],
        'require_false': [],
        'description': 'Invalid BIC4',
        'needs_directory': False
    },
    '8006': {
        'require_true': ['present'],
        'require_false': ['country_valid'],
        'description': 'Invalid country code',
        'needs_directory': False
    },
    
    # --- IBAN/Account Validation ---
    '8004': {
        'require_true': ['needs_iban'],
        'require_false': ['has_iban'],
        'description': 'IBAN cannot be derived',
        'needs_directory': True,
        'directory': 'IBAN_DERIVATION'
    },
    '8030': {
        'require_true': ['needs_iban'],
        'require_false': ['has_iban'],
        'description': 'IBAN derivation not supported for the country',
        'needs_directory': False
    },
    '8892': {
        'require_true': ['has_account'],
        'require_false': ['account_valid'],
        'description': 'Invalid Account number',
        'needs_directory': False
    },
    '8894': {
        'require_true': ['has_iban'],
        'require_false': ['iban_valid_format', 'iban_checksum_valid'],
        'description': 'Invalid IBAN',
        'needs_directory': False
    },
    '8897': {
        'require_true': ['has_account'],
        'require_false': ['bban_valid'],
        'description': 'Invalid BBAN',
        'needs_directory': False
    },
    '8898': {
        'require_true': ['has_iban'],
        'require_false': ['iban_checksum_valid'],
        'description': 'IBAN Check Digit calculation/validation failed',
        'needs_directory': False
    },
    
    # --- NCH/Routing Validation ---
    '8895': {
        'require_true': ['has_nch'],
        'require_false': ['nch_valid', 'fedaba_checksum_valid'],
        'description': 'Invalid NCH code',
        'needs_directory': False
    },
    '8896': {
        'require_true': ['has_account', 'is_domestic'],
        'require_false': ['domestic_account_valid'],
        'description': 'Invalid Domestic Account Number',
        'needs_directory': False
    },
    
    # --- Inconsistency Errors ---
    '8022': {
        'require_true': ['has_iban', 'has_bic'],
        'require_false': ['bic_iban_match'],
        'description': 'IBAN inconsistent with Account With Institution BIC',
        'needs_directory': False
    },
    '8023': {
        'require_true': ['has_iban'],
        'require_false': [],
        'description': 'IBAN inconsistency found in message',
        'needs_directory': False
    },
    '8024': {
        'require_true': ['has_account'],
        'require_false': [],
        'description': 'BBAN inconsistency found in message',
        'needs_directory': False
    },
    '8025': {
        'require_true': ['has_account', 'is_domestic'],
        'require_false': [],
        'description': 'Domestic Account Number inconsistency found in message',
        'needs_directory': False
    },
    '8026': {
        'require_true': ['has_nch'],
        'require_false': [],
        'description': 'NCH inconsistency found in message',
        'needs_directory': False
    },
    '8027': {
        'require_true': ['present'],
        'require_false': [],
        'description': 'ISO Country Code inconsistency found in message',
        'needs_directory': False
    },
    '8028': {
        'require_true': ['has_bic'],
        'require_false': [],
        'description': 'BIC4 inconsistency found in message',
        'needs_directory': False
    },
    '8029': {
        'require_true': ['has_account'],
        'require_false': [],
        'description': 'Account Number inconsistency found in message',
        'needs_directory': False
    },
    '8033': {
        'require_true': ['has_account'],
        'require_false': [],
        'description': 'CLABE inconsistency found in message',
        'needs_directory': False
    },
    
    # --- Format/Length Errors ---
    '8007': {
        'require_true': ['has_amount'],
        'require_false': [],
        'description': 'Number of fractional digits of amount exceeds maximum allowable',
        'needs_directory': False
    },
    '8851': {
        'require_true': ['present'],
        'require_false': [],
        'description': 'Incorrect field size',
        'needs_directory': False
    },
    '8852': {
        'require_true': ['has_account'],
        'require_false': [],
        'description': 'Incorrect length of attribute',
        'needs_directory': False
    },
    '8853': {
        'require_true': ['present'],
        'require_false': [],
        'description': 'Incorrect number format',
        'needs_directory': False
    },
    
    # --- Other Validation Errors ---
    '8003': {
        'require_true': [],
        'require_false': [],
        'description': 'File name derivation failed',
        'needs_directory': True
    },
    '8034': {
        'require_true': [],
        'require_false': [],
        'description': 'Forced Debit is not allowed and hence message cannot be processed',
        'needs_directory': False
    },
    '8035': {
        'require_true': ['has_account'],
        'require_false': [],
        'description': 'FCDA account validation failed',
        'needs_directory': True
    },
    '8036': {
        'require_true': ['has_account'],
        'require_false': [],
        'description': 'FCDA account name matching failed',
        'needs_directory': True
    },
    '8124': {
        'require_true': ['has_currency'],
        'require_false': ['currency_valid'],
        'description': 'Invalid currency',
        'needs_directory': False
    },
    '8464': {
        'require_true': [],
        'require_false': [],
        'description': 'Target Channel not derived',
        'needs_directory': True
    },
    '8465': {
        'require_true': [],
        'require_false': [],
        'description': 'Product code not found',
        'needs_directory': True
    },
    '8472': {
        'require_true': [],
        'require_false': [],
        'description': 'Fee code not derived',
        'needs_directory': True
    },
    '8905': {
        'require_true': [],
        'require_false': [],
        'description': 'Hash code Mismatch',
        'needs_directory': False
    },
    '8906': {
        'require_true': [],
        'require_false': [],
        'description': 'Message in wrong flow',
        'needs_directory': False
    },
    
    # ==========================================================================
    # 9XXX REPAIR/ENRICHMENT CODES
    # ==========================================================================
    
    # **************************************************************************
    # TRAINABLE 9XXX CODES - Can be predicted from message features alone
    # **************************************************************************
    
    # --- NCH/Routing Cleaning ---
    '9000': {
        'require_true': ['has_nch', 'nch_has_dirty_chars'],
        'require_false': [],
        'description': 'NCH code cleaned',
        'needs_directory': False
    },
    '9021': {
        'require_true': ['has_nch', 'nch_needs_formatting'],
        'require_false': [],
        'description': 'FEDABA Code correctly formatted',
        'needs_directory': False
    },
    '9028': {
        'require_true': ['has_nch'],
        'require_false': [],
        'description': 'NCH Code strip off',
        'needs_directory': False
    },
    
    # --- Account Cleaning ---
    '9002': {
        'require_true': ['has_account', 'account_has_dirty_chars'],
        'require_false': [],
        'description': 'Account number cleaned of non alpha numeric characters',
        'needs_directory': False
    },
    '9022': {
        'require_true': ['has_account'],
        'require_false': [],
        'description': 'Account number formatted to required length',
        'needs_directory': False
    },
    '9479': {
        'require_true': ['has_account', 'account_has_dirty_chars'],
        'require_false': [],
        'description': 'Account Number Cleaned',
        'needs_directory': False
    },
    
    # --- IBAN Cleaning/Formatting ---
    '9006': {
        'require_true': ['has_iban', 'iban_needs_formatting'],
        'require_false': [],
        'description': 'IBAN Cleaned',
        'needs_directory': False
    },
    '9012': {
        'require_true': ['has_iban', 'iban_needs_formatting'],
        'require_false': [],
        'description': 'IBAN Formatted in field',
        'needs_directory': False
    },
    '9014': {
        'require_true': ['has_account'],
        'require_false': [],
        'description': 'BBAN cleaned',
        'needs_directory': False
    },
    
    # --- Name/Address Cleaning ---
    '9001': {
        'require_true': ['present'],
        'require_false': [],
        'description': 'D Field deleted',
        'needs_directory': False
    },
    '9009': {
        'require_true': ['present'],
        'require_false': [],
        'description': 'Field Cleaned',
        'needs_directory': False
    },
    '9013': {
        'require_true': ['has_name'],
        'require_false': [],
        'description': 'Name and Address repair done',
        'needs_directory': False
    },
    '9015': {
        'require_true': ['has_account', 'account_has_dirty_chars'],
        'require_false': [],
        'description': 'Domestic account number cleaned',
        'needs_directory': False
    },
    '9967': {
        'require_true': ['has_address'],
        'require_false': [],
        'description': 'Cleared PO Box number',
        'needs_directory': False
    },
    
    # --- Party Identifier Cleaning ---
    '9019': {
        'require_true': ['present'],
        'require_false': [],
        'description': 'Party Identifier cleaned of non alpha numeric characters',
        'soft_require': ['iban_needs_formatting', 'account_has_dirty_chars', 'account_has_spaces', 'id_has_dirty_chars'],
        'needs_directory': False
    },
    '9020': {
        'require_true': ['present'],
        'require_false': [],
        'description': 'Party Identifier cleaned of non alpha numeric characters',
        'soft_require': ['id_has_dirty_chars'],
        'needs_directory': False
    },
    
    # --- Duplicate/Inconsistency Resolution ---
    '9017': {
        'require_true': ['present'],
        'require_false': [],
        'description': 'Multiple party information present',
        'needs_directory': False
    },
    '9018': {
        'require_true': ['present', 'has_duplicate_info'],
        'require_false': [],
        'description': 'Duplicate party information removed',
        'needs_directory': False
    },
    '9483': {
        'require_true': ['present', 'has_duplicate_info'],
        'require_false': [],
        'description': "Receiver's duplicate information removed",
        'needs_directory': False
    },
    '9487': {
        'require_true': ['has_bic'],
        'require_false': [],
        'description': "Receiver's BIC information removed",
        'needs_directory': False
    },
    '9990': {
        'require_true': ['present'],
        'require_false': [],
        'description': 'Duplicate information of CDT is removed BBI removed',
        'needs_directory': False
    },
    
    # --- Field Generation ---
    '9010': {
        'require_true': [],
        'require_false': [],
        'description': 'Field 23E generated',
        'needs_directory': False
    },
    '9026': {
        'require_true': [],
        'require_false': [],
        'description': 'Purpose Information found in Field 72',
        'needs_directory': False
    },
    '9938': {
        'require_true': [],
        'require_false': [],
        'description': 'Deleted redundant phrases from Sender to Receiver information',
        'needs_directory': False
    },
    
    # **************************************************************************
    # DIRECTORY-DEPENDENT 9XXX CODES - Require external lookup
    # **************************************************************************
    
    # --- IBAN Derivation ---
    '9004': {
        'require_true': ['present', 'has_account'],
        'require_false': ['has_iban'],
        'description': 'IBAN repaired in Account Number line',
        'needs_directory': True,
        'directory': 'IBAN_DERIVATION',
        'lookup_condition': 'LOOKUP(IBAN_DERIVATION, country={country}, account={account}) -> IF found EMIT 9004 ELSE EMIT 8004'
    },
    '9007': {
        'require_true': ['present', 'has_account'],
        'require_false': ['has_iban'],
        'description': 'Account number replaced by IBAN',
        'needs_directory': True,
        'directory': 'IBAN_DERIVATION',
        'lookup_condition': 'LOOKUP(IBAN_DERIVATION, country={country}, account={account}) -> IF found EMIT 9007 ELSE EMIT 8004'
    },
    
    # --- BIC Derivation ---
    '9005': {
        'require_true': ['present', 'has_nch'],
        'require_false': ['has_bic'],
        'description': 'BIC replaced by derived BIC',
        'needs_directory': True,
        'directory': 'NCH_TO_BIC',
        'lookup_condition': 'LOOKUP(NCH_TO_BIC, nch={nch_value}) -> IF found EMIT 9005 ELSE skip'
    },
    '9008': {
        'require_true': ['present', 'has_iban'],
        'require_false': ['has_bic'],
        'description': 'IBAN to BIC repair done',
        'needs_directory': True,
        'directory': 'IBAN_TO_BIC',
        'lookup_condition': 'LOOKUP(IBAN_TO_BIC, iban={iban}) -> IF found EMIT 9008 ELSE skip'
    },
    '9032': {
        'require_true': ['has_bic'],
        'require_false': [],
        'description': '8 char BIC insertion',
        'needs_directory': True,
        'directory': 'BIC_DIRECTORY'
    },
    '9910': {
        'require_true': [],
        'require_false': ['has_bic'],
        'description': 'BIC repaired from BIC found in field 72',
        'needs_directory': True,
        'directory': 'BIC_DIRECTORY'
    },
    '9917': {
        'require_true': [],
        'require_false': [],
        'description': 'BIC repaired from BIC Plus using field 72',
        'needs_directory': True,
        'directory': 'BIC_PLUS'
    },
    '9961': {
        'require_true': ['has_name'],
        'require_false': ['has_bic'],
        'description': 'BIC Derived from Name and Address',
        'needs_directory': True,
        'directory': 'NAME_TO_BIC'
    },
    '9970': {
        'require_true': ['has_name'],
        'require_false': ['has_bic'],
        'description': 'D-A using BIC from Name and Address',
        'needs_directory': True,
        'directory': 'NAME_TO_BIC'
    },
    '9978': {
        'require_true': [],
        'require_false': [],
        'description': 'Numeric BIC derived using field 58',
        'needs_directory': True,
        'directory': 'BIC_DIRECTORY'
    },
    '9979': {
        'require_true': [],
        'require_false': [],
        'description': 'Numeric BIC derived using field 57',
        'needs_directory': True,
        'directory': 'BIC_DIRECTORY'
    },
    '9980': {
        'require_true': ['has_bic'],
        'require_false': [],
        'description': 'Alpha BIC repaired with numeric BIC',
        'needs_directory': True,
        'directory': 'BIC_DIRECTORY'
    },
    '9981': {
        'require_true': [],
        'require_false': [],
        'description': 'Merged BIC derived using field 58',
        'needs_directory': True,
        'directory': 'BIC_DIRECTORY'
    },
    '9982': {
        'require_true': [],
        'require_false': [],
        'description': 'Merged BIC derived using field 57',
        'needs_directory': True,
        'directory': 'BIC_DIRECTORY'
    },
    '9983': {
        'require_true': [],
        'require_false': [],
        'description': 'Exception BIC derived using field 58',
        'needs_directory': True,
        'directory': 'BIC_DIRECTORY'
    },
    '9984': {
        'require_true': [],
        'require_false': [],
        'description': 'Exception BIC derived using field 57',
        'needs_directory': True,
        'directory': 'BIC_DIRECTORY'
    },
    '9985': {
        'require_true': ['has_nch'],
        'require_false': [],
        'description': 'BIC from CHIPS ABA Repaired',
        'needs_directory': True,
        'directory': 'CHIPS_ABA_TO_BIC'
    },
    '9986': {
        'require_true': ['has_bic'],
        'require_false': [],
        'description': 'Head Office BIC Repaired',
        'needs_directory': True,
        'directory': 'BIC_DIRECTORY'
    },
    '9901': {
        'require_true': [],
        'require_false': [],
        'description': 'D-A using BIC retrieved from the field',
        'needs_directory': True,
        'directory': 'BIC_DIRECTORY'
    },
    '9936': {
        'require_true': ['has_name'],
        'require_false': [],
        'description': 'D-A using BIC from Name and Address',
        'needs_directory': True,
        'directory': 'NAME_TO_BIC'
    },
    
    # --- NCH Derivation from Name/Address ---
    '9475': {
        'require_true': ['has_name'],
        'require_false': ['has_account'],
        'description': 'D-A using Account No. from Name and Address',
        'needs_directory': True,
        'directory': 'NAME_ADDRESS_PARSE'
    },
    '9476': {
        'require_true': ['has_name'],
        'require_false': ['has_nch'],
        'description': 'D-A using CHIPS ABA from Name and Address',
        'needs_directory': True,
        'directory': 'NAME_ADDRESS_PARSE'
    },
    '9477': {
        'require_true': ['has_name'],
        'require_false': ['has_nch'],
        'description': 'D-A using FED ABA from Name and Address',
        'needs_directory': True,
        'directory': 'NAME_ADDRESS_PARSE'
    },
    '9478': {
        'require_true': ['has_name'],
        'require_false': [],
        'description': 'D-A using CHIPS UID from Name and Address',
        'needs_directory': True,
        'directory': 'NAME_ADDRESS_PARSE'
    },
    '9485': {
        'require_true': ['has_name'],
        'require_false': ['has_nch'],
        'description': 'D-A using NCH from Name and Address',
        'needs_directory': True,
        'directory': 'NAME_ADDRESS_PARSE'
    },
    
    # --- Push Up/Down Operations ---
    '9024': {
        'require_true': ['present', 'has_bic'],
        'require_false': ['has_intermediary'],
        'description': 'Push Up performed',
        'needs_directory': True,
        'directory': 'ROUTING',
        'lookup_condition': 'LOOKUP(ROUTING, dest_bic={bic}) -> IF intermediary_required EMIT 9024 ELSE skip'
    },
    '9480': {
        'require_true': ['present'],
        'require_false': [],
        'description': 'Push Down of type Push 1 performed',
        'needs_directory': True,
        'directory': 'ROUTING'
    },
    '9481': {
        'require_true': ['present'],
        'require_false': [],
        'description': 'Push Down of type Push 2 performed',
        'needs_directory': True,
        'directory': 'ROUTING'
    },
    '9482': {
        'require_true': ['present'],
        'require_false': [],
        'description': 'Push Down of type partial Push 2 performed',
        'needs_directory': True,
        'directory': 'ROUTING'
    },
    '9486': {
        'require_true': ['present'],
        'require_false': [],
        'description': 'A to D performed',
        'needs_directory': True,
        'directory': 'ROUTING'
    },
    '9488': {
        'require_true': ['present'],
        'require_false': [],
        'description': 'Partial push down performed',
        'needs_directory': True,
        'directory': 'ROUTING'
    },
    '9991': {
        'require_true': ['present'],
        'require_false': [],
        'description': 'Account Number pushed down to BBI from BBK',
        'needs_directory': True,
        'directory': 'ROUTING'
    },
    '9992': {
        'require_true': ['has_nch'],
        'require_false': [],
        'description': 'NCH code has been translocated to BBI',
        'needs_directory': True,
        'directory': 'ROUTING'
    },
    
    # --- Account Repairs ---
    '9023': {
        'require_true': ['has_account'],
        'require_false': [],
        'description': 'Fund to Account repair done',
        'needs_directory': True,
        'directory': 'ACCOUNT_DIRECTORY'
    },
    '9025': {
        'require_true': ['has_account'],
        'require_false': [],
        'description': 'CLABE repaired',
        'needs_directory': True,
        'directory': 'CLABE_DIRECTORY'
    },
    '9027': {
        'require_true': ['has_account'],
        'require_false': [],
        'description': 'FCDA account reformatted',
        'needs_directory': True,
        'directory': 'FCDA_DIRECTORY'
    },
    '9029': {
        'require_true': [],
        'require_false': [],
        'description': 'GL account insertion for FED',
        'needs_directory': True,
        'directory': 'GL_DIRECTORY'
    },
    '9030': {
        'require_true': [],
        'require_false': [],
        'description': 'DDA to GL attribute type conversion',
        'needs_directory': True,
        'directory': 'GL_DIRECTORY'
    },
    '9031': {
        'require_true': [],
        'require_false': [],
        'description': 'Trust account to GL number insertion',
        'needs_directory': True,
        'directory': 'GL_DIRECTORY'
    },
    '9932': {
        'require_true': ['has_account'],
        'require_false': [],
        'description': 'Account number line repaired with Account number',
        'needs_directory': True,
        'directory': 'ACCOUNT_DIRECTORY'
    },
    '9935': {
        'require_true': ['has_nch'],
        'require_false': [],
        'description': 'Account number line repaired by NCH code',
        'needs_directory': True,
        'directory': 'NCH_DIRECTORY'
    },
    '9962': {
        'require_true': ['has_account'],
        'require_false': [],
        'description': 'Account number updated',
        'needs_directory': True,
        'directory': 'ACCOUNT_DIRECTORY'
    },
    '9963': {
        'require_true': [],
        'require_false': ['has_account'],
        'description': 'Account number added',
        'needs_directory': True,
        'directory': 'ACCOUNT_DIRECTORY'
    },
    '9484': {
        'require_true': ['present'],
        'require_false': [],
        'description': 'Repaired using info derived from Ban to Bank Info',
        'needs_directory': True,
        'directory': 'BANK_DIRECTORY'
    },
    
    # --- Fee/Charge Updates ---
    '9490': {
        'require_true': [],
        'require_false': [],
        'description': 'Fee Code Updated',
        'needs_directory': True,
        'directory': 'FEE_DIRECTORY'
    },
    '9491': {
        'require_true': [],
        'require_false': [],
        'description': 'Tariff Fee Updated',
        'needs_directory': True,
        'directory': 'FEE_DIRECTORY'
    },
    '9492': {
        'require_true': [],
        'require_false': [],
        'description': 'Charge Field updated with value B',
        'needs_directory': False
    },
    '9493': {
        'require_true': [],
        'require_false': [],
        'description': 'Debit Charge Field update with value N',
        'needs_directory': False
    },
    '9494': {
        'require_true': [],
        'require_false': [],
        'description': 'Field updated with GTYOUR',
        'needs_directory': False
    },
    '9495': {
        'require_true': [],
        'require_false': [],
        'description': 'Field updated with FCY',
        'needs_directory': False
    },
    '9496': {
        'require_true': [],
        'require_false': [],
        'description': 'Field updated with FCY OUR CHARGES',
        'needs_directory': False
    },
    '9497': {
        'require_true': [],
        'require_false': [],
        'description': 'Field updated with FCY GTYOUR',
        'needs_directory': False
    },
    '9498': {
        'require_true': [],
        'require_false': [],
        'description': 'PRESAM SecWir updated with Y',
        'needs_directory': False
    },
    
    # --- Other Updates ---
    '9918': {
        'require_true': [],
        'require_false': [],
        'description': 'Attribute type auto corrected',
        'needs_directory': False
    },
    '9964': {
        'require_true': [],
        'require_false': [],
        'description': 'Advice instructions modified',
        'needs_directory': False
    },
    '9965': {
        'require_true': [],
        'require_false': [],
        'description': 'Charge flag modified',
        'needs_directory': False
    },
    '9966': {
        'require_true': [],
        'require_false': [],
        'description': 'Secondary wire flag modified',
        'needs_directory': False
    },
    '9968': {
        'require_true': [],
        'require_false': [],
        'description': 'Parameter values added',
        'needs_directory': False
    },
    '9969': {
        'require_true': [],
        'require_false': [],
        'description': 'Related Amounts added',
        'needs_directory': False
    },
    '9971': {
        'require_true': [],
        'require_false': [],
        'description': 'Negative ref cancels pos ref and benededuct',
        'needs_directory': False
    },
    '9987': {
        'require_true': [],
        'require_false': [],
        'description': 'Code Consistent',
        'needs_directory': False
    },
    
    # --- Catch-all/Generic ---
    '9439': {
        'require_true': [],
        'require_false': [],
        'description': 'No Pattern Found',
        'needs_directory': False,
        'trainable': False
    },
    '9999': {
        'require_true': [],
        'require_false': [],
        'description': 'Field Repaired',
        'needs_directory': False,
        'trainable': False
    },
}

# =============================================================================
# 9XXX CODE CATEGORIZATION
# =============================================================================

# Trainable 9XXX codes - can be predicted from message features alone
TRAINABLE_9XXX_CODES = [
    code for code, info in CODE_TRIGGERS.items() 
    if code.startswith('9') and not info.get('needs_directory', False) and info.get('trainable', True)
]

# Directory-dependent 9XXX codes - need external lookup
DIRECTORY_DEPENDENT_9XXX_CODES = [
    code for code, info in CODE_TRIGGERS.items() 
    if code.startswith('9') and info.get('needs_directory', False)
]

# Non-trainable codes (too generic)
NON_TRAINABLE_9XXX_CODES = [
    code for code, info in CODE_TRIGGERS.items() 
    if code.startswith('9') and info.get('trainable') == False
]

def get_9xxx_category(code: str) -> str:
    """Get category for a 9XXX code."""
    base_code = code.split('_')[0] if '_' in code else code
    if base_code in TRAINABLE_9XXX_CODES:
        return 'TRAINABLE'
    elif base_code in DIRECTORY_DEPENDENT_9XXX_CODES:
        return 'DIRECTORY_DEPENDENT'
    elif base_code in NON_TRAINABLE_9XXX_CODES:
        return 'NON_TRAINABLE'
    return 'UNKNOWN'

def get_lookup_condition(code: str, features: dict = None) -> Optional[str]:
    """
    Get the directory lookup condition for a 9XXX code.
    
    Returns formatted string like:
    "LOOKUP(BIC_DIRECTORY, bic=DEUTDEFF) -> IF found EMIT 9477 ELSE skip"
    """
    base_code = code.split('_')[0] if '_' in code else code
    info = CODE_TRIGGERS.get(base_code, {})
    
    if not info.get('needs_directory'):
        return None
    
    lookup_template = info.get('lookup_condition', '')
    
    if features and lookup_template:
        # Extract party prefix from composite code
        party_prefix = None
        if '_' in code:
            party_hint = code.split('_')[1]
            party_prefix = SUFFIX_TO_PREFIX.get(party_hint)
        
        # Substitute feature values into template
        replacements = {}
        if party_prefix:
            replacements['country'] = features.get(f'{party_prefix}_country') or features.get(f'{party_prefix}_iban_country') or '?'
            replacements['account'] = features.get(f'{party_prefix}_account_value') or '?'
            replacements['bic'] = features.get(f'{party_prefix}_bic') or features.get(f'{party_prefix}_id_value') or '?'
            replacements['iban'] = features.get(f'{party_prefix}_iban') or features.get(f'{party_prefix}_account_value') or '?'
            replacements['nch_value'] = features.get(f'{party_prefix}_nch_value') or '?'
        
        for key, val in replacements.items():
            lookup_template = lookup_template.replace('{' + key + '}', str(val))
    
    return lookup_template

def format_9xxx_prediction(code: str, features: dict, probability: float) -> dict:
    """
    Format a 9XXX prediction with appropriate category and lookup info.
    """
    base_code = code.split('_')[0] if '_' in code else code
    info = CODE_TRIGGERS.get(base_code, {})
    category = get_9xxx_category(code)
    
    result = {
        'code': code,
        'description': info.get('description', 'Unknown'),
        'probability': round(probability, 4),
        'category': category
    }
    
    if category == 'DIRECTORY_DEPENDENT':
        result['needs_directory'] = True
        result['directory'] = info.get('directory', 'UNKNOWN')
        lookup = get_lookup_condition(code, features)
        if lookup:
            result['lookup_condition'] = lookup
        result['note'] = 'Prediction conditional on directory lookup result'
    elif category == 'TRAINABLE':
        result['needs_directory'] = False
        # Add trigger features
        triggers = []
        party_prefix = None
        if '_' in code:
            party_hint = code.split('_')[1]
            party_prefix = SUFFIX_TO_PREFIX.get(party_hint)
        
        for feat in info.get('require_true', []) + info.get('soft_require', []):
            full_feat = f'{party_prefix}_{feat}' if party_prefix else feat
            val = features.get(full_feat) or features.get(feat)
            if val:
                triggers.append({'feature': full_feat, 'value': val})
        
        if triggers:
            result['triggers'] = triggers
    
    return result

# Party suffix to feature prefix mapping
SUFFIX_TO_PREFIX = {
    'ORGPTY': 'orig',
    'SNDBNK': 'send',
    'DBTPTY': 'dbt',
    'CDTPTY': 'cdt',
    'INTBNK': 'intm',
    'BNFBNK': 'bnf',
    'BNPPTY': 'bnf',
}

# All possible party prefixes
PARTY_PREFIXES = ['orig', 'send', 'dbt', 'cdt', 'intm', 'bnf', 'ordi', 'acwi']

# Party prefix to human-readable name
PARTY_NAMES = {
    'orig': 'Originator',
    'send': 'Sending Bank',
    'dbt': 'Debit Party',
    'cdt': 'Credit Party',
    'intm': 'Intermediary Bank',
    'bnf': 'Beneficiary Bank',
    'ordi': 'Originator (alt)',
    'acwi': 'Credit Party (alt)',
}


# =============================================================================
# PREDICTION FILTER
# =============================================================================

def filter_predictions(predicted_codes: List[str], features: Dict) -> Tuple[List[str], List[dict]]:
    """
    Filter out predictions that don't meet ACE preconditions.
    
    Checks two conditions:
    1. require_true: Fields that MUST be True (e.g., has_iban for IBAN errors)
    2. require_false: Fields where at least ONE must be False (e.g., iban_valid for Invalid IBAN)
       - If ALL require_false fields are True, the error shouldn't fire (data is valid)
    
    Args:
        predicted_codes: List of predicted error codes (e.g., ['8022_BNFBNK', '8004_BNPPTY'])
        features: Dict of input features from IFML parser
        
    Returns:
        Tuple of:
        - filtered_codes: Predictions that pass precondition checks
        - filtered_out: List of {code, reason} for codes that were removed
        
    Example:
        >>> features = {'bnf_has_iban': True, 'bnf_iban_valid_format': True, 'bnf_iban_checksum_valid': True}
        >>> predicted = ['8894_BNFBNK']  # Invalid IBAN
        >>> filtered, removed = filter_predictions(predicted, features)
        >>> filtered
        []  # Filtered out because IBAN is valid
    """
    filtered_codes = []
    filtered_out = []
    
    for code in predicted_codes:
        base_code = code.split('_')[0]
        party_suffix = code.split('_')[1] if '_' in code else None
        
        # Get trigger definition
        trigger_def = CODE_TRIGGERS.get(base_code, {})
        require_true = trigger_def.get('require_true', [])
        require_false = trigger_def.get('require_false', [])
        
        if not require_true and not require_false:
            # No preconditions defined, keep the prediction
            filtered_codes.append(code)
            continue
        
        # Determine which party prefix to check
        if party_suffix:
            party_prefix = SUFFIX_TO_PREFIX.get(party_suffix)
            prefixes_to_check = [party_prefix] if party_prefix else PARTY_PREFIXES
        else:
            prefixes_to_check = PARTY_PREFIXES
        
        # Check conditions for at least one party
        passes = False
        failed_reason = None
        
        for prefix in prefixes_to_check:
            # === Check require_true: ALL must be True ===
            require_true_met = True
            true_failures = []
            
            for feat_base in require_true:
                feat_name = f'{prefix}_{feat_base}'
                value = features.get(feat_name)
                
                # Also check without prefix
                if value is None:
                    value = features.get(feat_base)
                    feat_name = feat_base
                
                if not (isinstance(value, bool) and value):
                    require_true_met = False
                    true_failures.append(f"{feat_name}={value}")
            
            if not require_true_met:
                failed_reason = f"requires {require_true}, got {true_failures}"
                continue  # Try next party
            
            # === Check require_false: At least ONE must be False ===
            # (Error fires when validation fails, so at least one validity check must be False)
            if require_false:
                any_false = False
                all_true_fields = []
                
                for feat_base in require_false:
                    feat_name = f'{prefix}_{feat_base}'
                    value = features.get(feat_name)
                    
                    # Also check without prefix
                    if value is None:
                        value = features.get(feat_base)
                        feat_name = feat_base
                    
                    if isinstance(value, bool):
                        if not value:
                            any_false = True  # Found an invalid/failed check
                            break
                        else:
                            all_true_fields.append(feat_name)
                    elif value is None:
                        # Field not present - could mean validation wasn't done
                        # Be conservative: treat as potentially invalid
                        any_false = True
                        break
                
                if not any_false and all_true_fields:
                    # All validity checks passed - this error shouldn't fire
                    failed_reason = f"all validity checks passed: {all_true_fields}"
                    continue  # Try next party
            
            # All conditions met for this party
            passes = True
            break
        
        if passes:
            filtered_codes.append(code)
        else:
            filtered_out.append({
                'code': code,
                'reason': f"Precondition failed: {failed_reason}"
            })
    
    return filtered_codes, filtered_out


# Alias for backward compatibility
filter_false_positives = filter_predictions


def get_code_description(code: str) -> Optional[str]:
    """Get ACE description for a code."""
    base_code = code.split('_')[0]
    trigger_def = CODE_TRIGGERS.get(base_code, {})
    return trigger_def.get('description')


def get_code_preconditions(code: str) -> dict:
    """Get preconditions for a code."""
    base_code = code.split('_')[0]
    return CODE_TRIGGERS.get(base_code, {})


# =============================================================================
# PREDICTION EXPLANATION
# =============================================================================

def _suffix_to_prefix(suffix: str) -> Optional[str]:
    """Convert party suffix to feature prefix."""
    return SUFFIX_TO_PREFIX.get(suffix)


def get_feature_explanation(feature_name: str) -> str:
    """Get human-readable explanation for a feature."""
    for prefix in PARTY_PREFIXES:
        if feature_name.startswith(f'{prefix}_'):
            base_name = feature_name[len(prefix)+1:]
            party_name = PARTY_NAMES.get(prefix, prefix)
            base_explanation = FEATURE_EXPLANATIONS.get(base_name, base_name)
            return f"{party_name}: {base_explanation}"
    
    return FEATURE_EXPLANATIONS.get(feature_name, feature_name)


def explain_prediction(features: Dict, predicted_codes: List[str], 
                       feature_importance: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Explain why each code was predicted using ACE Pelican trigger logic.
    
    Returns dict mapping code -> dict with:
        - triggers: list of triggering conditions
        - warning: optional warning if prediction may be false positive
        - description: code description from ACE
    """
    explanations = {}
    
    for code in predicted_codes:
        base_code = code.split('_')[0]
        party_suffix = code.split('_')[1] if '_' in code else None
        
        triggers = []
        warning = None
        
        # Get trigger definition for this code
        trigger_def = CODE_TRIGGERS.get(base_code, {})
        require_true = trigger_def.get('require_true', [])
        require_false = trigger_def.get('require_false', [])
        description = trigger_def.get('description', '')
        
        # Determine which party to check
        if party_suffix:
            party_prefix = _suffix_to_prefix(party_suffix)
            prefixes_to_check = [party_prefix] if party_prefix else PARTY_PREFIXES
        else:
            prefixes_to_check = PARTY_PREFIXES
        
        # Track if trigger conditions are met
        true_conditions_met = []
        true_conditions_failed = []
        false_conditions_met = []
        false_conditions_failed = []
        
        for prefix in prefixes_to_check:
            # Check require_true features (must be True for valid trigger)
            for feat_base in require_true:
                feat_name = f'{prefix}_{feat_base}'
                if feat_name in features:
                    value = features[feat_name]
                    if isinstance(value, bool):
                        if value:
                            true_conditions_met.append({
                                'feature': feat_name,
                                'value': value,
                                'explanation': get_feature_explanation(feat_name),
                                'trigger_type': 'present'
                            })
                        else:
                            true_conditions_failed.append(feat_name)
            
            # Check require_false features (must be False for valid trigger)
            for feat_base in require_false:
                feat_name = f'{prefix}_{feat_base}'
                if feat_name in features:
                    value = features[feat_name]
                    if isinstance(value, bool):
                        if not value:
                            false_conditions_met.append({
                                'feature': feat_name,
                                'value': value,
                                'explanation': get_feature_explanation(feat_name),
                                'trigger_type': 'invalid/missing'
                            })
                        else:
                            false_conditions_failed.append(feat_name)
        
        # Build trigger list - show conditions that match expected pattern
        triggers = true_conditions_met + false_conditions_met
        
        # Detect potential false positive
        if require_true and not true_conditions_met:
            warning = f"⚠️ POSSIBLE FALSE POSITIVE: {base_code} requires {require_true} but none found"
        elif true_conditions_failed and require_true:
            warning = f"⚠️ POSSIBLE FALSE POSITIVE: Expected {true_conditions_failed} = True but got False"
        
        # Sort by importance if available
        if feature_importance:
            triggers.sort(key=lambda x: -feature_importance.get(x['feature'], 0))
        
        explanations[code] = {
            'triggers': triggers,
            'warning': warning,
            'description': description
        }
    
    return explanations


def format_explanation(txn_id: str, features: Dict, predicted_codes: List[str], 
                       actual_codes: List[str], explanations: Dict,
                       filtered_out: List[dict] = None) -> List[str]:
    """Format explanation as markdown."""
    lines = []
    lines.append(f"### Transaction: {txn_id}")
    lines.append("")
    
    # Show prediction vs actual
    pred_str = ', '.join(predicted_codes) if predicted_codes else 'None'
    actual_str = ', '.join(actual_codes) if actual_codes else 'None'
    
    match = set(predicted_codes) == set(actual_codes)
    status = "✓ MATCH" if match else "✗ MISMATCH"
    
    lines.append(f"**Predicted:** {pred_str}")
    lines.append(f"**Actual:** {actual_str}")
    lines.append(f"**Status:** {status}")
    
    # Show filtered out codes
    if filtered_out:
        filtered_codes = [f['code'] for f in filtered_out]
        lines.append(f"**Filtered out (false positives):** {', '.join(filtered_codes)}")
    
    lines.append("")
    
    # Show explanations for each predicted code
    for code in predicted_codes:
        lines.append(f"#### Why {code}?")
        lines.append("")
        
        code_explanation = explanations.get(code, {})
        
        # Handle both old format (list) and new format (dict)
        if isinstance(code_explanation, list):
            triggers = code_explanation
            warning = None
            description = None
        else:
            triggers = code_explanation.get('triggers', [])
            warning = code_explanation.get('warning')
            description = code_explanation.get('description')
        
        # Show description
        if description:
            lines.append(f"**ACE Definition:** {description}")
            lines.append("")
        
        # Show warning if present
        if warning:
            lines.append(f"**{warning}**")
            lines.append("")
        
        if triggers:
            lines.append("| Field | Value | Explanation |")
            lines.append("|-------|-------|-------------|")
            for t in triggers[:5]:  # Top 5 triggers
                feat = t['feature']
                val = t['value']
                expl = t['explanation']
                lines.append(f"| `{feat}` | {val} | {expl} |")
            lines.append("")
        else:
            lines.append("*No specific triggers identified (may be pattern-based)*")
            lines.append("")
    
    # Show key input values
    lines.append("#### Key Input Fields")
    lines.append("")
    lines.append("```")
    
    key_fields = [
        'primary_currency', 'is_cross_border',
        'bnf_has_iban', 'bnf_needs_iban', 'bnf_has_bic', 'bnf_has_nch',
        'bnf_bic_iban_match', 'bnf_iban_valid_format', 'bnf_iban_checksum_valid',
        'cdt_has_iban', 'cdt_needs_iban', 'cdt_has_account',
        'intm_present', 'intm_has_bic',
    ]
    
    for field in key_fields:
        if field in features:
            lines.append(f"{field}: {features[field]}")
    
    lines.append("```")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    return lines


# =============================================================================
# MODEL LOADING AND PREDICTION
# =============================================================================

def load_feature_importance(model_dir: str) -> Dict[str, float]:
    """Load feature importance from model."""
    importance = {}
    
    try:
        info_path = Path(model_dir) / 'training_info.json'
        if info_path.exists():
            with open(info_path) as f:
                info = json.load(f)
                if 'feature_importance' in info:
                    for item in info['feature_importance']:
                        importance[item['feature']] = item['importance']
    except Exception as e:
        print(f"Warning: Could not load feature importance: {e}")
    
    return importance


def load_model_components(model_dir: str):
    """Load model, feature engineer, and class names."""
    import pandas as pd
    
    model_path = Path(model_dir) / 'model.pkl'
    fe_path = Path(model_dir) / 'feature_engineer.pkl'
    info_path = Path(model_dir) / 'training_info.json'
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    # Handle model as dict
    if isinstance(model_data, dict):
        model = model_data.get('model')
        print(f"Extracted model from dict, type: {type(model).__name__}")
    else:
        model = model_data
        print(f"Model type: {type(model).__name__}")
    
    with open(fe_path, 'rb') as f:
        feature_engineer = pickle.load(f)
    
    # If feature_engineer is a dict, try to reconstruct from module
    if isinstance(feature_engineer, dict):
        try:
            from feature_engineering import IFMLFeatureEngineer
            fe_obj = IFMLFeatureEngineer()
            if 'label_encoders' in feature_engineer:
                fe_obj.label_encoders = feature_engineer['label_encoders']
            if 'frequency_maps' in feature_engineer:
                fe_obj.frequency_maps = feature_engineer['frequency_maps']
            if 'feature_columns' in feature_engineer:
                fe_obj.feature_columns = feature_engineer['feature_columns']
            fe_obj.fitted = True
            feature_engineer = fe_obj
            print("Reconstructed feature engineer from dict")
        except ImportError:
            print("Warning: Could not import IFMLFeatureEngineer, using dict mode")
    
    with open(info_path) as f:
        info = json.load(f)
    
    class_names = info.get('class_names', [])
    threshold = info.get('threshold', 0.5)
    feature_columns = info.get('feature_columns', [])
    
    return model, feature_engineer, class_names, threshold, feature_columns


def transform_features(features: Dict, feature_engineer, feature_columns: List[str]):
    """Transform features dict to model input."""
    import pandas as pd
    
    # If feature_engineer is an object with transform method, use it
    if hasattr(feature_engineer, 'transform'):
        df = pd.DataFrame([features])
        return feature_engineer.transform(df)
    
    # If feature_engineer is a dict, manually build feature vector
    if isinstance(feature_engineer, dict):
        label_encoders = feature_engineer.get('label_encoders', {})
        frequency_maps = feature_engineer.get('frequency_maps', {})
        columns = feature_engineer.get('feature_columns', feature_columns)
        
        if not columns:
            raise ValueError("No feature_columns found")
        
        row = {}
        for col in columns:
            if col in features:
                val = features[col]
                
                if col in label_encoders:
                    le = label_encoders[col]
                    if hasattr(le, 'transform'):
                        try:
                            val = le.transform([str(val)])[0]
                        except:
                            val = 0
                    elif isinstance(le, dict):
                        val = le.get(str(val), 0)
                elif col in frequency_maps:
                    freq_map = frequency_maps[col]
                    val = freq_map.get(str(val), 0.0)
                elif isinstance(val, bool):
                    val = 1 if val else 0
                
                row[col] = val
            else:
                row[col] = 0
        
        df = pd.DataFrame([row])
        for col in columns:
            if col not in df.columns:
                df[col] = 0
        df = df[columns]
        
        return df.values
    
    raise ValueError(f"Unknown feature_engineer type: {type(feature_engineer)}")


def predict_from_features(features: Dict, model, feature_engineer, 
                          class_names: List[str], threshold: float = 0.5,
                          series: str = '8', feature_columns: List[str] = None) -> List[str]:
    """Run prediction directly from features dict."""
    X = transform_features(features, feature_engineer, feature_columns or [])
    
    if hasattr(model, 'predict_proba'):
        probas = model.predict_proba(X)
        
        if isinstance(probas, list):
            predicted = []
            for idx, proba in enumerate(probas):
                if idx < len(class_names):
                    code = class_names[idx]
                    if code.startswith(series):
                        if len(proba.shape) > 1 and proba.shape[1] >= 2:
                            prob = proba[0, 1]
                        else:
                            prob = proba[0]
                        if prob >= threshold:
                            predicted.append(code)
            return predicted
        else:
            predicted = []
            for idx, prob in enumerate(probas[0]):
                if idx < len(class_names):
                    code = class_names[idx]
                    if code.startswith(series) and prob >= threshold:
                        predicted.append(code)
            return predicted
    else:
        preds = model.predict(X)
        predicted = []
        for idx, pred in enumerate(preds[0]):
            if pred == 1 and idx < len(class_names):
                code = class_names[idx]
                if code.startswith(series):
                    predicted.append(code)
        return predicted


# =============================================================================
# CLI FUNCTIONS
# =============================================================================

def run_explanations(model_dir: str, data_dir: str, output_file: str, 
                     limit: int = 100, series: str = '8'):
    """Run predictions and generate explanations."""
    from data_pipeline import IFMLDataPipeline
    
    print(f"Loading model from {model_dir}...")
    model, feature_engineer, class_names, threshold, feature_columns = load_model_components(model_dir)
    importance = load_feature_importance(model_dir)
    
    print(f"Loading data from {data_dir}...")
    pipeline = IFMLDataPipeline()
    pipeline.load_combined_files(data_dir, '*.json')
    
    print(f"Loaded {len(pipeline.records)} records")
    
    lines = [
        f"# {series}XXX Prediction Explanations",
        "",
        "## Overview",
        "",
        f"This document shows what input fields triggered each {series}XXX prediction.",
        "",
        "---",
        ""
    ]
    
    count = 0
    total_filtered = 0
    
    for rec in pipeline.records:
        if count >= limit:
            break
        
        features = rec.request_features
        
        if hasattr(rec, 'composite_codes') and rec.composite_codes:
            actual = [c for c in rec.composite_codes if c.startswith(series)]
        else:
            actual = [c for c in rec.error_codes_only if c.startswith(series)]
        
        if not actual:
            continue
        
        # Run prediction
        raw_predicted = predict_from_features(features, model, feature_engineer, 
                                              class_names, threshold, series, feature_columns)
        
        # Apply false positive filter
        predicted, filtered_out = filter_predictions(raw_predicted, features)
        total_filtered += len(filtered_out)
        
        # Get explanations
        explanations = explain_prediction(features, predicted, importance)
        
        # Format
        txn_id = rec.transaction_id if hasattr(rec, 'transaction_id') else f"record_{count}"
        formatted = format_explanation(txn_id, features, predicted, actual, explanations, filtered_out)
        lines.extend(formatted)
        
        count += 1
        if count % 20 == 0:
            print(f"Processed {count} records...")
    
    # Add summary
    lines.insert(7, f"**Post-prediction filter removed {total_filtered} false positives**")
    lines.insert(8, "")
    
    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"Explanations written to {output_file}")
    print(f"Filtered out {total_filtered} false positive predictions")


def explain_single_payment(model_dir: str, payment_json: str, series: str = '8'):
    """Explain prediction for a single payment."""
    from ifml_parser import IFMLParser
    
    model, feature_engineer, class_names, threshold, feature_columns = load_model_components(model_dir)
    importance = load_feature_importance(model_dir)
    
    with open(payment_json) as f:
        data = json.load(f)
    
    parser = IFMLParser()
    features_obj = parser.parse(data)
    features = parser.to_dict(features_obj)
    
    # Predict and filter
    raw_predicted = predict_from_features(features, model, feature_engineer,
                                          class_names, threshold, series, feature_columns)
    predicted, filtered_out = filter_predictions(raw_predicted, features)
    
    # Explain
    explanations = explain_prediction(features, predicted, importance)
    
    print("\n" + "="*60)
    print("PREDICTION EXPLANATION")
    print("="*60)
    print(f"\nRaw predicted: {raw_predicted}")
    print(f"After filter: {predicted}")
    
    if filtered_out:
        print(f"\nFiltered out:")
        for f in filtered_out:
            print(f"  - {f['code']}: {f['reason']}")
    
    print("")
    
    for code in predicted:
        print(f"\n--- {code} ---")
        code_exp = explanations.get(code, {})
        
        if isinstance(code_exp, dict):
            desc = code_exp.get('description')
            triggers = code_exp.get('triggers', [])
            warning = code_exp.get('warning')
            
            if desc:
                print(f"ACE Definition: {desc}")
            if warning:
                print(f"{warning}")
            
            if triggers:
                print(f"{'Feature':<40} {'Value':<10} {'Explanation'}")
                print("-" * 70)
                for t in triggers:
                    print(f"{t['feature']:<40} {str(t['value']):<10} {t['explanation']}")
            else:
                print("No specific triggers identified")
        else:
            print("No explanation available")
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description='ACE Prediction Filter & Explainer')
    parser.add_argument('--model-dir', required=True, help='Model directory')
    parser.add_argument('--data-dir', default=None, help='Raw data directory')
    parser.add_argument('--payment', default=None, help='Single payment JSON file')
    parser.add_argument('--output', default='explanations.md', help='Output file')
    parser.add_argument('--limit', type=int, default=100, help='Max records to process')
    parser.add_argument('--series', default='8', help='Code series (8 or 9)')
    
    args = parser.parse_args()
    
    if args.payment:
        explain_single_payment(args.model_dir, args.payment, args.series)
    elif args.data_dir:
        run_explanations(args.model_dir, args.data_dir, args.output, args.limit, args.series)
    else:
        print("Error: Either --data-dir or --payment required")
        print("\nUsage examples:")
        print("  python prediction_utils.py --model-dir ./models_8x --data-dir ./raw_data --output explanations.md")
        print("  python prediction_utils.py --model-dir ./models_8x --payment ./test.json")
        print("\nAs a module:")
        print("  from prediction_utils import filter_predictions, explain_prediction")
        print("  filtered, removed = filter_predictions(predicted_codes, features)")


if __name__ == '__main__':
    main()
