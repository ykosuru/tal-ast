#!/usr/bin/env python3
"""
================================================================================
ENHANCED FEATURE EXTRACTOR FOR ACE ERROR CODE PREDICTION
================================================================================

This module adds comprehensive feature extraction including:
- Party counts and relationships
- Cross-border detection
- ID comparison across parties
- Country-specific validation (IBAN/BBAN per country)
- Routing chain analysis
- Message type detection
- Domestic vs international classification

Usage:
    from enhanced_features import EnhancedFeatureExtractor
    
    extractor = EnhancedFeatureExtractor()
    features = extractor.extract(ifml_dict)
    
    # features is a flat dict ready for ML

AUTHOR: ACE Pelican ML Team
VERSION: 1.0 (December 2025)
================================================================================
"""

import json
import re
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict

# Import base parser
try:
    from ifml_parser_v3 import IFMLParser, IFMLFeatures, PartyInfo, IBAN_LENGTHS, VALID_COUNTRY_CODES
except ImportError:
    from ifml_parser_v2 import IFMLParser, IFMLFeatures, PartyInfo
    IBAN_LENGTHS = {}
    VALID_COUNTRY_CODES = set()


# =============================================================================
# COUNTRY-SPECIFIC VALIDATION RULES
# =============================================================================

# Countries that require IBAN for international payments
IBAN_REQUIRED_COUNTRIES = {
    # Europe (SEPA)
    'AT', 'BE', 'BG', 'HR', 'CY', 'CZ', 'DK', 'EE', 'FI', 'FR', 'DE', 'GR',
    'HU', 'IS', 'IE', 'IT', 'LV', 'LI', 'LT', 'LU', 'MT', 'MC', 'NL', 'NO',
    'PL', 'PT', 'RO', 'SM', 'SK', 'SI', 'ES', 'SE', 'CH', 'GB',
    # Middle East
    'AE', 'BH', 'IL', 'JO', 'KW', 'LB', 'QA', 'SA', 'TR',
    # Africa
    'EG', 'MA', 'TN',
    # Other
    'BR', 'CR', 'GT', 'MU', 'PK',
}

# Countries using domestic account formats (not IBAN)
DOMESTIC_ACCOUNT_COUNTRIES = {
    'US': {'format': 'ABA+Account', 'routing_length': 9, 'account_length': (4, 17)},
    'CA': {'format': 'Transit+Account', 'routing_length': 9, 'account_length': (7, 12)},
    'AU': {'format': 'BSB+Account', 'routing_length': 6, 'account_length': (5, 10)},
    'NZ': {'format': 'Bank+Branch+Account', 'routing_length': 6, 'account_length': (7, 10)},
    'JP': {'format': 'Bank+Branch+Account', 'routing_length': 7, 'account_length': (7, 8)},
    'CN': {'format': 'CNAPS', 'routing_length': 12, 'account_length': (16, 19)},
    'IN': {'format': 'IFSC+Account', 'routing_length': 11, 'account_length': (9, 18)},
    'MX': {'format': 'CLABE', 'routing_length': 0, 'account_length': (18, 18)},
    'ZA': {'format': 'Branch+Account', 'routing_length': 6, 'account_length': (7, 11)},
    'HK': {'format': 'Bank+Account', 'routing_length': 3, 'account_length': (9, 12)},
    'SG': {'format': 'Bank+Branch+Account', 'routing_length': 7, 'account_length': (10, 14)},
}

# SWIFT message types
MESSAGE_TYPES = {
    '103': 'customer_transfer',
    '202': 'bank_transfer',
    '202COV': 'cover_payment',
    '200': 'bank_transfer_own',
    '205': 'bank_transfer_third',
    '910': 'confirmation_credit',
    '940': 'statement',
    '950': 'statement',
}


# =============================================================================
# ENHANCED FEATURE EXTRACTOR
# =============================================================================

class EnhancedFeatureExtractor:
    """
    Enhanced feature extraction for ACE error code prediction.
    Adds comprehensive cross-party analysis and country-specific validation.
    """
    
    def __init__(self):
        self.parser = IFMLParser()
        
        # Party type mappings
        self.party_prefixes = {
            'OriginatingPartyInfo': 'orig',
            'SendingBankInfo': 'send',
            'DebitPartyInfo': 'dbt',
            'CreditPartyInfo': 'cdt',
            'IntermediaryBankInfo': 'intm',
            'BeneficiaryBankInfo': 'bnf',
            'BeneficiaryPartyInfo': 'bnp',
            'AccountWithInstitution': 'acwi',
            'OrderingInstitution': 'ordi',
        }
        
        self.bank_parties = {'SendingBankInfo', 'IntermediaryBankInfo', 
                            'BeneficiaryBankInfo', 'AccountWithInstitution', 'OrderingInstitution'}
        self.non_bank_parties = {'OriginatingPartyInfo', 'DebitPartyInfo', 
                                 'CreditPartyInfo', 'BeneficiaryPartyInfo'}
    
    def extract(self, ifml: Dict) -> Dict[str, Any]:
        """
        Extract comprehensive features from IFML message.
        
        Returns flat dict with all features ready for ML.
        """
        # Parse with base parser first
        parsed = self.parser.parse(ifml)
        base_features = self.parser.to_dict(parsed)
        
        # Add enhanced features
        enhanced = self._extract_enhanced_features(parsed, ifml)
        
        # Merge
        base_features.update(enhanced)
        
        return base_features
    
    def _extract_enhanced_features(self, parsed: IFMLFeatures, 
                                    raw_ifml: Dict) -> Dict[str, Any]:
        """Extract additional enhanced features."""
        features = {}
        
        # === PARTY COUNTS ===
        features.update(self._count_parties(parsed))
        
        # === CROSS-BORDER ANALYSIS ===
        features.update(self._analyze_cross_border(parsed))
        
        # === ID COMPARISON ===
        features.update(self._compare_ids(parsed))
        
        # === COUNTRY-SPECIFIC VALIDATION ===
        features.update(self._validate_by_country(parsed))
        
        # === ROUTING CHAIN ANALYSIS ===
        features.update(self._analyze_routing(parsed, raw_ifml))
        
        # === MESSAGE TYPE DETECTION ===
        features.update(self._detect_message_type(parsed, raw_ifml))
        
        # === DOMESTIC VS INTERNATIONAL ===
        features.update(self._classify_payment_type(parsed))
        
        # === CONSISTENCY CHECKS ===
        features.update(self._check_consistency(parsed))
        
        return features
    
    def _count_parties(self, parsed: IFMLFeatures) -> Dict[str, Any]:
        """Count parties by type."""
        features = {}
        
        total = 0
        banks = 0
        non_banks = 0
        
        for party_type, party in parsed.parties.items():
            if party:
                total += 1
                if party_type in self.bank_parties:
                    banks += 1
                else:
                    non_banks += 1
        
        features['total_party_count'] = total
        features['bank_party_count'] = banks
        features['non_bank_party_count'] = non_banks
        features['has_originator'] = 'OriginatingPartyInfo' in parsed.parties and parsed.parties['OriginatingPartyInfo'] is not None
        features['has_debtor'] = 'DebitPartyInfo' in parsed.parties and parsed.parties['DebitPartyInfo'] is not None
        features['has_creditor'] = 'CreditPartyInfo' in parsed.parties and parsed.parties['CreditPartyInfo'] is not None
        features['has_beneficiary_party'] = 'BeneficiaryPartyInfo' in parsed.parties and parsed.parties['BeneficiaryPartyInfo'] is not None
        features['has_beneficiary_bank'] = 'BeneficiaryBankInfo' in parsed.parties and parsed.parties['BeneficiaryBankInfo'] is not None
        features['has_intermediary'] = 'IntermediaryBankInfo' in parsed.parties and parsed.parties['IntermediaryBankInfo'] is not None
        features['has_sending_bank'] = 'SendingBankInfo' in parsed.parties and parsed.parties['SendingBankInfo'] is not None
        
        # Intermediary count (from parsed features)
        features['intermediary_count'] = parsed.intm_count
        features['has_multiple_intermediaries'] = parsed.intm_has_multiple
        
        return features
    
    def _analyze_cross_border(self, parsed: IFMLFeatures) -> Dict[str, Any]:
        """Analyze cross-border payment characteristics."""
        features = {}
        
        # Collect all countries
        countries = {}  # party_type -> country
        
        for party_type, party in parsed.parties.items():
            if not party:
                continue
            
            # Get country from multiple sources
            country = (
                party.bic_country or 
                party.iban_country or 
                party.country or 
                party.mailing_country or
                party.residence_country or
                party.address_country
            )
            
            if country:
                country = country.upper()
                countries[party_type] = country
                prefix = self.party_prefixes.get(party_type, party_type[:4].lower())
                features[f'{prefix}_country_code'] = country
                features[f'{prefix}_is_iban_country'] = country in IBAN_REQUIRED_COUNTRIES
                features[f'{prefix}_is_domestic_format_country'] = country in DOMESTIC_ACCOUNT_COUNTRIES
        
        # Determine originator and beneficiary countries
        orig_country = (
            countries.get('OriginatingPartyInfo') or 
            countries.get('DebitPartyInfo') or
            countries.get('SendingBankInfo')
        )
        
        benef_country = (
            countries.get('BeneficiaryPartyInfo') or 
            countries.get('BeneficiaryBankInfo') or
            countries.get('CreditPartyInfo')
        )
        
        features['originator_country'] = orig_country
        features['beneficiary_country'] = benef_country
        
        # Cross-border detection
        if orig_country and benef_country:
            features['is_cross_border'] = orig_country != benef_country
            features['is_domestic'] = orig_country == benef_country
        else:
            features['is_cross_border'] = None
            features['is_domestic'] = None
        
        # Specific country pair flags
        features['orig_is_us'] = orig_country == 'US' if orig_country else False
        features['benef_is_us'] = benef_country == 'US' if benef_country else False
        features['is_us_domestic'] = orig_country == 'US' and benef_country == 'US'
        features['is_us_outbound'] = orig_country == 'US' and benef_country != 'US' if benef_country else False
        features['is_us_inbound'] = orig_country != 'US' and benef_country == 'US' if orig_country else False
        
        # EU/SEPA detection
        sepa_countries = {'AT', 'BE', 'BG', 'HR', 'CY', 'CZ', 'DK', 'EE', 'FI', 'FR', 
                         'DE', 'GR', 'HU', 'IS', 'IE', 'IT', 'LV', 'LI', 'LT', 'LU', 
                         'MT', 'MC', 'NL', 'NO', 'PL', 'PT', 'RO', 'SM', 'SK', 'SI', 
                         'ES', 'SE', 'CH', 'GB'}
        
        features['orig_is_sepa'] = orig_country in sepa_countries if orig_country else False
        features['benef_is_sepa'] = benef_country in sepa_countries if benef_country else False
        features['is_sepa_payment'] = (orig_country in sepa_countries and 
                                        benef_country in sepa_countries) if orig_country and benef_country else False
        
        # Count unique countries
        unique_countries = set(countries.values())
        features['unique_country_count'] = len(unique_countries)
        features['all_same_country'] = len(unique_countries) <= 1
        
        return features
    
    def _compare_ids(self, parsed: IFMLFeatures) -> Dict[str, Any]:
        """Compare IDs across parties to detect duplicates/inconsistencies."""
        features = {}
        
        # Collect all BICs
        bics = {}
        ibans = {}
        accounts = {}
        
        for party_type, party in parsed.parties.items():
            if not party:
                continue
            
            prefix = self.party_prefixes.get(party_type, party_type[:4].lower())
            
            if party.bic:
                bics[party_type] = party.bic.upper()
            if party.has_iban and party.account_value:
                ibans[party_type] = party.account_value.upper().replace(' ', '').replace('-', '')
            if party.account_value and not party.has_iban:
                accounts[party_type] = party.account_value
        
        # BIC analysis
        unique_bics = set(bics.values())
        features['unique_bic_count'] = len(unique_bics)
        features['all_bics_same'] = len(unique_bics) <= 1 and len(bics) > 0
        features['has_duplicate_bic'] = len(unique_bics) < len(bics)
        
        # Check if originator and beneficiary share BIC (unusual)
        orig_bic = bics.get('OriginatingPartyInfo') or bics.get('DebitPartyInfo') or bics.get('SendingBankInfo')
        benef_bic = bics.get('BeneficiaryBankInfo') or bics.get('BeneficiaryPartyInfo')
        features['orig_benef_same_bic'] = orig_bic == benef_bic if orig_bic and benef_bic else False
        
        # BIC4 (bank code) comparison
        bic4s = {k: v[:4] for k, v in bics.items() if len(v) >= 4}
        unique_bic4s = set(bic4s.values())
        features['unique_bic4_count'] = len(unique_bic4s)
        features['has_duplicate_bic4'] = len(unique_bic4s) < len(bic4s)
        
        # IBAN analysis
        unique_ibans = set(ibans.values())
        features['unique_iban_count'] = len(unique_ibans)
        features['has_duplicate_iban'] = len(unique_ibans) < len(ibans)
        features['iban_count'] = len(ibans)
        
        # IBAN country consistency with BIC
        for party_type in ibans.keys():
            party = parsed.parties.get(party_type)
            if party and party.bic_country and party.iban_country:
                prefix = self.party_prefixes.get(party_type, party_type[:4].lower())
                features[f'{prefix}_bic_iban_country_match'] = party.bic_country == party.iban_country
        
        # Account analysis
        features['account_count'] = len(accounts)
        features['has_multiple_accounts'] = len(accounts) > 1
        
        return features
    
    def _validate_by_country(self, parsed: IFMLFeatures) -> Dict[str, Any]:
        """Perform country-specific validation."""
        features = {}
        
        for party_type, party in parsed.parties.items():
            if not party:
                continue
            
            prefix = self.party_prefixes.get(party_type, party_type[:4].lower())
            
            # Determine party's country
            country = (
                party.bic_country or 
                party.iban_country or 
                party.country or 
                party.address_country
            )
            
            if not country:
                continue
            
            country = country.upper()
            
            # IBAN validation for IBAN-required countries
            if country in IBAN_REQUIRED_COUNTRIES:
                features[f'{prefix}_iban_required'] = True
                features[f'{prefix}_has_required_iban'] = party.has_iban
                features[f'{prefix}_missing_required_iban'] = not party.has_iban and not party.has_account
                
                # IBAN length validation
                if party.has_iban and party.account_value:
                    expected_len = IBAN_LENGTHS.get(country)
                    actual_len = len(party.account_value.replace(' ', '').replace('-', ''))
                    features[f'{prefix}_iban_correct_length'] = actual_len == expected_len if expected_len else True
                    features[f'{prefix}_iban_length'] = actual_len
                    features[f'{prefix}_iban_expected_length'] = expected_len
            else:
                features[f'{prefix}_iban_required'] = False
            
            # Domestic account validation
            if country in DOMESTIC_ACCOUNT_COUNTRIES:
                rules = DOMESTIC_ACCOUNT_COUNTRIES[country]
                features[f'{prefix}_domestic_format'] = rules['format']
                
                if party.account_value and not party.has_iban:
                    acct_len = len(party.account_value.replace(' ', '').replace('-', ''))
                    min_len, max_len = rules['account_length']
                    features[f'{prefix}_account_length_valid'] = min_len <= acct_len <= max_len
                
                # Check for routing number
                if party.has_nch or party.has_adr_bank_id:
                    features[f'{prefix}_has_routing'] = True
                    if party.nch_value:
                        routing_len = len(party.nch_value.replace(' ', '').replace('-', ''))
                        features[f'{prefix}_routing_length_valid'] = routing_len == rules['routing_length']
                else:
                    features[f'{prefix}_has_routing'] = False
        
        return features
    
    def _analyze_routing(self, parsed: IFMLFeatures, raw_ifml: Dict) -> Dict[str, Any]:
        """Analyze routing chain and correspondent banking."""
        features = {}
        
        # Count intermediaries
        features['intermediary_count'] = parsed.intm_count
        features['has_correspondent_chain'] = parsed.intm_count > 0
        features['correspondent_chain_length'] = parsed.intm_count + 2  # +2 for sender and receiver
        
        # Routing type detection
        has_fedwire = False
        has_chips = False
        has_swift = False
        has_target2 = False
        
        for party_type, party in parsed.parties.items():
            if not party:
                continue
            
            if party.nch_type == 'FEDABA' or party.is_fedaba:
                has_fedwire = True
            if party.nch_type == 'CHIPS' or party.is_chips_aba:
                has_chips = True
            if party.has_bic:
                has_swift = True
        
        features['uses_fedwire'] = has_fedwire
        features['uses_chips'] = has_chips
        features['uses_swift'] = has_swift
        features['routing_method_count'] = sum([has_fedwire, has_chips, has_swift])
        
        # Check for cover payment indicators
        features['may_be_cover_payment'] = parsed.intm_count > 0 and has_swift
        
        # Intermediary redundancy (already in parser, but add explicit flags)
        features['intm_likely_redundant'] = parsed.intm_has_redundant_info
        features['intm_share_country'] = parsed.intm_entries_share_country
        features['intm_share_bic'] = parsed.intm_entries_share_bic_prefix
        features['intm_share_adr_bank_id'] = parsed.intm_entries_share_adr_bank_id
        
        return features
    
    def _detect_message_type(self, parsed: IFMLFeatures, raw_ifml: Dict) -> Dict[str, Any]:
        """Detect SWIFT message type and characteristics."""
        features = {}
        
        # Get message type from parsed data
        msg_type = parsed.incoming_msg_type
        
        features['message_type'] = msg_type
        features['is_mt103'] = msg_type == '103'
        features['is_mt202'] = msg_type in ('202', '202COV')
        features['is_mt200'] = msg_type == '200'
        features['is_cover_payment'] = msg_type == '202COV'
        features['is_customer_payment'] = msg_type == '103'
        features['is_bank_transfer'] = msg_type in ('200', '202', '205')
        
        # Bank operation code
        boc = parsed.bank_operation_code
        features['bank_operation_code'] = boc
        features['is_credit_transfer'] = boc == 'CRED'
        features['is_direct_debit'] = boc == 'DRCT'
        
        # Source format
        features['source_code'] = parsed.source_code
        features['is_swift_source'] = parsed.source_code in ('SWF', 'SWIFT')
        features['is_fedwire_source'] = parsed.source_code in ('FED', 'FEDWIRE', 'FW')
        features['is_chips_source'] = parsed.source_code in ('CHP', 'CHIPS', 'CH')
        
        return features
    
    def _classify_payment_type(self, parsed: IFMLFeatures) -> Dict[str, Any]:
        """Classify payment as domestic or international with details."""
        features = {}
        
        # Get key party countries
        dbt = parsed.parties.get('DebitPartyInfo')
        cdt = parsed.parties.get('CreditPartyInfo')
        bnf = parsed.parties.get('BeneficiaryBankInfo')
        bnp = parsed.parties.get('BeneficiaryPartyInfo')
        
        # Debtor classification
        if dbt:
            dbt_country = dbt.bic_country or dbt.iban_country or dbt.country or dbt.address_country
            if dbt_country:
                features['debtor_country'] = dbt_country.upper()
                features['debtor_is_domestic_us'] = dbt_country.upper() == 'US'
                features['debtor_in_iban_zone'] = dbt_country.upper() in IBAN_REQUIRED_COUNTRIES
        
        # Creditor classification
        if cdt:
            cdt_country = cdt.bic_country or cdt.iban_country or cdt.country or cdt.address_country
            if cdt_country:
                features['creditor_country'] = cdt_country.upper()
                features['creditor_is_domestic_us'] = cdt_country.upper() == 'US'
                features['creditor_in_iban_zone'] = cdt_country.upper() in IBAN_REQUIRED_COUNTRIES
        
        # Beneficiary classification  
        benef_party = bnp or bnf
        if benef_party:
            benef_country = (benef_party.bic_country or benef_party.iban_country or 
                           benef_party.country or benef_party.address_country)
            if benef_country:
                features['beneficiary_final_country'] = benef_country.upper()
                features['beneficiary_is_domestic_us'] = benef_country.upper() == 'US'
                features['beneficiary_in_iban_zone'] = benef_country.upper() in IBAN_REQUIRED_COUNTRIES
                features['beneficiary_needs_iban'] = benef_country.upper() in IBAN_REQUIRED_COUNTRIES
                features['beneficiary_has_iban'] = benef_party.has_iban
                features['beneficiary_iban_missing'] = (benef_country.upper() in IBAN_REQUIRED_COUNTRIES and 
                                                        not benef_party.has_iban)
        
        return features
    
    def _check_consistency(self, parsed: IFMLFeatures) -> Dict[str, Any]:
        """Check for various inconsistencies that might trigger errors."""
        features = {}
        
        # Collect all IBANs and check consistency
        ibans = []
        iban_countries = set()
        
        for party_type, party in parsed.parties.items():
            if party and party.has_iban and party.account_value:
                iban = party.account_value.upper().replace(' ', '').replace('-', '')
                ibans.append(iban)
                if len(iban) >= 2:
                    iban_countries.add(iban[:2])
        
        features['total_iban_count'] = len(ibans)
        features['ibans_all_same_country'] = len(iban_countries) <= 1
        features['iban_countries_count'] = len(iban_countries)
        
        # Check for IBAN inconsistency (8023)
        if len(ibans) > 1:
            # Check if all IBANs have same first 8 chars (bank identifier)
            iban_banks = set(iban[:8] for iban in ibans if len(iban) >= 8)
            features['ibans_same_bank'] = len(iban_banks) <= 1
            features['ibans_consistent'] = len(set(ibans)) <= 1 or len(iban_banks) <= 1
        else:
            features['ibans_consistent'] = True
        
        # BIC consistency check (8028)
        bics = []
        for party_type, party in parsed.parties.items():
            if party and party.bic:
                bics.append(party.bic.upper())
        
        if len(bics) > 1:
            bic4s = set(bic[:4] for bic in bics if len(bic) >= 4)
            features['bic4s_consistent'] = len(bic4s) <= 1
        else:
            features['bic4s_consistent'] = True
        
        # Account number consistency (rare)
        accounts = []
        for party_type, party in parsed.parties.items():
            if party and party.account_value and not party.has_iban:
                accounts.append(party.account_value)
        
        features['domestic_account_count'] = len(accounts)
        
        return features


# =============================================================================
# INTEGRATION WITH DATA COLLECTOR
# =============================================================================

def extract_enhanced_features(ifml: Dict) -> Dict[str, Any]:
    """
    Convenience function to extract enhanced features from IFML.
    
    This is the main entry point for the data collector.
    """
    extractor = EnhancedFeatureExtractor()
    return extractor.extract(ifml)


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        with open(filepath) as f:
            data = json.load(f)
        
        extractor = EnhancedFeatureExtractor()
        features = extractor.extract(data)
        
        print("\n=== ENHANCED FEATURES ===\n")
        
        # Group by category for display
        categories = {
            'Party Counts': ['total_party_count', 'bank_party_count', 'non_bank_party_count',
                           'intermediary_count', 'has_multiple_intermediaries'],
            'Cross-Border': ['is_cross_border', 'is_domestic', 'originator_country', 
                           'beneficiary_country', 'is_us_domestic', 'is_sepa_payment'],
            'IDs': ['unique_bic_count', 'unique_iban_count', 'has_duplicate_bic',
                   'orig_benef_same_bic'],
            'Routing': ['uses_fedwire', 'uses_chips', 'uses_swift', 
                       'correspondent_chain_length'],
            'Message Type': ['message_type', 'is_mt103', 'is_mt202', 'is_cover_payment'],
            'Validation': ['ibans_consistent', 'bic4s_consistent'],
        }
        
        for cat, keys in categories.items():
            print(f"\n{cat}:")
            for k in keys:
                v = features.get(k)
                if v is not None:
                    print(f"  {k}: {v}")
        
        print(f"\n\nTotal features: {len(features)}")
    else:
        print("Usage: python enhanced_features.py <ifml_file.json>")
