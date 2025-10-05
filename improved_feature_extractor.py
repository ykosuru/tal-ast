#!/usr/bin/env python3
"""
Improved feature extraction that focuses on what's missing or incorrect
"""

import json
import numpy as np
from typing import Dict, List, Optional, Any
from collections import defaultdict

class ImprovedFeatureExtractor:
    """Feature extractor that identifies missing/incorrect elements"""
    
    def __init__(self):
        self.feature_names = []
        self._build_feature_names()
    
    def _build_feature_names(self):
        """Define features that indicate need for repairs"""
        
        # Missing critical fields (these often trigger repairs)
        self.feature_names.extend([
            # Agent completeness checks
            'cdtragt_missing_bic',
            'cdtragt_missing_name', 
            'cdtragt_missing_address',
            'cdtragt_missing_country',
            'dbtragt_missing_bic',
            'dbtragt_missing_name',
            'dbtragt_missing_address', 
            'dbtragt_missing_country',
            'instgagt_missing_bic',
            'instgagt_missing_name',
            'instdagt_missing_bic',
            'instdagt_missing_name',
            
            # Structural issues
            'has_clearing_no_bic',  # Common trigger for BIC lookup
            'has_bic_no_name',      # Triggers name lookup
            'has_partial_address',   # Triggers address enrichment
            'missing_country_code',  # Triggers country extraction
            
            # Format issues  
            'mmbid_at_wrong_level',  # MmbId not in ClrSysId structure
            'missing_clrsysid_structure',
            'missing_othr_structure',
            
            # Party completeness
            'cdtr_incomplete',
            'dbtr_incomplete',
            
            # Source/Clearing combinations
            'is_swf_fed',
            'is_swf_chips', 
            'is_ach_fed',
            'is_ach_chips',
            
            # Specific field presence (positive indicators)
            'has_ustrd_remittance',
            'has_struct_remittance',
            'has_account_type_code',
            
            # Cross-entity consistency
            'agent_country_mismatch',
            'missing_required_parties',
        ])
    
    def extract_for_prediction(self, payment: Dict) -> np.ndarray:
        """Extract features focused on what needs repair"""
        features = np.zeros(len(self.feature_names))
        
        # Normalize payment structure
        payment = self._normalize_keys(payment)
        
        idx = 0
        
        # Check each agent for completeness
        for agent in ['cdtragt', 'dbtragt']:
            agent_data = payment.get(agent, {})
            if isinstance(agent_data, dict):
                # For prediction, check the CURRENT state (not 'before')
                if 'before' in agent_data:
                    agent_data = agent_data['before']
                
                features[idx] = float(not self._has_bic(agent_data))  # missing_bic
                features[idx+1] = float(not self._has_name(agent_data))  # missing_name
                features[idx+2] = float(not self._has_full_address(agent_data))  # missing_address
                features[idx+3] = float(not self._has_country(agent_data))  # missing_country
            else:
                # Agent doesn't exist at all
                features[idx:idx+4] = 1.0  # All missing
            idx += 4
        
        # Instructing/Instructed agents (often less complete)
        for agent in ['instgagt', 'instdagt']:
            agent_data = payment.get(agent, {})
            if isinstance(agent_data, dict):
                if 'before' in agent_data:
                    agent_data = agent_data['before']
                features[idx] = float(not self._has_bic(agent_data))
                features[idx+1] = float(not self._has_name(agent_data))
            else:
                features[idx:idx+2] = 1.0
            idx += 2
        
        # Structural issues that trigger repairs
        features[idx] = float(self._has_clearing_no_bic(payment)); idx += 1
        features[idx] = float(self._has_bic_no_name(payment)); idx += 1
        features[idx] = float(self._has_partial_address(payment)); idx += 1
        features[idx] = float(self._missing_country_code(payment)); idx += 1
        
        # Format/structure issues
        features[idx] = float(self._mmbid_at_wrong_level(payment)); idx += 1
        features[idx] = float(self._missing_clrsysid_structure(payment)); idx += 1
        features[idx] = float(self._missing_othr_structure(payment)); idx += 1
        
        # Party completeness
        features[idx] = float(self._party_incomplete(payment.get('cdtr', {}))); idx += 1
        features[idx] = float(self._party_incomplete(payment.get('dbtr', {}))); idx += 1
        
        # Source/Clearing - strong indicators
        source = payment.get('source', '').upper()
        clearing = payment.get('clearing', '').upper()
        features[idx] = float(source == 'SWF' and clearing == 'FED'); idx += 1
        features[idx] = float(source == 'SWF' and clearing == 'CHIPS'); idx += 1
        features[idx] = float(source == 'ACH' and clearing == 'FED'); idx += 1
        features[idx] = float(source == 'ACH' and clearing == 'CHIPS'); idx += 1
        
        # Remittance info type
        rmtinf = payment.get('rmtinf', {})
        if isinstance(rmtinf, dict):
            if 'before' in rmtinf:
                rmtinf = rmtinf['before']
            features[idx] = float('ustrd' in str(rmtinf).lower()); idx += 1
            features[idx] = float('strd' in str(rmtinf).lower()); idx += 1
        else:
            idx += 2
        
        # Account type
        features[idx] = float(self._has_account_type(payment)); idx += 1
        
        # Cross-entity checks
        features[idx] = float(self._has_agent_country_mismatch(payment)); idx += 1
        features[idx] = float(self._missing_required_parties(payment)); idx += 1
        
        return features
    
    def extract_for_training(self, payment: Dict, has_repair: Dict[str, bool]) -> np.ndarray:
        """Extract features during training when we know the repairs"""
        # Same features but extracted from 'before' state
        before_state = self._extract_before_state(payment)
        return self.extract_for_prediction(before_state)
    
    def _extract_before_state(self, payment: Dict) -> Dict:
        """Extract the 'before' state from training data"""
        before_state = {
            'source': payment.get('source'),
            'clearing': payment.get('clearing'),
            'flags': payment.get('flags', {})
        }
        
        entities = ['cdtrAgt', 'dbtrAgt', 'instgAgt', 'instdAgt',
                   'cdtr', 'dbtr', 'cdtrAcct', 'dbtrAcct', 'rmtInf']
        
        for entity in entities:
            if entity in payment:
                entity_data = payment[entity]
                if isinstance(entity_data, dict) and 'before' in entity_data:
                    before_state[entity] = entity_data['before']
                elif isinstance(entity_data, dict) and 'after' not in entity_data:
                    # No after means this is the original state
                    before_state[entity] = entity_data
        
        return before_state
    
    def _normalize_keys(self, obj):
        """Normalize all keys to lowercase"""
        if isinstance(obj, dict):
            return {k.lower() if isinstance(k, str) else k: self._normalize_keys(v)
                   for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._normalize_keys(item) for item in obj]
        return obj
    
    def _has_bic(self, agent_data: Dict) -> bool:
        """Check if agent has BIC"""
        return self._find_field(agent_data, ['bicfi', 'bic']) is not None
    
    def _has_name(self, agent_data: Dict) -> bool:
        """Check if agent has name"""
        return self._find_field(agent_data, ['nm', 'name']) is not None
    
    def _has_full_address(self, agent_data: Dict) -> bool:
        """Check if agent has complete address"""
        adr_lines = self._find_field(agent_data, ['adrline'])
        if isinstance(adr_lines, list) and len(adr_lines) >= 2:
            return True
        return False
    
    def _has_country(self, agent_data: Dict) -> bool:
        """Check if agent has country"""
        return self._find_field(agent_data, ['ctryofres', 'ctry']) is not None
    
    def _has_clearing_no_bic(self, payment: Dict) -> bool:
        """Check if payment has clearing number but no BIC (triggers BIC lookup)"""
        for entity in ['cdtragt', 'dbtragt']:
            if entity in payment:
                agent = payment[entity]
                if isinstance(agent, dict):
                    if 'before' in agent:
                        agent = agent['before']
                    has_clearing = self._find_field(agent, ['mmbid', 'clrsysmmbid'])
                    has_bic = self._has_bic(agent)
                    if has_clearing and not has_bic:
                        return True
        return False
    
    def _has_bic_no_name(self, payment: Dict) -> bool:
        """Check if payment has BIC but no name (triggers name lookup)"""
        for entity in ['cdtragt', 'dbtragt']:
            if entity in payment:
                agent = payment[entity]
                if isinstance(agent, dict):
                    if 'before' in agent:
                        agent = agent['before']
                    if self._has_bic(agent) and not self._has_name(agent):
                        return True
        return False
    
    def _has_partial_address(self, payment: Dict) -> bool:
        """Check if payment has partial address info"""
        for entity in ['cdtragt', 'dbtragt']:
            if entity in payment:
                agent = payment[entity]
                if isinstance(agent, dict):
                    if 'before' in agent:
                        agent = agent['before']
                    has_town = self._find_field(agent, ['townname'])
                    has_full = self._has_full_address(agent)
                    if has_town and not has_full:
                        return True
        return False
    
    def _missing_country_code(self, payment: Dict) -> bool:
        """Check if country code is missing but could be extracted"""
        for entity in ['cdtr', 'dbtr', 'cdtragt', 'dbtragt']:
            if entity in payment:
                party = payment[entity]
                if isinstance(party, dict):
                    if 'before' in party:
                        party = party['before']
                    has_address = self._find_field(party, ['adrline'])
                    has_country = self._has_country(party)
                    if has_address and not has_country:
                        # Check if address contains country indicator
                        addr_str = str(has_address).upper()
                        if any(country in addr_str for country in ['US', 'USA', 'UNITED STATES', 'UK', 'GB']):
                            return True
        return False
    
    def _mmbid_at_wrong_level(self, payment: Dict) -> bool:
        """Check if MmbId is at wrong structural level"""
        for entity in ['cdtragt', 'dbtragt']:
            if entity in payment:
                agent = payment[entity]
                if isinstance(agent, dict):
                    if 'before' in agent:
                        agent = agent['before']
                    # Check if MmbId exists at top level instead of inside ClrSysId
                    if self._find_field(agent, ['mmbid'], max_depth=1):
                        # Should be inside clrsysid
                        if not self._find_field(agent, ['clrsysid']):
                            return True
        return False
    
    def _missing_clrsysid_structure(self, payment: Dict) -> bool:
        """Check if ClrSysId structure is missing when needed"""
        for entity in ['cdtragt', 'dbtragt']:
            if entity in payment:
                agent = payment[entity]
                if isinstance(agent, dict):
                    if 'before' in agent:
                        agent = agent['before']
                    has_clearing = self._find_field(agent, ['mmbid'])
                    has_clrsys = self._find_field(agent, ['clrsysid'])
                    if has_clearing and not has_clrsys:
                        return True
        return False
    
    def _missing_othr_structure(self, payment: Dict) -> bool:
        """Check if Othr structure is missing when needed"""
        for entity in ['cdtragt', 'dbtragt']:
            if entity in payment:
                agent = payment[entity]
                if isinstance(agent, dict):
                    if 'before' in agent:
                        agent = agent['before']
                    # Has some ID but no BIC and no Othr structure
                    has_id = self._find_field(agent, ['id'])
                    has_bic = self._has_bic(agent)
                    has_othr = self._find_field(agent, ['othr'])
                    if has_id and not has_bic and not has_othr:
                        return True
        return False
    
    def _party_incomplete(self, party: Dict) -> bool:
        """Check if party (cdtr/dbtr) is incomplete"""
        if not party:
            return False
        if isinstance(party, dict):
            if 'before' in party:
                party = party['before']
            has_name = self._find_field(party, ['nm', 'name'])
            has_address = self._find_field(party, ['adrline', 'pstladr'])
            return not (has_name and has_address)
        return False
    
    def _has_account_type(self, payment: Dict) -> bool:
        """Check if account type is present"""
        for acct in ['cdtracct', 'dbtracct']:
            if acct in payment:
                account = payment[acct]
                if isinstance(account, dict):
                    if 'before' in account:
                        account = account['before']
                    if self._find_field(account, ['tp', 'prtry']):
                        return True
        return False
    
    def _has_agent_country_mismatch(self, payment: Dict) -> bool:
        """Check if agents have mismatched country info"""
        countries = set()
        for entity in ['cdtragt', 'dbtragt']:
            if entity in payment:
                agent = payment[entity]
                if isinstance(agent, dict):
                    if 'before' in agent:
                        agent = agent['before']
                    country = self._find_field(agent, ['ctryofres', 'ctry'])
                    if country:
                        countries.add(str(country).upper())
        return len(countries) > 1
    
    def _missing_required_parties(self, payment: Dict) -> bool:
        """Check if required parties are missing"""
        # For FED clearing, certain parties are required
        clearing = payment.get('clearing', '').upper()
        if clearing == 'FED':
            required = ['cdtragt', 'dbtragt']
            for entity in required:
                if entity not in payment or not payment[entity]:
                    return True
        return False
    
    def _find_field(self, obj: Dict, field_names: List[str], max_depth: int = 5) -> Optional[Any]:
        """Find field by multiple possible names"""
        if not isinstance(obj, dict) or max_depth <= 0:
            return None
        
        for field_name in field_names:
            field_lower = field_name.lower()
            for key, value in obj.items():
                if field_lower in str(key).lower():
                    return value
        
        # Recursive search
        for value in obj.values():
            if isinstance(value, dict):
                result = self._find_field(value, field_names, max_depth - 1)
                if result is not None:
                    return result
        
        return None
    
    def get_feature_names(self) -> List[str]:
        """Get feature names for interpretation"""
        return self.feature_names
