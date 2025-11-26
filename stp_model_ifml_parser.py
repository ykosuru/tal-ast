"""
IFML Parser - Extract structured features from ACE Pelican IFML payment messages.
Handles nested JSON structure and normalizes fields for ML consumption.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class PartyInfo:
    """Normalized party information."""
    party_type: str
    has_id: bool = False
    id_value: Optional[str] = None
    id_type: Optional[str] = None
    has_bic: bool = False
    bic: Optional[str] = None
    bic_length: int = 0  # 4, 8, or 11 chars typically
    bic_country: Optional[str] = None  # Country code from BIC (chars 5-6)
    has_account: bool = False
    account_type: Optional[str] = None  # IBAN, BBAN, etc.
    account_value: Optional[str] = None
    iban_country: Optional[str] = None  # Country code from IBAN (first 2 chars)
    bic_iban_country_match: Optional[bool] = None  # Do BIC and IBAN countries match?
    country: Optional[str] = None
    mailing_country: Optional[str] = None
    residence_country: Optional[str] = None
    address_line_count: int = 0
    has_name: bool = False
    bank_flag: Optional[str] = None
    charge_flag: Optional[str] = None


@dataclass 
class MonetaryInfo:
    """Normalized monetary amount information."""
    amount_type: str
    currency: Optional[str] = None
    amount: float = 0.0


@dataclass
class IFMLFeatures:
    """Complete feature set extracted from an IFML request."""
    # Transaction identifiers
    transaction_id: Optional[str] = None
    transaction_uid: Optional[str] = None
    
    # Source/format info
    source_code: Optional[str] = None
    incoming_msg_type: Optional[str] = None
    incoming_format: Optional[str] = None
    bank_operation_code: Optional[str] = None
    
    # Location
    location_id_bank: Optional[str] = None
    
    # Monetary amounts
    amounts: List[MonetaryInfo] = field(default_factory=list)
    primary_currency: Optional[str] = None
    primary_amount: float = 0.0
    has_instructed_amount: bool = False
    instructed_amount: float = 0.0
    amount_mismatch: bool = False
    
    # Party information
    parties: Dict[str, PartyInfo] = field(default_factory=dict)
    
    # Derived cross-party features
    is_cross_border: bool = False
    has_intermediary: bool = False
    has_beneficiary_bank: bool = False
    originator_country: Optional[str] = None
    beneficiary_country: Optional[str] = None
    
    # Field presence flags
    has_bank_info: bool = False
    bank_info_count: int = 0
    
    # Raw data for debugging
    raw_json: Optional[dict] = None


class IFMLParser:
    """
    Parser for ACE Pelican IFML payment messages.
    Extracts structured features suitable for ML model training.
    """
    
    PARTY_TYPES = [
        'OriginatingPartyInfo',
        'SendingBankInfo', 
        'DebitPartyInfo',
        'CreditPartyInfo',
        'IntermediaryBankInfo',
        'BeneficiaryBankInfo',
        'AccountWithInstitution',
        'OrderingInstitution'
    ]
    
    def __init__(self, keep_raw: bool = False):
        self.keep_raw = keep_raw
    
    def parse_file(self, filepath: str) -> IFMLFeatures:
        """Parse IFML from a file path."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return self.parse(data)
    
    def parse_json_string(self, json_str: str) -> IFMLFeatures:
        """Parse IFML from a JSON string."""
        data = json.loads(json_str)
        return self.parse(data)
    
    def parse(self, data: dict) -> IFMLFeatures:
        """
        Parse IFML data structure and extract features.
        Handles both request and response formats.
        """
        features = IFMLFeatures()
        
        if self.keep_raw:
            features.raw_json = data
        
        # Navigate to BasicPayment - handle different wrapper structures
        basic_payment = self._find_basic_payment(data)
        
        if not basic_payment:
            return features
        
        # Extract transaction info
        features.transaction_id = basic_payment.get('TransactionID')
        features.transaction_uid = basic_payment.get('TransactionUID')
        
        # Extract source/format info
        features.source_code = basic_payment.get('SourceCode')
        features.incoming_msg_type = basic_payment.get('IncomingMsgType')
        features.incoming_format = basic_payment.get('IncomingFormat')
        features.bank_operation_code = basic_payment.get('BankOperationCode')
        
        # Extract location info
        location_info = basic_payment.get('LocationInfo', {})
        features.location_id_bank = location_info.get('IDBank')
        
        # Extract monetary amounts
        self._parse_monetary_amounts(basic_payment, features)
        
        # Extract party information
        self._parse_party_info(basic_payment, features)
        
        # Extract bank info
        self._parse_bank_info(basic_payment, features)
        
        # Derive cross-party features
        self._derive_cross_party_features(features)
        
        return features
    
    def _find_basic_payment(self, data: dict) -> Optional[dict]:
        """Navigate nested structure to find BasicPayment node."""
        # Try different wrapper patterns
        patterns = [
            # Pattern: {txn_id: {Request: {IFML: {File: {Message: {BasicPayment: ...}}}}}}
            lambda d: self._navigate_path(d, ['*', 'Request', 'IFML', 'File', 'Message', 'BasicPayment']),
            # Pattern: {Request: {IFML: {File: {Message: {BasicPayment: ...}}}}}
            lambda d: self._navigate_path(d, ['Request', 'IFML', 'File', 'Message', 'BasicPayment']),
            # Pattern: {IFML: {File: {Message: {BasicPayment: ...}}}}
            lambda d: self._navigate_path(d, ['IFML', 'File', 'Message', 'BasicPayment']),
            # Pattern: {BasicPayment: ...}
            lambda d: d.get('BasicPayment'),
        ]
        
        for pattern in patterns:
            result = pattern(data)
            if result:
                return result
        
        return None
    
    def _navigate_path(self, data: dict, path: List[str]) -> Optional[dict]:
        """Navigate a path through nested dicts, '*' matches any single key."""
        current = data
        
        for key in path:
            if not isinstance(current, dict):
                return None
            
            if key == '*':
                # Match any single key
                if len(current) == 1:
                    current = list(current.values())[0]
                else:
                    return None
            else:
                current = current.get(key)
                if current is None:
                    return None
        
        return current
    
    def _parse_monetary_amounts(self, basic_payment: dict, features: IFMLFeatures):
        """Extract and normalize monetary amount information."""
        amounts = basic_payment.get('MonetaryAmount', [])
        
        if isinstance(amounts, dict):
            amounts = [amounts]
        
        for amt in amounts:
            amt_type = amt.get('@Type', 'Unknown')
            currency = amt.get('@Currency') or amt.get('Currency')
            amount_val = self._parse_amount(amt.get('Amount', '0'))
            
            monetary_info = MonetaryInfo(
                amount_type=amt_type,
                currency=currency,
                amount=amount_val
            )
            features.amounts.append(monetary_info)
            
            # Track primary amount
            if amt_type == 'Amount':
                features.primary_currency = currency
                features.primary_amount = amount_val
            
            # Track instructed amount
            if amt_type == 'RelatedAmt' or amt.get('SubType') == 'INSTRUCTED AMOUNT':
                features.has_instructed_amount = True
                features.instructed_amount = amount_val
        
        # Check for amount mismatch
        if features.has_instructed_amount and features.primary_amount > 0:
            features.amount_mismatch = abs(features.primary_amount - features.instructed_amount) > 0.01
    
    def _parse_amount(self, amount_str: str) -> float:
        """Parse amount string to float."""
        try:
            return float(str(amount_str).replace(',', ''))
        except (ValueError, TypeError):
            return 0.0
    
    def _parse_party_info(self, basic_payment: dict, features: IFMLFeatures):
        """Extract party information for all party types."""
        party_info = basic_payment.get('PartyInfo', {})
        
        for party_type in self.PARTY_TYPES:
            party_data = party_info.get(party_type)
            if party_data:
                parsed_party = self._parse_single_party(party_type, party_data)
                features.parties[party_type] = parsed_party
    
    def _parse_single_party(self, party_type: str, party_data: dict) -> PartyInfo:
        """Parse a single party's information."""
        party = PartyInfo(party_type=party_type)
        
        # Handle nested structures - BasicPartyInfo, BasicIDInfo, AccountPartyInfo, BasicPartyBankInfo
        basic_info = (
            party_data.get('BasicPartyInfo') or 
            party_data.get('BasicIDInfo') or 
            party_data.get('AccountPartyInfo') or
            party_data.get('BasicPartyBankInfo') or
            party_data
        )
        
        # Extract ID
        id_field = basic_info.get('ID')
        if id_field:
            if isinstance(id_field, dict):
                party.has_id = True
                party.id_type = id_field.get('@Type')
                party.id_value = id_field.get('#text')
                if party.id_type == 'BIC':
                    party.has_bic = True
                    party.bic = party.id_value
            else:
                party.has_id = True
                party.id_value = str(id_field)
        
        # Check for BIC in ID type (only if ID is a dict)
        id_field_check = basic_info.get('ID')
        if isinstance(id_field_check, dict) and id_field_check.get('@Type') == 'BIC':
            party.has_bic = True
            party.bic = id_field_check.get('#text')
        
        # Extract account info
        acct_info = basic_info.get('AcctIDInfo')
        if acct_info:
            party.has_account = True
            if isinstance(acct_info, dict):
                acct_id = acct_info.get('ID', {})
                if isinstance(acct_id, dict):
                    party.account_type = acct_id.get('@Type')
                    party.account_value = acct_id.get('#text')
        
        # Check AdrBankID for additional account info
        adr_bank_id = basic_info.get('AdrBankID')
        if adr_bank_id:
            party.has_account = True
            if isinstance(adr_bank_id, dict):
                party.account_type = party.account_type or adr_bank_id.get('@Type')
        
        # Extract country codes
        party.country = (
            basic_info.get('Country') or 
            party_data.get('Country')
        )
        party.mailing_country = basic_info.get('MailingCountry')
        party.residence_country = basic_info.get('ResidenceCountry')
        
        # Extract address info
        address_info = basic_info.get('AddressInfo') or party_data.get('AddressInfo')
        if address_info:
            if isinstance(address_info, list):
                party.address_line_count = len(address_info)
                party.has_name = len(address_info) > 0
            elif isinstance(address_info, dict):
                party.address_line_count = 1
                party.has_name = True
        
        # Extract flags
        party.bank_flag = basic_info.get('BankFlag')
        party.charge_flag = basic_info.get('ChargeFlag')
        
        # Parse BIC structure
        if party.bic:
            party.bic_length = len(party.bic)
            # BIC country is characters 5-6 (0-indexed: 4-5)
            if len(party.bic) >= 6:
                party.bic_country = party.bic[4:6].upper()
        
        # Parse IBAN country
        if party.account_type == 'IBAN' and party.account_value:
            # IBAN country is first 2 characters
            if len(party.account_value) >= 2:
                party.iban_country = party.account_value[0:2].upper()
        
        # Check if BIC and IBAN countries match
        if party.bic_country and party.iban_country:
            party.bic_iban_country_match = (party.bic_country == party.iban_country)
        
        return party
    
    def _parse_bank_info(self, basic_payment: dict, features: IFMLFeatures):
        """Extract BankInfo field details."""
        bank_info = basic_payment.get('BankInfo', [])
        
        if isinstance(bank_info, dict):
            bank_info = [bank_info]
        
        features.has_bank_info = len(bank_info) > 0
        features.bank_info_count = len(bank_info)
    
    def _derive_cross_party_features(self, features: IFMLFeatures):
        """Derive features that span multiple parties."""
        # Get originator country
        orig_party = features.parties.get('OriginatingPartyInfo')
        if orig_party:
            features.originator_country = orig_party.country
        
        # Get beneficiary country (from CreditParty or BeneficiaryBank)
        credit_party = features.parties.get('CreditPartyInfo')
        benef_bank = features.parties.get('BeneficiaryBankInfo')
        
        if credit_party:
            features.beneficiary_country = (
                credit_party.country or 
                credit_party.mailing_country
            )
        if not features.beneficiary_country and benef_bank:
            features.beneficiary_country = benef_bank.country
        
        # Determine if cross-border
        if features.originator_country and features.beneficiary_country:
            features.is_cross_border = (
                features.originator_country != features.beneficiary_country
            )
        
        # Check for intermediary
        features.has_intermediary = 'IntermediaryBankInfo' in features.parties
        features.has_beneficiary_bank = 'BeneficiaryBankInfo' in features.parties
    
    def to_dict(self, features: IFMLFeatures) -> dict:
        """Convert features to a flat dictionary for ML consumption."""
        result = {
            'transaction_id': features.transaction_id,
            'source_code': features.source_code,
            'incoming_msg_type': features.incoming_msg_type,
            'incoming_format': features.incoming_format,
            'bank_operation_code': features.bank_operation_code,
            'location_id_bank': features.location_id_bank,
            'primary_currency': features.primary_currency,
            'primary_amount': features.primary_amount,
            'has_instructed_amount': features.has_instructed_amount,
            'instructed_amount': features.instructed_amount,
            'amount_mismatch': features.amount_mismatch,
            'is_cross_border': features.is_cross_border,
            'has_intermediary': features.has_intermediary,
            'has_beneficiary_bank': features.has_beneficiary_bank,
            'originator_country': features.originator_country,
            'beneficiary_country': features.beneficiary_country,
            'has_bank_info': features.has_bank_info,
            'bank_info_count': features.bank_info_count,
            'amount_count': len(features.amounts),
        }
        
        # Add party-specific features
        for party_type in self.PARTY_TYPES:
            prefix = self._party_type_to_prefix(party_type)
            party = features.parties.get(party_type)
            
            if party:
                result[f'{prefix}_present'] = True
                result[f'{prefix}_has_id'] = party.has_id
                result[f'{prefix}_has_bic'] = party.has_bic
                result[f'{prefix}_bic_length'] = party.bic_length
                result[f'{prefix}_bic_country'] = party.bic_country
                result[f'{prefix}_has_account'] = party.has_account
                result[f'{prefix}_account_type'] = party.account_type
                result[f'{prefix}_iban_country'] = party.iban_country
                result[f'{prefix}_bic_iban_match'] = party.bic_iban_country_match
                result[f'{prefix}_country'] = party.country
                result[f'{prefix}_mailing_country'] = party.mailing_country
                result[f'{prefix}_address_lines'] = party.address_line_count
                result[f'{prefix}_has_name'] = party.has_name
            else:
                result[f'{prefix}_present'] = False
                result[f'{prefix}_has_id'] = False
                result[f'{prefix}_has_bic'] = False
                result[f'{prefix}_bic_length'] = 0
                result[f'{prefix}_bic_country'] = None
                result[f'{prefix}_has_account'] = False
                result[f'{prefix}_account_type'] = None
                result[f'{prefix}_iban_country'] = None
                result[f'{prefix}_bic_iban_match'] = None
                result[f'{prefix}_country'] = None
                result[f'{prefix}_mailing_country'] = None
                result[f'{prefix}_address_lines'] = 0
                result[f'{prefix}_has_name'] = False
        
        return result
    
    def _party_type_to_prefix(self, party_type: str) -> str:
        """Convert party type to short prefix for feature names."""
        prefixes = {
            'OriginatingPartyInfo': 'orig',
            'SendingBankInfo': 'send',
            'DebitPartyInfo': 'dbt',
            'CreditPartyInfo': 'cdt',
            'IntermediaryBankInfo': 'intm',
            'BeneficiaryBankInfo': 'bnf',
            'AccountWithInstitution': 'acwi',
            'OrderingInstitution': 'ordi'
        }
        return prefixes.get(party_type, party_type.lower()[:4])


class IFMLResponseParser:
    """Parser for ACE Pelican IFML response messages with error codes."""
    
    def parse_file(self, filepath: str) -> Tuple[str, List[dict]]:
        """Parse response file and extract transaction ID and message codes."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return self.parse(data)
    
    def parse(self, data: dict) -> Tuple[Optional[str], List[dict]]:
        """
        Parse response and extract error codes.
        Returns (transaction_uid, list of code dictionaries).
        """
        # Navigate to Message
        message = self._find_message(data)
        
        if not message:
            return None, []
        
        # Extract transaction UID
        basic_payment = message.get('BasicPayment', {})
        transaction_uid = (
            basic_payment.get('TransactionUID') or 
            message.get('TransactionUID')
        )
        
        # Extract audit trail codes
        audit_trail = message.get('AuditTrail') or {}
        
        # Handle case where AuditTrail is at File level
        if not audit_trail:
            file_node = self._find_file(data)
            if file_node:
                audit_trail = file_node.get('AuditTrail', {})
        
        msg_status = audit_trail.get('MsgStatus', [])
        
        if isinstance(msg_status, dict):
            msg_status = [msg_status]
        
        codes = []
        for status in msg_status:
            info_data = status.get('InformationalData', '')
            party_short, party_full = self._extract_party_hint(info_data)
            codes.append({
                'code': status.get('Code'),
                'severity': status.get('Severity'),
                'info': info_data,
                'party_hint': party_short,  # Short code like CDTPTY, BNFBNK
                'party_type': party_full,   # Full type like CreditPartyInfo
            })
        
        return transaction_uid, codes
    
    def _find_message(self, data: dict) -> Optional[dict]:
        """Find Message node in response structure."""
        patterns = [
            lambda d: d.get('Response', {}).get('IFML', {}).get('File', {}).get('Message'),
            lambda d: d.get('IFML', {}).get('File', {}).get('Message'),
            lambda d: d.get('File', {}).get('Message'),
            lambda d: d.get('Message'),
        ]
        
        for pattern in patterns:
            result = pattern(data)
            if result:
                return result
        
        return None
    
    def _find_file(self, data: dict) -> Optional[dict]:
        """Find File node in response structure."""
        patterns = [
            lambda d: d.get('Response', {}).get('IFML', {}).get('File'),
            lambda d: d.get('IFML', {}).get('File'),
            lambda d: d.get('File'),
        ]
        
        for pattern in patterns:
            result = pattern(data)
            if result:
                return result
        
        return None
    
    def _extract_party_hint(self, info: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract party identifier from informational data.
        Returns (short_code, full_party_type) tuple.
        """
        # Common patterns: "BNFBNK ...", "CDTPTY ...", "BNPPTY ..."
        party_patterns = {
            'BNFBNK': 'BeneficiaryBankInfo',
            'BNFPTY': 'BeneficiaryPartyInfo',
            'BNPPTY': 'BeneficiaryPartyInfo',
            'CDTPTY': 'CreditPartyInfo',
            'DBTPTY': 'DebitPartyInfo',
            'ORGPTY': 'OriginatingPartyInfo',
            'INTBNK': 'IntermediaryBankInfo',
            'SNDBNK': 'SendingBankInfo',
            'ACWBNK': 'AccountWithInstitution',
            'ORDBNK': 'OrderingInstitution',
        }
        
        for pattern, party_type in party_patterns.items():
            if info.startswith(pattern):
                return pattern, party_type
        
        return None, None


if __name__ == '__main__':
    # Test with sample data
    import sys
    
    parser = IFMLParser(keep_raw=True)
    resp_parser = IFMLResponseParser()
    
    # Test parsing
    sample_request = {
        "2025092900000192": {
            "Request": {
                "IFML": {
                    "File": {
                        "Message": {
                            "BasicPayment": {
                                "TransactionID": "2025092900000XXX",
                                "SourceCode": "SWF",
                                "IncomingMsgType": "103",
                                "BankOperationCode": "CRED",
                                "PartyInfo": {
                                    "CreditPartyInfo": {
                                        "BasicPartyInfo": {
                                            "MailingCountry": "JP"
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    features = parser.parse(sample_request)
    print("Parsed features:")
    feature_dict = parser.to_dict(features)
    for k, v in feature_dict.items():
        if v is not None and v != False and v != 0:
            print(f"  {k}: {v}")
