#!/usr/bin/env python3
"""
Payment Repair Lookup Table Builder
Extracts repairs from payment data and creates a searchable lookup table
Version 1
"""

import json
import pickle
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import argparse
import re


@dataclass
class RepairInfo:
    """Information about a specific repair"""
    code: str
    field: str
    text: str
    entity: str
    repair_id: str
    
    def matches_error(self, error_message: str, field_name: str) -> float:
        """
        Calculate match score between this repair and a validation error
        Returns 0.0-1.0, higher is better match
        """
        score = 0.0
        error_lower = error_message.lower()
        field_lower = field_name.lower()
        repair_field_lower = self.field.lower()
        repair_text_lower = self.text.lower()
        
        # Exact field match
        if repair_field_lower in field_lower or field_lower in repair_field_lower:
            score += 0.4
        
        # Field mentioned in repair text
        if field_lower in repair_text_lower:
            score += 0.2
        
        # Error type matching
        error_keywords = {
            'missing': ['extracted', 'looked up', 'added'],
            'required': ['extracted', 'looked up', 'added'],
            'country': ['country', 'ctryofres', 'cdtpty'],
            'bic': ['bic', 'bicfi'],
            'name': ['name', 'nm'],
            'address': ['address', 'pstladr'],
            'routing': ['routing', 'clrsysmmbid'],
            'iban': ['iban'],
            'account': ['account', 'acct'],
        }
        
        for error_type, keywords in error_keywords.items():
            if error_type in error_lower:
                for keyword in keywords:
                    if keyword in repair_text_lower or keyword in repair_field_lower:
                        score += 0.3
                        break
        
        # Generic relevance
        words_in_common = set(error_lower.split()) & set(repair_text_lower.split())
        if len(words_in_common) > 2:
            score += 0.1
        
        return min(score, 1.0)


class RepairLookupBuilder:
    """Builds and manages repair lookup table"""
    
    def __init__(self):
        # Primary lookup: entity -> field -> [RepairInfo]
        self.repairs_by_entity_field: Dict[str, Dict[str, List[RepairInfo]]] = defaultdict(lambda: defaultdict(list))
        
        # Secondary lookup: repair_id -> RepairInfo
        self.repairs_by_id: Dict[str, RepairInfo] = {}
        
        # All repairs list
        self.all_repairs: List[RepairInfo] = []
        
        # Entity mappings
        self.entities = [
            'cdtr', 'dbtr', 'cdtrAgt', 'dbtrAgt', 'cdtrAcct', 'dbtrAcct',
            'instgAgt', 'instdAgt', 'intrmyAgt1', 'intrmyAgt2', 'intrmyAgt3',
            'rmtInf'
        ]
        
        # Statistics
        self.stats = {
            'files_processed': 0,
            'payments_processed': 0,
            'repairs_found': 0,
            'unique_repair_types': set()
        }
    
    def normalize_entity_name(self, name: str) -> str:
        """Normalize entity name to standard format"""
        # Convert to camelCase
        name_lower = name.lower()
        for entity in self.entities:
            if name_lower == entity.lower():
                return entity
        return name_lower
    
    def normalize_field_name(self, name: str) -> str:
        """Normalize field name to uppercase"""
        # Remove dots and convert to uppercase
        return name.upper().replace('.', '')
    
    def extract_repairs_from_payment(self, payment: Dict, txn_id: str = None):
        """Extract repairs from a single payment transaction"""
        
        # Look for repairs in the payment structure
        # Common patterns: 'repairs', 'Repairs', nested in entities
        
        for entity_name in self.entities:
            # Try to find entity (case-insensitive)
            entity_data = self._find_entity(payment, entity_name)
            
            if not entity_data:
                continue
            
            # Check if entity has before/after structure with repairs
            if isinstance(entity_data, dict):
                # Look for repairs at entity level
                repairs = self._extract_repairs_from_dict(entity_data, entity_name, txn_id)
                
                # Check before/after structure
                if 'before' in entity_data and 'after' in entity_data:
                    # Compare before and after to infer repairs
                    self._infer_repairs_from_changes(
                        entity_data.get('before', {}),
                        entity_data.get('after', {}),
                        entity_name,
                        txn_id
                    )
                
                # Check for explicit repairs field
                if 'repairs' in entity_data or 'Repairs' in entity_data:
                    repair_data = entity_data.get('repairs') or entity_data.get('Repairs')
                    self._process_repairs_field(repair_data, entity_name, txn_id)
        
        # Also check for global repairs field
        if 'repairs' in payment or 'Repairs' in payment:
            global_repairs = payment.get('repairs') or payment.get('Repairs')
            if isinstance(global_repairs, dict):
                for entity_name, repairs in global_repairs.items():
                    normalized_entity = self.normalize_entity_name(entity_name)
                    self._process_repairs_field(repairs, normalized_entity, txn_id)
    
    def _find_entity(self, payment: Dict, entity_name: str) -> Any:
        """Find entity in payment with case-insensitive search"""
        # Try exact match
        if entity_name in payment:
            return payment[entity_name]
        
        # Try capitalized
        capitalized = entity_name[0].upper() + entity_name[1:]
        if capitalized in payment:
            return payment[capitalized]
        
        # Try lowercase
        if entity_name.lower() in payment:
            return payment[entity_name.lower()]
        
        return None
    
    def _extract_repairs_from_dict(self, data: Dict, entity_name: str, txn_id: str) -> List[RepairInfo]:
        """Extract repairs from a dictionary"""
        repairs = []
        
        # Look for repair-like structures
        for key, value in data.items():
            if 'repair' in key.lower():
                if isinstance(value, dict):
                    repairs.extend(self._process_repairs_field(value, entity_name, txn_id))
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            repairs.extend(self._process_repairs_field(item, entity_name, txn_id))
        
        return repairs
    
    def _process_repairs_field(self, repairs_data: Any, entity_name: str, txn_id: str) -> List[RepairInfo]:
        """Process repairs field and extract RepairInfo objects"""
        repairs = []
        
        if isinstance(repairs_data, dict):
            # Format: { 'repair_id': { 'code': 'I', 'field': 'BICFI', 'text': '...' } }
            for repair_id, repair_details in repairs_data.items():
                if isinstance(repair_details, dict):
                    repair_info = self._create_repair_info(
                        repair_id, repair_details, entity_name, txn_id
                    )
                    if repair_info:
                        repairs.append(repair_info)
        
        elif isinstance(repairs_data, list):
            # Format: [ { 'id': '6021', 'code': 'I', 'field': 'BICFI', 'text': '...' } ]
            for i, repair_item in enumerate(repairs_data):
                if isinstance(repair_item, dict):
                    repair_id = repair_item.get('id', f"repair_{i}")
                    repair_info = self._create_repair_info(
                        repair_id, repair_item, entity_name, txn_id
                    )
                    if repair_info:
                        repairs.append(repair_info)
        
        return repairs
    
    def _create_repair_info(self, repair_id: str, repair_data: Dict, entity_name: str, txn_id: str) -> Optional[RepairInfo]:
        """Create RepairInfo from repair data"""
        code = repair_data.get('code', repair_data.get('Code', ''))
        field = repair_data.get('field', repair_data.get('Field', ''))
        text = repair_data.get('text', repair_data.get('Text', ''))
        
        if not field or not text:
            return None
        
        normalized_entity = self.normalize_entity_name(entity_name)
        normalized_field = self.normalize_field_name(field)
        
        repair_info = RepairInfo(
            code=code,
            field=normalized_field,
            text=text,
            entity=normalized_entity,
            repair_id=repair_id
        )
        
        # Add to lookups
        self.repairs_by_entity_field[normalized_entity][normalized_field].append(repair_info)
        self.repairs_by_id[repair_id] = repair_info
        self.all_repairs.append(repair_info)
        
        # Update stats
        self.stats['repairs_found'] += 1
        self.stats['unique_repair_types'].add(f"{normalized_entity}.{normalized_field}")
        
        return repair_info
    
    def _infer_repairs_from_changes(self, before: Dict, after: Dict, entity_name: str, txn_id: str):
        """Infer repairs by comparing before and after states"""
        if not isinstance(before, dict) or not isinstance(after, dict):
            return
        
        # Find fields added or changed
        for key, after_value in after.items():
            before_value = before.get(key)
            
            # Field was added
            if before_value is None or before_value == '':
                if after_value and after_value != '':
                    # Infer repair type based on field name
                    repair_text = self._infer_repair_text(key, after_value)
                    
                    repair_info = RepairInfo(
                        code='I',  # Inferred
                        field=self.normalize_field_name(key),
                        text=repair_text,
                        entity=self.normalize_entity_name(entity_name),
                        repair_id=f"inferred_{entity_name}_{key}"
                    )
                    
                    # Add to lookups
                    normalized_entity = self.normalize_entity_name(entity_name)
                    normalized_field = self.normalize_field_name(key)
                    
                    # Only add if not already present
                    existing = self.repairs_by_entity_field[normalized_entity][normalized_field]
                    if not any(r.text == repair_text for r in existing):
                        self.repairs_by_entity_field[normalized_entity][normalized_field].append(repair_info)
                        self.all_repairs.append(repair_info)
                        self.stats['repairs_found'] += 1
            
            # Recursively check nested objects
            elif isinstance(after_value, dict) and isinstance(before_value, dict):
                self._infer_repairs_from_changes(before_value, after_value, entity_name, txn_id)
    
    def _infer_repair_text(self, field_name: str, value: Any) -> str:
        """Infer repair description based on field name"""
        field_upper = field_name.upper()
        
        repair_patterns = {
            'BICFI': 'BIC looked up from Clearing System ID',
            'BIC': 'BIC looked up from Clearing System ID',
            'NM': 'Name looked up from BIC',
            'NAME': 'Name looked up from BIC',
            'PSTLADR': 'Address looked up from BIC',
            'ADDRESS': 'Address looked up from BIC',
            'CTRYOFRES': 'Country Code extracted from address',
            'COUNTRY': 'Country Code extracted from address',
            'CDTPTY': 'ISO Country Code extracted from field',
            'IBAN': 'IBAN validated and formatted',
            'CLRSYSMMBID': 'Clearing System Member ID extracted',
            'MMBID': 'Member ID extracted from clearing system',
        }
        
        for pattern, text in repair_patterns.items():
            if pattern in field_upper:
                return text
        
        return f"{field_name} field added or corrected"
    
    def analyze_directory(self, data_dir: str):
        """Analyze all JSON files in directory"""
        json_files = list(Path(data_dir).glob('**/*.json'))
        print(f"\n{'='*80}")
        print(f"BUILDING REPAIR LOOKUP TABLE FROM {len(json_files)} JSON FILES")
        print(f"{'='*80}\n")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                self.stats['files_processed'] += 1
                
                # Handle different formats
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            for txn_id, txn_data in item.items():
                                self.extract_repairs_from_payment(txn_data, txn_id)
                                self.stats['payments_processed'] += 1
                
                elif isinstance(data, dict):
                    # Check if it's a single payment or collection
                    if any(key in data for key in self.entities):
                        # Single payment
                        self.extract_repairs_from_payment(data)
                        self.stats['payments_processed'] += 1
                    else:
                        # Collection of payments
                        for txn_id, txn_data in data.items():
                            if isinstance(txn_data, dict):
                                self.extract_repairs_from_payment(txn_data, txn_id)
                                self.stats['payments_processed'] += 1
                
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
        
        print(f"Files processed: {self.stats['files_processed']}")
        print(f"Payments processed: {self.stats['payments_processed']}")
        print(f"Repairs found: {self.stats['repairs_found']}")
        print(f"Unique repair types: {len(self.stats['unique_repair_types'])}")
    
    def save_lookup_table(self, output_path: str):
        """Save lookup table to pickle file"""
        lookup_data = {
            'repairs_by_entity_field': dict(self.repairs_by_entity_field),
            'repairs_by_id': self.repairs_by_id,
            'all_repairs': self.all_repairs,
            'stats': {k: v if not isinstance(v, set) else list(v) 
                     for k, v in self.stats.items()}
        }
        
        with open(output_path, 'wb') as f:
            pickle.dump(lookup_data, f)
        
        print(f"\n{'='*80}")
        print(f"LOOKUP TABLE SAVED: {output_path}")
        print(f"{'='*80}\n")
    
    def save_json_export(self, output_path: str):
        """Save human-readable JSON export"""
        export_data = {
            'repairs_by_entity': {
                entity: {
                    field: [asdict(r) for r in repairs]
                    for field, repairs in fields.items()
                }
                for entity, fields in self.repairs_by_entity_field.items()
            },
            'stats': {k: v if not isinstance(v, set) else list(v) 
                     for k, v in self.stats.items()}
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"JSON export saved: {output_path}")
    
    def print_summary(self):
        """Print summary of repairs found"""
        print(f"\n{'='*80}")
        print("REPAIR LOOKUP TABLE SUMMARY")
        print(f"{'='*80}\n")
        
        for entity in sorted(self.repairs_by_entity_field.keys()):
            fields = self.repairs_by_entity_field[entity]
            if not fields:
                continue
            
            print(f"\n{entity}:")
            for field in sorted(fields.keys()):
                repairs = fields[field]
                print(f"  {field}: {len(repairs)} repair(s)")
                for repair in repairs:
                    print(f"    - [{repair.code}] {repair.text}")


class RepairLookup:
    """Lookup repairs from saved pickle file"""
    
    def __init__(self, lookup_path: str):
        with open(lookup_path, 'rb') as f:
            data = pickle.load(f)
        
        self.repairs_by_entity_field = data['repairs_by_entity_field']
        self.repairs_by_id = data['repairs_by_id']
        self.all_repairs = data['all_repairs']
        self.stats = data['stats']
    
    def find_repair(self, entity: str, field: str, error_message: str = "", 
                    threshold: float = 0.3) -> Optional[RepairInfo]:
        """
        Find best matching repair for an error
        
        Args:
            entity: Entity name (e.g., 'cdtrAgt')
            field: Field name (e.g., 'BICFI' or 'FinInstnId.BICFI')
            error_message: Error message from validator
            threshold: Minimum match score (0.0-1.0)
        
        Returns:
            Best matching RepairInfo or None
        """
        # Normalize inputs
        entity = entity.lower()
        field_normalized = self.normalize_field_name(field)
        
        # Get candidate repairs for this entity/field
        candidates = []
        
        if entity in self.repairs_by_entity_field:
            # Exact field match
            if field_normalized in self.repairs_by_entity_field[entity]:
                candidates.extend(self.repairs_by_entity_field[entity][field_normalized])
            
            # Partial field match (for nested fields like FinInstnId.BICFI)
            for repair_field, repairs in self.repairs_by_entity_field[entity].items():
                if repair_field in field_normalized or field_normalized in repair_field:
                    candidates.extend(repairs)
        
        if not candidates:
            return None
        
        # Score each candidate
        scored_repairs = []
        for repair in candidates:
            score = repair.matches_error(error_message, field)
            if score >= threshold:
                scored_repairs.append((score, repair))
        
        if not scored_repairs:
            return None
        
        # Return best match
        scored_repairs.sort(reverse=True, key=lambda x: x[0])
        return scored_repairs[0][1]
    
    def normalize_field_name(self, name: str) -> str:
        """Normalize field name"""
        # Extract last component if nested (e.g., FinInstnId.BICFI -> BICFI)
        if '.' in name:
            name = name.split('.')[-1]
        return name.upper()
    
    def get_all_repairs_for_entity(self, entity: str) -> Dict[str, List[RepairInfo]]:
        """Get all repairs for an entity"""
        entity = entity.lower()
        return self.repairs_by_entity_field.get(entity, {})


def main():
    parser = argparse.ArgumentParser(description='Build repair lookup table')
    parser.add_argument('--data_dir', required=True, help='Directory with JSON payment files')
    parser.add_argument('--output', default='repair_lookup.pkl', help='Output pickle file')
    parser.add_argument('--json-export', help='Optional JSON export file')
    
    args = parser.parse_args()
    
    # Build lookup table
    builder = RepairLookupBuilder()
    builder.analyze_directory(args.data_dir)
    builder.print_summary()
    builder.save_lookup_table(args.output)
    
    if args.json_export:
        builder.save_json_export(args.json_export)
    
    print(f"\nTo use the lookup table:")
    print(f"  from repair_lookup import RepairLookup")
    print(f"  lookup = RepairLookup('{args.output}')")
    print(f"  repair = lookup.find_repair('cdtrAgt', 'BICFI', 'REQUIRED field missing')")
    print(f"  if repair:")
    print(f"      print(repair.text)")


if __name__ == "__main__":
    main()
