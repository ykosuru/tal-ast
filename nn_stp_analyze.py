"""
ACE Repair Pattern Analyzer and Enhanced Predictor
==================================================
Fixed version with bug corrections
"""

import json
import argparse
import os
import logging
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# PATTERN ANALYZER
# ============================================================================

class RepairPatternAnalyzer:
    """
    Analyzes dataset to discover deterministic repair patterns.
    """
    
    def __init__(self):
        self.repair_stats = Counter()
        self.repair_patterns = defaultdict(list)
        self.repair_cooccurrence = defaultdict(Counter)
        self.field_changes = defaultdict(list)
        self.transactions_seen = set()  # Track unique transaction IDs
        
    def analyze(self, json_file: str) -> Dict:
        """Analyze dataset and return comprehensive statistics"""
        logger.info("="*70)
        logger.info("ANALYZING REPAIR PATTERNS")
        logger.info("="*70)
        
        with open(json_file, 'r') as f:
            raw_data = json.load(f)
        
        # Handle different formats
        if isinstance(raw_data, list):
            if len(raw_data) > 0 and isinstance(raw_data[0], dict):
                first_item = raw_data[0]
                if all(isinstance(v, dict) for v in first_item.values()):
                    data = first_item
                else:
                    data = {f"txn_{i:06d}": txn for i, txn in enumerate(raw_data)}
            else:
                raise ValueError("Unexpected array format")
        else:
            data = raw_data
        
        logger.info(f"Analyzing {len(data)} transactions...")
        
        # Analyze each transaction
        for txn_id, txn_data in data.items():
            self._analyze_transaction(txn_id, txn_data)
        
        # Generate insights
        analysis = self._generate_insights()
        
        # Print report
        self._print_report(analysis)
        
        return analysis
    
    def _analyze_transaction(self, txn_id: str, txn_data: Dict):
        """Analyze a single transaction for patterns"""
        repairs = [r['id'] for r in txn_data.get('ace', [])]
        
        if not repairs:
            return
        
        # Track transaction
        self.transactions_seen.add(txn_id)
        
        # Count repair frequency
        for repair_id in repairs:
            self.repair_stats[repair_id] += 1
        
        # Track co-occurrence
        for i, repair1 in enumerate(repairs):
            for repair2 in repairs[i+1:]:
                self.repair_cooccurrence[repair1][repair2] += 1
                self.repair_cooccurrence[repair2][repair1] += 1
        
        # Extract patterns for each repair
        for repair_id in repairs:
            pattern = self._extract_pattern(txn_data, repair_id, txn_id)
            self.repair_patterns[repair_id].append(pattern)
        
        # Analyze field changes
        for entity_key, entity_data in txn_data.items():
            if isinstance(entity_data, dict) and 'before' in entity_data and 'after' in entity_data:
                changes = self._analyze_entity_changes(entity_key, entity_data, repairs)
                for repair_id in repairs:
                    self.field_changes[repair_id].append(changes)
    
    def _extract_pattern(self, txn_data: Dict, repair_id: str, txn_id: str) -> Dict:
        """Extract features/conditions when a repair occurs"""
        pattern = {
            'repair_id': repair_id,
            'txn_id': txn_id,
            'has_bic': False,
            'has_iban': False,
            'has_clearing': False,
            'has_country': False,
            'has_name': False,
            'has_address': False,
            'entities_present': [],
            'field_changes': []
        }
        
        # Normalize to lowercase
        txn_normalized = self._normalize_dict(txn_data)
        
        # Check for key fields
        pattern['has_bic'] = self._has_field(txn_normalized, 'bicfi') or self._has_field(txn_normalized, 'bic')
        pattern['has_iban'] = self._has_field(txn_normalized, 'iban')
        pattern['has_clearing'] = self._has_field(txn_normalized, 'mmbid') or self._has_field(txn_normalized, 'clrsysmmbid')
        pattern['has_country'] = self._has_field(txn_normalized, 'ctryofres') or self._has_field(txn_normalized, 'ctry')
        pattern['has_name'] = self._has_field(txn_normalized, 'nm')
        pattern['has_address'] = self._has_field(txn_normalized, 'adrline') or self._has_field(txn_normalized, 'pstladr')
        
        # Track which entities have before/after
        for key in ['cdtr', 'dbtr', 'cdtragt', 'dbtragt', 'instgagt', 'instdagt', 'rmtinf']:
            if key in txn_normalized:
                entity_data = txn_normalized[key]
                if isinstance(entity_data, dict) and 'before' in entity_data:
                    pattern['entities_present'].append(key)
                    
                    # Track what changed
                    if 'diffs' in entity_data:
                        for diff in entity_data['diffs']:
                            pattern['field_changes'].append({
                                'entity': key,
                                'field': diff.get('key', ''),
                                'action': diff.get('msg', ''),
                                'value': str(diff.get('val', ''))[:50]  # Truncate
                            })
        
        return pattern
    
    def _analyze_entity_changes(self, entity_key: str, entity_data: Dict, repairs: List[str]) -> Dict:
        """Analyze what changed in an entity"""
        changes = {
            'entity': entity_key,
            'fields_added': [],
            'fields_removed': [],
            'fields_transformed': [],
            'repairs': repairs
        }
        
        if 'diffs' in entity_data:
            for diff in entity_data['diffs']:
                action = diff.get('msg', '')
                field = diff.get('key', '')
                
                if action == 'added':
                    changes['fields_added'].append(field)
                elif action == 'dropped':
                    changes['fields_removed'].append(field)
                elif action == 'transformed':
                    changes['fields_transformed'].append(field)
        
        return changes
    
    def _generate_insights(self) -> Dict:
        """Generate actionable insights from analysis"""
        total_repairs = sum(self.repair_stats.values())
        total_transactions = len(self.transactions_seen)  # FIXED: Use tracked transaction IDs
        
        insights = {
            'summary': {
                'total_transactions': total_transactions,
                'total_repair_instances': total_repairs,
                'unique_repairs': len(self.repair_stats),
                'avg_repairs_per_transaction': total_repairs / max(total_transactions, 1)
            },
            'top_repairs': [],
            'deterministic_rules': [],
            'repair_combinations': [],
            'field_change_patterns': []
        }
        
        # Top 15 repairs by frequency
        for repair_id, count in self.repair_stats.most_common(15):
            percentage = (count / total_transactions) * 100 if total_transactions > 0 else 0
            insights['top_repairs'].append({
                'repair_id': repair_id,
                'count': count,
                'percentage': percentage
            })
        
        # Find deterministic patterns
        for repair_id, patterns in self.repair_patterns.items():
            if len(patterns) < 3:  # Need at least 3 examples
                continue
            
            # Check if pattern is deterministic (100% consistent)
            rules = self._find_deterministic_rules(repair_id, patterns)
            if rules:
                insights['deterministic_rules'].extend(rules)
        
        # Common repair combinations
        combo_list = []
        for repair1, cooccur_dict in self.repair_cooccurrence.items():
            for repair2, count in cooccur_dict.most_common(3):
                # Avoid duplicates (only add if repair1 < repair2)
                if repair1 < repair2:
                    combo_list.append({
                        'repair1': repair1,
                        'repair2': repair2,
                        'count': count
                    })
        
        # Sort and take top 10
        combo_list.sort(key=lambda x: x['count'], reverse=True)
        insights['repair_combinations'] = combo_list[:10]
        
        # Field change patterns
        for repair_id, changes_list in self.field_changes.items():
            field_add_counter = Counter()
            field_transform_counter = Counter()
            
            for changes in changes_list:
                for field in changes['fields_added']:
                    field_add_counter[field] += 1
                for field in changes['fields_transformed']:
                    field_transform_counter[field] += 1
            
            if field_add_counter or field_transform_counter:
                insights['field_change_patterns'].append({
                    'repair_id': repair_id,
                    'common_additions': dict(field_add_counter.most_common(5)),
                    'common_transformations': dict(field_transform_counter.most_common(5))
                })
        
        return insights
    
    def _find_deterministic_rules(self, repair_id: str, patterns: List[Dict]) -> List[Dict]:
        """Find conditions that always predict this repair"""
        rules = []
        
        # Rule: BIC present + Country missing → Extract country from BIC
        if repair_id == '6021':
            has_bic_no_country = sum(1 for p in patterns if p['has_bic'] and not p['has_country'])
            if has_bic_no_country == len(patterns):
                rules.append({
                    'repair_id': repair_id,
                    'condition': 'has_bic AND NOT has_country',
                    'confidence': 1.0,
                    'occurrences': len(patterns),
                    'description': 'Extract ISO country code from BIC'
                })
        
        # Rule: Clearing ID present + BIC missing → Lookup BIC
        if repair_id == '6035':
            has_clearing_no_bic = sum(1 for p in patterns if p['has_clearing'] and not p['has_bic'])
            if has_clearing_no_bic == len(patterns):
                rules.append({
                    'repair_id': repair_id,
                    'condition': 'has_clearing AND NOT has_bic',
                    'confidence': 1.0,
                    'occurrences': len(patterns),
                    'description': 'Resolve BIC from clearing system ID'
                })
        
        # Rule: BIC present + Name missing → Lookup name
        if repair_id == '6036':
            has_bic_no_name = sum(1 for p in patterns if p['has_bic'] and not p['has_name'])
            if has_bic_no_name == len(patterns):
                rules.append({
                    'repair_id': repair_id,
                    'condition': 'has_bic AND NOT has_name',
                    'confidence': 1.0,
                    'occurrences': len(patterns),
                    'description': 'Lookup bank name from BIC directory'
                })
        
        # Check entity-specific patterns
        entity_consistency = defaultdict(int)
        for pattern in patterns:
            for entity in pattern['entities_present']:
                entity_consistency[entity] += 1
        
        # If repair always occurs in same entity with 95%+ consistency
        for entity, count in entity_consistency.items():
            consistency = count / len(patterns)
            if consistency >= 0.95:
                rules.append({
                    'repair_id': repair_id,
                    'condition': f'changes_in_{entity}',
                    'confidence': consistency,
                    'occurrences': count,
                    'description': f'Repair typically affects {entity} entity'
                })
        
        # Check for common field change patterns
        field_change_counter = Counter()
        for pattern in patterns:
            for change in pattern['field_changes']:
                key = f"{change['entity']}.{change['field']}.{change['action']}"
                field_change_counter[key] += 1
        
        # If a specific field change happens in 90%+ of cases
        for change_pattern, count in field_change_counter.items():
            consistency = count / len(patterns)
            if consistency >= 0.90:
                rules.append({
                    'repair_id': repair_id,
                    'condition': f'field_change:{change_pattern}',
                    'confidence': consistency,
                    'occurrences': count,
                    'description': f'Field change pattern: {change_pattern}'
                })
        
        return rules
    
    def _print_report(self, analysis: Dict):
        """Print analysis report"""
        logger.info("\n" + "="*70)
        logger.info("REPAIR PATTERN ANALYSIS REPORT")
        logger.info("="*70)
        
        summary = analysis['summary']
        logger.info(f"\nDataset Summary:")
        logger.info(f"  Total Transactions: {summary['total_transactions']}")
        logger.info(f"  Total Repair Instances: {summary['total_repair_instances']}")
        logger.info(f"  Unique Repair Codes: {summary['unique_repairs']}")
        logger.info(f"  Avg Repairs/Transaction: {summary['avg_repairs_per_transaction']:.2f}")
        
        logger.info(f"\nTop 15 Most Frequent Repairs:")
        for repair in analysis['top_repairs']:
            logger.info(f"  {repair['repair_id']}: {repair['count']} occurrences ({repair['percentage']:.1f}%)")
        
        logger.info(f"\nDeterministic Rules Found: {len(analysis['deterministic_rules'])}")
        for rule in analysis['deterministic_rules']:
            logger.info(f"  {rule['repair_id']}: {rule['condition']}")
            logger.info(f"    Confidence: {rule['confidence']:.1%}, Occurrences: {rule['occurrences']}")
            logger.info(f"    Description: {rule['description']}")
        
        if analysis['repair_combinations']:
            logger.info(f"\nCommon Repair Combinations:")
            for combo in analysis['repair_combinations'][:10]:
                logger.info(f"  {combo['repair1']} + {combo['repair2']}: {combo['count']} times")
        else:
            logger.info(f"\nNo common repair combinations found (single repairs only)")
        
        logger.info(f"\nField Change Patterns:")
        for pattern in analysis['field_change_patterns'][:10]:
            logger.info(f"  Repair {pattern['repair_id']}:")
            if pattern['common_additions']:
                logger.info(f"    Often adds: {list(pattern['common_additions'].keys())[:3]}")
            if pattern['common_transformations']:
                logger.info(f"    Often transforms: {list(pattern['common_transformations'].keys())[:3]}")
    
    def _normalize_dict(self, obj):
        """Recursively normalize dict keys to lowercase"""
        if isinstance(obj, dict):
            return {k.lower() if isinstance(k, str) else k: self._normalize_dict(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._normalize_dict(item) for item in obj]
        return obj
    
    def _has_field(self, obj, field_name: str) -> bool:
        """Check if field exists anywhere in object"""
        field_name = field_name.lower()
        
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(k, str) and field_name in k.lower():
                    if v is not None and v != '' and v != []:
                        return True
                if isinstance(v, (dict, list)):
                    if self._has_field(v, field_name):
                        return True
        elif isinstance(obj, list):
            for item in obj:
                if self._has_field(item, field_name):
                    return True
        
        return False


# ============================================================================
# CLI COMMANDS
# ============================================================================

def analyze_command(args):
    """Analyze dataset for patterns"""
    analyzer = RepairPatternAnalyzer()
    analysis = analyzer.analyze(args.input)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(analysis, f, indent=2)
        logger.info(f"\nAnalysis saved to {args.output}")


def main():
    parser = argparse.ArgumentParser(
        description='ACE Repair Pattern Analyzer',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Analyze
    analyze_parser = subparsers.add_parser('analyze', help='Analyze dataset for patterns')
    analyze_parser.add_argument('--input', required=True, help='Training data JSON file')
    analyze_parser.add_argument('--output', help='Output analysis JSON file')
    
    args = parser.parse_args()
    
    if args.command == 'analyze':
        analyze_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()


