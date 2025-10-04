"""
ACE Repair Pattern Analyzer
============================

Analyzes payment repair training data to discover patterns and rules.

Usage:
    python nn_stp_analyze.py analyze --input repairs_10k.json --output analysis_report.json

Author: Enhanced Version with Debug Logging
Date: 2025-10-04
"""

import json
import argparse
import os
import logging
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict, Counter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# PATTERN ANALYZER
# ============================================================================

class RepairPatternAnalyzer:
    """
    Analyzes payment repair dataset to discover patterns and rules.
    """
    
    def __init__(self):
        """Initialize counters and storage for analysis"""
        self.repair_stats = Counter()
        self.repair_patterns = defaultdict(list)
        self.repair_cooccurrence = defaultdict(Counter)
        self.field_changes = defaultdict(list)
        self.transactions_seen = set()
        
    def analyze(self, json_file: str) -> Dict:
        """
        Main analysis method - processes entire dataset.
        
        Args:
            json_file: Path to JSON file containing training data
            
        Returns:
            Dictionary containing analysis results
        """
        logger.info("="*70)
        logger.info("ANALYZING REPAIR PATTERNS")
        logger.info("="*70)
        
        # Load JSON data
        logger.info(f"\nLoading JSON file: {json_file}")
        with open(json_file, 'r') as f:
            raw_data = json.load(f)
        
        logger.info(f"Raw data type: {type(raw_data)}")
        if isinstance(raw_data, list):
            logger.info(f"Raw data is list with {len(raw_data)} elements")
        elif isinstance(raw_data, dict):
            logger.info(f"Raw data is dict with {len(raw_data)} keys")
        
        # Parse data into consistent format
        data = self._parse_input_format(raw_data)
        
        logger.info(f"\nParsed data contains {len(data)} transaction(s)")
        logger.info(f"\nTransaction IDs and repair counts:")
        
        # Show what we found
        for txn_id in data.keys():
            txn_data = data[txn_id]
            has_ace = 'ace' in txn_data
            ace_count = len(txn_data.get('ace', [])) if has_ace else 0
            
            if has_ace and ace_count > 0:
                repairs = [r['id'] for r in txn_data['ace']]
                logger.info(f"  {txn_id}: {ace_count} repair(s) - {repairs}")
            else:
                logger.info(f"  {txn_id}: NO REPAIRS FOUND")
        
        logger.info(f"\nStarting analysis loop...")
        
        # Process each transaction
        processed_count = 0
        skipped_count = 0
        
        for txn_id, txn_data in data.items():
            logger.info(f"\nChecking transaction: {txn_id}")
            
            # Check if transaction has repairs to analyze
            if 'ace' not in txn_data:
                skipped_count += 1
                logger.info(f"  -> SKIPPED: No 'ace' key found")
                continue
                
            if not txn_data['ace']:
                skipped_count += 1
                logger.info(f"  -> SKIPPED: 'ace' array is empty")
                continue
            
            # Process this transaction
            repairs = [r['id'] for r in txn_data['ace']]
            logger.info(f"  -> PROCESSING: {len(repairs)} repair(s) - {repairs}")
            
            self._analyze_transaction(txn_id, txn_data)
            processed_count += 1
            
            # Log progress every 100 transactions
            if processed_count % 100 == 0:
                logger.info(f"  Progress: {processed_count} transactions processed...")
        
        # Log completion stats
        logger.info(f"\n{'='*70}")
        logger.info(f"ANALYSIS COMPLETE")
        logger.info(f"{'='*70}")
        logger.info(f"Total transactions in file: {len(data)}")
        logger.info(f"Transactions with repairs: {processed_count}")
        logger.info(f"Transactions skipped (no repairs): {skipped_count}")
        logger.info(f"Total repair instances: {sum(self.repair_stats.values())}")
        logger.info(f"Unique repair codes: {len(self.repair_stats)}")
        
        # Generate insights from collected data
        analysis = self._generate_insights()
        
        # Print human-readable report
        self._print_report(analysis)
        
        return analysis
    
    def _parse_input_format(self, raw_data) -> Dict:
        """
        Parse various input formats into standard dict of transactions.
        
        Handles:
        1. Array containing single dict: [{txn1: {...}, txn2: {...}}]
        2. Array of dicts: [{...}, {...}]
        3. Dict of transactions: {txn1: {...}, txn2: {...}}
        
        Returns:
            Dict mapping transaction_id -> transaction_data
        """
        if isinstance(raw_data, list):
            logger.info(f"\nParsing list input with {len(raw_data)} element(s)")
            
            if len(raw_data) == 0:
                raise ValueError("Empty array in input file")
            
            # Check first element
            first_item = raw_data[0]
            logger.info(f"First element type: {type(first_item)}")
            
            if not isinstance(first_item, dict):
                raise ValueError(f"First array element is {type(first_item)}, expected dict")
            
            logger.info(f"First element has {len(first_item)} keys")
            logger.info(f"First 3 keys: {list(first_item.keys())[:3]}")
            
            # Check if all values in first element are dicts (Format 1)
            values_are_dicts = [isinstance(v, dict) for v in first_item.values()]
            logger.info(f"Values are dicts: {all(values_are_dicts)} (checked {len(values_are_dicts)} values)")
            
            if all(values_are_dicts):
                # Format 1: Array containing dict of transactions
                data = first_item
                logger.info(f"✓ Detected: Array containing dict of {len(data)} transactions")
            else:
                # Format 2: Array of transaction objects
                data = {f"txn_{i:06d}": txn for i, txn in enumerate(raw_data)}
                logger.info(f"✓ Detected: Array of {len(data)} transaction objects")
                
        elif isinstance(raw_data, dict):
            # Format 3: Direct dict of transactions
            data = raw_data
            logger.info(f"✓ Detected: Dict of {len(data)} transactions")
            
        else:
            raise ValueError(f"Unsupported data type: {type(raw_data)}")
        
        return data
    
    def _analyze_transaction(self, txn_id: str, txn_data: Dict):
        """
        Analyze a single transaction to extract repair patterns.
        
        Args:
            txn_id: Unique transaction identifier
            txn_data: Transaction data including 'ace' repairs and entities
        """
        # Extract repair IDs from 'ace' array
        repairs = [r['id'] for r in txn_data.get('ace', [])]
        
        if not repairs:
            return
        
        # Track that we processed this transaction
        self.transactions_seen.add(txn_id)
        
        # Count each repair occurrence
        for repair_id in repairs:
            self.repair_stats[repair_id] += 1
        
        # Track co-occurrence
        for i, repair1 in enumerate(repairs):
            for repair2 in repairs[i+1:]:
                self.repair_cooccurrence[repair1][repair2] += 1
                self.repair_cooccurrence[repair2][repair1] += 1
        
        # Extract pattern for each repair
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
        """Extract features/conditions present when a repair occurred."""
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
        
        # Normalize all keys to lowercase
        txn_normalized = self._normalize_dict(txn_data)
        
        # Check for presence of key fields
        pattern['has_bic'] = self._has_field(txn_normalized, 'bicfi') or \
                            self._has_field(txn_normalized, 'bic')
        pattern['has_iban'] = self._has_field(txn_normalized, 'iban')
        pattern['has_clearing'] = self._has_field(txn_normalized, 'mmbid') or \
                                  self._has_field(txn_normalized, 'clrsysmmbid')
        pattern['has_country'] = self._has_field(txn_normalized, 'ctryofres') or \
                                self._has_field(txn_normalized, 'ctry')
        pattern['has_name'] = self._has_field(txn_normalized, 'nm')
        pattern['has_address'] = self._has_field(txn_normalized, 'adrline') or \
                                self._has_field(txn_normalized, 'pstladr')
        
        # Track which entities have before/after - INCLUDES ACCOUNT ENTITIES
        entity_keys = [
            'cdtr', 'dbtr', 
            'cdtragt', 'dbtragt', 
            'cdtracct', 'dbtracct',  # ADDED: Account entities
            'instgagt', 'instdagt', 
            'rmtinf'
        ]
        
        for key in entity_keys:
            if key in txn_normalized:
                entity_data = txn_normalized[key]
                
                if isinstance(entity_data, dict) and 'before' in entity_data:
                    pattern['entities_present'].append(key)
                    
                    # Extract field changes from 'diffs'
                    if 'diffs' in entity_data:
                        for diff in entity_data['diffs']:
                            pattern['field_changes'].append({
                                'entity': key,
                                'field': diff.get('key', ''),
                                'action': diff.get('msg', ''),
                                'value': str(diff.get('val', ''))[:50]
                            })
        
        return pattern
    
    def _analyze_entity_changes(self, entity_key: str, entity_data: Dict, 
                                repairs: List[str]) -> Dict:
        """Analyze what changed in a specific entity."""
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
        """Generate actionable insights from collected data."""
        total_repairs = sum(self.repair_stats.values())
        total_transactions = len(self.transactions_seen)
        
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
        
        # Find deterministic patterns (need at least 3 examples)
        for repair_id, patterns in self.repair_patterns.items():
            if len(patterns) < 3:
                continue
            
            rules = self._find_deterministic_rules(repair_id, patterns)
            if rules:
                insights['deterministic_rules'].extend(rules)
        
        # Common repair combinations
        combo_list = []
        for repair1, cooccur_dict in self.repair_cooccurrence.items():
            for repair2, count in cooccur_dict.most_common(3):
                if repair1 < repair2:
                    combo_list.append({
                        'repair1': repair1,
                        'repair2': repair2,
                        'count': count
                    })
        
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
        """Identify deterministic rules with 100% confidence."""
        rules = []
        
        # Rule 1: Country from BIC
        if repair_id == '6021':
            has_bic_no_country = sum(1 for p in patterns 
                                    if p['has_bic'] and not p['has_country'])
            
            if has_bic_no_country == len(patterns):
                rules.append({
                    'repair_id': repair_id,
                    'condition': 'has_bic AND NOT has_country',
                    'confidence': 1.0,
                    'occurrences': len(patterns),
                    'description': 'Extract ISO country code from BIC'
                })
        
        # Rule 2: BIC from clearing
        if repair_id == '6035':
            has_clearing_no_bic = sum(1 for p in patterns 
                                     if p['has_clearing'] and not p['has_bic'])
            
            if has_clearing_no_bic == len(patterns):
                rules.append({
                    'repair_id': repair_id,
                    'condition': 'has_clearing AND NOT has_bic',
                    'confidence': 1.0,
                    'occurrences': len(patterns),
                    'description': 'Resolve BIC from clearing system ID'
                })
        
        # Rule 3: Name from BIC
        if repair_id == '6036':
            has_bic_no_name = sum(1 for p in patterns 
                                 if p['has_bic'] and not p['has_name'])
            
            if has_bic_no_name == len(patterns):
                rules.append({
                    'repair_id': repair_id,
                    'condition': 'has_bic AND NOT has_name',
                    'confidence': 1.0,
                    'occurrences': len(patterns),
                    'description': 'Lookup bank name from BIC directory'
                })
        
        # Entity-specific patterns
        entity_consistency = defaultdict(int)
        for pattern in patterns:
            for entity in pattern['entities_present']:
                entity_consistency[entity] += 1
        
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
        
        # Field change patterns
        field_change_counter = Counter()
        for pattern in patterns:
            for change in pattern['field_changes']:
                key = f"{change['entity']}.{change['field']}.{change['action']}"
                field_change_counter[key] += 1
        
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
        """Print human-readable analysis report."""
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
            logger.info(f"  {repair['repair_id']}: {repair['count']} occurrences "
                       f"({repair['percentage']:.1f}%)")
        
        logger.info(f"\nDeterministic Rules Found: {len(analysis['deterministic_rules'])}")
        if analysis['deterministic_rules']:
            for rule in analysis['deterministic_rules']:
                logger.info(f"  {rule['repair_id']}: {rule['condition']}")
                logger.info(f"    Confidence: {rule['confidence']:.1%}, "
                          f"Occurrences: {rule['occurrences']}")
                logger.info(f"    Description: {rule['description']}")
        else:
            logger.info("  (Need at least 3 examples per repair to detect patterns)")
        
        if analysis['repair_combinations']:
            logger.info(f"\nCommon Repair Combinations:")
            for combo in analysis['repair_combinations'][:10]:
                logger.info(f"  {combo['repair1']} + {combo['repair2']}: "
                          f"{combo['count']} times")
        else:
            logger.info(f"\nNo common repair combinations found")
        
        logger.info(f"\nField Change Patterns (Top 10 repairs):")
        for pattern in analysis['field_change_patterns'][:10]:
            logger.info(f"  Repair {pattern['repair_id']}:")
            if pattern['common_additions']:
                top_additions = list(pattern['common_additions'].keys())[:3]
                logger.info(f"    Often adds: {top_additions}")
            if pattern['common_transformations']:
                top_transforms = list(pattern['common_transformations'].keys())[:3]
                logger.info(f"    Often transforms: {top_transforms}")
    
    def _normalize_dict(self, obj):
        """Recursively normalize dict keys to lowercase."""
        if isinstance(obj, dict):
            return {k.lower() if isinstance(k, str) else k: self._normalize_dict(v) 
                   for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._normalize_dict(item) for item in obj]
        return obj
    
    def _has_field(self, obj, field_name: str) -> bool:
        """Check if field exists with non-empty value."""
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
# CLI
# ============================================================================

def analyze_command(args):
    """Execute analyze command."""
    analyzer = RepairPatternAnalyzer()
    analysis = analyzer.analyze(args.input)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(analysis, f, indent=2)
        logger.info(f"\nAnalysis saved to {args.output}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='ACE Repair Pattern Analyzer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python nn_stp_analyze.py analyze --input repairs_10k.json
  python nn_stp_analyze.py analyze --input repairs_10k.json --output analysis.json
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    analyze_parser = subparsers.add_parser('analyze', help='Analyze dataset')
    analyze_parser.add_argument('--input', required=True, help='Training data JSON file')
    analyze_parser.add_argument('--output', help='Output JSON file (optional)')
    
    args = parser.parse_args()
    
    if args.command == 'analyze':
        analyze_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
