"""
ACE Repair Pattern Analyzer
============================

Analyzes payment repair training data to discover patterns and rules.

Usage:
    python nn_stp_analyze.py analyze --input repairs_10k.json --output analysis_report.json

Author: Enhanced Version
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
    
    Identifies:
    - Most frequent repair codes
    - Deterministic rules (100% confidence patterns)
    - Repair co-occurrence patterns
    - Field change patterns per repair
    """
    
    def __init__(self):
        """Initialize counters and storage for analysis"""
        # Count how many times each repair appears
        self.repair_stats = Counter()
        
        # Store all patterns observed for each repair
        self.repair_patterns = defaultdict(list)
        
        # Track which repairs occur together
        self.repair_cooccurrence = defaultdict(Counter)
        
        # Track field changes per repair
        self.field_changes = defaultdict(list)
        
        # Track unique transaction IDs processed
        self.transactions_seen = set()
        
    def analyze(self, json_file: str) -> Dict:
        """
        Main analysis method - processes entire dataset.
        
        Args:
            json_file: Path to JSON file containing training data
            
        Returns:
            Dictionary with analysis results including summary, top repairs,
            deterministic rules, repair combinations, and field change patterns
        """
        logger.info("="*70)
        logger.info("ANALYZING REPAIR PATTERNS")
        logger.info("="*70)
        
        # Load JSON data
        logger.info(f"\nLoading JSON file: {json_file}")
        with open(json_file, 'r') as f:
            raw_data = json.load(f)
        
        logger.info(f"Raw data type: {type(raw_data).__name__}")
        if isinstance(raw_data, list):
            logger.info(f"Raw data is list with {len(raw_data)} elements")
        elif isinstance(raw_data, dict):
            logger.info(f"Raw data is dict with {len(raw_data)} keys")
        
        # Parse data into consistent format
        data = self._parse_input_format(raw_data)
        
        logger.info(f"\nParsed data contains {len(data)} transaction(s)")
        
        # Show sample of what we found
        logger.info(f"\nSample transaction IDs and repair counts:")
        sample_count = 0
        for txn_id in list(data.keys())[:5]:
            txn_data = data[txn_id]
            has_ace = 'ace' in txn_data
            ace_count = len(txn_data.get('ace', [])) if has_ace else 0
            
            if has_ace and ace_count > 0:
                repairs = [r['id'] for r in txn_data['ace']]
                logger.info(f"  {txn_id}: {ace_count} repair(s) - {repairs}")
                sample_count += 1
            else:
                logger.info(f"  {txn_id}: NO REPAIRS")
        
        if len(data) > 5:
            logger.info(f"  ... and {len(data) - 5} more transactions")
        
        logger.info(f"\nStarting analysis loop...")
        
        # Process each transaction
        processed_count = 0
        skipped_count = 0
        
        for txn_id, txn_data in data.items():
            # Check if transaction has repairs to analyze
            if 'ace' not in txn_data or not txn_data['ace']:
                skipped_count += 1
                continue
            
            # Process this transaction
            self._analyze_transaction(txn_id, txn_data)
            processed_count += 1
            
            # Log progress every 1000 transactions
            if processed_count % 1000 == 0:
                logger.info(f"  Progress: {processed_count}/{len(data)} transactions processed...")
        
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
        
        Handles three formats:
        1. Array of single-transaction dicts: [{txn1: {...}}, {txn2: {...}}]
        2. Array of transaction objects: [{...}, {...}]
        3. Dict of transactions: {txn1: {...}, txn2: {...}}
        
        Returns:
            Dict mapping transaction_id -> transaction_data
        """
        if isinstance(raw_data, list):
            logger.info(f"\nParsing list input with {len(raw_data)} element(s)")
            
            if len(raw_data) == 0:
                raise ValueError("Empty array in input file")
            
            # Check first element to determine format
            first_item = raw_data[0]
            logger.info(f"First element type: {type(first_item).__name__}")
            
            if not isinstance(first_item, dict):
                raise ValueError(f"First array element is {type(first_item).__name__}, expected dict")
            
            logger.info(f"First element has {len(first_item)} key(s)")
            
            # Check if values are transaction objects (dicts)
            values_are_dicts = all(isinstance(v, dict) for v in first_item.values())
            logger.info(f"All values are dicts: {values_are_dicts}")
            
            if values_are_dicts:
                # Format 1: Array of dicts, each containing one or more transactions
                # Merge all dicts into single dict
                data = {}
                for item in raw_data:
                    if isinstance(item, dict):
                        data.update(item)
                
                logger.info(f"✓ Detected: Array of {len(raw_data)} dicts merged into {len(data)} transactions")
                
            else:
                # Format 2: Array of transaction objects
                data = {f"txn_{i:06d}": txn for i, txn in enumerate(raw_data)}
                logger.info(f"✓ Detected: Array of {len(data)} transaction objects")
                
        elif isinstance(raw_data, dict):
            # Format 3: Direct dict of transactions
            data = raw_data
            logger.info(f"✓ Detected: Dict of {len(data)} transactions")
            
        else:
            raise ValueError(f"Unsupported data type: {type(raw_data).__name__}")
        
        return data
    
    def _analyze_transaction(self, txn_id: str, txn_data: Dict):
        """
        Analyze a single transaction to extract repair patterns.
        
        For each repair in the transaction:
        1. Count its occurrence
        2. Extract conditions when it occurred (pattern)
        3. Track which other repairs it occurred with (co-occurrence)
        4. Record field changes associated with it
        
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
        
        # Track co-occurrence (which repairs appear together)
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
        """
        Extract features/conditions present when a repair occurred.
        
        Captures the "state" of the payment when repair was needed:
        - Which key fields exist (BIC, IBAN, clearing, country, etc.)
        - Which entities are present
        - What specific changes occurred
        
        Args:
            txn_data: Full transaction data
            repair_id: The repair code being analyzed
            txn_id: Transaction identifier
            
        Returns:
            Pattern dict with boolean flags for conditions
        """
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
        
        # Normalize all keys to lowercase for consistent checking
        txn_normalized = self._normalize_dict(txn_data)
        
        # Check for presence of key fields anywhere in payment
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
        
        # Track which entities have before/after (are being modified)
        entity_keys = [
            'cdtr', 'dbtr',                    # Parties
            'cdtragt', 'dbtragt',              # Agents
            'cdtracct', 'dbtracct',            # Accounts
            'instgagt', 'instdagt',            # Instructing/Instructed agents
            'rmtinf'                           # Remittance info
        ]
        
        for key in entity_keys:
            if key in txn_normalized:
                entity_data = txn_normalized[key]
                
                # Only count entities that have transformations (before/after)
                if isinstance(entity_data, dict) and 'before' in entity_data:
                    pattern['entities_present'].append(key)
                    
                    # Extract specific field changes from 'diffs' array
                    if 'diffs' in entity_data:
                        for diff in entity_data['diffs']:
                            pattern['field_changes'].append({
                                'entity': key,
                                'field': diff.get('key', ''),
                                'action': diff.get('msg', ''),
                                'value': str(diff.get('val', ''))[:50]  # Truncate long values
                            })
        
        return pattern
    
    def _analyze_entity_changes(self, entity_key: str, entity_data: Dict, 
                                repairs: List[str]) -> Dict:
        """
        Analyze what changed in a specific entity.
        
        Categorizes changes into:
        - Fields added (new in 'after', not in 'before')
        - Fields removed (in 'before', gone in 'after')
        - Fields transformed (value changed)
        
        Args:
            entity_key: Entity name (e.g., 'cdtrAgt')
            entity_data: Entity dict with 'before', 'after', 'diffs'
            repairs: List of repairs applied to transaction
            
        Returns:
            Dict summarizing changes in this entity
        """
        changes = {
            'entity': entity_key,
            'fields_added': [],
            'fields_removed': [],
            'fields_transformed': [],
            'repairs': repairs
        }
        
        # Parse 'diffs' array to categorize changes
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
        """
        Generate actionable insights from collected data.
        
        Produces:
        1. Summary statistics
        2. Top 15 most frequent repairs
        3. Deterministic rules (100% confidence patterns)
        4. Common repair combinations
        5. Field change patterns per repair
        
        Returns:
            Dict with all insights
        """
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
        
        # Find deterministic patterns (need at least 3 examples to be confident)
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
                # Avoid duplicates - only add if repair1 < repair2 (alphabetically)
                if repair1 < repair2:
                    combo_list.append({
                        'repair1': repair1,
                        'repair2': repair2,
                        'count': count
                    })
        
        # Sort by frequency and take top 10
        combo_list.sort(key=lambda x: x['count'], reverse=True)
        insights['repair_combinations'] = combo_list[:10]
        
        # Field change patterns - what fields does each repair typically modify?
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
        """
        Identify deterministic rules - conditions that always predict a repair.
        
        A rule is deterministic if it occurs in 100% of cases for that repair.
        Example: "If BIC present AND country missing → always apply repair 6021"
        
        Args:
            repair_id: Repair code to analyze
            patterns: List of pattern dicts from all occurrences
            
        Returns:
            List of rule dicts with condition, confidence, description
        """
        rules = []
        
        # Rule 1: Country from BIC (repair 6021)
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
        
        # Rule 2: BIC from clearing system (repair 6035)
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
        
        # Rule 3: Bank name from BIC (repair 6036)
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
        
        # Entity-specific patterns: Does this repair always affect same entity?
        entity_consistency = defaultdict(int)
        for pattern in patterns:
            for entity in pattern['entities_present']:
                entity_consistency[entity] += 1
        
        # If repair occurs in same entity 95%+ of time, it's a strong pattern
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
        
        # Field change patterns: Does specific field change always trigger this repair?
        field_change_counter = Counter()
        for pattern in patterns:
            for change in pattern['field_changes']:
                # Create unique key: entity.field.action
                key = f"{change['entity']}.{change['field']}.{change['action']}"
                field_change_counter[key] += 1
        
        # If specific field change happens in 90%+ of cases, it's deterministic
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
        """
        Print human-readable analysis report to console.
        
        Args:
            analysis: Insights dict from _generate_insights()
        """
        logger.info("\n" + "="*70)
        logger.info("REPAIR PATTERN ANALYSIS REPORT")
        logger.info("="*70)
        
        # Summary section
        summary = analysis['summary']
        logger.info(f"\nDataset Summary:")
        logger.info(f"  Total Transactions: {summary['total_transactions']}")
        logger.info(f"  Total Repair Instances: {summary['total_repair_instances']}")
        logger.info(f"  Unique Repair Codes: {summary['unique_repairs']}")
        logger.info(f"  Avg Repairs/Transaction: {summary['avg_repairs_per_transaction']:.2f}")
        
        # Top repairs section
        logger.info(f"\nTop 15 Most Frequent Repairs:")
        for repair in analysis['top_repairs']:
            logger.info(f"  {repair['repair_id']}: {repair['count']} occurrences "
                       f"({repair['percentage']:.1f}%)")
        
        # Deterministic rules section
        logger.info(f"\nDeterministic Rules Found: {len(analysis['deterministic_rules'])}")
        if analysis['deterministic_rules']:
            for rule in analysis['deterministic_rules']:
                logger.info(f"  {rule['repair_id']}: {rule['condition']}")
                logger.info(f"    Confidence: {rule['confidence']:.1%}, "
                          f"Occurrences: {rule['occurrences']}")
                logger.info(f"    Description: {rule['description']}")
        else:
            logger.info("  (Need at least 3 examples per repair to detect patterns)")
        
        # Repair combinations section
        if analysis['repair_combinations']:
            logger.info(f"\nCommon Repair Combinations:")
            for combo in analysis['repair_combinations']:
                logger.info(f"  {combo['repair1']} + {combo['repair2']}: "
                          f"{combo['count']} times")
        else:
            logger.info(f"\nNo common repair combinations found")
        
        # Field change patterns section
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
        """
        Recursively normalize all dictionary keys to lowercase.
        
        Ensures consistent field access regardless of capitalization.
        
        Args:
            obj: Any object (dict, list, or primitive)
            
        Returns:
            Normalized version with lowercase keys
        """
        if isinstance(obj, dict):
            return {k.lower() if isinstance(k, str) else k: self._normalize_dict(v) 
                   for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._normalize_dict(item) for item in obj]
        return obj
    
    def _has_field(self, obj, field_name: str) -> bool:
        """
        Check if field exists anywhere in nested structure with non-empty value.
        
        Recursively searches through dicts and lists for field matching name.
        A field is only considered present if it has a non-empty value.
        
        Args:
            obj: Object to search (dict, list, or primitive)
            field_name: Field name to search for (case-insensitive)
            
        Returns:
            True if field exists with non-empty value, False otherwise
        """
        field_name = field_name.lower()
        
        if isinstance(obj, dict):
            for k, v in obj.items():
                # Check if key matches field name
                if isinstance(k, str) and field_name in k.lower():
                    # Only count as "has field" if value is non-empty
                    if v is not None and v != '' and v != []:
                        return True
                
                # Recursively search nested structures
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
    """
    Execute the analyze command from CLI.
    
    Args:
        args: Parsed command-line arguments
    """
    analyzer = RepairPatternAnalyzer()
    analysis = analyzer.analyze(args.input)
    
    # Save results to JSON if output path specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(analysis, f, indent=2)
        logger.info(f"\nAnalysis saved to {args.output}")


def main():
    """Main entry point for CLI"""
    parser = argparse.ArgumentParser(
        description='ACE Repair Pattern Analyzer - Discover patterns in repair training data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze repair patterns and print to console
  python nn_stp_analyze.py analyze --input repairs_10k.json
  
  # Analyze and save results to JSON file
  python nn_stp_analyze.py analyze --input repairs_10k.json --output analysis_report.json
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Analyze command
    analyze_parser = subparsers.add_parser(
        'analyze', 
        help='Analyze dataset for repair patterns'
    )
    analyze_parser.add_argument(
        '--input', 
        required=True, 
        help='Training data JSON file'
    )
    analyze_parser.add_argument(
        '--output', 
        help='Output JSON file for analysis results (optional)'
    )
    
    args = parser.parse_args()
    
    if args.command == 'analyze':
        analyze_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
