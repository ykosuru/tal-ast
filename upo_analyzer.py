#!/usr/bin/env python3
"""
Context-Aware Payment Data Structure Analyzer
Discovers field requirements based on source, clearing, and party context
"""

import json
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Any, Set, Tuple
from dataclasses import dataclass, field
import argparse


@dataclass
class FieldStats:
    """Statistics for a field"""
    total_count: int = 0
    present_count: int = 0
    value_samples: List[Any] = field(default_factory=list)
    value_lengths: List[int] = field(default_factory=list)
    has_dashes: int = 0
    data_types: Counter = field(default_factory=Counter)
    
    @property
    def presence_rate(self) -> float:
        return (self.present_count / self.total_count * 100) if self.total_count > 0 else 0
    
    @property
    def is_always_present(self) -> bool:
        return self.presence_rate == 100.0
    
    @property
    def is_common(self) -> bool:
        return self.presence_rate >= 80.0


class ContextAwareAnalyzer:
    """Analyzes payment data with context awareness"""
    
    def __init__(self):
        # Overall stats
        self.entity_stats = defaultdict(lambda: {
            'count': 0,
            'fields': defaultdict(FieldStats),
            'before_fields': defaultdict(FieldStats),
            'after_fields': defaultdict(FieldStats)
        })
        
        # Context-specific stats: context_key -> entity -> fields
        self.context_stats = defaultdict(lambda: defaultdict(lambda: {
            'count': 0,
            'before_fields': defaultdict(FieldStats),
            'after_fields': defaultdict(FieldStats)
        }))
        
        # Track context patterns
        self.context_patterns = Counter()
        self.total_payments = 0
    
    def create_context_key(self, source: str, clearing: str, parties: Dict) -> str:
        """Create context key from source, clearing, and parties"""
        # Get active parties
        active_parties = sorted([k for k, v in parties.items() if v])
        parties_str = ','.join(active_parties) if active_parties else 'none'
        
        return f"{source}|{clearing}|{parties_str}"
    
    def extract_context(self, payment: Dict) -> Tuple[str, str, Dict]:
        """Extract source, clearing, and parties from payment"""
        source = payment.get('source', 'UNKNOWN')
        clearing = payment.get('clearing', 'UNKNOWN')
        parties = payment.get('parties', {})
        
        return source, clearing, parties
    
    def analyze_directory(self, data_dir: str):
        """Analyze all JSON files in directory"""
        json_files = list(Path(data_dir).glob('**/*.json'))
        print(f"\n{'='*80}")
        print(f"CONTEXT-AWARE ANALYSIS OF {len(json_files)} JSON FILES")
        print(f"{'='*80}\n")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Handle different formats
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            for txn_id, txn_data in item.items():
                                self.analyze_payment(txn_data)
                elif isinstance(data, dict):
                    for key, value in data.items():
                        if isinstance(value, dict):
                            self.analyze_payment(value)
                
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
        
        print(f"Total payments analyzed: {self.total_payments}")
        print(f"Unique contexts discovered: {len(self.context_patterns)}")
    
    def analyze_payment(self, payment: Dict):
        """Analyze a single payment"""
        self.total_payments += 1
        
        # Extract context
        source, clearing, parties = self.extract_context(payment)
        context_key = self.create_context_key(source, clearing, parties)
        self.context_patterns[context_key] += 1
        
        # Known entities to check
        entities = [
            'cdtr', 'dbtr', 'cdtrAgt', 'dbtrAgt', 'cdtrAcct', 'dbtrAcct',
            'instgAgt', 'instdAgt', 'intrmyAgt1', 'intrmyAgt2', 'intrmyAgt3',
            'rmtInf'
        ]
        
        for entity_name in entities:
            # Check both camelCase and PascalCase
            entity_data = self._find_entity(payment, entity_name)
            
            if entity_data:
                # Overall stats
                stats = self.entity_stats[entity_name]
                stats['count'] += 1
                
                # Context-specific stats
                ctx_stats = self.context_stats[context_key][entity_name]
                ctx_stats['count'] += 1
                
                # Check if it has before/after structure
                if isinstance(entity_data, dict):
                    if 'before' in entity_data and 'after' in entity_data:
                        # Training data format
                        self.analyze_entity_fields(entity_data['before'], stats['before_fields'])
                        self.analyze_entity_fields(entity_data['after'], stats['after_fields'])
                        
                        # Context-specific
                        self.analyze_entity_fields(entity_data['before'], ctx_stats['before_fields'])
                        self.analyze_entity_fields(entity_data['after'], ctx_stats['after_fields'])
                    else:
                        # Direct format
                        self.analyze_entity_fields(entity_data, stats['fields'])
    
    def _find_entity(self, payment: Dict, entity_name: str) -> Any:
        """Find entity in payment with case-insensitive search"""
        if entity_name in payment:
            return payment[entity_name]
        elif entity_name.capitalize() in payment:
            return payment[entity_name.capitalize()]
        elif entity_name[0].upper() + entity_name[1:] in payment:
            return payment[entity_name[0].upper() + entity_name[1:]]
        return None
    
    def analyze_entity_fields(self, entity: Any, field_stats: Dict):
        """Recursively analyze entity fields"""
        if not isinstance(entity, dict):
            return
        
        for key, value in entity.items():
            stats = field_stats[key]
            stats.total_count += 1
            
            if value is not None and value != '':
                stats.present_count += 1
                
                # Sample values
                if len(stats.value_samples) < 5:
                    if isinstance(value, str):
                        stats.value_samples.append(value[:50])
                    elif not isinstance(value, (dict, list)):
                        stats.value_samples.append(value)
                
                # Track data type
                stats.data_types[type(value).__name__] += 1
                
                # For strings, track length and special chars
                if isinstance(value, str):
                    stats.value_lengths.append(len(value))
                    if '-' in value:
                        stats.has_dashes += 1
                
                # Recurse into nested objects
                if isinstance(value, dict):
                    self.analyze_entity_fields(value, field_stats)
    
    def generate_report(self):
        """Generate comprehensive context-aware analysis report"""
        print(f"\n{'='*80}")
        print("CONTEXT PATTERNS DISCOVERED")
        print(f"{'='*80}\n")
        
        print(f"Top 10 Context Combinations:")
        for context_key, count in self.context_patterns.most_common(10):
            parts = context_key.split('|')
            source = parts[0] if len(parts) > 0 else 'N/A'
            clearing = parts[1] if len(parts) > 1 else 'N/A'
            parties = parts[2] if len(parts) > 2 else 'N/A'
            print(f"  {count:4d} payments: source={source}, clearing={clearing}")
            print(f"        parties: {parties}")
        
        print(f"\n{'='*80}")
        print("CONTEXT-SPECIFIC FIELD REQUIREMENTS")
        print(f"{'='*80}\n")
        
        # Analyze each major context
        for context_key in self.context_patterns.most_common(5):
            self.print_context_analysis(context_key[0])
        
        print(f"\n{'='*80}")
        print("OVERALL ENTITY ANALYSIS (All Contexts)")
        print(f"{'='*80}\n")
        
        for entity_name, stats in sorted(self.entity_stats.items()):
            if stats['count'] == 0:
                continue
            
            print(f"\n{'-'*80}")
            print(f"ENTITY: {entity_name}")
            print(f"{'-'*80}")
            print(f"Present in: {stats['count']} / {self.total_payments} payments ({stats['count']/self.total_payments*100:.1f}%)")
            
            # Show fields added by repairs (before vs after)
            if stats['before_fields'] and stats['after_fields']:
                self.compare_before_after(entity_name, stats['before_fields'], stats['after_fields'])
    
    def print_context_analysis(self, context_key: str):
        """Print analysis for a specific context"""
        parts = context_key.split('|')
        source = parts[0] if len(parts) > 0 else 'N/A'
        clearing = parts[1] if len(parts) > 1 else 'N/A'
        parties = parts[2] if len(parts) > 2 else 'N/A'
        
        count = self.context_patterns[context_key]
        
        print(f"\n{'-'*80}")
        print(f"CONTEXT: source={source}, clearing={clearing}")
        print(f"         parties={parties}")
        print(f"         {count} payments")
        print(f"{'-'*80}")
        
        ctx_stats = self.context_stats[context_key]
        
        for entity_name in ['cdtrAgt', 'dbtrAgt', 'cdtrAcct', 'dbtrAcct', 'instgAgt']:
            if entity_name not in ctx_stats or ctx_stats[entity_name]['count'] == 0:
                continue
            
            entity_stats = ctx_stats[entity_name]
            
            print(f"\n  {entity_name}:")
            
            if entity_stats['after_fields']:
                # Show required fields in this context
                required = []
                optional = []
                
                for field_name, field_stats in entity_stats['after_fields'].items():
                    if field_stats.is_always_present:
                        required.append(field_name)
                    elif field_stats.presence_rate > 0:
                        optional.append(f"{field_name} ({field_stats.presence_rate:.0f}%)")
                
                if required:
                    print(f"    REQUIRED (100%): {', '.join(required)}")
                if optional:
                    print(f"    OPTIONAL: {', '.join(optional[:5])}")
                
                # Show fields added by repairs in this context
                if entity_stats['before_fields']:
                    added = []
                    for field_name, after_stats in entity_stats['after_fields'].items():
                        before_stats = entity_stats['before_fields'].get(field_name)
                        if not before_stats or before_stats.presence_rate == 0:
                            if after_stats.presence_rate == 100:
                                added.append(field_name)
                    
                    if added:
                        print(f"    ADDED BY REPAIRS: {', '.join(added)}")
    
    def compare_before_after(self, entity_name: str, before_fields: Dict, after_fields: Dict):
        """Compare before and after to find fields added by repairs"""
        print(f"\n  TRANSFORMATION PATTERNS (before → after):")
        
        added_fields = []
        for field_name, after_stats in after_fields.items():
            before_stats = before_fields.get(field_name)
            
            if not before_stats:
                if after_stats.presence_rate >= 80:
                    added_fields.append((field_name, 0, after_stats.presence_rate))
            elif after_stats.presence_rate > before_stats.presence_rate + 10:
                added_fields.append((field_name, before_stats.presence_rate, after_stats.presence_rate))
        
        if added_fields:
            for field_name, before_rate, after_rate in sorted(added_fields, key=lambda x: x[2], reverse=True):
                if before_rate == 0:
                    print(f"    + {field_name}: 0% → {after_rate:.1f}% (NEW)")
                else:
                    print(f"    ↑ {field_name}: {before_rate:.1f}% → {after_rate:.1f}%")
    
    def generate_context_specs(self) -> Dict:
        """Generate context-specific specifications"""
        context_specs = {}
        
        # For each major context
        for context_key, count in self.context_patterns.most_common(10):
            if count < 10:  # Skip rare contexts
                continue
            
            parts = context_key.split('|')
            source = parts[0] if len(parts) > 0 else 'UNKNOWN'
            clearing = parts[1] if len(parts) > 1 else 'UNKNOWN'
            parties_str = parts[2] if len(parts) > 2 else ''
            
            context_spec = {
                'source': source,
                'clearing': clearing,
                'parties': parties_str.split(',') if parties_str else [],
                'payment_count': count,
                'entities': {}
            }
            
            ctx_stats = self.context_stats[context_key]
            
            # For each entity in this context
            for entity_name, entity_stats in ctx_stats.items():
                if entity_stats['count'] == 0:
                    continue
                
                # Use after_fields if available (repaired state)
                field_source = entity_stats['after_fields'] if entity_stats['after_fields'] else entity_stats.get('before_fields', {})
                
                required = []
                optional = []
                
                for field_name, field_stats in field_source.items():
                    if field_stats.is_always_present:
                        required.append(field_name)
                    else:
                        optional.append(field_name)
                
                context_spec['entities'][entity_name] = {
                    'required': sorted(required),
                    'optional': sorted(optional)
                }
            
            context_specs[context_key] = context_spec
        
        return context_specs
    
    def save_context_specs(self, output_file: str):
        """Save context-specific specifications to JSON"""
        specs = self.generate_context_specs()
        
        with open(output_file, 'w') as f:
            json.dump(specs, f, indent=2)
        
        print(f"\n{'='*80}")
        print(f"CONTEXT-SPECIFIC SPECIFICATIONS SAVED: {output_file}")
        print(f"{'='*80}\n")
        
        # Print summary
        print("DISCOVERED CONTEXT-SPECIFIC REQUIREMENTS:\n")
        for context_key, spec in specs.items():
            print(f"Context: source={spec['source']}, clearing={spec['clearing']}")
            print(f"         parties={', '.join(spec['parties'])}")
            print(f"         {spec['payment_count']} payments\n")
            
            for entity_name, entity_spec in spec['entities'].items():
                if entity_spec['required']:
                    print(f"  {entity_name}:")
                    print(f"    Required: {', '.join(entity_spec['required'])}")
            print()


def main():
    parser = argparse.ArgumentParser(description='Context-aware payment data structure analyzer')
    parser.add_argument('--data_dir', required=True, help='Directory with JSON payment files')
    parser.add_argument('--output', default='context_specs.json', help='Output file for context-specific specs')
    
    args = parser.parse_args()
    
    analyzer = ContextAwareAnalyzer()
    analyzer.analyze_directory(args.data_dir)
    analyzer.generate_report()
    analyzer.save_context_specs(args.output)


if __name__ == "__main__":
    main()
