#!/usr/bin/env python3
"""
Payment Data Structure Analyzer
Discovers actual field requirements from real payment data
"""

import json
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Any, Set
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


class DataStructureAnalyzer:
    """Analyzes payment data to discover structure patterns"""
    
    def __init__(self):
        self.entity_stats = defaultdict(lambda: {
            'count': 0,
            'fields': defaultdict(FieldStats),
            'before_fields': defaultdict(FieldStats),
            'after_fields': defaultdict(FieldStats)
        })
        self.total_payments = 0
    
    def analyze_directory(self, data_dir: str):
        """Analyze all JSON files in directory"""
        json_files = list(Path(data_dir).glob('**/*.json'))
        print(f"\n{'='*80}")
        print(f"ANALYZING {len(json_files)} JSON FILES")
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
    
    def analyze_payment(self, payment: Dict):
        """Analyze a single payment"""
        self.total_payments += 1
        
        # Known entities to check
        entities = [
            'cdtr', 'dbtr', 'cdtrAgt', 'dbtrAgt', 'cdtrAcct', 'dbtrAcct',
            'instgAgt', 'instdAgt', 'intrmyAgt1', 'intrmyAgt2', 'intrmyAgt3',
            'rmtInf'
        ]
        
        for entity_name in entities:
            # Check both camelCase and PascalCase
            entity_data = None
            if entity_name in payment:
                entity_data = payment[entity_name]
            elif entity_name.capitalize() in payment:
                entity_data = payment[entity_name.capitalize()]
            elif entity_name[0].upper() + entity_name[1:] in payment:
                entity_data = payment[entity_name[0].upper() + entity_name[1:]]
            
            if entity_data:
                stats = self.entity_stats[entity_name]
                stats['count'] += 1
                
                # Check if it has before/after structure
                if isinstance(entity_data, dict):
                    if 'before' in entity_data and 'after' in entity_data:
                        # Training data format
                        self.analyze_entity_fields(entity_data['before'], stats['before_fields'])
                        self.analyze_entity_fields(entity_data['after'], stats['after_fields'])
                    else:
                        # Direct format (prediction input)
                        self.analyze_entity_fields(entity_data, stats['fields'])
    
    def analyze_entity_fields(self, entity: Any, field_stats: Dict):
        """Recursively analyze entity fields"""
        if not isinstance(entity, dict):
            return
        
        for key, value in entity.items():
            stats = field_stats[key]
            stats.total_count += 1
            
            if value is not None and value != '':
                stats.present_count += 1
                
                # Sample values (keep first 5)
                if len(stats.value_samples) < 5:
                    if isinstance(value, str):
                        stats.value_samples.append(value[:50])  # Truncate long values
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
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            self.analyze_entity_fields(item, field_stats)
    
    def generate_report(self):
        """Generate comprehensive analysis report"""
        print(f"\n{'='*80}")
        print("DATA STRUCTURE ANALYSIS REPORT")
        print(f"{'='*80}")
        print(f"Total payments analyzed: {self.total_payments}\n")
        
        for entity_name, stats in sorted(self.entity_stats.items()):
            if stats['count'] == 0:
                continue
            
            print(f"\n{'='*80}")
            print(f"ENTITY: {entity_name}")
            print(f"{'='*80}")
            print(f"Present in: {stats['count']} / {self.total_payments} payments ({stats['count']/self.total_payments*100:.1f}%)")
            
            # Analyze direct fields
            if stats['fields']:
                print(f"\n  FIELD ANALYSIS (Direct Format):")
                self.print_field_stats(stats['fields'])
            
            # Analyze before/after patterns
            if stats['before_fields']:
                print(f"\n  BEFORE STATE FIELDS:")
                self.print_field_stats(stats['before_fields'], prefix="    ")
            
            if stats['after_fields']:
                print(f"\n  AFTER STATE FIELDS:")
                self.print_field_stats(stats['after_fields'], prefix="    ")
                
                # Show fields added by repairs
                if stats['before_fields']:
                    self.compare_before_after(entity_name, stats['before_fields'], stats['after_fields'])
    
    def print_field_stats(self, field_stats: Dict, prefix: str = "  "):
        """Print statistics for fields"""
        # Sort by presence rate (descending)
        sorted_fields = sorted(
            field_stats.items(),
            key=lambda x: x[1].presence_rate,
            reverse=True
        )
        
        required_fields = []
        common_fields = []
        optional_fields = []
        
        for field_name, stats in sorted_fields:
            if stats.is_always_present:
                required_fields.append(field_name)
            elif stats.is_common:
                common_fields.append(field_name)
            else:
                optional_fields.append(field_name)
        
        if required_fields:
            print(f"{prefix}ALWAYS PRESENT (100%):")
            for field_name in required_fields:
                self.print_field_detail(field_name, field_stats[field_name], prefix + "  ")
        
        if common_fields:
            print(f"{prefix}COMMON (80-99%):")
            for field_name in common_fields:
                self.print_field_detail(field_name, field_stats[field_name], prefix + "  ")
        
        if optional_fields:
            print(f"{prefix}OPTIONAL (<80%):")
            for field_name in optional_fields:
                self.print_field_detail(field_name, field_stats[field_name], prefix + "  ")
    
    def print_field_detail(self, field_name: str, stats: FieldStats, prefix: str):
        """Print detailed field information"""
        # Basic info
        info = f"{prefix}{field_name}: {stats.presence_rate:.1f}% present ({stats.present_count}/{stats.total_count})"
        
        # Data type
        if stats.data_types:
            main_type = stats.data_types.most_common(1)[0][0]
            info += f", type={main_type}"
        
        # Length stats for strings
        if stats.value_lengths:
            min_len = min(stats.value_lengths)
            max_len = max(stats.value_lengths)
            avg_len = sum(stats.value_lengths) / len(stats.value_lengths)
            info += f", len={min_len}-{max_len} (avg {avg_len:.1f})"
        
        # Dash usage
        if stats.has_dashes > 0:
            dash_rate = stats.has_dashes / stats.present_count * 100
            info += f", {dash_rate:.1f}% with dashes"
        
        print(info)
        
        # Sample values
        if stats.value_samples:
            samples = ", ".join([f"'{s}'" for s in stats.value_samples[:3]])
            print(f"{prefix}  Examples: {samples}")
    
    def compare_before_after(self, entity_name: str, before_fields: Dict, after_fields: Dict):
        """Compare before and after to find fields added by repairs"""
        print(f"\n  FIELDS ADDED BY REPAIRS:")
        
        added_fields = []
        for field_name, after_stats in after_fields.items():
            before_stats = before_fields.get(field_name)
            
            if not before_stats:
                # Completely new field
                added_fields.append((field_name, 0, after_stats.presence_rate))
            elif after_stats.presence_rate > before_stats.presence_rate + 5:
                # Significantly more present in after
                added_fields.append((field_name, before_stats.presence_rate, after_stats.presence_rate))
        
        if added_fields:
            for field_name, before_rate, after_rate in sorted(added_fields, key=lambda x: x[2], reverse=True):
                if before_rate == 0:
                    print(f"    + {field_name}: NEW in after ({after_rate:.1f}%)")
                else:
                    print(f"    ↑ {field_name}: {before_rate:.1f}% → {after_rate:.1f}% (increased)")
        else:
            print(f"    (No fields consistently added)")
    
    def generate_discovered_spec(self) -> Dict:
        """Generate a discovered specification based on actual data"""
        discovered_spec = {}
        
        for entity_name, stats in self.entity_stats.items():
            if stats['count'] == 0:
                continue
            
            # Use 'after' fields if available (repaired state), otherwise use direct fields
            field_source = stats['after_fields'] if stats['after_fields'] else stats['fields']
            
            required = []
            optional = []
            
            for field_name, field_stats in field_source.items():
                if field_stats.is_always_present:
                    required.append(field_name)
                else:
                    optional.append(field_name)
            
            discovered_spec[entity_name] = {
                'required': sorted(required),
                'optional': sorted(optional),
                'presence_in_data': f"{stats['count']}/{self.total_payments}"
            }
        
        return discovered_spec
    
    def save_discovered_spec(self, output_file: str):
        """Save discovered specification to JSON"""
        spec = self.generate_discovered_spec()
        
        with open(output_file, 'w') as f:
            json.dump(spec, f, indent=2)
        
        print(f"\n{'='*80}")
        print(f"DISCOVERED SPECIFICATION SAVED: {output_file}")
        print(f"{'='*80}\n")
        
        # Print summary
        print("DISCOVERED REQUIREMENTS SUMMARY:\n")
        for entity_name, entity_spec in sorted(spec.items()):
            print(f"{entity_name}:")
            print(f"  Required: {', '.join(entity_spec['required']) if entity_spec['required'] else 'None'}")
            print(f"  Optional: {', '.join(entity_spec['optional'][:5]) if entity_spec['optional'] else 'None'}")
            if len(entity_spec['optional']) > 5:
                print(f"            ... and {len(entity_spec['optional']) - 5} more")
            print()


def main():
    parser = argparse.ArgumentParser(description='Analyze payment data structure')
    parser.add_argument('--data_dir', required=True, help='Directory with JSON payment files')
    parser.add_argument('--output', default='discovered_spec.json', help='Output file for discovered spec')
    
    args = parser.parse_args()
    
    analyzer = DataStructureAnalyzer()
    analyzer.analyze_directory(args.data_dir)
    analyzer.generate_report()
    analyzer.save_discovered_spec(args.output)


if __name__ == "__main__":
    main()
