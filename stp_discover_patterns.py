#!/usr/bin/env python3
"""
Dynamic Pattern Discovery and Memorization Model
Version: 11.0 - Automatic Pattern Discovery

This model automatically discovers patterns from training data:
- Learns what changes between "before" and "after" states
- Associates transformations with repairs
- No hardcoded rules - everything is learned from data
"""

import json
import pickle
import argparse
import logging
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
from tqdm import tqdm
import hashlib
from difflib import SequenceMatcher

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class DiscoveryConfig:
    """Configuration for pattern discovery model"""
    model_dir: str = "./discovery_model"
    similarity_threshold: float = 0.7  # For matching patterns
    min_pattern_frequency: int = 1  # Minimum occurrences to remember
    max_patterns: int = 10000  # Memory limit
    
    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: str):
        with open(path, 'r') as f:
            return cls(**json.load(f))


# ============================================================================
# PATTERN DISCOVERY ENGINE
# ============================================================================

class PatternDiscovery:
    """Discovers patterns automatically from data"""
    
    @staticmethod
    def extract_state_signature(state: Dict, max_depth: int = 5) -> Dict:
        """Extract structural signature from a state (what exists, what's missing)"""
        signature = {
            'fields': set(),
            'missing_fields': set(),
            'field_values': {},
            'structure': {}
        }
        
        def traverse(obj, path='', depth=0):
            if depth > max_depth:
                return
            
            if isinstance(obj, dict):
                for key, value in obj.items():
                    field_path = f"{path}.{key}" if path else key
                    signature['fields'].add(field_path)
                    
                    # Store actual values for leaf nodes
                    if not isinstance(value, (dict, list)):
                        if value and str(value).strip():
                            signature['field_values'][field_path] = str(value)[:100]
                        else:
                            signature['missing_fields'].add(field_path)
                    
                    # Record structure
                    signature['structure'][field_path] = type(value).__name__
                    
                    # Recurse
                    traverse(value, field_path, depth + 1)
                    
            elif isinstance(obj, list):
                for idx, item in enumerate(obj):
                    field_path = f"{path}[{idx}]"
                    traverse(item, field_path, depth + 1)
        
        traverse(state)
        return signature
    
    @staticmethod
    def discover_transformation(before: Dict, after: Dict) -> Dict:
        """Discover what transformed between before and after states"""
        before_sig = PatternDiscovery.extract_state_signature(before)
        after_sig = PatternDiscovery.extract_state_signature(after)
        
        transformation = {
            'added_fields': list(after_sig['fields'] - before_sig['fields']),
            'removed_fields': list(before_sig['fields'] - after_sig['fields']),
            'value_changes': {},
            'structure_changes': {},
            'filled_fields': []  # Fields that were empty/missing and now have values
        }
        
        # Find value changes
        common_fields = before_sig['fields'] & after_sig['fields']
        for field in common_fields:
            before_val = before_sig['field_values'].get(field, '')
            after_val = after_sig['field_values'].get(field, '')
            
            if before_val != after_val:
                transformation['value_changes'][field] = {
                    'before': before_val[:50],
                    'after': after_val[:50]
                }
            
            # Check if field was filled (empty → has value)
            if field in before_sig['missing_fields'] and field not in after_sig['missing_fields']:
                transformation['filled_fields'].append(field)
        
        # Find structure changes
        for field in common_fields:
            before_type = before_sig['structure'].get(field)
            after_type = after_sig['structure'].get(field)
            if before_type != after_type:
                transformation['structure_changes'][field] = {
                    'before': before_type,
                    'after': after_type
                }
        
        return transformation
    
    @staticmethod
    def create_pattern_fingerprint(state: Dict) -> str:
        """Create a fingerprint for pattern matching"""
        sig = PatternDiscovery.extract_state_signature(state)
        
        # Create a string representation of key characteristics
        fingerprint_parts = []
        
        # Add field presence patterns
        for field in sorted(sig['fields']):
            # Normalize field names (remove indices, lowercase)
            normalized = field.lower().replace('[0]', '').replace('[1]', '')
            fingerprint_parts.append(f"has:{normalized}")
        
        # Add missing value patterns  
        for field in sorted(sig['missing_fields']):
            normalized = field.lower().replace('[0]', '').replace('[1]', '')
            fingerprint_parts.append(f"missing_val:{normalized}")
        
        # Create hash
        fingerprint_str = '|'.join(fingerprint_parts)
        return hashlib.md5(fingerprint_str.encode()).hexdigest()[:16]
    
    @staticmethod
    def calculate_similarity(pattern1: Dict, pattern2: Dict) -> float:
        """Calculate similarity between two patterns"""
        sig1 = pattern1 if 'fields' in pattern1 else PatternDiscovery.extract_state_signature(pattern1)
        sig2 = pattern2 if 'fields' in pattern2 else PatternDiscovery.extract_state_signature(pattern2)
        
        # Compare field sets
        fields1 = sig1['fields']
        fields2 = sig2['fields']
        
        if not fields1 and not fields2:
            return 1.0
        if not fields1 or not fields2:
            return 0.0
        
        # Jaccard similarity for fields
        intersection = len(fields1 & fields2)
        union = len(fields1 | fields2)
        field_similarity = intersection / union if union > 0 else 0
        
        # Compare missing fields
        missing1 = sig1['missing_fields']
        missing2 = sig2['missing_fields']
        
        if missing1 or missing2:
            missing_intersection = len(missing1 & missing2)
            missing_union = len(missing1 | missing2)
            missing_similarity = missing_intersection / missing_union if missing_union > 0 else 0
        else:
            missing_similarity = 1.0
        
        # Weighted average
        return 0.7 * field_similarity + 0.3 * missing_similarity


# ============================================================================
# DYNAMIC MEMORIZATION MODEL
# ============================================================================

class DynamicMemorizationModel:
    """Model that discovers and memorizes patterns automatically"""
    
    def __init__(self, config: DiscoveryConfig):
        self.config = config
        self.discovery = PatternDiscovery()
        
        # Core memory: fingerprint → (before_pattern, transformation, repairs)
        self.memory = {}
        
        # Statistics
        self.repair_stats = Counter()
        self.transformation_stats = defaultdict(Counter)
        self.discovered_patterns = defaultdict(list)
        
    def train(self, data_dir: str):
        """Train by discovering and memorizing patterns"""
        logger.info("="*70)
        logger.info("DYNAMIC PATTERN DISCOVERY")
        logger.info("="*70)
        
        json_files = list(Path(data_dir).glob('**/*.json'))
        logger.info(f"Discovering patterns from {len(json_files)} files")
        
        patterns_discovered = 0
        transactions_processed = 0
        
        for json_file in tqdm(json_files, desc="Discovering patterns"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                transactions = self._extract_transactions(data)
                
                for txn_id, txn_data in transactions.items():
                    if not isinstance(txn_data, dict):
                        continue
                    
                    # Skip if no repairs
                    repairs = txn_data.get('ace', [])
                    if not repairs:
                        continue
                    
                    # Extract repair IDs
                    repair_ids = []
                    for repair in repairs:
                        if isinstance(repair, dict):
                            repair_id = str(repair.get('id', 'unknown'))
                        else:
                            repair_id = str(repair)
                        repair_ids.append(repair_id)
                        self.repair_stats[repair_id] += 1
                    
                    # Process each entity with before/after
                    entities_with_changes = []
                    
                    for entity_name, entity_data in txn_data.items():
                        if not isinstance(entity_data, dict):
                            continue
                        
                        if 'before' in entity_data and 'after' in entity_data:
                            before = entity_data['before']
                            after = entity_data['after']
                            
                            if before and after:
                                # Discover transformation
                                transformation = self.discovery.discover_transformation(before, after)
                                
                                # If transformation is significant, memorize it
                                if (transformation['added_fields'] or 
                                    transformation['value_changes'] or 
                                    transformation['filled_fields']):
                                    
                                    # Create fingerprint of before state
                                    fingerprint = self.discovery.create_pattern_fingerprint(before)
                                    
                                    # Memorize: before pattern → transformation + repairs
                                    if fingerprint not in self.memory:
                                        self.memory[fingerprint] = []
                                    
                                    self.memory[fingerprint].append({
                                        'entity': entity_name,
                                        'before_pattern': self.discovery.extract_state_signature(before),
                                        'transformation': transformation,
                                        'repairs': repair_ids,
                                        'source': txn_data.get('source'),
                                        'clearing': txn_data.get('clearing')
                                    })
                                    
                                    patterns_discovered += 1
                                    
                                    # Track transformation statistics
                                    for field in transformation['added_fields']:
                                        self.transformation_stats['added'][field] += 1
                                    for field in transformation['filled_fields']:
                                        self.transformation_stats['filled'][field] += 1
                                    
                                    # Track pattern associations
                                    for repair_id in repair_ids:
                                        pattern_desc = self._describe_transformation(transformation)
                                        self.discovered_patterns[repair_id].append(pattern_desc)
                    
                    transactions_processed += 1
                    
                    # Memory limit
                    if len(self.memory) > self.config.max_patterns:
                        # Keep most frequent patterns
                        self._prune_memory()
                        
            except Exception as e:
                logger.warning(f"Error processing {json_file}: {e}")
        
        # Analyze discovered patterns
        self._analyze_discoveries()
        
        logger.info(f"\nDiscovery complete!")
        logger.info(f"Transactions processed: {transactions_processed}")
        logger.info(f"Unique patterns discovered: {len(self.memory)}")
        logger.info(f"Total pattern instances: {patterns_discovered}")
    
    def predict(self, payment_data: Dict) -> Dict:
        """Predict repairs based on discovered patterns"""
        predictions = {
            'matched_patterns': [],
            'suggested_transformations': [],
            'predicted_repairs': Counter(),
            'ace': []
        }
        
        # Check each entity in the payment
        for entity_name, entity_data in payment_data.items():
            if not isinstance(entity_data, dict):
                continue
            
            # Get the current state (handle before/after or direct data)
            if 'before' in entity_data:
                current_state = entity_data['before']
            elif 'after' not in entity_data:
                current_state = entity_data
            else:
                continue
            
            if not current_state:
                continue
            
            # Find matching patterns in memory
            fingerprint = self.discovery.create_pattern_fingerprint(current_state)
            
            # Exact match
            if fingerprint in self.memory:
                for memory_item in self.memory[fingerprint]:
                    predictions['matched_patterns'].append({
                        'entity': entity_name,
                        'match_type': 'exact',
                        'repairs': memory_item['repairs']
                    })
                    
                    # Add repairs with confidence
                    for repair_id in memory_item['repairs']:
                        predictions['predicted_repairs'][repair_id] += 1.0
                    
                    # Suggest transformation
                    predictions['suggested_transformations'].append({
                        'entity': entity_name,
                        'transformation': memory_item['transformation']
                    })
            
            # Similarity-based matching
            else:
                current_sig = self.discovery.extract_state_signature(current_state)
                best_matches = []
                
                for mem_fingerprint, memory_items in self.memory.items():
                    for memory_item in memory_items:
                        if memory_item['entity'] != entity_name:
                            continue
                        
                        similarity = self.discovery.calculate_similarity(
                            current_sig, 
                            memory_item['before_pattern']
                        )
                        
                        if similarity >= self.config.similarity_threshold:
                            best_matches.append((similarity, memory_item))
                
                # Use top matches
                best_matches.sort(key=lambda x: x[0], reverse=True)
                for similarity, memory_item in best_matches[:3]:
                    predictions['matched_patterns'].append({
                        'entity': entity_name,
                        'match_type': 'similar',
                        'similarity': similarity,
                        'repairs': memory_item['repairs']
                    })
                    
                    # Add repairs weighted by similarity
                    for repair_id in memory_item['repairs']:
                        predictions['predicted_repairs'][repair_id] += similarity
        
        # Create final ACE array from top predicted repairs
        for repair_id, score in predictions['predicted_repairs'].most_common():
            if score >= 0.5:  # Threshold
                predictions['ace'].append({
                    'id': repair_id,
                    'confidence': min(score, 1.0),
                    'code': 'I',
                    'field': 'AUTO_DISCOVERED'
                })
        
        # Add summary
        predictions['summary'] = {
            'total_matches': len(predictions['matched_patterns']),
            'unique_repairs_predicted': len(predictions['predicted_repairs']),
            'final_repairs': len(predictions['ace'])
        }
        
        return predictions
    
    def _describe_transformation(self, transformation: Dict) -> str:
        """Create human-readable description of transformation"""
        parts = []
        
        if transformation['added_fields']:
            # Extract field names without paths
            field_names = [f.split('.')[-1] for f in transformation['added_fields'][:3]]
            parts.append(f"added:{','.join(field_names)}")
        
        if transformation['filled_fields']:
            field_names = [f.split('.')[-1] for f in transformation['filled_fields'][:3]]
            parts.append(f"filled:{','.join(field_names)}")
        
        if transformation['value_changes']:
            parts.append(f"changed:{len(transformation['value_changes'])}_values")
        
        return '|'.join(parts) if parts else 'no_change'
    
    def _analyze_discoveries(self):
        """Analyze and print discovered patterns"""
        logger.info("\n" + "="*50)
        logger.info("DISCOVERED PATTERNS ANALYSIS")
        logger.info("="*50)
        
        # Top repairs
        logger.info("\nTop 10 Discovered Repairs:")
        for repair_id, count in self.repair_stats.most_common(10):
            logger.info(f"  {repair_id}: {count} occurrences")
            
            # Show pattern examples for this repair
            if repair_id in self.discovered_patterns:
                pattern_counts = Counter(self.discovered_patterns[repair_id])
                top_pattern = pattern_counts.most_common(1)[0]
                logger.info(f"    Common pattern: {top_pattern[0]}")
        
        # Common transformations
        logger.info("\nMost Common Field Additions:")
        for field, count in self.transformation_stats['added'].most_common(10):
            field_name = field.split('.')[-1]
            logger.info(f"  {field_name}: added {count} times")
        
        logger.info("\nMost Common Field Fills:")
        for field, count in self.transformation_stats['filled'].most_common(10):
            field_name = field.split('.')[-1]
            logger.info(f"  {field_name}: filled {count} times")
        
        # Pattern diversity
        logger.info(f"\nPattern Statistics:")
        logger.info(f"  Unique fingerprints: {len(self.memory)}")
        total_instances = sum(len(items) for items in self.memory.values())
        logger.info(f"  Total memorized instances: {total_instances}")
        
        # Find patterns that lead to specific repairs
        repair_patterns = defaultdict(set)
        for fingerprint, memory_items in self.memory.items():
            for item in memory_items:
                for repair_id in item['repairs']:
                    pattern_desc = f"{item['entity']}:{self._describe_transformation(item['transformation'])}"
                    repair_patterns[repair_id].add(pattern_desc)
        
        logger.info("\nRepair → Pattern Mappings (top 5):")
        for repair_id in ['6021', '8852', '6001', '8001', '7743']:
            if repair_id in repair_patterns:
                patterns = list(repair_patterns[repair_id])[:2]
                logger.info(f"  {repair_id}: {patterns}")
    
    def _prune_memory(self):
        """Prune memory to stay within limits"""
        # Keep patterns with multiple instances
        pruned = {}
        for fingerprint, items in self.memory.items():
            if len(items) >= self.config.min_pattern_frequency:
                pruned[fingerprint] = items[-100:]  # Keep last 100 instances
        
        self.memory = pruned
        logger.debug(f"Pruned memory to {len(self.memory)} patterns")
    
    def _extract_transactions(self, data: Any) -> Dict:
        """Extract transactions from various formats"""
        transactions = {}
        
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict):
                    transactions[str(key)] = value
                    
        elif isinstance(data, list):
            for idx, item in enumerate(data):
                if isinstance(item, dict):
                    if len(item) == 1:
                        txn_id = list(item.keys())[0]
                        transactions[str(txn_id)] = item[txn_id]
                    else:
                        transactions[f"txn_{idx}"] = item
        
        return transactions
    
    def save(self, model_dir: str):
        """Save discovered patterns"""
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        
        # Save config
        self.config.save(Path(model_dir) / 'config.json')
        
        # Save discovered patterns
        model_data = {
            'memory': self.memory,
            'repair_stats': dict(self.repair_stats),
            'transformation_stats': dict(self.transformation_stats),
            'discovered_patterns': dict(self.discovered_patterns)
        }
        
        with open(Path(model_dir) / 'patterns.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {model_dir}")
    
    @classmethod
    def load(cls, model_dir: str):
        """Load discovered patterns"""
        config = DiscoveryConfig.load(Path(model_dir) / 'config.json')
        model = cls(config)
        
        with open(Path(model_dir) / 'patterns.pkl', 'rb') as f:
            model_data = pickle.load(f)
        
        model.memory = model_data['memory']
        model.repair_stats = Counter(model_data['repair_stats'])
        model.transformation_stats = defaultdict(Counter, model_data['transformation_stats'])
        model.discovered_patterns = defaultdict(list, model_data['discovered_patterns'])
        
        logger.info(f"Loaded model with {len(model.memory)} discovered patterns")
        return model


# ============================================================================
# MAIN CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Dynamic Pattern Discovery Model')
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Discover patterns from data')
    train_parser.add_argument('--data_dir', required=True, help='Directory with training JSON files')
    train_parser.add_argument('--model_dir', default='./discovery_model', help='Where to save model')
    
    # Predict command  
    predict_parser = subparsers.add_parser('predict', help='Predict repairs')
    predict_parser.add_argument('--input', required=True, help='Input JSON file')
    predict_parser.add_argument('--model_dir', default='./discovery_model', help='Model directory')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a payment')
    analyze_parser.add_argument('--input', required=True, help='Input JSON file')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        config = DiscoveryConfig(model_dir=args.model_dir)
        model = DynamicMemorizationModel(config)
        model.train(args.data_dir)
        model.save(args.model_dir)
        
    elif args.command == 'predict':
        model = DynamicMemorizationModel.load(args.model_dir)
        
        # Load payment
        with open(args.input, 'r') as f:
            data = json.load(f)
        
        # Extract payment
        if isinstance(data, dict) and len(data) == 1:
            payment = data[list(data.keys())[0]]
        else:
            payment = data
        
        # Predict
        results = model.predict(payment)
        
        # Print results
        print("\n" + "="*70)
        print("PREDICTION RESULTS")
        print("="*70)
        
        print(f"\nMatched Patterns ({len(results['matched_patterns'])}):")
        for match in results['matched_patterns'][:5]:
            print(f"  Entity: {match['entity']}")
            print(f"  Match Type: {match['match_type']}")
            if 'similarity' in match:
                print(f"  Similarity: {match['similarity']:.1%}")
            print(f"  Suggested Repairs: {match['repairs']}")
        
        print(f"\nPredicted Repairs ({len(results['ace'])}):")
        for repair in results['ace']:
            print(f"  ID: {repair['id']}, Confidence: {repair['confidence']:.1%}")
        
        print(f"\nSummary:")
        for key, value in results['summary'].items():
            print(f"  {key}: {value}")
    
    elif args.command == 'analyze':
        discovery = PatternDiscovery()
        
        with open(args.input, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, dict) and len(data) == 1:
            payment = data[list(data.keys())[0]]
        else:
            payment = data
        
        print("\n" + "="*70)
        print("PAYMENT ANALYSIS")
        print("="*70)
        
        # Analyze each entity
        for entity_name, entity_data in payment.items():
            if not isinstance(entity_data, dict):
                continue
            
            if 'before' in entity_data and 'after' in entity_data:
                print(f"\nEntity: {entity_name}")
                
                transformation = discovery.discover_transformation(
                    entity_data['before'], 
                    entity_data['after']
                )
                
                print("  Transformation discovered:")
                if transformation['added_fields']:
                    print(f"    Added fields: {transformation['added_fields'][:5]}")
                if transformation['filled_fields']:
                    print(f"    Filled fields: {transformation['filled_fields'][:5]}")
                if transformation['value_changes']:
                    print(f"    Value changes: {len(transformation['value_changes'])} fields")
                
                # Show fingerprint
                fingerprint = discovery.create_pattern_fingerprint(entity_data['before'])
                print(f"    Before fingerprint: {fingerprint}")
        
        if 'ace' in payment:
            print("\nActual Repairs:")
            for repair in payment['ace'][:10]:
                if isinstance(repair, dict):
                    print(f"  ID: {repair.get('id')}")
                else:
                    print(f"  {repair}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
