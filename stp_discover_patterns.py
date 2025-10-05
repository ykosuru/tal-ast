#!/usr/bin/env python3
"""
Context-Aware Pattern Discovery and Memorization Model
Version: 12.0 - Includes Source, Clearing, and Parties Context

This model discovers patterns with full context awareness:
- Network context: source + clearing
- Party context: which parties are present
- Field-to-party mapping: ace.field → required party
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
# FIELD-TO-PARTY MAPPING
# ============================================================================

# Maps ACE field codes to required party flags
FIELD_PARTY_MAPPING = {
    'CDTPTY': 'bnf_pty',      # Credit party = beneficiary
    'BNFBNK': 'bnf_pty',      # Beneficiary bank
    'CDTAGT': 'bnf_pty',      # Credit agent
    'DBTRAGT': 'bbk_pty',     # Debtor agent = ordering bank
    'DBTR': 'org_pty',        # Debtor = originator
    'ORDPTY': 'org_pty',      # Ordering party
    'INTAGT': 'ibk_pty',      # Intermediary agent
    'INTRMY': 'crdt_agt_intrmy_pty',  # Intermediary
}

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class DiscoveryConfig:
    """Configuration for pattern discovery model"""
    model_dir: str = "./discovery_model"
    similarity_threshold: float = 0.7
    min_pattern_frequency: int = 1
    max_patterns: int = 10000
    use_context: bool = True  # Enable context-aware matching
    
    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: str):
        with open(path, 'r') as f:
            return cls(**json.load(f))


# ============================================================================
# CONTEXT EXTRACTOR
# ============================================================================

class ContextExtractor:
    """Extracts context information from transactions"""
    
    @staticmethod
    def extract_network_context(txn_data: Dict) -> Dict:
        """Extract source and clearing (network context)"""
        return {
            'source': txn_data.get('source', ''),
            'clearing': txn_data.get('clearing', '')
        }
    
    @staticmethod
    def extract_party_context(txn_data: Dict) -> Dict:
        """Extract parties information"""
        parties = txn_data.get('parties', {})
        
        # Convert to canonical format
        party_context = {
            'bbk_pty': parties.get('bbk_pty', False),
            'bnf_pty': parties.get('bnf_pty', False),
            'crdt_agt_intrmy_pty': parties.get('crdt_agt_intrmy_pty', False),
            'crdt_agt_pty': parties.get('crdt_agt_pty', False),
            'dbtr_agt_pty': parties.get('dbtr_agt_pty', False),
            'ibk_pty': parties.get('ibk_pty', False),
            'ogb_pty': parties.get('ogb_pty', False),
            'org_pty': parties.get('org_pty', False)
        }
        
        return party_context
    
    @staticmethod
    def create_context_signature(network: Dict, parties: Dict) -> str:
        """Create a signature for context matching"""
        # Format: source|clearing|party1|party2|...
        parts = [
            network.get('source', ''),
            network.get('clearing', '')
        ]
        
        # Add present parties in sorted order
        present_parties = sorted([k for k, v in parties.items() if v])
        parts.extend(present_parties)
        
        return '|'.join(parts)
    
    @staticmethod
    def check_repair_context_match(repair: Dict, parties: Dict) -> bool:
        """Check if repair's required party is present"""
        field = repair.get('field', '')
        
        # Get required party for this field
        required_party = FIELD_PARTY_MAPPING.get(field)
        
        if required_party:
            # Check if required party is present
            return parties.get(required_party, False)
        
        # If no specific party required, allow
        return True


# ============================================================================
# PATTERN DISCOVERY ENGINE
# ============================================================================

class PatternDiscovery:
    """Discovers patterns automatically from data"""
    
    @staticmethod
    def extract_state_signature(state: Dict, max_depth: int = 5) -> Dict:
        """Extract structural signature from a state"""
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
                    
                    if not isinstance(value, (dict, list)):
                        if value and str(value).strip():
                            signature['field_values'][field_path] = str(value)[:100]
                        else:
                            signature['missing_fields'].add(field_path)
                    
                    signature['structure'][field_path] = type(value).__name__
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
            'filled_fields': []
        }
        
        common_fields = before_sig['fields'] & after_sig['fields']
        for field in common_fields:
            before_val = before_sig['field_values'].get(field, '')
            after_val = after_sig['field_values'].get(field, '')
            
            if before_val != after_val:
                transformation['value_changes'][field] = {
                    'before': before_val[:50],
                    'after': after_val[:50]
                }
            
            if field in before_sig['missing_fields'] and field not in after_sig['missing_fields']:
                transformation['filled_fields'].append(field)
        
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
    def create_pattern_fingerprint(state: Dict, context_sig: str = "") -> str:
        """Create a fingerprint for pattern matching (includes context)"""
        sig = PatternDiscovery.extract_state_signature(state)
        
        fingerprint_parts = []
        
        # Add context signature first
        if context_sig:
            fingerprint_parts.append(f"ctx:{context_sig}")
        
        # Add field presence patterns
        for field in sorted(sig['fields']):
            normalized = field.lower().replace('[0]', '').replace('[1]', '')
            fingerprint_parts.append(f"has:{normalized}")
        
        # Add missing value patterns  
        for field in sorted(sig['missing_fields']):
            normalized = field.lower().replace('[0]', '').replace('[1]', '')
            fingerprint_parts.append(f"missing_val:{normalized}")
        
        fingerprint_str = '|'.join(fingerprint_parts)
        return hashlib.md5(fingerprint_str.encode()).hexdigest()[:16]
    
    @staticmethod
    def calculate_similarity(pattern1: Dict, pattern2: Dict) -> float:
        """Calculate similarity between two patterns"""
        sig1 = pattern1 if 'fields' in pattern1 else PatternDiscovery.extract_state_signature(pattern1)
        sig2 = pattern2 if 'fields' in pattern2 else PatternDiscovery.extract_state_signature(pattern2)
        
        fields1 = sig1['fields']
        fields2 = sig2['fields']
        
        if not fields1 and not fields2:
            return 1.0
        if not fields1 or not fields2:
            return 0.0
        
        intersection = len(fields1 & fields2)
        union = len(fields1 | fields2)
        field_similarity = intersection / union if union > 0 else 0
        
        missing1 = sig1['missing_fields']
        missing2 = sig2['missing_fields']
        
        if missing1 or missing2:
            missing_intersection = len(missing1 & missing2)
            missing_union = len(missing1 | missing2)
            missing_similarity = missing_intersection / missing_union if missing_union > 0 else 0
        else:
            missing_similarity = 1.0
        
        return 0.7 * field_similarity + 0.3 * missing_similarity


# ============================================================================
# CONTEXT-AWARE MEMORIZATION MODEL
# ============================================================================

class DynamicMemorizationModel:
    """Model that discovers and memorizes patterns with context"""
    
    def __init__(self, config: DiscoveryConfig):
        self.config = config
        self.discovery = PatternDiscovery()
        self.context_extractor = ContextExtractor()
        
        # Core memory: fingerprint → (context, before_pattern, transformation, repairs)
        self.memory = {}
        
        # Context-indexed memory for fast lookup
        self.context_index = defaultdict(list)  # context_sig → [fingerprints]
        
        # Statistics
        self.repair_stats = Counter()
        self.repair_context_stats = defaultdict(lambda: {'networks': Counter(), 'parties': Counter()})
        self.transformation_stats = defaultdict(Counter)
        self.discovered_patterns = defaultdict(list)
        
    def train(self, data_dir: str):
        """Train by discovering and memorizing patterns with context"""
        logger.info("="*70)
        logger.info("CONTEXT-AWARE PATTERN DISCOVERY")
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
                    
                    repairs = txn_data.get('ace', [])
                    if not repairs:
                        continue
                    
                    # Extract context
                    network_context = self.context_extractor.extract_network_context(txn_data)
                    party_context = self.context_extractor.extract_party_context(txn_data)
                    context_sig = self.context_extractor.create_context_signature(
                        network_context, party_context
                    )
                    
                    # Extract repair IDs
                    repair_ids = []
                    for repair in repairs:
                        if isinstance(repair, dict):
                            repair_id = str(repair.get('id', 'unknown'))
                            repair_field = repair.get('field', '')
                            
                            # Validate repair against context
                            if self.context_extractor.check_repair_context_match(repair, party_context):
                                repair_ids.append((repair_id, repair_field))
                                self.repair_stats[repair_id] += 1
                                
                                # Track context for this repair
                                network_key = f"{network_context['source']}|{network_context['clearing']}"
                                self.repair_context_stats[repair_id]['networks'][network_key] += 1
                                
                                # Track which parties this repair appears with
                                present_parties = tuple(sorted([k for k, v in party_context.items() if v]))
                                self.repair_context_stats[repair_id]['parties'][present_parties] += 1
                        else:
                            repair_id = str(repair)
                            repair_ids.append((repair_id, ''))
                    
                    if not repair_ids:
                        continue
                    
                    # Process entities with transformations
                    for entity_name, entity_data in txn_data.items():
                        if not isinstance(entity_data, dict):
                            continue
                        
                        if 'before' in entity_data and 'after' in entity_data:
                            before = entity_data['before']
                            after = entity_data['after']
                            
                            if before and after:
                                transformation = self.discovery.discover_transformation(before, after)
                                
                                if (transformation['added_fields'] or 
                                    transformation['value_changes'] or 
                                    transformation['filled_fields']):
                                    
                                    # Create context-aware fingerprint
                                    if self.config.use_context:
                                        fingerprint = self.discovery.create_pattern_fingerprint(
                                            before, context_sig
                                        )
                                    else:
                                        fingerprint = self.discovery.create_pattern_fingerprint(before)
                                    
                                    # Memorize with context
                                    if fingerprint not in self.memory:
                                        self.memory[fingerprint] = []
                                    
                                    memory_item = {
                                        'entity': entity_name,
                                        'before_pattern': self.discovery.extract_state_signature(before),
                                        'transformation': transformation,
                                        'repairs': repair_ids,
                                        'network_context': network_context,
                                        'party_context': party_context,
                                        'context_signature': context_sig
                                    }
                                    
                                    self.memory[fingerprint].append(memory_item)
                                    self.context_index[context_sig].append(fingerprint)
                                    
                                    patterns_discovered += 1
                                    
                                    # Track statistics
                                    for field in transformation['added_fields']:
                                        self.transformation_stats['added'][field] += 1
                                    for field in transformation['filled_fields']:
                                        self.transformation_stats['filled'][field] += 1
                                    
                                    for repair_id, repair_field in repair_ids:
                                        pattern_desc = self._describe_transformation(transformation)
                                        self.discovered_patterns[repair_id].append(pattern_desc)
                    
                    transactions_processed += 1
                    
                    if len(self.memory) > self.config.max_patterns:
                        self._prune_memory()
                        
            except Exception as e:
                logger.warning(f"Error processing {json_file}: {e}")
        
        self._analyze_discoveries()
        
        logger.info(f"\nDiscovery complete!")
        logger.info(f"Transactions processed: {transactions_processed}")
        logger.info(f"Unique patterns discovered: {len(self.memory)}")
        logger.info(f"Total pattern instances: {patterns_discovered}")
        logger.info(f"Unique contexts: {len(self.context_index)}")
    
    def predict(self, payment_data: Dict) -> Dict:
        """Predict repairs based on discovered patterns with context awareness"""
        predictions = {
            'matched_patterns': [],
            'suggested_transformations': [],
            'predicted_repairs': Counter(),
            'context_filters_applied': {},
            'ace': []
        }
        
        # Extract context from input payment
        network_context = self.context_extractor.extract_network_context(payment_data)
        party_context = self.context_extractor.extract_party_context(payment_data)
        context_sig = self.context_extractor.create_context_signature(
            network_context, party_context
        )
        
        predictions['context_filters_applied'] = {
            'network': network_context,
            'parties': party_context,
            'signature': context_sig
        }
        
        # Check each entity in the payment
        for entity_name, entity_data in payment_data.items():
            if not isinstance(entity_data, dict):
                continue
            
            # Skip metadata fields
            if entity_name in ['source', 'clearing', 'parties', 'flags', 'ace']:
                continue
            
            # For prediction input, entities are direct (no 'before' wrapper)
            # For training data, they have 'before' and 'after'
            if 'before' in entity_data:
                current_state = entity_data['before']
            else:
                # Direct entity data (prediction input format)
                current_state = entity_data
            
            if not current_state:
                continue
            
            # Create context-aware fingerprint
            if self.config.use_context:
                fingerprint = self.discovery.create_pattern_fingerprint(
                    current_state, context_sig
                )
            else:
                fingerprint = self.discovery.create_pattern_fingerprint(current_state)
            
            # Exact match (with context)
            if fingerprint in self.memory:
                for memory_item in self.memory[fingerprint]:
                    # Verify context match
                    if self._contexts_match(network_context, party_context, memory_item):
                        predictions['matched_patterns'].append({
                            'entity': entity_name,
                            'match_type': 'exact',
                            'repairs': memory_item['repairs'],
                            'context_match': 'exact'
                        })
                        
                        # Add repairs with confidence, filtering by party requirements
                        for repair_id, repair_field in memory_item['repairs']:
                            # Check if this repair makes sense for current parties
                            if self._repair_valid_for_parties(repair_field, party_context):
                                predictions['predicted_repairs'][repair_id] += 1.0
                        
                        predictions['suggested_transformations'].append({
                            'entity': entity_name,
                            'transformation': memory_item['transformation']
                        })
            
            # Context-aware similarity matching
            else:
                current_sig = self.discovery.extract_state_signature(current_state)
                best_matches = []
                
                # Only search patterns from same context
                if context_sig in self.context_index:
                    candidate_fingerprints = self.context_index[context_sig]
                else:
                    # Fall back to all patterns if no exact context match
                    candidate_fingerprints = self.memory.keys()
                
                for mem_fingerprint in candidate_fingerprints:
                    for memory_item in self.memory.get(mem_fingerprint, []):
                        if memory_item['entity'] != entity_name:
                            continue
                        
                        # Check context compatibility
                        if not self._contexts_match(network_context, party_context, memory_item):
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
                        'repairs': memory_item['repairs'],
                        'context_match': 'compatible'
                    })
                    
                    # Add repairs weighted by similarity, with party filtering
                    for repair_id, repair_field in memory_item['repairs']:
                        if self._repair_valid_for_parties(repair_field, party_context):
                            predictions['predicted_repairs'][repair_id] += similarity
        
        # Create final ACE array from top predicted repairs
        for repair_id, score in predictions['predicted_repairs'].most_common():
            if score >= 0.5:
                predictions['ace'].append({
                    'id': repair_id,
                    'confidence': min(score, 1.0),
                    'code': 'I',
                    'field': 'AUTO_DISCOVERED'
                })
        
        predictions['summary'] = {
            'total_matches': len(predictions['matched_patterns']),
            'unique_repairs_predicted': len(predictions['predicted_repairs']),
            'final_repairs': len(predictions['ace']),
            'context_used': self.config.use_context
        }
        
        return predictions
    
    def _contexts_match(self, network: Dict, parties: Dict, memory_item: Dict) -> bool:
        """Check if contexts are compatible"""
        # Network must match exactly
        mem_network = memory_item['network_context']
        if network.get('source') != mem_network.get('source'):
            return False
        if network.get('clearing') != mem_network.get('clearing'):
            return False
        
        # Parties should overlap significantly
        mem_parties = memory_item['party_context']
        
        # Check key parties match
        important_parties = ['bnf_pty', 'bbk_pty', 'org_pty']
        for party in important_parties:
            if parties.get(party) != mem_parties.get(party):
                # If one has it and other doesn't, not compatible
                if parties.get(party) or mem_parties.get(party):
                    return False
        
        return True
    
    def _repair_valid_for_parties(self, repair_field: str, parties: Dict) -> bool:
        """Check if repair is valid given current parties"""
        required_party = FIELD_PARTY_MAPPING.get(repair_field)
        
        if required_party:
            return parties.get(required_party, False)
        
        return True
    
    def _describe_transformation(self, transformation: Dict) -> str:
        """Create human-readable description of transformation"""
        parts = []
        
        if transformation['added_fields']:
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
        logger.info("CONTEXT-AWARE PATTERNS ANALYSIS")
        logger.info("="*50)
        
        logger.info("\nTop 10 Discovered Repairs:")
        for repair_id, count in self.repair_stats.most_common(10):
            logger.info(f"  {repair_id}: {count} occurrences")
            
            # Show context for this repair
            if repair_id in self.repair_context_stats:
                ctx_stats = self.repair_context_stats[repair_id]
                
                top_network = ctx_stats['networks'].most_common(1)
                if top_network:
                    logger.info(f"    Most common network: {top_network[0][0]} ({top_network[0][1]} times)")
                
                top_parties = ctx_stats['parties'].most_common(1)
                if top_parties:
                    party_list = ', '.join([p for p in top_parties[0][0] if 'pty' in p])
                    logger.info(f"    Common parties: {party_list}")
            
            if repair_id in self.discovered_patterns:
                pattern_counts = Counter(self.discovered_patterns[repair_id])
                top_pattern = pattern_counts.most_common(1)[0]
                logger.info(f"    Common pattern: {top_pattern[0]}")
        
        logger.info("\nMost Common Field Additions:")
        for field, count in self.transformation_stats['added'].most_common(10):
            field_name = field.split('.')[-1]
            logger.info(f"  {field_name}: added {count} times")
        
        logger.info(f"\nPattern Statistics:")
        logger.info(f"  Unique fingerprints: {len(self.memory)}")
        logger.info(f"  Unique contexts: {len(self.context_index)}")
        total_instances = sum(len(items) for items in self.memory.values())
        logger.info(f"  Total memorized instances: {total_instances}")
    
    def _prune_memory(self):
        """Prune memory to stay within limits"""
        pruned = {}
        for fingerprint, items in self.memory.items():
            if len(items) >= self.config.min_pattern_frequency:
                pruned[fingerprint] = items[-100:]
        
        self.memory = pruned
        
        # Rebuild context index
        self.context_index.clear()
        for fingerprint, items in self.memory.items():
            for item in items:
                context_sig = item['context_signature']
                self.context_index[context_sig].append(fingerprint)
        
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
        
        self.config.save(Path(model_dir) / 'config.json')
        
        model_data = {
            'memory': self.memory,
            'context_index': dict(self.context_index),
            'repair_stats': dict(self.repair_stats),
            'repair_context_stats': dict(self.repair_context_stats),
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
        model.context_index = defaultdict(list, model_data.get('context_index', {}))
        model.repair_stats = Counter(model_data['repair_stats'])
        model.repair_context_stats = defaultdict(
            lambda: {'networks': Counter(), 'parties': Counter()}, 
            model_data.get('repair_context_stats', {})
        )
        model.transformation_stats = defaultdict(Counter, model_data['transformation_stats'])
        model.discovered_patterns = defaultdict(list, model_data['discovered_patterns'])
        
        logger.info(f"Loaded model with {len(model.memory)} patterns across {len(model.context_index)} contexts")
        return model


# ============================================================================
# MAIN CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Context-Aware Pattern Discovery Model')
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Discover patterns from data')
    train_parser.add_argument('--data_dir', required=True, help='Directory with training JSON files')
    train_parser.add_argument('--model_dir', default='./discovery_model', help='Where to save model')
    train_parser.add_argument('--no-context', action='store_true', help='Disable context-aware matching')
    
    # Predict command  
    predict_parser = subparsers.add_parser('predict', help='Predict repairs')
    predict_parser.add_argument('--input', required=True, help='Input JSON file')
    predict_parser.add_argument('--model_dir', default='./discovery_model', help='Model directory')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a payment')
    analyze_parser.add_argument('--input', required=True, help='Input JSON file')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        config = DiscoveryConfig(
            model_dir=args.model_dir,
            use_context=not args.no_context
        )
        model = DynamicMemorizationModel(config)
        model.train(args.data_dir)
        model.save(args.model_dir)
        
    elif args.command == 'predict':
        model = DynamicMemorizationModel.load(args.model_dir)
        
        with open(args.input, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, dict) and len(data) == 1:
            payment = data[list(data.keys())[0]]
        else:
            payment = data
        
        results = model.predict(payment)
        
        print("\n" + "="*70)
        print("CONTEXT-AWARE PREDICTION RESULTS")
        print("="*70)
        
        print(f"\nContext Applied:")
        print(f"  Network: {results['context_filters_applied']['network']}")
        print(f"  Parties: {[k for k,v in results['context_filters_applied']['parties'].items() if v]}")
        
        print(f"\nMatched Patterns ({len(results['matched_patterns'])}):")
        for match in results['matched_patterns'][:5]:
            print(f"  Entity: {match['entity']}")
            print(f"  Match Type: {match['match_type']} (context: {match.get('context_match', 'N/A')})")
            if 'similarity' in match:
                print(f"  Similarity: {match['similarity']:.1%}")
            repairs_str = ', '.join([f"{r[0]}" for r in match['repairs'][:3]])
            print(f"  Suggested Repairs: {repairs_str}")
        
        print(f"\nPredicted Repairs ({len(results['ace'])}):")
        for repair in results['ace']:
            print(f"  ID: {repair['id']}, Confidence: {repair['confidence']:.1%}")
        
        print(f"\nSummary:")
        for key, value in results['summary'].items():
            print(f"  {key}: {value}")
    
    elif args.command == 'analyze':
        discovery = PatternDiscovery()
        context_extractor = ContextExtractor()
        
        with open(args.input, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, dict) and len(data) == 1:
            payment = data[list(data.keys())[0]]
        else:
            payment = data
        
        print("\n" + "="*70)
        print("PAYMENT CONTEXT ANALYSIS")
        print("="*70)
        
        network_ctx = context_extractor.extract_network_context(payment)
        party_ctx = context_extractor.extract_party_context(payment)
        
        print(f"\nNetwork Context:")
        print(f"  Source: {network_ctx['source']}")
        print(f"  Clearing: {network_ctx['clearing']}")
        
        print(f"\nParty Context:")
        for party, present in party_ctx.items():
            if present:
                print(f"  {party}: {present}")
        
        print(f"\nACE Repairs:")
        if 'ace' in payment:
            for repair in payment['ace']:
                if isinstance(repair, dict):
                    repair_id = repair.get('id')
                    repair_field = repair.get('field', '')
                    required_party = FIELD_PARTY_MAPPING.get(repair_field, 'N/A')
                    party_present = party_ctx.get(required_party, False) if required_party != 'N/A' else True
                    
                    print(f"  ID: {repair_id}, Field: {repair_field}")
                    print(f"    Required Party: {required_party}, Present: {party_present}")
        
        print(f"\nEntity Transformations:")
        for entity_name, entity_data in payment.items():
            if not isinstance(entity_data, dict):
                continue
            
            # Skip metadata fields
            if entity_name in ['source', 'clearing', 'parties', 'flags', 'ace']:
                continue
            
            if 'before' in entity_data and 'after' in entity_data:
                print(f"\n  Entity: {entity_name}")
                
                transformation = discovery.discover_transformation(
                    entity_data['before'], 
                    entity_data['after']
                )
                
                if transformation['added_fields']:
                    print(f"    Added fields: {transformation['added_fields'][:5]}")
                if transformation['filled_fields']:
                    print(f"    Filled fields: {transformation['filled_fields'][:5]}")
            else:
                print(f"\n  Entity: {entity_name}")
                print(f"    Current state only (no transformation to analyze)")
                # Show fingerprint for current state
                fingerprint = discovery.create_pattern_fingerprint(entity_data)
                print(f"    Pattern fingerprint: {fingerprint}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
