#!/usr/bin/env python3
"""
Comprehensive diagnostic script - analyzes ALL entities
"""

import json
import pickle
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path
import sys
import os

# Define all entities to check
ALL_ENTITIES = [
    'cdtr',           # Creditor
    'dbtr',           # Debtor  
    'cdtrAgt',        # Creditor Agent
    'dbtrAgt',        # Debtor Agent
    'cdtrAcct',       # Creditor Account
    'dbtrAcct',       # Debtor Account
    'instgAgt',       # Instructing Agent
    'instdAgt',       # Instructed Agent
    'intrmyAgt1',     # Intermediary Agent 1
    'intrmyAgt2',     # Intermediary Agent 2
    'intrmyAgt3',     # Intermediary Agent 3
    'prvtId',         # Private Identification
    'orgId',          # Organization Identification
    'rmtInf',         # Remittance Information
    'purp',           # Purpose
    'rgltryRptg',     # Regulatory Reporting
    'rltdRmtInf',     # Related Remittance Info
    'rltdPties',      # Related Parties
    'ultmtCdtr',      # Ultimate Creditor
    'ultmtDbtr',      # Ultimate Debtor
    'initgPty',       # Initiating Party
    'fwdgAgt',        # Forwarding Agent
    'splmtryData'     # Supplementary Data
]

AGENT_ENTITIES = ['cdtrAgt', 'dbtrAgt', 'instgAgt', 'instdAgt', 'intrmyAgt1', 'intrmyAgt2', 'intrmyAgt3', 'fwdgAgt']
PARTY_ENTITIES = ['cdtr', 'dbtr', 'ultmtCdtr', 'ultmtDbtr', 'initgPty']
ACCOUNT_ENTITIES = ['cdtrAcct', 'dbtrAcct']

def diagnose_model_basic(model_dir: str):
    """Basic diagnosis without unpickling complex objects"""
    
    print("="*70)
    print("MODEL DIAGNOSTICS (SAFE MODE)")
    print("="*70)
    
    model_path = Path(model_dir)
    
    # 1. Check what files exist
    print("\n1. MODEL FILES CHECK:")
    expected_files = [
        'config.json',
        'processor.pkl', 
        'rf_model.pkl',
        'nn_model.pt',
        'model_info.json'
    ]
    
    for file in expected_files:
        file_path = model_path / file
        if file_path.exists():
            size_kb = file_path.stat().st_size / 1024
            print(f"   ✓ {file}: {size_kb:.1f} KB")
        else:
            print(f"   ❌ {file}: MISSING")
    
    # 2. Check config
    config_path = model_path / 'config.json'
    if config_path.exists():
        print("\n2. CONFIGURATION:")
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"   Hidden dim: {config.get('hidden_dim', 'N/A')}")
        print(f"   Batch size: {config.get('batch_size', 'N/A')}")
        print(f"   Learning rate: {config.get('learning_rate', 'N/A')}")
        print(f"   Epochs: {config.get('num_epochs', 'N/A')}")
    
    # 3. Check model info
    model_info_path = model_path / 'model_info.json'
    if model_info_path.exists():
        print("\n3. MODEL ARCHITECTURE:")
        with open(model_info_path, 'r') as f:
            model_info = json.load(f)
        print(f"   Input features: {model_info.get('num_features', 'N/A')}")
        print(f"   Output repairs: {model_info.get('num_repairs', 'N/A')}")
        print(f"   Hidden dim: {model_info.get('hidden_dim', 'N/A')}")
        
        if model_info.get('num_repairs', 0) == 0:
            print("   ❌ ERROR: Model has 0 output repairs! Training failed.")
    
    # 4. Try to inspect processor without full unpickling
    print("\n4. PROCESSOR INSPECTION (Limited):")
    processor_path = model_path / 'processor.pkl'
    if processor_path.exists():
        try:
            with open(processor_path, 'rb') as f:
                content = f.read()
                
                # Look for repair IDs in the binary content
                repair_ids = []
                common_repairs = [b'6021', b'8852', b'7743', b'6001', b'8001', b'6022', b'8853']
                for repair in common_repairs:
                    if repair in content:
                        repair_ids.append(repair.decode())
                
                if repair_ids:
                    print(f"   Found repair IDs in vocabulary: {repair_ids}")
                else:
                    print("   ⚠️ No common repair IDs found in vocabulary")
                
                # Check size as indicator
                size_mb = len(content) / (1024 * 1024)
                print(f"   Processor size: {size_mb:.2f} MB")
                if size_mb < 0.1:
                    print("   ⚠️ Processor seems too small, may not have learned patterns")
        except Exception as e:
            print(f"   Could not inspect processor: {e}")

def analyze_all_entities(data_dir: str, max_files: int = 10):
    """Comprehensively analyze ALL entities in training data"""
    print("\n" + "="*70)
    print("COMPREHENSIVE ENTITY ANALYSIS")
    print("="*70)
    
    json_files = list(Path(data_dir).glob('**/*.json'))[:max_files]
    print(f"\nAnalyzing {len(json_files)} sample files...")
    
    stats = {
        'total_transactions': 0,
        'has_ace_repairs': 0,
        'repair_ids': Counter(),
        'source_types': Counter(),
        'clearing_types': Counter(),
        'entities_present': Counter(),
        'entities_with_before': Counter(),
        'entities_with_after': Counter(),
        'entities_with_diffs': Counter(),
        'missing_patterns': defaultdict(Counter)
    }
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Handle different formats
            transactions = {}
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, dict):
                        transactions[key] = value
            elif isinstance(data, list):
                for idx, item in enumerate(data):
                    if isinstance(item, dict):
                        if len(item) == 1:
                            key = list(item.keys())[0]
                            transactions[key] = item[key]
                        else:
                            transactions[f"txn_{idx}"] = item
            
            # Analyze each transaction
            for txn_id, txn_data in transactions.items():
                if not isinstance(txn_data, dict):
                    continue
                
                stats['total_transactions'] += 1
                
                # Check source/clearing
                if 'source' in txn_data:
                    stats['source_types'][txn_data['source']] += 1
                if 'clearing' in txn_data:
                    stats['clearing_types'][txn_data['clearing']] += 1
                
                # Check for ACE repairs
                if 'ace' in txn_data and txn_data['ace']:
                    stats['has_ace_repairs'] += 1
                    for repair in txn_data['ace']:
                        if isinstance(repair, dict):
                            repair_id = repair.get('id', 'unknown')
                        else:
                            repair_id = str(repair)
                        stats['repair_ids'][repair_id] += 1
                
                # Analyze EVERY entity
                for entity in ALL_ENTITIES:
                    if entity in txn_data:
                        stats['entities_present'][entity] += 1
                        entity_data = txn_data[entity]
                        
                        if isinstance(entity_data, dict):
                            if 'before' in entity_data:
                                stats['entities_with_before'][entity] += 1
                                analyze_entity_issues(entity, entity_data['before'], stats['missing_patterns'])
                            if 'after' in entity_data:
                                stats['entities_with_after'][entity] += 1
                            if 'diffs' in entity_data:
                                stats['entities_with_diffs'][entity] += 1
                            
                            # If no before/after, it's the current state
                            if 'before' not in entity_data and 'after' not in entity_data:
                                analyze_entity_issues(entity, entity_data, stats['missing_patterns'])
                    
        except Exception as e:
            print(f"Error reading {json_file}: {e}")
    
    # Print comprehensive results
    print(f"\nTotal transactions: {stats['total_transactions']}")
    
    if stats['total_transactions'] > 0:
        print(f"Transactions with repairs: {stats['has_ace_repairs']}")
        
        print("\n=== TOP REPAIR IDS ===")
        for repair_id, count in stats['repair_ids'].most_common(15):
            print(f"  {repair_id}: {count}")
        
        print("\n=== ENTITY PRESENCE ===")
        for entity in ALL_ENTITIES:
            count = stats['entities_present'].get(entity, 0)
            if count > 0:
                pct = 100 * count / stats['total_transactions']
                before_count = stats['entities_with_before'].get(entity, 0)
                after_count = stats['entities_with_after'].get(entity, 0)
                diff_count = stats['entities_with_diffs'].get(entity, 0)
                
                print(f"\n{entity}:")
                print(f"  Present: {count} ({pct:.1f}%)")
                print(f"  Has before: {before_count}")
                print(f"  Has after: {after_count}")
                print(f"  Has diffs: {diff_count}")
                
                # Show common issues
                entity_issues = stats['missing_patterns'].get(entity, {})
                if entity_issues:
                    print(f"  Common issues:")
                    for issue, issue_count in entity_issues.most_common(3):
                        print(f"    - {issue}: {issue_count}")
        
        print("\n=== SOURCE/CLEARING COMBINATIONS ===")
        for source, count in stats['source_types'].most_common():
            print(f"  Source {source}: {count}")
        for clearing, count in stats['clearing_types'].most_common():
            print(f"  Clearing {clearing}: {count}")

def analyze_entity_issues(entity_name: str, entity_data: dict, missing_patterns: dict):
    """Analyze what's missing or incorrect in an entity"""
    if not isinstance(entity_data, dict):
        return
    
    issues = []
    
    # For agents - check completeness
    if entity_name in AGENT_ENTITIES:
        fin_inst = entity_data.get('FinInstnId', entity_data.get('finInstnId', {}))
        if isinstance(fin_inst, dict):
            # Check for BIC
            if not fin_inst.get('BICFI') and not fin_inst.get('bicFi') and not fin_inst.get('BIC'):
                issues.append('missing_bic')
            
            # Check for name
            if not fin_inst.get('Nm') and not fin_inst.get('nm'):
                issues.append('missing_name')
            
            # Check for address
            postal = fin_inst.get('PstlAdr', fin_inst.get('pstlAdr', {}))
            if isinstance(postal, dict):
                if not postal.get('AdrLine') and not postal.get('adrLine'):
                    issues.append('missing_address')
                if not postal.get('Ctry') and not postal.get('ctry'):
                    issues.append('missing_country')
            else:
                issues.append('missing_postal_address')
            
            # Check for clearing system
            clr_sys = fin_inst.get('ClrSysMmbId', fin_inst.get('clrSysMmbId', {}))
            if isinstance(clr_sys, dict):
                if not clr_sys.get('MmbId') and not clr_sys.get('mmbId'):
                    issues.append('missing_member_id')
            
            # Check for MmbId at wrong level
            if (fin_inst.get('MmbId') or fin_inst.get('mmbId')) and not clr_sys:
                issues.append('mmbid_at_wrong_level')
    
    # For parties - check completeness
    elif entity_name in PARTY_ENTITIES:
        if not entity_data.get('Nm') and not entity_data.get('nm'):
            issues.append('missing_name')
        
        postal = entity_data.get('PstlAdr', entity_data.get('pstlAdr', {}))
        if isinstance(postal, dict):
            if not postal.get('AdrLine') and not postal.get('adrLine'):
                issues.append('missing_address')
            if not postal.get('Ctry') and not postal.get('ctry'):
                issues.append('missing_country')
        else:
            issues.append('missing_postal_address')
        
        # Check identification
        if not entity_data.get('Id') and not entity_data.get('id'):
            issues.append('missing_identification')
    
    # For accounts - check completeness
    elif entity_name in ACCOUNT_ENTITIES:
        id_data = entity_data.get('Id', entity_data.get('id', {}))
        if isinstance(id_data, dict):
            if not id_data.get('IBAN') and not id_data.get('iban'):
                if not id_data.get('Othr') and not id_data.get('othr'):
                    issues.append('missing_account_id')
        
        if not entity_data.get('Tp') and not entity_data.get('tp'):
            issues.append('missing_account_type')
        
        if not entity_data.get('Ccy') and not entity_data.get('ccy'):
            issues.append('missing_currency')
    
    # For remittance info
    elif entity_name == 'rmtInf':
        if not entity_data.get('Ustrd') and not entity_data.get('ustrd'):
            if not entity_data.get('Strd') and not entity_data.get('strd'):
                issues.append('missing_remittance_info')
    
    # Record issues
    for issue in issues:
        missing_patterns[entity_name][issue] += 1

def test_comprehensive_extraction(test_file: str):
    """Comprehensively test feature extraction for ALL entities"""
    print("\n" + "="*70)
    print("COMPREHENSIVE FEATURE EXTRACTION TEST")
    print("="*70)
    
    try:
        with open(test_file, 'r') as f:
            data = json.load(f)
        
        # Extract transaction
        if isinstance(data, dict) and len(data) == 1:
            txn_id = list(data.keys())[0]
            txn_data = data[txn_id]
        else:
            txn_data = data
            txn_id = "test"
        
        print(f"\nTransaction ID: {txn_id}")
        print(f"Source: {txn_data.get('source', 'N/A')}")
        print(f"Clearing: {txn_data.get('clearing', 'N/A')}")
        
        # Analyze ALL entities
        print("\n=== ENTITY ANALYSIS ===")
        
        entity_summary = {
            'agents': {},
            'parties': {},
            'accounts': {},
            'other': {}
        }
        
        for entity in ALL_ENTITIES:
            if entity not in txn_data:
                continue
            
            entity_data = txn_data[entity]
            
            # Categorize entity
            if entity in AGENT_ENTITIES:
                category = 'agents'
            elif entity in PARTY_ENTITIES:
                category = 'parties'
            elif entity in ACCOUNT_ENTITIES:
                category = 'accounts'
            else:
                category = 'other'
            
            # Analyze structure
            structure_info = analyze_entity_structure(entity, entity_data)
            entity_summary[category][entity] = structure_info
        
        # Print detailed analysis
        for category, entities in entity_summary.items():
            if entities:
                print(f"\n{category.upper()}:")
                for entity, info in entities.items():
                    print(f"\n  {entity}:")
                    print(f"    Structure: {info['structure']}")
                    if info['issues']:
                        print(f"    Issues found:")
                        for issue in info['issues']:
                            print(f"      - {issue}")
                    if info['repairs_needed']:
                        print(f"    Likely repairs needed:")
                        for repair in info['repairs_needed']:
                            print(f"      - {repair}")
        
        # Check for ACE repairs
        if 'ace' in txn_data:
            repairs = txn_data['ace']
            print(f"\n=== ACTUAL REPAIRS ===")
            print(f"Count: {len(repairs) if repairs else 0}")
            if repairs:
                for repair in repairs[:10]:  # Show first 10
                    if isinstance(repair, dict):
                        print(f"  - ID: {repair.get('id')}, Field: {repair.get('field')}, Code: {repair.get('code')}")
                    else:
                        print(f"  - {repair}")
        
    except Exception as e:
        print(f"Error analyzing test file: {e}")

def analyze_entity_structure(entity_name: str, entity_data):
    """Analyze structure and issues for a single entity"""
    info = {
        'structure': 'unknown',
        'issues': [],
        'repairs_needed': []
    }
    
    if isinstance(entity_data, dict):
        # Check structure type
        if 'before' in entity_data and 'after' in entity_data:
            info['structure'] = 'has_before_and_after'
        elif 'before' in entity_data:
            info['structure'] = 'has_before_only'
        elif 'after' in entity_data:
            info['structure'] = 'has_after_only'
        else:
            info['structure'] = 'direct_data'
        
        # Analyze the relevant state
        check_data = entity_data.get('before', entity_data)
        
        if entity_name in AGENT_ENTITIES and isinstance(check_data, dict):
            # Check agent completeness
            fin_inst = check_data.get('FinInstnId', check_data.get('finInstnId', {}))
            if isinstance(fin_inst, dict):
                # Missing BIC
                if not any(fin_inst.get(k) for k in ['BICFI', 'bicFi', 'BIC']):
                    info['issues'].append('Missing BIC')
                    if any(fin_inst.get(k) for k in ['ClrSysMmbId', 'clrSysMmbId']):
                        info['repairs_needed'].append('BIC lookup (8852)')
                
                # Missing name
                if not any(fin_inst.get(k) for k in ['Nm', 'nm']):
                    info['issues'].append('Missing bank name')
                    if any(fin_inst.get(k) for k in ['BICFI', 'bicFi']):
                        info['repairs_needed'].append('Name lookup')
                
                # Check address
                postal = fin_inst.get('PstlAdr', fin_inst.get('pstlAdr', {}))
                if isinstance(postal, dict):
                    if not any(postal.get(k) for k in ['Ctry', 'ctry', 'CtryOfRes', 'ctryOfRes']):
                        info['issues'].append('Missing country')
                        info['repairs_needed'].append('Country extraction (6021)')
                    if not any(postal.get(k) for k in ['AdrLine', 'adrLine']):
                        info['issues'].append('Missing address lines')
                        info['repairs_needed'].append('Address enrichment')
                
                # MmbId structure issue
                if any(fin_inst.get(k) for k in ['MmbId', 'mmbId']):
                    if not any(fin_inst.get(k) for k in ['ClrSysId', 'clrSysId']):
                        info['issues'].append('MmbId not in ClrSysId structure')
                        info['repairs_needed'].append('Structural repair')
        
        elif entity_name in PARTY_ENTITIES and isinstance(check_data, dict):
            # Check party completeness
            if not any(check_data.get(k) for k in ['Nm', 'nm']):
                info['issues'].append('Missing party name')
            
            postal = check_data.get('PstlAdr', check_data.get('pstlAdr', {}))
            if isinstance(postal, dict):
                if not any(postal.get(k) for k in ['Ctry', 'ctry']):
                    info['issues'].append('Missing country')
                    info['repairs_needed'].append('Country extraction (6021)')
        
        elif entity_name in ACCOUNT_ENTITIES and isinstance(check_data, dict):
            # Check account completeness
            id_data = check_data.get('Id', check_data.get('id', {}))
            if isinstance(id_data, dict):
                if not any(id_data.get(k) for k in ['IBAN', 'iban']):
                    if not any(id_data.get(k) for k in ['Othr', 'othr']):
                        info['issues'].append('Missing account identifier')
            
            if not any(check_data.get(k) for k in ['Ccy', 'ccy']):
                info['issues'].append('Missing currency')
    
    elif entity_data is None:
        info['structure'] = 'null'
        info['issues'].append('Entity is null')
    else:
        info['structure'] = f'{type(entity_data).__name__}'
    
    return info

def print_recommendations():
    """Print comprehensive recommendations"""
    print("\n" + "="*70)
    print("RECOMMENDATIONS FOR IMPROVING PREDICTIONS")
    print("="*70)
    
    print("""
1. ENSURE ALL ENTITIES ARE CHECKED:
   Agents: cdtrAgt, dbtrAgt, instgAgt, instdAgt, intrmyAgt1-3
   Parties: cdtr, dbtr, ultmtCdtr, ultmtDbtr, initgPty
   Accounts: cdtrAcct, dbtrAcct
   Other: rmtInf, purp, rgltryRptg, etc.

2. COMMON REPAIR PATTERNS BY ENTITY:
   
   AGENTS (cdtrAgt, dbtrAgt):
   - Missing BIC + has clearing → Repair 8852 (BIC lookup)
   - Missing name + has BIC → Name lookup repair
   - Missing country in address → Repair 6021 (country extraction)
   - MmbId not in ClrSysId → Structural repair
   
   PARTIES (cdtr, dbtr):
   - Missing country → Repair 6021
   - Missing postal address → Address enrichment
   - Missing name → Party identification
   
   ACCOUNTS (cdtrAcct, dbtrAcct):
   - Missing IBAN → Account identification
   - Missing currency → Currency addition
   - Wrong account type → Type correction

3. CHECK ENTITY RELATIONSHIPS:
   - If cdtrAgt incomplete → affects cdtr
   - If dbtrAgt incomplete → affects dbtr
   - Missing intermediary agents → routing issues

4. SOURCE/CLEARING PATTERNS:
   - SWF + FED → specific agent requirements
   - ACH + CHIPS → different validation rules
   - Each combination has typical repairs

5. TRAINING DATA REQUIREMENTS:
   - Must have before/after states for entities
   - Diffs array shows what changed
   - ACE array contains actual repair IDs

6. FEATURE EXTRACTION MUST CHECK:
   - What's missing in EACH entity
   - Structural problems (wrong nesting)
   - Cross-entity consistency
   - Source/clearing requirements
""")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive diagnostic tool')
    parser.add_argument('--model_dir', default='./models', help='Model directory')
    parser.add_argument('--test_file', help='Test JSON file')
    parser.add_argument('--data_dir', help='Training data directory')
    
    args = parser.parse_args()
    
    # Run diagnostics
    if args.model_dir:
        diagnose_model_basic(args.model_dir)
    
    if args.data_dir:
        analyze_all_entities(args.data_dir)
    
    if args.test_file:
        test_comprehensive_extraction(args.test_file)
    
    print_recommendations()
    
    print("\n" + "="*70)
    print("COMPREHENSIVE DIAGNOSTIC COMPLETE")
    print("="*70)
