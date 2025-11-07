"""
Practical Usage Guide - TAL Code Indexer for Payment Systems

This guide shows concrete examples of using the indexer for
common payment system analysis tasks.
"""

import tal_code_indexer
from tal_query_interface import TALQueryInterface
from pathlib import Path
import json


def example_1_find_ofac_code():
    """
    Example 1: Find all code related to OFAC sanctions screening
    
    Use case: Compliance audit requires documenting all OFAC-related code
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Find OFAC Sanctions Screening Code")
    print("=" * 60)
    
    # Load index
    indexer = tal_code_indexer.TALCodeIndexer.load_binary('payment_system_index.pkl')
    
    # Method 1: Direct keyword search
    print("\nMethod 1: Keyword Search")
    print("-" * 40)
    ofac_results = indexer.search("OFAC screening sanctions SDN", top_k=20)
    
    print(f"Found {len(ofac_results)} functions via keyword search:")
    for name, score, elem in ofac_results[:10]:
        print(f"  • {name} ({Path(elem.file_path).name}:{elem.line_number})")
        if elem.calls:
            print(f"    Calls: {', '.join(elem.calls[:3])}")
    
    # Method 2: Capability-based search
    print("\nMethod 2: Capability-Based Search")
    print("-" * 40)
    capability_functions = indexer.get_capability_functions('compliance.ofac.screening')
    
    print(f"Found {len(capability_functions)} functions via capability mapping:")
    for func in capability_functions[:10]:
        print(f"  • {func.name} ({Path(func.file_path).name}:{func.line_number})")
    
    # Method 3: Find everything that calls OFAC functions
    print("\nMethod 3: Find All Code That Uses OFAC Functions")
    print("-" * 40)
    
    all_ofac_code = set()
    
    # Add direct matches
    for _, _, elem in ofac_results:
        all_ofac_code.add(elem.name)
    
    # Add capability matches
    for func in capability_functions:
        all_ofac_code.add(func.name)
    
    # Find transitive callers (anything that eventually calls OFAC code)
    def find_transitive_callers(func_name, depth=0, max_depth=3, visited=None):
        if visited is None:
            visited = set()
        if depth > max_depth or func_name in visited:
            return
        visited.add(func_name)
        
        for caller in indexer.get_callers(func_name):
            all_ofac_code.add(caller.name)
            find_transitive_callers(caller.name, depth + 1, max_depth, visited)
    
    for ofac_func in list(all_ofac_code):
        find_transitive_callers(ofac_func)
    
    print(f"\nTotal OFAC-related code (including callers): {len(all_ofac_code)} functions")
    
    # Export for documentation
    ofac_report = {
        'analysis_type': 'OFAC Compliance Audit',
        'total_functions': len(all_ofac_code),
        'direct_ofac_functions': len(capability_functions),
        'functions': [
            {
                'name': func_name,
                'details': indexer.elements[func_name].to_dict() if func_name in indexer.elements else None
            }
            for func_name in list(all_ofac_code)[:50]  # Limit for report
        ]
    }
    
    with open('ofac_compliance_report.json', 'w') as f:
        json.dump(ofac_report, f, indent=2)
    
    print("\n✓ Exported OFAC compliance report to: ofac_compliance_report.json")


def example_2_wire_transfer_flow():
    """
    Example 2: Trace complete wire transfer processing flow
    
    Use case: Documentation needs for wire transfer process
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Wire Transfer Processing Flow")
    print("=" * 60)
    
    indexer = tal_code_indexer.TALCodeIndexer.load_binary('payment_system_index.pkl')
    interface = TALQueryInterface(indexer)
    
    # Find entry point for wire transfers
    print("\nFinding wire transfer entry point...")
    entry_results = indexer.search("wire transfer process initiate main", top_k=10)
    
    if entry_results:
        entry_function = entry_results[0][2].name  # Get top result
        print(f"Entry point: {entry_function}")
        
        # Get complete context
        context = interface.get_function_context(
            entry_function,
            include_callers=True,
            include_callees=True,
            max_depth=3
        )
        
        # Build flow diagram
        print(f"\nWire Transfer Flow Starting from {entry_function}:")
        print("-" * 40)
        
        # Show immediate flow
        if context.get('callees'):
            print(f"\n{entry_function} calls:")
            for callee in context['callees'][:10]:
                print(f"  → {callee['name']}")
                
                # Show what each sub-function does
                if callee['exists'] and callee['capabilities']:
                    caps = ', '.join(callee['capabilities'])
                    print(f"     Capabilities: {caps}")
                
                # Show second level
                if callee['name'] in indexer.elements:
                    sub_calls = indexer.elements[callee['name']].calls[:3]
                    if sub_calls:
                        print(f"     Calls: {', '.join(sub_calls)}")
        
        # Show full call chains
        print(f"\nComplete Call Chains (depth 3):")
        print("-" * 40)
        chains = indexer.get_call_chain(entry_function, max_depth=3)
        
        for i, chain in enumerate(chains[:10], 1):
            print(f"{i}. {' → '.join(chain)}")
        
        if len(chains) > 10:
            print(f"   ... and {len(chains) - 10} more chains")
        
        # Export flow for documentation
        flow_doc = {
            'entry_point': entry_function,
            'capabilities': list(context['function']['business_capabilities']),
            'call_chains': [' → '.join(chain) for chain in chains],
            'key_functions': [
                {
                    'name': callee['name'],
                    'capabilities': callee['capabilities']
                }
                for callee in context.get('callees', [])[:20]
            ]
        }
        
        with open('wire_transfer_flow.json', 'w') as f:
            json.dump(flow_doc, f, indent=2)
        
        print("\n✓ Exported wire transfer flow to: wire_transfer_flow.json")


def example_3_iso20022_usage():
    """
    Example 3: Find all ISO 20022 message handling code
    
    Use case: Upgrading ISO 20022 implementation
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 3: ISO 20022 Message Handling Analysis")
    print("=" * 60)
    
    indexer = tal_code_indexer.TALCodeIndexer.load_binary('payment_system_index.pkl')
    
    # Find ISO 20022 related code
    print("\nSearching for ISO 20022 code...")
    iso_results = indexer.search("pacs.008 pacs008 ISO 20022 message format", top_k=30)
    
    # Group by message type
    by_message_type = {}
    for name, score, elem in iso_results:
        # Try to detect message type from name/content
        content_upper = elem.content.upper()
        
        if 'PACS.008' in content_upper or 'PACS008' in content_upper:
            msg_type = 'pacs.008'
        elif 'PACS.002' in content_upper or 'PACS002' in content_upper:
            msg_type = 'pacs.002'
        elif 'PAIN.001' in content_upper or 'PAIN001' in content_upper:
            msg_type = 'pain.001'
        else:
            msg_type = 'other'
        
        if msg_type not in by_message_type:
            by_message_type[msg_type] = []
        
        by_message_type[msg_type].append({
            'name': name,
            'file': elem.file_path,
            'line': elem.line_number,
            'score': score
        })
    
    # Report by message type
    print("\nISO 20022 Message Types Found:")
    print("-" * 40)
    for msg_type, functions in sorted(by_message_type.items()):
        print(f"\n{msg_type.upper()}: {len(functions)} functions")
        for func in functions[:5]:
            print(f"  • {func['name']} ({Path(func['file']).name}:{func['line']})")
        if len(functions) > 5:
            print(f"  ... and {len(functions) - 5} more")
    
    # Find message creation vs parsing
    print("\nMessage Creation vs Parsing:")
    print("-" * 40)
    
    creation_funcs = []
    parsing_funcs = []
    
    for name, score, elem in iso_results:
        content_upper = elem.content.upper()
        if any(kw in content_upper for kw in ['CREATE', 'BUILD', 'FORMAT', 'GENERATE']):
            creation_funcs.append(elem.name)
        if any(kw in content_upper for kw in ['PARSE', 'READ', 'EXTRACT', 'DECODE']):
            parsing_funcs.append(elem.name)
    
    print(f"Message Creation: {len(creation_funcs)} functions")
    for func in creation_funcs[:5]:
        print(f"  • {func}")
    
    print(f"\nMessage Parsing: {len(parsing_funcs)} functions")
    for func in parsing_funcs[:5]:
        print(f"  • {func}")
    
    # Export upgrade impact analysis
    upgrade_report = {
        'total_iso20022_functions': len(iso_results),
        'by_message_type': {
            msg_type: len(funcs) for msg_type, funcs in by_message_type.items()
        },
        'creation_functions': creation_funcs,
        'parsing_functions': parsing_funcs,
        'high_priority_functions': [
            {
                'name': name,
                'score': score,
                'file': elem.file_path,
                'line': elem.line_number
            }
            for name, score, elem in iso_results[:20]
        ]
    }
    
    with open('iso20022_upgrade_analysis.json', 'w') as f:
        json.dump(upgrade_report, f, indent=2)
    
    print("\n✓ Exported ISO 20022 analysis to: iso20022_upgrade_analysis.json")


def example_4_payment_repair_rag():
    """
    Example 4: Build RAG context for payment repair prediction model
    
    Use case: Training ML model to predict payment repairs
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Payment Repair RAG Context")
    print("=" * 60)
    
    indexer = tal_code_indexer.TALCodeIndexer.load_binary('payment_system_index.pkl')
    interface = TALQueryInterface(indexer)
    
    # Find all repair-related code
    print("\nBuilding repair prediction context...")
    repair_results = indexer.search("repair ACE exception error fix correction", top_k=50)
    
    # Build comprehensive context for RAG
    rag_contexts = []
    
    for name, score, elem in repair_results[:20]:  # Top 20 functions
        # Get full context
        context = interface.get_function_context(elem.name, max_depth=2)
        
        # Format for LLM
        llm_context = f"""
Function: {elem.name}
File: {Path(elem.file_path).name}:{elem.line_number}
Purpose: {', '.join(elem.business_capabilities) if elem.business_capabilities else 'Payment repair processing'}

Parameters: {', '.join(elem.parameters) if elem.parameters else 'None'}
Return Type: {elem.return_type or 'void'}

Calls:
{chr(10).join(f'  - {call}' for call in elem.calls[:10]) if elem.calls else '  None'}

Called By:
{chr(10).join(f'  - {caller["name"]}' for caller in context.get('callers', [])[:5])}

Key Logic:
{elem.content[:500]}...
"""
        
        rag_contexts.append({
            'function_name': elem.name,
            'context': llm_context,
            'metadata': {
                'file': elem.file_path,
                'line': elem.line_number,
                'capabilities': list(elem.business_capabilities),
                'parameters': elem.parameters,
                'calls': elem.calls
            }
        })
    
    # Export for RAG system
    rag_export = {
        'query': 'payment repair prediction',
        'context_count': len(rag_contexts),
        'contexts': rag_contexts
    }
    
    with open('payment_repair_rag_context.json', 'w') as f:
        json.dump(rag_export, f, indent=2)
    
    print(f"\n✓ Generated {len(rag_contexts)} RAG contexts")
    print("✓ Exported to: payment_repair_rag_context.json")
    
    # Show example context
    if rag_contexts:
        print("\nExample RAG Context:")
        print("-" * 40)
        print(rag_contexts[0]['context'])


def example_5_impact_analysis():
    """
    Example 5: Analyze impact of changing a core function
    
    Use case: Planning refactoring of BIC validation logic
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Change Impact Analysis")
    print("=" * 60)
    
    indexer = tal_code_indexer.TALCodeIndexer.load_binary('payment_system_index.pkl')
    interface = TALQueryInterface(indexer)
    
    # Function to analyze
    target_function = 'VALIDATE_BIC_CODE'  # Example
    
    print(f"\nAnalyzing impact of changing: {target_function}")
    print("-" * 40)
    
    # Find direct callers
    direct_callers = indexer.get_callers(target_function)
    print(f"\nDirect Callers: {len(direct_callers)}")
    for caller in direct_callers[:10]:
        print(f"  • {caller.name} ({Path(caller.file_path).name}:{caller.line_number})")
        if caller.business_capabilities:
            print(f"    Affects: {', '.join(caller.business_capabilities)}")
    
    # Find transitive callers (full impact)
    all_affected = set()
    visited = set()
    
    def find_all_callers(func_name, depth=0, max_depth=5):
        if depth > max_depth or func_name in visited:
            return
        visited.add(func_name)
        
        for caller in indexer.get_callers(func_name):
            all_affected.add(caller.name)
            find_all_callers(caller.name, depth + 1, max_depth)
    
    find_all_callers(target_function)
    
    print(f"\nTotal Affected Functions: {len(all_affected)}")
    
    # Group by capability
    affected_by_capability = {}
    for func_name in all_affected:
        if func_name in indexer.elements:
            elem = indexer.elements[func_name]
            for cap in elem.business_capabilities:
                if cap not in affected_by_capability:
                    affected_by_capability[cap] = []
                affected_by_capability[cap].append(func_name)
    
    print("\nImpact by Business Capability:")
    print("-" * 40)
    for cap, funcs in sorted(affected_by_capability.items(), 
                            key=lambda x: len(x[1]), reverse=True):
        if cap in indexer.capabilities:
            cap_name = indexer.capabilities[cap].name
        else:
            cap_name = cap
        print(f"\n{cap_name}: {len(funcs)} functions affected")
        for func in funcs[:5]:
            print(f"  • {func}")
        if len(funcs) > 5:
            print(f"  ... and {len(funcs) - 5} more")
    
    # Export impact analysis
    impact_report = {
        'target_function': target_function,
        'direct_callers': len(direct_callers),
        'total_affected': len(all_affected),
        'affected_capabilities': {
            cap: len(funcs) for cap, funcs in affected_by_capability.items()
        },
        'affected_functions': list(all_affected),
        'high_risk_areas': [
            cap for cap, funcs in affected_by_capability.items() 
            if len(funcs) >= 5
        ]
    }
    
    with open(f'{target_function}_impact_analysis.json', 'w') as f:
        json.dump(impact_report, f, indent=2)
    
    print(f"\n✓ Exported impact analysis to: {target_function}_impact_analysis.json")


def main():
    """Run all examples."""
    print("TAL Code Indexer - Practical Usage Examples")
    print("=" * 60)
    print("\nNote: These examples assume you have an indexed codebase.")
    print("Run: python tal_indexer_example.py /path/to/code first")
    print()
    
    # Run examples (commented out - uncomment as needed)
    
    try:
        example_1_find_ofac_code()
    except Exception as e:
        print(f"\nExample 1 error: {e}")
    
    try:
        example_2_wire_transfer_flow()
    except Exception as e:
        print(f"\nExample 2 error: {e}")
    
    try:
        example_3_iso20022_usage()
    except Exception as e:
        print(f"\nExample 3 error: {e}")
    
    try:
        example_4_payment_repair_rag()
    except Exception as e:
        print(f"\nExample 4 error: {e}")
    
    try:
        example_5_impact_analysis()
    except Exception as e:
        print(f"\nExample 5 error: {e}")
    
    print("\n" + "=" * 60)
    print("All examples complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
