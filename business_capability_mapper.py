"""
Business Capability Mapper - Integrates Wire Processing Taxonomy

This module maps the real-world BusinessCapabilityTaxonomy to the
TAL code indexer, converting the hierarchical categories and keywords
into searchable capability definitions.
"""

import tal_code_indexer
from typing import Dict, List, Set


class BusinessCapabilityTaxonomy:
    """Wire Processing Business Capabilities taxonomy"""
    
    CAPABILITIES = {
        "Core Payment & Network": [
            "clearing networks", "fed", "chips", "swift", "clearing house",
            "network gateways", "network connectivity", "network acknowledgments",
            "network admin", "network certification", "lterm", "ack", "nak"
        ],
        
        "Payment Processing & Execution": [
            "payment initiation", "payment routing", "payment execution",
            "preadvising", "cover payments", "liquidity management",
            "debit confirmation", "credit confirmation", "outbound payment",
            "hard posting", "cutoffs", "workflow scheduling", "orchestration",
            "split advising", "intraday liquidity", "book transfer",
            "eod processing", "fee determination", "payment agreements",
            "payment returns", "payment prioritization", "warehousing",
            "exceptions processing"
        ],
        
        "Instruction & Validation": [
            "instruction management", "straight thru processing", "stp",
            "pay thru validation", "method of payment", "payment enrichment",
            "payment repair", "payment verify", "sod releasing",
            "auto repair", "date validation", "time validation",
            "account validation", "amount validation", "currency validation",
            "standing orders", "repetitive orders", "party association"
        ],
        
        "Controls & Risk Management": [
            "controls services", "anomalies detection", "ca&d",
            "sanctions screening", "fircosoft", "ofac", "funds control",
            "fraud checking", "debit authority", "duplicate checking",
            "debit blocks", "credit blocks", "memo posting",
            "ceo fraud", "cfm", "anti-money laundering", "aml", "newton",
            "risk control system", "rcs"
        ],
        
        "Data & Reporting": [
            "data management", "report distribution", "financial crimes reporting",
            "risk analysis reporting", "historical data", "payment reconciliation",
            "general ledger", "gl feeds", "account activity reporting",
            "adhoc reporting", "scheduled reporting", "event notification",
            "alert", "fee charges", "analysis charges", "product capabilities",
            "data service integration", "ai ml modeling", "report archiving",
            "axcis", "client billing", "statements", "client reconciliation",
            "transaction info", "balance info", "electronic window",
            "intelligence analytics", "trend analysis", "ecosystem analytics"
        ],
        
        "Service Integration": [
            "data masking", "obfuscation", "transaction replay",
            "data encryption", "decryption", "channel acknowledgments",
            "service api", "endpoint publishing", "duplicate detection",
            "api invocation", "service invocation", "queues", "topics",
            "format transformation", "id generation", "schema validation"
        ],
        
        "User Experience": [
            "business activity monitoring", "alert dispositioning",
            "queue drilldown", "telemetry", "ui maintenance",
            "user entitlements", "payment data inquiry", "trend analysis",
            "stp analysis", "risk event information", "smart alerting"
        ],
        
        "Channel & Integration": [
            "client authentication", "client preference", "channel connectivity",
            "canonical management", "shared services", "global services",
            "investigations", "pega", "intellitracs", "bank reconciliation",
            "1bkr", "intraday posting", "tms", "middleware", "gabs",
            "fx services", "wxchg", "opics", "revenue profit", "rpm",
            "enterprise fax", "gfx", "online wires", "olw",
            "treasury workstation", "pstw", "ceo api wires",
            "secure fax", "voice response", "vru", "position management",
            "loan q", "liq", "approval queue", "onq", "cyberpay",
            "1cyb", "1trx", "js-gds"
        ],
        
        "ISO Standards & Formats": [
            "iso20022", "iso 20022", "pacs.008", "pacs.009", "pacs.002",
            "pain.001", "camt.053", "mt103", "mt202", "mt199",
            "fedwire", "chips format", "swift format", "xml", "json"
        ],
        
        "Validation & Screening": [
            "bic validation", "bic code", "iban validation", "iban",
            "party validation", "account validation", "sanctions check",
            "watchlist screening", "name screening", "address validation",
            "routing validation", "aba", "sort code"
        ],
        
        "Transaction Processing": [
            "wire transfer", "wire payment", "domestic wire", "international wire",
            "cross-border payment", "same-day payment", "rtgs", "ach",
            "clearing", "settlement", "netting", "gross settlement"
        ]
    }
    
    # Synonyms and variations
    SYNONYMS = {
        "ofac": ["ofac screening", "sanctions", "ofac_screen_party"],
        "ace": ["automated clearing", "ace repair", "ace code"],
        "stp": ["straight through", "straight thru", "straight-through"],
        "aml": ["anti money laundering", "money laundering"],
        "bic": ["bank identifier", "swift code", "bic code"],
        "fed": ["federal reserve", "fedwire", "federal wire"],
        "chips": ["clearing house interbank", "chips network"],
        "iso20022": ["iso 20022", "iso-20022", "pacs", "pain", "camt"]
    }


def convert_taxonomy_to_capabilities(taxonomy: BusinessCapabilityTaxonomy) -> Dict[str, Dict]:
    """
    Convert the BusinessCapabilityTaxonomy into indexer-compatible format.
    
    Args:
        taxonomy: The business capability taxonomy instance
        
    Returns:
        Dictionary of capability definitions for the indexer
    """
    capabilities = {}
    
    for category_name, keywords in taxonomy.CAPABILITIES.items():
        # Create capability ID (lowercase, underscores)
        capability_id = category_name.lower().replace(' ', '_').replace('&', 'and')
        
        # Expand keywords with synonyms
        expanded_keywords = set()
        for keyword in keywords:
            expanded_keywords.add(keyword.lower())
            
            # Add variations without spaces/hyphens
            expanded_keywords.add(keyword.lower().replace(' ', ''))
            expanded_keywords.add(keyword.lower().replace(' ', '_'))
            expanded_keywords.add(keyword.lower().replace('-', ''))
            
            # Check if this keyword has synonyms
            for syn_key, syn_list in taxonomy.SYNONYMS.items():
                if keyword.lower() == syn_key or keyword.lower() in syn_list:
                    # Add all synonym variations
                    for syn in syn_list:
                        expanded_keywords.add(syn.lower())
                        expanded_keywords.add(syn.lower().replace(' ', ''))
                        expanded_keywords.add(syn.lower().replace(' ', '_'))
        
        capabilities[capability_id] = {
            'name': category_name,
            'description': f'Functions related to {category_name.lower()}',
            'keywords': list(expanded_keywords)
        }
    
    return capabilities


def register_taxonomy_with_indexer(indexer: tal_code_indexer.TALCodeIndexer) -> None:
    """
    Register all capabilities from the taxonomy with the indexer.
    
    Args:
        indexer: TAL code indexer instance to register capabilities with
    """
    taxonomy = BusinessCapabilityTaxonomy()
    capabilities = convert_taxonomy_to_capabilities(taxonomy)
    
    print(f"Registering {len(capabilities)} business capabilities from taxonomy...")
    
    for capability_id, cap_data in capabilities.items():
        indexer.add_business_capability(
            capability_id=capability_id,
            name=cap_data['name'],
            description=cap_data['description'],
            keywords=cap_data['keywords']
        )
    
    print(f"✓ Registered {len(capabilities)} capabilities with {sum(len(c['keywords']) for c in capabilities.values())} total keywords")


def get_capability_statistics(indexer: tal_code_indexer.TALCodeIndexer) -> Dict:
    """
    Generate statistics about capability mappings.
    
    Args:
        indexer: Populated indexer instance
        
    Returns:
        Dictionary with capability statistics
    """
    stats = {
        'total_capabilities': len(indexer.capabilities),
        'capabilities_with_code': 0,
        'total_function_mappings': 0,
        'by_category': {}
    }
    
    for cap_id, capability in indexer.capabilities.items():
        function_count = len(capability.implementing_functions)
        
        if function_count > 0:
            stats['capabilities_with_code'] += 1
            stats['total_function_mappings'] += function_count
        
        stats['by_category'][cap_id] = {
            'name': capability.name,
            'function_count': function_count,
            'keyword_count': len(capability.keywords)
        }
    
    return stats


def print_capability_coverage_report(indexer: tal_code_indexer.TALCodeIndexer) -> None:
    """
    Print a detailed report of capability coverage.
    
    Args:
        indexer: Populated indexer instance
    """
    stats = get_capability_statistics(indexer)
    
    print("\n" + "=" * 60)
    print("BUSINESS CAPABILITY COVERAGE REPORT")
    print("=" * 60)
    
    print(f"\nOverview:")
    print(f"  Total Capabilities: {stats['total_capabilities']}")
    print(f"  Capabilities with Code: {stats['capabilities_with_code']}")
    print(f"  Coverage: {stats['capabilities_with_code'] / stats['total_capabilities'] * 100:.1f}%")
    print(f"  Total Function Mappings: {stats['total_function_mappings']}")
    
    # Sort by function count
    sorted_caps = sorted(
        stats['by_category'].items(),
        key=lambda x: x[1]['function_count'],
        reverse=True
    )
    
    print(f"\nTop 10 Capabilities by Code Volume:")
    print("-" * 60)
    for i, (cap_id, cap_stats) in enumerate(sorted_caps[:10], 1):
        print(f"{i:2d}. {cap_stats['name']}")
        print(f"    Functions: {cap_stats['function_count']}, Keywords: {cap_stats['keyword_count']}")
    
    # Show capabilities with no code
    no_code_caps = [
        (cap_id, cap_stats['name'])
        for cap_id, cap_stats in sorted_caps
        if cap_stats['function_count'] == 0
    ]
    
    if no_code_caps:
        print(f"\nCapabilities with No Code Found: {len(no_code_caps)}")
        print("-" * 60)
        for cap_id, name in no_code_caps[:10]:
            print(f"  • {name}")
        if len(no_code_caps) > 10:
            print(f"  ... and {len(no_code_caps) - 10} more")


def find_cross_capability_functions(indexer: tal_code_indexer.TALCodeIndexer,
                                    min_capabilities: int = 2) -> List[Dict]:
    """
    Find functions that implement multiple business capabilities.
    
    These are often orchestrator or gateway functions.
    
    Args:
        indexer: Populated indexer instance
        min_capabilities: Minimum number of capabilities to qualify
        
    Returns:
        List of functions with their capabilities
    """
    cross_capability = []
    
    for name, element in indexer.elements.items():
        if len(element.business_capabilities) >= min_capabilities:
            cross_capability.append({
                'name': name,
                'type': element.element_type,
                'capability_count': len(element.business_capabilities),
                'capabilities': list(element.business_capabilities),
                'file': element.file_path,
                'line': element.line_number
            })
    
    # Sort by capability count
    cross_capability.sort(key=lambda x: x['capability_count'], reverse=True)
    
    return cross_capability


def search_by_multiple_capabilities(indexer: tal_code_indexer.TALCodeIndexer,
                                   capability_ids: List[str]) -> Dict:
    """
    Find functions that implement ALL of the specified capabilities.
    
    Useful for finding integration points between capabilities.
    
    Args:
        indexer: Populated indexer instance
        capability_ids: List of capability IDs to search for
        
    Returns:
        Dictionary with results
    """
    # Get functions for each capability
    capability_functions = {}
    for cap_id in capability_ids:
        if cap_id in indexer.capabilities:
            funcs = indexer.get_capability_functions(cap_id)
            capability_functions[cap_id] = set(f.name for f in funcs)
    
    if not capability_functions:
        return {'error': 'No valid capabilities found'}
    
    # Find intersection
    all_sets = list(capability_functions.values())
    intersection = all_sets[0]
    for func_set in all_sets[1:]:
        intersection = intersection & func_set
    
    # Get full function details
    result_functions = []
    for func_name in intersection:
        if func_name in indexer.elements:
            elem = indexer.elements[func_name]
            result_functions.append({
                'name': elem.name,
                'type': elem.element_type,
                'file': elem.file_path,
                'line': elem.line_number,
                'all_capabilities': list(elem.business_capabilities)
            })
    
    return {
        'query_capabilities': capability_ids,
        'match_count': len(result_functions),
        'functions': result_functions
    }


def export_capability_mapping_for_rag(indexer: tal_code_indexer.TALCodeIndexer,
                                      output_path: str) -> None:
    """
    Export capability mappings in a format optimized for RAG retrieval.
    
    Args:
        indexer: Populated indexer instance
        output_path: Path to save the export file
    """
    import json
    
    rag_export = {
        'taxonomy_version': '1.0',
        'capabilities': []
    }
    
    for cap_id, capability in indexer.capabilities.items():
        functions = indexer.get_capability_functions(cap_id)
        
        cap_export = {
            'id': cap_id,
            'name': capability.name,
            'description': capability.description,
            'keywords': list(capability.keywords),
            'function_count': len(functions),
            'functions': []
        }
        
        # Export detailed function info
        for func in functions:
            func_export = {
                'name': func.name,
                'type': func.element_type,
                'file': func.file_path,
                'line': func.line_number,
                'signature': {
                    'return_type': func.return_type,
                    'parameters': func.parameters
                },
                'relationships': {
                    'calls': func.calls[:20],  # Limit for size
                    'caller_count': len(indexer.get_callers(func.name))
                },
                'content_preview': func.content[:200] if func.content else '',
                'other_capabilities': [
                    c for c in func.business_capabilities if c != cap_id
                ]
            }
            cap_export['functions'].append(func_export)
        
        rag_export['capabilities'].append(cap_export)
    
    # Add summary statistics
    rag_export['summary'] = get_capability_statistics(indexer)
    
    with open(output_path, 'w') as f:
        json.dump(rag_export, f, indent=2)
    
    print(f"✓ Exported RAG-optimized capability mapping to: {output_path}")


# Example usage
if __name__ == '__main__':
    print("Business Capability Mapper - Wire Processing Taxonomy")
    print("=" * 60)
    
    # Create and show taxonomy
    taxonomy = BusinessCapabilityTaxonomy()
    
    print(f"\nTaxonomy contains {len(taxonomy.CAPABILITIES)} categories:")
    for category_name, keywords in taxonomy.CAPABILITIES.items():
        print(f"  • {category_name}: {len(keywords)} keywords")
    
    print(f"\nSynonyms defined: {len(taxonomy.SYNONYMS)}")
    for key, values in taxonomy.SYNONYMS.items():
        print(f"  • {key} → {', '.join(values)}")
    
    # Convert to capability format
    capabilities = convert_taxonomy_to_capabilities(taxonomy)
    
    print(f"\n✓ Converted to {len(capabilities)} searchable capabilities")
    
    # Show example
    example_cap = list(capabilities.items())[0]
    print(f"\nExample Capability:")
    print(f"  ID: {example_cap[0]}")
    print(f"  Name: {example_cap[1]['name']}")
    print(f"  Keywords ({len(example_cap[1]['keywords'])}): {', '.join(list(example_cap[1]['keywords'])[:10])}...")
    
    print("\n" + "=" * 60)
    print("To use with indexer:")
    print("  from business_capability_mapper import register_taxonomy_with_indexer")
    print("  register_taxonomy_with_indexer(indexer)")
    print("  indexer.map_capabilities()")
