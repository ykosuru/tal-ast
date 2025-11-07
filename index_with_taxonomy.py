"""
TAL Indexer with Real Wire Processing Taxonomy - Complete Example

This example shows how to use your actual BusinessCapabilityTaxonomy
instead of the simplified example taxonomy.
"""

import sys
import os
from pathlib import Path

# Import the parsers and indexer
import tal_proc_parser
import enhanced_tal_parser
import tal_code_indexer
from business_capability_mapper import (
    register_taxonomy_with_indexer,
    print_capability_coverage_report,
    find_cross_capability_functions,
    search_by_multiple_capabilities,
    export_capability_mapping_for_rag,
    get_capability_statistics
)


def main():
    """
    Main example showing real taxonomy usage.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Index TAL code with Wire Processing Taxonomy',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /path/to/tal/code                 # Index with real taxonomy
  %(prog)s /path/to/tal/code -o ./output     # Index and export
  %(prog)s --load index.pkl --report         # Load and show coverage
        """
    )
    
    parser.add_argument('directory', nargs='?', help='Directory containing TAL files')
    parser.add_argument('-o', '--output', default='./wire_processing_index',
                       help='Output directory for exports')
    parser.add_argument('--load', help='Load existing binary index')
    parser.add_argument('--report', action='store_true',
                       help='Show capability coverage report')
    parser.add_argument('--cross-cap', action='store_true',
                       help='Show functions implementing multiple capabilities')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("TAL CODE INDEXER - WIRE PROCESSING TAXONOMY")
    print("=" * 60)
    
    # Create indexer
    indexer = tal_code_indexer.TALCodeIndexer()
    
    # Register the real taxonomy (not the simple example)
    register_taxonomy_with_indexer(indexer)
    
    # Load existing index or parse directory
    if args.load:
        print(f"\nLoading existing index from: {args.load}")
        indexer = tal_code_indexer.TALCodeIndexer.load_binary(args.load)
        print("✓ Index loaded successfully")
    
    elif args.directory:
        if not os.path.exists(args.directory):
            print(f"Error: Directory '{args.directory}' not found")
            return 1
        
        # Parse and index all TAL files
        directory = Path(args.directory)
        tal_files = list(directory.glob('**/*.tal')) + list(directory.glob('**/*.TAL'))
        
        if not tal_files:
            print(f"No TAL files found in {args.directory}")
            return 1
        
        print(f"\nFound {len(tal_files)} TAL files")
        print("=" * 60)
        
        parser_instance = enhanced_tal_parser.EnhancedTALParser()
        
        for i, tal_file in enumerate(tal_files, 1):
            print(f"\n[{i}/{len(tal_files)}] Processing: {tal_file.name}")
            
            try:
                result = parser_instance.parse_file(str(tal_file))
                
                if result.get('success') and result.get('ast'):
                    indexer.index_ast(result['ast'], str(tal_file))
                    print(f"  ✓ Successfully indexed")
                else:
                    print(f"  ⚠ Parse had issues, indexed partial results")
                    if result.get('ast'):
                        indexer.index_ast(result['ast'], str(tal_file))
            
            except Exception as e:
                print(f"  ✗ Error: {e}")
                continue
        
        # Map capabilities after indexing
        print("\n" + "=" * 60)
        print("MAPPING BUSINESS CAPABILITIES")
        print("=" * 60)
        indexer.map_capabilities()
    
    else:
        parser.print_help()
        return 1
    
    # Print statistics
    indexer.print_statistics()
    
    # Show capability coverage report
    if args.report or not args.load:
        print_capability_coverage_report(indexer)
    
    # Show cross-capability functions (orchestrators)
    if args.cross_cap:
        print("\n" + "=" * 60)
        print("CROSS-CAPABILITY FUNCTIONS (Orchestrators)")
        print("=" * 60)
        
        cross_cap_funcs = find_cross_capability_functions(indexer, min_capabilities=2)
        
        if cross_cap_funcs:
            print(f"\nFound {len(cross_cap_funcs)} functions implementing multiple capabilities:")
            print("-" * 60)
            
            for i, func in enumerate(cross_cap_funcs[:20], 1):
                print(f"\n{i}. {func['name']} ({func['capability_count']} capabilities)")
                print(f"   File: {Path(func['file']).name}:{func['line']}")
                print(f"   Capabilities:")
                for cap_id in func['capabilities']:
                    if cap_id in indexer.capabilities:
                        cap_name = indexer.capabilities[cap_id].name
                        print(f"     • {cap_name}")
            
            if len(cross_cap_funcs) > 20:
                print(f"\n   ... and {len(cross_cap_funcs) - 20} more")
        else:
            print("\nNo cross-capability functions found")
    
    # Example searches with real taxonomy
    print("\n" + "=" * 60)
    print("EXAMPLE SEARCHES WITH REAL TAXONOMY")
    print("=" * 60)
    
    example_queries = [
        ("OFAC Sanctions", "ofac sanctions screening"),
        ("Wire Transfer Processing", "wire transfer payment processing"),
        ("ISO 20022", "pacs.008 iso20022 message"),
        ("Payment Repair", "payment repair ace auto"),
        ("CHIPS Network", "chips clearing network"),
        ("Risk Controls", "fraud checking aml controls")
    ]
    
    for query_name, query_text in example_queries:
        print(f"\n{query_name}: '{query_text}'")
        print("-" * 40)
        
        results = indexer.search(query_text, top_k=5)
        
        if results:
            for rank, (name, score, element) in enumerate(results, 1):
                print(f"  {rank}. {name} (score: {score:.3f})")
                if element.business_capabilities:
                    caps = [indexer.capabilities[c].name for c in element.business_capabilities 
                           if c in indexer.capabilities]
                    if caps:
                        print(f"     Categories: {', '.join(caps[:3])}")
        else:
            print("  No results found")
    
    # Export everything
    output_path = Path(args.output)
    output_path.mkdir(exist_ok=True)
    
    print("\n" + "=" * 60)
    print("EXPORTING RESULTS")
    print("=" * 60)
    
    # 1. Binary index for fast loading
    pickle_path = output_path / "wire_processing_index.pkl"
    indexer.save_binary(str(pickle_path))
    print(f"✓ Binary index: {pickle_path}")
    
    # 2. JSON index for integration
    json_path = output_path / "wire_processing_index.json"
    indexer.export_index(str(json_path))
    print(f"✓ JSON index: {json_path}")
    
    # 3. RAG-optimized capability mapping
    rag_path = output_path / "capability_mapping_rag.json"
    export_capability_mapping_for_rag(indexer, str(rag_path))
    print(f"✓ RAG capability mapping: {rag_path}")
    
    # 4. Statistics report
    stats_path = output_path / "capability_statistics.json"
    import json
    stats = get_capability_statistics(indexer)
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"✓ Statistics: {stats_path}")
    
    print("\n" + "=" * 60)
    print("INDEXING COMPLETE!")
    print("=" * 60)
    print(f"\nYour indexed codebase is ready for:")
    print("  • Keyword search (TF-IDF)")
    print("  • Business capability queries")
    print("  • Call graph analysis")
    print("  • RAG system integration")
    print(f"\nTo query interactively:")
    print(f"  python tal_query_interface.py {pickle_path}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
