#!/usr/bin/env python3
"""
Complete Workflow Demo: Parse TAL Code -> Export Graph -> Generate Visualization

This script demonstrates the full pipeline:
1. Parse TAL files and build knowledge graph
2. Export graph data to JSON
3. Generate interactive HTML visualization
"""

import sys
from pathlib import Path

# Add necessary imports (adjust paths as needed)
try:
    from knowledge_graph import KnowledgeGraph
    from parsers_updated import export_for_visualization, export_knowledge_graph
    from graph_visualizer import generate_standalone_html
    print("✓ All modules imported successfully")
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("\nMake sure these files are in the same directory:")
    print("  - knowledge_graph.py")
    print("  - parsers_updated.py (or parsers.py with updated functions)")
    print("  - graph_visualizer.py")
    sys.exit(1)


def demo_workflow(tal_directory: str = None, output_dir: str = "./demo_output"):
    """
    Run complete workflow demonstration
    
    Args:
        tal_directory: Directory containing TAL files (optional for demo)
        output_dir: Output directory for exports and visualizations
    """
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║           Knowledge Graph Workflow Demonstration                     ║
╚══════════════════════════════════════════════════════════════════════╝

This demo shows the complete pipeline from parsing to visualization.
    """)
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # ========================================================================
    # STEP 1: Create Sample Knowledge Graph
    # ========================================================================
    
    print("\n" + "="*70)
    print("STEP 1: Creating Sample Knowledge Graph")
    print("="*70 + "\n")
    
    kg = KnowledgeGraph(backend="networkx")
    
    if tal_directory and Path(tal_directory).exists():
        print(f"Parsing TAL files from: {tal_directory}")
        # Here you would call your TAL parsing code
        # For demo, we'll create sample data
    else:
        print("Creating sample graph data for demonstration...\n")
        create_sample_graph(kg)
    
    stats = kg.get_statistics()
    print(f"\nGraph created successfully!")
    print(f"  Entities: {stats['total_entities']}")
    print(f"  Relationships: {stats['total_relationships']}")
    
    # ========================================================================
    # STEP 2: Export Graph Data
    # ========================================================================
    
    print("\n" + "="*70)
    print("STEP 2: Exporting Graph Data")
    print("="*70 + "\n")
    
    vis_file = export_knowledge_graph(kg, output_dir=output_dir)
    
    print(f"\n✓ Graph data exported successfully")
    print(f"  Visualization data: {vis_file}")
    
    # ========================================================================
    # STEP 3: Generate HTML Visualization
    # ========================================================================
    
    print("\n" + "="*70)
    print("STEP 3: Generating Interactive HTML Visualization")
    print("="*70 + "\n")
    
    html_file = output_path / "graph_visualization.html"
    generate_standalone_html(
        json_file=vis_file,
        output_file=str(html_file),
        title="Demo Knowledge Graph"
    )
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    print("\n" + "="*70)
    print("✓ WORKFLOW COMPLETE!")
    print("="*70)
    
    print(f"""
Generated Files:
  📄 Full graph data:      {output_path / 'knowledge_graph.json'}
  📄 Visualization data:   {output_path / 'graph_data.json'}
  📄 Procedures summary:   {output_path / 'procedures.json'}
  📄 Call graph:           {output_path / 'call_graph.json'}
  📄 Statistics:           {output_path / 'statistics.json'}
  
  🌐 Interactive HTML:     {html_file}

Next Steps:
  1. Open {html_file} in your web browser
  2. Use the interactive controls to explore the graph
  3. Search for nodes, filter by type, zoom and pan
  4. Click nodes to highlight connections
  5. Export to SVG if needed

Real Usage:
  # Parse your TAL code
  python parsers.py ./your_tal_directory --export ./output
  
  # Generate visualization
  python graph_visualizer.py ./output/graph_data.json
  
  # Open graph_visualization.html in browser
    """)
    
    print("="*70 + "\n")


def create_sample_graph(kg: KnowledgeGraph):
    """Create sample graph data for demonstration"""
    from knowledge_graph import Entity, Relationship, EntityType, RelationType
    
    # Create file entity
    file1 = Entity(
        id="",
        type=EntityType.FILE,
        name="payment_processor.tal",
        qualified_name="payment_processor.tal",
        file_path="/path/to/payment_processor.tal",
        language="TAL",
        metadata={'extension': '.tal'}
    )
    kg.add_entity(file1)
    
    # Create main procedure
    main_proc = Entity(
        id="",
        type=EntityType.PROCEDURE,
        name="PROCESS_PAYMENT",
        qualified_name="payment_processor.tal::PROCESS_PAYMENT",
        file_path="/path/to/payment_processor.tal",
        start_line=10,
        language="TAL",
        metadata={
            'is_main': True,
            'return_type': 'INT',
            'parameter_count': 3,
            'statement_count': 45
        }
    )
    kg.add_entity(main_proc)
    
    # Create helper procedures
    validate_proc = Entity(
        id="",
        type=EntityType.PROCEDURE,
        name="VALIDATE_AMOUNT",
        qualified_name="payment_processor.tal::VALIDATE_AMOUNT",
        file_path="/path/to/payment_processor.tal",
        start_line=60,
        language="TAL",
        metadata={
            'return_type': 'INT',
            'parameter_count': 1,
            'statement_count': 15
        }
    )
    kg.add_entity(validate_proc)
    
    check_proc = Entity(
        id="",
        type=EntityType.PROCEDURE,
        name="CHECK_BALANCE",
        qualified_name="payment_processor.tal::CHECK_BALANCE",
        file_path="/path/to/payment_processor.tal",
        start_line=80,
        language="TAL",
        metadata={
            'return_type': 'INT',
            'parameter_count': 2,
            'statement_count': 20
        }
    )
    kg.add_entity(check_proc)
    
    # External reference
    ext_proc = Entity(
        id="",
        type=EntityType.PROCEDURE,
        name="UPDATE_LEDGER",
        qualified_name="external::UPDATE_LEDGER",
        language="TAL",
        metadata={
            'is_external': True,
            'return_type': 'INT'
        }
    )
    kg.add_entity(ext_proc)
    
    # Create some variables
    var1 = Entity(
        id="",
        type=EntityType.VARIABLE,
        name="payment_amount",
        qualified_name="payment_processor.tal::PROCESS_PAYMENT::payment_amount",
        file_path="/path/to/payment_processor.tal",
        start_line=12,
        language="TAL",
        metadata={
            'data_type': 'INT',
            'scope': 'local'
        }
    )
    kg.add_entity(var1)
    
    var2 = Entity(
        id="",
        type=EntityType.VARIABLE,
        name="account_balance",
        qualified_name="payment_processor.tal::PROCESS_PAYMENT::account_balance",
        file_path="/path/to/payment_processor.tal",
        start_line=13,
        language="TAL",
        metadata={
            'data_type': 'FIXED(10,2)',
            'scope': 'local'
        }
    )
    kg.add_entity(var2)
    
    # Create relationships
    kg.add_relationship(Relationship(
        source_id=file1.id,
        target_id=main_proc.id,
        type=RelationType.DEFINES
    ))
    
    kg.add_relationship(Relationship(
        source_id=main_proc.id,
        target_id=validate_proc.id,
        type=RelationType.CALLS,
        metadata={'line': 25}
    ))
    
    kg.add_relationship(Relationship(
        source_id=main_proc.id,
        target_id=check_proc.id,
        type=RelationType.CALLS,
        metadata={'line': 30}
    ))
    
    kg.add_relationship(Relationship(
        source_id=main_proc.id,
        target_id=ext_proc.id,
        type=RelationType.CALLS,
        metadata={'line': 40, 'external': True}
    ))
    
    kg.add_relationship(Relationship(
        source_id=main_proc.id,
        target_id=var1.id,
        type=RelationType.CONTAINS
    ))
    
    kg.add_relationship(Relationship(
        source_id=main_proc.id,
        target_id=var2.id,
        type=RelationType.CONTAINS
    ))
    
    print("Sample entities created:")
    print(f"  • 1 file")
    print(f"  • 4 procedures (1 main, 1 external)")
    print(f"  • 2 variables")
    print(f"  • 6 relationships")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Demo: Complete knowledge graph workflow')
    parser.add_argument('--tal-dir', help='Optional: Directory with TAL files to parse')
    parser.add_argument('--output', default='./demo_output', help='Output directory')
    
    args = parser.parse_args()
    
    demo_workflow(tal_directory=args.tal_dir, output_dir=args.output)
