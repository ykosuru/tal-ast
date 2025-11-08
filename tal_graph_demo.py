#!/usr/bin/env python3
"""
Demo Script for Enhanced TAL Knowledge Graph System

This script demonstrates all three new features:
1. GraphViz visualization
2. Enhanced LLM context for code rewriting
3. Fixed Kuzu persistence

Run this after setting up the necessary TAL parser modules.
"""

import sys
from pathlib import Path

def print_banner(text):
    """Print a formatted banner"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")


def demo_feature_1_visualization():
    """Demonstrate GraphViz visualization"""
    print_banner("FEATURE 1: GraphViz Visualization")
    
    print("""
This feature generates visual diagrams of your TAL code structure:

1. Full Graph - Complete knowledge graph with all entities
2. Call Graph - Procedure call relationships
3. File Structure - Organization of code by file
4. Procedure Context - Detailed view of specific procedures

Example Usage:
""")
    
    code = """
from graph_visualizer import visualize_knowledge_graph, KnowledgeGraphVisualizer
from knowledge_graph import KnowledgeGraph

# After parsing your TAL code...
kg = KnowledgeGraph()

# Generate all standard visualizations
viz_paths = visualize_knowledge_graph(
    kg,
    output_dir="./visualizations",
    format="png"
)

# Or create specific visualizations
visualizer = KnowledgeGraphVisualizer(kg)

# Full graph
visualizer.visualize_full_graph(
    "full_graph",
    max_nodes=200,
    include_files=False
)

# Call graph
visualizer.visualize_call_graph(
    "call_graph",
    main_only=True,
    max_depth=5
)

# Specific procedure context
visualizer.visualize_procedure_subgraph(
    "PROCESS_PAYMENT",
    depth=2
)
"""
    
    print(code)
    print("\nOutput: PNG/SVG/PDF files in ./visualizations/")
    print("Example: full_graph.png, call_graph.png, file_structure.png")


def demo_feature_2_llm_context():
    """Demonstrate enhanced LLM context"""
    print_banner("FEATURE 2: Enhanced LLM Context for Code Rewriting")
    
    print("""
This feature generates comprehensive context for LLMs to help rewrite TAL code
in other languages (Java, Python, etc.).

What's Included:
- Procedure signatures with actual code
- Business logic summaries
- Call chain analysis
- Data structure definitions
- Language-specific migration hints

Example Usage:
""")
    
    code = """
from code_rewriting_context import create_rewriting_context
from knowledge_graph import KnowledgeGraph

# After parsing your TAL code...
kg = KnowledgeGraph()

# Generate context for Java migration
context = create_rewriting_context(
    kg,
    search_term="drawdown",
    target_language="Java",
    output_file="drawdown_migration.md"
)

# The context includes:
# - All procedures related to "drawdown"
# - Actual TAL source code snippets
# - Call chains and dependencies
# - Java-specific migration hints
# - Data structure mappings

# Use this context with Claude/GPT-4:
# "Using the following TAL code context, rewrite it in Java..."
"""
    
    print(code)
    print("\nOutput: Markdown file ready for LLM consumption")
    print("Example content:")
    print("""
# TAL to Java Migration Context
## Functionality: `drawdown`

### 1. `PROCESS_DRAWDOWN`
| Property | Value |
|----------|-------|
| Return Type | INT |
| Parameters | account_id, amount |
| Statements | 45 |

**Source Code:**
```tal
PROC PROCESS_DRAWDOWN(account_id, amount);
  INT status;
  CALL VALIDATE_ACCOUNT(account_id);
  ...
END;
```

## Migration Hints for Java
- Convert procedures to service methods
- Use BigDecimal for money amounts
- Implement proper exception handling
""")


def demo_feature_3_kuzu_fix():
    """Demonstrate Kuzu fix"""
    print_banner("FEATURE 3: Fixed Kuzu Persistence")
    
    print("""
Previous Issue:
  "Database path cannot be a directory" error

Fix Applied:
  - Proper database path handling
  - Automatic cleanup of conflicting schemas
  - Stable persistence layer

Example Usage:
""")
    
    code = """
from knowledge_graph import KnowledgeGraph

# Old way (would fail):
# kg = KnowledgeGraph(backend="kuzu")  # ERROR!

# New way (works perfectly):
kg = KnowledgeGraph(
    backend="kuzu",
    db_path="./my_knowledge_graph"
)

# Parse and add data
# ... data is now persisted in Kuzu ...

# Reconnect later - data is still there!
kg2 = KnowledgeGraph(
    backend="kuzu",
    db_path="./my_knowledge_graph"
)
stats = kg2.get_statistics()  # Loads persisted data

# Or use the fix module directly
from kuzu_fix import initialize_kuzu_database

db, conn = initialize_kuzu_database(
    "./my_db",
    clean=True  # Clean start
)
"""
    
    print(code)
    print("\nBenefits:")
    print("✓ Persistent storage across sessions")
    print("✓ No more path errors")
    print("✓ Production-ready database")


def demo_complete_workflow():
    """Demonstrate complete integrated workflow"""
    print_banner("COMPLETE WORKFLOW: All Features Together")
    
    print("""
Use all three features in a single workflow:
""")
    
    code = """
from kg_integration import EnhancedKnowledgeGraphWorkflow

# 1. Initialize with Kuzu backend (Feature 3)
workflow = EnhancedKnowledgeGraphWorkflow(
    backend="kuzu",
    db_path="./payment_system_kg"
)

# 2. Parse TAL files
results = workflow.parse_directory(
    "./tal_source/payment_system",
    recursive=True,
    resolve_refs=True
)

print(f"Parsed {results['total_entities']} entities")

# 3. Generate visualizations (Feature 1)
viz_paths = workflow.visualize(
    output_dir="./visualizations",
    format="png"
)

print(f"Generated visualizations:")
for viz_type, path in viz_paths.items():
    print(f"  {viz_type}: {path}")

# 4. Generate migration context (Feature 2)
context = workflow.generate_migration_context(
    search_term="drawdown",
    target_language="Java",
    output_file="drawdown_to_java.md"
)

print(f"Migration context: {len(context)} characters")

# 5. Search for functionality
search_results = workflow.search_functionality("payment")

# 6. Export everything
workflow.export_graph(output_dir="./export")

print("Workflow complete!")
print("  - Knowledge graph persisted in Kuzu")
print("  - Visualizations in ./visualizations/")
print("  - Migration context in drawdown_to_java.md")
print("  - Graph data exported to ./export/")
"""
    
    print(code)


def demo_command_line():
    """Demonstrate command line usage"""
    print_banner("COMMAND LINE USAGE")
    
    print("Quick one-liners for common tasks:\n")
    
    commands = [
        ("Parse and visualize", 
         "python kg_integration.py ./tal_source --visualize"),
        
        ("Generate Java migration context",
         "python kg_integration.py ./tal_source --migrate drawdown --target-language Java"),
        
        ("Use Kuzu backend",
         "python kg_integration.py ./tal_source --backend kuzu --visualize"),
        
        ("Complete workflow",
         "python kg_integration.py ./tal_source --visualize --migrate payment --export"),
        
        ("Search only",
         "python kg_integration.py ./tal_source --search drawdown"),
        
        ("Export to SVG",
         "python kg_integration.py ./tal_source --visualize --viz-format svg"),
    ]
    
    for task, cmd in commands:
        print(f"{task}:")
        print(f"  $ {cmd}\n")


def main():
    """Run the demo"""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║     Enhanced TAL Knowledge Graph System - Feature Demonstration      ║
╚══════════════════════════════════════════════════════════════════════╝

This demo shows the three new features and how to use them.
""")
    
    if len(sys.argv) > 1 and sys.argv[1] == "--run-example":
        print("\nRunning example workflow...\n")
        
        # Check if we have the necessary modules
        try:
            from kg_integration import EnhancedKnowledgeGraphWorkflow
            print("✓ Modules loaded successfully")
            
            # You would run an actual example here with real TAL files
            print("\nTo run with your TAL files:")
            print("  python demo.py <tal_directory>")
            
        except ImportError as e:
            print(f"✗ Missing module: {e}")
            print("\nMake sure all required files are present:")
            print("  - knowledge_graph.py")
            print("  - parsers.py")
            print("  - graph_visualizer.py")
            print("  - code_rewriting_context.py")
            print("  - kuzu_fix.py")
            print("  - kg_integration.py")
    else:
        # Show feature demos
        demo_feature_1_visualization()
        input("\nPress Enter to continue...")
        
        demo_feature_2_llm_context()
        input("\nPress Enter to continue...")
        
        demo_feature_3_kuzu_fix()
        input("\nPress Enter to continue...")
        
        demo_complete_workflow()
        input("\nPress Enter to continue...")
        
        demo_command_line()
        
        print("\n" + "="*70)
        print("  Demo Complete!")
        print("="*70)
        print("\nNext Steps:")
        print("  1. Review the code examples above")
        print("  2. Install required dependencies (see README.md)")
        print("  3. Run: python kg_integration.py <your_tal_directory>")
        print("\nFor help:")
        print("  python kg_integration.py --help")
        print()


if __name__ == "__main__":
    main()
