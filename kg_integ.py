"""
Knowledge Graph Integration Module
Combines all enhancements: visualization, code rewriting context, and Kuzu fixes

This module provides a unified interface for:
1. Graph visualization with GraphViz
2. Enhanced LLM context for code rewriting
3. Fixed Kuzu persistence
4. All existing functionality preserved
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# Import core modules (these need to be in the same directory or Python path)
try:
    from knowledge_graph import KnowledgeGraph, EntityType, RelationType
    from parsers import (
        parse_tal_directory_recursive,
        resolve_external_references,
        display_parsing_summary,
        export_knowledge_graph,
        KnowledgeGraphSearch
    )
except ImportError as e:
    print(f"Error importing core modules: {e}")
    print("Make sure knowledge_graph.py and parsers.py are in the same directory")
    sys.exit(1)

# Import new modules
try:
    from graph_visualizer import visualize_knowledge_graph, KnowledgeGraphVisualizer
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False
    print("Warning: graph_visualizer not available. Visualization features disabled.")

try:
    from code_rewriting_context import create_rewriting_context, CodeRewritingContextBuilder
    REWRITING_CONTEXT_AVAILABLE = True
except ImportError:
    REWRITING_CONTEXT_AVAILABLE = False
    print("Warning: code_rewriting_context not available. Enhanced context features disabled.")

try:
    from kuzu_fix import initialize_kuzu_database, patch_kuzu_database, verify_kuzu_installation
    patch_kuzu_database()  # Apply fix immediately
    KUZU_FIX_AVAILABLE = True
except ImportError:
    KUZU_FIX_AVAILABLE = False
    print("Warning: kuzu_fix not available. Kuzu fixes not applied.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Enhanced Knowledge Graph Workflow
# ============================================================================

class EnhancedKnowledgeGraphWorkflow:
    """
    Complete workflow for parsing, visualizing, and generating migration context
    """
    
    def __init__(self, backend: str = "networkx", db_path: Optional[str] = None):
        """
        Initialize workflow
        
        Args:
            backend: Database backend ('networkx' or 'kuzu')
            db_path: Path for Kuzu database (if using Kuzu)
        """
        self.backend = backend
        self.db_path = db_path or "./knowledge_graph_db"
        self.kg = None
        self.search = None
        
        logger.info(f"Initializing Enhanced Knowledge Graph Workflow")
        logger.info(f"Backend: {backend}")
        
        # Verify Kuzu if needed
        if backend == "kuzu":
            if not verify_kuzu_installation():
                logger.error("Kuzu not installed. Install with: pip install kuzu")
                raise ImportError("Kuzu not available")
    
    def parse_directory(self, 
                       directory: str,
                       recursive: bool = True,
                       resolve_refs: bool = True) -> Dict[str, Any]:
        """
        Parse TAL directory and build knowledge graph
        
        Args:
            directory: Directory containing TAL files
            recursive: Search subdirectories
            resolve_refs: Resolve external references
        
        Returns:
            Parsing results dictionary
        """
        logger.info(f"Starting directory parsing: {directory}")
        
        # Create knowledge graph
        self.kg = KnowledgeGraph(backend=self.backend, db_path=self.db_path)
        
        # Parse files
        results = parse_tal_directory_recursive(directory, self.kg, recursive=recursive)
        
        # Resolve external references
        resolution = None
        if resolve_refs and results.get('parsed_successfully'):
            logger.info("Resolving external references...")
            resolution = resolve_external_references(self.kg)
            results['resolution'] = resolution
        
        # Initialize search
        self.search = KnowledgeGraphSearch(self.kg)
        
        # Display summary
        display_parsing_summary(results, resolution)
        
        return results
    
    def visualize(self,
                 output_dir: str = "./visualizations",
                 format: str = "png",
                 create_all: bool = True) -> Dict[str, str]:
        """
        Generate visualizations
        
        Args:
            output_dir: Output directory for visualizations
            format: Output format (png, svg, pdf)
            create_all: Create all standard visualizations
        
        Returns:
            Dict mapping visualization type to file path
        """
        if not GRAPHVIZ_AVAILABLE:
            logger.error("GraphViz module not available")
            return {}
        
        if not self.kg:
            logger.error("No knowledge graph available. Parse directory first.")
            return {}
        
        logger.info("Generating visualizations...")
        
        if create_all:
            return visualize_knowledge_graph(self.kg, output_dir, format)
        else:
            # Just create basic visualization
            visualizer = KnowledgeGraphVisualizer(self.kg)
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True, parents=True)
            
            result_path = visualizer.visualize_full_graph(
                output_file=str(output_path / "knowledge_graph"),
                format=format,
                max_nodes=200
            )
            return {'full_graph': result_path}
    
    def generate_migration_context(self,
                                   search_term: str,
                                   target_language: str = "Java",
                                   output_file: Optional[str] = None) -> str:
        """
        Generate enhanced LLM context for code migration
        
        Args:
            search_term: Functionality to search for
            target_language: Target language (Java, Python, etc.)
            output_file: Optional file to save context
        
        Returns:
            Formatted migration context
        """
        if not REWRITING_CONTEXT_AVAILABLE:
            logger.error("Code rewriting context module not available")
            return ""
        
        if not self.kg:
            logger.error("No knowledge graph available. Parse directory first.")
            return ""
        
        logger.info(f"Generating migration context for: {search_term}")
        
        context = create_rewriting_context(
            self.kg,
            search_term,
            target_language,
            output_file
        )
        
        return context
    
    def search_functionality(self, search_term: str) -> Dict[str, Any]:
        """
        Search for functionality in knowledge graph
        
        Args:
            search_term: Term to search for
        
        Returns:
            Search results
        """
        if not self.search:
            logger.error("Search not initialized. Parse directory first.")
            return {}
        
        logger.info(f"Searching for: {search_term}")
        results = self.search.find_by_functionality(search_term)
        self.search.display_search_results(results)
        
        return results
    
    def export_graph(self, output_dir: str = "./output"):
        """
        Export knowledge graph in multiple formats
        
        Args:
            output_dir: Output directory
        """
        if not self.kg:
            logger.error("No knowledge graph available. Parse directory first.")
            return
        
        export_knowledge_graph(self.kg, output_dir)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge graph statistics"""
        if not self.kg:
            return {}
        return self.kg.get_statistics()


# ============================================================================
# Command Line Interface
# ============================================================================

def main():
    """Enhanced CLI for knowledge graph analysis"""
    
    parser = argparse.ArgumentParser(
        description='Enhanced TAL Knowledge Graph Analysis with Visualization and Migration Context',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Parse, visualize, and search
  python kg_integration.py ./tal_source --visualize --search drawdown
  
  # Generate migration context for Java
  python kg_integration.py ./tal_source --migrate drawdown --target-language Java
  
  # Use Kuzu backend (with fix)
  python kg_integration.py ./tal_source --backend kuzu --visualize
  
  # Complete workflow
  python kg_integration.py ./tal_source --visualize --export --migrate payment
        """
    )
    
    # Required arguments
    parser.add_argument('directory', help='Directory containing TAL files')
    
    # Backend options
    parser.add_argument('--backend', default='networkx', choices=['networkx', 'kuzu'],
                       help='Graph database backend (default: networkx)')
    parser.add_argument('--db-path', help='Path for Kuzu database (default: ./knowledge_graph_db)')
    
    # Parsing options
    parser.add_argument('--no-recursive', action='store_true',
                       help="Don't search subdirectories")
    parser.add_argument('--no-resolve', action='store_true',
                       help="Don't resolve external references")
    
    # Visualization options
    parser.add_argument('--visualize', action='store_true',
                       help='Generate graph visualizations')
    parser.add_argument('--viz-dir', default='./visualizations',
                       help='Output directory for visualizations (default: ./visualizations)')
    parser.add_argument('--viz-format', default='png', choices=['png', 'svg', 'pdf'],
                       help='Visualization format (default: png)')
    
    # Migration context options
    parser.add_argument('--migrate', metavar='TERM',
                       help='Generate migration context for functionality')
    parser.add_argument('--target-language', default='Java',
                       help='Target language for migration (default: Java)')
    parser.add_argument('--context-output', help='Output file for migration context')
    
    # Search options
    parser.add_argument('--search', metavar='TERM',
                       help='Search for functionality after parsing')
    
    # Export options
    parser.add_argument('--export', action='store_true',
                       help='Export knowledge graph data')
    parser.add_argument('--export-dir', default='./output',
                       help='Output directory for exports (default: ./output)')
    
    args = parser.parse_args()
    
    # Validate directory
    dir_path = Path(args.directory)
    if not dir_path.exists():
        print(f"Error: Directory '{args.directory}' does not exist")
        sys.exit(1)
    
    try:
        # Initialize workflow
        workflow = EnhancedKnowledgeGraphWorkflow(
            backend=args.backend,
            db_path=args.db_path
        )
        
        # Parse directory
        print(f"\n{'='*70}")
        print("STEP 1: PARSING TAL FILES")
        print(f"{'='*70}\n")
        
        results = workflow.parse_directory(
            args.directory,
            recursive=not args.no_recursive,
            resolve_refs=not args.no_resolve
        )
        
        if not results.get('success'):
            print(f"Error: Parsing failed")
            sys.exit(1)
        
        # Generate visualizations
        if args.visualize:
            print(f"\n{'='*70}")
            print("STEP 2: GENERATING VISUALIZATIONS")
            print(f"{'='*70}\n")
            
            viz_paths = workflow.visualize(
                output_dir=args.viz_dir,
                format=args.viz_format
            )
            
            if viz_paths:
                print("\nGenerated visualizations:")
                for viz_type, path in viz_paths.items():
                    print(f"  • {viz_type}: {path}")
        
        # Export graph data
        if args.export:
            print(f"\n{'='*70}")
            print("STEP 3: EXPORTING GRAPH DATA")
            print(f"{'='*70}\n")
            
            workflow.export_graph(args.export_dir)
        
        # Generate migration context
        if args.migrate:
            print(f"\n{'='*70}")
            print("STEP 4: GENERATING MIGRATION CONTEXT")
            print(f"{'='*70}\n")
            
            context_file = args.context_output or f"{args.migrate}_migration_context.md"
            
            context = workflow.generate_migration_context(
                args.migrate,
                args.target_language,
                context_file
            )
            
            if context:
                print(f"\nMigration context generated:")
                print(f"  Functionality: {args.migrate}")
                print(f"  Target Language: {args.target_language}")
                print(f"  Output File: {context_file}")
                print(f"  Size: {len(context)} characters")
        
        # Search functionality
        if args.search:
            print(f"\n{'='*70}")
            print("STEP 5: SEARCHING KNOWLEDGE GRAPH")
            print(f"{'='*70}\n")
            
            workflow.search_functionality(args.search)
        
        # Final statistics
        print(f"\n{'='*70}")
        print("WORKFLOW COMPLETE")
        print(f"{'='*70}\n")
        
        stats = workflow.get_statistics()
        print("Knowledge Graph Statistics:")
        print(f"  Total Entities: {stats.get('total_entities', 0)}")
        print(f"  Total Relationships: {stats.get('total_relationships', 0)}")
        
        print("\nNext steps:")
        if not args.visualize:
            print(f"  • Generate visualizations: --visualize")
        if not args.migrate:
            print(f"  • Generate migration context: --migrate <functionality>")
        if not args.export:
            print(f"  • Export graph data: --export")
        
        print("\n" + "="*70 + "\n")
        
    except KeyboardInterrupt:
        print("\n\nWorkflow interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
