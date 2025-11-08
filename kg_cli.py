#!/usr/bin/env python3
"""
Knowledge Graph CLI Tool
Command-line interface for code analysis using the knowledge graph

Usage:
    python kg_cli.py init --backend networkx
    python kg_cli.py parse --file code.tal --language TAL --ast ast.json
    python kg_cli.py query calls --procedure MAIN_PROC
    python kg_cli.py analyze impact --procedure VALIDATE_ACCOUNT
    python kg_cli.py export rag --output embeddings.json
    python kg_cli.py stats
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from knowledge_graph import KnowledgeGraph, EntityType
from parsers import ParserFactory, BatchParser, TALProcedure, COBOLParagraph
from query_utils import QueryBuilder, CodeMetricsAnalyzer, RAGExporter, GraphVisualizer


class KnowledgeGraphCLI:
    """CLI interface for knowledge graph operations"""
    
    def __init__(self):
        self.kg: Optional[KnowledgeGraph] = None
        self.kg_file = "knowledge_graph.json"
    
    def load_or_create_kg(self, backend: str = "networkx", db_path: Optional[str] = None):
        """Load existing graph or create new one"""
        kg_json = Path(self.kg_file)
        
        if kg_json.exists() and backend == "networkx":
            print(f"Loading existing graph from {self.kg_file}")
            self.kg = KnowledgeGraph(backend=backend)
            self.kg.load_from_json(str(kg_json))
        else:
            print(f"Creating new {backend} graph")
            self.kg = KnowledgeGraph(backend=backend, db_path=db_path)
        
        return self.kg
    
    def save_kg(self):
        """Save graph to JSON"""
        if self.kg and self.kg.backend == "networkx":
            self.kg.save_to_json(self.kg_file)
            print(f"Saved graph to {self.kg_file}")
    
    def cmd_init(self, args):
        """Initialize a new knowledge graph"""
        self.load_or_create_kg(args.backend, args.db_path)
        self.save_kg()
        print("✓ Knowledge graph initialized")
    
    def cmd_parse(self, args):
        """Parse a file and add to graph"""
        self.load_or_create_kg(args.backend)
        
        # Load AST from file or stdin
        if args.ast:
            with open(args.ast, 'r') as f:
                ast_data = json.load(f)
        elif args.ast_stdin:
            ast_data = json.load(sys.stdin)
        else:
            print("Error: Must provide --ast or --ast-stdin")
            return 1
        
        # Parse file
        parser = ParserFactory.create_parser(args.language, self.kg)
        parser.parse_file(args.file, ast_data)
        
        self.save_kg()
        print(f"✓ Parsed {args.file}")
        
        # Show stats
        stats = self.kg.get_statistics()
        print(f"  Total entities: {stats['total_entities']}")
        print(f"  Total relationships: {stats['total_relationships']}")
    
    def cmd_batch_parse(self, args):
        """Batch parse multiple files"""
        self.load_or_create_kg(args.backend)
        
        # Load file list
        with open(args.filelist, 'r') as f:
            file_data = json.load(f)
        
        # Parse files
        batch = BatchParser(self.kg)
        batch.parse_files(file_data)
        
        self.save_kg()
        print(f"✓ Parsed {len(file_data)} files")
    
    def cmd_query_calls(self, args):
        """Query procedure calls"""
        self.load_or_create_kg(args.backend)
        qb = QueryBuilder(self.kg)
        
        if args.direction == "callers":
            results = qb.find_procedures_calling(args.procedure)
            print(f"Procedures calling {args.procedure}:")
        else:
            results = qb.find_procedures_called_by(args.procedure)
            print(f"Procedures called by {args.procedure}:")
        
        for proc in results:
            print(f"  - {proc.name} ({proc.file_path}:{proc.start_line})")
        
        print(f"\nTotal: {len(results)}")
    
    def cmd_query_callgraph(self, args):
        """Generate call graph"""
        self.load_or_create_kg(args.backend)
        qb = QueryBuilder(self.kg)
        
        call_graph = qb.find_call_graph(args.procedure, depth=args.depth)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(call_graph, f, indent=2)
            print(f"✓ Call graph saved to {args.output}")
        else:
            print(json.dumps(call_graph, indent=2))
    
    def cmd_analyze_impact(self, args):
        """Analyze impact of changing a procedure"""
        self.load_or_create_kg(args.backend)
        qb = QueryBuilder(self.kg)
        
        impact = qb.get_impact_analysis(args.procedure)
        
        print(f"\n🎯 Impact Analysis: {args.procedure}")
        print("=" * 60)
        print(f"Direct impact: {impact['direct_impact_count']} procedures")
        print(f"Total impact: {impact['total_impact_count']} procedures")
        print(f"\nImpacted procedures:")
        for proc_name in impact['impacted_procedures'][:20]:
            print(f"  - {proc_name}")
        
        if len(impact['impacted_procedures']) > 20:
            print(f"  ... and {len(impact['impacted_procedures']) - 20} more")
        
        print(f"\nImpacted files:")
        for file_path, procs in impact['impacted_files'].items():
            print(f"  {file_path}: {len(procs)} procedures")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(impact, f, indent=2)
            print(f"\n✓ Impact analysis saved to {args.output}")
    
    def cmd_analyze_deadcode(self, args):
        """Find dead code"""
        self.load_or_create_kg(args.backend)
        qb = QueryBuilder(self.kg)
        
        unused = qb.find_unused_procedures()
        
        print(f"\n🔍 Dead Code Analysis")
        print("=" * 60)
        print(f"Found {len(unused)} unused procedures:\n")
        
        for proc in unused:
            print(f"  - {proc.name}")
            print(f"    File: {proc.file_path}")
            print(f"    Lines: {proc.start_line}-{proc.end_line}")
            print()
        
        if args.output:
            data = [
                {
                    'name': p.name,
                    'file': p.file_path,
                    'start_line': p.start_line,
                    'end_line': p.end_line
                }
                for p in unused
            ]
            with open(args.output, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"✓ Dead code report saved to {args.output}")
    
    def cmd_analyze_complexity(self, args):
        """Find complexity hotspots"""
        self.load_or_create_kg(args.backend)
        qb = QueryBuilder(self.kg)
        
        hotspots = qb.find_complexity_hotspots()[:args.limit]
        
        print(f"\n🔥 Top {len(hotspots)} Complexity Hotspots")
        print("=" * 60)
        
        for proc, complexity in hotspots:
            print(f"{complexity:3d} | {proc.name}")
            print(f"      {proc.file_path}:{proc.start_line}")
            print()
        
        if args.output:
            data = [
                {
                    'name': p.name,
                    'complexity': c,
                    'file': p.file_path,
                    'start_line': p.start_line
                }
                for p, c in hotspots
            ]
            with open(args.output, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"✓ Complexity report saved to {args.output}")
    
    def cmd_export_rag(self, args):
        """Export for RAG systems"""
        self.load_or_create_kg(args.backend)
        rag = RAGExporter(self.kg)
        
        documents = rag.export_for_embedding()
        
        with open(args.output, 'w') as f:
            json.dump(documents, f, indent=2)
        
        print(f"✓ Exported {len(documents)} documents to {args.output}")
        print(f"  Ready for embedding and vector database insertion")
    
    def cmd_export_graphviz(self, args):
        """Export to GraphViz"""
        self.load_or_create_kg(args.backend)
        viz = GraphVisualizer(self.kg)
        
        entity_ids = None
        if args.procedures:
            # Find specific procedures
            entity_ids = []
            qb = QueryBuilder(self.kg)
            for proc_name in args.procedures:
                procs = qb.find_by_name_pattern(proc_name, EntityType.PROCEDURE)
                entity_ids.extend([p.id for p in procs])
        
        viz.export_to_graphviz(entity_ids=entity_ids, output_file=args.output)
        print(f"✓ GraphViz export saved to {args.output}")
        print(f"  Render with: dot -Tpng {args.output} -o graph.png")
    
    def cmd_stats(self, args):
        """Show graph statistics"""
        self.load_or_create_kg(args.backend)
        
        stats = self.kg.get_statistics()
        analyzer = CodeMetricsAnalyzer(self.kg)
        summary = analyzer.get_summary_statistics()
        
        print("\n📊 Knowledge Graph Statistics")
        print("=" * 60)
        print(f"Total Entities: {stats['total_entities']}")
        print(f"Total Relationships: {stats['total_relationships']}")
        print(f"\nEntity Breakdown:")
        for entity_type, count in stats['entity_counts'].items():
            print(f"  {entity_type:20s}: {count:6d}")
        
        print(f"\nRelationship Breakdown:")
        for rel_type, count in stats['relationship_counts'].items():
            print(f"  {rel_type:20s}: {count:6d}")
        
        if 'total_lines_of_code' in summary:
            print(f"\nCode Metrics:")
            print(f"  Total LOC: {summary['total_lines_of_code']:,}")
            print(f"  Avg Procedure LOC: {summary['avg_procedure_loc']:.1f}")
            print(f"  Avg Complexity: {summary['avg_procedure_complexity']:.1f}")
    
    def cmd_list(self, args):
        """List entities"""
        self.load_or_create_kg(args.backend)
        
        entity_type = EntityType(args.type) if args.type else None
        entities = self.kg.query_entities(entity_type=entity_type)
        
        print(f"\n📋 Entities ({len(entities)} total)")
        print("=" * 60)
        
        for entity in entities[:args.limit]:
            print(f"{entity.type.value:15s} | {entity.name:40s} | {entity.file_path or ''}")
        
        if len(entities) > args.limit:
            print(f"\n... and {len(entities) - args.limit} more")


def main():
    parser = argparse.ArgumentParser(
        description="Knowledge Graph CLI for Code Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--backend', default='networkx', choices=['networkx', 'kuzu'],
                       help='Graph database backend')
    parser.add_argument('--db-path', help='Database path (for Kuzu backend)')
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # init command
    init_parser = subparsers.add_parser('init', help='Initialize knowledge graph')
    
    # parse command
    parse_parser = subparsers.add_parser('parse', help='Parse a file')
    parse_parser.add_argument('--file', required=True, help='Source file path')
    parse_parser.add_argument('--language', required=True, help='Language (TAL, COBOL, etc.)')
    parse_parser.add_argument('--ast', help='AST JSON file')
    parse_parser.add_argument('--ast-stdin', action='store_true', help='Read AST from stdin')
    
    # batch-parse command
    batch_parser = subparsers.add_parser('batch-parse', help='Batch parse files')
    batch_parser.add_argument('--filelist', required=True, help='JSON file with list of files to parse')
    
    # query commands
    query_parser = subparsers.add_parser('query', help='Query the graph')
    query_subparsers = query_parser.add_subparsers(dest='query_type')
    
    calls_parser = query_subparsers.add_parser('calls', help='Query procedure calls')
    calls_parser.add_argument('--procedure', required=True, help='Procedure name')
    calls_parser.add_argument('--direction', choices=['callers', 'callees'], default='callees')
    
    callgraph_parser = query_subparsers.add_parser('callgraph', help='Generate call graph')
    callgraph_parser.add_argument('--procedure', required=True, help='Start procedure')
    callgraph_parser.add_argument('--depth', type=int, default=3, help='Traversal depth')
    callgraph_parser.add_argument('--output', help='Output JSON file')
    
    # analyze commands
    analyze_parser = subparsers.add_parser('analyze', help='Analyze the graph')
    analyze_subparsers = analyze_parser.add_subparsers(dest='analyze_type')
    
    impact_parser = analyze_subparsers.add_parser('impact', help='Impact analysis')
    impact_parser.add_argument('--procedure', required=True, help='Procedure name')
    impact_parser.add_argument('--output', help='Output JSON file')
    
    deadcode_parser = analyze_subparsers.add_parser('deadcode', help='Find dead code')
    deadcode_parser.add_argument('--output', help='Output JSON file')
    
    complexity_parser = analyze_subparsers.add_parser('complexity', help='Find complexity hotspots')
    complexity_parser.add_argument('--limit', type=int, default=10, help='Number of results')
    complexity_parser.add_argument('--output', help='Output JSON file')
    
    # export commands
    export_parser = subparsers.add_parser('export', help='Export data')
    export_subparsers = export_parser.add_subparsers(dest='export_type')
    
    rag_parser = export_subparsers.add_parser('rag', help='Export for RAG systems')
    rag_parser.add_argument('--output', required=True, help='Output JSON file')
    
    graphviz_parser = export_subparsers.add_parser('graphviz', help='Export to GraphViz')
    graphviz_parser.add_argument('--output', default='graph.dot', help='Output DOT file')
    graphviz_parser.add_argument('--procedures', nargs='+', help='Specific procedures to include')
    
    # stats command
    stats_parser = subparsers.add_parser('stats', help='Show statistics')
    
    # list command
    list_parser = subparsers.add_parser('list', help='List entities')
    list_parser.add_argument('--type', help='Entity type filter')
    list_parser.add_argument('--limit', type=int, default=50, help='Max results')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    cli = KnowledgeGraphCLI()
    
    command_map = {
        'init': cli.cmd_init,
        'parse': cli.cmd_parse,
        'batch-parse': cli.cmd_batch_parse,
        'stats': cli.cmd_stats,
        'list': cli.cmd_list,
    }
    
    if args.command == 'query':
        if args.query_type == 'calls':
            return cli.cmd_query_calls(args)
        elif args.query_type == 'callgraph':
            return cli.cmd_query_callgraph(args)
    elif args.command == 'analyze':
        if args.analyze_type == 'impact':
            return cli.cmd_analyze_impact(args)
        elif args.analyze_type == 'deadcode':
            return cli.cmd_analyze_deadcode(args)
        elif args.analyze_type == 'complexity':
            return cli.cmd_analyze_complexity(args)
    elif args.command == 'export':
        if args.export_type == 'rag':
            return cli.cmd_export_rag(args)
        elif args.export_type == 'graphviz':
            return cli.cmd_export_graphviz(args)
    elif args.command in command_map:
        return command_map[args.command](args)
    
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
