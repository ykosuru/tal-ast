"""
Query Analyzer - Keyword Distribution Analyzer
Takes a user query and shows distribution of matched keywords across documents
"""

import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import pandas as pd
import re


class QueryKeywordAnalyzer:
    """
    Analyzes user queries to find matching keywords and their document distribution
    """
    
    def __init__(
        self,
        keywords_yaml_path: str = "./keywords.yaml",
        mappings_dir: str = "./keyword_mappings"
    ):
        """
        Initialize analyzer
        
        Args:
            keywords_yaml_path: Path to keywords.yaml
            mappings_dir: Directory containing mapping outputs
        """
        self.keywords_yaml_path = Path(keywords_yaml_path)
        self.mappings_dir = Path(mappings_dir)
        
        # Load keywords configuration
        print(f"Loading keywords from: {self.keywords_yaml_path}")
        with open(self.keywords_yaml_path, 'r') as f:
            self.keywords_config = yaml.safe_load(f)
        
        # Build flat keyword list with metadata
        self.all_keywords = []
        for category, data in self.keywords_config.items():
            priority = data.get('priority', 'medium')
            for keyword in data.get('keywords', []):
                self.all_keywords.append({
                    'keyword': keyword.lower(),
                    'original': keyword,
                    'category': category,
                    'priority': priority
                })
        
        print(f"✓ Loaded {len(self.all_keywords)} keywords")
        
        # Load mappings
        print(f"Loading mappings from: {self.mappings_dir}")
        self._load_mappings()
    
    def _load_mappings(self):
        """Load the most recent mapping files"""
        # Load top files JSON
        top_files_pattern = list(self.mappings_dir.glob("top_5_files_per_keyword_*.json"))
        if not top_files_pattern:
            raise FileNotFoundError(f"No mapping files found in {self.mappings_dir}")
        
        top_files_json = sorted(top_files_pattern)[-1]
        print(f"  Loading: {top_files_json.name}")
        
        with open(top_files_json) as f:
            self.top_files_data = json.load(f)
        
        # Load CSV for easier analysis
        csv_pattern = list(self.mappings_dir.glob("top_5_files_per_keyword_*.csv"))
        if csv_pattern:
            csv_file = sorted(csv_pattern)[-1]
            self.mappings_df = pd.read_csv(csv_file)
            print(f"  Loading: {csv_file.name}")
        else:
            self.mappings_df = None
        
        print("✓ Mappings loaded successfully\n")
    
    def match_keywords_in_query(self, query: str) -> List[Dict[str, Any]]:
        """
        Find all keywords from keywords.yaml that appear in the query
        Case-insensitive matching
        
        Args:
            query: User's search query
            
        Returns:
            List of matched keywords with metadata
        """
        query_lower = query.lower()
        matched = []
        
        for kw_data in self.all_keywords:
            keyword = kw_data['keyword']
            
            # Check for exact match or word boundary match (case-insensitive)
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(keyword) + r'\b'
            
            if re.search(pattern, query_lower, re.IGNORECASE):
                matched.append(kw_data.copy())
        
        # Sort by priority and length (longer matches first)
        priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        matched.sort(key=lambda x: (
            priority_order.get(x['priority'], 4),
            -len(x['keyword'])
        ))
        
        return matched
    
    def get_document_distribution(
        self,
        keywords: List[str],
        top_n_docs: int = 10
    ) -> Dict[str, Any]:
        """
        Get document distribution for a list of keywords
        
        Args:
            keywords: List of keywords to analyze
            top_n_docs: Number of top documents to return
            
        Returns:
            Dictionary with distribution data
        """
        if not self.mappings_df is not None:
            print("Warning: CSV data not available, using JSON only")
        
        # Collect document data
        doc_data = defaultdict(lambda: {
            'keywords_found': set(),
            'keyword_details': [],
            'total_chunks': 0,
            'max_relevance': 0,
            'avg_relevance': [],
            'combined_scores': []
        })
        
        for keyword in keywords:
            if keyword not in self.top_files_data['mappings']:
                continue
            
            keyword_info = self.top_files_data['mappings'][keyword]
            
            for file_data in keyword_info['top_files']:
                doc = file_data['document']
                doc_data[doc]['keywords_found'].add(keyword)
                doc_data[doc]['keyword_details'].append({
                    'keyword': keyword,
                    'category': keyword_info['category'],
                    'priority': keyword_info['priority'],
                    'max_relevance': file_data['max_relevance_score'],
                    'chunk_count': file_data['chunk_count'],
                    'combined_score': file_data['combined_file_score']
                })
                doc_data[doc]['total_chunks'] += file_data['chunk_count']
                doc_data[doc]['max_relevance'] = max(
                    doc_data[doc]['max_relevance'],
                    file_data['max_relevance_score']
                )
                doc_data[doc]['avg_relevance'].append(file_data['max_relevance_score'])
                doc_data[doc]['combined_scores'].append(file_data['combined_file_score'])
        
        # Calculate aggregate scores
        doc_results = []
        for doc, data in doc_data.items():
            avg_rel = sum(data['avg_relevance']) / len(data['avg_relevance'])
            avg_combined = sum(data['combined_scores']) / len(data['combined_scores'])
            
            doc_results.append({
                'document': doc,
                'keywords_matched': len(data['keywords_found']),
                'keywords_list': sorted(list(data['keywords_found'])),
                'total_chunks': data['total_chunks'],
                'max_relevance': data['max_relevance'],
                'avg_relevance': avg_rel,
                'avg_combined_score': avg_combined,
                'coverage_ratio': len(data['keywords_found']) / len(keywords),
                'keyword_details': data['keyword_details']
            })
        
        # Sort by coverage ratio, then by avg combined score
        doc_results.sort(
            key=lambda x: (x['coverage_ratio'], x['avg_combined_score']),
            reverse=True
        )
        
        return {
            'total_documents': len(doc_results),
            'documents': doc_results[:top_n_docs],
            'all_documents': doc_results
        }
    
    def analyze_query(
        self,
        query: str,
        top_n_docs: int = 10,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Complete query analysis: match keywords and get document distribution
        
        Args:
            query: User's search query
            top_n_docs: Number of top documents to show
            verbose: Print detailed output
            
        Returns:
            Complete analysis results
        """
        if verbose:
            print("="*80)
            print("QUERY ANALYSIS")
            print("="*80)
            print(f"\nQuery: \"{query}\"\n")
        
        # Step 1: Match keywords
        matched_keywords = self.match_keywords_in_query(query)
        
        if verbose:
            if matched_keywords:
                print(f"✓ Found {len(matched_keywords)} matching keywords:\n")
                for kw in matched_keywords:
                    print(f"  • {kw['original']:40s} [{kw['priority']:8s}] ({kw['category']})")
            else:
                print("⚠ No matching keywords found in query")
                return {
                    'query': query,
                    'matched_keywords': [],
                    'distribution': None
                }
        
        # Step 2: Get document distribution
        keyword_list = [kw['original'] for kw in matched_keywords]
        distribution = self.get_document_distribution(keyword_list, top_n_docs)
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"DOCUMENT DISTRIBUTION")
            print(f"{'='*80}\n")
            print(f"Found {distribution['total_documents']} documents containing these keywords\n")
            
            if distribution['documents']:
                print(f"Top {len(distribution['documents'])} Documents:\n")
                
                for rank, doc in enumerate(distribution['documents'], 1):
                    print(f"[{rank}] {doc['document']}")
                    print(f"    Keywords Matched: {doc['keywords_matched']}/{len(keyword_list)} "
                          f"({doc['coverage_ratio']*100:.0f}% coverage)")
                    print(f"    Total Chunks: {doc['total_chunks']}")
                    print(f"    Max Relevance: {doc['max_relevance']}/5")
                    print(f"    Avg Relevance: {doc['avg_relevance']:.2f}/5")
                    print(f"    Combined Score: {doc['avg_combined_score']:.2f}")
                    print(f"    Keywords: {', '.join(doc['keywords_list'])}")
                    print()
        
        return {
            'query': query,
            'matched_keywords': matched_keywords,
            'distribution': distribution
        }
    
    def compare_documents(
        self,
        query: str,
        min_keywords: int = 2
    ) -> pd.DataFrame:
        """
        Create a comparison matrix showing which documents contain which keywords
        
        Args:
            query: User's search query
            min_keywords: Minimum number of keywords a document must have
            
        Returns:
            DataFrame with document-keyword matrix
        """
        matched_keywords = self.match_keywords_in_query(query)
        keyword_list = [kw['original'] for kw in matched_keywords]
        
        if not keyword_list:
            print("No keywords matched")
            return None
        
        # Build matrix
        matrix_data = []
        
        for keyword in keyword_list:
            if keyword not in self.top_files_data['mappings']:
                continue
            
            keyword_info = self.top_files_data['mappings'][keyword]
            
            for file_data in keyword_info['top_files']:
                matrix_data.append({
                    'Document': file_data['document'],
                    'Keyword': keyword,
                    'Relevance': file_data['max_relevance_score'],
                    'Chunks': file_data['chunk_count']
                })
        
        if not matrix_data:
            print("No document data found")
            return None
        
        df = pd.DataFrame(matrix_data)
        
        # Filter documents with minimum keywords
        doc_keyword_counts = df.groupby('Document')['Keyword'].nunique()
        qualified_docs = doc_keyword_counts[doc_keyword_counts >= min_keywords].index
        
        df_filtered = df[df['Document'].isin(qualified_docs)]
        
        # Create pivot table
        pivot = df_filtered.pivot_table(
            index='Document',
            columns='Keyword',
            values='Relevance',
            aggfunc='max'
        )
        
        # Add summary columns
        pivot['Keywords_Found'] = pivot.notna().sum(axis=1)
        pivot['Avg_Relevance'] = pivot.iloc[:, :-1].mean(axis=1)
        
        # Sort by keywords found, then avg relevance
        pivot = pivot.sort_values(['Keywords_Found', 'Avg_Relevance'], ascending=False)
        
        return pivot
    
    def export_query_analysis(
        self,
        query: str,
        output_file: str = None
    ):
        """
        Export query analysis to a text file
        
        Args:
            query: User's search query
            output_file: Output filename (auto-generated if None)
        """
        analysis = self.analyze_query(query, top_n_docs=20, verbose=False)
        
        if output_file is None:
            # Generate filename from query
            safe_query = re.sub(r'[^\w\s-]', '', query).strip()
            safe_query = re.sub(r'[-\s]+', '_', safe_query)[:50]
            output_file = f"query_analysis_{safe_query}.txt"
        
        output_path = self.mappings_dir / output_file
        
        with open(output_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("QUERY ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Query: \"{analysis['query']}\"\n")
            f.write(f"Generated: {pd.Timestamp.now()}\n\n")
            
            f.write("="*80 + "\n")
            f.write("MATCHED KEYWORDS\n")
            f.write("="*80 + "\n\n")
            
            if analysis['matched_keywords']:
                for kw in analysis['matched_keywords']:
                    f.write(f"• {kw['original']}\n")
                    f.write(f"  Category: {kw['category']}\n")
                    f.write(f"  Priority: {kw['priority']}\n\n")
            else:
                f.write("No keywords matched\n")
            
            if analysis['distribution']:
                f.write("\n" + "="*80 + "\n")
                f.write("DOCUMENT DISTRIBUTION\n")
                f.write("="*80 + "\n\n")
                
                f.write(f"Total Documents: {analysis['distribution']['total_documents']}\n\n")
                
                for rank, doc in enumerate(analysis['distribution']['documents'], 1):
                    f.write(f"[{rank}] {doc['document']}\n")
                    f.write(f"    Keywords Matched: {doc['keywords_matched']}")
                    f.write(f"/{len(analysis['matched_keywords'])} ")
                    f.write(f"({doc['coverage_ratio']*100:.0f}%)\n")
                    f.write(f"    Total Chunks: {doc['total_chunks']}\n")
                    f.write(f"    Max Relevance: {doc['max_relevance']}/5\n")
                    f.write(f"    Avg Relevance: {doc['avg_relevance']:.2f}/5\n")
                    f.write(f"    Keywords: {', '.join(doc['keywords_list'])}\n\n")
                    
                    f.write(f"    Keyword Details:\n")
                    for kw_detail in doc['keyword_details']:
                        f.write(f"      - {kw_detail['keyword']:30s} ")
                        f.write(f"Relevance: {kw_detail['max_relevance']}/5, ")
                        f.write(f"Chunks: {kw_detail['chunk_count']}\n")
                    f.write("\n")
        
        print(f"✓ Analysis exported to: {output_path}")
        return output_path
    
    def visualize_distribution(
        self,
        query: str,
        show_matrix: bool = True
    ):
        """
        Create visualizations of keyword distribution
        
        Args:
            query: User's search query
            show_matrix: Whether to show document-keyword matrix
        """
        analysis = self.analyze_query(query, verbose=False)
        
        if not analysis['matched_keywords']:
            print("No keywords to visualize")
            return
        
        print("\n" + "="*80)
        print("KEYWORD COVERAGE SUMMARY")
        print("="*80 + "\n")
        
        # Summary table
        keyword_list = [kw['original'] for kw in analysis['matched_keywords']]
        
        keyword_doc_counts = []
        for kw in keyword_list:
            if kw in self.top_files_data['mappings']:
                top_files = self.top_files_data['mappings'][kw]['top_files']
                keyword_doc_counts.append({
                    'Keyword': kw,
                    'Documents': len(top_files),
                    'Category': self.top_files_data['mappings'][kw]['category'],
                    'Priority': self.top_files_data['mappings'][kw]['priority']
                })
        
        if keyword_doc_counts:
            summary_df = pd.DataFrame(keyword_doc_counts)
            print(summary_df.to_string(index=False))
        
        # Document-keyword matrix
        if show_matrix and len(keyword_list) > 1:
            print("\n" + "="*80)
            print("DOCUMENT-KEYWORD MATRIX")
            print("="*80 + "\n")
            
            matrix = self.compare_documents(query, min_keywords=1)
            if matrix is not None and not matrix.empty:
                print("Legend: Numbers show relevance scores (1-5), '-' means not found\n")
                print(matrix.fillna('-').to_string())
            else:
                print("No matrix data available")


def main():
    """Main execution with example queries"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze query to find keyword distribution across documents"
    )
    parser.add_argument(
        "--query",
        required=True,
        help="User query to analyze"
    )
    parser.add_argument(
        "--keywords-yaml",
        default="./keywords.yaml",
        help="Path to keywords.yaml"
    )
    parser.add_argument(
        "--mappings-dir",
        default="./keyword_mappings",
        help="Directory containing mapping files"
    )
    parser.add_argument(
        "--top-docs",
        type=int,
        default=10,
        help="Number of top documents to show (default: 10)"
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export analysis to file"
    )
    parser.add_argument(
        "--show-matrix",
        action="store_true",
        help="Show document-keyword matrix"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("QUERY KEYWORD ANALYZER")
    print("="*80 + "\n")
    
    # Initialize analyzer
    analyzer = QueryKeywordAnalyzer(
        keywords_yaml_path=args.keywords_yaml,
        mappings_dir=args.mappings_dir
    )
    
    # Analyze query
    analysis = analyzer.analyze_query(
        query=args.query,
        top_n_docs=args.top_docs,
        verbose=True
    )
    
    # Show visualizations
    if analysis['matched_keywords']:
        analyzer.visualize_distribution(
            query=args.query,
            show_matrix=args.show_matrix
        )
    
    # Export if requested
    if args.export:
        analyzer.export_query_analysis(args.query)
    
    print("\n" + "="*80)
    print("✓ ANALYSIS COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
