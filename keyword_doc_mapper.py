"""
Keyword-to-Document Mapper
Maps each keyword from keywords.yaml to relevant documents with 1-5 relevance scoring
5 = Strongest match, 1 = Weakest match
"""

import yaml
import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import numpy as np
from datetime import datetime

# Import the searcher from the indexer
import sys
import os

# Assuming the indexer script is in the same directory or accessible
# If not, you may need to adjust the import or copy the WireProcessingSearcher class


class KeywordDocumentMapper:
    """
    Maps keywords to documents with 1-5 relevance scoring
    Uses the existing wire processing search infrastructure
    """
    
    def __init__(
        self,
        index_path: str,
        keywords_yaml_path: str,
        output_dir: str = "./keyword_mappings"
    ):
        """
        Initialize mapper
        
        Args:
            index_path: Path to the existing search index
            keywords_yaml_path: Path to keywords.yaml file
            output_dir: Directory to save mapping outputs
        """
        self.index_path = Path(index_path)
        self.keywords_yaml_path = Path(keywords_yaml_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load keywords from YAML
        print(f"Loading keywords from: {self.keywords_yaml_path}")
        self.keywords_config = self._load_keywords()
        
        # Initialize searcher
        print(f"Loading search index from: {self.index_path}")
        # Import here to avoid circular dependency
        try:
            from wire_processing_indexer import WireProcessingSearcher
            self.searcher = WireProcessingSearcher(
                index_path=str(self.index_path),
                enable_query_expansion=True,
                expansion_level="medium",
                enable_query_preprocessing=True
            )
            print("✓ Search index loaded successfully")
        except ImportError:
            print("ERROR: Could not import WireProcessingSearcher")
            print("Make sure the indexer script is in the same directory or in PYTHONPATH")
            raise
        
        # Storage for mappings
        self.keyword_mappings = defaultdict(list)
        self.document_keywords = defaultdict(list)
        self.category_stats = {}
        
    def _load_keywords(self) -> Dict[str, Any]:
        """Load keywords from YAML file"""
        with open(self.keywords_yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Count keywords by category
        total_keywords = 0
        for category, data in config.items():
            keyword_count = len(data.get('keywords', []))
            total_keywords += keyword_count
            print(f"  {category}: {keyword_count} keywords (priority: {data.get('priority', 'N/A')})")
        
        print(f"✓ Loaded {total_keywords} keywords across {len(config)} categories")
        return config
    
    def _normalize_score_to_scale(
        self,
        score: float,
        scores: List[float],
        min_threshold: float = 0.05
    ) -> int:
        """
        Normalize a score to 1-5 scale
        
        Args:
            score: The score to normalize
            scores: All scores for context
            min_threshold: Minimum score to be considered (below this = 0, not included)
            
        Returns:
            Integer score from 1-5, or 0 if below threshold
        """
        if score < min_threshold:
            return 0
        
        # Use percentile-based bucketing for better distribution
        if not scores or len(scores) == 1:
            return 5 if score > 0.5 else 3
        
        # Calculate percentiles
        scores_sorted = sorted(scores, reverse=True)
        
        # Find position in sorted list
        try:
            percentile = (scores_sorted.index(score) + 1) / len(scores_sorted)
        except ValueError:
            # If exact score not found, estimate
            higher = sum(1 for s in scores_sorted if s > score)
            percentile = higher / len(scores_sorted)
        
        # Map percentiles to 1-5 scale
        # Top 20% = 5, Next 20% = 4, Next 20% = 3, Next 20% = 2, Bottom 20% = 1
        if percentile <= 0.20:
            return 5
        elif percentile <= 0.40:
            return 4
        elif percentile <= 0.60:
            return 3
        elif percentile <= 0.80:
            return 2
        else:
            return 1
    
    def map_keyword(
        self,
        keyword: str,
        category: str,
        priority: str,
        max_docs: int = 20,
        min_score_threshold: float = 0.05
    ) -> List[Dict[str, Any]]:
        """
        Map a single keyword to relevant documents
        
        Args:
            keyword: The keyword to search for
            category: Category this keyword belongs to
            priority: Priority level (critical, high, medium, low)
            max_docs: Maximum number of documents to return
            min_score_threshold: Minimum relevance score to include
            
        Returns:
            List of document mappings with scores
        """
        # Search for keyword
        results = self.searcher.search(
            query=keyword,
            top_k=max_docs,
            verbose=False
        )
        
        if not results:
            return []
        
        # Extract all scores for normalization
        all_scores = [r.get('combined_score', 0) for r in results]
        
        # Map each result
        mappings = []
        for result in results:
            raw_score = result.get('combined_score', 0)
            
            # Normalize to 1-5
            relevance_score = self._normalize_score_to_scale(
                raw_score,
                all_scores,
                min_score_threshold
            )
            
            if relevance_score == 0:
                continue
            
            mapping = {
                'keyword': keyword,
                'category': category,
                'priority': priority,
                'document': result['source_file'],
                'chunk_index': result['chunk_index'],
                'relevance_score': relevance_score,
                'raw_score': raw_score,
                'normalized_score': result.get('normalized_score', 0),
                'bm25_score': result.get('bm25_score', 0),
                'primary_capability': result.get('primary_capability', 'unknown'),
                'capabilities': result.get('capabilities', [])[:3],
                'keywords_found': result.get('keywords', [])[:5],
                'text_preview': result['text'][:200] + "..." if len(result['text']) > 200 else result['text']
            }
            
            mappings.append(mapping)
        
        return mappings
    
    def generate_all_mappings(
        self,
        max_docs_per_keyword: int = 20,
        min_score_threshold: float = 0.05
    ):
        """
        Generate mappings for all keywords in the configuration
        
        Args:
            max_docs_per_keyword: Max documents to map per keyword
            min_score_threshold: Minimum relevance threshold
        """
        print("\n" + "="*80)
        print("GENERATING KEYWORD-TO-DOCUMENT MAPPINGS")
        print("="*80)
        
        total_keywords = sum(
            len(data.get('keywords', []))
            for data in self.keywords_config.values()
        )
        
        current_keyword = 0
        
        for category, data in self.keywords_config.items():
            priority = data.get('priority', 'medium')
            keywords = data.get('keywords', [])
            
            print(f"\n[{category}] Priority: {priority.upper()} - {len(keywords)} keywords")
            print("-" * 80)
            
            category_mappings = []
            
            for keyword in keywords:
                current_keyword += 1
                print(f"  [{current_keyword}/{total_keywords}] Mapping: '{keyword}'", end=" ")
                
                mappings = self.map_keyword(
                    keyword=keyword,
                    category=category,
                    priority=priority,
                    max_docs=max_docs_per_keyword,
                    min_score_threshold=min_score_threshold
                )
                
                if mappings:
                    print(f"✓ Found {len(mappings)} relevant docs")
                    
                    # Store in keyword_mappings
                    self.keyword_mappings[keyword] = mappings
                    
                    # Build reverse index: document -> keywords
                    for mapping in mappings:
                        doc_key = f"{mapping['document']}#chunk{mapping['chunk_index']}"
                        self.document_keywords[doc_key].append({
                            'keyword': keyword,
                            'category': category,
                            'priority': priority,
                            'relevance_score': mapping['relevance_score']
                        })
                    
                    category_mappings.extend(mappings)
                else:
                    print("⚠ No relevant docs found")
            
            # Category statistics
            if category_mappings:
                avg_score = np.mean([m['relevance_score'] for m in category_mappings])
                unique_docs = len(set([m['document'] for m in category_mappings]))
                
                self.category_stats[category] = {
                    'total_mappings': len(category_mappings),
                    'unique_documents': unique_docs,
                    'avg_relevance_score': float(avg_score),
                    'keywords_count': len(keywords),
                    'priority': priority
                }
                
                print(f"\n  Category Summary:")
                print(f"    Total mappings: {len(category_mappings)}")
                print(f"    Unique documents: {unique_docs}")
                print(f"    Avg relevance: {avg_score:.2f}/5")
        
        print("\n" + "="*80)
        print(f"✓ Mapping complete! Processed {total_keywords} keywords")
        print("="*80)
    
    def save_mappings(self):
        """Save all mappings to various output formats"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("\n" + "="*80)
        print("SAVING MAPPINGS")
        print("="*80)
        
        # 1. Save detailed JSON (keyword -> documents)
        json_path = self.output_dir / f"keyword_to_documents_{timestamp}.json"
        json_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_keywords': len(self.keyword_mappings),
                'total_documents': len(self.document_keywords),
                'index_path': str(self.index_path),
                'keywords_yaml': str(self.keywords_yaml_path)
            },
            'category_statistics': self.category_stats,
            'mappings': {
                keyword: mappings
                for keyword, mappings in self.keyword_mappings.items()
            }
        }
        
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"✓ Saved detailed JSON: {json_path}")
        
        # 2. Save reverse index JSON (document -> keywords)
        reverse_json_path = self.output_dir / f"document_to_keywords_{timestamp}.json"
        reverse_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_documents': len(self.document_keywords),
                'total_keywords': len(self.keyword_mappings)
            },
            'mappings': dict(self.document_keywords)
        }
        
        with open(reverse_json_path, 'w') as f:
            json.dump(reverse_data, f, indent=2)
        print(f"✓ Saved reverse index JSON: {reverse_json_path}")
        
        # 3. Save CSV (flat format for easy analysis)
        csv_path = self.output_dir / f"keyword_document_mappings_{timestamp}.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Keyword',
                'Category',
                'Priority',
                'Document',
                'Chunk_Index',
                'Relevance_Score_1to5',
                'Raw_Score',
                'Primary_Capability',
                'Top_Capabilities',
                'Text_Preview'
            ])
            
            for keyword, mappings in sorted(self.keyword_mappings.items()):
                for mapping in mappings:
                    writer.writerow([
                        mapping['keyword'],
                        mapping['category'],
                        mapping['priority'],
                        mapping['document'],
                        mapping['chunk_index'],
                        mapping['relevance_score'],
                        f"{mapping['raw_score']:.4f}",
                        mapping['primary_capability'],
                        '; '.join(mapping['capabilities']),
                        mapping['text_preview']
                    ])
        
        print(f"✓ Saved CSV: {csv_path}")
        
        # 4. Save summary statistics
        summary_path = self.output_dir / f"mapping_summary_{timestamp}.txt"
        with open(summary_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("KEYWORD-TO-DOCUMENT MAPPING SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Keywords: {len(self.keyword_mappings)}\n")
            f.write(f"Total Unique Documents: {len(self.document_keywords)}\n\n")
            
            f.write("="*80 + "\n")
            f.write("CATEGORY STATISTICS\n")
            f.write("="*80 + "\n\n")
            
            for category, stats in sorted(self.category_stats.items()):
                f.write(f"{category.upper()}\n")
                f.write(f"  Priority: {stats['priority']}\n")
                f.write(f"  Keywords: {stats['keywords_count']}\n")
                f.write(f"  Total Mappings: {stats['total_mappings']}\n")
                f.write(f"  Unique Documents: {stats['unique_documents']}\n")
                f.write(f"  Avg Relevance Score: {stats['avg_relevance_score']:.2f}/5\n")
                f.write("\n")
            
            f.write("="*80 + "\n")
            f.write("TOP KEYWORDS BY DOCUMENT COUNT\n")
            f.write("="*80 + "\n\n")
            
            keyword_doc_counts = [
                (kw, len(mappings))
                for kw, mappings in self.keyword_mappings.items()
            ]
            keyword_doc_counts.sort(key=lambda x: x[1], reverse=True)
            
            for kw, count in keyword_doc_counts[:20]:
                f.write(f"  {kw:40s} - {count:3d} documents\n")
            
            f.write("\n")
            f.write("="*80 + "\n")
            f.write("TOP DOCUMENTS BY KEYWORD COUNT\n")
            f.write("="*80 + "\n\n")
            
            doc_keyword_counts = [
                (doc, len(keywords))
                for doc, keywords in self.document_keywords.items()
            ]
            doc_keyword_counts.sort(key=lambda x: x[1], reverse=True)
            
            for doc, count in doc_keyword_counts[:20]:
                f.write(f"  {doc:60s} - {count:3d} keywords\n")
        
        print(f"✓ Saved summary: {summary_path}")
        
        # 5. Save high-relevance only (score >= 4)
        high_relevance_path = self.output_dir / f"high_relevance_mappings_{timestamp}.json"
        high_relevance_data = {}
        
        for keyword, mappings in self.keyword_mappings.items():
            high_rel = [m for m in mappings if m['relevance_score'] >= 4]
            if high_rel:
                high_relevance_data[keyword] = high_rel
        
        with open(high_relevance_path, 'w') as f:
            json.dump({
                'metadata': {
                    'description': 'Only mappings with relevance score >= 4 (out of 5)',
                    'generated_at': datetime.now().isoformat(),
                    'total_high_relevance_keywords': len(high_relevance_data)
                },
                'mappings': high_relevance_data
            }, f, indent=2)
        
        print(f"✓ Saved high-relevance only: {high_relevance_path}")
        
        print("\n" + "="*80)
        print("✓ All mappings saved successfully!")
        print("="*80)
    
    def print_summary(self):
        """Print a summary of the mappings"""
        print("\n" + "="*80)
        print("MAPPING SUMMARY")
        print("="*80)
        
        print(f"\nTotal Keywords Mapped: {len(self.keyword_mappings)}")
        print(f"Total Unique Documents: {len(self.document_keywords)}")
        
        # Calculate overall statistics
        all_scores = []
        score_distribution = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        
        for mappings in self.keyword_mappings.values():
            for mapping in mappings:
                score = mapping['relevance_score']
                all_scores.append(score)
                score_distribution[score] += 1
        
        if all_scores:
            print(f"Total Mappings: {len(all_scores)}")
            print(f"Average Relevance Score: {np.mean(all_scores):.2f}/5")
            
            print("\nRelevance Score Distribution:")
            for score in [5, 4, 3, 2, 1]:
                count = score_distribution[score]
                percentage = (count / len(all_scores)) * 100
                bar = "█" * int(percentage / 2)
                print(f"  {score}: {bar} {count:5d} ({percentage:5.1f}%)")
        
        print("\nCategory Statistics:")
        for category, stats in sorted(self.category_stats.items()):
            print(f"  {category:30s} - {stats['total_mappings']:4d} mappings, "
                  f"{stats['unique_documents']:3d} docs, "
                  f"avg score: {stats['avg_relevance_score']:.2f}")


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Map keywords to documents with 1-5 relevance scoring"
    )
    parser.add_argument(
        "--index-path",
        required=True,
        help="Path to the wire processing search index"
    )
    parser.add_argument(
        "--keywords-yaml",
        required=True,
        help="Path to keywords.yaml file"
    )
    parser.add_argument(
        "--output-dir",
        default="./keyword_mappings",
        help="Output directory for mappings (default: ./keyword_mappings)"
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=20,
        help="Maximum documents per keyword (default: 20)"
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.05,
        help="Minimum relevance score threshold (default: 0.05)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("KEYWORD-TO-DOCUMENT MAPPER")
    print("Maps keywords to documents with 1-5 relevance scoring")
    print("="*80 + "\n")
    
    # Initialize mapper
    mapper = KeywordDocumentMapper(
        index_path=args.index_path,
        keywords_yaml_path=args.keywords_yaml,
        output_dir=args.output_dir
    )
    
    # Generate mappings
    mapper.generate_all_mappings(
        max_docs_per_keyword=args.max_docs,
        min_score_threshold=args.min_score
    )
    
    # Print summary
    mapper.print_summary()
    
    # Save all outputs
    mapper.save_mappings()
    
    print("\n" + "="*80)
    print("✓ MAPPING COMPLETE!")
    print(f"✓ Check output directory: {args.output_dir}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
