"""
Example Analysis Script
Demonstrates how to analyze the keyword-to-document mappings
"""

import json
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns


class MappingAnalyzer:
    """Analyze keyword-to-document mappings"""
    
    def __init__(self, mappings_dir: str):
        """
        Initialize analyzer
        
        Args:
            mappings_dir: Directory containing mapping output files
        """
        self.mappings_dir = Path(mappings_dir)
        
        # Find the most recent files
        json_files = list(self.mappings_dir.glob("keyword_to_documents_*.json"))
        csv_files = list(self.mappings_dir.glob("keyword_document_mappings_*.csv"))
        
        if not json_files or not csv_files:
            raise FileNotFoundError(f"No mapping files found in {mappings_dir}")
        
        # Load most recent
        self.json_file = sorted(json_files)[-1]
        self.csv_file = sorted(csv_files)[-1]
        
        print(f"Loading mappings from: {self.json_file.name}")
        
        with open(self.json_file) as f:
            self.data = json.load(f)
        
        self.df = pd.read_csv(self.csv_file)
        
        print(f"✓ Loaded {len(self.data['mappings'])} keywords")
        print(f"✓ Loaded {len(self.df)} total mappings")
    
    def get_top_keywords_by_coverage(self, n=10):
        """Find keywords that appear in the most documents"""
        keyword_doc_counts = self.df.groupby('Keyword')['Document'].nunique()
        top_keywords = keyword_doc_counts.nlargest(n)
        
        print(f"\n{'='*60}")
        print(f"TOP {n} KEYWORDS BY DOCUMENT COVERAGE")
        print(f"{'='*60}")
        for keyword, count in top_keywords.items():
            print(f"  {keyword:40s} - {count:3d} documents")
        
        return top_keywords
    
    def get_top_documents_by_keyword_coverage(self, n=10):
        """Find documents that match the most keywords"""
        doc_key = self.df['Document'] + '#chunk' + self.df['Chunk_Index'].astype(str)
        doc_keyword_counts = doc_key.value_counts()
        
        print(f"\n{'='*60}")
        print(f"TOP {n} DOCUMENT CHUNKS BY KEYWORD COVERAGE")
        print(f"{'='*60}")
        for doc, count in doc_keyword_counts.head(n).items():
            print(f"  {doc:50s} - {count:3d} keywords")
        
        return doc_keyword_counts.head(n)
    
    def analyze_by_category(self):
        """Analyze mappings by category"""
        print(f"\n{'='*60}")
        print("ANALYSIS BY CATEGORY")
        print(f"{'='*60}")
        
        category_stats = self.df.groupby('Category').agg({
            'Keyword': 'nunique',
            'Document': 'nunique',
            'Relevance_Score_1to5': ['mean', 'median', 'count']
        })
        
        category_stats.columns = ['Unique_Keywords', 'Unique_Docs', 
                                  'Avg_Score', 'Median_Score', 'Total_Mappings']
        category_stats = category_stats.sort_values('Total_Mappings', ascending=False)
        
        print(category_stats.to_string())
        
        return category_stats
    
    def analyze_by_priority(self):
        """Analyze mappings by priority level"""
        print(f"\n{'='*60}")
        print("ANALYSIS BY PRIORITY LEVEL")
        print(f"{'='*60}")
        
        priority_stats = self.df.groupby('Priority').agg({
            'Keyword': 'nunique',
            'Document': 'nunique',
            'Relevance_Score_1to5': ['mean', 'count']
        })
        
        priority_stats.columns = ['Unique_Keywords', 'Unique_Docs', 
                                  'Avg_Score', 'Total_Mappings']
        
        # Order by priority
        priority_order = ['critical', 'high', 'medium', 'low']
        priority_stats = priority_stats.reindex(
            [p for p in priority_order if p in priority_stats.index]
        )
        
        print(priority_stats.to_string())
        
        return priority_stats
    
    def find_keyword_gaps(self, min_docs=3):
        """Find keywords with few matching documents (potential gaps)"""
        keyword_doc_counts = self.df.groupby('Keyword')['Document'].nunique()
        gaps = keyword_doc_counts[keyword_doc_counts < min_docs]
        
        print(f"\n{'='*60}")
        print(f"KEYWORDS WITH < {min_docs} MATCHING DOCUMENTS (Potential Gaps)")
        print(f"{'='*60}")
        
        if len(gaps) == 0:
            print("  No gaps found - all keywords have good coverage!")
        else:
            for keyword, count in gaps.sort_values().items():
                priority = self.df[self.df['Keyword'] == keyword]['Priority'].iloc[0]
                category = self.df[self.df['Keyword'] == keyword]['Category'].iloc[0]
                print(f"  [{priority:8s}] {keyword:40s} - {count} doc(s) ({category})")
        
        return gaps
    
    def analyze_score_distribution(self):
        """Analyze the distribution of relevance scores"""
        print(f"\n{'='*60}")
        print("RELEVANCE SCORE DISTRIBUTION")
        print(f"{'='*60}")
        
        score_counts = self.df['Relevance_Score_1to5'].value_counts().sort_index(ascending=False)
        total = len(self.df)
        
        for score in [5, 4, 3, 2, 1]:
            count = score_counts.get(score, 0)
            percentage = (count / total) * 100
            bar = "█" * int(percentage / 2)
            print(f"  Score {score}: {bar:30s} {count:5d} ({percentage:5.1f}%)")
        
        print(f"\n  Mean Score: {self.df['Relevance_Score_1to5'].mean():.2f}")
        print(f"  Median Score: {self.df['Relevance_Score_1to5'].median():.0f}")
        
        return score_counts
    
    def find_document_by_keywords(self, keywords: list, min_score=3):
        """
        Find documents that match multiple keywords
        Useful for finding comprehensive documents
        """
        print(f"\n{'='*60}")
        print(f"DOCUMENTS MATCHING KEYWORDS: {', '.join(keywords)}")
        print(f"(Minimum score: {min_score})")
        print(f"{'='*60}")
        
        # Filter by keywords and score
        filtered = self.df[
            (self.df['Keyword'].isin(keywords)) & 
            (self.df['Relevance_Score_1to5'] >= min_score)
        ]
        
        # Group by document
        doc_matches = filtered.groupby(['Document', 'Chunk_Index']).agg({
            'Keyword': lambda x: list(x),
            'Relevance_Score_1to5': 'mean'
        }).reset_index()
        
        doc_matches['keyword_count'] = doc_matches['Keyword'].apply(len)
        doc_matches = doc_matches.sort_values('keyword_count', ascending=False)
        
        for _, row in doc_matches.head(10).iterrows():
            print(f"\n  {row['Document']} (chunk {row['Chunk_Index']})")
            print(f"    Matches {row['keyword_count']}/{len(keywords)} keywords")
            print(f"    Avg score: {row['Relevance_Score_1to5']:.1f}")
            print(f"    Keywords: {', '.join(row['Keyword'])}")
        
        return doc_matches
    
    def get_keyword_recommendations(self, keyword: str, n=5):
        """
        Find related keywords based on document co-occurrence
        """
        print(f"\n{'='*60}")
        print(f"RELATED KEYWORDS TO: '{keyword}'")
        print(f"{'='*60}")
        
        # Get documents for this keyword
        keyword_docs = set(
            self.df[self.df['Keyword'] == keyword]['Document'].unique()
        )
        
        if not keyword_docs:
            print(f"  No documents found for '{keyword}'")
            return []
        
        # Find keywords that appear in same documents
        related_keywords = defaultdict(int)
        for doc in keyword_docs:
            doc_keywords = self.df[self.df['Document'] == doc]['Keyword'].unique()
            for kw in doc_keywords:
                if kw != keyword:
                    related_keywords[kw] += 1
        
        # Sort by co-occurrence count
        top_related = sorted(
            related_keywords.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:n]
        
        print(f"\n  Based on {len(keyword_docs)} documents:")
        for related_kw, count in top_related:
            percentage = (count / len(keyword_docs)) * 100
            category = self.df[self.df['Keyword'] == related_kw]['Category'].iloc[0]
            print(f"    {related_kw:40s} - {count:2d}/{len(keyword_docs)} docs "
                  f"({percentage:5.1f}%) [{category}]")
        
        return top_related
    
    def export_high_priority_report(self, output_file="high_priority_report.txt"):
        """Generate a report focused on high-priority keywords"""
        output_path = self.mappings_dir / output_file
        
        critical_high = self.df[self.df['Priority'].isin(['critical', 'high'])]
        
        with open(output_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("HIGH PRIORITY KEYWORD COVERAGE REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Total Critical/High Priority Keywords: "
                   f"{critical_high['Keyword'].nunique()}\n")
            f.write(f"Total Mappings: {len(critical_high)}\n")
            f.write(f"Unique Documents: {critical_high['Document'].nunique()}\n")
            f.write(f"Average Score: {critical_high['Relevance_Score_1to5'].mean():.2f}\n\n")
            
            # By category
            f.write("="*80 + "\n")
            f.write("CRITICAL/HIGH PRIORITY BY CATEGORY\n")
            f.write("="*80 + "\n\n")
            
            for category in critical_high['Category'].unique():
                cat_data = critical_high[critical_high['Category'] == category]
                f.write(f"{category.upper()}\n")
                f.write(f"  Keywords: {cat_data['Keyword'].nunique()}\n")
                f.write(f"  Mappings: {len(cat_data)}\n")
                f.write(f"  Documents: {cat_data['Document'].nunique()}\n")
                f.write(f"  Avg Score: {cat_data['Relevance_Score_1to5'].mean():.2f}\n\n")
            
            # Keywords with limited coverage
            f.write("="*80 + "\n")
            f.write("HIGH PRIORITY KEYWORDS WITH LIMITED COVERAGE (<5 docs)\n")
            f.write("="*80 + "\n\n")
            
            keyword_counts = critical_high.groupby('Keyword')['Document'].nunique()
            limited = keyword_counts[keyword_counts < 5].sort_values()
            
            for keyword, count in limited.items():
                priority = critical_high[critical_high['Keyword'] == keyword]['Priority'].iloc[0]
                f.write(f"  [{priority:8s}] {keyword:40s} - {count} document(s)\n")
        
        print(f"\n✓ High priority report saved to: {output_path}")


def main():
    """Example usage of the analyzer"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze keyword-to-document mappings"
    )
    parser.add_argument(
        "--mappings-dir",
        default="./keyword_mappings",
        help="Directory containing mapping files"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("KEYWORD MAPPING ANALYZER")
    print("="*80 + "\n")
    
    analyzer = MappingAnalyzer(args.mappings_dir)
    
    # Run various analyses
    analyzer.get_top_keywords_by_coverage(n=15)
    analyzer.get_top_documents_by_keyword_coverage(n=15)
    analyzer.analyze_by_category()
    analyzer.analyze_by_priority()
    analyzer.analyze_score_distribution()
    analyzer.find_keyword_gaps(min_docs=3)
    
    # Example: Find documents covering compliance
    print("\n" + "="*60)
    print("EXAMPLE: Finding comprehensive compliance documents")
    print("="*60)
    
    compliance_keywords = [
        'OFAC screening',
        'sanctions screening',
        'AML checks',
        'party validation'
    ]
    analyzer.find_document_by_keywords(compliance_keywords, min_score=3)
    
    # Example: Find related keywords
    analyzer.get_keyword_recommendations('OFAC screening', n=10)
    
    # Export high-priority report
    analyzer.export_high_priority_report()
    
    print("\n" + "="*80)
    print("✓ Analysis complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
