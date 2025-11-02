"""
Example Analysis Script
Demonstrates how to analyze the keyword-to-document mappings
Enhanced with full keyword distribution from keywords.yaml
"""

import json
import pandas as pd
import yaml
from pathlib import Path
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns


class MappingAnalyzer:
    """Analyze keyword-to-document mappings"""
    
    def __init__(self, mappings_dir: str, keywords_yaml: str = "./keywords.yaml"):
        """
        Initialize analyzer
        
        Args:
            mappings_dir: Directory containing mapping output files
            keywords_yaml: Path to keywords.yaml file
        """
        self.mappings_dir = Path(mappings_dir)
        self.keywords_yaml = Path(keywords_yaml)
        
        # Load keywords configuration
        if self.keywords_yaml.exists():
            with open(self.keywords_yaml, 'r') as f:
                self.keywords_config = yaml.safe_load(f)
            print(f"✓ Loaded keywords from {self.keywords_yaml}")
        else:
            self.keywords_config = None
            print(f"⚠ Keywords file not found: {self.keywords_yaml}")
        
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
        
        # Load keywords.yaml if available
        self.keywords_config = None
        keywords_yaml = self.mappings_dir.parent / "keywords.yaml"
        if keywords_yaml.exists():
            with open(keywords_yaml) as f:
                self.keywords_config = yaml.safe_load(f)
            print(f"✓ Loaded keywords configuration from {keywords_yaml.name}")
        else:
            print(f"⚠ keywords.yaml not found at {keywords_yaml}")
            print(f"  Some distribution features will be limited")
    
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
    
    def analyze_keyword_distribution(self):
        """
        Comprehensive analysis of ALL keywords from keywords.yaml
        Shows distribution across documents, coverage, and gaps
        """
        if not self.keywords_config:
            print("\n❌ keywords.yaml not loaded - cannot perform full distribution analysis")
            print("   Place keywords.yaml in the parent directory of mappings folder")
            return
        
        print("\n" + "="*80)
        print("COMPREHENSIVE KEYWORD DISTRIBUTION ANALYSIS")
        print("="*80)
        
        # Collect all keywords from yaml with their metadata
        all_keywords = []
        for category, data in self.keywords_config.items():
            priority = data.get('priority', 'medium')
            for keyword in data.get('keywords', []):
                all_keywords.append({
                    'keyword': keyword,
                    'category': category,
                    'priority': priority
                })
        
        print(f"\nTotal keywords in keywords.yaml: {len(all_keywords)}")
        
        # Check which keywords have mappings
        mapped_keywords = set(self.data['mappings'].keys())
        
        # Build distribution data
        distribution_data = []
        
        for kw_info in all_keywords:
            keyword = kw_info['keyword']
            
            if keyword in mapped_keywords:
                # Get stats from mappings
                kw_data = self.df[self.df['Keyword'] == keyword]
                
                doc_count = kw_data['Document'].nunique()
                chunk_count = len(kw_data)
                avg_relevance = kw_data['Relevance_Score_1to5'].mean()
                max_relevance = kw_data['Relevance_Score_1to5'].max()
                
                distribution_data.append({
                    'Keyword': keyword,
                    'Category': kw_info['category'],
                    'Priority': kw_info['priority'],
                    'Documents': doc_count,
                    'Chunks': chunk_count,
                    'Avg_Relevance': avg_relevance,
                    'Max_Relevance': max_relevance,
                    'Status': 'Mapped'
                })
            else:
                # Keyword has no mappings (gap)
                distribution_data.append({
                    'Keyword': keyword,
                    'Category': kw_info['category'],
                    'Priority': kw_info['priority'],
                    'Documents': 0,
                    'Chunks': 0,
                    'Avg_Relevance': 0,
                    'Max_Relevance': 0,
                    'Status': 'No Mapping'
                })
        
        dist_df = pd.DataFrame(distribution_data)
        
        # Overall Statistics
        print("\n" + "="*80)
        print("OVERALL STATISTICS")
        print("="*80)
        
        mapped_count = len(dist_df[dist_df['Status'] == 'Mapped'])
        unmapped_count = len(dist_df[dist_df['Status'] == 'No Mapping'])
        
        print(f"\nKeywords with mappings: {mapped_count} ({mapped_count/len(dist_df)*100:.1f}%)")
        print(f"Keywords without mappings: {unmapped_count} ({unmapped_count/len(dist_df)*100:.1f}%)")
        
        if mapped_count > 0:
            print(f"\nAverage documents per keyword: {dist_df[dist_df['Documents'] > 0]['Documents'].mean():.1f}")
            print(f"Average chunks per keyword: {dist_df[dist_df['Chunks'] > 0]['Chunks'].mean():.1f}")
            print(f"Average relevance: {dist_df[dist_df['Status'] == 'Mapped']['Avg_Relevance'].mean():.2f}/5")
        
        # By Category
        print("\n" + "="*80)
        print("DISTRIBUTION BY CATEGORY")
        print("="*80 + "\n")
        
        category_summary = dist_df.groupby('Category').agg({
            'Keyword': 'count',
            'Documents': ['sum', 'mean'],
            'Chunks': 'sum',
            'Avg_Relevance': 'mean',
            'Status': lambda x: (x == 'Mapped').sum()
        }).round(2)
        
        category_summary.columns = ['Total_Keywords', 'Total_Docs', 'Avg_Docs_Per_KW', 
                                    'Total_Chunks', 'Avg_Relevance', 'Mapped_Count']
        
        # Add coverage percentage
        category_summary['Coverage_%'] = (
            category_summary['Mapped_Count'] / category_summary['Total_Keywords'] * 100
        ).round(1)
        
        # Sort by priority then coverage
        priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        
        # Add priority for sorting
        cat_priority = {}
        for category, data in self.keywords_config.items():
            cat_priority[category] = data.get('priority', 'medium')
        
        category_summary['Priority_Order'] = category_summary.index.map(
            lambda x: priority_order.get(cat_priority.get(x, 'medium'), 4)
        )
        
        category_summary = category_summary.sort_values(['Priority_Order', 'Coverage_%'], 
                                                        ascending=[True, False])
        category_summary = category_summary.drop('Priority_Order', axis=1)
        
        print(category_summary.to_string())
        
        # Top Covered Keywords
        print("\n" + "="*80)
        print("TOP 20 KEYWORDS BY DOCUMENT COVERAGE")
        print("="*80 + "\n")
        
        top_keywords = dist_df[dist_df['Status'] == 'Mapped'].nlargest(20, 'Documents')
        
        for idx, row in top_keywords.iterrows():
            print(f"{row['Keyword']:40s} - {int(row['Documents']):3d} docs, "
                  f"{int(row['Chunks']):4d} chunks, "
                  f"{row['Avg_Relevance']:.1f}/5 avg relevance "
                  f"[{row['Priority']:8s}]")
        
        # Gaps Analysis
        print("\n" + "="*80)
        print("KEYWORDS WITH NO MAPPINGS (GAPS)")
        print("="*80 + "\n")
        
        gaps = dist_df[dist_df['Status'] == 'No Mapping']
        
        if len(gaps) > 0:
            print(f"Found {len(gaps)} keywords with no document mappings:\n")
            
            # Group by priority
            for priority in ['critical', 'high', 'medium', 'low']:
                priority_gaps = gaps[gaps['Priority'] == priority]
                
                if len(priority_gaps) > 0:
                    print(f"\n{priority.upper()} Priority ({len(priority_gaps)} keywords):")
                    print("-" * 80)
                    
                    for idx, row in priority_gaps.iterrows():
                        print(f"  • {row['Keyword']:40s} ({row['Category']})")
        else:
            print("✓ No gaps! All keywords have document mappings.")
        
        # Limited Coverage (1-2 docs only)
        print("\n" + "="*80)
        print("KEYWORDS WITH LIMITED COVERAGE (1-2 DOCUMENTS)")
        print("="*80 + "\n")
        
        limited = dist_df[(dist_df['Documents'] > 0) & (dist_df['Documents'] <= 2)]
        
        if len(limited) > 0:
            print(f"Found {len(limited)} keywords with limited coverage:\n")
            
            # Group by priority
            for priority in ['critical', 'high', 'medium', 'low']:
                priority_limited = limited[limited['Priority'] == priority]
                
                if len(priority_limited) > 0:
                    print(f"\n{priority.upper()} Priority ({len(priority_limited)} keywords):")
                    print("-" * 80)
                    
                    for idx, row in priority_limited.iterrows():
                        print(f"  • {row['Keyword']:40s} - {int(row['Documents'])} doc(s), "
                              f"{int(row['Chunks'])} chunks ({row['Category']})")
        else:
            print("✓ All mapped keywords have good coverage (3+ documents)")
        
        # Priority-based summary
        print("\n" + "="*80)
        print("COVERAGE BY PRIORITY LEVEL")
        print("="*80 + "\n")
        
        priority_summary = dist_df.groupby('Priority').agg({
            'Keyword': 'count',
            'Documents': 'sum',
            'Status': lambda x: (x == 'Mapped').sum()
        })
        
        priority_summary.columns = ['Total_Keywords', 'Total_Documents', 'Mapped_Count']
        priority_summary['Coverage_%'] = (
            priority_summary['Mapped_Count'] / priority_summary['Total_Keywords'] * 100
        ).round(1)
        
        # Sort by priority order
        priority_summary['Order'] = priority_summary.index.map(
            lambda x: priority_order.get(x, 4)
        )
        priority_summary = priority_summary.sort_values('Order').drop('Order', axis=1)
        
        print(priority_summary.to_string())
        
        # Return the dataframe for further analysis
        return dist_df
    
    def analyze_keyword_distribution(self):
        """
        Comprehensive analysis of ALL keywords from keywords.yaml
        Shows distribution across documents
        """
        if not self.keywords_config:
            print("\n⚠ Keywords configuration not available")
            return
        
        print("\n" + "="*80)
        print("COMPREHENSIVE KEYWORD DISTRIBUTION ANALYSIS")
        print("="*80)
        
        # Collect all keywords from yaml
        all_keywords = []
        for category, data in self.keywords_config.items():
            priority = data.get('priority', 'medium')
            for keyword in data.get('keywords', []):
                all_keywords.append({
                    'keyword': keyword,
                    'category': category,
                    'priority': priority
                })
        
        print(f"\nTotal keywords in keywords.yaml: {len(all_keywords)}")
        print(f"Total keywords with mappings: {len(self.data['mappings'])}\n")
        
        # Analyze each keyword
        keyword_stats = []
        
        for kw_data in all_keywords:
            keyword = kw_data['keyword']
            
            if keyword in self.data['mappings']:
                mappings = self.data['mappings'][keyword]
                
                # Get unique documents
                unique_docs = set([m['document'] for m in mappings])
                
                # Calculate statistics
                total_chunks = sum([m['chunk_count'] for m in mappings])
                avg_relevance = sum([m['relevance_score'] for m in mappings]) / len(mappings)
                max_relevance = max([m['relevance_score'] for m in mappings])
                
                keyword_stats.append({
                    'Keyword': keyword,
                    'Category': kw_data['category'],
                    'Priority': kw_data['priority'],
                    'Documents': len(unique_docs),
                    'Total_Chunks': total_chunks,
                    'Avg_Relevance': avg_relevance,
                    'Max_Relevance': max_relevance,
                    'Status': 'Mapped'
                })
            else:
                # Keyword not found in mappings
                keyword_stats.append({
                    'Keyword': keyword,
                    'Category': kw_data['category'],
                    'Priority': kw_data['priority'],
                    'Documents': 0,
                    'Total_Chunks': 0,
                    'Avg_Relevance': 0,
                    'Max_Relevance': 0,
                    'Status': 'Not Found'
                })
        
        # Create DataFrame
        dist_df = pd.DataFrame(keyword_stats)
        
        # Analysis by status
        print("="*80)
        print("MAPPING COVERAGE")
        print("="*80 + "\n")
        
        mapped_count = len(dist_df[dist_df['Status'] == 'Mapped'])
        not_found_count = len(dist_df[dist_df['Status'] == 'Not Found'])
        
        print(f"✓ Mapped keywords: {mapped_count} ({mapped_count/len(dist_df)*100:.1f}%)")
        print(f"✗ Not found: {not_found_count} ({not_found_count/len(dist_df)*100:.1f}%)")
        
        # Show unmapped keywords
        if not_found_count > 0:
            print(f"\n{'='*80}")
            print(f"UNMAPPED KEYWORDS ({not_found_count} total)")
            print(f"{'='*80}\n")
            
            unmapped = dist_df[dist_df['Status'] == 'Not Found']
            
            # Group by priority
            priority_order = ['critical', 'high', 'medium', 'low']
            for priority in priority_order:
                priority_unmapped = unmapped[unmapped['Priority'] == priority]
                if not priority_unmapped.empty:
                    print(f"\n{priority.upper()} Priority ({len(priority_unmapped)} keywords):")
                    for _, row in priority_unmapped.iterrows():
                        print(f"  ✗ {row['Keyword']:40s} ({row['Category']})")
        
        # Distribution by category
        print(f"\n{'='*80}")
        print("DISTRIBUTION BY CATEGORY")
        print(f"{'='*80}\n")
        
        category_dist = dist_df.groupby('Category').agg({
            'Documents': ['mean', 'sum', 'count'],
            'Total_Chunks': 'sum',
            'Avg_Relevance': 'mean'
        }).round(2)
        
        category_dist.columns = ['Avg_Docs_Per_Keyword', 'Total_Docs', 'Keyword_Count', 
                                  'Total_Chunks', 'Avg_Relevance']
        category_dist = category_dist.sort_values('Total_Docs', ascending=False)
        
        print(category_dist.to_string())
        
        # Distribution by priority
        print(f"\n{'='*80}")
        print("DISTRIBUTION BY PRIORITY")
        print(f"{'='*80}\n")
        
        priority_dist = dist_df.groupby('Priority').agg({
            'Documents': ['mean', 'sum', 'count'],
            'Total_Chunks': 'sum',
            'Avg_Relevance': 'mean'
        }).round(2)
        
        priority_dist.columns = ['Avg_Docs_Per_Keyword', 'Total_Docs', 'Keyword_Count',
                                 'Total_Chunks', 'Avg_Relevance']
        
        # Sort by priority order
        priority_order_dict = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        priority_dist['Order'] = priority_dist.index.map(lambda x: priority_order_dict.get(x, 4))
        priority_dist = priority_dist.sort_values('Order').drop('Order', axis=1)
        
        print(priority_dist.to_string())
        
        # Top keywords by document coverage
        print(f"\n{'='*80}")
        print("TOP 20 KEYWORDS BY DOCUMENT COVERAGE")
        print(f"{'='*80}\n")
        
        mapped_df = dist_df[dist_df['Status'] == 'Mapped'].copy()
        top_keywords = mapped_df.nlargest(20, 'Documents')
        
        for idx, row in top_keywords.iterrows():
            print(f"  {row['Keyword']:40s} - {row['Documents']:3d} docs, "
                  f"{row['Total_Chunks']:4d} chunks [{row['Priority']:8s}]")
        
        # Keywords with limited coverage
        print(f"\n{'='*80}")
        print("KEYWORDS WITH LIMITED COVERAGE (< 3 documents)")
        print(f"{'='*80}\n")
        
        limited = mapped_df[mapped_df['Documents'] < 3].copy()
        limited = limited.sort_values(['Priority', 'Documents'])
        
        if not limited.empty:
            print(f"Found {len(limited)} keywords with limited coverage:\n")
            
            for priority in ['critical', 'high', 'medium', 'low']:
                priority_limited = limited[limited['Priority'] == priority]
                if not priority_limited.empty:
                    print(f"\n{priority.upper()} Priority ({len(priority_limited)} keywords):")
                    for _, row in priority_limited.iterrows():
                        print(f"  ⚠ {row['Keyword']:40s} - {row['Documents']} doc(s), "
                              f"{row['Total_Chunks']} chunks ({row['Category']})")
        else:
            print("✓ All mapped keywords have good coverage (3+ documents)")
        
        # Keywords with excellent coverage
        print(f"\n{'='*80}")
        print("KEYWORDS WITH EXCELLENT COVERAGE (10+ documents)")
        print(f"{'='*80}\n")
        
        excellent = mapped_df[mapped_df['Documents'] >= 10].copy()
        excellent = excellent.sort_values('Documents', ascending=False)
        
        if not excellent.empty:
            print(f"Found {len(excellent)} keywords with excellent coverage:\n")
            for _, row in excellent.iterrows():
                print(f"  ✓ {row['Keyword']:40s} - {row['Documents']:3d} docs, "
                      f"{row['Total_Chunks']:4d} chunks [{row['Priority']:8s}]")
        else:
            print("No keywords with 10+ document coverage")
        
        # Save detailed distribution report
        output_file = self.mappings_dir / "keyword_distribution_analysis.csv"
        dist_df.to_csv(output_file, index=False)
        print(f"\n✓ Detailed distribution saved to: {output_file}")
        
        return dist_df
    
    def create_keyword_heatmap(self, top_n_keywords: int = 20, top_n_docs: int = 15):
        """
        Create a heatmap showing keyword-document distribution
        
        Args:
            top_n_keywords: Number of top keywords to include
            top_n_docs: Number of top documents to include
        """
        print(f"\n{'='*80}")
        print(f"KEYWORD-DOCUMENT HEATMAP")
        print(f"{'='*80}\n")
        
        # Get top keywords by document count
        keyword_doc_counts = self.df.groupby('Keyword')['Document'].nunique()
        top_keywords = keyword_doc_counts.nlargest(top_n_keywords).index.tolist()
        
        # Get top documents by keyword count
        doc_keyword_counts = self.df.groupby('Document')['Keyword'].nunique()
        top_docs = doc_keyword_counts.nlargest(top_n_docs).index.tolist()
        
        # Filter dataframe
        heatmap_df = self.df[
            (self.df['Keyword'].isin(top_keywords)) &
            (self.df['Document'].isin(top_docs))
        ]
        
        # Create pivot table
        pivot = heatmap_df.pivot_table(
            index='Keyword',
            columns='Document',
            values='Relevance_Score_1to5',
            aggfunc='max'
        )
        
        print(f"Showing top {len(top_keywords)} keywords × top {len(top_docs)} documents")
        print(f"(Relevance scores 1-5, blank = no match)\n")
        print(pivot.fillna('-').to_string())
        
        return pivot


def main():
    """Example usage of the analyzer"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze keyword-to-document mappings with full distribution from keywords.yaml"
    )
    parser.add_argument(
        "--mappings-dir",
        default="./keyword_mappings",
        help="Directory containing mapping files"
    )
    parser.add_argument(
        "--keywords-yaml",
        default="./keywords.yaml",
        help="Path to keywords.yaml file (default: ./keywords.yaml)"
    )
    parser.add_argument(
        "--show-heatmap",
        action="store_true",
        help="Show keyword-document heatmap"
    )
    parser.add_argument(
        "--distribution-only",
        action="store_true",
        help="Show only keyword distribution analysis"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("KEYWORD MAPPING ANALYZER - ENHANCED")
    print("="*80 + "\n")
    
    analyzer = MappingAnalyzer(args.mappings_dir, args.keywords_yaml)
    
    if args.distribution_only:
        # Show only distribution analysis
        analyzer.analyze_keyword_distribution()
        
        if args.show_heatmap:
            analyzer.create_keyword_heatmap()
    else:
        # Run all analyses
        print("\n" + "="*80)
        print("PART 1: BASIC STATISTICS")
        print("="*80)
        
        analyzer.get_top_keywords_by_coverage(n=15)
        analyzer.get_top_documents_by_keyword_coverage(n=15)
        analyzer.analyze_by_category()
        analyzer.analyze_by_priority()
        analyzer.analyze_score_distribution()
        analyzer.find_keyword_gaps(min_docs=3)
        
        # Example: Find documents covering compliance
        print("\n" + "="*80)
        print("PART 2: EXAMPLE ANALYSES")
        print("="*80)
        
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
        
        # NEW: Comprehensive keyword distribution analysis
        print("\n" + "="*80)
        print("PART 3: COMPREHENSIVE KEYWORD DISTRIBUTION")
        print("="*80)
        
        analyzer.analyze_keyword_distribution()
        
        if args.show_heatmap:
            analyzer.create_keyword_heatmap()
    
    print("\n" + "="*80)
    print("✓ Analysis complete!")
    print("\n💡 TIP: Use --distribution-only for just keyword distribution")
    print("💡 TIP: Use --show-heatmap to see keyword-document matrix")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
