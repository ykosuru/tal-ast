"""
Keyword Index Search Tool
Search the keyword index created by keyword_mapper.py
"""

import json
import sys
from pathlib import Path
from typing import List, Dict
import re


class KeywordIndexSearcher:
    """Search and explore the keyword index"""
    
    def __init__(self, index_path: str):
        self.index_path = index_path
        
        with open(index_path, 'r') as f:
            data = json.load(f)
        
        self.metadata = data['metadata']
        self.keyword_index = data['keyword_index']
        self.file_index = data['file_index']
        self.category_index = data.get('category_index', {})
        
        print(f"✓ Loaded index from: {index_path}")
        print(f"  - Total files: {self.metadata['total_files']}")
        print(f"  - Total matches: {self.metadata['total_matches']}")
        print(f"  - Unique keywords: {self.metadata['unique_keywords']}")
    
    def search(self, query: str, case_sensitive: bool = False) -> List[Dict]:
        """Search for keywords (supports partial matching)"""
        results = []
        
        if case_sensitive:
            pattern = re.compile(re.escape(query))
        else:
            pattern = re.compile(re.escape(query), re.IGNORECASE)
        
        for keyword, locations in self.keyword_index.items():
            if pattern.search(keyword):
                results.append({
                    'keyword': keyword,
                    'occurrences': len(locations),
                    'locations': locations
                })
        
        return results
    
    def get_exact_keyword(self, keyword: str) -> List[Dict]:
        """Get locations for exact keyword match"""
        return self.keyword_index.get(keyword, [])
    
    def get_keywords_in_file(self, filename: str) -> List[str]:
        """Get all keywords found in a file"""
        return self.file_index.get(filename, [])
    
    def get_keywords_by_category(self, category: str) -> List[str]:
        """Get all keywords in a category"""
        keywords = []
        for keyword, locations in self.keyword_index.items():
            if any(loc['category'] == category for loc in locations):
                keywords.append(keyword)
        return keywords
    
    def get_keywords_by_priority(self, priority: str) -> List[str]:
        """Get all keywords with a specific priority"""
        keywords = []
        for keyword, locations in self.keyword_index.items():
            if any(loc['priority'] == priority for loc in locations):
                keywords.append(keyword)
        return keywords
    
    def list_all_keywords(self):
        """List all keywords found"""
        return sorted(self.metadata['keywords_found'])
    
    def print_search_results(self, results: List[Dict], max_per_keyword: int = 5):
        """Pretty print search results"""
        if not results:
            print("\nNo matches found")
            return
        
        print(f"\n{'='*80}")
        print(f"FOUND {len(results)} MATCHING KEYWORD(S)")
        print(f"{'='*80}\n")
        
        for i, result in enumerate(results, start=1):
            keyword = result['keyword']
            occurrences = result['occurrences']
            locations = result['locations']
            
            # Get priority and category from first location
            priority = locations[0]['priority']
            category = locations[0]['category']
            
            print(f"[{i}] {keyword}")
            print(f"    Priority: {priority} | Category: {category}")
            print(f"    Found in {occurrences} location(s):\n")
            
            for j, loc in enumerate(locations[:max_per_keyword], start=1):
                print(f"    {j}. {loc['file']}:{loc['line_number']}")
                print(f"       {loc['context'][:100]}")
                print()
            
            if occurrences > max_per_keyword:
                print(f"    ... and {occurrences - max_per_keyword} more location(s)\n")
            
            print("-" * 80)
    
    def print_file_summary(self, filename: str):
        """Print summary of keywords in a file"""
        keywords = self.get_keywords_in_file(filename)
        
        if not keywords:
            print(f"\nNo keywords found in file: {filename}")
            return
        
        print(f"\n{'='*80}")
        print(f"FILE: {filename}")
        print(f"{'='*80}")
        print(f"Found {len(keywords)} unique keywords:\n")
        
        # Group by category
        by_category = {}
        for kw in keywords:
            locations = self.keyword_index[kw]
            for loc in locations:
                if loc['file'] == filename:
                    category = loc['category']
                    if category not in by_category:
                        by_category[category] = []
                    by_category[category].append(kw)
                    break
        
        for category, kw_list in sorted(by_category.items()):
            print(f"\n{category}:")
            for kw in sorted(set(kw_list)):
                # Count occurrences in this file
                count = sum(1 for loc in self.keyword_index[kw] 
                           if loc['file'] == filename)
                print(f"  • {kw:35s} ({count} occurrence(s))")
    
    def print_category_summary(self):
        """Print summary by category"""
        print(f"\n{'='*80}")
        print("KEYWORDS BY CATEGORY")
        print(f"{'='*80}\n")
        
        by_category = {}
        for keyword, locations in self.keyword_index.items():
            for loc in locations:
                category = loc['category']
                if category not in by_category:
                    by_category[category] = []
                by_category[category].append(keyword)
        
        for category in sorted(by_category.keys()):
            unique_keywords = set(by_category[category])
            total_occurrences = sum(len(self.keyword_index[kw]) 
                                   for kw in unique_keywords)
            
            print(f"\n{category}")
            print(f"  Unique keywords: {len(unique_keywords)}")
            print(f"  Total occurrences: {total_occurrences}")
            print(f"  Keywords:")
            for kw in sorted(unique_keywords)[:10]:
                count = len(self.keyword_index[kw])
                print(f"    • {kw:30s} ({count}x)")
            if len(unique_keywords) > 10:
                print(f"    ... and {len(unique_keywords) - 10} more")


def main():
    """Interactive search interface"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Search keyword index"
    )
    parser.add_argument("--index", required=True, help="Path to keyword_index.json")
    parser.add_argument("--search", help="Search for keyword")
    parser.add_argument("--file", help="Show keywords in file")
    parser.add_argument("--category", help="Show keywords by category")
    parser.add_argument("--priority", help="Show keywords by priority (critical/high/medium/low)")
    parser.add_argument("--list-all", action="store_true", help="List all keywords")
    parser.add_argument("--summary", action="store_true", help="Show category summary")
    parser.add_argument("--max-results", type=int, default=5, help="Max results per keyword")
    
    args = parser.parse_args()
    
    # Load index
    searcher = KeywordIndexSearcher(args.index)
    
    # Execute command
    if args.search:
        results = searcher.search(args.search)
        searcher.print_search_results(results, max_per_keyword=args.max_results)
    
    elif args.file:
        searcher.print_file_summary(args.file)
    
    elif args.category:
        keywords = searcher.get_keywords_by_category(args.category)
        print(f"\nKeywords in category '{args.category}':")
        for kw in sorted(keywords):
            count = len(searcher.keyword_index[kw])
            print(f"  • {kw:40s} ({count} occurrences)")
    
    elif args.priority:
        keywords = searcher.get_keywords_by_priority(args.priority)
        print(f"\nKeywords with priority '{args.priority}':")
        for kw in sorted(keywords):
            count = len(searcher.keyword_index[kw])
            print(f"  • {kw:40s} ({count} occurrences)")
    
    elif args.list_all:
        keywords = searcher.list_all_keywords()
        print(f"\nAll {len(keywords)} keywords:")
        for kw in keywords:
            count = len(searcher.keyword_index[kw])
            locations = searcher.keyword_index[kw]
            priority = locations[0]['priority'] if locations else 'unknown'
            print(f"  [{priority:8s}] {kw:40s} ({count}x)")
    
    elif args.summary:
        searcher.print_category_summary()
    
    else:
        print("\nNo action specified. Use --help for options")
        print("\nQuick examples:")
        print(f"  python search_index.py --index {args.index} --search 'OFAC'")
        print(f"  python search_index.py --index {args.index} --file 'test_sample.tal'")
        print(f"  python search_index.py --index {args.index} --category compliance_screening")
        print(f"  python search_index.py --index {args.index} --summary")


if __name__ == "__main__":
    main()
