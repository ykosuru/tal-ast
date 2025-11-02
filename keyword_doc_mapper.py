"""
Pattern-Based Keyword Mapper
1. Parse keywords.yaml → Build regex patterns
2. Scan files → Find keyword matches with line numbers
3. Create index: keyword → [(file, line_num, context), ...]
"""

import os
import json
import re
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any, Set, Tuple
import yaml

# PDF processing
try:
    import PyPDF2
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("Warning: PDF support disabled (install: pip install PyPDF2 pdfplumber)")


SUPPORTED_EXTENSIONS = {
    'pdf': ['.pdf'],
    'code': [
        '.py', '.c', '.cpp', '.h', '.hpp', '.tal', '.cbl', '.cobol',
        '.java', '.js', '.ts', '.sql', '.pl', '.sh', '.rb', '.php',
        '.cs', '.go', '.rs', '.swift'
    ],
    'text': ['.txt', '.md', '.rst', '.log'],
    'config': ['.json', '.xml', '.yaml', '.yml', '.toml', '.ini', '.cfg']
}

ALL_EXTENSIONS = set()
for exts in SUPPORTED_EXTENSIONS.values():
    ALL_EXTENSIONS.update(exts)


class KeywordPatternBuilder:
    """
    Parse keywords.yaml and build optimized regex patterns
    """
    
    def __init__(self, yaml_path: str):
        self.yaml_path = yaml_path
        self.keywords_by_category = {}
        self.keyword_priorities = {}
        self.patterns = []  # List of (compiled_regex, keyword, category, priority)
        
        print(f"\n{'='*80}")
        print(f"Loading keywords from: {yaml_path}")
        print(f"{'='*80}")
        
        self._load_yaml()
        self._build_patterns()
    
    def _load_yaml(self):
        """Load keywords from YAML"""
        if not os.path.exists(self.yaml_path):
            raise FileNotFoundError(f"Keywords file not found: {self.yaml_path}")
        
        with open(self.yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if not config:
            raise ValueError("Empty YAML configuration")
        
        for category, details in config.items():
            priority = details.get('priority', 'medium')
            keywords = details.get('keywords', [])
            
            if not keywords:
                continue
            
            self.keywords_by_category[category] = {
                'priority': priority,
                'keywords': keywords
            }
            
            for kw in keywords:
                self.keyword_priorities[kw.lower()] = priority
            
            print(f"  ✓ {category:30s} [{priority:8s}] - {len(keywords):3d} keywords")
        
        total_keywords = sum(len(v['keywords']) for v in self.keywords_by_category.values())
        print(f"\nTotal keywords loaded: {total_keywords}")
    
    def _build_patterns(self):
        """Build regex patterns for all keywords"""
        print(f"\n{'='*80}")
        print("Building regex patterns...")
        print(f"{'='*80}")
        
        # Collect all keywords
        all_keywords = []
        for category, data in self.keywords_by_category.items():
            for keyword in data['keywords']:
                all_keywords.append({
                    'text': keyword,
                    'category': category,
                    'priority': data['priority']
                })
        
        # Sort by length (longest first) to match longer phrases before shorter ones
        all_keywords.sort(key=lambda x: len(x['text']), reverse=True)
        
        # Build patterns
        multi_word_count = 0
        single_word_count = 0
        
        for kw_data in all_keywords:
            keyword = kw_data['text']
            category = kw_data['category']
            priority = kw_data['priority']
            
            # Escape special regex characters but preserve word boundaries
            escaped = re.escape(keyword)
            
            # Build pattern with word boundaries
            # Handle multi-word phrases
            if ' ' in keyword:
                # Multi-word: match exactly with word boundaries
                pattern = r'\b' + escaped.replace(r'\ ', r'\s+') + r'\b'
                multi_word_count += 1
            else:
                # Single word: match with word boundaries
                pattern = r'\b' + escaped + r'\b'
                single_word_count += 1
            
            # Compile with case-insensitive flag
            try:
                compiled = re.compile(pattern, re.IGNORECASE)
                self.patterns.append({
                    'regex': compiled,
                    'keyword': keyword,
                    'category': category,
                    'priority': priority
                })
            except re.error as e:
                print(f"  ⚠ Error compiling pattern for '{keyword}': {e}")
        
        print(f"  ✓ Built {len(self.patterns)} regex patterns")
        print(f"    - Multi-word phrases: {multi_word_count}")
        print(f"    - Single-word terms: {single_word_count}")
    
    def get_patterns(self) -> List[Dict]:
        """Return all compiled patterns"""
        return self.patterns


class FileScanner:
    """
    Scan files and find keyword matches with line numbers
    """
    
    def __init__(self, patterns: List[Dict]):
        self.patterns = patterns
        self.pdf_available = PDF_AVAILABLE
    
    def scan_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Scan a single file and return keyword matches
        
        Returns:
            {
                'file': str,
                'file_type': str,
                'total_lines': int,
                'matches': [
                    {
                        'keyword': str,
                        'category': str,
                        'priority': str,
                        'line_number': int,
                        'line_text': str,
                        'match_context': str
                    },
                    ...
                ]
            }
        """
        result = {
            'file': str(file_path.name),
            'file_path': str(file_path),
            'file_type': self._get_file_type(file_path),
            'total_lines': 0,
            'matches': []
        }
        
        # Extract text
        text_lines = self._extract_text_lines(file_path, result['file_type'])
        
        if not text_lines:
            return result
        
        result['total_lines'] = len(text_lines)
        
        # Scan each line for keywords
        for line_num, line_text in enumerate(text_lines, start=1):
            for pattern_data in self.patterns:
                regex = pattern_data['regex']
                keyword = pattern_data['keyword']
                category = pattern_data['category']
                priority = pattern_data['priority']
                
                # Find all matches in this line
                matches = regex.finditer(line_text)
                for match in matches:
                    # Get context around match
                    start = max(0, match.start() - 30)
                    end = min(len(line_text), match.end() + 30)
                    context = line_text[start:end].strip()
                    
                    result['matches'].append({
                        'keyword': keyword,
                        'category': category,
                        'priority': priority,
                        'line_number': line_num,
                        'line_text': line_text.strip(),
                        'match_context': context,
                        'match_position': match.start()
                    })
        
        return result
    
    def _get_file_type(self, file_path: Path) -> str:
        """Determine file type"""
        ext = file_path.suffix.lower()
        for file_type, extensions in SUPPORTED_EXTENSIONS.items():
            if ext in extensions:
                return file_type
        return 'unknown'
    
    def _extract_text_lines(self, file_path: Path, file_type: str) -> List[str]:
        """Extract text as list of lines"""
        if file_type == 'pdf':
            return self._extract_pdf_lines(file_path)
        else:
            return self._extract_text_file_lines(file_path)
    
    def _extract_pdf_lines(self, pdf_path: Path) -> List[str]:
        """Extract lines from PDF"""
        if not self.pdf_available:
            return []
        
        lines = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text() or ""
                    page_lines = text.split('\n')
                    # Add page markers
                    lines.append(f"[PAGE {page_num}]")
                    lines.extend(page_lines)
        except Exception as e:
            print(f"  ⚠ Error reading PDF {pdf_path.name}: {e}")
        
        return lines
    
    def _extract_text_file_lines(self, file_path: Path) -> List[str]:
        """Extract lines from text/code file"""
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.readlines()
            except (UnicodeDecodeError, Exception):
                continue
        
        print(f"  ⚠ Could not read {file_path.name}")
        return []


class KeywordIndex:
    """
    Build and manage the keyword index
    Maps: keyword → [(file, line_num, context), ...]
    """
    
    def __init__(self):
        self.keyword_index = defaultdict(list)  # keyword → list of locations
        self.file_index = defaultdict(list)  # file → list of keywords found
        self.category_index = defaultdict(list)  # category → list of files
        self.priority_index = defaultdict(int)  # keyword → count by priority
        
        self.total_files = 0
        self.total_matches = 0
    
    def add_file_results(self, scan_result: Dict[str, Any]):
        """Add scan results to index"""
        file_name = scan_result['file']
        file_path = scan_result['file_path']
        
        if not scan_result['matches']:
            return
        
        self.total_files += 1
        keywords_in_file = set()
        
        for match in scan_result['matches']:
            keyword = match['keyword']
            category = match['category']
            priority = match['priority']
            line_num = match['line_number']
            context = match['match_context']
            
            # Add to keyword index
            self.keyword_index[keyword].append({
                'file': file_name,
                'file_path': file_path,
                'line_number': line_num,
                'context': context,
                'category': category,
                'priority': priority
            })
            
            keywords_in_file.add(keyword)
            self.total_matches += 1
            
            # Track priority stats
            priority_key = f"{priority}_{keyword}"
            self.priority_index[priority_key] += 1
            
            # Track category
            if file_path not in self.category_index[category]:
                self.category_index[category].append(file_path)
        
        # Track which keywords appear in which files
        for kw in keywords_in_file:
            self.file_index[file_name].append(kw)
    
    def get_keyword_locations(self, keyword: str) -> List[Dict]:
        """Get all locations for a keyword"""
        return self.keyword_index.get(keyword.lower(), [])
    
    def get_file_keywords(self, file_name: str) -> List[str]:
        """Get all keywords found in a file"""
        return self.file_index.get(file_name, [])
    
    def get_top_keywords(self, n: int = 20) -> List[Tuple[str, int]]:
        """Get top N keywords by frequency"""
        keyword_counts = {
            kw: len(locations) 
            for kw, locations in self.keyword_index.items()
        }
        sorted_keywords = sorted(
            keyword_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        return sorted_keywords[:n]
    
    def save(self, output_path: str):
        """Save index to JSON"""
        output = {
            'metadata': {
                'total_files': self.total_files,
                'total_matches': self.total_matches,
                'unique_keywords': len(self.keyword_index),
                'keywords_found': list(self.keyword_index.keys())
            },
            'keyword_index': dict(self.keyword_index),
            'file_index': dict(self.file_index),
            'category_index': dict(self.category_index)
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n✓ Index saved to: {output_path}")


class KeywordMapper:
    """
    Main class: orchestrates the entire mapping process
    """
    
    def __init__(
        self, 
        keywords_yaml: str,
        files_folder: str,
        file_extensions: List[str] = None
    ):
        self.keywords_yaml = keywords_yaml
        self.files_folder = Path(files_folder)
        self.file_extensions = file_extensions or list(ALL_EXTENSIONS)
        
        # Build patterns from YAML
        pattern_builder = KeywordPatternBuilder(keywords_yaml)
        self.patterns = pattern_builder.get_patterns()
        
        # Initialize scanner and index
        self.scanner = FileScanner(self.patterns)
        self.index = KeywordIndex()
    
    def scan_folder(self):
        """Scan all files in folder"""
        print(f"\n{'='*80}")
        print(f"Scanning folder: {self.files_folder}")
        print(f"{'='*80}")
        
        # Find all files
        files = []
        for ext in self.file_extensions:
            found = list(self.files_folder.glob(f"**/*{ext}"))
            files.extend(found)
        
        if not files:
            print("No files found!")
            return
        
        print(f"Found {len(files)} files to scan\n")
        
        # Scan each file
        for idx, file_path in enumerate(files, start=1):
            print(f"[{idx}/{len(files)}] Scanning: {file_path.name} ...", end=' ')
            
            result = self.scanner.scan_file(file_path)
            
            if result['matches']:
                print(f"✓ {len(result['matches'])} matches")
                self.index.add_file_results(result)
            else:
                print("(no matches)")
    
    def save_index(self, output_path: str = "./keyword_index.json"):
        """Save the index"""
        self.index.save(output_path)
    
    def print_statistics(self):
        """Print indexing statistics"""
        print(f"\n{'='*80}")
        print("INDEXING STATISTICS")
        print(f"{'='*80}")
        
        print(f"\nTotal files scanned: {self.index.total_files}")
        print(f"Total keyword matches: {self.index.total_matches}")
        print(f"Unique keywords found: {len(self.index.keyword_index)}")
        
        # Top keywords
        print(f"\n{'='*80}")
        print("TOP 20 DOMAIN KEYWORDS (by frequency)")
        print(f"{'='*80}")
        
        top_keywords = self.index.get_top_keywords(n=20)
        
        for i, (keyword, count) in enumerate(top_keywords, start=1):
            # Get priority from first occurrence
            locations = self.index.get_keyword_locations(keyword)
            priority = locations[0]['priority'] if locations else 'unknown'
            
            # Get categories
            categories = set(loc['category'] for loc in locations[:3])
            category_str = ', '.join(list(categories)[:2])
            
            print(f"{i:2d}. {keyword:35s} [{priority:8s}] - {count:4d} occurrences")
            print(f"    Categories: {category_str}")
        
        # Category distribution
        print(f"\n{'='*80}")
        print("KEYWORD MATCHES BY CATEGORY")
        print(f"{'='*80}")
        
        category_counts = defaultdict(int)
        for keyword, locations in self.index.keyword_index.items():
            for loc in locations:
                category_counts[loc['category']] += 1
        
        for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {category:35s} - {count:5d} matches")
    
    def search_keyword(self, keyword: str, max_results: int = 10):
        """Search for a specific keyword"""
        print(f"\n{'='*80}")
        print(f"SEARCH: '{keyword}'")
        print(f"{'='*80}")
        
        locations = self.index.get_keyword_locations(keyword.lower())
        
        if not locations:
            print(f"No matches found for '{keyword}'")
            return
        
        print(f"Found {len(locations)} occurrences\n")
        
        for i, loc in enumerate(locations[:max_results], start=1):
            print(f"[{i}] {loc['file']}:{loc['line_number']}")
            print(f"    Category: {loc['category']} | Priority: {loc['priority']}")
            print(f"    Context: ...{loc['context']}...")
            print()
        
        if len(locations) > max_results:
            print(f"... and {len(locations) - max_results} more occurrences")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Pattern-Based Keyword Mapper - Map YAML keywords to code/docs"
    )
    parser.add_argument("--keywords-yaml", required=True, help="Path to keywords.yaml")
    parser.add_argument("--folder", required=True, help="Folder to scan")
    parser.add_argument("--extensions", nargs='+', help="File extensions (e.g., .py .tal .pdf)")
    parser.add_argument("--output", default="./keyword_index.json", help="Output index file")
    parser.add_argument("--search", help="Search for a specific keyword")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("PATTERN-BASED KEYWORD MAPPER")
    print("=" * 80)
    
    # Initialize mapper
    mapper = KeywordMapper(
        keywords_yaml=args.keywords_yaml,
        files_folder=args.folder,
        file_extensions=args.extensions
    )
    
    # Scan folder
    mapper.scan_folder()
    
    # Save index
    mapper.save_index(args.output)
    
    # Print statistics
    mapper.print_statistics()
    
    # Search if requested
    if args.search:
        mapper.search_keyword(args.search)
    
    print(f"\n{'='*80}")
    print("DONE!")
    print(f"{'='*80}")
    print(f"\nIndex saved to: {args.output}")
    print(f"To search: python keyword_mapper.py --keywords-yaml {args.keywords_yaml} \\")
    print(f"                                     --folder {args.folder} \\")
    print(f"                                     --search 'OFAC screening'")


if __name__ == "__main__":
    main()
