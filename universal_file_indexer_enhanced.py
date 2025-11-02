"""
Universal File Indexer with Domain-Focused Keyword Extraction
Now prioritizes business domain keywords over system-level code keywords
"""

import os
import json
import math
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional, Tuple, Set
import hashlib
import re

# YAML support for custom keywords
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    print("Warning: PyYAML not available, using built-in taxonomy only")
    print("Install with: pip install pyyaml")

# PDF processing (optional)
try:
    import PyPDF2
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("Warning: PDF libraries not available, PDF indexing disabled")

# Stemming
try:
    from nltk.stem import PorterStemmer
    import nltk
    STEMMER_AVAILABLE = True
    print("✓ NLTK PorterStemmer available")
except ImportError:
    STEMMER_AVAILABLE = False
    print("Warning: nltk not available, stemming disabled")

# Vector embeddings (optional)
import numpy as np
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("Warning: sentence_transformers not available, embeddings disabled")

# Vector database (optional)
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: faiss not available, using keyword search only")

import pickle

# BM25 for fast keyword search
from rank_bm25 import BM25Okapi


# Supported file extensions by category
SUPPORTED_EXTENSIONS = {
    'pdf': ['.pdf'],
    'code': [
        '.py', '.c', '.cpp', '.h', '.hpp', '.cc', '.cxx',
        '.java', '.scala', '.kt', '.groovy',
        '.js', '.ts', '.jsx', '.tsx',
        '.go', '.rs', '.swift',
        '.tal', '.cbl', '.cobol', '.cob',
        '.sql', '.pl', '.pm',
        '.rb', '.php', '.sh', '.bash',
        '.cs', '.vb', '.fs'
    ],
    'text': [
        '.txt', '.md', '.rst', '.log', '.text',
        '.doc', '.rtf'
    ],
    'config': [
        '.json', '.xml', '.yaml', '.yml', '.toml', '.ini', '.cfg',
        '.properties', '.conf', '.config'
    ],
    'markup': [
        '.html', '.htm', '.xhtml', '.css', '.scss', '.sass'
    ]
}

# Flatten all extensions
ALL_SUPPORTED_EXTENSIONS = set()
for extensions in SUPPORTED_EXTENSIONS.values():
    ALL_SUPPORTED_EXTENSIONS.update(extensions)


class TextStemmer:
    """Centralized stemming utility using Porter Stemmer"""
    
    def __init__(self, use_stemming: bool = True):
        self.use_stemming = use_stemming and STEMMER_AVAILABLE
        
        if self.use_stemming:
            self.stemmer = PorterStemmer()
            print("✓ Stemming enabled (Porter Stemmer)")
        else:
            self.stemmer = None
            print("⚠ Stemming disabled")
        
        self._stem_cache = {}
    
    def stem(self, word: str) -> str:
        """Stem a single word with caching"""
        if not self.use_stemming or not word:
            return word.lower()
        
        word_lower = word.lower()
        if word_lower in self._stem_cache:
            return self._stem_cache[word_lower]
        
        stemmed = self.stemmer.stem(word_lower)
        self._stem_cache[word_lower] = stemmed
        return stemmed
    
    def stem_tokens(self, tokens: List[str]) -> List[str]:
        """Stem a list of tokens"""
        return [self.stem(token) for token in tokens]
    
    def stem_text(self, text: str) -> str:
        """Stem all words in text, preserving structure"""
        words = re.findall(r'\b[\w-]+\b', text)
        stemmed_words = [self.stem(word) for word in words]
        return ' '.join(stemmed_words)


class DomainKeywordLoader:
    """
    Load and manage domain-specific keywords from YAML configuration
    Supports priority levels: critical, high, medium, low
    """
    
    def __init__(self, yaml_path: Optional[str] = None, stemmer: Optional[TextStemmer] = None):
        self.stemmer = stemmer
        self.keywords_by_priority = {
            'critical': [],
            'high': [],
            'medium': [],
            'low': []
        }
        self.all_domain_keywords = set()
        self.keyword_to_category = {}
        self.keyword_to_priority = {}
        
        # Priority weights for scoring
        self.priority_weights = {
            'critical': 10.0,
            'high': 5.0,
            'medium': 2.0,
            'low': 1.0
        }
        
        if yaml_path and YAML_AVAILABLE:
            self.load_from_yaml(yaml_path)
        else:
            print("⚠ No YAML keywords loaded, using built-in taxonomy only")
    
    def load_from_yaml(self, yaml_path: str):
        """Load keywords from YAML configuration"""
        try:
            with open(yaml_path, 'r') as f:
                config = yaml.safe_load(f)
            
            for category, details in config.items():
                priority = details.get('priority', 'medium')
                keywords = details.get('keywords', [])
                
                for kw in keywords:
                    kw_normalized = kw.lower().strip()
                    
                    # Store original and stemmed versions
                    self.all_domain_keywords.add(kw_normalized)
                    self.keyword_to_category[kw_normalized] = category
                    self.keyword_to_priority[kw_normalized] = priority
                    self.keywords_by_priority[priority].append(kw_normalized)
                    
                    # Also store stemmed version if stemmer available
                    if self.stemmer and self.stemmer.use_stemming:
                        kw_stemmed = self.stemmer.stem_text(kw_normalized)
                        if kw_stemmed != kw_normalized:
                            self.all_domain_keywords.add(kw_stemmed)
                            self.keyword_to_category[kw_stemmed] = category
                            self.keyword_to_priority[kw_stemmed] = priority
            
            print(f"✓ Loaded {len(self.all_domain_keywords)} domain keywords from {yaml_path}")
            print(f"  - Critical: {len(self.keywords_by_priority['critical'])}")
            print(f"  - High: {len(self.keywords_by_priority['high'])}")
            print(f"  - Medium: {len(self.keywords_by_priority['medium'])}")
            print(f"  - Low: {len(self.keywords_by_priority['low'])}")
            
        except Exception as e:
            print(f"Error loading YAML keywords: {e}")
    
    def get_priority_weight(self, keyword: str) -> float:
        """Get priority weight for a keyword"""
        priority = self.keyword_to_priority.get(keyword.lower(), 'low')
        return self.priority_weights[priority]
    
    def is_domain_keyword(self, keyword: str) -> bool:
        """Check if keyword is a domain keyword"""
        return keyword.lower() in self.all_domain_keywords
    
    def get_category(self, keyword: str) -> Optional[str]:
        """Get category for a keyword"""
        return self.keyword_to_category.get(keyword.lower())


class BusinessCapabilityTaxonomy:
    """Wire Processing Business Capabilities taxonomy (built-in fallback)"""
    
    CAPABILITIES = {
        "Core Payment & Network": [
            "clearing networks", "fed", "chips", "swift", "clearing house",
            "network gateways", "network connectivity", "network acknowledgments",
            "fedwire", "rtgs", "ach"
        ],
        
        "Payment Processing & Execution": [
            "payment initiation", "payment routing", "payment execution",
            "preadvising", "cover payments", "liquidity management",
            "debit confirmation", "credit confirmation", "outbound payment",
            "hard posting", "cutoffs", "workflow scheduling", "orchestration",
            "split advising", "intraday liquidity", "book transfer",
            "eod processing", "fee determination", "payment enrichment",
            "payment repair", "payment validation", "straight through processing",
            "stp", "payment returns", "payment prioritization", "warehousing"
        ],
        
        "Compliance & Screening": [
            "sanctions screening", "fircosoft", "ofac", "ofac screening",
            "fraud checking", "anti-money laundering", "aml", "kyc validation",
            "watchlist screening", "name screening", "ceo fraud", "cfm"
        ],
        
        "Validation": [
            "bic validation", "bic code", "iban validation", "iban",
            "party validation", "account validation", "date validation",
            "amount validation", "currency validation", "routing validation",
            "aba", "sort code", "address validation"
        ],
        
        "ISO Standards & Formats": [
            "iso20022", "iso 20022", "pacs.008", "pacs.009", "pacs.002",
            "pain.001", "camt.053", "mt103", "mt202", "mt199"
        ],
        
        "Transaction Processing": [
            "wire transfer", "wire payment", "domestic wire", "international wire",
            "cross-border payment", "same-day payment", "clearing", "settlement",
            "netting", "gross settlement", "ace repair", "automated clearing enhancement"
        ]
    }
    
    @classmethod
    def get_all_keywords(cls) -> List[str]:
        """Get all keywords across all capabilities"""
        all_keywords = []
        for keywords in cls.CAPABILITIES.values():
            all_keywords.extend(keywords)
        return list(set(all_keywords))


class UniversalFileExtractor:
    """Extract text from various file types"""
    
    def __init__(self, enable_ocr: bool = False):
        self.enable_ocr = enable_ocr
        self.pdf_available = PDF_AVAILABLE
    
    def get_file_type(self, file_path: Path) -> str:
        """Determine file type from extension"""
        ext = file_path.suffix.lower()
        for file_type, extensions in SUPPORTED_EXTENSIONS.items():
            if ext in extensions:
                return file_type
        return 'unknown'
    
    def extract(self, file_path: Path) -> Dict[str, Any]:
        """Universal file extraction dispatcher"""
        file_type = self.get_file_type(file_path)
        
        if file_type == 'pdf':
            return self._extract_pdf(file_path)
        elif file_type == 'code':
            return self._extract_code(file_path)
        elif file_type in ['text', 'config', 'markup']:
            return self._extract_text(file_path)
        else:
            return self._extract_text(file_path)
    
    def _extract_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract text from PDF files"""
        if not self.pdf_available:
            return {
                'text': '', 'file_type': 'pdf', 'error': 'PDF libraries not available',
                'line_count': 0, 'has_tables': False
            }
        
        content = {
            "text": "", "file_type": "pdf", "has_tables": False,
            "tables": [], "images": [], "line_count": 0
        }
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text() or ""
                    content["text"] += page_text + "\n"
                    
                    tables = page.extract_tables()
                    if tables:
                        content["has_tables"] = True
                        for table in tables:
                            table_text = self._table_to_text(table)
                            content["tables"].append({"page": page_num + 1, "text": table_text})
                            content["text"] += f"\n[TABLE]\n{table_text}\n[/TABLE]\n"
        except Exception as e:
            print(f"Error extracting from {pdf_path}: {e}")
        
        content["line_count"] = content["text"].count('\n')
        return content
    
    def _extract_code(self, code_path: Path) -> Dict[str, Any]:
        """Extract text from code files"""
        content = {
            "text": "", "file_type": "code",
            "language": code_path.suffix[1:], "line_count": 0, "encoding": "utf-8"
        }
        
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        for encoding in encodings:
            try:
                with open(code_path, 'r', encoding=encoding) as f:
                    content["text"] = f.read()
                content["encoding"] = encoding
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                content["error"] = str(e)
                break
        
        content["line_count"] = content["text"].count('\n')
        content["metadata"] = self._extract_code_metadata(content["text"], content["language"])
        return content
    
    def _extract_text(self, text_path: Path) -> Dict[str, Any]:
        """Extract text from plain text files"""
        content = {"text": "", "file_type": "text", "line_count": 0, "encoding": "utf-8"}
        
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        for encoding in encodings:
            try:
                with open(text_path, 'r', encoding=encoding) as f:
                    content["text"] = f.read()
                content["encoding"] = encoding
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                content["error"] = str(e)
                break
        
        content["line_count"] = content["text"].count('\n')
        return content
    
    def _extract_code_metadata(self, code_text: str, language: str) -> Dict[str, Any]:
        """Extract metadata from code"""
        metadata = {"functions": [], "classes": [], "imports": [], "has_main": False}
        
        patterns = {
            'python': {
                'function': r'def\s+(\w+)\s*\(',
                'class': r'class\s+(\w+)\s*[:\(]',
                'import': r'(?:from\s+\S+\s+)?import\s+(\S+)'
            },
            'java': {
                'function': r'(?:public|private|protected)?\s+\w+\s+(\w+)\s*\(',
                'class': r'(?:public|private)?\s+class\s+(\w+)',
                'import': r'import\s+([\w\.]+);'
            },
            'c': {
                'function': r'\w+\s+(\w+)\s*\([^)]*\)\s*\{',
                'class': r'(?:struct|typedef\s+struct)\s+(\w+)',
                'include': r'#include\s+[<"]([^>"]+)[>"]'
            },
            'tal': {
                'function': r'PROC\s+(\w+)',
                'subproc': r'SUBPROC\s+(\w+)',
            }
        }
        
        lang_patterns = patterns.get(language, patterns.get('python', {}))
        
        if 'function' in lang_patterns:
            functions = re.findall(lang_patterns['function'], code_text, re.IGNORECASE)
            metadata['functions'] = list(set(functions))[:50]
            if 'main' in [f.lower() for f in functions]:
                metadata['has_main'] = True
        
        if 'class' in lang_patterns:
            classes = re.findall(lang_patterns['class'], code_text)
            metadata['classes'] = list(set(classes))[:50]
        
        import_key = 'import' if 'import' in lang_patterns else 'include'
        if import_key in lang_patterns:
            imports = re.findall(lang_patterns[import_key], code_text)
            metadata['imports'] = list(set(imports))[:50]
        
        return metadata
    
    def _table_to_text(self, table: List[List]) -> str:
        """Convert table structure to text"""
        if not table:
            return ""
        text_rows = []
        for row in table:
            cleaned_row = [str(cell) if cell else "" for cell in row]
            text_rows.append(" | ".join(cleaned_row))
        return "\n".join(text_rows)


class DomainFocusedKeywordExtractor:
    """
    Domain-focused keyword extraction
    Prioritizes business keywords over system/code keywords
    """
    
    def __init__(
        self, 
        stemmer: Optional[TextStemmer] = None,
        domain_keywords: Optional[DomainKeywordLoader] = None
    ):
        self.stemmer = stemmer
        self.domain_keywords = domain_keywords
        
        # System-level keywords to SUPPRESS (low priority)
        self.system_keywords = {
            # TAL system functions
            '$len', '$numeric', '$scan', '$offset', '$type', '$occurs',
            '$dbl', '$ifix', '$fix', '$flt', '$fltu', '$comp', '$ladr',
            '$xadr', '$carry', '$overflow', '$special',
            
            # Generic programming constructs
            'if', 'else', 'then', 'endif', 'while', 'for', 'loop',
            'return', 'call', 'goto', 'exit', 'break', 'continue',
            'begin', 'end', 'proc', 'subproc', 'int', 'string',
            'char', 'void', 'static', 'public', 'private', 'protected',
            
            # Generic variable/field names
            'i', 'j', 'k', 'x', 'y', 'z', 'temp', 'tmp', 'count',
            'index', 'flag', 'status', 'result', 'value', 'data'
        }
        
        # Domain-specific patterns (HIGH PRIORITY)
        self.domain_patterns = [
            # ISO standards
            (r'ISO[\s-]?20022', 'iso_standard', 8.0),
            (r'pacs\.\d+(?:\.\d+)*', 'payment_message', 8.0),
            (r'pain\.\d+(?:\.\d+)*', 'payment_message', 8.0),
            (r'camt\.\d+(?:\.\d+)*', 'payment_message', 8.0),
            (r'MT\d{3}', 'swift_message', 7.0),
            
            # ACE codes
            (r'ACE[\s-]?\d+', 'ace_code', 7.0),
            (r'ACE[\s_]?[A-Z]+', 'ace_code', 7.0),
            
            # BIC and IBAN
            (r'[A-Z]{6}[A-Z0-9]{2}(?:[A-Z0-9]{3})?', 'bic_code', 6.0),
            (r'[A-Z]{2}\d{2}[A-Z0-9]{10,30}', 'iban', 6.0),
            
            # Business functions (NOT system functions)
            (r'\$OFAC_SCREEN_PARTY', 'domain_function', 10.0),
            (r'\$VALIDATE_[A-Z_]+', 'domain_function', 8.0),
            (r'\$PAYMENT_[A-Z_]+', 'domain_function', 8.0),
            (r'\$SCREEN_[A-Z_]+', 'domain_function', 8.0),
        ]
        
        # Common stopwords
        self.stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
            'to', 'for', 'of', 'with', 'from', 'by', 'as', 'is', 'was',
            'are', 'were', 'been', 'be', 'have', 'has', 'had', 'do',
            'does', 'did', 'will', 'would', 'should', 'could', 'may',
            'might', 'must', 'can', 'this', 'that', 'these', 'those'
        }
    
    def extract(self, text: str, max_keywords: int = 20) -> List[Tuple[str, float]]:
        """
        Extract keywords with DOMAIN FOCUS
        Returns: List of (keyword, confidence) tuples
        """
        if not text or len(text.strip()) < 10:
            return []
        
        keyword_scores = defaultdict(float)
        text_lower = text.lower()
        
        # Tokenize and stem
        words = re.findall(r'\b[\w-]+\b', text_lower)
        if self.stemmer and self.stemmer.use_stemming:
            stemmed_words = [self.stemmer.stem(w) for w in words]
        else:
            stemmed_words = words
        
        # 1. HIGHEST PRIORITY: Domain patterns
        for pattern, pattern_type, score_boost in self.domain_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                keyword_scores[match.lower()] += score_boost
        
        # 2. HIGHEST PRIORITY: YAML domain keywords
        if self.domain_keywords:
            for kw in self.domain_keywords.all_domain_keywords:
                # Check for multi-word keywords
                if ' ' in kw:
                    kw_tokens = kw.split()
                    # Look for phrase matches
                    for i in range(len(words) - len(kw_tokens) + 1):
                        if words[i:i+len(kw_tokens)] == kw_tokens:
                            priority_weight = self.domain_keywords.get_priority_weight(kw)
                            keyword_scores[kw] += priority_weight * 5.0
                else:
                    # Single word keyword
                    if self.stemmer and self.stemmer.use_stemming:
                        kw_stem = self.stemmer.stem(kw)
                        count = stemmed_words.count(kw_stem)
                    else:
                        count = words.count(kw)
                    
                    if count > 0:
                        priority_weight = self.domain_keywords.get_priority_weight(kw)
                        keyword_scores[kw] += priority_weight * math.log1p(count) * 3.0
        
        # 3. Built-in business capability keywords
        capability_keywords = BusinessCapabilityTaxonomy.get_all_keywords()
        for kw in capability_keywords:
            if ' ' in kw:
                # Multi-word business term
                if kw in text_lower:
                    count = text_lower.count(kw)
                    keyword_scores[kw] += 4.0 * math.log1p(count)
            else:
                # Single word
                if self.stemmer and self.stemmer.use_stemming:
                    kw_stem = self.stemmer.stem(kw)
                    count = stemmed_words.count(kw_stem)
                else:
                    count = words.count(kw)
                
                if count > 0:
                    keyword_scores[kw] += 3.0 * math.log1p(count)
        
        # 4. Capitalized words (lower priority now)
        capitalized_words = re.findall(r'\b[A-Z][a-z]{2,}\b', text)
        for word in capitalized_words:
            word_lower = word.lower()
            if (word_lower not in self.stopwords and 
                word_lower not in self.system_keywords):
                keyword_scores[word_lower] += 1.0
        
        # 5. SUPPRESS system keywords
        for system_kw in self.system_keywords:
            if system_kw in keyword_scores:
                # Reduce score dramatically
                keyword_scores[system_kw] *= 0.1
        
        # Filter and normalize
        filtered = []
        for kw, score in keyword_scores.items():
            # Skip if too short, stopword, or system keyword with low score
            if (len(kw) > 2 and 
                kw not in self.stopwords and
                not all(c.isdigit() for c in kw) and
                score > 0.5):  # Minimum threshold
                filtered.append((kw, score))
        
        # Sort by score and return top N
        filtered.sort(key=lambda x: x[1], reverse=True)
        return filtered[:max_keywords]


class ImprovedQueryProcessor:
    """Pre-process queries with domain focus"""
    
    def __init__(self, stemmer: Optional[TextStemmer] = None):
        self.stemmer = stemmer
        
        self.stopwords = {
            'how', 'what', 'when', 'where', 'why', 'who', 'which',
            'do', 'does', 'did', 'is', 'are', 'was', 'were', 'be',
            'have', 'has', 'had', 'will', 'would', 'should', 'could',
            'implement', 'create', 'build', 'make', 'develop', 'setup',
            'explain', 'describe', 'show', 'tell', 'need', 'want', 'help',
            'a', 'an', 'the', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'from', 'by', 'as', 'into', 'through', 'this', 'that'
        }
        
        if self.stemmer and self.stemmer.use_stemming:
            self.stopwords = {self.stemmer.stem(sw) for sw in self.stopwords}
    
    def extract_core_terms(self, query: str) -> Dict[str, Any]:
        """Extract core searchable terms from query"""
        original = query.strip()
        query_lower = query.lower()
        
        is_question = any(query_lower.startswith(q) for q in 
                         ['how', 'what', 'when', 'where', 'why', 'who'])
        
        tokens = re.findall(r'\b[\w-]+\b', query_lower)
        
        core_terms = []
        for token in tokens:
            if len(token) <= 1 or token.isdigit():
                continue
            
            if self.stemmer and self.stemmer.use_stemming:
                stemmed_token = self.stemmer.stem(token)
            else:
                stemmed_token = token
            
            if stemmed_token not in self.stopwords:
                core_terms.append(stemmed_token)
        
        cleaned_query = ' '.join(core_terms)
        
        return {
            'core_terms': core_terms,
            'original_query': original,
            'stemmed_query': cleaned_query,
            'is_question': is_question
        }


class CapabilityMapper:
    """Map documents to business capabilities"""
    
    def __init__(
        self, 
        stemmer: Optional[TextStemmer] = None,
        domain_keywords: Optional[DomainKeywordLoader] = None
    ):
        self.taxonomy = BusinessCapabilityTaxonomy()
        self.stemmer = stemmer
        self.domain_keywords = domain_keywords
        self.capability_keywords = {}
        
        # Build keyword to capability mapping
        for capability, keywords in self.taxonomy.CAPABILITIES.items():
            for kw in keywords:
                if self.stemmer and self.stemmer.use_stemming:
                    kw_stem = self.stemmer.stem_text(kw)
                else:
                    kw_stem = kw
                
                if kw_stem not in self.capability_keywords:
                    self.capability_keywords[kw_stem] = []
                self.capability_keywords[kw_stem].append(capability)
    
    def map_to_capabilities(
        self, 
        keywords: List[Tuple[str, float]], 
        text: str
    ) -> List[Tuple[str, float]]:
        """Map extracted keywords to business capabilities"""
        capability_scores = defaultdict(float)
        text_lower = text.lower()
        
        if self.stemmer and self.stemmer.use_stemming:
            text_stemmed = self.stemmer.stem_text(text_lower)
        else:
            text_stemmed = text_lower
        
        # Score based on keywords
        for keyword, kw_score in keywords:
            if keyword in self.capability_keywords:
                for capability in self.capability_keywords[keyword]:
                    capability_scores[capability] += kw_score
        
        # Direct matching
        for capability, keywords_list in self.taxonomy.CAPABILITIES.items():
            for kw in keywords_list:
                if self.stemmer and self.stemmer.use_stemming:
                    kw_stem = self.stemmer.stem_text(kw)
                    count = text_stemmed.count(kw_stem)
                else:
                    count = text_lower.count(kw)
                
                if count > 0:
                    capability_scores[capability] += math.log1p(count) * 1.5
        
        # Normalize
        if capability_scores:
            max_score = max(capability_scores.values())
            capability_scores = {
                cap: score / max_score 
                for cap, score in capability_scores.items()
            }
        
        sorted_capabilities = sorted(
            capability_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return sorted_capabilities


class UniversalFileIndexer:
    """Universal file indexer with domain-focused keyword extraction"""
    
    def __init__(
        self,
        files_folder: str,
        index_path: str = "./universal_index",
        keywords_yaml: Optional[str] = None,
        file_extensions: Optional[List[str]] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        use_stemming: bool = True,
        use_embeddings: bool = False
    ):
        self.files_folder = Path(files_folder)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        self.use_embeddings = use_embeddings and EMBEDDINGS_AVAILABLE and FAISS_AVAILABLE
        self.use_stemming = use_stemming
        
        if file_extensions:
            self.file_extensions = [ext if ext.startswith('.') else f'.{ext}' 
                                   for ext in file_extensions]
        else:
            self.file_extensions = list(ALL_SUPPORTED_EXTENSIONS)
        
        print(f"✓ Indexing files with extensions: {', '.join(self.file_extensions[:10])}...")
        
        # Initialize components
        self.file_extractor = UniversalFileExtractor()
        self.stemmer = TextStemmer(use_stemming=use_stemming)
        
        # Load domain keywords from YAML
        self.domain_keywords = DomainKeywordLoader(
            yaml_path=keywords_yaml,
            stemmer=self.stemmer
        )
        
        # Initialize DOMAIN-FOCUSED keyword extractor
        self.keyword_extractor = DomainFocusedKeywordExtractor(
            stemmer=self.stemmer,
            domain_keywords=self.domain_keywords
        )
        
        self.capability_mapper = CapabilityMapper(
            stemmer=self.stemmer,
            domain_keywords=self.domain_keywords
        )
        
        if self.use_embeddings:
            print("Loading embedding model...")
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
            self.embedding_dim = self.embedder.get_sentence_embedding_dimension()
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        else:
            print("⚡ Embeddings disabled - using pure keyword/BM25 search")
            self.embedder = None
            self.embedding_dim = None
            self.faiss_index = None
        
        self.metadata_store = []
        self.document_store = []
        self.document_store_stemmed = []
        self.keyword_doc_counts = Counter()
        self.capability_doc_counts = Counter()
        self.filetype_counts = Counter()
        self.total_chunks = 0
    
    def scan_files(self) -> List[Path]:
        """Scan folder for supported files"""
        print(f"Scanning folder: {self.files_folder}")
        
        files = []
        for ext in self.file_extensions:
            found = list(self.files_folder.glob(f"**/*{ext}"))
            files.extend(found)
            if found:
                print(f"  Found {len(found)} {ext} files")
        
        return files
    
    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            
            if len(chunk_text.strip()) < 50:
                continue
            
            chunk_id = hashlib.md5(chunk_text.encode()).hexdigest()
            
            chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "chunk_index": len(chunks),
                "start_word": i,
                "end_word": i + len(chunk_words),
                **metadata
            })
        
        return chunks
    
    def index_files(self, batch_size: int = 32):
        """Main indexing pipeline"""
        files = self.scan_files()
        print(f"\nTotal files to index: {len(files)}")
        
        if not files:
            print("No files found to index!")
            return None
        
        all_chunks = []
        
        print("\n=== Phase 1: Extracting Content ===")
        for idx, file_path in enumerate(files, 1):
            print(f"[{idx}/{len(files)}] Processing: {file_path.name}")
            
            content = self.file_extractor.extract(file_path)
            
            if not content.get("text") or len(content["text"].strip()) < 10:
                print(f"  ⚠ No text extracted, skipping")
                continue
            
            file_type = content.get("file_type", "unknown")
            self.filetype_counts[file_type] += 1
            
            # Extract DOMAIN-FOCUSED keywords
            keywords = self.keyword_extractor.extract(content["text"])
            print(f"  Domain Keywords: {', '.join([kw for kw, _ in keywords[:5]])}")
            
            # Map capabilities
            capabilities = self.capability_mapper.map_to_capabilities(keywords, content["text"])
            
            # Metadata
            doc_metadata = {
                "source_file": str(file_path.name),
                "source_path": str(file_path),
                "file_type": file_type,
                "file_size": file_path.stat().st_size,
                "language": content.get("language", ""),
                "line_count": content.get("line_count", 0),
                "has_tables": content.get("has_tables", False)
            }
            
            if file_type == "code" and "metadata" in content:
                doc_metadata.update(content["metadata"])
            
            chunks = self.chunk_text(content["text"], doc_metadata)
            
            for chunk in chunks:
                chunk["keywords"] = [kw for kw, score in keywords]
                chunk["keyword_scores"] = dict(keywords)
                chunk["capabilities"] = [cap for cap, score in capabilities]
                chunk["capability_scores"] = dict(capabilities)
                
                self.keyword_doc_counts.update(set(chunk["keywords"]))
                self.capability_doc_counts.update(set(chunk["capabilities"]))
            
            all_chunks.extend(chunks)
            print(f"  Created {len(chunks)} chunks")
        
        self.total_chunks = len(all_chunks)
        print(f"\nTotal chunks: {self.total_chunks}")
        
        print("\n=== Phase 2: Computing TF-IDF ===")
        idf_scores = {}
        for keyword, doc_count in self.keyword_doc_counts.items():
            if doc_count > 0:
                idf = math.log(self.total_chunks / doc_count)
                idf_scores[keyword] = idf
        
        for chunk in all_chunks:
            keyword_weights = {}
            for kw in chunk["keywords"]:
                if kw in idf_scores:
                    keyword_weights[kw] = idf_scores[kw]
            chunk["keyword_weights"] = keyword_weights
            
            sorted_kw = sorted(keyword_weights.items(), key=lambda x: x[1], reverse=True)
            chunk["top_keywords"] = [kw for kw, _ in sorted_kw[:5]]
        
        print("\n=== Phase 3: Storing Index ===")
        for chunk in all_chunks:
            top_capabilities = chunk["capabilities"][:3]
            metadata = {
                "source_file": chunk["source_file"],
                "file_type": chunk["file_type"],
                "language": chunk.get("language", ""),
                "chunk_index": chunk["chunk_index"],
                "has_tables": chunk.get("has_tables", False),
                "keywords": chunk["keywords"][:10],
                "top_keywords": chunk["top_keywords"],
                "capabilities": top_capabilities,
                "primary_capability": top_capabilities[0] if top_capabilities else "unknown",
            }
            self.metadata_store.append(metadata)
            self.document_store.append(chunk["text"])
            
            if self.use_stemming:
                stemmed_text = self.stemmer.stem_text(chunk["text"])
                self.document_store_stemmed.append(stemmed_text)
        
        print("\n=== Saving Index ===")
        
        with open(self.index_path / "metadata.pkl", 'wb') as f:
            pickle.dump(self.metadata_store, f)
        print(f"✓ Saved metadata")
        
        with open(self.index_path / "documents.pkl", 'wb') as f:
            pickle.dump(self.document_store, f)
        print(f"✓ Saved documents")
        
        if self.use_stemming:
            with open(self.index_path / "documents_stemmed.pkl", 'wb') as f:
                pickle.dump(self.document_store_stemmed, f)
            print(f"✓ Saved stemmed documents")
        
        stats = {
            "total_chunks": self.total_chunks,
            "total_keywords": len(idf_scores),
            "total_capabilities": len(self.capability_doc_counts),
            "total_files": len(files),
            "file_types": dict(self.filetype_counts),
            "idf_scores": idf_scores,
            "capability_distribution": dict(self.capability_doc_counts),
            "use_stemming": self.use_stemming,
            "file_extensions": self.file_extensions,
            "domain_keywords_loaded": len(self.domain_keywords.all_domain_keywords)
        }
        
        with open(self.index_path / "stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"✓ Saved statistics")
        
        return stats
    
    def get_statistics(self):
        """Display indexing statistics"""
        print("\n=== Indexing Statistics ===")
        print(f"Total chunks: {self.total_chunks}")
        print(f"Unique keywords: {len(self.keyword_doc_counts)}")
        print(f"Domain keywords loaded: {len(self.domain_keywords.all_domain_keywords)}")
        print(f"Unique capabilities: {len(self.capability_doc_counts)}")
        print(f"Stemming: {'Enabled' if self.use_stemming else 'Disabled'}")
        
        print("\n=== File Types ===")
        for file_type, count in self.filetype_counts.items():
            print(f"  {file_type:15s} - {count:4d} files")
        
        print("\n=== Top 10 Capabilities ===")
        for cap, count in self.capability_doc_counts.most_common(10):
            print(f"  {cap:40s} - {count:4d} chunks")
        
        print("\n=== Top 20 Domain Keywords ===")
        for kw, count in self.keyword_doc_counts.most_common(20):
            print(f"  {kw:30s} - {count:4d} chunks")


class UniversalFileSearcher:
    """Fast search across all indexed files"""
    
    def __init__(self, index_path: str = "./universal_index"):
        self.index_path = Path(index_path)
        
        with open(self.index_path / "stats.json", 'r') as f:
            self.stats = json.load(f)
        
        self.use_stemming = self.stats.get('use_stemming', False)
        self.stemmer = TextStemmer(use_stemming=self.use_stemming)
        
        with open(self.index_path / "metadata.pkl", 'rb') as f:
            self.metadata_store = pickle.load(f)
        
        with open(self.index_path / "documents.pkl", 'rb') as f:
            self.document_store = pickle.load(f)
        
        if self.use_stemming:
            stemmed_path = self.index_path / "documents_stemmed.pkl"
            if stemmed_path.exists():
                with open(stemmed_path, 'rb') as f:
                    self.document_store_stemmed = pickle.load(f)
            else:
                self.document_store_stemmed = [
                    self.stemmer.stem_text(doc) for doc in self.document_store
                ]
        else:
            self.document_store_stemmed = self.document_store
        
        tokenized_docs = [doc.split() for doc in self.document_store_stemmed]
        self.bm25 = BM25Okapi(tokenized_docs)
        print(f"✓ BM25 index built")
        
        self.query_processor = ImprovedQueryProcessor(stemmer=self.stemmer)
    
    def search(
        self,
        query: str,
        top_k: int = 20,
        file_type_filter: Optional[str] = None,
        verbose: bool = False
    ) -> List[Dict[str, Any]]:
        """Search indexed files"""
        if verbose:
            print(f"\n{'='*80}")
            print(f"Query: {query}")
            print(f"{'='*80}")
        
        query_analysis = self.query_processor.extract_core_terms(query)
        search_query = query_analysis['stemmed_query'] if self.use_stemming else query_analysis['original_query']
        
        if verbose:
            print(f"Processed query: {search_query}")
        
        query_tokens = search_query.split()
        scores = self.bm25.get_scores(query_tokens)
        
        top_indices = np.argsort(scores)[::-1][:top_k * 5]
        
        results = []
        for idx in top_indices:
            if scores[idx] < 0.01:
                break
            
            metadata = self.metadata_store[idx]
            
            if file_type_filter and metadata.get('file_type') != file_type_filter:
                continue
            
            results.append({
                "text": self.document_store[idx],
                "source_file": metadata['source_file'],
                "file_type": metadata['file_type'],
                "language": metadata.get('language', ''),
                "chunk_index": metadata['chunk_index'],
                "score": scores[idx],
                "capabilities": metadata.get('capabilities', []),
                "keywords": metadata.get('keywords', [])
            })
            
            if len(results) >= top_k:
                break
        
        return results


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Universal File Indexer with Domain Focus"
    )
    parser.add_argument("--folder", required=True, help="Path to files folder")
    parser.add_argument("--index-path", default="./universal_index", help="Index storage path")
    parser.add_argument("--keywords-yaml", help="Path to keywords YAML file")
    parser.add_argument("--action", choices=["index", "search", "stats"], default="index")
    parser.add_argument("--query", help="Search query")
    parser.add_argument("--extensions", nargs='+', help="File extensions to index")
    parser.add_argument("--file-type", help="Filter by file type")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results")
    parser.add_argument("--disable-stemming", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Universal File Indexer - DOMAIN FOCUSED")
    print("=" * 70)
    print()
    
    if args.action == "index":
        indexer = UniversalFileIndexer(
            files_folder=args.folder,
            index_path=args.index_path,
            keywords_yaml=args.keywords_yaml,
            file_extensions=args.extensions,
            use_stemming=not args.disable_stemming,
            use_embeddings=False
        )
        
        stats = indexer.index_files()
        indexer.get_statistics()
        
    elif args.action == "search":
        if not args.query:
            print("Error: --query required for search")
            return
        
        searcher = UniversalFileSearcher(index_path=args.index_path)
        
        results = searcher.search(
            args.query, 
            top_k=args.top_k,
            file_type_filter=args.file_type,
            verbose=args.verbose
        )
        
        print(f"\n{'='*80}")
        print(f"Found {len(results)} results")
        print(f"{'='*80}\n")
        
        for i, result in enumerate(results, 1):
            print(f"[{i}] Score: {result['score']:.3f}")
            print(f"File: {result['source_file']} ({result['file_type']})")
            if result['language']:
                print(f"Language: {result['language']}")
            print(f"Keywords: {', '.join(result['keywords'][:5])}")
            print(f"Text: {result['text'][:200]}...")
            print("-" * 80)
    
    elif args.action == "stats":
        searcher = UniversalFileSearcher(index_path=args.index_path)
        print("\n=== File Types ===")
        for file_type, count in searcher.stats.get('file_types', {}).items():
            print(f"  {file_type:15s} - {count:4d} files")


if __name__ == "__main__":
    main()
