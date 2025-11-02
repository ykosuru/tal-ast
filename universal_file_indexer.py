"""
Universal File Indexer with Stemming - Supports PDFs, Code, and Text Files
Handles: .pdf, .c, .py, .java, .tal, .cbl, .txt, .md, .json, .xml, etc.
"""

import os
import json
import math
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional, Tuple, Set
import hashlib
import re

# PDF processing (optional)
try:
    import PyPDF2
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("Warning: PDF libraries not available, PDF indexing disabled")
    print("Install with: pip install PyPDF2 pdfplumber")

# OCR for images (optional)
try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

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
        '.py', '.c', '.cpp', '.h', '.hpp', '.cc', '.cxx',  # Python, C/C++
        '.java', '.scala', '.kt', '.groovy',  # JVM languages
        '.js', '.ts', '.jsx', '.tsx',  # JavaScript/TypeScript
        '.go', '.rs', '.swift',  # Go, Rust, Swift
        '.tal', '.cbl', '.cobol', '.cob',  # TAL, COBOL
        '.sql', '.pl', '.pm',  # SQL, Perl
        '.rb', '.php', '.sh', '.bash',  # Ruby, PHP, Shell
        '.cs', '.vb', '.fs'  # .NET languages
    ],
    'text': [
        '.txt', '.md', '.rst', '.log', '.text',
        '.doc', '.rtf'  # Note: .doc needs special handling
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
    """
    Centralized stemming utility using Porter Stemmer
    Handles all word variations: payment→payments, process→processing, etc.
    """
    
    def __init__(self, use_stemming: bool = True):
        self.use_stemming = use_stemming and STEMMER_AVAILABLE
        
        if self.use_stemming:
            self.stemmer = PorterStemmer()
            print("✓ Stemming enabled (Porter Stemmer)")
        else:
            self.stemmer = None
            print("⚠ Stemming disabled")
        
        # Cache for performance
        self._stem_cache = {}
    
    def stem(self, word: str) -> str:
        """Stem a single word with caching"""
        if not self.use_stemming or not word:
            return word.lower()
        
        word_lower = word.lower()
        
        # Check cache first
        if word_lower in self._stem_cache:
            return self._stem_cache[word_lower]
        
        # Stem and cache
        stemmed = self.stemmer.stem(word_lower)
        self._stem_cache[word_lower] = stemmed
        
        return stemmed
    
    def stem_tokens(self, tokens: List[str]) -> List[str]:
        """Stem a list of tokens"""
        return [self.stem(token) for token in tokens]
    
    def stem_text(self, text: str) -> str:
        """Stem all words in text, preserving structure"""
        words = re.findall(r'\b[\w-]+\b', text)
        stemmed_words = []
        
        for word in words:
            stemmed_words.append(self.stem(word))
        
        return ' '.join(stemmed_words)


class BusinessCapabilityTaxonomy:
    """Wire Processing Business Capabilities taxonomy"""
    
    CAPABILITIES = {
        "Core Payment & Network": [
            "clearing networks", "fed", "chips", "swift", "clearing house",
            "network gateways", "network connectivity", "network acknowledgments",
            "network admin", "network certification", "lterm", "ack", "nak"
        ],
        
        "Payment Processing & Execution": [
            "payment initiation", "payment routing", "payment execution",
            "preadvising", "cover payments", "liquidity management",
            "debit confirmation", "credit confirmation", "outbound payment",
            "hard posting", "cutoffs", "workflow scheduling", "orchestration",
            "split advising", "intraday liquidity", "book transfer",
            "eod processing", "fee determination", "payment agreements",
            "payment returns", "payment prioritization", "warehousing",
            "exceptions processing"
        ],
        
        "Instruction & Validation": [
            "instruction management", "straight thru processing", "stp",
            "pay thru validation", "method of payment", "payment enrichment",
            "payment repair", "payment verify", "sod releasing",
            "auto repair", "date validation", "time validation",
            "account validation", "amount validation", "currency validation",
            "standing orders", "repetitive orders", "party association"
        ],
        
        "Controls & Risk Management": [
            "controls services", "anomalies detection", "ca&d",
            "sanctions screening", "fircosoft", "ofac", "funds control",
            "fraud checking", "debit authority", "duplicate checking",
            "debit blocks", "credit blocks", "memo posting",
            "ceo fraud", "cfm", "anti-money laundering", "aml", "newton",
            "risk control system", "rcs"
        ],
        
        "Data & Reporting": [
            "data management", "report distribution", "financial crimes reporting",
            "risk analysis reporting", "historical data", "payment reconciliation",
            "general ledger", "gl feeds", "account activity reporting",
            "adhoc reporting", "scheduled reporting", "event notification",
            "alert", "fee charges", "analysis charges", "product capabilities",
            "data service integration", "ai ml modeling", "report archiving",
            "axcis", "client billing", "statements", "client reconciliation",
            "transaction info", "balance info", "electronic window",
            "intelligence analytics", "trend analysis", "ecosystem analytics"
        ],
        
        "Service Integration": [
            "data masking", "obfuscation", "transaction replay",
            "data encryption", "decryption", "channel acknowledgments",
            "service api", "endpoint publishing", "duplicate detection",
            "api invocation", "service invocation", "queues", "topics",
            "format transformation", "id generation", "schema validation"
        ],
        
        "User Experience": [
            "business activity monitoring", "alert dispositioning",
            "queue drilldown", "telemetry", "ui maintenance",
            "user entitlements", "payment data inquiry", "trend analysis",
            "stp analysis", "risk event information", "smart alerting"
        ],
        
        "Channel & Integration": [
            "client authentication", "client preference", "channel connectivity",
            "canonical management", "shared services", "global services",
            "investigations", "pega", "intellitracs", "bank reconciliation",
            "1bkr", "intraday posting", "tms", "middleware", "gabs",
            "fx services", "wxchg", "opics", "revenue profit", "rpm",
            "enterprise fax", "gfx", "online wires", "olw",
            "treasury workstation", "pstw", "ceo api wires",
            "secure fax", "voice response", "vru", "position management",
            "loan q", "liq", "approval queue", "onq", "cyberpay",
            "1cyb", "1trx", "js-gds"
        ],
        
        "ISO Standards & Formats": [
            "iso20022", "iso 20022", "pacs.008", "pacs.009", "pacs.002",
            "pain.001", "camt.053", "mt103", "mt202", "mt199",
            "fedwire", "chips format", "swift format", "xml", "json"
        ],
        
        "Validation & Screening": [
            "bic validation", "bic code", "iban validation", "iban",
            "party validation", "account validation", "sanctions check",
            "watchlist screening", "name screening", "address validation",
            "routing validation", "aba", "sort code"
        ],
        
        "Transaction Processing": [
            "wire transfer", "wire payment", "domestic wire", "international wire",
            "cross-border payment", "same-day payment", "rtgs", "ach",
            "clearing", "settlement", "netting", "gross settlement"
        ]
    }
    
    # Synonyms and variations
    SYNONYMS = {
        "ofac": ["ofac screening", "sanctions", "ofac_screen_party"],
        "ace": ["automated clearing", "ace repair", "ace code"],
        "stp": ["straight through", "straight thru", "straight-through"],
        "aml": ["anti money laundering", "money laundering"],
        "bic": ["bank identifier", "swift code", "bic code"],
        "fed": ["federal reserve", "fedwire", "federal wire"],
        "chips": ["clearing house interbank", "chips network"],
        "iso20022": ["iso 20022", "iso-20022", "pacs", "pain", "camt"]
    }
    
    @classmethod
    def get_all_keywords(cls) -> List[str]:
        """Get all keywords across all capabilities"""
        all_keywords = []
        for keywords in cls.CAPABILITIES.values():
            all_keywords.extend(keywords)
        return list(set(all_keywords))
    
    @classmethod
    def expand_with_synonyms(cls, keyword: str) -> List[str]:
        """Expand keyword with synonyms"""
        expanded = [keyword]
        for main_term, synonyms in cls.SYNONYMS.items():
            if keyword.lower() in [main_term] + [s.lower() for s in synonyms]:
                expanded.extend([main_term] + synonyms)
        return list(set(expanded))
    
    @classmethod
    def get_related_capabilities(cls, capability: str) -> List[str]:
        """Get keywords related to a specific capability"""
        return cls.CAPABILITIES.get(capability, [])


class UniversalFileExtractor:
    """
    Extract text from various file types:
    - PDFs (.pdf)
    - Code files (.py, .c, .java, .tal, etc.)
    - Text files (.txt, .md, .log)
    - Config files (.json, .xml, .yaml)
    """
    
    def __init__(self, enable_ocr: bool = False):
        self.enable_ocr = enable_ocr and OCR_AVAILABLE
        self.pdf_available = PDF_AVAILABLE
    
    def get_file_type(self, file_path: Path) -> str:
        """Determine file type from extension"""
        ext = file_path.suffix.lower()
        
        for file_type, extensions in SUPPORTED_EXTENSIONS.items():
            if ext in extensions:
                return file_type
        
        return 'unknown'
    
    def extract(self, file_path: Path) -> Dict[str, Any]:
        """
        Universal file extraction dispatcher
        
        Returns:
            {
                'text': str,
                'file_type': str,
                'language': str (for code),
                'line_count': int,
                'has_tables': bool (for PDFs),
                'metadata': dict
            }
        """
        file_type = self.get_file_type(file_path)
        
        if file_type == 'pdf':
            return self._extract_pdf(file_path)
        elif file_type == 'code':
            return self._extract_code(file_path)
        elif file_type in ['text', 'config', 'markup']:
            return self._extract_text(file_path)
        else:
            # Try as text file anyway
            return self._extract_text(file_path)
    
    def _extract_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract text from PDF files"""
        if not self.pdf_available:
            return {
                'text': '',
                'file_type': 'pdf',
                'error': 'PDF libraries not available',
                'line_count': 0,
                'has_tables': False
            }
        
        content = {
            "text": "",
            "file_type": "pdf",
            "has_tables": False,
            "tables": [],
            "images": [],
            "line_count": 0
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
                            content["tables"].append({
                                "page": page_num + 1,
                                "text": table_text
                            })
                            content["text"] += f"\n[TABLE]\n{table_text}\n[/TABLE]\n"
        
        except Exception as e:
            print(f"Error extracting from {pdf_path}: {e}")
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        content["text"] += page.extract_text() + "\n"
            except Exception as e2:
                print(f"Fallback also failed: {e2}")
                content["error"] = str(e2)
        
        content["line_count"] = content["text"].count('\n')
        return content
    
    def _extract_code(self, code_path: Path) -> Dict[str, Any]:
        """
        Extract text from code files with syntax awareness
        Preserves structure: function names, class names, comments
        """
        content = {
            "text": "",
            "file_type": "code",
            "language": code_path.suffix[1:],  # Remove dot
            "line_count": 0,
            "encoding": "utf-8"
        }
        
        # Try different encodings
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
        
        # Extract code-specific metadata
        content["metadata"] = self._extract_code_metadata(content["text"], content["language"])
        
        return content
    
    def _extract_text(self, text_path: Path) -> Dict[str, Any]:
        """Extract text from plain text files"""
        content = {
            "text": "",
            "file_type": "text",
            "line_count": 0,
            "encoding": "utf-8"
        }
        
        # Try different encodings
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
        """
        Extract metadata from code:
        - Function names
        - Class names
        - Imports
        - Comments
        """
        metadata = {
            "functions": [],
            "classes": [],
            "imports": [],
            "has_main": False
        }
        
        # Language-specific patterns
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
            'cpp': {
                'function': r'\w+\s+(\w+)\s*\([^)]*\)\s*\{',
                'class': r'class\s+(\w+)',
                'include': r'#include\s+[<"]([^>"]+)[>"]'
            }
        }
        
        lang_patterns = patterns.get(language, patterns.get('python', {}))
        
        # Extract functions
        if 'function' in lang_patterns:
            functions = re.findall(lang_patterns['function'], code_text)
            metadata['functions'] = list(set(functions))[:50]  # Limit to 50
            
            # Check for main
            if 'main' in functions:
                metadata['has_main'] = True
        
        # Extract classes
        if 'class' in lang_patterns:
            classes = re.findall(lang_patterns['class'], code_text)
            metadata['classes'] = list(set(classes))[:50]
        
        # Extract imports/includes
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


class ImprovedQueryProcessor:
    """
    Pre-process queries to extract core terms and handle natural language
    NOW WITH STEMMING: Ensures "payments" matches "payment"
    """
    
    def __init__(self, stemmer: Optional[TextStemmer] = None):
        self.stemmer = stemmer
        
        # Comprehensive stopwords
        self.stopwords = {
            # Question words
            'how', 'what', 'when', 'where', 'why', 'who', 'which', 'whose',
            
            # Common verbs
            'do', 'does', 'did', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'will', 'would', 'should', 'could', 'can',
            'may', 'might', 'must', 'shall',
            
            # Implementation/action words
            'implement', 'create', 'build', 'make', 'develop', 'setup', 'configure',
            'explain', 'describe', 'show', 'tell', 'need', 'want', 'help',
            
            # Articles and prepositions
            'a', 'an', 'the', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'from', 'by', 'as', 'into', 'through', 'during', 'before', 'after',
            
            # Pronouns
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her',
            'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their',
            
            # Conjunctions
            'and', 'or', 'but', 'if', 'then', 'than', 'so', 'because',
            
            # Other common words
            'this', 'that', 'these', 'those', 'there', 'here',
        }
        
        # Stem stopwords if stemmer available
        if self.stemmer and self.stemmer.use_stemming:
            self.stopwords = {self.stemmer.stem(sw) for sw in self.stopwords}
        
        # Words that suggest action/question but should be removed for search
        self.action_indicators = {
            'how', 'implement', 'create', 'build', 'setup', 'configure',
            'explain', 'describe', 'show', 'tell', 'need', 'want', 'help'
        }
    
    def extract_core_terms(self, query: str) -> Dict[str, Any]:
        """
        Extract core searchable terms from query with STEMMING
        """
        original = query.strip()
        query_lower = query.lower()
        
        # Detect if it's a question
        is_question = any(query_lower.startswith(q) for q in 
                         ['how', 'what', 'when', 'where', 'why', 'who'])
        
        # Detect action type
        action_type = None
        for action in self.action_indicators:
            if action in query_lower:
                action_type = action
                break
        
        # Tokenize - keep hyphenated terms together
        tokens = re.findall(r'\b[\w-]+\b', query_lower)
        
        # Extract core terms (non-stopwords)
        core_terms = []
        for token in tokens:
            # Skip single letters and numbers
            if len(token) <= 1 or token.isdigit():
                continue
            
            # Stem token if available
            if self.stemmer and self.stemmer.use_stemming:
                stemmed_token = self.stemmer.stem(token)
            else:
                stemmed_token = token
            
            # Skip stopwords (check against stemmed version)
            if stemmed_token in self.stopwords:
                continue
            
            core_terms.append(stemmed_token)
        
        # Reconstruct cleaned query (stemmed)
        cleaned_query = ' '.join(core_terms)
        
        # Also create un-stemmed version for display
        unstemmed_core = []
        for token in tokens:
            if len(token) > 1 and not token.isdigit() and token not in self.stopwords:
                unstemmed_core.append(token)
        
        return {
            'core_terms': core_terms,  # Stemmed
            'original_query': original,
            'cleaned_query': ' '.join(unstemmed_core),  # Un-stemmed for display
            'stemmed_query': cleaned_query,  # Stemmed for search
            'is_question': is_question,
            'action_type': action_type,
            'stopwords_removed': len(tokens) - len(core_terms)
        }
    
    def extract_phrases(self, query: str) -> List[str]:
        """Extract important multi-word phrases"""
        phrases = []
        
        # Extract quoted phrases
        quoted = re.findall(r'"([^"]+)"', query)
        phrases.extend(quoted)
        
        # Extract capitalized phrases (likely important)
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', query)
        phrases.extend(capitalized)
        
        # Extract technical patterns
        technical_patterns = [
            r'pacs\.\d+',
            r'ISO[\s-]?\d+',
            r'MT\d{3}',
            r'[A-Z]{3,5}(?:\s+[A-Z]{3,5})*'  # Acronym sequences
        ]
        
        for pattern in technical_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            phrases.extend(matches)
        
        return list(set(phrases))


class FastKeywordExtractor:
    """
    Extract keywords using regex patterns - NOW WITH STEMMING
    Properly handles: payment/payments, process/processing, validate/validation
    """
    
    def __init__(self, stemmer: Optional[TextStemmer] = None):
        self.stemmer = stemmer
        
        # Domain-specific patterns
        self.patterns = [
            (r'ISO[\s-]?\d+', 'iso_standard'),
            (r'pacs\.\d+(?:\.\d+)*', 'payment_message'),
            (r'pain\.\d+(?:\.\d+)*', 'payment_message'),
            (r'camt\.\d+(?:\.\d+)*', 'payment_message'),
            (r'MT\d{3}', 'swift_message'),
            (r'ACE[\s-]?\d+', 'ace_code'),
            (r'[A-Z]{6}[A-Z0-9]{2}(?:[A-Z0-9]{3})?', 'bic_code'),
            (r'\$[A-Z_]+', 'system_function'),
            (r'[A-Z]{2}\d{2}[A-Z0-9]+', 'iban'),
        ]
        
        # Important domain-specific terms
        self.important_terms = {
            # Core payment terms
            'credit', 'debit', 'party', 'determination', 'validation',
            'payment', 'wire', 'transfer', 'transaction', 'settlement',
            'clearing', 'posting', 'execution', 'routing', 'initiation',
            
            # Time-sensitive processing
            'cutoff', 'cutoffs', 'deadline', 'eod', 'sod', 'intraday',
            'real-time', 'batch', 'schedule', 'warehousing',
            
            # Validation & screening
            'screening', 'sanctions', 'ofac', 'fircosoft', 'compliance',
            'verification', 'confirmation', 'approval', 'authorization',
            
            # Networks & standards
            'fedwire', 'chips', 'swift', 'ach', 'sepa', 'target',
            'iso20022', 'pacs', 'pain', 'camt',
            
            # Entities & accounts
            'creditor', 'debtor', 'beneficiary', 'originator', 'intermediary',
            'nostro', 'vostro', 'account', 'customer', 'client',
            
            # Business capabilities
            'repair', 'enrichment', 'orchestration', 'workflow', 'processing',
            'reconciliation', 'exception', 'investigation', 'monitoring',
            
            # Technical
            'format', 'message', 'instruction', 'queue', 'endpoint',
            'service', 'api', 'integration', 'interface', 'channel',
            
            # Code-specific
            'function', 'class', 'method', 'variable', 'parameter',
            'return', 'loop', 'condition', 'exception', 'error'
        }
        
        # Stem important terms if stemmer available
        if self.stemmer and self.stemmer.use_stemming:
            self.important_terms_stemmed = {self.stemmer.stem(term) for term in self.important_terms}
        else:
            self.important_terms_stemmed = self.important_terms
        
        # Common stopwords
        self.stopwords = {
            'process', 'system', 'data', 'information',
            'general', 'related', 'based', 'using', 'including', 'provides',
            'allows', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
            'to', 'for', 'of', 'with', 'from', 'by', 'as', 'is', 'was', 'are',
            'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'should', 'could', 'may', 'might', 'must', 'can',
            'this', 'that', 'these', 'those', 'it', 'its', 'they', 'their', 'them'
        }
        
        # Stem stopwords if stemmer available
        if self.stemmer and self.stemmer.use_stemming:
            self.stopwords_stemmed = {self.stemmer.stem(sw) for sw in self.stopwords}
        else:
            self.stopwords_stemmed = self.stopwords
    
    def extract(self, text: str, max_keywords: int = 20) -> List[Tuple[str, float]]:
        """
        Extract keywords with confidence scores using regex + STEMMING
        Returns: List of (keyword, confidence) tuples
        """
        if not text or len(text.strip()) < 10:
            return []
        
        keyword_scores = defaultdict(float)
        text_lower = text.lower()
        
        # Tokenize text and stem
        words = re.findall(r'\b[\w-]+\b', text_lower)
        if self.stemmer and self.stemmer.use_stemming:
            stemmed_words = [self.stemmer.stem(w) for w in words]
        else:
            stemmed_words = words
        
        # 1. Domain-specific patterns (highest confidence)
        for pattern, pattern_type in self.patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                keyword_scores[match.lower()] += 3.0
        
        # 2. Important domain terms (WITH STEMMING)
        for term_stem in self.important_terms_stemmed:
            count = stemmed_words.count(term_stem)
            if count > 0:
                keyword_scores[term_stem] += 3.0 * math.log1p(count)
        
        # 3. Capitalized words
        capitalized_words = re.findall(r'\b[A-Z][a-z]{2,}\b', text)
        for word in capitalized_words:
            if self.stemmer and self.stemmer.use_stemming:
                word_stem = self.stemmer.stem(word.lower())
            else:
                word_stem = word.lower()
            
            if (word_stem not in self.stopwords_stemmed and 
                word_stem not in self.important_terms_stemmed):
                keyword_scores[word_stem] += 1.5
        
        # 4. Business capability keywords
        capability_keywords = BusinessCapabilityTaxonomy.get_all_keywords()
        for kw in capability_keywords:
            if self.stemmer and self.stemmer.use_stemming:
                kw_stem = self.stemmer.stem_text(kw)
                kw_tokens = kw_stem.split()
                matches = sum(1 for i in range(len(stemmed_words) - len(kw_tokens) + 1)
                             if stemmed_words[i:i+len(kw_tokens)] == kw_tokens)
                if matches > 0:
                    keyword_scores[kw_stem] += 2.0 * math.log1p(matches)
            else:
                if kw in text_lower:
                    count = text_lower.count(kw)
                    keyword_scores[kw] += 2.0 * math.log1p(count)
        
        # Filter and normalize
        filtered = []
        for kw, score in keyword_scores.items():
            if (len(kw) > 2 and 
                kw not in self.stopwords_stemmed and
                not all(c.isdigit() for c in kw)):
                filtered.append((kw, score))
        
        # Sort by score and return top N
        filtered.sort(key=lambda x: x[1], reverse=True)
        return filtered[:max_keywords]


class CapabilityMapper:
    """Map documents to business capabilities - NOW WITH STEMMING"""
    
    def __init__(self, stemmer: Optional[TextStemmer] = None):
        self.taxonomy = BusinessCapabilityTaxonomy()
        self.stemmer = stemmer
        self.capability_keywords = {}
        
        # Build keyword to capability mapping (with stemming)
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
        
        # Stem text if available
        if self.stemmer and self.stemmer.use_stemming:
            text_stemmed = self.stemmer.stem_text(text_lower)
        else:
            text_stemmed = text_lower
        
        # Score each capability based on keyword matches
        for keyword, kw_score in keywords:
            if keyword in self.capability_keywords:
                for capability in self.capability_keywords[keyword]:
                    capability_scores[capability] += kw_score
        
        # Direct capability keyword matching in text
        for capability, keywords_list in self.taxonomy.CAPABILITIES.items():
            for kw in keywords_list:
                if self.stemmer and self.stemmer.use_stemming:
                    kw_stem = self.stemmer.stem_text(kw)
                    count = text_stemmed.count(kw_stem)
                else:
                    count = text_lower.count(kw)
                
                if count > 0:
                    capability_scores[capability] += math.log1p(count) * 1.5
        
        # Normalize scores
        if capability_scores:
            max_score = max(capability_scores.values())
            capability_scores = {
                cap: score / max_score 
                for cap, score in capability_scores.items()
            }
        
        # Sort by score
        sorted_capabilities = sorted(
            capability_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return sorted_capabilities


class UniversalFileIndexer:
    """Universal file indexer - supports PDFs, code, text files"""
    
    def __init__(
        self,
        files_folder: str,
        index_path: str = "./universal_index",
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
        
        # Determine which extensions to index
        if file_extensions:
            self.file_extensions = [ext if ext.startswith('.') else f'.{ext}' 
                                   for ext in file_extensions]
        else:
            # Default: all supported extensions
            self.file_extensions = list(ALL_SUPPORTED_EXTENSIONS)
        
        print(f"✓ Indexing files with extensions: {', '.join(self.file_extensions[:10])}...")
        
        # Initialize components
        self.file_extractor = UniversalFileExtractor()
        self.stemmer = TextStemmer(use_stemming=use_stemming)
        self.keyword_extractor = FastKeywordExtractor(stemmer=self.stemmer)
        self.capability_mapper = CapabilityMapper(stemmer=self.stemmer)
        
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
            
            # Extract content
            content = self.file_extractor.extract(file_path)
            
            if not content.get("text") or len(content["text"].strip()) < 10:
                print(f"  ⚠ No text extracted, skipping")
                continue
            
            file_type = content.get("file_type", "unknown")
            self.filetype_counts[file_type] += 1
            
            # Extract keywords
            keywords = self.keyword_extractor.extract(content["text"])
            print(f"  Keywords: {', '.join([kw for kw, _ in keywords[:5]])}")
            
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
            
            # Add code-specific metadata
            if file_type == "code" and "metadata" in content:
                doc_metadata.update(content["metadata"])
            
            # Chunk the content
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
        
        # Continue with TF-IDF, embeddings, etc. (same as before)
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
        
        # Save to disk
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
            "file_extensions": self.file_extensions
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
        print(f"Unique capabilities: {len(self.capability_doc_counts)}")
        print(f"Stemming: {'Enabled' if self.use_stemming else 'Disabled'}")
        
        print("\n=== File Types ===")
        for file_type, count in self.filetype_counts.items():
            print(f"  {file_type:15s} - {count:4d} files")
        
        print("\n=== Top 10 Capabilities ===")
        for cap, count in self.capability_doc_counts.most_common(10):
            print(f"  {cap:40s} - {count:4d} chunks")


class UniversalFileSearcher:
    """Fast search across all indexed files"""
    
    def __init__(
        self, 
        index_path: str = "./universal_index",
        enable_query_expansion: bool = False,
        expansion_level: str = "medium"
    ):
        self.index_path = Path(index_path)
        
        # Load stats
        with open(self.index_path / "stats.json", 'r') as f:
            self.stats = json.load(f)
        
        self.use_stemming = self.stats.get('use_stemming', False)
        self.stemmer = TextStemmer(use_stemming=self.use_stemming)
        
        # Load metadata and documents
        with open(self.index_path / "metadata.pkl", 'rb') as f:
            self.metadata_store = pickle.load(f)
        
        with open(self.index_path / "documents.pkl", 'rb') as f:
            self.document_store = pickle.load(f)
        
        # Load stemmed documents if available
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
        
        # Build BM25 index
        tokenized_docs = [doc.split() for doc in self.document_store_stemmed]
        self.bm25 = BM25Okapi(tokenized_docs)
        print(f"✓ BM25 index built with {'stemmed' if self.use_stemming else 'original'} documents")
        
        # Query processing
        self.query_processor = ImprovedQueryProcessor(stemmer=self.stemmer)
        self.keyword_extractor = FastKeywordExtractor(stemmer=self.stemmer)
        self.capability_mapper = CapabilityMapper(stemmer=self.stemmer)
    
    def search(
        self,
        query: str,
        top_k: int = 20,
        file_type_filter: Optional[str] = None,
        verbose: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Search indexed files
        
        Args:
            query: Search query
            top_k: Number of results
            file_type_filter: Filter by file type ('code', 'pdf', 'text', etc.)
            verbose: Show detailed process
        """
        if verbose:
            print(f"\n{'='*80}")
            print(f"Query: {query}")
            print(f"{'='*80}")
        
        # Process query
        query_analysis = self.query_processor.extract_core_terms(query)
        search_query = query_analysis['stemmed_query'] if self.use_stemming else query_analysis['cleaned_query']
        
        if verbose:
            print(f"Processed query: {search_query}")
        
        # Search with BM25
        query_tokens = search_query.split()
        scores = self.bm25.get_scores(query_tokens)
        
        top_indices = np.argsort(scores)[::-1][:top_k * 5]  # Get extra for filtering
        
        # Format results
        results = []
        for idx in top_indices:
            if scores[idx] < 0.01:
                break
            
            metadata = self.metadata_store[idx]
            
            # Filter by file type if specified
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
    """Example usage - UNIVERSAL FILE INDEXER"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Universal File Indexer - PDFs, Code, Text files"
    )
    parser.add_argument("--folder", required=True, help="Path to files folder")
    parser.add_argument("--index-path", default="./universal_index", help="Index storage path")
    parser.add_argument("--action", choices=["index", "search", "stats"], default="index")
    parser.add_argument("--query", help="Search query")
    parser.add_argument("--extensions", nargs='+', help="File extensions to index (e.g., .py .c .java)")
    parser.add_argument("--file-type", help="Filter by file type (code, pdf, text)")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results")
    parser.add_argument("--disable-stemming", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Universal File Indexer")
    print("Supports: .pdf, .py, .c, .java, .tal, .txt, .md, and more")
    print("=" * 70)
    print()
    
    if args.action == "index":
        indexer = UniversalFileIndexer(
            files_folder=args.folder,
            index_path=args.index_path,
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

print("\n" + "=" * 70)
print("✓ Universal File Indexer")
print("  • Supports: .pdf, .py, .c, .java, .tal, .txt, .md, .json, .xml")
print("  • Automatic file type detection")
print("  • Code-aware extraction (functions, classes, imports)")
print("  • Stemming support for better matching")
print("=" * 70)
