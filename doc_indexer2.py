"""
Universal File Indexer v2.0 - Enhanced Edition
Supports: PDFs, Code (Python, Java, C, TAL, COBOL, SQL), Text, Config files
Features: BM25 search, stemming, business capability mapping, code metadata extraction
         MULTI-TERM SEARCH, PHRASE MATCHING, MULTI-STAGE RANKING, RELEVANCE BOOSTING

Enhancements:
- Multi-word term detection (e.g., "cutoff times", "cover payments")
- Position index for proximity and phrase search
- Multi-stage ranking with 12+ relevance signals
- Exact phrase matching with high boost
- Keyword/capability/title matching with custom weights
- Query explanations for debugging
"""

import os
import json
import math
import re
import pickle
import hashlib
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional, Tuple
from rank_bm25 import BM25Okapi

# Optional: PDF support
try:
    import pdfplumber
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("⚠ PDF libraries not available. Install: pip install pdfplumber PyPDF2")

# Optional: Stemming
try:
    from nltk.stem import PorterStemmer
    STEMMER_AVAILABLE = True
except ImportError:
    STEMMER_AVAILABLE = False
    print("⚠ NLTK not available. Install: pip install nltk")


# ================================================================================
# File Type Definitions
# ================================================================================

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
        '.cs', '.vb', '.fs', '.TXT'
    ],
    'text': ['.txt', '.md', '.rst', '.log', '.text'],
    'config': ['.json', '.xml', '.yaml', '.yml', '.toml', '.ini', '.cfg'],
    'markup': ['.html', '.htm', '.xhtml', '.css', '.scss']
}

ALL_EXTENSIONS = set()
for extensions in SUPPORTED_EXTENSIONS.values():
    ALL_EXTENSIONS.update(extensions)


# ================================================================================
# Shared NLP Components
# ================================================================================

STOPWORDS = [
    'how', 'what', 'when', 'where', 'why', 'who', 'which',
    'do', 'does', 'did', 'is', 'are', 'was', 'were', 'be', 'been',
    'have', 'has', 'had', 'will', 'would', 'should', 'could', 'can',
    'implement', 'create', 'build', 'make', 'develop', 'setup',
    'a', 'an', 'the', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
    'from', 'by', 'as', 'into', 'through', 'during', 'before', 'after',
    'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her',
    'and', 'or', 'but', 'if', 'then', 'than', 'so', 'because',
    'this', 'that', 'these', 'those', 'there', 'here',
    'process', 'system', 'data', 'information', 'file', 'code'
]


class TextStemmer:
    """Porter Stemmer with caching for performance"""

    def __init__(self, use_stemming: bool = True):
        self.use_stemming = use_stemming and STEMMER_AVAILABLE

        if self.use_stemming:
            self.stemmer = PorterStemmer()
        else:
            self.stemmer = None

        self._cache = {}

    def stem(self, word: str) -> str:
        """Stem a single word"""
        if not self.use_stemming or not word:
            return word.lower()

        word_lower = word.lower()
        if word_lower in self._cache:
            return self._cache[word_lower]

        stemmed = self.stemmer.stem(word_lower)
        self._cache[word_lower] = stemmed
        return stemmed

    def stem_tokens(self, tokens: List[str]) -> List[str]:
        """Stem list of tokens"""
        return [self.stem(t) for t in tokens]

    def stem_text(self, text: str) -> str:
        """Stem all words in text"""
        words = re.findall(r'\b[\w-]+\b', text)
        return ' '.join([self.stem(w) for w in words])


class DomainQueryExpander:
    """Expand queries with domain-specific synonyms"""

    # EPX business capability-oriented synonyms (functional intent)
    SYNONYMS = {
        # Core payment concepts
        'payment': [
            'transaction', 'transfer', 'wire', 'remittance',
            'book transfer', 'funds movement', 'pay instruction', 'outbound payment',
            'credit transfer', 'customer transfer'
        ],
        'drawdown': ['advance', 'disbursement', 'funding', 'loan', 'withdraw'],
        'stp': [
            'straight thru processing', 'straight-through processing', 'straight through processing',
            'stp analysis', 'stp', 'ace', 'pelican'
        ],
        'validation': [
            'verification', 'checking', 'screening', 'confirmation',
            'date validation', 'time validation', 'amount validation', 'currency validation',
            'debit account validation', 'credit account validation', 'schema validation',
            'format validation', 'technical duplicate detection', 'business duplicate checking'
        ],
        'repair': [
            'fix', 'correction', 'enrichment', 'amendment',
            'manual repair', 'auto repair', 'payment enrichment', 'manual verify', 'tech repair'
        ],
        'routing': [
            'route', 'network determination', 'method of payment determination',
            'payment routing', 'line selection', 'lterm selection', 'internation routing'
        ],
        'advising': [
            'pre-advising', 'preadvising', 'pre advice', 'pre-advice',
            'cover payments', 'split advising', 'advice', 'advisements',
            'fed advising', 'chips advising', 'swift advising'
        ],
        'acknowledgement': [
            'ack', 'acks', 'acknowledgment', 'acknowledgement',
            'channel acknowledgements', 'network acknowledgements',
            'payment engine acknowledgments', 'engine ack', 'level 1', 'level 2', 'confirms'
        ],
        'prioritization': ['payment prioritization', 'priority setting', 'queue priority'],
        'warehousing': ['payment warehousing', 'hold and release', 'store and forward'],
        'returns': ['payment returns', 'return items', 'rtrn processing'],
        'exceptions': ['exceptions processing', 'exception handling', 'repair queue'],
        'agreements': ['payment agreements', 'service agreements', 'entitlements'],
        'orchestration': [
            'workflow scheduling', 'workflow orchestration', 'scheduling', 'job orchestration'
        ],
        'cutoffs': ['fed cutoffs', 'chips cutoffs', 'network cutoffs', 'cut-off times', 'cutoff times'],
        'liquidity': [
            'liquidity management', 'intraday liquidity', 'intraday liq', 'ilms',
            'intraday monitoring'
        ],
        'posting': [
            'hard posting', 'memo posting', 'intraday posting', 'sod processing', 'eod processing'
        ],
        'id_generation': [
            'id generation', 'network id generation', 'sequence generation', 'reference generation'
        ],
        'encryption': ['data encryption', 'encryption', 'crypto'],
        'decryption': ['data decryption', 'decryption'],
        'masking': ['data masking', 'obfuscation', 'redaction'],
        'transformation': [
            'format transformation', 'mapping', 'iso mapping', 'iso20022 mapping',
            'mapping iso to upo', 'schema mapping'
        ],
        'api': [
            'service api', 'api/service invocation', 'api invocation',
            'service api endpoint publishing', 'endpoint publishing'
        ],
        'connectivity': [
            'channel connectivity', 'network connectivity', 'queues connectivity',
            'topics connectivity', 'mq connectivity', 'integration layer'
        ],
        'reconciliation': [
            'client & account reconciliation', 'ops bank reconciliation', 'bank rec', 'recon'
        ],
        'billing': ['client billing', 'statements', 'billing & statements', 'fee billing'],
        'fees': ['fee determination', 'fees management', 'dodd frank fees', 'analysis charges'],
        'ring_fencing': ['ring fencing', 'funds ring-fencing', 'ringfence'],
        'agreements_entitlements': ['user entitlements', 'entitlements maintenance'],
        'monitoring': [
            'business activity monitoring', 'telemetry', 'trend analysis',
            'straight thru processing analysis', 'risk event information'
        ],
        'reporting': [
            'report archiving', 'report viewing', 'report file distribution',
            'adhoc bi', 'scheduled reporting', 'reporting - risk analysis',
            'reporting - account activity', 'reporting - financial crimes'
        ],
        'alerts': ['alert dispositioning', 'payment event notification', 'notifications', 'alerts'],

        # Parties
        'party': ['entity', 'participant', 'customer', 'client', 'counterparty'],
        'creditor': ['beneficiary', 'payee', 'receiver'],
        'debtor': ['originator', 'payer', 'sender'],

        # Risk, sanctions, fraud, controls
        'sanctions': [
            'ofac', 'sanction screening', 'sanctions screening', 'watchlist', 'fircosoft'
        ],
        'aml': ['anti-money laundering', 'newton', 'financial crimes', 'kyc'],
        'fraud': [
            'fraud checking', 'fraud management', 'ceo fraud management', 'cfm', 'wholesale cust fraud'
        ],
        'funds_control': ['funds control', 'debit authority', 'debit authority management'],
        'controls': ['controls', 'anomalies detection', 'controls/anomalies & detection', 'kill switch'],
        'risk': ['risk control system', 'rcs', 'risk controls'],

        # Networks / rails / standards
        'fedwire': ['fedwire', 'fed', 'federal reserve', 'frb', 'wire network'],
        'chips': ['chips', 'chips network'],
        'swift': ['swift', 'swift network', 'society for worldwide interbank financial telecommunication'],
        'ach': ['ach', 'automated clearing house'],
        'rtp': ['rtp', 'real-time payments'],
        'fednow': ['fednow', 'instant payments', 'instant rail'],
        'wires': ['wires', 'wire transfer', 'wire'],
        'confirmation': ['confirmations', 'network confirmations', 'cust confirmations'],

        # Channels / apps
        'channels': [
            'channel', 'channel connectivity', 'channel acknowledgements', 'network acknowledgements'
        ],
        'ceo_wires': ['ceo & api wires', 'ceo wires', 'api wires'],
        'online_wires': ['online wires', 'olw'],
        'treasury_workstation': ['treasury workstation', 'pstw', 'tsys workstation'],
        'secure_fax': ['secure fax', 'fax channel'],
        'cyberpay': ['cyberpay', 'icyb'],
        'intellitracs': ['intellitracs', 'itrx'],
        'i5_gds': ['i5-gds', 'gds'],

        # Payment services specifics
        'debit_credit_determination': [
            'debit/credit determination', 'debit credit confirmation', 'dc confirmation'
        ],
        'cover_payments': ['cover payments', 'cover'],
        'payment_engine': ['payment engine acknowledgments', 'engine ack', 'engine response'],

        # Integrations / systems
        'dda_core': ['account dda', 'hogan', 'iis'],
        'ilms': ['intraday liq. monitoring system', 'ilms'],
        'tms': ['intraday posting', 'tms'],
        'gabs': ['m/w integration layer', 'gabs', 'middleware'],
        'axcis': ['report archiving & viewing', 'axcis'],
        'rpm': ['customer revenue & profit', 'rpm'],
        'fx': ['fx services', 'wxchg', 'opics', 'foreign exchange'],
        'et_window': ['electronic window', 'et', 'txn & balance info'],
        'investigations': ['investigations', 'pega', 'intellitracs', 'cases'],

        # Data & analytics
        'ai_ml': ['ai/ml', 'machine learning', 'modeling'],
        'data_inquiry': ['payment data inquiry', 'data inquiry', 'account activity inquiry'],
        'data_security': ['masking', 'encryption', 'decryption'],
        'archiving': ['report archiving', 'archive & viewing'],

        # Admin / network ops
        'certifications': ['network certification testing', 'certifications', 'cert testing'],
        'network_admin': ['network admin message generation', 'admin messages', 'service messages'],
        'sequence': ['network id/sequence generation', 'sequence generation'],
        'mapping': ['mapping iso to upo', 'lterms mapping', 'format transformation'],

        # Compliance / regulatory named items
        'dodd_frank': ['dodd frank fees management', 'dodd-frank', 'dfa fees'],

        # Existing keys from your prior list (kept for backward-compat)
        'ofac': ['sanctions', 'watchlist'],
        'function': ['procedure', 'subroutine', 'method', 'proc'],

        # Generic network label (for broad matches where needed)
        'network': ['rail', 'fed', 'chips', 'swift', 'chps', 'swf', 'fedwire']
    }
    
    # Multi-word terms dictionary
    MULTI_WORD_TERMS = {
        'cutoff times': ['cut-off times', 'cutoff time', 'deadline', 'cut-off deadline'],
        'cover payments': ['covering payment', 'cover', 'split advising'],
        'wire transfer': ['wire transfers', 'wires', 'wire payment'],
        'standing orders': ['standing order', 'recurring payment'],
        'payment repair': ['payment fix', 'repair queue', 'manual repair'],
        'network cutoffs': ['fed cutoffs', 'chips cutoffs'],
        'straight thru processing': ['stp', 'straight through processing'],
        'payment routing': ['route determination', 'network determination'],
        'liquidity management': ['intraday liquidity', 'ilms'],
        'book transfer': ['internal transfer', 'book entry'],
    }

    def __init__(self, stemmer: Optional[TextStemmer] = None):
        self.stemmer = stemmer

        if stemmer and stemmer.use_stemming:
            self.synonyms_stemmed = {}
            for key, values in self.SYNONYMS.items():
                key_stem = stemmer.stem(key)
                values_stem = [stemmer.stem(v) for v in values]
                self.synonyms_stemmed[key_stem] = values_stem
        else:
            self.synonyms_stemmed = self.SYNONYMS

    def expand_query(self, query: str, max_expansions: int = 2) -> str:
        """Expand query with synonyms"""
        terms = query.lower().split()
        expanded = set(terms)

        for term in terms:
            if self.stemmer:
                term_stem = self.stemmer.stem(term)
            else:
                term_stem = term

            if term_stem in self.synonyms_stemmed:
                synonyms = self.synonyms_stemmed[term_stem][:max_expansions]
                expanded.update(synonyms)

        return ' '.join(expanded)
    
    def detect_multi_word_terms(self, query: str) -> List[Dict[str, Any]]:
        """
        Detect known multi-word terms in query
        Returns list of detected terms with metadata
        """
        query_lower = query.lower()
        detected = []
        
        for mwt, variations in self.MULTI_WORD_TERMS.items():
            if mwt in query_lower:
                detected.append({
                    'term': mwt,
                    'variations': variations,
                    'type': 'exact'
                })
            else:
                for variation in variations:
                    if variation in query_lower:
                        detected.append({
                            'term': mwt,
                            'variations': variations,
                            'matched_variation': variation,
                            'type': 'variation'
                        })
                        break
        
        # If no exact match but query has 2-3 words, treat as potential MWT
        words = query_lower.split()
        if len(words) in [2, 3] and not detected:
            detected.append({
                'term': query_lower,
                'variations': [],
                'type': 'potential_mwt'
            })
        
        return detected


# ================================================================================
# Business Capability Taxonomy
# ================================================================================

class BusinessCapabilityTaxonomy:
    """Wire Processing Business Capabilities"""

    CAPABILITIES = {
        "Core Payment & Network": [
            "clearing networks", "fed", "chips", "swift", "clearing house",
            "network gateways", "network connectivity", "lterm", "ack", "nak"
        ],

        "Payment Processing & Execution": [
            "payment initiation", "payment routing", "payment execution",
            "preadvising", "cover payments", "liquidity management",
            "debit confirmation", "credit confirmation", "outbound payment",
            "hard posting", "cutoffs", "workflow scheduling", "orchestration",
            "intraday liquidity", "book transfer", "eod processing",
            "fee determination", "payment returns", "warehousing"
        ],

        "Instruction & Validation": [
            "instruction management", "straight thru processing", "stp",
            "pay thru validation", "method of payment", "payment enrichment",
            "payment repair", "payment verify", "auto repair",
            "date validation", "time validation", "account validation",
            "amount validation", "currency validation", "standing orders"
        ],

        "Controls & Risk Management": [
            "controls services", "anomalies detection", "sanctions screening",
            "fircosoft", "ofac", "funds control", "fraud checking",
            "debit authority", "duplicate checking", "debit blocks",
            "credit blocks", "ceo fraud", "anti money laundering", "aml"
        ],

        "Data & Reporting": [
            "data management", "report distribution", "financial crimes reporting",
            "risk analysis reporting", "historical data", "payment reconciliation",
            "general ledger", "gl feeds", "account activity reporting",
            "adhoc reporting", "event notification", "alert", "statements"
        ],

        "Service Integration": [
            "data masking", "transaction replay", "data encryption",
            "channel acknowledgements", "service api", "endpoint publishing",
            "duplicate detection", "api invocation", "queues", "topics",
            "format transformation", "id generation", "schema validation"
        ],

        "ISO Standards & Formats": [
            "iso20022", "iso 20022", "pacs.008", "pacs.009", "pacs.002",
            "pain.001", "camt.053", "mt103", "mt202", "fedwire", "xml"
        ],

        "Validation & Screening": [
            "bic validation", "bic code", "iban validation", "iban",
            "party validation", "sanctions check", "watchlist screening",
            "name screening", "address validation", "routing validation"
        ],

        "Transaction Processing": [
            "wire transfer", "wire payment", "domestic wire", "international wire",
            "cross border payment", "rtgs", "ach", "clearing", "settlement"
        ]
    }

    @classmethod
    def get_all_keywords(cls) -> List[str]:
        """Get all keywords across all capabilities"""
        keywords = []
        for kws in cls.CAPABILITIES.values():
            keywords.extend(kws)
        return list(set(keywords))


class CapabilityMapper:
    """Map documents to business capabilities"""

    def __init__(self, stemmer: Optional[TextStemmer] = None):
        self.taxonomy = BusinessCapabilityTaxonomy()
        self.stemmer = stemmer
        self.capability_keywords = {}

        # Build keyword to capability mapping
        for capability, keywords in self.taxonomy.CAPABILITIES.items():
            for kw in keywords:
                if stemmer and stemmer.use_stemming:
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
        """Map keywords to business capabilities"""
        capability_scores = defaultdict(float)

        text_lower = text.lower()
        if self.stemmer and self.stemmer.use_stemming:
            text_stemmed = self.stemmer.stem_text(text_lower)
        else:
            text_stemmed = text_lower

        # Score from keywords
        for keyword, kw_score in keywords:
            if keyword in self.capability_keywords:
                for cap in self.capability_keywords[keyword]:
                    capability_scores[cap] += kw_score

        # Direct text matching
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

        # Sort and return
        sorted_caps = sorted(
            capability_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_caps


# ================================================================================
# File Extraction
# ================================================================================

class UniversalFileExtractor:
    """Extract text from PDFs, code, and text files"""

    def __init__(self):
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
        else:
            return self._extract_text(file_path)

    def _extract_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract text from PDF"""
        content = {
            "text": "",
            "file_type": "pdf",
            "line_count": 0,
            "has_tables": False
        }

        if not self.pdf_available:
            return content

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    content["text"] += page_text + "\n"

                    tables = page.extract_tables()
                    if tables:
                        content["has_tables"] = True
                        for table in tables:
                            table_text = self._table_to_text(table)
                            content["text"] += f"\n[TABLE]\n{table_text}\n[/TABLE]\n"
        except Exception as e:
            print(f"Error extracting {pdf_path}: {e}")

        content["line_count"] = content["text"].count('\n')
        return content

    def _extract_code(self, code_path: Path) -> Dict[str, Any]:
        """Extract text from code files with metadata"""
        content = {
            "text": "",
            "file_type": "code",
            "language": code_path.suffix[1:],
            "line_count": 0,
            "metadata": {}
        }

        # Try different encodings
        for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
            try:
                with open(code_path, 'r', encoding=encoding) as f:
                    content["text"] = f.read()
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"Error reading {code_path}: {e}")
                break

        content["line_count"] = content["text"].count('\n')

        # Extract code metadata
        content["metadata"] = self._extract_code_metadata(
            content["text"],
            content["language"]
        )

        return content

    def _extract_text(self, text_path: Path) -> Dict[str, Any]:
        """Extract text from plain text files"""
        content = {
            "text": "",
            "file_type": "text",
            "line_count": 0
        }

        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                with open(text_path, 'r', encoding=encoding) as f:
                    content["text"] = f.read()
                break
            except UnicodeDecodeError:
                continue

        content["line_count"] = content["text"].count('\n')
        return content

    def _extract_code_metadata(self, code_text: str, language: str) -> Dict[str, Any]:
        """Extract metadata from code (functions, classes, imports)"""
        metadata = {
            "functions": [],
            "classes": [],
            "imports": [],
            "system_calls": [],
            "has_main": False
        }

        # Language-specific patterns
        patterns = {
            'python': {
                'function': r'def\s+(\w+)\s*\(',
                'class': r'class\s+(\w+)\s*[\(:]',
                'import': r'(?:from\s+\S+\s+)?import\s+(\S+)'
            },
            'java': {
                'function': r'(?:public|private|protected)?\s+\w+\s+(\w+)\s*\(',
                'class': r'(?:public|private)?\s+class\s+(\w+)',
                'import': r'import\s+([\w\.]+);'
            },
            'c': {
                'function': r'\w+\s+(\w+)\s*\([^\)]*\)\s*\{',
                'class': r'(?:struct|typedef\s+struct)\s+(\w+)',
                'include': r'#include\s+[<"]([^>"]+)[>"]'
            },
            'cpp': {
                'function': r'\w+\s+(\w+)\s*\([^\)]*\)\s*\{',
                'class': r'class\s+(\w+)',
                'include': r'#include\s+[<"]([^>"]+)[>"]'
            },
            'tal': {
                'procedure': r'(?:PROC|PROCEDURE)\s+(\w+)',
                'subproc': r'SUBPROC\s+(\w+)',
                'function': r'INT\s+PROCEDURE\s+(\w+)',
                'system_call': r'\$\(\w+\)',
                'variable': r'(?:INT|STRING|FIXED|REAL)\s+(\w+)'
            },
            'cbl': {
                'procedure': r'(?:PROCEDURE\s+DIVISION|PERFORM)\s+(\w+(?:-\w+)*)',
                'section': r'(\w+(?:-\w+)*)\s+SECTION',
                'paragraph': r'^(\w+(?:-\w+)*)\./',
                'working_storage': r'01\s+(\w+(?:-\w+)*)',
                'copybook': r'COPY\s+(\w+)'
            },
            'cobol': {
                'procedure': r'(?:PROCEDURE\s+DIVISION|PERFORM)\s+(\w+(?:-\w+)*)',
                'section': r'(\w+(?:-\w+)*)\s+SECTION',
                'working_storage': r'01\s+(\w+(?:-\w+)*)'
            },
            'sql': {
                'procedure': r'CREATE\s+(?:OR\s+REPLACE\s+)?PROCEDURE\s+(\w+)',
                'function': r'CREATE\s+(?:OR\s+REPLACE\s+)?FUNCTION\s+(\w+)',
                'table': r'(?:FROM|JOIN)\s+(\w+)',
                'cte': r'WITH\s+(\w+)\s+AS'
            }
        }

        lang_patterns = patterns.get(language, patterns.get('python', {}))

        # Extract functions/procedures
        for key in ['function', 'procedure', 'subproc']:
            if key in lang_patterns:
                matches = re.findall(
                    lang_patterns[key],
                    code_text,
                    re.IGNORECASE | re.MULTILINE
                )
                metadata['functions'].extend(list(set(matches))[:50])

        # Extract classes
        if 'class' in lang_patterns:
            classes = re.findall(
                lang_patterns['class'],
                code_text,
                re.IGNORECASE | re.MULTILINE
            )
            metadata['classes'] = list(set(classes))[:50]

        # Extract imports/includes
        for key in ['import', 'include', 'copybook']:
            if key in lang_patterns:
                imports = re.findall(
                    lang_patterns[key],
                    code_text,
                    re.IGNORECASE | re.MULTILINE
                )
                metadata['imports'].extend(list(set(imports))[:50])

        # Extract system calls (TAL-specific)
        if 'system_call' in lang_patterns:
            system_calls = re.findall(lang_patterns['system_call'], code_text)
            metadata['system_calls'] = list(set(system_calls))[:50]

        # Check for main
        if 'main' in [f.lower() for f in metadata['functions']]:
            metadata['has_main'] = True

        return metadata

    def _table_to_text(self, table: List[List]) -> str:
        """Convert table to text"""
        if not table:
            return ""

        text_rows = []
        for row in table:
            cleaned = [str(cell) if cell else "" for cell in row]
            text_rows.append(" | ".join(cleaned))

        return "\n".join(text_rows)


# ================================================================================
# Keyword Extraction
# ================================================================================

class KeywordExtractor:
    """Extract keywords from text using patterns and domain knowledge"""

    def __init__(self, stemmer: Optional[TextStemmer] = None):
        self.stemmer = stemmer

        # Domain-specific patterns
        self.patterns = [
            (r'ISO[\s-]?\d+', 'iso_standard'),
            (r'pacs\.\d+\{\?\.\d+\}*', 'payment_message'),
            (r'pain\.\d+\{\?\.\d+\}*', 'payment_message'),
            (r'camt\.\d+\{\?\.\d+\}*', 'payment_message'),
            (r'MT\d{3}', 'swift_message'),
            (r'ACE[\s-]?\d+', 'ace_code'),
            (r'[A-Z]{6}[A-Z0-9]{2}(?:[A-Z0-9]{3})?', 'bic_code'),
            (r'\$\{[A-Z_]+\}', 'system_function'),
        ]

        # Important domain terms
        self.important_terms = {
            'credit', 'debit', 'party', 'determination', 'validation',
            'payment', 'wire', 'transfer', 'transaction', 'settlement',
            'clearing', 'posting', 'execution', 'routing', 'initiation',
            'cutoff', 'deadline', 'eod', 'sod', 'intraday', 'batch',
            'screening', 'sanctions', 'ofac', 'fircosoft', 'compliance',
            'fedwire', 'chips', 'swift', 'ach', 'sepa',
            'creditor', 'debtor', 'beneficiary', 'originator',
            'repair', 'enrichment', 'orchestration', 'workflow',
            'function', 'class', 'method', 'procedure'
        }

        if stemmer and stemmer.use_stemming:
            self.important_terms_stemmed = {stemmer.stem(t) for t in self.important_terms}
        else:
            self.important_terms_stemmed = self.important_terms

        # Stopwords
        if stemmer and stemmer.use_stemming:
            self.stopwords = {stemmer.stem(sw) for sw in STOPWORDS}
        else:
            self.stopwords = STOPWORDS

    def extract(self, text: str, max_keywords: int = 20) -> List[Tuple[str, float]]:
        """Extract keywords with confidence scores"""
        if not text or len(text.strip()) < 10:
            return []

        keyword_scores = defaultdict(float)
        text_lower = text.lower()

        # Tokenize
        words = re.findall(r'\b[\w-]+\b', text_lower)
        if self.stemmer and self.stemmer.use_stemming:
            stemmed_words = [self.stemmer.stem(w) for w in words]
        else:
            stemmed_words = words

        # 1. Domain patterns (highest confidence)
        for pattern, pattern_type in self.patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                keyword_scores[match.lower()] += 3.0

        # 2. Important domain terms
        for term_stem in self.important_terms_stemmed:
            count = stemmed_words.count(term_stem)
            if count > 0:
                keyword_scores[term_stem] += 3.0 * math.log1p(count)

        # 3. Capitalized words
        capitalized = re.findall(r'\b[A-Z][a-z]{2,}\b', text)
        for word in capitalized:
            if self.stemmer and self.stemmer.use_stemming:
                word_stem = self.stemmer.stem(word.lower())
            else:
                word_stem = word.lower()

            if word_stem not in self.stopwords:
                keyword_scores[word_stem] += 1.5

        # 4. Business capability keywords
        capability_keywords = BusinessCapabilityTaxonomy.get_all_keywords()
        for kw in capability_keywords:
            if self.stemmer and self.stemmer.use_stemming:
                kw_stem = self.stemmer.stem_text(kw)
                count = text_lower.count(kw)
            else:
                kw_stem = kw
                count = text_lower.count(kw)

            if count > 0:
                keyword_scores[kw_stem] += 2.0 * math.log1p(count)

        # Filter and sort
        filtered = []
        for kw, score in keyword_scores.items():
            if (len(kw) > 2 and
                kw not in self.stopwords and
                not kw.isdigit()):
                filtered.append((kw, score))

        filtered.sort(key=lambda x: x[1], reverse=True)
        return filtered[:max_keywords]


# ================================================================================
# ENHANCEMENT: Position Index Builder
# ================================================================================

class PositionIndexBuilder:
    """Build position index for phrase and proximity search"""
    
    def __init__(self):
        self.position_index = defaultdict(list)
    
    def add_document(self, doc_id: int, text: str):
        """Add document to position index"""
        words = text.lower().split()
        
        for position, word in enumerate(words):
            # Clean word
            word_clean = re.sub(r'[^\w-]', '', word)
            if word_clean and len(word_clean) > 1:
                self.position_index[word_clean].append((doc_id, position))
    
    def get_index(self) -> Dict[str, List[Tuple[int, int]]]:
        """Get the position index"""
        return dict(self.position_index)
    
    def search_phrase(self, phrase: str, doc_ids: Optional[set] = None) -> List[Dict]:
        """
        Find exact phrase matches using position index
        Returns list of {doc_id, position} for phrase matches
        """
        words = phrase.lower().split()
        if not words:
            return []
        
        # Get positions for first word
        if words[0] not in self.position_index:
            return []
        
        first_word_positions = self.position_index[words[0]]
        
        # Filter by doc_ids if provided
        if doc_ids:
            first_word_positions = [(doc_id, pos) for doc_id, pos in first_word_positions if doc_id in doc_ids]
        
        matches = []
        
        # Check each occurrence of first word
        for doc_id, start_pos in first_word_positions:
            is_match = True
            
            # Check if subsequent words appear at consecutive positions
            for i, word in enumerate(words[1:], 1):
                if word not in self.position_index:
                    is_match = False
                    break
                
                # Check if word appears at start_pos + i in same document
                expected_pos = start_pos + i
                word_positions = self.position_index[word]
                
                if not any(d == doc_id and p == expected_pos for d, p in word_positions):
                    is_match = False
                    break
            
            if is_match:
                matches.append({
                    'doc_id': doc_id,
                    'position': start_pos,
                    'phrase': phrase
                })
        
        return matches
    
    def search_proximity(self, term1: str, term2: str, max_distance: int = 5) -> List[Dict]:
        """
        Find documents where term1 and term2 appear within max_distance words
        """
        if term1 not in self.position_index or term2 not in self.position_index:
            return []
        
        matches = []
        
        # Group positions by document
        doc_positions1 = defaultdict(list)
        for doc_id, pos in self.position_index[term1]:
            doc_positions1[doc_id].append(pos)
        
        doc_positions2 = defaultdict(list)
        for doc_id, pos in self.position_index[term2]:
            doc_positions2[doc_id].append(pos)
        
        # Find common documents
        common_docs = set(doc_positions1.keys()) & set(doc_positions2.keys())
        
        for doc_id in common_docs:
            pos1_list = doc_positions1[doc_id]
            pos2_list = doc_positions2[doc_id]
            
            # Find closest pair
            min_distance = float('inf')
            best_pos1 = None
            best_pos2 = None
            
            for pos1 in pos1_list:
                for pos2 in pos2_list:
                    distance = abs(pos1 - pos2) - 1
                    if distance < min_distance and distance <= max_distance:
                        min_distance = distance
                        best_pos1 = pos1
                        best_pos2 = pos2
            
            if min_distance <= max_distance:
                matches.append({
                    'doc_id': doc_id,
                    'distance': min_distance,
                    'term1_pos': best_pos1,
                    'term2_pos': best_pos2
                })
        
        return matches


# ================================================================================
# ENHANCEMENT: Multi-Stage Ranker
# ================================================================================

class MultiStageRanker:
    """
    Multi-stage ranking with rich features
    Stage 1: BM25 retrieval (top 100 candidates)
    Stage 2: Re-ranking with 12+ features
    Stage 3: Diversity filtering
    """
    
    def __init__(self):
        # Feature weights (tunable)
        self.feature_weights = {
            'bm25_score': 1.0,
            'keyword_match': 2.5,
            'capability_match': 2.0,
            'title_match': 3.0,
            'exact_phrase': 4.0,
            'function_match': 2.5,
            'code_metadata': 1.5,
            'match_density': 2.0,
            'position_score': 1.5,
            'file_type_boost': 1.0,
            'domain_pattern': 3.0,
            'query_coverage': 1.8,
        }
    
    def rank(
        self,
        query_info: Dict,
        bm25_scores: np.ndarray,
        metadata_store: List[Dict],
        document_store: List[str],
        position_index: PositionIndexBuilder,
        top_k: int = 20,
        explain: bool = False
    ) -> List[Dict]:
        """
        Main ranking function
        """
        # Stage 1: Get top 100 BM25 candidates
        candidates = self._get_candidates(bm25_scores, metadata_store, document_store, top_n=100)
        
        if not candidates:
            return []
        
        # Stage 2: Re-rank with features
        for candidate in candidates:
            features = self._extract_features(query_info, candidate, position_index)
            final_score = self._compute_final_score(features)
            
            candidate['features'] = features
            candidate['final_score'] = final_score
        
        # Sort by final score
        candidates.sort(key=lambda x: x['final_score'], reverse=True)
        
        # Stage 3: Apply diversity
        diverse_results = self._apply_diversity(candidates, top_k)
        
        # Add explanations
        if explain:
            diverse_results = self._add_explanations(diverse_results)
        
        return diverse_results
    
    def _get_candidates(self, bm25_scores, metadata_store, document_store, top_n=100):
        """Stage 1: Get top N candidates"""
        top_indices = np.argsort(bm25_scores)[::-1][:top_n]
        
        candidates = []
        for idx in top_indices:
            if bm25_scores[idx] > 0.01:
                candidates.append({
                    'doc_id': idx,
                    'bm25_score': float(bm25_scores[idx]),
                    'metadata': metadata_store[idx],
                    'text': document_store[idx]
                })
        
        return candidates
    
    def _extract_features(self, query_info, candidate, position_index):
        """Extract all ranking features"""
        features = {}
        
        metadata = candidate['metadata']
        text = candidate['text']
        doc_id = candidate['doc_id']
        
        # Feature 1: BM25 baseline
        features['bm25_score'] = candidate['bm25_score']
        
        # Feature 2: Keyword match
        features['keyword_match'] = self._compute_keyword_match(
            query_info['tokens'], metadata.get('keywords', [])
        )
        
        # Feature 3: Capability match
        features['capability_match'] = self._compute_capability_match(
            query_info, metadata.get('capabilities', [])
        )
        
        # Feature 4: Title match
        features['title_match'] = self._compute_title_match(
            query_info['original_query'], metadata.get('source_file', '')
        )
        
        # Feature 5: Exact phrase match (CRITICAL for multi-word terms!)
        features['exact_phrase'] = self._compute_exact_phrase_match(
            query_info, text, position_index, doc_id
        )
        
        # Feature 6: Function/class match
        features['function_match'] = self._compute_code_match(
            query_info['tokens'], metadata.get('code_metadata', {})
        )
        
        # Feature 7: Code metadata match
        features['code_metadata'] = self._compute_metadata_match(
            query_info['tokens'], metadata.get('code_metadata', {})
        )
        
        # Feature 8: Match density
        features['match_density'] = self._compute_match_density(
            query_info['tokens'], text
        )
        
        # Feature 9: Position score
        features['position_score'] = self._compute_position_score(
            query_info['tokens'], text
        )
        
        # Feature 10: File type boost
        features['file_type_boost'] = self._compute_file_type_boost(
            metadata.get('file_type', ''), metadata.get('language', '')
        )
        
        # Feature 11: Domain pattern match
        features['domain_pattern'] = self._compute_domain_pattern_match(
            query_info['original_query'], text
        )
        
        # Feature 12: Query coverage
        features['query_coverage'] = self._compute_query_coverage(
            query_info['tokens'], text
        )
        
        return features
    
    def _compute_keyword_match(self, query_tokens, doc_keywords):
        """How many extracted keywords match query?"""
        if not doc_keywords:
            return 0.0
        
        query_set = set(query_tokens)
        doc_keyword_set = set(doc_keywords)
        
        matches = query_set & doc_keyword_set
        
        if not matches:
            return 0.0
        
        # F1 score
        precision = len(matches) / len(query_set) if query_set else 0
        recall = len(matches) / len(doc_keyword_set) if doc_keyword_set else 0
        
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
            return f1 * 10.0
        
        return 0.0
    
    def _compute_capability_match(self, query_info, doc_capabilities):
        """Does document capability match query?"""
        if not doc_capabilities:
            return 0.0
        
        # Simple heuristic: check if any query capability matches
        # In production, you'd infer query capabilities from the query
        return 5.0 if doc_capabilities else 0.0
    
    def _compute_title_match(self, query, filename):
        """Match in filename is STRONG signal"""
        filename_lower = filename.lower()
        query_lower = query.lower()
        
        score = 0.0
        
        # Exact query in filename
        if query_lower in filename_lower:
            score += 10.0
        
        # Individual words in filename
        for word in query_lower.split():
            if word in filename_lower:
                score += 3.0
        
        return score
    
    def _compute_exact_phrase_match(self, query_info, text, position_index, doc_id):
        """
        CRITICAL: Exact phrase matching for multi-word terms
        """
        score = 0.0
        
        # Check for exact phrase in original query
        original_query = query_info['original_query'].lower()
        text_lower = text.lower()
        
        # Count exact occurrences
        count = text_lower.count(original_query)
        if count > 0:
            score += min(math.log1p(count) * 15.0, 40.0)
        
        # Use position index for more precise phrase matching
        if 'multi_word_terms' in query_info:
            for mwt in query_info['multi_word_terms']:
                phrase_matches = position_index.search_phrase(mwt['term'])
                doc_matches = [m for m in phrase_matches if m['doc_id'] == doc_id]
                if doc_matches:
                    score += len(doc_matches) * 10.0
        
        return score
    
    def _compute_code_match(self, query_tokens, code_metadata):
        """Match in function/class names"""
        if not code_metadata:
            return 0.0
        
        score = 0.0
        
        # Check functions
        for func in code_metadata.get('functions', []):
            func_lower = func.lower()
            for token in query_tokens:
                if token.lower() in func_lower:
                    score += 3.0
        
        # Check classes
        for cls in code_metadata.get('classes', []):
            cls_lower = cls.lower()
            for token in query_tokens:
                if token.lower() in cls_lower:
                    score += 3.0
        
        return min(score, 20.0)
    
    def _compute_metadata_match(self, query_tokens, code_metadata):
        """Match in imports/system calls"""
        if not code_metadata:
            return 0.0
        
        score = 0.0
        
        for imp in code_metadata.get('imports', []):
            imp_lower = imp.lower()
            for token in query_tokens:
                if token.lower() in imp_lower:
                    score += 1.0
        
        for call in code_metadata.get('system_calls', []):
            call_lower = call.lower()
            for token in query_tokens:
                if token.lower() in call_lower:
                    score += 2.0
        
        return min(score, 10.0)
    
    def _compute_match_density(self, query_tokens, text):
        """Are query terms concentrated?"""
        text_lower = text.lower()
        words = text_lower.split()
        
        if len(words) < 2:
            return 0.0
        
        # Find all positions
        positions = []
        for token in query_tokens:
            for i, word in enumerate(words):
                if token.lower() in word:
                    positions.append(i)
        
        if len(positions) < 2:
            return 0.0
        
        positions.sort()
        span = positions[-1] - positions[0] + 1
        density = len(positions) / span if span > 0 else 0
        
        return min(density * 20.0, 15.0)
    
    def _compute_position_score(self, query_tokens, text):
        """Matches early in document are better"""
        text_lower = text.lower()
        words = text_lower.split()
        
        if not words:
            return 0.0
        
        earliest_position = len(words)
        
        for token in query_tokens:
            for i, word in enumerate(words):
                if token.lower() in word:
                    earliest_position = min(earliest_position, i)
                    break
        
        if earliest_position == len(words):
            return 0.0
        
        position_ratio = earliest_position / len(words)
        return (1.0 - position_ratio) * 10.0
    
    def _compute_file_type_boost(self, file_type, language):
        """Boost certain file types"""
        boost = 0.0
        
        if file_type == 'code':
            boost += 2.0
            if language in ['tal', 'cobol', 'sql']:
                boost += 1.0
        elif file_type == 'pdf':
            boost += 1.5
        
        return boost
    
    def _compute_domain_pattern_match(self, query, text):
        """Match domain patterns (ISO, SWIFT, etc.)"""
        score = 0.0
        
        # ISO standards
        if re.search(r'ISO[\s-]?\d+', query, re.IGNORECASE):
            iso_matches = re.findall(r'ISO[\s-]?\d+', text, re.IGNORECASE)
            score += len(iso_matches) * 5.0
        
        # SWIFT messages
        swift_patterns = [r'MT\d{3}', r'pacs\.\d+', r'pain\.\d+', r'camt\.\d+']
        for pattern in swift_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                matches = re.findall(pattern, text, re.IGNORECASE)
                score += len(matches) * 5.0
        
        # ACE codes
        if re.search(r'ACE', query, re.IGNORECASE):
            ace_matches = re.findall(r'ACE[\s-]?\d+', text, re.IGNORECASE)
            score += len(ace_matches) * 4.0
        
        return min(score, 25.0)
    
    def _compute_query_coverage(self, query_tokens, text):
        """What % of query terms appear?"""
        text_lower = text.lower()
        
        matched = sum(1 for token in query_tokens if token.lower() in text_lower)
        total = len(query_tokens)
        
        if total == 0:
            return 0.0
        
        coverage = matched / total
        return coverage * 10.0
    
    def _compute_final_score(self, features):
        """Combine all features"""
        final_score = 0.0
        
        for feature_name, feature_value in features.items():
            weight = self.feature_weights.get(feature_name, 1.0)
            final_score += feature_value * weight
        
        return final_score
    
    def _apply_diversity(self, candidates, top_k):
        """Ensure result diversity (max 3 per file)"""
        diverse_results = []
        file_counts = Counter()
        max_per_file = 3
        
        for candidate in candidates:
            source_file = candidate['metadata'].get('source_file', '')
            
            if file_counts[source_file] < max_per_file:
                diverse_results.append(candidate)
                file_counts[source_file] += 1
                
                if len(diverse_results) >= top_k:
                    break
        
        return diverse_results
    
    def _add_explanations(self, results):
        """Add human-readable explanations"""
        for result in results:
            features = result.get('features', {})
            
            # Top contributing features
            sorted_features = sorted(
                features.items(),
                key=lambda x: x[1] * self.feature_weights.get(x[0], 1.0),
                reverse=True
            )
            
            explanation = []
            for feature_name, feature_value in sorted_features[:5]:
                if feature_value > 0:
                    weight = self.feature_weights.get(feature_name, 1.0)
                    contribution = feature_value * weight
                    explanation.append(
                        f"{feature_name}: {feature_value:.2f} × {weight} = {contribution:.2f}"
                    )
            
            result['explanation'] = explanation
        
        return results


# ================================================================================
# Main Indexer (Enhanced)
# ================================================================================

class UniversalFileIndexer:
    """Universal file indexer with BM25, stemming, and capability mapping"""

    def __init__(
        self,
        files_folder: str,
        index_path: str = "./universal_index",
        file_extensions: Optional[List[str]] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        use_stemming: bool = True
    ):
        self.files_folder = Path(files_folder)
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_stemming = use_stemming

        # File extensions
        if file_extensions:
            self.file_extensions = [
                ext if ext.startswith('.') else f'.{ext}'
                for ext in file_extensions
            ]
        else:
            self.file_extensions = list(ALL_EXTENSIONS)

        # Components
        self.file_extractor = UniversalFileExtractor()
        self.stemmer = TextStemmer(use_stemming=use_stemming)
        self.keyword_extractor = KeywordExtractor(stemmer=self.stemmer)
        self.capability_mapper = CapabilityMapper(stemmer=self.stemmer)
        self.query_expander = DomainQueryExpander(stemmer=self.stemmer)

        # Storage
        self.metadata_store = []
        self.document_store = []
        self.bm25 = None
        
        # ENHANCEMENT: Position index
        self.position_index_builder = PositionIndexBuilder()

        # Statistics
        self.keyword_doc_counts = Counter()
        self.capability_doc_counts = Counter()
        self.filetype_counts = Counter()
        self.total_chunks = 0

    def scan_files(self, verbose: bool = False) -> List[Path]:
        """Scan folder for supported files (case-insensitive)"""
        print(f"Scanning: {self.files_folder}")

        files = []
        seen = set()

        all_files = list(self.files_folder.glob("**/*"))

        if verbose:
            print(f"\nTotal files found: {len([f for f in all_files if f.is_file()])}")
            print(f"Looking for extensions: {', '.join(self.file_extensions[:20])}...")

        extensions_lower = [ext.lower() for ext in self.file_extensions]

        for file_path in all_files:
            if file_path.is_file():
                file_ext_lower = file_path.suffix.lower()
                if file_ext_lower in extensions_lower:
                    if str(file_path) not in seen:
                        files.append(file_path)
                        seen.add(str(file_path))
                        if verbose:
                            print(f"  ✓ {file_path.name}")
            elif verbose and file_path.suffix:
                print(f"  ✗ {file_path.name} (extension: {file_path.suffix})")

        ext_counts = {}
        for f in files:
            ext = f.suffix.lower()
            ext_counts[ext] = ext_counts.get(ext, 0) + 1

        print(f"\nFound {len(files)} matching files:")
        for ext, count in sorted(ext_counts.items()):
            print(f"  {ext}: {count} files")

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

    def index_files(self, verbose_scan: bool = False):
        """Main indexing pipeline"""
        files = self.scan_files(verbose=verbose_scan)

        if not files:
            print("No files found!")
            return None

        all_chunks = []

        print("\n=== Phase 1: Extracting Content ===")
        for idx, file_path in enumerate(files, 1):
            print(f"[{idx}/{len(files)}] {file_path.name}")

            content = self.file_extractor.extract(file_path)

            if not content.get("text") or len(content["text"].strip()) < 10:
                print(f"  ⚠ No text extracted, skipping")
                continue

            file_type = content.get("file_type", "unknown")
            self.filetype_counts[file_type] += 1

            keywords = self.keyword_extractor.extract(content["text"])
            capabilities = self.capability_mapper.map_to_capabilities(
                keywords,
                content["text"]
            )

            doc_metadata = {
                "source_file": str(file_path.name),
                "source_path": str(file_path),
                "file_type": file_type,
                "file_size": file_path.stat().st_size,
                "language": content.get("language", ""),
                "line_count": content.get("line_count", 0),
                "has_tables": content.get("has_tables", False),
                "code_metadata": content.get("metadata", {})
            }

            chunks = self.chunk_text(content["text"], doc_metadata)

            for chunk in chunks:
                chunk["keywords"] = [kw for kw, _ in keywords[:10]]
                chunk["capabilities"] = [cap for cap, _ in capabilities[:3]]

                self.keyword_doc_counts.update(set(chunk["keywords"]))
                self.capability_doc_counts.update(set(chunk["capabilities"]))

            all_chunks.extend(chunks)
            print(f"  Created {len(chunks)} chunks")

        self.total_chunks = len(all_chunks)
        print(f"\nTotal chunks: {self.total_chunks}")

        # Phase 2: Compute TF-IDF
        print("\n=== Phase 2: Computing TF-IDF ===")
        idf_scores = {}
        for keyword, doc_count in self.keyword_doc_counts.items():
            if doc_count > 0:
                idf = math.log(self.total_chunks / doc_count)
                idf_scores[keyword] = idf

        # Phase 3: Store documents + BUILD POSITION INDEX
        print("\n=== Phase 3: Storing Documents & Building Position Index ===")
        for doc_id, chunk in enumerate(all_chunks):
            # Store processed text
            if self.use_stemming:
                processed_text = self.stemmer.stem_text(chunk["text"])
            else:
                processed_text = chunk["text"]

            self.document_store.append(processed_text)
            
            # ENHANCEMENT: Build position index
            self.position_index_builder.add_document(doc_id, chunk["text"])

            # Store metadata
            top_capabilities = chunk["capabilities"][:3]
            metadata = {
                "source_file": chunk["source_file"],
                "file_type": chunk["file_type"],
                "language": chunk.get("language", ""),
                "chunk_index": chunk["chunk_index"],
                "keywords": chunk["keywords"],
                "capabilities": top_capabilities,
                "primary_capability": top_capabilities[0] if top_capabilities else "unknown",
                "text_snippet": chunk["text"][:500]
            }

            self.metadata_store.append(metadata)
        
        print(f"✓ Position index built with {len(self.position_index_builder.position_index)} terms")

        # Phase 4: Build BM25 index
        print("\n=== Phase 4: Building BM25 Index ===")
        tokenized_docs = [doc.split() for doc in self.document_store]
        self.bm25 = BM25Okapi(tokenized_docs)
        print("✓ BM25 index built")

        # Phase 5: Save everything
        print("\n=== Phase 5: Saving Index ===")

        with open(self.index_path / "metadata.pkl", 'wb') as f:
            pickle.dump(self.metadata_store, f)
        print("✓ Saved metadata")

        with open(self.index_path / "documents.pkl", 'wb') as f:
            pickle.dump(self.document_store, f)
        print("✓ Saved documents")

        with open(self.index_path / "bm25.pkl", 'wb') as f:
            pickle.dump(self.bm25, f)
        print("✓ Saved BM25 index")
        
        # ENHANCEMENT: Save position index
        with open(self.index_path / "position_index.pkl", 'wb') as f:
            pickle.dump(self.position_index_builder.get_index(), f)
        print("✓ Saved position index")

        # Save statistics
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
        print("✓ Saved statistics")

        self.print_statistics()

        return stats

    def print_statistics(self):
        """Print indexing statistics"""
        print("\n" + "="*70)
        print("INDEXING STATISTICS")
        print("="*70)
        print(f"Total chunks:            {self.total_chunks}")
        print(f"Unique keywords:         {len(self.keyword_doc_counts)}")
        print(f"Unique capabilities:     {len(self.capability_doc_counts)}")
        print(f"Stemming:                {'Enabled' if self.use_stemming else 'Disabled'}")

        print("\nFile Types:")
        for file_type, count in self.filetype_counts.items():
            print(f"  {file_type:15s} {count:4d} files")

        print("\nTop 10 Capabilities:")
        for cap, count in self.capability_doc_counts.most_common(10):
            print(f"  {cap:40s} {count:4d} chunks")


# ================================================================================
# ENHANCED SEARCHER with Multi-Term & Relevance Boosting
# ================================================================================

class UniversalFileSearcher:
    """
    Enhanced searcher with:
    - Multi-term/phrase detection
    - Position-based phrase matching
    - Multi-stage ranking with 12+ features
    - Query explanations
    """

    def __init__(self, index_path: str = "./universal_index"):
        self.index_path = Path(index_path)

        # Load stats
        with open(self.index_path / "stats.json", 'r') as f:
            self.stats = json.load(f)

        self.use_stemming = self.stats.get('use_stemming', False)

        # Load components
        self.stemmer = TextStemmer(use_stemming=self.use_stemming)
        self.query_expander = DomainQueryExpander(stemmer=self.stemmer)

        # Load indexes
        with open(self.index_path / "metadata.pkl", 'rb') as f:
            self.metadata_store = pickle.load(f)

        with open(self.index_path / "documents.pkl", 'rb') as f:
            self.document_store = pickle.load(f)

        with open(self.index_path / "bm25.pkl", 'rb') as f:
            self.bm25 = pickle.load(f)
        
        # ENHANCEMENT: Load position index
        try:
            with open(self.index_path / "position_index.pkl", 'rb') as f:
                position_index_dict = pickle.load(f)
                self.position_index = PositionIndexBuilder()
                self.position_index.position_index = defaultdict(list, position_index_dict)
        except FileNotFoundError:
            print("⚠ Position index not found. Phrase search will be limited.")
            self.position_index = None
        
        # ENHANCEMENT: Multi-stage ranker
        self.ranker = MultiStageRanker()

        print(f"✓ Enhanced index loaded: {len(self.document_store)} chunks")

    def search(
        self,
        query: str,
        top_k: int = 20,
        file_type_filter: Optional[str] = None,
        use_query_expansion: bool = True,
        enable_multi_term: bool = True,
        explain: bool = True,
        verbose: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Enhanced search with multi-term support and relevance boosting

        Args:
            query: Search query
            top_k: Number of results
            file_type_filter: Filter by 'code', 'pdf', 'text', etc.
            use_query_expansion: Expand query with synonyms
            enable_multi_term: Enable multi-term detection
            explain: Show ranking explanations
            verbose: Show details
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"Query: {query}")
            print(f"{'='*70}")

        # Build query info
        query_info = {
            'original_query': query,
            'tokens': query.lower().split()
        }
        
        # ENHANCEMENT: Detect multi-word terms
        if enable_multi_term:
            multi_word_terms = self.query_expander.detect_multi_word_terms(query)
            if multi_word_terms:
                query_info['multi_word_terms'] = multi_word_terms
                if verbose:
                    print(f"Detected multi-word terms: {[m['term'] for m in multi_word_terms]}")

        # Expand query
        processed_query = query.lower()
        if use_query_expansion:
            expanded_query = self.query_expander.expand_query(processed_query)
            if verbose:
                print(f"Expanded: {expanded_query}")
            processed_query = expanded_query

        # Stem query
        if self.use_stemming:
            processed_query = self.stemmer.stem_text(processed_query)
        
        query_info['expanded_tokens'] = processed_query.split()

        # BM25 search
        query_tokens = processed_query.split()
        bm25_scores = self.bm25.get_scores(query_tokens)

        # ENHANCEMENT: Multi-stage ranking
        results = self.ranker.rank(
            query_info=query_info,
            bm25_scores=bm25_scores,
            metadata_store=self.metadata_store,
            document_store=self.document_store,
            position_index=self.position_index if self.position_index else PositionIndexBuilder(),
            top_k=top_k,
            explain=explain
        )

        # Filter by file type
        if file_type_filter:
            results = [r for r in results if r['metadata'].get('file_type') == file_type_filter]

        # Format results
        formatted_results = []
        for r in results:
            formatted_results.append({
                "text": r['metadata'].get('text_snippet', self.document_store[r['doc_id']][:500]),
                "source_file": r['metadata']['source_file'],
                "file_type": r['metadata']['file_type'],
                "language": r['metadata'].get('language', ''),
                "chunk_index": r['metadata']['chunk_index'],
                "score": r['final_score'],
                "bm25_score": r['bm25_score'],
                "capabilities": r['metadata'].get('capabilities', []),
                "keywords": r['metadata'].get('keywords', []),
                "explanation": r.get('explanation', [])
            })

        if verbose and formatted_results:
            print(f"\n🎯 Top Result:")
            print(f"   File: {formatted_results[0]['source_file']}")
            print(f"   Score: {formatted_results[0]['score']:.2f}")
            print(f"   Explanation:")
            for line in formatted_results[0]['explanation']:
                print(f"      {line}")

        return formatted_results


# ================================================================================
# Main Entry Point
# ================================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Universal File Indexer v2.0 - Enhanced Edition"
    )
    parser.add_argument("--folder", required=True, help="Path to files folder")
    parser.add_argument("--index-path", default="./universal_index", help="Index storage path")
    parser.add_argument("--action", choices=["index", "search"], default="index")
    parser.add_argument("--query", help="Search query")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results")
    parser.add_argument("--extensions", nargs='+', help="File extensions to index")
    parser.add_argument("--disable-stemming", action="store_true")
    parser.add_argument("--no-explain", action="store_true", help="Disable explanations")
    parser.add_argument("--verbose", action="store_true", help="Show detailed output")

    args = parser.parse_args()

    if args.action == "index":
        indexer = UniversalFileIndexer(
            files_folder=args.folder,
            index_path=args.index_path,
            file_extensions=args.extensions,
            use_stemming=not args.disable_stemming
        )
        indexer.index_files(verbose_scan=args.verbose)

    elif args.action == "search":
        if not args.query:
            print("Error: --query required for search")
            return

        searcher = UniversalFileSearcher(index_path=args.index_path)
        results = searcher.search(
            args.query,
            top_k=args.top_k,
            explain=not args.no_explain,
            verbose=args.verbose
        )

        print(f"\n{'='*70}")
        print(f"Results for: {args.query}")
        print(f"{'='*70}\n")

        for i, result in enumerate(results, 1):
            print(f"[{i}] {result['source_file']} (score: {result['score']:.2f})")
            print(f"    Type: {result['file_type']}")
            if result['language']:
                print(f"    Language: {result['language']}")
            if result['capabilities']:
                print(f"    Capabilities: {', '.join(result['capabilities'][:2])}")
            if result['explanation'] and not args.no_explain:
                print(f"    Top signals:")
                for exp in result['explanation'][:3]:
                    print(f"      • {exp}")
            print(f"    {result['text'][:150]}...")
            print()


if __name__ == "__main__":
    main()
