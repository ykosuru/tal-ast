"""
Universal File Indexer v2.0 - Production Ready
Supports: PDFs, Code (Python, Java, C, TAL, COBOL, SQL), Text, Config files
Features: BM25 search, stemming, business capability mapping, code metadata extraction

All patches integrated, no modifications needed.
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
        'cutoffs': ['fed cutoffs', 'chips cutoffs', 'network cutoffs', 'cut-off times'],
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

        # Channels / apps (aliases included for matching)
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
# Main Indexer
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

        # Statistics
        self.keyword_doc_counts = Counter()
        self.capability_doc_counts = Counter()
        self.filetype_counts = Counter()
        self.total_chunks = 0

    def scan_files(self, verbose: bool = False) -> List[Path]:
        """Scan folder for supported files (case-insensitive)"""
        print(f"Scanning: {self.files_folder}")

        files = []
        seen = set()  # Avoid duplicates

        # Get all files recursively
        all_files = list(self.files_folder.glob("**/*"))

        if verbose:
            print(f"\nTotal files found: {len([f for f in all_files if f.is_file()])}")
            print(f"Looking for extensions: {', '.join(self.file_extensions[:20])}...")

        # Filter by extension (case-insensitive)
        extensions_lower = [ext.lower() for ext in self.file_extensions]

        for file_path in all_files:
            if file_path.is_file():
                file_ext_lower = file_path.suffix.lower()
                if file_ext_lower in extensions_lower:
                    # Avoid duplicates
                    if str(file_path) not in seen:
                        files.append(file_path)
                        seen.add(str(file_path))
                        if verbose:
                            print(f"  ✓ {file_path.name}")
            elif verbose and file_path.suffix:
                print(f"  ✗ {file_path.name} (extension: {file_path.suffix})")

        # Print breakdown by extension
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

            # Extract content
            content = self.file_extractor.extract(file_path)

            if not content.get("text") or len(content["text"].strip()) < 10:
                print(f"  ⚠ No text extracted, skipping")
                continue

            file_type = content.get("file_type", "unknown")
            self.filetype_counts[file_type] += 1

            # Extract keywords
            keywords = self.keyword_extractor.extract(content["text"])

            # Map capabilities
            capabilities = self.capability_mapper.map_to_capabilities(
                keywords,
                content["text"]
            )

            # Metadata
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

            # Chunk
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

        # Phase 3: Store documents
        print("\n=== Phase 3: Storing Documents ===")
        for chunk in all_chunks:
            # Store processed text
            if self.use_stemming:
                processed_text = self.stemmer.stem_text(chunk["text"])
            else:
                processed_text = chunk["text"]

            self.document_store.append(processed_text)

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
                "text_snippet": chunk["text"][:500]  # For display
            }

            self.metadata_store.append(metadata)

        # Phase 4: Build BM25 index (CRITICAL - saves index!)
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

        # CRITICAL: Save BM25 index
        with open(self.index_path / "bm25.pkl", 'wb') as f:
            pickle.dump(self.bm25, f)
        print("✓ Saved BM25 index")

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


class UniversalFileSearcher:
    """Search indexed files with BM25 + query expansion"""

    def __init__(self, index_path: str = "./universal_index"):
        self.index_path = Path(index_path)

        # Load stats
        with open(self.index_path / "stats.json", 'r') as f:
            self.stats = json.load(f)

        self.use_stemming = self.stats.get('use_stemming', False)

        # Load components
        self.stemmer = TextStemmer(use_stemming=self.use_stemming)
        self.query_expander = DomainQueryExpander(stemmer=self.stemmer)

        # Load metadata and documents
        with open(self.index_path / "metadata.pkl", 'rb') as f:
            self.metadata_store = pickle.load(f)

        with open(self.index_path / "documents.pkl", 'rb') as f:
            self.document_store = pickle.load(f)

        # Load BM25 index
        with open(self.index_path / "bm25.pkl", 'rb') as f:
            self.bm25 = pickle.load(f)

        print(f"✓ Index loaded: {len(self.document_store)} chunks")

    def search(
        self,
        query: str,
        top_k: int = 20,
        file_type_filter: Optional[str] = None,
        use_query_expansion: bool = True,
        verbose: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Search indexed files

        Args:
            query: Search query
            top_k: Number of results
            file_type_filter: Filter by 'code', 'pdf', 'text', etc.
            use_query_expansion: Expand query with synonyms
            verbose: Show details
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"Query: {query}")
            print(f"{'='*70}")

        # Process query
        processed_query = query.lower()

        # Expand query
        if use_query_expansion:
            expanded_query = self.query_expander.expand_query(processed_query)
            #if verbose and expanded_query != processed_query:
            print(f"Expanded: {expanded_query}")
            processed_query = expanded_query

        # Stem query
        if self.use_stemming:
            processed_query = self.stemmer.stem_text(processed_query)

        # Search with BM25
        query_tokens = processed_query.split()
        scores = self.bm25.get_scores(query_tokens)

        # Get top results
        top_indices = np.argsort(scores)[::-1][:top_k * 5]

        # Format results
        results = []
        for idx in top_indices:
            if scores[idx] < 0.01:
                break

            metadata = self.metadata_store[idx]

            # Filter by file type
            if file_type_filter and metadata.get('file_type') != file_type_filter:
                continue

            results.append({
                "text": metadata.get('text_snippet', self.document_store[idx][:500]),
                "source_file": metadata['source_file'],
                "file_type": metadata['file_type'],
                "language": metadata.get('language', ''),
                "chunk_index": metadata['chunk_index'],
                "score": float(scores[idx]),
                "capabilities": metadata.get('capabilities', []),
                "keywords": metadata.get('keywords', [])
            })

            if len(results) >= top_k:
                break

        if verbose:
            print(f"\nFound {len(results)} results")

        return results


# ================================================================================
# Main Entry Point
# ================================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Universal File Indexer v2.0 - Production Ready"
    )
    parser.add_argument("--folder", required=True, help="Path to files folder")
    parser.add_argument("--index-path", default="./universal_index", help="Index storage path")
    parser.add_argument("--action", choices=["index", "search"], default="index")
    parser.add_argument("--query", help="Search query")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results")
    parser.add_argument("--extensions", nargs='+', help="File extensions to index")
    parser.add_argument("--disable-stemming", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="Show detailed file scanning")

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
            verbose=args.verbose
        )

        print(f"\n{'='*70}")
        print(f"Results for: {args.query}")
        print(f"{'='*70}\n")

        for i, result in enumerate(results, 1):
            print(f"[{i}] {result['source_file']} (score: {result['score']:.3f})")
            print(f"    Type: {result['file_type']}")
            if result['language']:
                print(f"    Language: {result['language']}")
            if result['capabilities']:
                print(f"    Capabilities: {', '.join(result['capabilities'][:2])}")
            print(f"    {result['text'][:150]}...")
            print()


if __name__ == "__main__":
    main()
