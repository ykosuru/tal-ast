"""
Wire Processing Indexer with Custom Keyword Mapping
Supports user-provided domain keywords with priorities and categories
"""

import os
import json
import yaml
import math
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional, Tuple, Set
import hashlib
import re

# PDF processing
import PyPDF2
import pdfplumber

# Vector embeddings (optional)
import numpy as np
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

import pickle
from rank_bm25 import BM25Okapi

# YAML for configuration (optional)
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    print("Warning: PyYAML not available, use JSON for config files")


class CustomKeywordTaxonomy:
    """
    User-provided keyword taxonomy with priorities and categories
    Supports YAML/JSON configuration files
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.categories = {}
        self.keyword_to_category = {}
        self.keyword_priorities = {}
        self.all_keywords = set()
        
        if config_path and os.path.exists(config_path):
            self.load_from_file(config_path)
        else:
            # Default keywords if no config provided
            self._load_default_keywords()
    
    def load_from_file(self, config_path: str):
        """Load keywords from YAML or JSON file"""
        print(f"Loading custom keywords from: {config_path}")
        
        with open(config_path, 'r') as f:
            if (config_path.endswith('.yaml') or config_path.endswith('.yml')) and YAML_AVAILABLE:
                config = yaml.safe_load(f)
            else:
                config = json.load(f)
        
        self._parse_config(config)
        print(f"✓ Loaded {len(self.all_keywords)} keywords in {len(self.categories)} categories")
    
    def _parse_config(self, config: Dict[str, Any]):
        """Parse configuration structure"""
        for category_name, category_data in config.items():
            priority = category_data.get('priority', 'medium')
            keywords = category_data.get('keywords', [])
            
            # Store category
            self.categories[category_name] = {
                'priority': priority,
                'keywords': keywords
            }
            
            # Map keywords to category and priority
            for kw in keywords:
                kw_lower = kw.lower()
                self.all_keywords.add(kw_lower)
                self.keyword_to_category[kw_lower] = category_name
                self.keyword_priorities[kw_lower] = self._priority_to_score(priority)
    
    def _priority_to_score(self, priority: str) -> float:
        """Convert priority level to numeric score"""
        priority_map = {
            'critical': 3.0,
            'high': 2.5,
            'medium': 2.0,
            'low': 1.5
        }
        return priority_map.get(priority.lower(), 2.0)
    
    def _load_default_keywords(self):
        """Load default wire processing keywords"""
        default_config = {
            "payment_processing": {
                "priority": "high",
                "keywords": [
                    "credit party determination", "debit party determination",
                    "payment routing", "payment execution", "liquidity management",
                    "fee determination", "payment enrichment", "payment repair"
                ]
            },
            "validation_screening": {
                "priority": "critical",
                "keywords": [
                    "OFAC screening", "sanctions screening", "Fircosoft",
                    "party validation", "account validation", "BIC validation",
                    "IBAN validation", "watchlist screening"
                ]
            },
            "networks_standards": {
                "priority": "high",
                "keywords": [
                    "ISO20022", "pacs.008", "pain.001", "MT103", "MT202",
                    "Fedwire", "CHIPS", "SWIFT", "ACH"
                ]
            }
        }
        self._parse_config(default_config)
    
    def get_keyword_priority(self, keyword: str) -> float:
        """Get priority score for a keyword"""
        return self.keyword_priorities.get(keyword.lower(), 1.0)
    
    def get_keyword_category(self, keyword: str) -> Optional[str]:
        """Get category for a keyword"""
        return self.keyword_to_category.get(keyword.lower())
    
    def get_keywords_by_category(self, category: str) -> List[str]:
        """Get all keywords in a category"""
        if category in self.categories:
            return self.categories[category]['keywords']
        return []
    
    def get_all_categories(self) -> List[str]:
        """Get all category names"""
        return list(self.categories.keys())


class EnhancedKeywordExtractor:
    """
    Enhanced keyword extractor with custom keyword support
    Maps documents to user-provided keywords
    """
    
    def __init__(self, custom_taxonomy: CustomKeywordTaxonomy):
        self.custom_taxonomy = custom_taxonomy
        
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
        ]
        
        # Stopwords - expanded to catch generic business terms
        self.stopwords = {
            # Basic stopwords
            'process', 'system', 'data', 'information', 'general', 'related',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'from', 'by', 'as', 'is', 'was', 'are',
            'be', 'been', 'have', 'has', 'had', 'will', 'would', 'should',
            'could', 'may', 'might', 'must', 'can', 'this', 'that', 'these',
            'those', 'it', 'its', 'they', 'their', 'them',
            
            # Generic business terms (VERY COMMON - FILTER THESE OUT!)
            'business', 'approved', 'technical', 'approval', 'requirement',
            'requirements', 'document', 'documentation', 'section', 'page',
            'table', 'figure', 'example', 'note', 'notes', 'reference',
            'references', 'description', 'overview', 'summary', 'details',
            'detail', 'status', 'type', 'types', 'item', 'items', 'value',
            'values', 'field', 'fields', 'name', 'names', 'number', 'numbers',
            'date', 'dates', 'time', 'times', 'version', 'versions', 'change',
            'changes', 'update', 'updates', 'new', 'old', 'current', 'previous',
            'next', 'first', 'last', 'high', 'low', 'medium', 'level', 'levels',
            'phase', 'phases', 'step', 'steps', 'procedure', 'procedures',
            'method', 'methods', 'function', 'functions', 'operation', 'operations',
            'activity', 'activities', 'action', 'actions', 'result', 'results',
            'output', 'outputs', 'input', 'inputs', 'parameter', 'parameters',
            'attribute', 'attributes', 'property', 'properties', 'component',
            'components', 'element', 'elements', 'module', 'modules', 'service',
            'services', 'application', 'applications', 'user', 'users', 'customer',
            'customers', 'client', 'clients', 'vendor', 'vendors', 'provider',
            'providers', 'source', 'sources', 'target', 'targets', 'destination',
            'destinations', 'origin', 'origins', 'object', 'objects', 'entity',
            'entities', 'record', 'records', 'entry', 'entries', 'list', 'lists',
            'set', 'sets', 'group', 'groups', 'class', 'classes', 'category',
            'categories', 'code', 'codes', 'identifier', 'identifiers', 'key',
            'keys', 'index', 'indices', 'flag', 'flags', 'option', 'options',
            'setting', 'settings', 'configuration', 'configurations', 'setup',
            'setups', 'installation', 'installations', 'deployment', 'deployments',
            'implementation', 'implementations', 'definition', 'definitions',
            'specification', 'specifications', 'standard', 'standards', 'policy',
            'policies', 'rule', 'rules', 'guideline', 'guidelines', 'principle',
            'principles', 'best', 'practice', 'practices', 'pattern', 'patterns',
            'model', 'models', 'template', 'templates', 'format', 'formats',
            'structure', 'structures', 'schema', 'schemas', 'design', 'designs',
            'architecture', 'architectures', 'framework', 'frameworks', 'platform',
            'platforms', 'technology', 'technologies', 'tool', 'tools', 'utility',
            'utilities', 'helper', 'helpers', 'support', 'supports', 'enable',
            'enables', 'disable', 'disables', 'allow', 'allows', 'prevent',
            'prevents', 'restrict', 'restricts', 'limit', 'limits', 'control',
            'controls', 'manage', 'manages', 'handle', 'handles', 'perform',
            'performs', 'execute', 'executes', 'run', 'runs', 'start', 'starts',
            'stop', 'stops', 'end', 'ends', 'begin', 'begins', 'complete',
            'completes', 'finish', 'finishes', 'cancel', 'cancels', 'abort',
            'aborts', 'terminate', 'terminates', 'close', 'closes', 'open',
            'opens', 'create', 'creates', 'delete', 'deletes', 'remove', 'removes',
            'add', 'adds', 'insert', 'inserts', 'modify', 'modifies', 'update',
            'updates', 'change', 'changes', 'alter', 'alters', 'adjust', 'adjusts',
            'set', 'sets', 'get', 'gets', 'retrieve', 'retrieves', 'fetch',
            'fetches', 'load', 'loads', 'save', 'saves', 'store', 'stores',
            'read', 'reads', 'write', 'writes', 'send', 'sends', 'receive',
            'receives', 'transmit', 'transmits', 'transfer', 'transfers', 'move',
            'moves', 'copy', 'copies', 'duplicate', 'duplicates', 'clone', 'clones',
            'merge', 'merges', 'split', 'splits', 'combine', 'combines', 'separate',
            'separates', 'join', 'joins', 'connect', 'connects', 'disconnect',
            'disconnects', 'link', 'links', 'unlink', 'unlinks', 'associate',
            'associates', 'disassociate', 'disassociates', 'attach', 'attaches',
            'detach', 'detaches', 'bind', 'binds', 'unbind', 'unbinds', 'map',
            'maps', 'unmap', 'unmaps', 'convert', 'converts', 'transform',
            'transforms', 'translate', 'translates', 'parse', 'parses', 'format',
            'formats', 'validate', 'validates', 'verify', 'verifies', 'check',
            'checks', 'test', 'tests', 'compare', 'compares', 'match', 'matches',
            'search', 'searches', 'find', 'finds', 'locate', 'locates', 'identify',
            'identifies', 'detect', 'detects', 'discover', 'discovers', 'scan',
            'scans', 'filter', 'filters', 'sort', 'sorts', 'order', 'orders',
            'rank', 'ranks', 'prioritize', 'prioritizes', 'organize', 'organizes',
            'arrange', 'arranges', 'structure', 'structures', 'format', 'formats',
        }
    
    def extract(
        self, 
        text: str, 
        max_keywords: int = 30
    ) -> Tuple[List[Tuple[str, float]], Dict[str, List[str]]]:
        """
        Extract keywords with scores AND map to custom keyword categories
        
        Returns:
            - List of (keyword, score) tuples
            - Dict mapping category -> matched keywords
        """
        if not text or len(text.strip()) < 10:
            return [], {}
        
        keyword_scores = defaultdict(float)
        text_lower = text.lower()
        
        # 1. Extract patterns
        for pattern, _ in self.patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                keyword_scores[match.lower()] += 3.0
        
        # 2. Extract custom user keywords (HIGHEST PRIORITY!)
        for kw in self.custom_taxonomy.all_keywords:
            if kw in text_lower:
                count = text_lower.count(kw)
                priority_score = self.custom_taxonomy.get_keyword_priority(kw)
                # BOOST: Custom keywords get 2x higher scores
                keyword_scores[kw] += (priority_score * 2.0) * math.log1p(count)
        
        # 3. Extract individual words from multi-word custom keywords
        for kw in self.custom_taxonomy.all_keywords:
            words = kw.split()
            if len(words) > 1:  # Multi-word phrase
                for word in words:
                    if (len(word) > 3 and 
                        word not in self.stopwords and 
                        word in text_lower):
                        count = text_lower.count(word)
                        # BOOST: Words from custom keywords get higher scores
                        keyword_scores[word] += 2.0 * math.log1p(count)
        
        # 4. Capitalized words (LOWER PRIORITY - only if not in stopwords)
        capitalized_words = re.findall(r'\b[A-Z][a-z]{2,}\b', text)
        for word in capitalized_words:
            word_lower = word.lower()
            # Only add if NOT a stopword and NOT already captured by custom keywords
            if (word_lower not in self.stopwords and 
                word_lower not in self.custom_taxonomy.all_keywords):
                # REDUCED score for generic capitalized words
                keyword_scores[word_lower] += 1.0
        
        # 5. Acronyms (ONLY if 3-5 letters and not in stopwords)
        acronyms = re.findall(r'\b[A-Z]{3,5}\b', text)
        for acronym in acronyms:
            acronym_lower = acronym.lower()
            if (acronym_lower not in self.stopwords and
                acronym_lower not in self.custom_taxonomy.all_keywords):
                keyword_scores[acronym_lower] += 1.5
        
        # Filter and sort
        filtered = [
            (kw, score) for kw, score in keyword_scores.items() 
            if len(kw) > 2 and kw not in self.stopwords
        ]
        filtered.sort(key=lambda x: x[1], reverse=True)
        
        # Map to categories
        category_mapping = self._map_to_categories(
            [kw for kw, _ in filtered[:max_keywords]]
        )
        
        return filtered[:max_keywords], category_mapping
    
    def _map_to_categories(self, keywords: List[str]) -> Dict[str, List[str]]:
        """Map extracted keywords to custom categories"""
        category_mapping = defaultdict(list)
        
        for kw in keywords:
            category = self.custom_taxonomy.get_keyword_category(kw)
            if category:
                category_mapping[category].append(kw)
        
        return dict(category_mapping)


class KeywordMappedIndexer:
    """
    Indexer that maps documents to user-provided keywords
    Creates a keyword-document matrix for easy navigation
    """
    
    def __init__(
        self,
        pdf_folder: str,
        index_path: str = "./wire_index",
        keyword_config: Optional[str] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        use_embeddings: bool = False
    ):
        self.pdf_folder = Path(pdf_folder)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        self.use_embeddings = use_embeddings and EMBEDDINGS_AVAILABLE and FAISS_AVAILABLE
        
        # Load custom keywords
        print("=" * 70)
        print("Loading Custom Keyword Taxonomy")
        print("=" * 70)
        self.custom_taxonomy = CustomKeywordTaxonomy(keyword_config)
        
        print("\nInitializing enhanced keyword extractor...")
        self.keyword_extractor = EnhancedKeywordExtractor(self.custom_taxonomy)
        
        if self.use_embeddings:
            print(f"Loading embedding model...")
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
        
        # Keyword mapping structures
        self.keyword_doc_counts = Counter()
        self.category_doc_counts = Counter()
        self.keyword_document_matrix = defaultdict(set)  # keyword -> set of doc IDs
        self.category_document_matrix = defaultdict(set)  # category -> set of doc IDs
        
        self.total_chunks = 0
    
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text from PDF"""
        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    text += page_text + "\n"
        except Exception as e:
            print(f"Error extracting from {pdf_path}: {e}")
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
            except Exception as e2:
                print(f"Fallback also failed: {e2}")
        
        return text
    
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
    
    def index_pdfs(self, batch_size: int = 32):
        """Main indexing pipeline with keyword mapping"""
        print(f"\nScanning PDF folder: {self.pdf_folder}")
        pdf_files = list(self.pdf_folder.glob("**/*.pdf"))
        print(f"Found {len(pdf_files)} PDF files")
        
        all_chunks = []
        
        print("\n" + "=" * 70)
        print("Phase 1: Extracting Content & Mapping Keywords")
        print("=" * 70)
        
        for idx, pdf_path in enumerate(pdf_files, 1):
            print(f"\n[{idx}/{len(pdf_files)}] Processing: {pdf_path.name}")
            
            text = self.extract_text_from_pdf(pdf_path)
            if not text.strip():
                print(f"  ⚠ No text extracted, skipping")
                continue
            
            # Extract keywords with category mapping
            keywords, category_mapping = self.keyword_extractor.extract(text)
            
            print(f"  ✓ Extracted {len(keywords)} keywords")
            if keywords:
                print(f"    Top 5: {', '.join([kw for kw, _ in keywords[:5]])}")
            
            # Show category mapping
            if category_mapping:
                print(f"  ✓ Mapped to {len(category_mapping)} categories:")
                for cat, kws in list(category_mapping.items())[:3]:
                    print(f"    - {cat}: {', '.join(kws[:3])}")
            
            doc_metadata = {
                "source_file": str(pdf_path.name),
                "source_path": str(pdf_path),
                "file_size": pdf_path.stat().st_size,
            }
            
            chunks = self.chunk_text(text, doc_metadata)
            
            # Add keywords and categories to each chunk
            for chunk_idx, chunk in enumerate(chunks):
                chunk["keywords"] = [kw for kw, score in keywords]
                chunk["keyword_scores"] = dict(keywords)
                chunk["category_mapping"] = category_mapping
                chunk["categories"] = list(category_mapping.keys())
                
                # Update document counts
                self.keyword_doc_counts.update(set(chunk["keywords"]))
                self.category_doc_counts.update(set(chunk["categories"]))
                
                # Update keyword-document matrix
                doc_id = f"{pdf_path.name}:{chunk_idx}"
                for kw in chunk["keywords"]:
                    self.keyword_document_matrix[kw].add(doc_id)
                for cat in chunk["categories"]:
                    self.category_document_matrix[cat].add(doc_id)
            
            all_chunks.extend(chunks)
            print(f"  ✓ Created {len(chunks)} chunks")
        
        self.total_chunks = len(all_chunks)
        print(f"\n{'='*70}")
        print(f"Total chunks created: {self.total_chunks}")
        print(f"{'='*70}")
        
        # Compute TF-IDF
        print("\nPhase 2: Computing TF-IDF Scores")
        idf_scores = {}
        for keyword, doc_count in self.keyword_doc_counts.items():
            if doc_count > 0:
                idf = math.log(self.total_chunks / doc_count)
                idf_scores[keyword] = idf
        
        print(f"✓ Computed IDF for {len(idf_scores)} unique keywords")
        
        for chunk in all_chunks:
            keyword_weights = {}
            for kw in chunk["keywords"]:
                if kw in idf_scores:
                    keyword_weights[kw] = idf_scores[kw]
            chunk["keyword_weights"] = keyword_weights
            
            sorted_kw = sorted(keyword_weights.items(), key=lambda x: x[1], reverse=True)
            chunk["top_keywords"] = [kw for kw, _ in sorted_kw[:5]]
        
        # Store chunks
        print("\nPhase 3: Storing Index")
        for chunk in all_chunks:
            metadata = {
                "source_file": chunk["source_file"],
                "chunk_index": chunk["chunk_index"],
                "keywords": chunk["keywords"][:15],
                "top_keywords": chunk["top_keywords"],
                "categories": chunk["categories"],
                "category_mapping": chunk["category_mapping"],
                "keyword_scores": chunk["keyword_scores"]
            }
            self.metadata_store.append(metadata)
            self.document_store.append(chunk["text"])
        
        print(f"✓ Stored {len(all_chunks)} chunks")
        
        # Save to disk
        print("\nPhase 4: Saving to Disk")
        
        metadata_path = self.index_path / "metadata.pkl"
        documents_path = self.index_path / "documents.pkl"
        
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata_store, f)
        print(f"✓ Saved metadata: {metadata_path}")
        
        with open(documents_path, 'wb') as f:
            pickle.dump(self.document_store, f)
        print(f"✓ Saved documents: {documents_path}")
        
        # Save keyword mappings
        keyword_map_path = self.index_path / "keyword_mappings.json"
        keyword_mappings = {
            "keyword_document_matrix": {
                kw: list(docs) for kw, docs in self.keyword_document_matrix.items()
            },
            "category_document_matrix": {
                cat: list(docs) for cat, docs in self.category_document_matrix.items()
            },
            "keyword_doc_counts": dict(self.keyword_doc_counts),
            "category_doc_counts": dict(self.category_doc_counts)
        }
        
        with open(keyword_map_path, 'w') as f:
            json.dump(keyword_mappings, f, indent=2)
        print(f"✓ Saved keyword mappings: {keyword_map_path}")
        
        # Save statistics
        stats_path = self.index_path / "stats.json"
        stats = {
            "total_chunks": len(all_chunks),
            "total_keywords": len(idf_scores),
            "total_categories": len(self.category_doc_counts),
            "pdf_files": len(pdf_files),
            "custom_keywords_used": len(self.custom_taxonomy.all_keywords),
            "categories_defined": list(self.custom_taxonomy.get_all_categories()),
            "idf_scores": idf_scores,
            "use_embeddings": self.use_embeddings
        }
        
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"✓ Saved statistics: {stats_path}")
        
        return stats
    
    def get_statistics(self):
        """Display comprehensive statistics"""
        print("\n" + "=" * 70)
        print("INDEXING STATISTICS")
        print("=" * 70)
        
        print(f"\nTotal chunks: {self.total_chunks}")
        print(f"Unique keywords: {len(self.keyword_doc_counts)}")
        print(f"Custom keywords used: {len(self.custom_taxonomy.all_keywords)}")
        print(f"Categories: {len(self.category_doc_counts)}")
        
        print("\n" + "=" * 70)
        print("TOP 10 CATEGORIES BY DOCUMENT COUNT")
        print("=" * 70)
        for cat, count in self.category_doc_counts.most_common(10):
            print(f"  {cat:40s} - {count:4d} chunks")
        
        print("\n" + "=" * 70)
        print("TOP 10 KEYWORDS BY DOCUMENT COUNT")
        print("=" * 70)
        for kw, count in self.keyword_doc_counts.most_common(10):
            category = self.custom_taxonomy.get_keyword_category(kw)
            priority = self.custom_taxonomy.get_keyword_priority(kw)
            cat_str = f"({category})" if category else ""
            print(f"  {kw:30s} {cat_str:30s} - {count:4d} chunks (priority: {priority:.1f})")
        
        print("\n" + "=" * 70)
        print("KEYWORD COVERAGE BY CATEGORY")
        print("=" * 70)
        for category in self.custom_taxonomy.get_all_categories():
            category_keywords = self.custom_taxonomy.get_keywords_by_category(category)
            found_keywords = [
                kw for kw in category_keywords
                if kw.lower() in self.keyword_doc_counts
            ]
            coverage = len(found_keywords) / len(category_keywords) * 100 if category_keywords else 0
            print(f"  {category:40s} - {len(found_keywords)}/{len(category_keywords)} keywords ({coverage:.1f}%)")


class KeywordMappedSearcher:
    """
    Searcher with custom keyword awareness
    Supports search by query, keyword, or category
    """
    
    def __init__(
        self, 
        index_path: str = "./wire_index",
        keyword_config: Optional[str] = None
    ):
        self.index_path = Path(index_path)
        
        # Load custom keywords
        self.custom_taxonomy = CustomKeywordTaxonomy(keyword_config)
        
        # Load stats
        stats_path = self.index_path / "stats.json"
        with open(stats_path, 'r') as f:
            self.stats = json.load(f)
        
        # Load metadata and documents
        metadata_path = self.index_path / "metadata.pkl"
        documents_path = self.index_path / "documents.pkl"
        
        with open(metadata_path, 'rb') as f:
            self.metadata_store = pickle.load(f)
        
        with open(documents_path, 'rb') as f:
            self.document_store = pickle.load(f)
        
        # Load keyword mappings
        keyword_map_path = self.index_path / "keyword_mappings.json"
        if keyword_map_path.exists():
            with open(keyword_map_path, 'r') as f:
                self.keyword_mappings = json.load(f)
        else:
            self.keyword_mappings = {}
        
        # Build BM25 index
        tokenized_docs = [doc.lower().split() for doc in self.document_store]
        self.bm25 = BM25Okapi(tokenized_docs)
        
        print(f"✓ Loaded index with {len(self.document_store)} chunks")
        print(f"✓ Custom keywords: {len(self.custom_taxonomy.all_keywords)}")
        print(f"✓ Categories: {len(self.custom_taxonomy.get_all_categories())}")
    
    def search(
        self,
        query: str,
        top_k: int = 20,
        category_filter: Optional[str] = None,
        keyword_filter: Optional[str] = None,
        verbose: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Search with custom keyword awareness
        
        Args:
            query: Search query
            top_k: Number of results
            category_filter: Only return docs in this category
            keyword_filter: Only return docs with this keyword
            verbose: Show detailed search process
        """
        if verbose:
            print(f"\n{'='*80}")
            print(f"Query: {query}")
            if category_filter:
                print(f"Category Filter: {category_filter}")
            if keyword_filter:
                print(f"Keyword Filter: {keyword_filter}")
            print(f"{'='*80}")
        
        # Tokenize query
        query_tokens = query.lower().split()
        
        # Get BM25 scores
        bm25_scores = self.bm25.get_scores(query_tokens)
        
        # Boost scores for custom keywords
        for idx, metadata in enumerate(self.metadata_store):
            # Check if doc contains custom keywords from query
            doc_keywords = set([kw.lower() for kw in metadata.get('keywords', [])])
            query_keywords = set(query.lower().split())
            
            # Boost if custom keyword match
            for qk in query_keywords:
                if qk in self.custom_taxonomy.all_keywords:
                    if qk in doc_keywords:
                        priority = self.custom_taxonomy.get_keyword_priority(qk)
                        bm25_scores[idx] *= (1.0 + priority * 0.3)
        
        # Get top candidates
        search_k = min(1000, top_k * 20) if category_filter or keyword_filter else top_k
        top_indices = np.argsort(bm25_scores)[::-1][:search_k]
        
        # Format results with filtering
        results = []
        for idx in top_indices:
            if bm25_scores[idx] < 0.01:
                continue
            
            metadata = self.metadata_store[idx]
            
            # Apply category filter
            if category_filter:
                if category_filter not in metadata.get('categories', []):
                    continue
            
            # Apply keyword filter
            if keyword_filter:
                if keyword_filter.lower() not in [kw.lower() for kw in metadata.get('keywords', [])]:
                    continue
            
            # Normalize score
            max_score = bm25_scores.max() if bm25_scores.max() > 0 else 1.0
            normalized_score = bm25_scores[idx] / max_score
            
            results.append({
                "text": self.document_store[idx],
                "source_file": metadata['source_file'],
                "chunk_index": metadata['chunk_index'],
                "bm25_score": float(bm25_scores[idx]),
                "normalized_score": float(normalized_score),
                "keywords": metadata.get('keywords', [])[:10],
                "top_keywords": metadata.get('top_keywords', []),
                "categories": metadata.get('categories', []),
                "category_mapping": metadata.get('category_mapping', {})
            })
            
            if len(results) >= top_k:
                break
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"Found {len(results)} results")
            print(f"{'='*80}\n")
        
        return results
    
    def search_by_keyword(
        self,
        keyword: str,
        top_k: int = 20
    ) -> List[Dict[str, Any]]:
        """Search documents containing a specific keyword"""
        results = []
        
        keyword_lower = keyword.lower()
        
        for idx, metadata in enumerate(self.metadata_store):
            doc_keywords = [kw.lower() for kw in metadata.get('keywords', [])]
            if keyword_lower in doc_keywords:
                results.append({
                    "text": self.document_store[idx],
                    "source_file": metadata['source_file'],
                    "chunk_index": metadata['chunk_index'],
                    "keywords": metadata.get('keywords', [])[:10],
                    "categories": metadata.get('categories', []),
                    "category_mapping": metadata.get('category_mapping', {})
                })
                
                if len(results) >= top_k:
                    break
        
        return results
    
    def search_by_category(
        self,
        category: str,
        top_k: int = 20
    ) -> List[Dict[str, Any]]:
        """Search documents in a specific category"""
        results = []
        
        for idx, metadata in enumerate(self.metadata_store):
            if category in metadata.get('categories', []):
                results.append({
                    "text": self.document_store[idx],
                    "source_file": metadata['source_file'],
                    "chunk_index": metadata['chunk_index'],
                    "keywords": metadata.get('keywords', [])[:10],
                    "categories": metadata.get('categories', []),
                    "category_mapping": metadata.get('category_mapping', {})
                })
                
                if len(results) >= top_k:
                    break
        
        return results
    
    def get_keyword_documents(self, keyword: str) -> List[str]:
        """Get list of documents containing a keyword"""
        keyword_lower = keyword.lower()
        if 'keyword_document_matrix' in self.keyword_mappings:
            return self.keyword_mappings['keyword_document_matrix'].get(keyword_lower, [])
        return []
    
    def get_category_documents(self, category: str) -> List[str]:
        """Get list of documents in a category"""
        if 'category_document_matrix' in self.keyword_mappings:
            return self.keyword_mappings['category_document_matrix'].get(category, [])
        return []
    
    def get_keyword_coverage(self) -> Dict[str, Any]:
        """Get keyword coverage statistics"""
        coverage = {}
        
        for category in self.custom_taxonomy.get_all_categories():
            category_keywords = self.custom_taxonomy.get_keywords_by_category(category)
            
            found_keywords = []
            for kw in category_keywords:
                kw_lower = kw.lower()
                if 'keyword_doc_counts' in self.keyword_mappings:
                    if kw_lower in self.keyword_mappings['keyword_doc_counts']:
                        count = self.keyword_mappings['keyword_doc_counts'][kw_lower]
                        found_keywords.append((kw, count))
            
            coverage[category] = {
                'total_keywords': len(category_keywords),
                'found_keywords': len(found_keywords),
                'coverage_percent': (len(found_keywords) / len(category_keywords) * 100) if category_keywords else 0,
                'keyword_details': found_keywords
            }
        
        return coverage


def create_example_keyword_config():
    """Create an example keyword configuration file"""
    example_config = {
        "payment_processing": {
            "priority": "high",
            "keywords": [
                "credit party determination",
                "debit party determination",
                "payment routing",
                "payment execution",
                "liquidity management",
                "fee determination",
                "payment enrichment",
                "payment repair",
                "payment validation",
                "straight through processing"
            ]
        },
        "compliance_screening": {
            "priority": "critical",
            "keywords": [
                "OFAC screening",
                "sanctions screening",
                "Fircosoft",
                "watchlist screening",
                "party validation",
                "account validation",
                "BIC validation",
                "IBAN validation",
                "AML checks",
                "KYC validation"
            ]
        },
        "networks_standards": {
            "priority": "high",
            "keywords": [
                "ISO20022",
                "pacs.008",
                "pacs.009",
                "pain.001",
                "camt.053",
                "MT103",
                "MT202",
                "Fedwire",
                "CHIPS",
                "SWIFT",
                "ACH"
            ]
        },
        "systems_platforms": {
            "priority": "high",
            "keywords": [
                "GFIMTS",
                "PSTW",
                "OPICS",
                "AXCIS",
                "Newton",
                "RCS",
                "TMS",
                "GABS",
                "IntelliTracs",
                "PEGA"
            ]
        },
        "business_processes": {
            "priority": "medium",
            "keywords": [
                "ACE repair",
                "exception handling",
                "payment reconciliation",
                "intraday liquidity",
                "cutoff processing",
                "EOD processing",
                "SOD releasing",
                "warehousing",
                "preadvising"
            ]
        }
    }
    
    return example_config


def main():
    """Example usage with custom keywords - WITH SEARCH"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Wire Processing Indexer with Custom Keyword Mapping"
    )
    parser.add_argument("--pdf-folder", help="Path to PDF folder")
    parser.add_argument("--index-path", default="./wire_index", help="Index storage path")
    parser.add_argument("--keywords", help="Path to custom keywords YAML/JSON file")
    parser.add_argument("--create-example", action="store_true",
                       help="Create example keywords.yaml file")
    parser.add_argument("--action", choices=["index", "search", "stats", "coverage"], 
                       default="index", help="Action to perform")
    
    # Search options
    parser.add_argument("--query", help="Search query")
    parser.add_argument("--category", help="Filter by category")
    parser.add_argument("--keyword", help="Search by specific keyword")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.create_example:
        example = create_example_keyword_config()
        if YAML_AVAILABLE:
            with open("keywords_example.yaml", 'w') as f:
                yaml.dump(example, f, default_flow_style=False, sort_keys=False)
            print("✓ Created keywords_example.yaml")
        else:
            with open("keywords_example.json", 'w') as f:
                json.dump(example, f, indent=2)
            print("✓ Created keywords_example.json (PyYAML not available)")
        return
    
    print("=" * 70)
    print("Wire Processing Indexer - Custom Keyword Mapping")
    print("=" * 70)
    
    if args.action == "index":
        if not args.pdf_folder:
            print("Error: --pdf-folder required for indexing")
            return
            
        indexer = KeywordMappedIndexer(
            pdf_folder=args.pdf_folder,
            index_path=args.index_path,
            keyword_config=args.keywords,
            use_embeddings=False
        )
        
        stats = indexer.index_pdfs()
        indexer.get_statistics()
        
    elif args.action == "search":
        if not args.query and not args.category and not args.keyword:
            print("Error: --query, --category, or --keyword required for search")
            return
        
        searcher = KeywordMappedSearcher(
            index_path=args.index_path,
            keyword_config=args.keywords
        )
        
        if args.keyword:
            # Search by specific keyword
            results = searcher.search_by_keyword(args.keyword, args.top_k)
            print(f"\n{'='*80}")
            print(f"Documents containing keyword: {args.keyword}")
            print(f"{'='*80}\n")
        elif args.category:
            # Search by category
            results = searcher.search_by_category(args.category, args.top_k)
            print(f"\n{'='*80}")
            print(f"Documents in category: {args.category}")
            print(f"{'='*80}\n")
        else:
            # Regular search with optional filters
            results = searcher.search(
                query=args.query,
                top_k=args.top_k,
                category_filter=args.category if args.category else None,
                keyword_filter=args.keyword if args.keyword else None,
                verbose=args.verbose
            )
        
        # Display results
        print(f"Found {len(results)} results\n")
        
        for i, result in enumerate(results, 1):
            score = result.get('normalized_score', result.get('bm25_score', 0))
            print(f"[{i}] Score: {score:.3f}")
            print(f"Source: {result['source_file']}")
            
            if result.get('categories'):
                print(f"Categories: {', '.join(result['categories'][:3])}")
            
            if result.get('top_keywords'):
                print(f"Keywords: {', '.join(result['top_keywords'][:5])}")
            
            print(f"Text: {result['text'][:200]}...")
            print("-" * 80)
    
    elif args.action == "stats":
        indexer = KeywordMappedIndexer(
            pdf_folder=".",  # Not used for stats
            index_path=args.index_path,
            keyword_config=args.keywords,
            use_embeddings=False
        )
        indexer.get_statistics()
    
    elif args.action == "coverage":
        searcher = KeywordMappedSearcher(
            index_path=args.index_path,
            keyword_config=args.keywords
        )
        
        coverage = searcher.get_keyword_coverage()
        
        print("\n" + "=" * 70)
        print("KEYWORD COVERAGE REPORT")
        print("=" * 70)
        
        for category, data in coverage.items():
            print(f"\n{category}:")
            print(f"  Coverage: {data['found_keywords']}/{data['total_keywords']} keywords ({data['coverage_percent']:.1f}%)")
            
            if data['keyword_details']:
                print(f"  Found keywords:")
                for kw, count in sorted(data['keyword_details'], key=lambda x: x[1], reverse=True)[:5]:
                    print(f"    • {kw} - {count} chunks")
        
        print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

print("\n" + "=" * 70)
print("✓ Custom Keyword Mapping Indexer Ready")
print("  • Load your own domain keywords")
print("  • Assign priorities (critical/high/medium/low)")
print("  • Map documents to keyword categories")
print("  • Track keyword coverage across documents")
print("  • Search by query, keyword, or category")
print("=" * 70)
