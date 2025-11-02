"""
Memory-Efficient Hybrid Search with Line Number Tracking
Fixes: KILLED: 9 (OOM) issue by using batching and aggressive memory management
"""

import os
import json
import math
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional, Tuple, Set
import hashlib
import re
import gc

# Core libraries
import numpy as np
from scipy.sparse import csr_matrix, save_npz, load_npz
from scipy.sparse.linalg import svds
import pickle

# Stemming
try:
    from nltk.stem import PorterStemmer
    STEMMER_AVAILABLE = True
except ImportError:
    STEMMER_AVAILABLE = False
    print("Warning: nltk not available, stemming disabled")

# BM25
from rank_bm25 import BM25Okapi

# PDF processing (optional)
try:
    import PyPDF2
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False


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
    'text': ['.txt', '.md', '.rst', '.log', '.text'],
    'config': ['.json', '.xml', '.yaml', '.yml', '.toml', '.ini', '.cfg'],
    'markup': ['.html', '.htm', '.xhtml', '.css', '.scss', '.sass']
}

ALL_SUPPORTED_EXTENSIONS = set()
for extensions in SUPPORTED_EXTENSIONS.values():
    ALL_SUPPORTED_EXTENSIONS.update(extensions)


class TextStemmer:
    """Porter Stemmer with caching"""
    
    def __init__(self, use_stemming: bool = True):
        self.use_stemming = use_stemming and STEMMER_AVAILABLE
        
        if self.use_stemming:
            self.stemmer = PorterStemmer()
            print("✓ Stemming enabled")
        else:
            self.stemmer = None
        
        self._stem_cache = {}
    
    def stem(self, word: str) -> str:
        if not self.use_stemming or not word:
            return word.lower()
        
        word_lower = word.lower()
        if word_lower in self._stem_cache:
            return self._stem_cache[word_lower]
        
        stemmed = self.stemmer.stem(word_lower)
        self._stem_cache[word_lower] = stemmed
        return stemmed
    
    def stem_text(self, text: str) -> str:
        words = re.findall(r'\b[\w-]+\b', text)
        return ' '.join([self.stem(w) for w in words])


class MemoryEfficientLSI:
    """
    Memory-efficient LSI using batched processing and aggressive filtering
    Prevents OOM kills by:
    1. Aggressive vocabulary filtering
    2. Batched matrix construction
    3. Memory cleanup
    """
    
    def __init__(
        self,
        n_components: int = 200,  # Reduced from 300
        min_df: int = 3,           # Increased from 2
        max_df: float = 0.7,       # Reduced from 0.8
        max_features: int = 5000   # NEW: Cap vocabulary size
    ):
        self.n_components = n_components
        self.min_df = min_df
        self.max_df = max_df
        self.max_features = max_features
        
        self.vocabulary_ = {}
        self.idf_ = {}
        self.U = None
        self.sigma = None
        self.Vt = None
        
        self.n_documents = 0
        self.n_terms = 0
    
    def build_vocabulary(self, documents: List[str]) -> Dict[str, int]:
        """Build vocabulary with AGGRESSIVE filtering to save memory"""
        print("  Building vocabulary with aggressive filtering...")
        
        df = Counter()
        for doc in documents:
            unique_terms = set(doc.split())
            df.update(unique_terms)
        
        self.n_documents = len(documents)
        max_doc_freq = int(self.max_df * self.n_documents)
        
        # Filter by document frequency
        filtered = {
            term: freq for term, freq in df.items()
            if self.min_df <= freq <= max_doc_freq
        }
        
        print(f"    After df filter: {len(filtered)} terms (from {len(df)})")
        
        # AGGRESSIVE: Keep only top N by frequency
        if len(filtered) > self.max_features:
            sorted_terms = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
            filtered = dict(sorted_terms[:self.max_features])
            print(f"    Limiting to top {self.max_features} terms")
        
        # Build final vocabulary
        vocabulary = {}
        for idx, (term, doc_freq) in enumerate(filtered.items()):
            vocabulary[term] = idx
            self.idf_[term] = math.log(self.n_documents / doc_freq)
        
        self.n_terms = len(vocabulary)
        print(f"  Final vocabulary: {self.n_terms} terms")
        
        return vocabulary
    
    def build_tfidf_matrix_batched(
        self,
        documents: List[str],
        batch_size: int = 1000
    ) -> csr_matrix:
        """Build TF-IDF matrix in batches to save memory"""
        print(f"  Building TF-IDF matrix in batches of {batch_size}...")
        
        all_rows = []
        all_cols = []
        all_data = []
        
        n_batches = (len(documents) + batch_size - 1) // batch_size
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(documents))
            batch_docs = documents[start_idx:end_idx]
            
            # Process batch
            for doc_offset, doc in enumerate(batch_docs):
                doc_idx = start_idx + doc_offset
                
                tf = Counter(doc.split())
                doc_length = sum(tf.values())
                
                for term, count in tf.items():
                    if term in self.vocabulary_:
                        term_idx = self.vocabulary_[term]
                        
                        tf_score = count / doc_length
                        idf_score = self.idf_.get(term, 0)
                        tfidf = tf_score * idf_score
                        
                        all_rows.append(term_idx)
                        all_cols.append(doc_idx)
                        all_data.append(tfidf)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"    Processed {end_idx}/{len(documents)} documents")
                gc.collect()  # Force garbage collection
        
        # Create sparse matrix
        matrix = csr_matrix(
            (all_data, (all_rows, all_cols)),
            shape=(self.n_terms, self.n_documents)
        )
        
        print(f"  TF-IDF matrix: {matrix.shape} ({matrix.nnz:,} non-zero entries)")
        print(f"  Memory usage: {matrix.data.nbytes / 1024 / 1024:.1f} MB")
        
        # Cleanup
        del all_rows, all_cols, all_data
        gc.collect()
        
        return matrix
    
    def fit(self, documents: List[str]):
        """Fit LSI with memory-efficient processing"""
        print("\n=== Computing Memory-Efficient LSI ===")
        
        # Build vocabulary
        self.vocabulary_ = self.build_vocabulary(documents)
        
        # Build TF-IDF matrix in batches
        tfidf_matrix = self.build_tfidf_matrix_batched(documents, batch_size=1000)
        
        # Reduce n_components if needed
        actual_components = min(
            self.n_components,
            min(tfidf_matrix.shape) - 1
        )
        
        if actual_components < self.n_components:
            print(f"  Reducing n_components: {self.n_components} → {actual_components}")
            self.n_components = actual_components
        
        # Check available memory before SVD
        import psutil
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024**3)
        
        print(f"\n  Available memory: {available_gb:.1f} GB")
        print(f"  Computing SVD with {self.n_components} components...")
        
        if available_gb < 2.0:
            print("  ⚠️  WARNING: Low memory! Consider reducing n_components or docs")
        
        try:
            # Compute SVD
            self.U, self.sigma, self.Vt = svds(
                tfidf_matrix,
                k=self.n_components,
                which='LM'
            )
            
            # Sort by singular values
            idx = np.argsort(self.sigma)[::-1]
            self.sigma = self.sigma[idx]
            self.U = self.U[:, idx]
            self.Vt = self.Vt[idx, :]
            
            print(f"  ✓ SVD complete!")
            print(f"  Top 5 singular values: {self.sigma[:5]}")
            
        except MemoryError as e:
            print(f"\n  ❌ SVD failed due to memory error!")
            print(f"  Error: {e}")
            print(f"\n  Solutions:")
            print(f"  1. Reduce --lsi-components (try 100 or 50)")
            print(f"  2. Use --max-docs to limit corpus size")
            print(f"  3. Use --disable-lsi to skip LSI entirely")
            raise
        
        # Cleanup
        del tfidf_matrix
        gc.collect()
        
        return self
    
    def transform_query(self, query: str) -> np.ndarray:
        """Transform query to concept space"""
        terms = query.split()
        tf = Counter(terms)
        query_length = len(terms)
        
        vec = np.zeros(self.n_terms)
        for term, count in tf.items():
            if term in self.vocabulary_:
                term_idx = self.vocabulary_[term]
                tf_score = count / query_length
                idf_score = self.idf_.get(term, 0)
                vec[term_idx] = tf_score * idf_score
        
        query_concept = vec @ self.U
        return query_concept
    
    def save(self, path: Path):
        """Save LSI model"""
        model_data = {
            'n_components': self.n_components,
            'vocabulary_': self.vocabulary_,
            'idf_': self.idf_,
            'U': self.U,
            'sigma': self.sigma,
            'Vt': self.Vt,
            'n_documents': self.n_documents,
            'n_terms': self.n_terms
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load(cls, path: Path):
        """Load LSI model"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        lsi = cls(n_components=model_data['n_components'])
        lsi.vocabulary_ = model_data['vocabulary_']
        lsi.idf_ = model_data['idf_']
        lsi.U = model_data['U']
        lsi.sigma = model_data['sigma']
        lsi.Vt = model_data['Vt']
        lsi.n_documents = model_data['n_documents']
        lsi.n_terms = model_data['n_terms']
        
        return lsi


class UniversalFileExtractor:
    """Extract text from files with line tracking"""
    
    def __init__(self):
        self.pdf_available = PDF_AVAILABLE
    
    def get_file_type(self, file_path: Path) -> str:
        ext = file_path.suffix.lower()
        for file_type, extensions in SUPPORTED_EXTENSIONS.items():
            if ext in extensions:
                return file_type
        return 'unknown'
    
    def extract(self, file_path: Path) -> Dict[str, Any]:
        """Extract with line tracking"""
        file_type = self.get_file_type(file_path)
        
        if file_type == 'pdf' and self.pdf_available:
            return self._extract_pdf(file_path)
        else:
            return self._extract_text(file_path)
    
    def _extract_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        content = {
            "text": "",
            "file_type": "pdf",
            "line_count": 0,
            "lines": []
        }
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    page_lines = page_text.split('\n')
                    content["lines"].extend(page_lines)
                    content["text"] += page_text + "\n"
        except Exception as e:
            print(f"Error: {e}")
        
        content["line_count"] = len(content["lines"])
        return content
    
    def _extract_text(self, text_path: Path) -> Dict[str, Any]:
        content = {
            "text": "",
            "file_type": "text",
            "line_count": 0,
            "lines": []
        }
        
        encodings = ['utf-8', 'latin-1', 'cp1252']
        for encoding in encodings:
            try:
                with open(text_path, 'r', encoding=encoding) as f:
                    content["lines"] = f.readlines()
                    content["text"] = ''.join(content["lines"])
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"Error reading {text_path}: {e}")
                break
        
        content["line_count"] = len(content["lines"])
        return content


class FastKeywordExtractor:
    """Extract keywords"""
    
    def __init__(self, stemmer: Optional[TextStemmer] = None):
        self.stemmer = stemmer
        
        self.important_terms = {
            'credit', 'debit', 'party', 'determination', 'validation',
            'payment', 'wire', 'transfer', 'transaction', 'settlement',
            'clearing', 'posting', 'execution', 'routing', 'screening',
            'ofac', 'sanctions', 'compliance', 'fedwire', 'chips', 'swift'
        }
        
        if self.stemmer and self.stemmer.use_stemming:
            self.important_terms_stemmed = {self.stemmer.stem(t) for t in self.important_terms}
        else:
            self.important_terms_stemmed = self.important_terms
        
        self.stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
            'to', 'for', 'of', 'with', 'from', 'by', 'as', 'is', 'was'
        }
    
    def extract(self, text: str, max_keywords: int = 20) -> List[Tuple[str, float]]:
        if not text or len(text.strip()) < 10:
            return []
        
        keyword_scores = defaultdict(float)
        
        words = re.findall(r'\b[\w-]+\b', text.lower())
        if self.stemmer and self.stemmer.use_stemming:
            stemmed_words = [self.stemmer.stem(w) for w in words]
        else:
            stemmed_words = words
        
        for term_stem in self.important_terms_stemmed:
            count = stemmed_words.count(term_stem)
            if count > 0:
                keyword_scores[term_stem] += 3.0 * math.log1p(count)
        
        capitalized = re.findall(r'\b[A-Z][a-z]{2,}\b', text)
        for word in capitalized:
            stem = self.stemmer.stem(word.lower()) if self.stemmer else word.lower()
            if stem not in self.stopwords:
                keyword_scores[stem] += 1.5
        
        filtered = [(kw, score) for kw, score in keyword_scores.items() 
                   if len(kw) > 2 and kw not in self.stopwords]
        
        filtered.sort(key=lambda x: x[1], reverse=True)
        return filtered[:max_keywords]


class LineAwareChunker:
    """Chunk text while tracking line numbers"""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 128):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_by_lines(
        self,
        lines: List[str],
        metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Chunk by lines with line number tracking"""
        chunks = []
        
        total_words = sum(len(line.split()) for line in lines)
        avg_words_per_line = total_words / len(lines) if lines else 10
        
        lines_per_chunk = max(1, int(self.chunk_size / avg_words_per_line))
        lines_overlap = max(1, int(self.chunk_overlap / avg_words_per_line))
        
        current_line = 0
        chunk_index = 0
        
        while current_line < len(lines):
            end_line = min(current_line + lines_per_chunk, len(lines))
            chunk_lines = lines[current_line:end_line]
            
            chunk_text = ''.join(chunk_lines)
            if len(chunk_text.strip()) < 50:
                current_line = end_line
                continue
            
            chunk = {
                "text": chunk_text,
                "start_line": current_line + 1,
                "end_line": end_line,
                "chunk_index": chunk_index,
                "total_lines": len(chunk_lines),
                **metadata
            }
            
            chunks.append(chunk)
            chunk_index += 1
            
            current_line = end_line - lines_overlap
            if current_line <= 0:
                current_line = end_line
        
        return chunks


class MemoryEfficientSearchEngine:
    """
    Memory-efficient search engine with line tracking
    Prevents OOM kills!
    """
    
    def __init__(
        self,
        files_folder: str,
        index_path: str = "./efficient_index",
        file_extensions: Optional[List[str]] = None,
        use_stemming: bool = True,
        use_lsi: bool = True,
        lsi_components: int = 200,  # Reduced default
        max_docs: Optional[int] = None,  # NEW: Limit corpus size
        chunk_size: int = 512,
        chunk_overlap: int = 128
    ):
        self.files_folder = Path(files_folder)
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        self.use_stemming = use_stemming
        self.use_lsi = use_lsi
        self.lsi_components = lsi_components
        self.max_docs = max_docs
        
        if file_extensions:
            self.file_extensions = [ext if ext.startswith('.') else f'.{ext}' 
                                   for ext in file_extensions]
        else:
            self.file_extensions = list(ALL_SUPPORTED_EXTENSIONS)
        
        self.file_extractor = UniversalFileExtractor()
        self.stemmer = TextStemmer(use_stemming=use_stemming)
        self.keyword_extractor = FastKeywordExtractor(stemmer=self.stemmer)
        self.chunker = LineAwareChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        self.metadata_store = []
        self.document_store = []
        self.document_store_stemmed = []
        self.bm25 = None
        self.lsi = None
    
    def index_files(self):
        """Index with memory-efficient processing"""
        
        # Check available memory
        try:
            import psutil
            mem = psutil.virtual_memory()
            available_gb = mem.available / (1024**3)
            total_gb = mem.total / (1024**3)
            
            print(f"\n=== System Memory ===")
            print(f"Total: {total_gb:.1f} GB")
            print(f"Available: {available_gb:.1f} GB")
            print(f"Used: {(total_gb - available_gb):.1f} GB ({mem.percent}%)")
            
            if available_gb < 2.0:
                print(f"\n⚠️  WARNING: Low memory!")
                print(f"Consider:")
                print(f"  - Use --disable-lsi")
                print(f"  - Use --max-docs 1000")
                print(f"  - Reduce --lsi-components 100")
        except ImportError:
            print("Install psutil for memory monitoring: pip install psutil")
        
        print(f"\nScanning folder: {self.files_folder}")
        
        files = []
        for ext in self.file_extensions:
            found = list(self.files_folder.glob(f"**/*{ext}"))
            files.extend(found)
        
        print(f"Found {len(files)} files")
        
        if not files:
            print("No files found!")
            return None
        
        # Limit corpus size if specified
        if self.max_docs and len(files) > self.max_docs:
            print(f"⚠️  Limiting to first {self.max_docs} files to save memory")
            files = files[:self.max_docs]
        
        print("\n=== Phase 1: Extracting Content ===")
        all_chunks = []
        
        for idx, file_path in enumerate(files, 1):
            if idx % 100 == 0:
                print(f"Processed {idx}/{len(files)} files")
                gc.collect()
            
            content = self.file_extractor.extract(file_path)
            if not content.get("lines") or len(content["lines"]) < 1:
                continue
            
            file_metadata = {
                "source_file": str(file_path.name),
                "source_path": str(file_path.absolute()),
                "file_type": content.get("file_type", "unknown"),
                "total_file_lines": content.get("line_count", 0)
            }
            
            chunks = self.chunker.chunk_by_lines(content["lines"], file_metadata)
            
            for chunk in chunks:
                keywords = self.keyword_extractor.extract(chunk["text"])
                chunk["keywords"] = [kw for kw, _ in keywords[:10]]
            
            all_chunks.extend(chunks)
        
        print(f"\nTotal chunks: {len(all_chunks)}")
        
        # Store documents
        print("\n=== Phase 2: Storing Documents ===")
        for chunk in all_chunks:
            self.metadata_store.append({
                "source_file": chunk["source_file"],
                "source_path": chunk["source_path"],
                "file_type": chunk["file_type"],
                "chunk_index": chunk["chunk_index"],
                "start_line": chunk["start_line"],
                "end_line": chunk["end_line"],
                "total_lines": chunk["total_lines"],
                "total_file_lines": chunk["total_file_lines"],
                "keywords": chunk["keywords"]
            })
            self.document_store.append(chunk["text"])
            
            if self.use_stemming:
                stemmed = self.stemmer.stem_text(chunk["text"])
                self.document_store_stemmed.append(stemmed)
        
        # Build BM25
        print("\n=== Phase 3: Building BM25 Index ===")
        docs_for_bm25 = self.document_store_stemmed if self.use_stemming else self.document_store
        tokenized = [doc.split() for doc in docs_for_bm25]
        self.bm25 = BM25Okapi(tokenized)
        print("✓ BM25 index complete")
        
        # Build LSI
        if self.use_lsi:
            try:
                self.lsi = MemoryEfficientLSI(
                    n_components=self.lsi_components,
                    min_df=3,
                    max_df=0.7,
                    max_features=5000
                )
                
                self.lsi.fit(docs_for_bm25)
                
            except MemoryError as e:
                print(f"\n❌ LSI failed due to low memory")
                print(f"Continuing with BM25 only...")
                self.use_lsi = False
                self.lsi = None
        
        # Save
        print("\n=== Saving Index ===")
        
        with open(self.index_path / "metadata.pkl", 'wb') as f:
            pickle.dump(self.metadata_store, f)
        print("✓ Saved metadata")
        
        with open(self.index_path / "documents.pkl", 'wb') as f:
            pickle.dump(self.document_store, f)
        print("✓ Saved documents")
        
        with open(self.index_path / "documents_stemmed.pkl", 'wb') as f:
            pickle.dump(self.document_store_stemmed, f)
        print("✓ Saved stemmed documents")
        
        with open(self.index_path / "bm25.pkl", 'wb') as f:
            pickle.dump(self.bm25, f)
        print("✓ Saved BM25")
        
        if self.use_lsi and self.lsi:
            self.lsi.save(self.index_path / "lsi_model.pkl")
            print("✓ Saved LSI model")
        
        stats = {
            "total_chunks": len(all_chunks),
            "total_files": len(files),
            "use_stemming": self.use_stemming,
            "use_lsi": self.use_lsi and self.lsi is not None,
            "lsi_components": self.lsi_components if self.use_lsi and self.lsi else 0
        }
        
        with open(self.index_path / "stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
        print("✓ Saved statistics")
        
        print("\n✅ Indexing complete!")
        return stats


class HybridSearcher:
    """Search with line number results"""
    
    def __init__(self, index_path: str = "./efficient_index"):
        self.index_path = Path(index_path)
        
        with open(self.index_path / "stats.json", 'r') as f:
            self.stats = json.load(f)
        
        self.use_stemming = self.stats['use_stemming']
        self.use_lsi = self.stats['use_lsi']
        
        self.stemmer = TextStemmer(use_stemming=self.use_stemming)
        
        with open(self.index_path / "metadata.pkl", 'rb') as f:
            self.metadata_store = pickle.load(f)
        
        with open(self.index_path / "documents.pkl", 'rb') as f:
            self.document_store = pickle.load(f)
        
        with open(self.index_path / "bm25.pkl", 'rb') as f:
            self.bm25 = pickle.load(f)
        
        if self.use_lsi:
            try:
                self.lsi = MemoryEfficientLSI.load(self.index_path / "lsi_model.pkl")
                print(f"✓ LSI loaded ({self.lsi.n_components} components)")
            except FileNotFoundError:
                print("⚠️  LSI model not found, using BM25 only")
                self.lsi = None
                self.use_lsi = False
        else:
            self.lsi = None
        
        print(f"✓ Index loaded: {len(self.document_store)} documents")
    
    def search(
        self,
        query: str,
        top_k: int = 20,
        use_lsi: bool = True,
        bm25_weight: float = 0.7,  # Favor BM25 more
        lsi_weight: float = 0.3,
        verbose: bool = False
    ) -> List[Dict[str, Any]]:
        """Search with line numbers"""
        
        query_processed = self.stemmer.stem_text(query) if self.use_stemming else query
        query_tokens = query_processed.split()
        
        # BM25 scores
        bm25_scores = self.bm25.get_scores(query_tokens)
        bm25_scores = bm25_scores / (bm25_scores.max() + 1e-10)
        
        # LSI scores
        if use_lsi and self.lsi is not None:
            try:
                query_vector = self.lsi.transform_query(query_processed)
                
                lsi_scores = np.zeros(len(self.document_store))
                
                # Compute similarities efficiently
                for idx in range(len(self.document_store)):
                    doc_text = self.document_store_stemmed[idx] if hasattr(self, 'document_store_stemmed') else self.document_store[idx]
                    doc_vector = self.lsi.transform_query(doc_text)
                    
                    cos_sim = np.dot(query_vector, doc_vector) / (
                        np.linalg.norm(query_vector) * np.linalg.norm(doc_vector) + 1e-10
                    )
                    lsi_scores[idx] = cos_sim
                
                lsi_scores = (lsi_scores - lsi_scores.min()) / (lsi_scores.max() - lsi_scores.min() + 1e-10)
            except Exception as e:
                print(f"LSI scoring failed: {e}, using BM25 only")
                lsi_scores = np.zeros(len(self.document_store))
        else:
            lsi_scores = np.zeros(len(self.document_store))
        
        # Combine
        combined_scores = bm25_weight * bm25_scores + lsi_weight * lsi_scores
        top_indices = np.argsort(combined_scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if combined_scores[idx] < 0.01:
                break
            
            metadata = self.metadata_store[idx]
            location = f"{metadata['source_file']}:{metadata['start_line']}-{metadata['end_line']}"
            
            results.append({
                "source_file": metadata["source_file"],
                "source_path": metadata["source_path"],
                "file_type": metadata["file_type"],
                "chunk_index": metadata["chunk_index"],
                "start_line": metadata["start_line"],
                "end_line": metadata["end_line"],
                "total_lines": metadata["total_lines"],
                "location": location,
                "text": self.document_store[idx],
                "keywords": metadata.get("keywords", []),
                "bm25_score": float(bm25_scores[idx]),
                "lsi_score": float(lsi_scores[idx]),
                "combined_score": float(combined_scores[idx])
            })
        
        return results
    
    def get_surrounding_lines(
        self,
        file_path: str,
        start_line: int,
        end_line: int,
        context_lines: int = 5
    ) -> Dict[str, Any]:
        """Get chunk with surrounding context"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                all_lines = f.readlines()
            
            start_idx = max(0, start_line - 1 - context_lines)
            end_idx = min(len(all_lines), end_line + context_lines)
            
            chunk_start = max(0, start_line - 1)
            chunk_end = min(len(all_lines), end_line)
            
            return {
                'before': all_lines[start_idx:chunk_start],
                'chunk': all_lines[chunk_start:chunk_end],
                'after': all_lines[chunk_end:end_idx],
                'full_text': ''.join(all_lines[start_idx:end_idx]),
                'context_start_line': start_idx + 1,
                'context_end_line': end_idx
            }
        except Exception as e:
            print(f"Error: {e}")
            return None


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Memory-Efficient Hybrid Search (Fixes KILLED: 9)"
    )
    parser.add_argument("--folder", required=True)
    parser.add_argument("--index-path", default="./efficient_index")
    parser.add_argument("--action", choices=["index", "search"], default="index")
    parser.add_argument("--query", help="Search query")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--lsi-components", type=int, default=200, help="LSI dimensions (lower = less memory)")
    parser.add_argument("--max-docs", type=int, help="Limit corpus size to save memory")
    parser.add_argument("--disable-lsi", action="store_true", help="Skip LSI (use BM25 only)")
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Memory-Efficient Hybrid Search with Line Tracking")
    print("Fixes: KILLED: 9 (Out of Memory)")
    print("=" * 80)
    
    if args.action == "index":
        engine = MemoryEfficientSearchEngine(
            files_folder=args.folder,
            index_path=args.index_path,
            use_lsi=not args.disable_lsi,
            lsi_components=args.lsi_components,
            max_docs=args.max_docs
        )
        engine.index_files()
        
    elif args.action == "search":
        if not args.query:
            print("Error: --query required")
            return
        
        searcher = HybridSearcher(index_path=args.index_path)
        results = searcher.search(
            args.query,
            top_k=args.top_k,
            verbose=args.verbose
        )
        
        print(f"\n{'='*80}")
        print(f"Found {len(results)} results")
        print(f"{'='*80}\n")
        
        for i, result in enumerate(results, 1):
            print(f"[{i}] {result['location']}")
            print(f"    Scores: BM25={result['bm25_score']:.3f} | LSI={result['lsi_score']:.3f}")
            print(f"    Preview: {result['text'][:150]}...")
            print("-" * 80)


if __name__ == "__main__":
    main()

print("\n" + "=" * 80)
print("✓ Memory-Efficient Search (Fixes OOM Kills)")
print("  • Aggressive vocabulary filtering")
print("  • Batched matrix construction")
print("  • Memory monitoring")
print("  • Fallback to BM25 if needed")
print("=" * 80)
