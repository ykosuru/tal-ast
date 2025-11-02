"""
Universal File Indexer with LSI/SVD (Eigenvector-Based Semantic Search)
Combines: BM25 + Stemming + Latent Semantic Indexing for optimal results

LSI Benefits:
- Semantic matching: "payment" matches "transaction" automatically
- Dimensionality reduction: 10,000 terms → 300 dimensions (faster search)
- Discovers synonyms from usage patterns
- Handles polysemy (same word, different meanings)
"""

import os
import json
import math
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional, Tuple, Set
import hashlib
import re

# Core libraries
import numpy as np
from scipy.sparse import csr_matrix
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
    print("Warning: PDF libraries not available")


# File type mappings
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
        """Stem all words in text"""
        words = re.findall(r'\b[\w-]+\b', text)
        return ' '.join([self.stem(w) for w in words])


class LatentSemanticIndexer:
    """
    Latent Semantic Indexing using SVD (Singular Value Decomposition)
    
    Theory:
    - Term-Document matrix: A (terms × documents)
    - SVD: A = U * Σ * V^T
    - U: term-concept matrix (eigenvectors of A*A^T)
    - Σ: singular values (strength of concepts)
    - V^T: concept-document matrix (eigenvectors of A^T*A)
    
    Benefits:
    - Reduces from 10k terms to 300 concepts (faster)
    - Captures semantic relationships (synonyms, related terms)
    - Noise reduction (low singular values discarded)
    """
    
    def __init__(
        self,
        n_components: int = 300,
        min_df: int = 2,
        max_df: float = 0.8
    ):
        """
        Initialize LSI
        
        Args:
            n_components: Number of latent concepts (dimensions after SVD)
            min_df: Minimum document frequency (ignore rare terms)
            max_df: Maximum document frequency (ignore common terms like "the")
        """
        self.n_components = n_components
        self.min_df = min_df
        self.max_df = max_df
        
        # Will be populated during fit
        self.vocabulary_ = {}  # term -> index
        self.idf_ = {}  # term -> IDF score
        self.U = None  # term-concept matrix
        self.sigma = None  # singular values
        self.Vt = None  # concept-document matrix
        
        self.n_documents = 0
        self.n_terms = 0
    
    def build_vocabulary(self, documents: List[str]) -> Dict[str, int]:
        """
        Build vocabulary with document frequency filtering
        
        Returns:
            vocabulary: {term: index}
        """
        # Count document frequencies
        df = Counter()
        for doc in documents:
            unique_terms = set(doc.split())
            df.update(unique_terms)
        
        self.n_documents = len(documents)
        
        # Filter by min_df and max_df
        max_doc_freq = int(self.max_df * self.n_documents)
        
        vocabulary = {}
        idx = 0
        for term, doc_freq in df.items():
            if self.min_df <= doc_freq <= max_doc_freq:
                vocabulary[term] = idx
                # Compute IDF
                self.idf_[term] = math.log(self.n_documents / doc_freq)
                idx += 1
        
        self.n_terms = len(vocabulary)
        print(f"  Vocabulary: {self.n_terms} terms (filtered from {len(df)} total)")
        
        return vocabulary
    
    def build_tfidf_matrix(self, documents: List[str]) -> csr_matrix:
        """
        Build TF-IDF term-document matrix
        
        Returns:
            Sparse matrix of shape (n_terms, n_documents)
        """
        rows = []
        cols = []
        data = []
        
        for doc_idx, doc in enumerate(documents):
            # Count term frequencies in this document
            tf = Counter(doc.split())
            doc_length = sum(tf.values())
            
            for term, count in tf.items():
                if term in self.vocabulary_:
                    term_idx = self.vocabulary_[term]
                    
                    # TF-IDF score
                    tf_score = count / doc_length
                    idf_score = self.idf_.get(term, 0)
                    tfidf = tf_score * idf_score
                    
                    rows.append(term_idx)
                    cols.append(doc_idx)
                    data.append(tfidf)
        
        # Create sparse matrix
        matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(self.n_terms, self.n_documents)
        )
        
        print(f"  TF-IDF matrix: {matrix.shape} ({matrix.nnz:,} non-zero entries)")
        return matrix
    
    def fit(self, documents: List[str]):
        """
        Fit LSI model using SVD
        
        This computes the eigenvectors:
        - U contains term eigenvectors (semantic term relationships)
        - V contains document eigenvectors (semantic document relationships)
        """
        print("\n=== Computing LSI (SVD Decomposition) ===")
        
        # Build vocabulary and TF-IDF matrix
        self.vocabulary_ = self.build_vocabulary(documents)
        tfidf_matrix = self.build_tfidf_matrix(documents)
        
        # Determine actual n_components (can't exceed matrix dimensions)
        actual_components = min(
            self.n_components,
            min(tfidf_matrix.shape) - 1
        )
        
        if actual_components < self.n_components:
            print(f"  Reducing n_components from {self.n_components} to {actual_components}")
            self.n_components = actual_components
        
        # Compute SVD (Singular Value Decomposition)
        print(f"  Computing SVD with {self.n_components} components...")
        print(f"  This finds the {self.n_components} most important 'concepts' (eigenvectors)")
        
        # svds computes the k largest singular values/vectors
        # Returns: U, sigma, Vt where A ≈ U @ diag(sigma) @ Vt
        self.U, self.sigma, self.Vt = svds(
            tfidf_matrix,
            k=self.n_components,
            which='LM'  # Largest Magnitude
        )
        
        # Sort by singular values (descending)
        idx = np.argsort(self.sigma)[::-1]
        self.sigma = self.sigma[idx]
        self.U = self.U[:, idx]
        self.Vt = self.Vt[idx, :]
        
        print(f"  ✓ SVD complete!")
        print(f"  Top 5 singular values: {self.sigma[:5]}")
        print(f"  Explained variance ratio: {self.sigma[:10].sum() / self.sigma.sum():.2%} (top 10)")
        
        return self
    
    def transform_documents(self, documents: List[str]) -> np.ndarray:
        """
        Transform documents to concept space (dimensionality reduction)
        
        Returns:
            Document vectors in concept space: (n_documents, n_components)
        """
        # Build TF-IDF vectors for documents
        doc_vectors = []
        
        for doc in documents:
            tf = Counter(doc.split())
            doc_length = sum(tf.values())
            
            # Create TF-IDF vector
            vec = np.zeros(self.n_terms)
            for term, count in tf.items():
                if term in self.vocabulary_:
                    term_idx = self.vocabulary_[term]
                    tf_score = count / doc_length
                    idf_score = self.idf_.get(term, 0)
                    vec[term_idx] = tf_score * idf_score
            
            doc_vectors.append(vec)
        
        doc_vectors = np.array(doc_vectors)
        
        # Project to concept space: doc_concept = doc_tfidf @ U
        doc_concept = doc_vectors @ self.U
        
        return doc_concept
    
    def transform_query(self, query: str) -> np.ndarray:
        """
        Transform query to concept space
        
        Returns:
            Query vector in concept space: (n_components,)
        """
        # Tokenize query
        terms = query.split()
        tf = Counter(terms)
        query_length = len(terms)
        
        # Build TF-IDF vector
        vec = np.zeros(self.n_terms)
        for term, count in tf.items():
            if term in self.vocabulary_:
                term_idx = self.vocabulary_[term]
                tf_score = count / query_length
                idf_score = self.idf_.get(term, 0)
                vec[term_idx] = tf_score * idf_score
        
        # Project to concept space
        query_concept = vec @ self.U
        
        return query_concept
    
    def get_similar_terms(self, term: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Find semantically similar terms using eigenvector similarity
        
        This uses the term-concept matrix (U) to find terms that appear
        in similar contexts (similar eigenvector patterns)
        """
        if term not in self.vocabulary_:
            return []
        
        term_idx = self.vocabulary_[term]
        term_vector = self.U[term_idx, :]
        
        # Compute cosine similarity with all terms
        similarities = []
        for other_term, other_idx in self.vocabulary_.items():
            if other_term == term:
                continue
            
            other_vector = self.U[other_idx, :]
            
            # Cosine similarity
            cos_sim = np.dot(term_vector, other_vector) / (
                np.linalg.norm(term_vector) * np.linalg.norm(other_vector) + 1e-10
            )
            
            similarities.append((other_term, cos_sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def save(self, path: Path):
        """Save LSI model to disk"""
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
        """Load LSI model from disk"""
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
        """Universal file extraction"""
        file_type = self.get_file_type(file_path)
        
        if file_type == 'pdf' and self.pdf_available:
            return self._extract_pdf(file_path)
        else:
            return self._extract_text(file_path)
    
    def _extract_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract text from PDF"""
        content = {"text": "", "file_type": "pdf", "line_count": 0}
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    content["text"] += page_text + "\n"
        except Exception as e:
            print(f"Error: {e}")
        
        content["line_count"] = content["text"].count('\n')
        return content
    
    def _extract_text(self, text_path: Path) -> Dict[str, Any]:
        """Extract text from text/code files"""
        content = {"text": "", "file_type": "text", "line_count": 0}
        
        encodings = ['utf-8', 'latin-1', 'cp1252']
        for encoding in encodings:
            try:
                with open(text_path, 'r', encoding=encoding) as f:
                    content["text"] = f.read()
                break
            except UnicodeDecodeError:
                continue
        
        content["line_count"] = content["text"].count('\n')
        return content


class FastKeywordExtractor:
    """Extract keywords with stemming"""
    
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
        """Extract keywords with scores"""
        if not text or len(text.strip()) < 10:
            return []
        
        keyword_scores = defaultdict(float)
        
        words = re.findall(r'\b[\w-]+\b', text.lower())
        if self.stemmer and self.stemmer.use_stemming:
            stemmed_words = [self.stemmer.stem(w) for w in words]
        else:
            stemmed_words = words
        
        # Score important terms
        for term_stem in self.important_terms_stemmed:
            count = stemmed_words.count(term_stem)
            if count > 0:
                keyword_scores[term_stem] += 3.0 * math.log1p(count)
        
        # Score capitalized words
        capitalized = re.findall(r'\b[A-Z][a-z]{2,}\b', text)
        for word in capitalized:
            stem = self.stemmer.stem(word.lower()) if self.stemmer else word.lower()
            if stem not in self.stopwords:
                keyword_scores[stem] += 1.5
        
        filtered = [(kw, score) for kw, score in keyword_scores.items() 
                   if len(kw) > 2 and kw not in self.stopwords]
        
        filtered.sort(key=lambda x: x[1], reverse=True)
        return filtered[:max_keywords]


class HybridSearchEngine:
    """
    Hybrid search combining:
    1. BM25 (keyword matching)
    2. LSI (semantic/eigenvector matching)
    3. Stemming (word variation handling)
    """
    
    def __init__(
        self,
        files_folder: str,
        index_path: str = "./hybrid_index",
        file_extensions: Optional[List[str]] = None,
        use_stemming: bool = True,
        use_lsi: bool = True,
        lsi_components: int = 300
    ):
        self.files_folder = Path(files_folder)
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        self.use_stemming = use_stemming
        self.use_lsi = use_lsi
        self.lsi_components = lsi_components
        
        # File extensions
        if file_extensions:
            self.file_extensions = [ext if ext.startswith('.') else f'.{ext}' 
                                   for ext in file_extensions]
        else:
            self.file_extensions = list(ALL_SUPPORTED_EXTENSIONS)
        
        # Components
        self.file_extractor = UniversalFileExtractor()
        self.stemmer = TextStemmer(use_stemming=use_stemming)
        self.keyword_extractor = FastKeywordExtractor(stemmer=self.stemmer)
        
        # Will be populated
        self.metadata_store = []
        self.document_store = []
        self.document_store_stemmed = []
        self.bm25 = None
        self.lsi = None
        self.lsi_doc_vectors = None
    
    def index_files(self):
        """Index all files with BM25 + LSI"""
        print(f"Scanning folder: {self.files_folder}")
        
        # Find all files
        files = []
        for ext in self.file_extensions:
            found = list(self.files_folder.glob(f"**/*{ext}"))
            files.extend(found)
        
        print(f"Found {len(files)} files")
        
        if not files:
            print("No files found!")
            return None
        
        print("\n=== Phase 1: Extracting Content ===")
        all_chunks = []
        
        for idx, file_path in enumerate(files, 1):
            print(f"[{idx}/{len(files)}] {file_path.name}")
            
            content = self.file_extractor.extract(file_path)
            if not content.get("text") or len(content["text"].strip()) < 10:
                continue
            
            # Simple chunking
            text = content["text"]
            words = text.split()
            
            chunk_size = 512
            for i in range(0, len(words), chunk_size - 128):
                chunk_text = " ".join(words[i:i + chunk_size])
                if len(chunk_text.strip()) < 50:
                    continue
                
                keywords = self.keyword_extractor.extract(chunk_text)
                
                all_chunks.append({
                    "text": chunk_text,
                    "source_file": str(file_path.name),
                    "file_type": content.get("file_type", "unknown"),
                    "keywords": [kw for kw, _ in keywords[:10]]
                })
        
        print(f"\nTotal chunks: {len(all_chunks)}")
        
        # Store documents
        for chunk in all_chunks:
            self.metadata_store.append({
                "source_file": chunk["source_file"],
                "file_type": chunk["file_type"],
                "keywords": chunk["keywords"]
            })
            self.document_store.append(chunk["text"])
            
            if self.use_stemming:
                stemmed = self.stemmer.stem_text(chunk["text"])
                self.document_store_stemmed.append(stemmed)
        
        # Build BM25 index
        print("\n=== Phase 2: Building BM25 Index ===")
        docs_for_bm25 = self.document_store_stemmed if self.use_stemming else self.document_store
        tokenized = [doc.split() for doc in docs_for_bm25]
        self.bm25 = BM25Okapi(tokenized)
        print("✓ BM25 index complete")
        
        # Build LSI index (eigenvector approach)
        if self.use_lsi:
            print(f"\n=== Phase 3: Building LSI Index (Eigenvectors) ===")
            self.lsi = LatentSemanticIndexer(
                n_components=self.lsi_components,
                min_df=2,
                max_df=0.8
            )
            
            self.lsi.fit(docs_for_bm25)
            
            # Transform all documents to concept space
            print("  Transforming documents to concept space...")
            self.lsi_doc_vectors = self.lsi.transform_documents(docs_for_bm25)
            print(f"  ✓ Document vectors: {self.lsi_doc_vectors.shape}")
        
        # Save everything
        print("\n=== Saving Index ===")
        
        with open(self.index_path / "metadata.pkl", 'wb') as f:
            pickle.dump(self.metadata_store, f)
        
        with open(self.index_path / "documents.pkl", 'wb') as f:
            pickle.dump(self.document_store, f)
        
        with open(self.index_path / "documents_stemmed.pkl", 'wb') as f:
            pickle.dump(self.document_store_stemmed, f)
        
        with open(self.index_path / "bm25.pkl", 'wb') as f:
            pickle.dump(self.bm25, f)
        
        if self.use_lsi:
            self.lsi.save(self.index_path / "lsi_model.pkl")
            np.save(self.index_path / "lsi_doc_vectors.npy", self.lsi_doc_vectors)
        
        stats = {
            "total_chunks": len(all_chunks),
            "total_files": len(files),
            "use_stemming": self.use_stemming,
            "use_lsi": self.use_lsi,
            "lsi_components": self.lsi_components if self.use_lsi else 0
        }
        
        with open(self.index_path / "stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        print("✓ Index saved")
        return stats
    
    def search(
        self,
        query: str,
        top_k: int = 20,
        use_lsi: bool = True,
        bm25_weight: float = 0.5,
        lsi_weight: float = 0.5,
        verbose: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining BM25 and LSI
        
        Args:
            query: Search query
            top_k: Number of results
            use_lsi: Use LSI semantic matching
            bm25_weight: Weight for BM25 scores (0-1)
            lsi_weight: Weight for LSI scores (0-1)
            verbose: Show details
        """
        if verbose:
            print(f"\nQuery: {query}")
            print(f"Hybrid search: BM25 ({bm25_weight:.1f}) + LSI ({lsi_weight:.1f})")
        
        # Prepare query
        query_processed = self.stemmer.stem_text(query) if self.use_stemming else query
        query_tokens = query_processed.split()
        
        # BM25 scores
        bm25_scores = self.bm25.get_scores(query_tokens)
        bm25_scores = bm25_scores / (bm25_scores.max() + 1e-10)  # Normalize
        
        if verbose:
            print(f"BM25: max={bm25_scores.max():.3f}, mean={bm25_scores.mean():.3f}")
        
        # LSI scores (semantic similarity using eigenvectors)
        if use_lsi and self.lsi is not None:
            query_vector = self.lsi.transform_query(query_processed)
            
            # Cosine similarity with all documents
            lsi_scores = np.zeros(len(self.document_store))
            for idx in range(len(self.document_store)):
                doc_vector = self.lsi_doc_vectors[idx]
                
                cos_sim = np.dot(query_vector, doc_vector) / (
                    np.linalg.norm(query_vector) * np.linalg.norm(doc_vector) + 1e-10
                )
                lsi_scores[idx] = cos_sim
            
            # Normalize to [0, 1]
            lsi_scores = (lsi_scores - lsi_scores.min()) / (lsi_scores.max() - lsi_scores.min() + 1e-10)
            
            if verbose:
                print(f"LSI: max={lsi_scores.max():.3f}, mean={lsi_scores.mean():.3f}")
        else:
            lsi_scores = np.zeros(len(self.document_store))
        
        # Combine scores
        combined_scores = bm25_weight * bm25_scores + lsi_weight * lsi_scores
        
        # Get top results
        top_indices = np.argsort(combined_scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if combined_scores[idx] < 0.01:
                break
            
            metadata = self.metadata_store[idx]
            
            results.append({
                "text": self.document_store[idx],
                "source_file": metadata["source_file"],
                "file_type": metadata["file_type"],
                "keywords": metadata["keywords"],
                "bm25_score": float(bm25_scores[idx]),
                "lsi_score": float(lsi_scores[idx]),
                "combined_score": float(combined_scores[idx])
            })
        
        return results
    
    def find_similar_terms(self, term: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Find semantically similar terms using LSI eigenvectors"""
        if not self.lsi:
            return []
        
        return self.lsi.get_similar_terms(term, top_k)


class HybridSearcher:
    """Load and search existing index"""
    
    def __init__(self, index_path: str = "./hybrid_index"):
        self.index_path = Path(index_path)
        
        # Load stats
        with open(self.index_path / "stats.json", 'r') as f:
            self.stats = json.load(f)
        
        self.use_stemming = self.stats['use_stemming']
        self.use_lsi = self.stats['use_lsi']
        
        # Load components
        self.stemmer = TextStemmer(use_stemming=self.use_stemming)
        
        with open(self.index_path / "metadata.pkl", 'rb') as f:
            self.metadata_store = pickle.load(f)
        
        with open(self.index_path / "documents.pkl", 'rb') as f:
            self.document_store = pickle.load(f)
        
        with open(self.index_path / "bm25.pkl", 'rb') as f:
            self.bm25 = pickle.load(f)
        
        if self.use_lsi:
            self.lsi = LatentSemanticIndexer.load(self.index_path / "lsi_model.pkl")
            self.lsi_doc_vectors = np.load(self.index_path / "lsi_doc_vectors.npy")
            print(f"✓ LSI index loaded ({self.lsi.n_components} components)")
        else:
            self.lsi = None
            self.lsi_doc_vectors = None
        
        print(f"✓ Index loaded: {len(self.document_store)} documents")
    
    def search(
        self,
        query: str,
        top_k: int = 20,
        use_lsi: bool = True,
        bm25_weight: float = 0.5,
        lsi_weight: float = 0.5,
        verbose: bool = False
    ) -> List[Dict[str, Any]]:
        """Search with BM25 + LSI hybrid"""
        
        # Prepare query
        query_processed = self.stemmer.stem_text(query) if self.use_stemming else query
        query_tokens = query_processed.split()
        
        # BM25 scores
        bm25_scores = self.bm25.get_scores(query_tokens)
        bm25_scores = bm25_scores / (bm25_scores.max() + 1e-10)
        
        # LSI scores
        if use_lsi and self.lsi is not None:
            query_vector = self.lsi.transform_query(query_processed)
            
            lsi_scores = np.zeros(len(self.document_store))
            for idx in range(len(self.document_store)):
                doc_vector = self.lsi_doc_vectors[idx]
                cos_sim = np.dot(query_vector, doc_vector) / (
                    np.linalg.norm(query_vector) * np.linalg.norm(doc_vector) + 1e-10
                )
                lsi_scores[idx] = cos_sim
            
            lsi_scores = (lsi_scores - lsi_scores.min()) / (lsi_scores.max() - lsi_scores.min() + 1e-10)
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
            results.append({
                "text": self.document_store[idx],
                "source_file": metadata["source_file"],
                "file_type": metadata["file_type"],
                "bm25_score": float(bm25_scores[idx]),
                "lsi_score": float(lsi_scores[idx]),
                "combined_score": float(combined_scores[idx])
            })
        
        return results
    
    def find_similar_terms(self, term: str, top_k: int = 10):
        """Find similar terms using eigenvectors"""
        if not self.lsi:
            return []
        return self.lsi.get_similar_terms(term, top_k)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Hybrid Search with LSI/SVD (Eigenvector Approach)"
    )
    parser.add_argument("--folder", required=True)
    parser.add_argument("--index-path", default="./hybrid_index")
    parser.add_argument("--action", choices=["index", "search", "similar"], default="index")
    parser.add_argument("--query", help="Search query")
    parser.add_argument("--term", help="Term for similarity search")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--bm25-weight", type=float, default=0.5)
    parser.add_argument("--lsi-weight", type=float, default=0.5)
    parser.add_argument("--disable-lsi", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Hybrid Search Engine: BM25 + LSI (Eigenvector-Based)")
    print("=" * 70)
    
    if args.action == "index":
        engine = HybridSearchEngine(
            files_folder=args.folder,
            index_path=args.index_path,
            use_lsi=not args.disable_lsi
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
            use_lsi=not args.disable_lsi,
            bm25_weight=args.bm25_weight,
            lsi_weight=args.lsi_weight,
            verbose=args.verbose
        )
        
        print(f"\nFound {len(results)} results\n")
        for i, result in enumerate(results, 1):
            print(f"[{i}] {result['source_file']}")
            print(f"BM25: {result['bm25_score']:.3f} | LSI: {result['lsi_score']:.3f} | Combined: {result['combined_score']:.3f}")
            print(f"{result['text'][:200]}...")
            print("-" * 70)
    
    elif args.action == "similar":
        if not args.term:
            print("Error: --term required")
            return
        
        searcher = HybridSearcher(index_path=args.index_path)
        similar = searcher.find_similar_terms(args.term, args.top_k)
        
        print(f"\nTerms similar to '{args.term}':\n")
        for term, score in similar:
            print(f"  {term:30s} - {score:.3f}")


if __name__ == "__main__":
    main()

print("\n" + "=" * 70)
print("✓ Hybrid Search with LSI/SVD (Eigenvector-Based)")
print("  • BM25: Fast keyword matching")
print("  • LSI/SVD: Semantic matching using eigenvectors")
print("  • Discovers synonyms automatically from usage patterns")
print("  • Dimensionality reduction: 10k terms → 300 concepts")
print("=" * 70)
