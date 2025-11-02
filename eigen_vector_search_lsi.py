"""
Hybrid Search with LSI/SVD + Line Number Tracking
Returns: filename, chunk#, line numbers for precise code/doc snippet retrieval
Perfect for RAG-based code generation!
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
    """
    
    def __init__(
        self,
        n_components: int = 300,
        min_df: int = 2,
        max_df: float = 0.8
    ):
        self.n_components = n_components
        self.min_df = min_df
        self.max_df = max_df
        
        self.vocabulary_ = {}
        self.idf_ = {}
        self.U = None
        self.sigma = None
        self.Vt = None
        
        self.n_documents = 0
        self.n_terms = 0
    
    def build_vocabulary(self, documents: List[str]) -> Dict[str, int]:
        """Build vocabulary with document frequency filtering"""
        df = Counter()
        for doc in documents:
            unique_terms = set(doc.split())
            df.update(unique_terms)
        
        self.n_documents = len(documents)
        
        max_doc_freq = int(self.max_df * self.n_documents)
        
        vocabulary = {}
        idx = 0
        for term, doc_freq in df.items():
            if self.min_df <= doc_freq <= max_doc_freq:
                vocabulary[term] = idx
                self.idf_[term] = math.log(self.n_documents / doc_freq)
                idx += 1
        
        self.n_terms = len(vocabulary)
        print(f"  Vocabulary: {self.n_terms} terms (filtered from {len(df)} total)")
        
        return vocabulary
    
    def build_tfidf_matrix(self, documents: List[str]) -> csr_matrix:
        """Build TF-IDF term-document matrix"""
        rows = []
        cols = []
        data = []
        
        for doc_idx, doc in enumerate(documents):
            tf = Counter(doc.split())
            doc_length = sum(tf.values())
            
            for term, count in tf.items():
                if term in self.vocabulary_:
                    term_idx = self.vocabulary_[term]
                    
                    tf_score = count / doc_length
                    idf_score = self.idf_.get(term, 0)
                    tfidf = tf_score * idf_score
                    
                    rows.append(term_idx)
                    cols.append(doc_idx)
                    data.append(tfidf)
        
        matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(self.n_terms, self.n_documents)
        )
        
        print(f"  TF-IDF matrix: {matrix.shape} ({matrix.nnz:,} non-zero entries)")
        return matrix
    
    def fit(self, documents: List[str]):
        """Fit LSI model using SVD"""
        print("\n=== Computing LSI (SVD Decomposition) ===")
        
        self.vocabulary_ = self.build_vocabulary(documents)
        tfidf_matrix = self.build_tfidf_matrix(documents)
        
        actual_components = min(
            self.n_components,
            min(tfidf_matrix.shape) - 1
        )
        
        if actual_components < self.n_components:
            print(f"  Reducing n_components from {self.n_components} to {actual_components}")
            self.n_components = actual_components
        
        print(f"  Computing SVD with {self.n_components} components...")
        
        self.U, self.sigma, self.Vt = svds(
            tfidf_matrix,
            k=self.n_components,
            which='LM'
        )
        
        idx = np.argsort(self.sigma)[::-1]
        self.sigma = self.sigma[idx]
        self.U = self.U[:, idx]
        self.Vt = self.Vt[idx, :]
        
        print(f"  ✓ SVD complete!")
        print(f"  Top 5 singular values: {self.sigma[:5]}")
        
        return self
    
    def transform_documents(self, documents: List[str]) -> np.ndarray:
        """Transform documents to concept space"""
        doc_vectors = []
        
        for doc in documents:
            tf = Counter(doc.split())
            doc_length = sum(tf.values())
            
            vec = np.zeros(self.n_terms)
            for term, count in tf.items():
                if term in self.vocabulary_:
                    term_idx = self.vocabulary_[term]
                    tf_score = count / doc_length
                    idf_score = self.idf_.get(term, 0)
                    vec[term_idx] = tf_score * idf_score
            
            doc_vectors.append(vec)
        
        doc_vectors = np.array(doc_vectors)
        doc_concept = doc_vectors @ self.U
        
        return doc_concept
    
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
    
    def get_similar_terms(self, term: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Find semantically similar terms using eigenvector similarity"""
        if term not in self.vocabulary_:
            return []
        
        term_idx = self.vocabulary_[term]
        term_vector = self.U[term_idx, :]
        
        similarities = []
        for other_term, other_idx in self.vocabulary_.items():
            if other_term == term:
                continue
            
            other_vector = self.U[other_idx, :]
            
            cos_sim = np.dot(term_vector, other_vector) / (
                np.linalg.norm(term_vector) * np.linalg.norm(other_vector) + 1e-10
            )
            
            similarities.append((other_term, cos_sim))
        
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
        """Universal file extraction with line-by-line tracking"""
        file_type = self.get_file_type(file_path)
        
        if file_type == 'pdf' and self.pdf_available:
            return self._extract_pdf(file_path)
        else:
            return self._extract_text(file_path)
    
    def _extract_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract text from PDF"""
        content = {
            "text": "",
            "file_type": "pdf",
            "line_count": 0,
            "lines": []  # NEW: Track individual lines
        }
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    content["text"] += page_text + "\n"
                    
                    # Track lines
                    page_lines = page_text.split('\n')
                    content["lines"].extend(page_lines)
        except Exception as e:
            print(f"Error: {e}")
        
        content["line_count"] = len(content["lines"])
        return content
    
    def _extract_text(self, text_path: Path) -> Dict[str, Any]:
        """Extract text from text/code files with line tracking"""
        content = {
            "text": "",
            "file_type": "text",
            "line_count": 0,
            "lines": []  # NEW: Track individual lines
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
        
        content["line_count"] = len(content["lines"])
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


class LineAwareChunker:
    """
    Chunk text while tracking line numbers
    CRITICAL for code generation: Know exactly where each chunk comes from
    """
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 128):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_by_lines(
        self,
        lines: List[str],
        metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Chunk text by lines, maintaining line number tracking
        
        Returns chunks with:
        - text: The actual chunk text
        - start_line: Starting line number (1-indexed)
        - end_line: Ending line number (1-indexed)
        - chunk_index: Sequential chunk number
        """
        chunks = []
        
        # Estimate words per line
        total_words = sum(len(line.split()) for line in lines)
        avg_words_per_line = total_words / len(lines) if lines else 10
        
        # Calculate lines per chunk
        lines_per_chunk = max(1, int(self.chunk_size / avg_words_per_line))
        lines_overlap = max(1, int(self.chunk_overlap / avg_words_per_line))
        
        current_line = 0
        chunk_index = 0
        
        while current_line < len(lines):
            # Get chunk lines
            end_line = min(current_line + lines_per_chunk, len(lines))
            chunk_lines = lines[current_line:end_line]
            
            # Skip if too small
            chunk_text = ''.join(chunk_lines)
            if len(chunk_text.strip()) < 50:
                current_line = end_line
                continue
            
            # Create chunk with line tracking
            chunk = {
                "text": chunk_text,
                "start_line": current_line + 1,  # 1-indexed
                "end_line": end_line,  # 1-indexed
                "chunk_index": chunk_index,
                "total_lines": len(chunk_lines),
                **metadata
            }
            
            chunks.append(chunk)
            chunk_index += 1
            
            # Move to next chunk with overlap
            current_line = end_line - lines_overlap
            if current_line <= 0:
                current_line = end_line
        
        return chunks


class HybridSearchEngine:
    """
    Hybrid search with LINE NUMBER TRACKING
    Perfect for RAG-based code generation!
    """
    
    def __init__(
        self,
        files_folder: str,
        index_path: str = "./hybrid_index",
        file_extensions: Optional[List[str]] = None,
        use_stemming: bool = True,
        use_lsi: bool = True,
        lsi_components: int = 300,
        chunk_size: int = 512,
        chunk_overlap: int = 128
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
        self.chunker = LineAwareChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        # Will be populated
        self.metadata_store = []
        self.document_store = []
        self.document_store_stemmed = []
        self.bm25 = None
        self.lsi = None
        self.lsi_doc_vectors = None
    
    def index_files(self):
        """Index all files with BM25 + LSI + LINE TRACKING"""
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
        
        print("\n=== Phase 1: Extracting Content (WITH LINE TRACKING) ===")
        all_chunks = []
        
        for idx, file_path in enumerate(files, 1):
            print(f"[{idx}/{len(files)}] {file_path.name}")
            
            # Extract with line tracking
            content = self.file_extractor.extract(file_path)
            if not content.get("lines") or len(content["lines"]) < 1:
                continue
            
            # Chunk with line tracking
            file_metadata = {
                "source_file": str(file_path.name),
                "source_path": str(file_path.absolute()),
                "file_type": content.get("file_type", "unknown"),
                "total_file_lines": content.get("line_count", 0)
            }
            
            chunks = self.chunker.chunk_by_lines(content["lines"], file_metadata)
            
            # Add keywords to each chunk
            for chunk in chunks:
                keywords = self.keyword_extractor.extract(chunk["text"])
                chunk["keywords"] = [kw for kw, _ in keywords[:10]]
            
            all_chunks.extend(chunks)
            print(f"  Created {len(chunks)} chunks with line tracking")
        
        print(f"\nTotal chunks: {len(all_chunks)}")
        
        # Store documents
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
        
        # Build BM25 index
        print("\n=== Phase 2: Building BM25 Index ===")
        docs_for_bm25 = self.document_store_stemmed if self.use_stemming else self.document_store
        tokenized = [doc.split() for doc in docs_for_bm25]
        self.bm25 = BM25Okapi(tokenized)
        print("✓ BM25 index complete")
        
        # Build LSI index
        if self.use_lsi:
            print(f"\n=== Phase 3: Building LSI Index (Eigenvectors) ===")
            self.lsi = LatentSemanticIndexer(
                n_components=self.lsi_components,
                min_df=2,
                max_df=0.8
            )
            
            self.lsi.fit(docs_for_bm25)
            
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
        
        print("✓ Index saved with line number tracking!")
        return stats


class HybridSearcher:
    """
    Load and search with LINE NUMBER RESULTS
    Perfect for code generation: Get exact file:line references!
    """
    
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
        """
        Search with LINE NUMBER TRACKING
        
        Returns:
            List of results with:
            - source_file: filename
            - source_path: full path
            - chunk_index: chunk number
            - start_line: starting line (1-indexed)
            - end_line: ending line (1-indexed)
            - location: formatted as "file.py:10-25"
            - text: chunk content
            - scores: BM25, LSI, combined
        """
        
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
            
            # Format location string
            location = f"{metadata['source_file']}:{metadata['start_line']}-{metadata['end_line']}"
            
            results.append({
                "source_file": metadata["source_file"],
                "source_path": metadata["source_path"],
                "file_type": metadata["file_type"],
                "chunk_index": metadata["chunk_index"],
                "start_line": metadata["start_line"],
                "end_line": metadata["end_line"],
                "total_lines": metadata["total_lines"],
                "location": location,  # NEW: formatted location
                "text": self.document_store[idx],
                "keywords": metadata.get("keywords", []),
                "bm25_score": float(bm25_scores[idx]),
                "lsi_score": float(lsi_scores[idx]),
                "combined_score": float(combined_scores[idx])
            })
        
        return results
    
    def get_full_file_context(self, file_path: str) -> str:
        """
        Get full file content for context
        Useful when generating code based on search results
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return ""
    
    def get_surrounding_lines(
        self,
        file_path: str,
        start_line: int,
        end_line: int,
        context_lines: int = 5
    ) -> Dict[str, Any]:
        """
        Get chunk with surrounding context
        
        Returns:
            {
                'before': lines before chunk,
                'chunk': the actual chunk lines,
                'after': lines after chunk,
                'full_text': combined text
            }
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                all_lines = f.readlines()
            
            # Adjust for 0-indexed
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
    
    def find_similar_terms(self, term: str, top_k: int = 10):
        """Find similar terms using eigenvectors"""
        if not self.lsi:
            return []
        return self.lsi.get_similar_terms(term, top_k)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Hybrid Search with Line Number Tracking (Perfect for RAG!)"
    )
    parser.add_argument("--folder", required=True)
    parser.add_argument("--index-path", default="./hybrid_index")
    parser.add_argument("--action", choices=["index", "search", "context"], default="index")
    parser.add_argument("--query", help="Search query")
    parser.add_argument("--file", help="File path for context")
    parser.add_argument("--start-line", type=int, help="Start line")
    parser.add_argument("--end-line", type=int, help="End line")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--bm25-weight", type=float, default=0.5)
    parser.add_argument("--lsi-weight", type=float, default=0.5)
    parser.add_argument("--disable-lsi", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Hybrid Search with Line Number Tracking")
    print("Perfect for RAG-based Code Generation!")
    print("=" * 80)
    
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
        
        print(f"\n{'='*80}")
        print(f"Found {len(results)} results")
        print(f"{'='*80}\n")
        
        for i, result in enumerate(results, 1):
            print(f"[{i}] {result['location']}")
            print(f"    File: {result['source_file']}")
            print(f"    Lines: {result['start_line']}-{result['end_line']} ({result['total_lines']} lines)")
            print(f"    Chunk: #{result['chunk_index']}")
            print(f"    Scores: BM25={result['bm25_score']:.3f} | LSI={result['lsi_score']:.3f} | Combined={result['combined_score']:.3f}")
            print(f"    Keywords: {', '.join(result['keywords'][:5])}")
            print(f"    Preview: {result['text'][:150]}...")
            print("-" * 80)
        
        # Show how to use for code generation
        if results:
            print("\n" + "="*80)
            print("RAG Code Generation Example:")
            print("="*80)
            best_result = results[0]
            print(f"\n# To generate code based on this snippet:")
            print(f"# File: {best_result['source_path']}")
            print(f"# Lines: {best_result['start_line']}-{best_result['end_line']}")
            print(f"\n# Get surrounding context:")
            context = searcher.get_surrounding_lines(
                best_result['source_path'],
                best_result['start_line'],
                best_result['end_line'],
                context_lines=5
            )
            if context:
                print(f"\n# Context (lines {context['context_start_line']}-{context['context_end_line']}):")
                print(context['full_text'][:500])
    
    elif args.action == "context":
        if not args.file or not args.start_line or not args.end_line:
            print("Error: --file, --start-line, --end-line required")
            return
        
        searcher = HybridSearcher(index_path=args.index_path)
        context = searcher.get_surrounding_lines(
            args.file,
            args.start_line,
            args.end_line,
            context_lines=10
        )
        
        if context:
            print(f"\n=== Context for {args.file}:{args.start_line}-{args.end_line} ===\n")
            print(context['full_text'])


if __name__ == "__main__":
    main()

print("\n" + "=" * 80)
print("✓ Hybrid Search with Line Number Tracking")
print("  • Returns: filename, chunk#, line numbers")
print("  • Perfect for RAG-based code generation")
print("  • Get surrounding context for better prompts")
print("  • Eigenvector-based semantic search (LSI/SVD)")
print("=" * 80)
