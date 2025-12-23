#!/usr/bin/env python3
"""
TAL Code Intelligence Searcher v3.0
====================================
Enhanced with cascading search: exact → fuzzy → contains → token

New Features:
- Fuzzy matching (Levenshtein distance for typos)
- Contains matching (substring search)
- Token matching (individual word matching)
- Better suggestions when no results found

Based on tal_searcher_v2.py
#YK234
"""

import re
import os
import json
import math
import argparse
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum

# BM25 - try to import, otherwise provide simple fallback
try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False

# Optional: requests for OpenAI-compatible API
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# Optional: Qdrant for vector storage
try:
    from qdrant_client import QdrantClient
    HAS_QDRANT = True
except ImportError:
    HAS_QDRANT = False

# Optional: HuggingFace transformers
HAS_TRANSFORMERS = False
try:
    import torch
    from transformers import AutoModel, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    pass


# =============================================================================
# MATCH TYPE ENUM
# =============================================================================

class MatchType(Enum):
    """Type of match found"""
    EXACT = "exact"           # Perfect name match
    FUZZY = "fuzzy"           # Close match (edit distance)
    CONTAINS = "contains"     # Query is substring of name
    TOKEN = "token"           # Individual terms match
    HYBRID = "hybrid"         # BM25/vector search
    PARTIAL = "partial"       # Partial token overlap


# =============================================================================
# EMBEDDING PROVIDER (from v2)
# =============================================================================

class EmbeddingProvider:
    """Abstraction for different embedding providers"""
    
    def __init__(self, db_path: str = None):
        self.provider = None
        self.model = None
        self.tokenizer = None
        self.qdrant = None
        self.db_path = db_path
        self._init_provider()
    
    def _init_provider(self):
        """Initialize the best available embedding provider"""
        
        # Option 1: OpenAI-compatible API
        api_url = os.environ.get('EMBEDDING_API_URL')
        if api_url and HAS_REQUESTS:
            self.provider = 'openai_compatible'
            self.api_url = api_url.rstrip('/')
            self.api_key = os.environ.get('EMBEDDING_API_KEY', '')
            self.model_name = os.environ.get('EMBEDDING_MODEL', 'text-embedding-ada-002')
            self.dimension = int(os.environ.get('EMBEDDING_DIMENSION', '1536'))
            print(f"   📡 Using OpenAI-compatible embeddings: {self.api_url}")
            self._init_qdrant()
            return
        
        # Option 2: HuggingFace transformers
        if HAS_TRANSFORMERS:
            try:
                self.provider = 'transformers'
                model_name = 'sentence-transformers/all-MiniLM-L6-v2'
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name)
                self.model.eval()
                self.dimension = 384
                print("   🤗 Using HuggingFace transformers embeddings")
                self._init_qdrant()
                return
            except Exception as e:
                print(f"   ⚠️  Could not load transformers: {e}")
        
        # Option 3: No embeddings - BM25 only
        self.provider = None
        self.dimension = 0
        print("   📚 Using BM25 search only (no vector embeddings)")
    
    def _init_qdrant(self):
        """Initialize Qdrant if available"""
        if HAS_QDRANT and self.db_path:
            try:
                self.qdrant = QdrantClient(path=self.db_path)
            except Exception as e:
                print(f"   ⚠️  Could not init Qdrant: {e}")
                self.qdrant = None
    
    def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode texts to embeddings"""
        if not self.provider:
            return []
        
        if isinstance(texts, str):
            texts = [texts]
        
        if self.provider == 'openai_compatible':
            return self._encode_openai(texts)
        elif self.provider == 'transformers':
            return self._encode_transformers(texts)
        
        return []
    
    def _encode_openai(self, texts: List[str]) -> List[List[float]]:
        """Encode using OpenAI-compatible API"""
        try:
            headers = {'Content-Type': 'application/json'}
            if self.api_key:
                headers['Authorization'] = f'Bearer {self.api_key}'
            
            response = requests.post(
                f"{self.api_url}/embeddings",
                headers=headers,
                json={
                    'model': self.model_name,
                    'input': texts
                },
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            embeddings = [item['embedding'] for item in sorted(data['data'], key=lambda x: x['index'])]
            return embeddings
        
        except Exception as e:
            print(f"   ⚠️  Embedding API error: {e}")
            return []
    
    def _encode_transformers(self, texts: List[str]) -> List[List[float]]:
        """Encode using HuggingFace transformers"""
        try:
            inputs = self.tokenizer(texts, padding=True, truncation=True, 
                                   max_length=512, return_tensors='pt')
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)
            return embeddings.tolist()
        except Exception as e:
            print(f"   ⚠️  Transformers encoding error: {e}")
            return []
    
    def is_available(self) -> bool:
        """Check if embeddings are available"""
        return self.provider is not None


# =============================================================================
# SIMPLE BM25 FALLBACK
# =============================================================================

class SimpleBM25:
    """Simple BM25 implementation as fallback"""
    
    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.doc_len = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_len) / len(corpus) if corpus else 0
        self.idf = {}
        self._calc_idf()
    
    def _calc_idf(self):
        """Calculate IDF for all terms"""
        df = defaultdict(int)
        for doc in self.corpus:
            for term in set(doc):
                df[term] += 1
        
        n = len(self.corpus)
        for term, freq in df.items():
            self.idf[term] = math.log((n - freq + 0.5) / (freq + 0.5) + 1)
    
    def get_scores(self, query: List[str]) -> List[float]:
        """Get BM25 scores for query"""
        scores = []
        for i, doc in enumerate(self.corpus):
            score = 0
            doc_len = self.doc_len[i]
            term_freqs = defaultdict(int)
            for term in doc:
                term_freqs[term] += 1
            
            for term in query:
                if term not in self.idf:
                    continue
                tf = term_freqs.get(term, 0)
                idf = self.idf[term]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                score += idf * numerator / denominator
            
            scores.append(score)
        return scores


# =============================================================================
# ENHANCED SEARCH RESULT
# =============================================================================

@dataclass
class SearchResult:
    """Search result with match metadata"""
    score: float
    result_type: str  # 'procedure', 'symbol', 'struct', 'rule'
    name: str
    file: str
    line: int
    text: str
    context: str
    match_type: MatchType
    matched_terms: List[str] = field(default_factory=list)
    data: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'score': self.score,
            'result_type': self.result_type,
            'name': self.name,
            'file': self.file,
            'line': self.line,
            'match_type': self.match_type.value,
            'matched_terms': self.matched_terms
        }


# =============================================================================
# LLM INTEGRATION (from v2)
# =============================================================================

def CALL_LLM(system_prompt: str, user_prompt: str, model: str = "gpt-4") -> str:
    """Call LLM API - set LLM_API_URL to enable"""
    api_url = os.environ.get('LLM_API_URL')
    api_key = os.environ.get('LLM_API_KEY', '')
    
    if not api_url:
        return "[LLM not configured. Set LLM_API_URL environment variable.]"
    
    if not HAS_REQUESTS:
        return "[requests library not installed.]"
    
    try:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        response = requests.post(
            f"{api_url.rstrip('/')}/chat/completions",
            headers=headers,
            json={
                "model": os.environ.get('LLM_MODEL', model),
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 4000
            },
            timeout=120
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    
    except Exception as e:
        return f"[LLM API error: {e}]"


# =============================================================================
# ENHANCED TAL SEARCHER v3
# =============================================================================

class TalSearcherV3:
    """
    Enhanced TAL code searcher with cascading search strategy:
    1. EXACT - Direct name match
    2. FUZZY - Levenshtein distance for typos
    3. CONTAINS - Substring matching
    4. TOKEN - Individual word matching
    5. HYBRID - BM25 + Vector search
    """
    
    def __init__(self, db_path: str = "./tal_index", collection_name: str = "tal"):
        print("🔍 Loading TAL Intelligence System v3.0 (Enhanced Search)...")
        
        self.db_path = db_path
        self.collection_name = collection_name
        self.logic_coll = f"{collection_name}_logic"
        self.symbol_coll = f"{collection_name}_symbols"
        
        # Initialize embedding provider
        self.embedder = EmbeddingProvider(db_path)
        
        # Load JSON stores
        self.procedures = self._load_json('procedures.json')
        self.symbols = self._load_json('symbols.json')
        self.structs = self._load_json('structs.json')
        self.call_graph = self._load_json('call_graph.json')
        self.data_usage = self._load_json('data_usage.json')
        self.business_rules = self._load_json('business_rules.json')
        self.preprocessor = self._load_json('preprocessor.json')
        
        # Build lookup indexes
        self._build_lookup_indexes()
        
        # Build BM25 indexes
        self._build_bm25_indices()
        
        # Build token index for fast token matching
        self._build_token_index()
        
        print(f"   Loaded {len(self.procedures)} procedures, {len(self.symbols)} symbols")
        print(f"   Loaded {len(self.structs)} structs, {len(self.business_rules)} business rules")
        print(f"   🔤 Token index: {len(self.token_index)} unique tokens")
    
    def _load_json(self, filename: str) -> any:
        """Load JSON file from db_path"""
        path = os.path.join(self.db_path, filename)
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"   ⚠️  Error loading {filename}: {e}")
                return [] if filename.endswith('s.json') else {}
        return [] if filename.endswith('s.json') else {}
    
    def _build_lookup_indexes(self):
        """Build fast lookup indexes"""
        # Procedures by name
        self.proc_by_name = {p['name'].upper(): p for p in self.procedures}
        
        # Populate calls from call_graph if not in procedure records
        for proc_name, proc in self.proc_by_name.items():
            if not proc.get('calls') and proc_name in self.call_graph:
                callees = self.call_graph[proc_name].get('callees', [])
                proc['calls'] = [{'target': c} for c in callees]
        
        # Symbols by name (can have duplicates)
        self.symbol_by_name = defaultdict(list)
        for s in self.symbols:
            self.symbol_by_name[s['name'].upper().lstrip('.')].append(s)
        
        # Structs by name
        self.struct_by_name = {s['name'].upper(): s for s in self.structs}
        
        # Business rules by procedure
        self.rules_by_proc = defaultdict(list)
        for r in self.business_rules:
            self.rules_by_proc[r.get('procedure', '').upper()].append(r)
        
        # All known names for suggestions and fuzzy matching
        self.all_proc_names = set(self.proc_by_name.keys())
        self.all_symbol_names = set(self.symbol_by_name.keys())
        self.all_struct_names = set(self.struct_by_name.keys())
        self.all_names = self.all_proc_names | self.all_symbol_names | self.all_struct_names
    
    def _build_bm25_indices(self):
        """Build BM25 indices"""
        BM25Class = BM25Okapi if HAS_BM25 else SimpleBM25
        
        # Procedures
        proc_texts = [
            f"{p.get('name', '')} {p.get('code', '')}"
            for p in self.procedures
        ]
        self.proc_bm25 = BM25Class([self._tokenize(t) for t in proc_texts]) if proc_texts else None
        
        # Symbols
        symbol_texts = [
            f"{s['name']} {s.get('data_type', '')} {s.get('section', '')}"
            for s in self.symbols
        ]
        self.symbol_bm25 = BM25Class([self._tokenize(t) for t in symbol_texts]) if symbol_texts else None
        
        # Structs
        struct_texts = [
            f"{s['name']} STRUCT {' '.join(f.get('name', '') for f in s.get('fields', []))}"
            for s in self.structs
        ]
        self.struct_bm25 = BM25Class([self._tokenize(t) for t in struct_texts]) if struct_texts else None
    
    def _build_token_index(self):
        """Build inverted index: token -> [(name, type), ...]"""
        self.token_index: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        
        # Index procedures
        for proc in self.procedures:
            name = proc.get('name', '').upper()
            for token in self._split_name(name):
                self.token_index[token].append((name, 'procedure'))
        
        # Index symbols
        for sym in self.symbols:
            name = sym.get('name', '').upper().lstrip('.')
            for token in self._split_name(name):
                self.token_index[token].append((name, 'symbol'))
        
        # Index structs
        for struct in self.structs:
            name = struct.get('name', '').upper()
            for token in self._split_name(name):
                self.token_index[token].append((name, 'struct'))
    
    def _split_name(self, name: str) -> List[str]:
        """Split identifier into tokens"""
        # Split on underscores and caret (^)
        parts = re.split(r'[_^]', name.upper())
        tokens = [p for p in parts if len(p) >= 2]
        # Also include the full name
        if name:
            tokens.append(name.upper())
        return tokens
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25"""
        return re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_^]*\b', text.lower())
    
    # =========================================================================
    # LEVENSHTEIN DISTANCE (Fuzzy Matching)
    # =========================================================================
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein edit distance"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def _similarity_score(self, s1: str, s2: str) -> float:
        """Calculate similarity score (0-1)"""
        if not s1 or not s2:
            return 0.0
        
        distance = self._levenshtein_distance(s1.upper(), s2.upper())
        max_len = max(len(s1), len(s2))
        
        return 1.0 - (distance / max_len)
    
    def _find_fuzzy_matches(self, query: str, names: Set[str], 
                           threshold: float = 0.6, limit: int = 10) -> List[Tuple[str, float]]:
        """Find names similar to query using fuzzy matching"""
        query_upper = query.upper()
        matches = []
        
        for name in names:
            score = self._similarity_score(query_upper, name)
            if score >= threshold:
                matches.append((name, score))
        
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:limit]
    
    # =========================================================================
    # CASCADING SEARCH METHODS
    # =========================================================================
    
    def search_exact(self, query: str) -> List[SearchResult]:
        """Search for exact name match"""
        query_upper = query.strip().upper()
        results = []
        
        # Check structs first (highest priority for struct names)
        if query_upper in self.struct_by_name:
            struct = self.struct_by_name[query_upper]
            field_count = len(struct.get('fields', []))
            inline_tag = " (inline)" if struct.get('inline') else ""
            results.append(SearchResult(
                score=1.0,
                result_type='struct',
                name=struct['name'],
                file=struct['file'],
                line=struct.get('start_line', 1),
                text=f"{field_count} fields{inline_tag}",
                context="STRUCT definition",
                match_type=MatchType.EXACT,
                matched_terms=[query_upper],
                data=struct
            ))
        
        # Check procedures
        if query_upper in self.proc_by_name:
            proc = self.proc_by_name[query_upper]
            results.append(SearchResult(
                score=1.0,
                result_type='procedure',
                name=proc['name'],
                file=proc['file'],
                line=proc.get('start_line', 1),
                text=proc.get('code', '')[:200],
                context=f"Type: {proc.get('proc_type', 'PROC')} | Calls: {len(proc.get('calls', []))}",
                match_type=MatchType.EXACT,
                matched_terms=[query_upper],
                data=proc
            ))
        
        # Check symbols
        if query_upper in self.symbol_by_name:
            for sym in self.symbol_by_name[query_upper]:
                results.append(SearchResult(
                    score=1.0,
                    result_type='symbol',
                    name=sym['name'],
                    file=sym['file'],
                    line=sym.get('line', 1),
                    text=sym.get('data_type', 'N/A'),
                    context=f"Section: {sym.get('section', 'GLOBAL')}",
                    match_type=MatchType.EXACT,
                    matched_terms=[query_upper],
                    data=sym
                ))
        
        return results
    
    def search_fuzzy(self, query: str, threshold: float = 0.6) -> List[SearchResult]:
        """Search for fuzzy/close matches (handles typos)"""
        results = []
        
        # Find fuzzy matches in procedures
        proc_matches = self._find_fuzzy_matches(query, self.all_proc_names, threshold)
        for name, score in proc_matches:
            proc = self.proc_by_name[name]
            results.append(SearchResult(
                score=score,
                result_type='procedure',
                name=name,
                file=proc['file'],
                line=proc.get('start_line', 1),
                text=proc.get('code', '')[:200],
                context=f"Type: {proc.get('proc_type', 'PROC')}",
                match_type=MatchType.FUZZY,
                matched_terms=[query.upper()],
                data=proc
            ))
        
        # Find fuzzy matches in symbols
        sym_matches = self._find_fuzzy_matches(query, self.all_symbol_names, threshold)
        for name, score in sym_matches:
            for sym in self.symbol_by_name[name]:
                results.append(SearchResult(
                    score=score,
                    result_type='symbol',
                    name=name,
                    file=sym['file'],
                    line=sym.get('line', 1),
                    text=sym.get('data_type', 'N/A'),
                    context=f"Section: {sym.get('section', 'GLOBAL')}",
                    match_type=MatchType.FUZZY,
                    matched_terms=[query.upper()],
                    data=sym
                ))
        
        # Find fuzzy matches in structs
        struct_matches = self._find_fuzzy_matches(query, self.all_struct_names, threshold)
        for name, score in struct_matches:
            struct = self.struct_by_name[name]
            field_count = len(struct.get('fields', []))
            results.append(SearchResult(
                score=score,
                result_type='struct',
                name=name,
                file=struct['file'],
                line=struct.get('start_line', 1),
                text=f"{field_count} fields",
                context="STRUCT definition",
                match_type=MatchType.FUZZY,
                matched_terms=[query.upper()],
                data=struct
            ))
        
        results.sort(key=lambda r: r.score, reverse=True)
        return results
    
    def search_contains(self, query: str) -> List[SearchResult]:
        """Search for names containing query as substring"""
        query_upper = query.strip().upper()
        results = []
        
        # Search procedures
        for name, proc in self.proc_by_name.items():
            if query_upper in name:
                score = len(query_upper) / len(name)
                results.append(SearchResult(
                    score=score,
                    result_type='procedure',
                    name=name,
                    file=proc['file'],
                    line=proc.get('start_line', 1),
                    text=proc.get('code', '')[:200],
                    context=f"Type: {proc.get('proc_type', 'PROC')}",
                    match_type=MatchType.CONTAINS,
                    matched_terms=[query_upper],
                    data=proc
                ))
        
        # Search symbols
        for name, syms in self.symbol_by_name.items():
            if query_upper in name:
                score = len(query_upper) / len(name)
                for sym in syms:
                    results.append(SearchResult(
                        score=score,
                        result_type='symbol',
                        name=name,
                        file=sym['file'],
                        line=sym.get('line', 1),
                        text=sym.get('data_type', 'N/A'),
                        context=f"Section: {sym.get('section', 'GLOBAL')}",
                        match_type=MatchType.CONTAINS,
                        matched_terms=[query_upper],
                        data=sym
                    ))
        
        # Search structs
        for name, struct in self.struct_by_name.items():
            if query_upper in name:
                score = len(query_upper) / len(name)
                field_count = len(struct.get('fields', []))
                results.append(SearchResult(
                    score=score,
                    result_type='struct',
                    name=name,
                    file=struct['file'],
                    line=struct.get('start_line', 1),
                    text=f"{field_count} fields",
                    context="STRUCT definition",
                    match_type=MatchType.CONTAINS,
                    matched_terms=[query_upper],
                    data=struct
                ))
        
        results.sort(key=lambda r: r.score, reverse=True)
        return results
    
    def search_tokens(self, query: str) -> List[SearchResult]:
        """Search by matching individual query terms"""
        # Extract query tokens
        query_tokens = set(self._split_name(query.upper()))
        
        if not query_tokens:
            return []
        
        # Find names that match any query token
        matches: Dict[str, Tuple[str, float, Set[str]]] = {}  # name -> (type, score, matched_tokens)
        
        for token in query_tokens:
            token_upper = token.upper()
            
            # Direct token match
            if token_upper in self.token_index:
                for name, item_type in self.token_index[token_upper]:
                    if name not in matches:
                        matches[name] = (item_type, 0, set())
                    matches[name] = (
                        matches[name][0],
                        matches[name][1] + 1,
                        matches[name][2] | {token_upper}
                    )
            
            # Partial token matches (token is substring of indexed token)
            for indexed_token, items in self.token_index.items():
                if len(token_upper) >= 3 and (token_upper in indexed_token or indexed_token in token_upper):
                    if indexed_token != token_upper:  # Don't double count exact matches
                        for name, item_type in items:
                            if name not in matches:
                                matches[name] = (item_type, 0, set())
                            matches[name] = (
                                matches[name][0],
                                matches[name][1] + 0.5,  # Partial match counts less
                                matches[name][2] | {token_upper}
                            )
        
        # Convert to results
        results = []
        for name, (item_type, count, matched) in matches.items():
            score = count / len(query_tokens)
            match_type = MatchType.TOKEN if count >= 1 else MatchType.PARTIAL
            
            # Get item data
            if item_type == 'procedure' and name in self.proc_by_name:
                proc = self.proc_by_name[name]
                results.append(SearchResult(
                    score=score,
                    result_type='procedure',
                    name=name,
                    file=proc['file'],
                    line=proc.get('start_line', 1),
                    text=proc.get('code', '')[:200],
                    context=f"Type: {proc.get('proc_type', 'PROC')}",
                    match_type=match_type,
                    matched_terms=list(matched),
                    data=proc
                ))
            elif item_type == 'symbol' and name in self.symbol_by_name:
                for sym in self.symbol_by_name[name]:
                    results.append(SearchResult(
                        score=score,
                        result_type='symbol',
                        name=name,
                        file=sym['file'],
                        line=sym.get('line', 1),
                        text=sym.get('data_type', 'N/A'),
                        context=f"Section: {sym.get('section', 'GLOBAL')}",
                        match_type=match_type,
                        matched_terms=list(matched),
                        data=sym
                    ))
            elif item_type == 'struct' and name in self.struct_by_name:
                struct = self.struct_by_name[name]
                field_count = len(struct.get('fields', []))
                results.append(SearchResult(
                    score=score,
                    result_type='struct',
                    name=name,
                    file=struct['file'],
                    line=struct.get('start_line', 1),
                    text=f"{field_count} fields",
                    context="STRUCT definition",
                    match_type=match_type,
                    matched_terms=list(matched),
                    data=struct
                ))
        
        results.sort(key=lambda r: r.score, reverse=True)
        return results
    
    def search_hybrid(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """BM25 + Vector hybrid search"""
        query_tokens = self._tokenize(query)
        results = []
        seen = set()
        
        # Embed query for vector search
        query_vec = []
        if self.embedder.is_available():
            embeddings = self.embedder.encode([query])
            query_vec = embeddings[0] if embeddings else []
        
        # Procedures
        if self.proc_bm25:
            bm25_scores = self.proc_bm25.get_scores(query_tokens)
            bm25_ranked = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)[:20]
            bm25_results = [(idx, score) for idx, score in bm25_ranked if score > 0.1]
            
            # Vector search
            vector_results = []
            if self.embedder.qdrant and query_vec:
                try:
                    vec_results = self.embedder.qdrant.query_points(self.logic_coll, query=query_vec, limit=20).points
                    vector_results = [(p.id, p.score) for p in vec_results]
                except Exception:
                    pass
            
            # RRF fusion
            fused = self._rrf_fusion(bm25_results, vector_results)
            
            for idx, rrf_score in fused[:top_k]:
                if 0 <= idx < len(self.procedures):
                    proc = self.procedures[idx]
                    key = (proc['file'], proc['name'])
                    if key not in seen:
                        results.append(SearchResult(
                            score=rrf_score,
                            result_type='procedure',
                            name=proc['name'],
                            file=proc['file'],
                            line=proc.get('start_line', 1),
                            text=proc.get('code', '')[:200],
                            context=f"Type: {proc.get('proc_type', 'PROC')}",
                            match_type=MatchType.HYBRID,
                            data=proc
                        ))
                        seen.add(key)
        
        # Symbols
        if self.symbol_bm25:
            sym_scores = self.symbol_bm25.get_scores(query_tokens)
            sym_ranked = sorted(enumerate(sym_scores), key=lambda x: x[1], reverse=True)[:10]
            
            for idx, score in sym_ranked:
                if score > 0.1 and 0 <= idx < len(self.symbols):
                    sym = self.symbols[idx]
                    key = (sym['file'], sym['name'])
                    if key not in seen:
                        results.append(SearchResult(
                            score=score * 0.5,
                            result_type='symbol',
                            name=sym['name'],
                            file=sym['file'],
                            line=sym.get('line', 1),
                            text=sym.get('data_type', 'N/A'),
                            context=f"Section: {sym.get('section', 'GLOBAL')}",
                            match_type=MatchType.HYBRID,
                            data=sym
                        ))
                        seen.add(key)
        
        # Structs
        if self.struct_bm25:
            struct_scores = self.struct_bm25.get_scores(query_tokens)
            struct_ranked = sorted(enumerate(struct_scores), key=lambda x: x[1], reverse=True)[:5]
            
            for idx, score in struct_ranked:
                if score > 0.1 and 0 <= idx < len(self.structs):
                    struct = self.structs[idx]
                    key = (struct['file'], struct['name'])
                    if key not in seen:
                        field_count = len(struct.get('fields', []))
                        results.append(SearchResult(
                            score=score * 0.5,
                            result_type='struct',
                            name=struct['name'],
                            file=struct['file'],
                            line=struct.get('start_line', 1),
                            text=f"{field_count} fields",
                            context="STRUCT definition",
                            match_type=MatchType.HYBRID,
                            data=struct
                        ))
                        seen.add(key)
        
        return results
    
    def _rrf_fusion(
        self,
        bm25_results: List[Tuple[int, float]],
        vector_results: List[Tuple[int, float]],
        k: int = 60
    ) -> List[Tuple[int, float]]:
        """Reciprocal Rank Fusion"""
        scores = defaultdict(float)
        
        for rank, (idx, _) in enumerate(bm25_results):
            scores[idx] += 1.0 / (k + rank + 1)
        
        for rank, (idx, _) in enumerate(vector_results):
            scores[idx] += 1.0 / (k + rank + 1)
        
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # =========================================================================
    # MAIN SEARCH METHOD (CASCADING)
    # =========================================================================
    
    def search(self, query: str, top_k: int = 20,
               include_fuzzy: bool = True,
               include_contains: bool = True,
               include_tokens: bool = True,
               include_hybrid: bool = True,
               include_code: bool = True,
               fuzzy_threshold: float = 0.6) -> List[SearchResult]:
        """
        Main cascading search method.
        
        Search Order:
        1. EXACT - Direct name match (highest priority)
        2. FUZZY - Close matches for typos
        3. CONTAINS - Substring in names
        4. TOKEN - Individual word matching in names
        5. CODE - Search within procedure code content (including comments)
        6. HYBRID - BM25 + Vector search (semantic)
        
        Args:
            query: Search query
            top_k: Maximum results
            include_fuzzy: Include fuzzy matches
            include_contains: Include substring matches
            include_tokens: Include token-based matches
            include_hybrid: Include BM25/vector search
            include_code: Include code content search
            fuzzy_threshold: Minimum similarity for fuzzy (0-1)
        
        Returns:
            List of SearchResult ordered by relevance
        """
        all_results = []
        seen_names = set()
        
        def add_results(results: List[SearchResult], boost: float = 1.0):
            """Add results while avoiding duplicates"""
            for r in results:
                if r.name not in seen_names:
                    seen_names.add(r.name)
                    r.score *= boost
                    all_results.append(r)
        
        # Clean up query - remove common question words for better matching
        query_clean = query.strip()
        question_words = {'how', 'to', 'what', 'is', 'the', 'a', 'an', 'does', 'do', 'can', 'where', 'when', 'why'}
        
        # Handle "code" as a potential command prefix
        if query_clean.lower().startswith('code '):
            query_clean = query_clean[5:].strip()
        
        query_words = query_clean.split()
        meaningful_words = [w for w in query_words if w.lower() not in question_words and len(w) >= 2]
        meaningful_query = ' '.join(meaningful_words) if meaningful_words else query_clean
        
        # 1. EXACT MATCH (always first)
        exact_results = self.search_exact(query_clean)
        add_results(exact_results, boost=1.0)
        
        # Also try meaningful query for exact match
        if meaningful_query != query_clean:
            exact_results2 = self.search_exact(meaningful_query)
            add_results(exact_results2, boost=0.95)
        
        # If we have exact matches, return them
        if exact_results:
            return all_results[:top_k]
        
        # 2. FUZZY MATCH (typo correction)
        if include_fuzzy:
            fuzzy_results = self.search_fuzzy(meaningful_query, threshold=fuzzy_threshold)
            add_results(fuzzy_results, boost=0.9)
        
        # 3. CONTAINS MATCH (substring in names)
        if include_contains:
            contains_results = self.search_contains(meaningful_query)
            add_results(contains_results, boost=0.85)
        
        # 4. TOKEN MATCH (individual words in names)
        if include_tokens:
            token_results = self.search_tokens(meaningful_query)
            add_results(token_results, boost=0.8)
        
        # 5. CODE CONTENT SEARCH (search within procedure code and comments)
        if include_code:
            # Use meaningful query for code search
            code_results = self.search_in_code(meaningful_query)
            add_results(code_results, boost=0.75)
            
            # Also search comments specifically
            comment_results = self.search_in_comments(meaningful_query)
            add_results(comment_results, boost=0.7)
        
        # 6. HYBRID SEARCH (BM25 + Vector)
        if include_hybrid and len(all_results) < 5:
            # Only use hybrid if we haven't found much
            hybrid_results = self.search_hybrid(meaningful_query, top_k)
            add_results(hybrid_results, boost=0.6)
        
        # Sort by match quality and score
        def sort_key(r: SearchResult) -> Tuple[int, float]:
            type_order = {
                MatchType.EXACT: 0,
                MatchType.FUZZY: 1,
                MatchType.CONTAINS: 2,
                MatchType.TOKEN: 3,
                MatchType.HYBRID: 4,
                MatchType.PARTIAL: 5
            }
            return (type_order.get(r.match_type, 6), -r.score)
        
        all_results.sort(key=sort_key)
        
        # If still no results, provide suggestions
        if not all_results:
            suggestions = self.suggest(meaningful_query)
            if suggestions:
                print(f"   💡 Did you mean: {', '.join(suggestions[:5])}")
        
        return all_results[:top_k]
    
    def suggest(self, query: str, limit: int = 5) -> List[str]:
        """Get search suggestions for partial query"""
        # Try fuzzy matching with lower threshold
        fuzzy = self._find_fuzzy_matches(query, self.all_names, threshold=0.4, limit=limit)
        return [name for name, _ in fuzzy]
    
    # =========================================================================
    # SPECIALIZED SEARCH METHODS
    # =========================================================================
    
    def search_procedures(self, query: str, limit: int = 10) -> List[SearchResult]:
        """Search only procedures"""
        results = self.search(query, top_k=limit * 2)
        return [r for r in results if r.result_type == 'procedure'][:limit]
    
    def search_symbols(self, query: str, limit: int = 10) -> List[SearchResult]:
        """Search only symbols"""
        results = self.search(query, top_k=limit * 2)
        return [r for r in results if r.result_type == 'symbol'][:limit]
    
    def search_structs(self, query: str, limit: int = 10) -> List[SearchResult]:
        """Search only structs"""
        results = self.search(query, top_k=limit * 2)
        return [r for r in results if r.result_type == 'struct'][:limit]
    
    def search_in_code(self, query: str, limit: int = 20) -> List[SearchResult]:
        """Search within procedure code content (including comments)"""
        results = []
        
        # Handle multi-word queries - search for the phrase
        query_clean = query.strip()
        query_upper = query_clean.upper()
        
        # Also prepare individual words for fallback
        query_words = [w.upper() for w in re.findall(r'[A-Za-z_][A-Za-z0-9_]*', query_clean) if len(w) >= 3]
        
        for proc in self.procedures:
            # Search in raw_code (with comments) first, then normalized code
            code = proc.get('raw_code', proc.get('code', ''))
            code_upper = code.upper()
            name = proc.get('name', '')
            
            score = 0
            matched = []
            
            # 1. Exact phrase match (highest value)
            if query_upper in code_upper:
                count = code_upper.count(query_upper)
                score += count * 10
                matched.append(query_clean)
            
            # 2. Individual word matches (lower value but still useful)
            for word in query_words:
                if word in code_upper:
                    word_count = code_upper.count(word)
                    score += word_count * 2
                    if word not in matched:
                        matched.append(word)
            
            if score > 0:
                # Find the line where the match occurs for context
                match_context = ""
                for line in code.split('\n'):
                    if query_upper in line.upper() or any(w in line.upper() for w in query_words[:2]):
                        match_context = line.strip()[:80]
                        break
                
                results.append(SearchResult(
                    score=min(1.0, score / 20),  # Normalize score
                    result_type='procedure',
                    name=name,
                    file=proc.get('file', ''),
                    line=proc.get('start_line', 1),
                    text=match_context if match_context else f"Found in code",
                    context=f"Type: {proc.get('proc_type', 'PROC')} | Matches: {score//2}",
                    match_type=MatchType.CONTAINS,
                    matched_terms=matched,
                    data=proc
                ))
        
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]
    
    def search_in_comments(self, query: str, limit: int = 20) -> List[SearchResult]:
        """Search specifically in code comments (lines starting with !)"""
        results = []
        query_upper = query.strip().upper()
        query_words = [w.upper() for w in re.findall(r'[A-Za-z_][A-Za-z0-9_]*', query) if len(w) >= 3]
        
        for proc in self.procedures:
            code = proc.get('raw_code', proc.get('code', ''))
            name = proc.get('name', '')
            
            score = 0
            matched_lines = []
            
            for line in code.split('\n'):
                # Check if it's a comment line
                line_stripped = line.strip()
                if line_stripped.startswith('!') or '!' in line:
                    comment_part = line_stripped
                    if '!' in line and not line_stripped.startswith('!'):
                        comment_part = line[line.index('!'):]
                    
                    comment_upper = comment_part.upper()
                    
                    # Check for phrase match
                    if query_upper in comment_upper:
                        score += 10
                        matched_lines.append(comment_part.strip()[:60])
                    else:
                        # Check word matches
                        for word in query_words:
                            if word in comment_upper:
                                score += 2
                                if len(matched_lines) < 3:
                                    matched_lines.append(comment_part.strip()[:60])
                                break
            
            if score > 0:
                results.append(SearchResult(
                    score=min(1.0, score / 20),
                    result_type='procedure',
                    name=name,
                    file=proc.get('file', ''),
                    line=proc.get('start_line', 1),
                    text=matched_lines[0] if matched_lines else "Found in comments",
                    context=f"Comment matches: {len(matched_lines)}",
                    match_type=MatchType.CONTAINS,
                    matched_terms=query_words[:3],
                    data=proc
                ))
        
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]
    
    # =========================================================================
    # TRACE & USAGE (from v2)
    # =========================================================================
    
    def trace_procedure(self, proc_name: str) -> Dict:
        """Trace procedure call relationships"""
        name_upper = proc_name.strip().upper()
        
        if name_upper not in self.call_graph:
            suggestions = self.suggest(name_upper)
            return {'error': f"Procedure '{proc_name}' not found", 'suggestions': suggestions}
        
        node = self.call_graph[name_upper]
        proc_info = self.proc_by_name.get(name_upper, {})
        
        direct_callers = node.get('callers', [])
        direct_callees = node.get('callees', node.get('targets', []))
        
        # Calculate impact
        all_callers = set(direct_callers)
        to_check = list(direct_callers)
        checked = set()
        
        while to_check:
            caller = to_check.pop()
            if caller in checked:
                continue
            checked.add(caller)
            
            if caller in self.call_graph:
                indirect = self.call_graph[caller].get('callers', [])
                for c in indirect:
                    if c not in all_callers:
                        all_callers.add(c)
                        to_check.append(c)
        
        return {
            'name': name_upper,
            'file': node.get('file'),
            'type': node.get('proc_type', node.get('type')),
            'return_type': proc_info.get('return_type'),
            'attributes': proc_info.get('attributes', []),
            'direct_callers': direct_callers,
            'direct_callees': direct_callees,
            'all_callers': list(all_callers - set(direct_callers)),
            'impact_count': len(all_callers)
        }
    
    def find_data_usage(self, symbol: str) -> Tuple[List[Dict], List[str]]:
        """Find all places where a symbol is used"""
        sym_upper = symbol.strip().upper().lstrip('.')
        
        if sym_upper in self.data_usage:
            return self.data_usage[sym_upper], []
        
        suggestions = self.suggest(sym_upper)
        return [], suggestions
    
    def get_struct(self, struct_name: str) -> Dict:
        """Get struct definition with fields"""
        name_upper = struct_name.strip().upper()
        
        if name_upper not in self.struct_by_name:
            suggestions = self.suggest(name_upper)
            return {'error': f"Struct '{struct_name}' not found", 'suggestions': suggestions}
        
        return self.struct_by_name[name_upper]
    
    def explain_procedure(self, proc_name: str) -> Dict:
        """Get full context for a procedure"""
        name_upper = proc_name.strip().upper()
        
        if name_upper not in self.proc_by_name:
            suggestions = self.suggest(name_upper)
            return {'error': f"Procedure '{proc_name}' not found", 'suggestions': suggestions}
        
        proc = self.proc_by_name[name_upper]
        rules = self.rules_by_proc.get(name_upper, [])
        
        return {
            'procedure': proc,
            'business_rules': rules,
            'call_info': self.trace_procedure(proc_name),
            'data_refs': proc.get('data_refs', [])
        }
    
    # =========================================================================
    # LIST METHODS
    # =========================================================================
    
    def list_procedures(self, pattern: str = None) -> List[Dict]:
        """List procedures, optionally filtered by pattern"""
        if not pattern:
            return [{'name': p['name'], 'file': p['file'], 'type': p.get('proc_type', 'PROC')} 
                    for p in self.procedures]
        
        pattern_upper = pattern.upper()
        return [
            {'name': p['name'], 'file': p['file'], 'type': p.get('proc_type', 'PROC')}
            for p in self.procedures
            if pattern_upper in p['name']
        ]
    
    def list_defines(self, pattern: str = None) -> List[Dict]:
        """List DEFINE macros"""
        defines = [s for s in self.symbols if s.get('data_type') == 'DEFINE']
        
        if pattern:
            pattern_upper = pattern.upper()
            defines = [d for d in defines if pattern_upper in d['name']]
        
        return defines
    
    def list_literals(self, pattern: str = None) -> List[Dict]:
        """List LITERAL constants"""
        literals = [s for s in self.symbols if s.get('data_type') == 'LITERAL']
        
        if pattern:
            pattern_upper = pattern.upper()
            literals = [l for l in literals if pattern_upper in l['name']]
        
        return literals
    
    def close(self):
        """Close resources"""
        if self.embedder.qdrant:
            try:
                self.embedder.qdrant.close()
            except Exception:
                pass


# =============================================================================
# PRETTY PRINTING
# =============================================================================

def print_search_results(results: List[SearchResult]):
    """Pretty print search results with match type indicators"""
    if not results:
        print("   No results found")
        return
    
    # Match type indicators
    match_icons = {
        MatchType.EXACT: "✓ exact",
        MatchType.FUZZY: "≈ fuzzy",
        MatchType.CONTAINS: "⊃ contains",
        MatchType.TOKEN: "∩ token",
        MatchType.HYBRID: "🔍 hybrid",
        MatchType.PARTIAL: "~ partial"
    }
    
    # Type icons
    type_icons = {
        'struct': '🏗️',
        'procedure': '📋',
        'symbol': '🏷️'
    }
    
    # Group by type
    structs = [r for r in results if r.result_type == 'struct']
    procs = [r for r in results if r.result_type == 'procedure']
    symbols = [r for r in results if r.result_type == 'symbol']
    
    print(f"\n   Found {len(results)} result(s):\n")
    
    if structs:
        print("   🏗️ STRUCTS:")
        for r in structs[:5]:
            match_info = match_icons.get(r.match_type, r.match_type.value)
            print(f"      {r.name}")
            print(f"         {r.file}:L{r.line} | {r.text} | {match_info} ({r.score:.2f})")
            if r.matched_terms:
                print(f"         matched: {', '.join(r.matched_terms)}")
    
    if procs:
        print("\n   📋 PROCEDURES:")
        for r in procs[:10]:
            match_info = match_icons.get(r.match_type, r.match_type.value)
            print(f"      {r.name}")
            print(f"         {r.file}:L{r.line} | {r.context} | {match_info} ({r.score:.2f})")
            if r.matched_terms:
                print(f"         matched: {', '.join(r.matched_terms)}")
    
    if symbols:
        print("\n   🏷️ SYMBOLS:")
        for r in symbols[:10]:
            match_info = match_icons.get(r.match_type, r.match_type.value)
            print(f"      {r.name}")
            print(f"         {r.file}:L{r.line} | {r.text} | {match_info} ({r.score:.2f})")


def print_trace_result(trace: Dict):
    """Pretty print trace result"""
    if 'error' in trace:
        print(f"   ❌ {trace['error']}")
        if trace.get('suggestions'):
            print(f"   💡 Did you mean: {', '.join(trace['suggestions'][:5])}")
        return
    
    print(f"\n📍 {trace['name']}")
    print(f"   File: {trace['file']}")
    print(f"   Type: {trace['type']}")
    
    print(f"\n   ⬆️ Called by ({len(trace['direct_callers'])}):")
    for caller in trace['direct_callers'][:10]:
        print(f"      • {caller}")
    
    print(f"\n   ⬇️ Calls ({len(trace['direct_callees'])}):")
    for callee in trace['direct_callees'][:10]:
        print(f"      • {callee}")
    
    if trace['all_callers']:
        print(f"\n   🌐 Impact: {trace['impact_count']} procedures in call chain")


def print_usage_result(usages: List[Dict], suggestions: List[str], symbol: str):
    """Pretty print usage results"""
    if suggestions and not usages:
        print(f"   ❌ Symbol '{symbol}' not found")
        print(f"   💡 Did you mean: {', '.join(suggestions[:5])}")
        return
    
    if not usages:
        print(f"   No usages found for '{symbol}'")
        return
    
    by_file = defaultdict(list)
    for u in usages:
        by_file[u['file']].append(u)
    
    print(f"\n📊 Usage of '{symbol}' ({len(usages)} references):")
    for file, refs in sorted(by_file.items()):
        print(f"\n   📁 {file}:")
        for ref in refs:
            print(f"      • {ref['procedure']} ({ref.get('type', 'PROC')}) L{ref['line']}")


def print_struct_result(struct: Dict):
    """Pretty print struct result"""
    if 'error' in struct:
        print(f"   ❌ {struct['error']}")
        if struct.get('suggestions'):
            print(f"   💡 Did you mean: {', '.join(struct['suggestions'][:5])}")
        return
    
    inline_tag = " (inline parameter)" if struct.get('inline') else ""
    print(f"\n🏗️ STRUCT: {struct['name']}{inline_tag}")
    print(f"   File: {struct['file']}:L{struct.get('start_line', 1)}-{struct.get('end_line', '?')}")
    
    fields = struct.get('fields', [])
    print(f"\n   Fields ({len(fields)}):")
    for f in fields:
        ptr_tag = " (pointer)" if f.get('is_pointer') else ""
        bounds = f" [{f.get('array_bounds')}]" if f.get('array_bounds') else ""
        print(f"      • {f['name']}: {f.get('data_type', '?')}{bounds}{ptr_tag}")


# =============================================================================
# INTERACTIVE CLI
# =============================================================================

def interactive_search(searcher: TalSearcherV3):
    """Interactive search REPL"""
    print("\n" + "=" * 60)
    print("TAL Intelligence System v3.0 (Enhanced Search)")
    print("=" * 60)
    print("\nCommands:")
    print("  <query>                - Cascading search (exact→fuzzy→contains→token)")
    print("  TRACE <proc>           - Show call graph")
    print("  USAGE <symbol>         - Find symbol references")
    print("  EXPLAIN <proc>         - Full procedure context")
    print("  STRUCT <name>          - Show struct definition")
    print("  CODE <query>           - Search within code content")
    print("  LIST PROC [pattern]    - List procedures")
    print("  LIST DEFINE [pattern]  - List DEFINE macros")
    print("  HELP                   - Show this help")
    print("  QUIT                   - Exit")
    print("\n💡 Try: 'VALIDTE_CREDIT' (typo) or 'validate iban' (tokens)")
    print("=" * 60)
    
    while True:
        try:
            query = input("\n❓ Search: ").strip()
            
            if not query:
                continue
            
            query_upper = query.upper()
            
            if query_upper in ('QUIT', 'EXIT', 'Q'):
                print("👋 Goodbye!")
                break
            
            if query_upper == 'HELP':
                print("\nSearch Types:")
                print("  ✓ exact    - Direct name match")
                print("  ≈ fuzzy    - Close match (handles typos)")
                print("  ⊃ contains - Substring in name")
                print("  ∩ token    - Individual word match")
                print("  🔍 hybrid   - BM25 + vector search")
                print("\nExamples:")
                print("  VALIDATE_CREDIT_PARTY  → exact match")
                print("  VALIDTE_CREDIT         → fuzzy match (typo)")
                print("  CREDIT                 → contains match")
                print("  validate iban          → token match")
                continue
            
            # TRACE
            if query_upper.startswith('TRACE '):
                proc_name = query[6:].strip()
                result = searcher.trace_procedure(proc_name)
                print_trace_result(result)
                continue
            
            # USAGE
            if query_upper.startswith('USAGE '):
                symbol = query[6:].strip()
                usages, suggestions = searcher.find_data_usage(symbol)
                print_usage_result(usages, suggestions, symbol)
                continue
            
            # EXPLAIN
            if query_upper.startswith('EXPLAIN '):
                proc_name = query[8:].strip()
                result = searcher.explain_procedure(proc_name)
                if 'error' in result:
                    print(f"   ❌ {result['error']}")
                    if result.get('suggestions'):
                        print(f"   💡 Did you mean: {', '.join(result['suggestions'][:5])}")
                else:
                    proc = result['procedure']
                    print(f"\n📖 {proc['name']} ({proc['file']})")
                    print(f"   Type: {proc.get('proc_type', 'PROC')}")
                    code_preview = proc.get('code', '')[:500]
                    print(f"\n   Code:\n{code_preview}")
                continue
            
            # STRUCT
            if query_upper.startswith('STRUCT '):
                struct_name = query[7:].strip()
                result = searcher.get_struct(struct_name)
                print_struct_result(result)
                continue
            
            # CODE
            if query_upper.startswith('CODE '):
                code_query = query[5:].strip()
                results = searcher.search_in_code(code_query)
                print_search_results(results)
                continue
            
            # LIST
            if query_upper.startswith('LIST '):
                rest = query[5:].strip()
                parts = rest.split(None, 1)
                list_type = parts[0].upper() if parts else ''
                pattern = parts[1] if len(parts) > 1 else None
                
                if list_type in ('PROC', 'PROCS', 'PROCEDURE', 'PROCEDURES'):
                    items = searcher.list_procedures(pattern)
                    print(f"\n📋 Procedures ({len(items)}):")
                    for item in items[:30]:
                        print(f"   • {item['name']} ({item['file']})")
                elif list_type in ('DEFINE', 'DEFINES'):
                    items = searcher.list_defines(pattern)
                    print(f"\n🏷️ Defines ({len(items)}):")
                    for item in items[:30]:
                        print(f"   • {item['name']} ({item['file']})")
                else:
                    print(f"   Unknown list type: {list_type}")
                continue
            
            # Default: cascading search
            print(f"\n🔍 Searching: '{query}'")
            results = searcher.search(query)
            print_search_results(results)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"   ❌ Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="TAL Intelligence System v3.0")
    parser.add_argument("--db-path", default="./tal_index", help="Path to index")
    args = parser.parse_args()
    
    searcher = TalSearcherV3(db_path=args.db_path)
    try:
        interactive_search(searcher)
    finally:
        searcher.close()


if __name__ == "__main__":
    main()
