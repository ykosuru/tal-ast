#!/usr/bin/env python3
"""
TAL Code Intelligence Searcher v2.0
====================================
Interactive search interface for TAL codebase

Features:
- Hybrid search (BM25 + Vector + RRF fusion)
- TRACE: Call graph analysis
- USAGE: Data reference tracking
- EXPLAIN: Procedure context with business rules
- STRUCT: Structure definition display
- LIST: Browse procedures, defines, literals, rules

Usage:
    python tal_searcher_v2.py --db-path ./tal_index
#YK123
"""

import re
import os
import json
import torch
import argparse
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass
from collections import defaultdict

from transformers import AutoModel, AutoTokenizer
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi


@dataclass
class SearchResult:
    """Search result container"""
    score: float
    result_type: str  # 'procedure', 'symbol', 'struct', 'rule'
    name: str
    file: str
    line: int
    text: str
    context: str
    match_type: str  # 'exact', 'hybrid'


class TalSearcherV2:
    """
    Production-grade TAL code searcher with hybrid search.
    """
    
    def __init__(self, db_path: str = "./tal_index", collection_name: str = "tal"):
        print("🔍 Loading TAL Intelligence System v2.0...")
        
        # Embedding model
        model_name = 'nomic-ai/nomic-embed-text-v1.5'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).eval()
        
        # Vector DB
        self.db_path = db_path
        self.qdrant = QdrantClient(path=db_path)
        self.logic_coll = f"{collection_name}_logic"
        self.symbol_coll = f"{collection_name}_symbols"
        
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
        
        print(f"   Loaded {len(self.procedures)} procedures, {len(self.symbols)} symbols")
        print(f"   Loaded {len(self.structs)} structs, {len(self.business_rules)} business rules")
    
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
        
        # All known names for suggestions
        self.all_proc_names = set(self.proc_by_name.keys())
        self.all_symbol_names = set(self.symbol_by_name.keys())
        self.all_struct_names = set(self.struct_by_name.keys())
    
    def _build_bm25_indices(self):
        """Build BM25 indices"""
        # Procedures: name + code
        proc_texts = [
            f"{p.get('name', '')} {p.get('code', '')}"
            for p in self.procedures
        ]
        self.proc_bm25 = BM25Okapi([self._tokenize(t) for t in proc_texts]) if proc_texts else None
        
        # Symbols: name + type
        symbol_texts = [
            f"{s['name']} {s.get('data_type', '')} {s.get('section', '')}"
            for s in self.symbols
        ]
        self.symbol_bm25 = BM25Okapi([self._tokenize(t) for t in symbol_texts]) if symbol_texts else None
        
        # Structs: name + fields
        struct_texts = [
            f"{s['name']} STRUCT {' '.join(f.get('name', '') for f in s.get('fields', []))}"
            for s in self.structs
        ]
        self.struct_bm25 = BM25Okapi([self._tokenize(t) for t in struct_texts]) if struct_texts else None
        
        # Business rules
        rule_texts = [
            f"{r.get('rule_type', '')} {r.get('description', '')} {r.get('source_code', '')}"
            for r in self.business_rules
        ]
        self.rule_bm25 = BM25Okapi([self._tokenize(t) for t in rule_texts]) if rule_texts else None
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        return re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_^]*\b', text.lower())
    
    def _embed_query(self, query: str) -> List[float]:
        """Embed query text"""
        prefixed = "search_query: " + query
        inputs = self.tokenizer(
            [prefixed],
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=512
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = torch.nn.functional.normalize(
                outputs.last_hidden_state.mean(dim=1),
                p=2, dim=1
            )
            return embedding[0].cpu().numpy().tolist()
    
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
    
    def _find_similar(self, query: str, candidates: Set[str]) -> List[str]:
        """Find similar names for suggestions"""
        query = query.upper().lstrip('.')
        scored = []
        
        for candidate in candidates:
            score = 0
            cand_clean = candidate.lstrip('.')
            
            if query in cand_clean:
                score += 50
            elif cand_clean in query:
                score += 40
            
            # Word overlap (handle ^ separator)
            query_parts = set(re.split(r'[_^]', query))
            cand_parts = set(re.split(r'[_^]', cand_clean))
            common = query_parts & cand_parts
            if common:
                score += len(common) * 10
            
            if score > 15:
                scored.append((candidate, score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return [item for item, _ in scored[:10]]
    
    def search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Main hybrid search"""
        query_upper = query.strip().upper()
        query_tokens = self._tokenize(query)
        
        results = []
        seen = set()
        
        # 1. EXACT MATCHES
        for token in query_tokens:
            token_upper = token.upper()
            
            # Check structs FIRST (highest priority for struct names)
            if token_upper in self.struct_by_name:
                struct = self.struct_by_name[token_upper]
                key = (struct['file'], struct['name'], struct['start_line'])
                if key not in seen:
                    field_count = len(struct.get('fields', []))
                    inline_tag = " (inline)" if struct.get('inline') else ""
                    results.append(SearchResult(
                        score=1.0,
                        result_type='struct',
                        name=struct['name'],
                        file=struct['file'],
                        line=struct['start_line'],
                        text=f"{field_count} fields{inline_tag}",
                        context="STRUCT definition",
                        match_type='exact'
                    ))
                    seen.add(key)
            
            # Check procedures
            if token_upper in self.proc_by_name:
                proc = self.proc_by_name[token_upper]
                key = (proc['file'], proc['name'], proc['start_line'])
                if key not in seen:
                    results.append(SearchResult(
                        score=1.0,
                        result_type='procedure',
                        name=proc['name'],
                        file=proc['file'],
                        line=proc['start_line'],
                        text=proc.get('code', '')[:200],
                        context=f"Type: {proc.get('proc_type', 'PROC')} | Calls: {len(proc.get('calls', []))}",
                        match_type='exact'
                    ))
                    seen.add(key)
            
            # Check symbols
            if token_upper in self.symbol_by_name:
                for sym in self.symbol_by_name[token_upper]:
                    key = (sym['file'], sym['name'], sym['line'])
                    if key not in seen:
                        results.append(SearchResult(
                            score=1.0,
                            result_type='symbol',
                            name=sym['name'],
                            file=sym['file'],
                            line=sym['line'],
                            text=f"{sym.get('data_type', 'N/A')}",
                            context=f"Section: {sym.get('section', 'GLOBAL')}",
                            match_type='exact'
                        ))
                        seen.add(key)
        
        # 2. HYBRID SEARCH
        query_vec = self._embed_query(query)
        
        # Procedures
        if self.proc_bm25:
            bm25_scores = self.proc_bm25.get_scores(query_tokens)
            bm25_ranked = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)[:20]
            bm25_results = [(idx, score) for idx, score in bm25_ranked if score > 0.1]
            
            vec_results = self.qdrant.query_points(self.logic_coll, query=query_vec, limit=20).points
            vector_results = [(p.id, p.score) for p in vec_results]
            
            fused = self._rrf_fusion(bm25_results, vector_results)
            
            for idx, rrf_score in fused[:top_k]:
                if 0 <= idx < len(self.procedures):
                    proc = self.procedures[idx]
                    key = (proc['file'], proc['name'], proc.get('start_line', 0))
                    if key not in seen:
                        results.append(SearchResult(
                            score=rrf_score,
                            result_type='procedure',
                            name=proc['name'],
                            file=proc['file'],
                            line=proc.get('start_line', 1),
                            text=proc.get('code', '')[:200],
                            context=f"Type: {proc.get('proc_type', 'PROC')}",
                            match_type='hybrid'
                        ))
                        seen.add(key)
        
        # Symbols
        if self.symbol_bm25:
            sym_bm25_scores = self.symbol_bm25.get_scores(query_tokens)
            sym_bm25_ranked = sorted(enumerate(sym_bm25_scores), key=lambda x: x[1], reverse=True)[:10]
            sym_bm25_results = [(idx, score) for idx, score in sym_bm25_ranked if score > 0.1]
            
            sym_vec = self.qdrant.query_points(self.symbol_coll, query=query_vec, limit=10).points
            sym_vector_results = [(p.id - 1000000, p.score) for p in sym_vec if p.id >= 1000000]
            
            fused_sym = self._rrf_fusion(sym_bm25_results, sym_vector_results)
            
            for idx, rrf_score in fused_sym[:5]:
                if 0 <= idx < len(self.symbols):
                    sym = self.symbols[idx]
                    key = (sym['file'], sym['name'], sym['line'])
                    if key not in seen:
                        results.append(SearchResult(
                            score=rrf_score,
                            result_type='symbol',
                            name=sym['name'],
                            file=sym['file'],
                            line=sym['line'],
                            text=sym.get('data_type', 'N/A'),
                            context=f"Section: {sym.get('section', 'GLOBAL')}",
                            match_type='hybrid'
                        ))
                        seen.add(key)
        
        # Structs (BM25 only - no vector index)
        if self.struct_bm25:
            struct_bm25_scores = self.struct_bm25.get_scores(query_tokens)
            struct_bm25_ranked = sorted(enumerate(struct_bm25_scores), key=lambda x: x[1], reverse=True)[:5]
            
            for idx, score in struct_bm25_ranked:
                if score > 0.1 and 0 <= idx < len(self.structs):
                    struct = self.structs[idx]
                    key = (struct['file'], struct['name'], struct['start_line'])
                    if key not in seen:
                        field_count = len(struct.get('fields', []))
                        inline_tag = " (inline)" if struct.get('inline') else ""
                        results.append(SearchResult(
                            score=score * 0.5,  # Scale down BM25-only score
                            result_type='struct',
                            name=struct['name'],
                            file=struct['file'],
                            line=struct['start_line'],
                            text=f"{field_count} fields{inline_tag}",
                            context="STRUCT definition",
                            match_type='hybrid'
                        ))
                        seen.add(key)
        
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]
    
    def find_data_usage(self, symbol: str) -> Tuple[List[Dict], List[str]]:
        """Find all places where a symbol is used"""
        sym_upper = symbol.strip().upper().lstrip('.')
        
        if sym_upper in self.data_usage:
            return self.data_usage[sym_upper], []
        
        suggestions = self._find_similar(sym_upper, set(self.data_usage.keys()))
        return [], suggestions
    
    def trace_procedure(self, proc_name: str) -> Dict:
        """Trace procedure call relationships"""
        name_upper = proc_name.strip().upper()
        
        if name_upper not in self.call_graph:
            suggestions = self._find_similar(name_upper, self.all_proc_names)
            return {'error': f"Procedure '{proc_name}' not found", 'suggestions': suggestions}
        
        node = self.call_graph[name_upper]
        proc_info = self.proc_by_name.get(name_upper, {})
        
        # Get direct callers and callees
        direct_callers = node.get('callers', [])
        direct_callees = node.get('callees', [])  # Fixed: was 'targets'
        
        # Calculate impact (indirect callers)
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
            'type': node.get('proc_type', node.get('type')),  # Fixed: check both field names
            'return_type': proc_info.get('return_type'),
            'attributes': proc_info.get('attributes', []),
            'direct_callers': direct_callers,
            'direct_callees': direct_callees,
            'all_callers': list(all_callers - set(direct_callers)),
            'impact_count': len(all_callers)
        }
    
    def explain_procedure(self, proc_name: str) -> Dict:
        """Get full context for a procedure"""
        name_upper = proc_name.strip().upper()
        
        if name_upper not in self.proc_by_name:
            suggestions = self._find_similar(name_upper, self.all_proc_names)
            return {'error': f"Procedure '{proc_name}' not found", 'suggestions': suggestions}
        
        proc = self.proc_by_name[name_upper]
        rules = self.rules_by_proc.get(name_upper, [])
        
        return {
            'procedure': proc,
            'business_rules': rules,
            'call_info': self.trace_procedure(proc_name),
            'data_refs': proc.get('data_refs', [])
        }
    
    def get_struct(self, struct_name: str) -> Dict:
        """Get struct definition with fields"""
        name_upper = struct_name.strip().upper()
        
        if name_upper not in self.struct_by_name:
            suggestions = self._find_similar(name_upper, self.all_struct_names)
            return {'error': f"Struct '{struct_name}' not found", 'suggestions': suggestions}
        
        return self.struct_by_name[name_upper]
    
    def list_procedures(self, pattern: str = None) -> List[Dict]:
        """List procedures, optionally filtered by pattern"""
        if not pattern:
            return [{'name': p['name'], 'file': p['file'], 'type': p['proc_type']} 
                    for p in self.procedures]
        
        pattern_upper = pattern.upper()
        return [
            {'name': p['name'], 'file': p['file'], 'type': p['proc_type']}
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
    
    def list_business_rules(self, rule_type: str = None) -> List[Dict]:
        """List business rules by type"""
        if not rule_type:
            return self.business_rules
        
        type_upper = rule_type.upper()
        return [r for r in self.business_rules if r.get('rule_type', '').upper() == type_upper]
    
    def ask_question(self, question: str, debug: bool = False) -> Dict:
        """
        Answer a natural language question about business logic.
        Synthesizes information from procedures, rules, and data.
        """
        question_lower = question.lower()
        question_upper = question.upper()
        
        if debug:
            print(f"DEBUG: question = '{question}'")
            print(f"DEBUG: proc_by_name has {len(self.proc_by_name)} entries")
            print(f"DEBUG: call_graph has {len(self.call_graph)} entries")
        
        # Extract key words for matching
        words = re.findall(r'[A-Za-z_][A-Za-z0-9_]*', question)
        words_upper = [w.upper() for w in words]
        
        if debug:
            print(f"DEBUG: words = {words_upper}")
        
        # First, check if question contains a specific procedure name
        potential_proc_names = []
        
        # Direct match: check if any word is a known procedure
        for word in words_upper:
            if word in self.proc_by_name:
                potential_proc_names.append(word)
        
        # Try to build compound procedure name from words
        # e.g., "validate credit party" -> "VALIDATE_CREDIT_PARTY"
        if len(words) >= 2:
            compound_name = '_'.join(w.upper() for w in words if len(w) > 2)
            if compound_name in self.proc_by_name:
                potential_proc_names.insert(0, compound_name)
            
            # Try subsets of consecutive words
            for start in range(len(words)):
                for end in range(start + 2, min(start + 6, len(words) + 1)):
                    subset = '_'.join(w.upper() for w in words[start:end] if len(w) > 1)
                    if subset in self.proc_by_name:
                        potential_proc_names.insert(0, subset)
        
        if debug:
            print(f"DEBUG: after compound matching, potential_proc_names = {potential_proc_names}")
        
        # Score-based matching for procedures
        if not potential_proc_names:
            scored_procs = []
            significant_words = [w.upper() for w in words if len(w) > 3]
            
            # Also extract special domain keywords even if short
            domain_keywords = {'IBAN', 'BIC', 'ABA', 'OFAC', 'AML', 'KYC', 'EDD', 'SWIFT', 'ACH'}
            for w in words:
                if w.upper() in domain_keywords:
                    significant_words.append(w.upper())
            
            significant_words = list(set(significant_words))  # Remove duplicates
            
            if debug:
                print(f"DEBUG: significant_words = {significant_words}")
            
            for proc_name, proc in self.proc_by_name.items():
                score = 0
                proc_parts = set(proc_name.split('_'))
                
                # Match in procedure name
                for word in significant_words:
                    if word in proc_parts:
                        score += 10  # Exact part match
                    elif word in proc_name:
                        score += 5   # Substring match
                
                # Match in procedure calls (e.g., IBAN -> VALIDATE_IBAN_FORMAT)
                calls = proc.get('calls', [])
                if calls:
                    # Handle both list of strings and list of dicts
                    if isinstance(calls[0], dict):
                        calls_text = ' '.join(c.get('target', '') for c in calls).upper()
                    else:
                        calls_text = ' '.join(str(c) for c in calls).upper()
                    
                    for word in significant_words:
                        if word in calls_text:
                            score += 8  # Found in calls
                
                # Match in step descriptions
                steps = proc.get('validation_steps', [])
                if steps:
                    steps_text = ' '.join(s.get('description', '') for s in steps).upper()
                    for word in significant_words:
                        if word in steps_text:
                            score += 7  # Found in step descriptions
                
                # Match in code
                code_text = proc.get('raw_code', proc.get('code', '')).upper()
                for word in significant_words:
                    if len(word) >= 4 and word in code_text:
                        score += 3  # Found in code
                
                # Penalize if query has "credit" but proc has "debit" (or vice versa)
                if 'CREDIT' in significant_words and 'DEBIT' in proc_name:
                    score -= 20
                if 'DEBIT' in significant_words and 'CREDIT' in proc_name:
                    score -= 20
                
                if score > 0:
                    scored_procs.append((proc_name, score))
                    if debug and score > 5:
                        print(f"DEBUG: {proc_name} score={score}")
            
            # Sort by score descending
            scored_procs.sort(key=lambda x: x[1], reverse=True)
            potential_proc_names = [p[0] for p in scored_procs[:5]]
        
        if debug:
            print(f"DEBUG: after scoring, potential_proc_names = {potential_proc_names}")
        
        # Additional fallback: search for domain keywords in all procedures
        if not potential_proc_names:
            domain_keywords_found = [w.upper() for w in words if w.upper() in 
                                    {'IBAN', 'BIC', 'ABA', 'OFAC', 'AML', 'KYC', 'SWIFT', 'SANCTION', 
                                     'CORRESPONDENT', 'BENEFICIARY', 'ROUTING', 'ACCOUNT', 'BALANCE'}]
            if debug:
                print(f"DEBUG: domain_keywords_found = {domain_keywords_found}")
            if domain_keywords_found:
                for proc_name, proc in self.proc_by_name.items():
                    # Search in calls
                    calls = proc.get('calls', [])
                    calls_text = ''
                    if calls:
                        if isinstance(calls[0], dict):
                            calls_text = ' '.join(c.get('target', '') for c in calls).upper()
                        else:
                            calls_text = ' '.join(str(c) for c in calls).upper()
                    
                    for kw in domain_keywords_found:
                        if kw in calls_text or kw in proc_name:
                            if proc_name not in potential_proc_names:
                                potential_proc_names.append(proc_name)
                                if debug:
                                    print(f"DEBUG: domain match: {proc_name} (keyword={kw})")
                            break
        
        # Build list of relevant procedures
        relevant_procs = []
        for proc_name in potential_proc_names:
            if proc_name in self.proc_by_name:
                relevant_procs.append(self.proc_by_name[proc_name])
        
        if debug:
            print(f"DEBUG: relevant_procs count = {len(relevant_procs)}")
        
        # Fall back to semantic search if no direct matches
        if not relevant_procs:
            try:
                search_results = self.search(question, top_k=5)
                if debug:
                    print(f"DEBUG: semantic search returned {len(search_results)} results")
                for r in search_results:
                    if r.result_type == 'procedure' and r.name in self.proc_by_name:
                        # Apply credit/debit filter
                        if 'credit' in question_lower and 'DEBIT' in r.name:
                            continue
                        if 'debit' in question_lower and 'CREDIT' in r.name:
                            continue
                        relevant_procs.append(self.proc_by_name[r.name])
            except Exception as e:
                if debug:
                    print(f"DEBUG: semantic search error: {e}")
                pass  # Silently fail and continue
        
        # Last resort: just get any validation procedure
        if not relevant_procs:
            if debug:
                print("DEBUG: using last resort - any VALIDATE procedure")
            for proc_name in self.proc_by_name:
                if 'VALIDATE' in proc_name:
                    relevant_procs.append(self.proc_by_name[proc_name])
                    if len(relevant_procs) >= 3:
                        break
        
        if not relevant_procs:
            if debug:
                print("DEBUG: NO procedures found at all!")
                print(f"DEBUG: proc_by_name keys sample: {list(self.proc_by_name.keys())[:5]}")
            return {
                'answer': "I couldn't find relevant procedures for your question.",
                'suggestions': list(self.proc_by_name.keys())[:10]
            }
        
        # Build comprehensive answer
        primary_proc = relevant_procs[0]
        proc_name = primary_proc['name']
        
        # Get all context
        rules = self.rules_by_proc.get(proc_name, [])
        calls = primary_proc.get('calls', [])
        data_refs = primary_proc.get('data_refs', [])
        
        # Use pre-indexed validation steps if available, otherwise extract
        steps = primary_proc.get('validation_steps', [])
        if not steps:
            # Fallback: try to extract from raw_code if available
            raw_code = primary_proc.get('raw_code', primary_proc.get('code', ''))
            steps = self._extract_validation_steps_dynamic(raw_code)
        
        # Extract error codes and their meanings
        error_codes = self._extract_error_codes(primary_proc)
        
        # For procedures without steps, try to extract logic from code
        logic_summary = []
        if not steps:
            logic_summary = self._extract_logic_summary(primary_proc)
        
        # Build the answer
        answer = {
            'question': question,
            'primary_procedure': proc_name,
            'file': primary_proc['file'],
            'proc_type': primary_proc.get('proc_type', 'PROC'),
            'parameters': primary_proc.get('parameters', []),
            'summary': self._generate_summary(primary_proc, steps),
            'steps': steps,
            'logic': logic_summary,
            'validations': [r for r in rules if r['rule_type'] == 'VALIDATION'][:10],
            'error_codes': error_codes,
            'sub_procedures': [c['target'] for c in calls],
            'data_used': data_refs[:20],
            'related_procedures': [p['name'] for p in relevant_procs[1:5]],
            'focused_answer': self._generate_focused_answer(question, steps, error_codes, primary_proc)
        }
        
        return answer
    
    def _generate_focused_answer(self, question: str, steps: List[Dict], 
                                  error_codes: List[Dict], proc: Dict) -> Dict:
        """Generate a focused answer highlighting the most relevant step"""
        question_lower = question.lower()
        question_words = set(re.findall(r'[a-z]+', question_lower))
        
        # Skip focused answer for general overview questions
        # e.g., "how to validate credit party" should show all steps, not one
        overview_patterns = [
            r'^how\s+to\s+(validate|implement|process|handle)',
            r'^what\s+are\s+the\s+steps',
            r'^what\s+is\s+the\s+process',
            r'^explain\s+',
            r'^describe\s+',
        ]
        for pattern in overview_patterns:
            if re.search(pattern, question_lower):
                # Check if question is about the procedure as a whole (not a specific topic)
                proc_name_lower = proc.get('name', '').lower().replace('_', ' ')
                if any(word in proc_name_lower for word in question_words if len(word) > 4):
                    # Question matches procedure name - show overview, not focused step
                    return None
        
        # Keywords that indicate specific topics
        topic_keywords = {
            'iban': ['iban', 'account', 'number'],
            'bic': ['bic', 'swift', 'routing'],
            'ofac': ['ofac', 'sanction', 'sanctions', 'screening'],
            'aml': ['aml', 'money', 'laundering', 'monitoring'],
            'kyc': ['kyc', 'know', 'customer', 'compliance'],
            'name': ['name', 'beneficiary', 'party'],
            'address': ['address', 'city', 'country', 'postal'],
            'amount': ['amount', 'currency', 'limit', 'balance'],
            'correspondent': ['correspondent', 'bank', 'nostro'],
            'format': ['format', 'message', 'invalid'],
            'domestic': ['domestic', 'aba', 'fedwire'],
            'international': ['international', 'cross', 'border'],
        }
        
        # Find which topic the question is about
        matched_topics = []
        for topic, keywords in topic_keywords.items():
            if any(kw in question_lower for kw in keywords):
                matched_topics.append(topic)
        
        # Score each step based on keyword matches
        best_step = None
        best_score = 0
        
        for step in steps:
            score = 0
            step_text = (step.get('description', '') + ' ' + 
                        ' '.join(step.get('calls', []))).lower()
            
            # Check for topic matches
            for topic in matched_topics:
                if topic in step_text:
                    score += 10
                for kw in topic_keywords.get(topic, []):
                    if kw in step_text:
                        score += 5
            
            # Check for direct word matches
            for word in question_words:
                if len(word) > 3 and word in step_text:
                    score += 3
            
            if score > best_score:
                best_score = score
                best_step = step
        
        if not best_step or best_score < 5:
            return None
        
        # Find relevant error codes for this step
        step_errors = best_step.get('errors', [])
        relevant_errors = [e for e in error_codes if e['code'] in step_errors]
        
        # Determine outcome based on error types
        outcome = "VALIDATION_STATUS := 2 (REJECTED)"
        if relevant_errors and all(e['type'] == 'WARNING' for e in relevant_errors):
            outcome = "VALIDATION_STATUS := 1 (WARNING) - continues with flag"
        
        # Generate focused answer text
        focus_text = self._generate_focus_text(question_lower, best_step, relevant_errors)
        
        return {
            'step': best_step,
            'step_num': best_step.get('step_num', '?'),
            'description': best_step.get('description', ''),
            'calls': best_step.get('calls', []),
            'errors': relevant_errors,
            'outcome': outcome,
            'explanation': focus_text,
            'score': best_score
        }
    
    def _generate_focus_text(self, question: str, step: Dict, errors: List[Dict]) -> str:
        """Generate natural language explanation for the focused answer"""
        desc = step.get('description', '').lower()
        calls = step.get('calls', [])
        
        # Detect question type
        if 'what happens' in question or 'when' in question:
            if errors:
                error_types = set(e['type'] for e in errors)
                if 'ERROR' in error_types:
                    return "The transaction is REJECTED and cannot proceed."
                else:
                    return "A WARNING is raised but the transaction may continue with additional review."
            return "The validation step is executed."
        
        elif 'how' in question:
            if calls:
                return f"This is handled by calling {', '.join(calls[:3])}."
            return "This is handled in the validation step shown."
        
        elif 'why' in question:
            if errors:
                return f"To ensure compliance and catch issues like: {errors[0]['message']}"
            return "To ensure the transaction meets all requirements."
        
        else:
            if errors:
                return f"If validation fails, error codes {', '.join(e['code'] for e in errors[:3])} are raised."
            return "This step validates the relevant data."
    
    def _extract_logic_summary(self, proc: Dict) -> List[Dict]:
        """Extract logic conditions from a procedure"""
        # Try raw_code first, fall back to code
        code = proc.get('raw_code', '') or proc.get('code', '')
        if not code:
            return []
        
        logic = []
        lines = code.splitlines()
        
        # Track nested IF depth for context
        current_context = []  # Stack of conditions
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            line_upper = line.upper()
            
            # Skip comments
            if line.startswith('!'):
                i += 1
                continue
            
            # Track ELSE branches
            if line_upper.startswith('ELSE'):
                if current_context:
                    # We're in an ELSE branch
                    pass
            
            # Look for IF conditions
            if re.search(r'\bIF\b', line_upper) and 'ENDIF' not in line_upper:
                # Handle multi-line IF (condition may span lines)
                full_line = line
                j = i
                while 'THEN' not in full_line.upper() and j < len(lines) - 1:
                    j += 1
                    full_line += ' ' + lines[j].strip()
                
                # Extract condition
                if_match = re.search(r'\bIF\s+(.+?)\s+THEN\b', full_line, re.IGNORECASE | re.DOTALL)
                if if_match:
                    condition = if_match.group(1).strip()
                    condition = re.sub(r'\s+', ' ', condition)  # Normalize whitespace
                    
                    # Find the action
                    action = self._find_action_after_then(lines, j, full_line)
                    
                    logic.append({
                        'condition': condition[:80],
                        'action': action
                    })
                
                i = j + 1
                continue
            
            # Look for CASE statements
            if 'CASE ' in line_upper and ' OF' in line_upper:
                case_match = re.search(r'CASE\s+(\w+)', line, re.IGNORECASE)
                if case_match:
                    logic.append({
                        'condition': f"CASE {case_match.group(1)}",
                        'action': '(multiple branches - see code)'
                    })
            
            i += 1
        
        # If no IF logic found, show key assignments
        if not logic:
            logic = self._extract_key_assignments(lines)
        
        return logic[:12]  # Limit to 12 items
    
    def _find_action_after_then(self, lines: List[str], then_line_idx: int, then_line: str) -> str:
        """Find what happens after THEN"""
        # Check if action is on same line after THEN
        # Handle both "THEN action" and "THEN" at end of line
        then_match = re.search(r'\bTHEN\s*(.*)$', then_line, re.IGNORECASE)
        if then_match:
            after_then = then_match.group(1).strip().rstrip(';')
            
            # If nothing after THEN, look at next line
            if not after_then or after_then.startswith('!'):
                # Fall through to next line handling below
                pass
            elif after_then.upper() == 'BEGIN':
                # Look inside BEGIN block for first meaningful action
                for k in range(then_line_idx + 1, min(then_line_idx + 15, len(lines))):
                    inner = lines[k].strip()
                    inner_upper = inner.upper()
                    
                    # Skip empty lines, comments, BEGIN
                    if not inner or inner.startswith('!') or inner_upper == 'BEGIN':
                        continue
                    
                    # Stop at END
                    if inner_upper in ('END', 'END;'):
                        break
                    
                    # Found an assignment
                    if ':=' in inner and 'IF ' not in inner_upper:
                        m = re.search(r'(\w+)\s*:=\s*(.+?)(?:;|$)', inner)
                        if m:
                            return f"{m.group(1)} := {m.group(2).strip()[:25]}"
                    
                    # Found a CALL
                    if inner_upper.startswith('CALL '):
                        cm = re.search(r'CALL\s+(\w+)', inner, re.IGNORECASE)
                        if cm:
                            return f"CALL {cm.group(1)}"
                    
                    # Found a nested IF - describe it
                    if inner_upper.startswith('IF '):
                        return "(nested conditions)"
                    
                    # Found something else meaningful
                    if inner_upper not in ('ELSE', 'ELSE;'):
                        return inner[:40]
                
                return "(code block)"
            
            elif ':=' in after_then:
                m = re.search(r'(\w+)\s*:=\s*(.+?)(?:;|$)', after_then)
                if m:
                    return f"{m.group(1)} := {m.group(2).strip()[:25]}"
            elif after_then.upper().startswith('CALL '):
                cm = re.search(r'CALL\s+(\w+)', after_then, re.IGNORECASE)
                if cm:
                    return f"CALL {cm.group(1)}"
            elif after_then:
                return after_then[:40]
        
        # Look at next lines for action
        for offset in range(1, 8):
            if then_line_idx + offset >= len(lines):
                break
            next_line = lines[then_line_idx + offset].strip()
            next_upper = next_line.upper()
            
            # Skip empty/comments
            if not next_line or next_line.startswith('!'):
                continue
            
            if ':=' in next_line and 'IF ' not in next_upper:
                m = re.search(r'(\w+)\s*:=\s*(.+?)(?:;|$)', next_line)
                if m:
                    return f"{m.group(1)} := {m.group(2).strip()[:25]}"
            elif next_upper.startswith('CALL '):
                cm = re.search(r'CALL\s+(\w+)', next_line, re.IGNORECASE)
                if cm:
                    return f"CALL {cm.group(1)}"
            elif next_upper == 'BEGIN':
                # Look into BEGIN block
                for k in range(then_line_idx + offset + 1, min(then_line_idx + offset + 15, len(lines))):
                    inner = lines[k].strip()
                    inner_upper = inner.upper()
                    if not inner or inner.startswith('!') or inner_upper == 'BEGIN':
                        continue
                    if inner_upper in ('END', 'END;'):
                        break
                    if ':=' in inner and 'IF ' not in inner_upper:
                        m = re.search(r'(\w+)\s*:=\s*(.+?)(?:;|$)', inner)
                        if m:
                            return f"{m.group(1)} := {m.group(2).strip()[:25]}"
                    if inner_upper.startswith('CALL '):
                        cm = re.search(r'CALL\s+(\w+)', inner, re.IGNORECASE)
                        if cm:
                            return f"CALL {cm.group(1)}"
                    if inner_upper.startswith('IF '):
                        return "(nested conditions)"
                return "(code block)"
            elif next_upper not in ('END', 'END;', 'ELSE', 'ELSE;'):
                return next_line[:35]
        
        return "(see code)"
    
    def _extract_key_assignments(self, lines: List[str]) -> List[Dict]:
        """Extract key variable assignments when no IF logic found"""
        assignments = []
        for line in lines:
            line_stripped = line.strip()
            if line_stripped.startswith('!'):
                continue
            if ':=' in line_stripped:
                m = re.search(r'(\w+)\s*:=\s*(.+?)(?:;|$)', line_stripped)
                if m:
                    var_name = m.group(1)
                    value = m.group(2).strip()[:40]
                    # Skip temp/loop variables
                    if not var_name.upper().startswith(('TEMP', 'I', 'J', 'K', 'IDX', 'CNT')):
                        assignments.append({
                            'condition': 'Assignment',
                            'action': f"{var_name} := {value}"
                        })
        return assignments[:8]
    
    def _extract_validation_steps_dynamic(self, code: str) -> List[Dict]:
        """Extract validation steps from code (fallback for older indexes)"""
        lines = code.splitlines()
        steps = []
        current_step = None
        
        for line in lines:
            line_stripped = line.strip()
            
            # Pattern: ! STEP N: DESCRIPTION
            step_match = re.search(
                r'^[!\*\s]*(?:=+\s*)?STEP\s*(\d+)\s*[:\.\-]\s*(.+?)(?:\s*=+)?$',
                line_stripped, re.IGNORECASE
            )
            if step_match:
                if current_step:
                    steps.append(current_step)
                current_step = {
                    'step_num': int(step_match.group(1)),
                    'description': step_match.group(2).strip().rstrip('=*-! '),
                    'errors': [],
                    'calls': []
                }
                continue
            
            if current_step:
                # Error codes
                for error_match in re.finditer(r'["\']([A-Z]{2,4}\d{3})["\']', line):
                    code = error_match.group(1)
                    if code not in current_step['errors']:
                        current_step['errors'].append(code)
                
                # CALL statements
                call_match = re.search(r'\bCALL\s+([A-Za-z_]\w*)', line, re.IGNORECASE)
                if call_match:
                    call_name = call_match.group(1).upper()
                    if call_name not in current_step['calls']:
                        current_step['calls'].append(call_name)
        
        if current_step:
            steps.append(current_step)
        
        # If no steps found, infer from calls
        if not steps:
            steps = self._infer_steps_from_code({'code': code})
        
        return steps
    
    def _infer_steps_from_code(self, proc: Dict) -> List[Dict]:
        """Infer validation steps from code when comments aren't present"""
        code = proc.get('code', '')
        steps = []
        
        # Look for major CALL patterns that indicate steps
        call_pattern = re.compile(r'\bCALL\s+([A-Za-z_]\w*)', re.IGNORECASE)
        error_pattern = re.compile(r'["\']([A-Z]{2,4}\d{3})["\']')
        
        # Group calls by their prefix to infer steps
        call_groups = {}
        for match in call_pattern.finditer(code):
            call_name = match.group(1).upper()
            # Skip helper calls
            if call_name in ('ADD_VALIDATION_ERROR', 'ADD_VALIDATION_WARNING', 'LOG_VALIDATION_RESULT'):
                continue
            
            # Group by first word
            prefix = call_name.split('_')[0] if '_' in call_name else call_name
            if prefix not in call_groups:
                call_groups[prefix] = []
            if call_name not in call_groups[prefix]:
                call_groups[prefix].append(call_name)
        
        # Create steps from groups
        step_num = 1
        for prefix, calls in call_groups.items():
            steps.append({
                'step_num': step_num,
                'description': f"{prefix.title()} Validation",
                'validations': [],
                'errors': [],
                'calls': calls
            })
            step_num += 1
        
        return steps
    
    def _extract_error_codes(self, proc: Dict) -> List[Dict]:
        """Extract error codes and their messages from procedure"""
        code = proc.get('code', '')
        error_codes = []
        
        # Pattern: ADD_VALIDATION_ERROR("CODE", "message", ...)
        pattern = r'ADD_VALIDATION_(?:ERROR|WARNING)\s*\(\s*["\']([A-Z]{2,4}\d{3})["\']\s*,\s*["\']([^"\']+)["\']'
        
        for match in re.finditer(pattern, code, re.IGNORECASE):
            error_codes.append({
                'code': match.group(1),
                'message': match.group(2),
                'type': 'ERROR' if 'ERROR' in match.group(0).upper() else 'WARNING'
            })
        
        # Deduplicate
        seen = set()
        unique = []
        for ec in error_codes:
            if ec['code'] not in seen:
                seen.add(ec['code'])
                unique.append(ec)
        
        return unique
    
    def _generate_summary(self, proc: Dict, steps: List[Dict]) -> str:
        """Generate a human-readable summary of the procedure"""
        name = proc['name']
        proc_type = proc.get('proc_type', 'PROC')
        params = proc.get('parameters', [])
        
        # Infer purpose from name
        name_parts = re.split(r'[_^]', name.lower())
        action = name_parts[0] if name_parts else 'process'
        subject = ' '.join(name_parts[1:]) if len(name_parts) > 1 else 'data'
        
        # Build summary
        summary_lines = []
        
        # Fix grammar for common verbs
        action_verb = action
        if action.endswith('fy'):
            action_verb = action[:-1] + 'ies'  # classify -> classifies
        elif action.endswith('y') and len(action) > 2 and action[-2] not in 'aeiou':
            action_verb = action[:-1] + 'ies'  # verify -> verifies
        elif action.endswith('ss') or action.endswith('sh') or action.endswith('ch') or action.endswith('x'):
            action_verb = action + 'es'  # process -> processes
        elif action.endswith('e'):
            action_verb = action + 's'  # validate -> validates
        else:
            action_verb = action + 's'  # check -> checks
        
        # Type indicator
        if proc_type == 'SUBPROC':
            summary_lines.append(f"This is a SUBPROC (local procedure) that {action_verb} {subject}.")
        else:
            # Main description based on action verb
            if 'validate' in action.lower():
                summary_lines.append(f"This procedure validates the {subject} information.")
            elif 'classify' in action.lower():
                summary_lines.append(f"This procedure classifies the {subject}.")
            elif 'check' in action.lower():
                summary_lines.append(f"This procedure checks {subject} conditions.")
            elif 'add' in action.lower():
                summary_lines.append(f"This procedure adds {subject} to a collection.")
            elif 'normalize' in action.lower():
                summary_lines.append(f"This procedure normalizes {subject} format.")
            elif 'format' in action.lower():
                summary_lines.append(f"This procedure formats {subject}.")
            else:
                summary_lines.append(f"This procedure handles {subject}.")
        
        # Parameters
        if params:
            summary_lines.append(f"\nParameters: {', '.join(params)}")
        
        # Steps summary (if any)
        if steps:
            summary_lines.append(f"\nIt performs {len(steps)} main validation steps:")
            for step in steps[:8]:
                desc = step.get('description', '').strip()
                if desc:
                    error_count = len(step.get('errors', []))
                    error_note = f" ({error_count} error codes)" if error_count else ""
                    summary_lines.append(f"  {step['step_num']}. {desc}{error_note}")
            if len(steps) > 8:
                summary_lines.append(f"  ... and {len(steps) - 8} more steps")
        
        # Key calls (if no steps, show calls as the main content)
        calls = proc.get('calls', [])
        if calls:
            # Filter out helper procedures
            key_calls = [c['target'] for c in calls 
                        if c['target'] not in ('ADD_VALIDATION_ERROR', 'ADD_VALIDATION_WARNING', 'LOG_VALIDATION_RESULT')][:8]
            if key_calls:
                if not steps:
                    summary_lines.append(f"\nIt calls: {', '.join(key_calls)}")
                else:
                    summary_lines.append(f"\nKey operations: {', '.join(key_calls)}")
        
        # If no steps and no calls, describe from code patterns
        if not steps and not calls:
            code = proc.get('code', '')
            # Look for key patterns
            if 'IF ' in code.upper():
                summary_lines.append("\nContains conditional logic for decision-making.")
            if ':=' in code:
                summary_lines.append("Sets output values based on input conditions.")
        
        return '\n'.join(summary_lines)
    
    def close(self):
        """Close database connection"""
        self.qdrant.close()


def print_search_results(results: List[SearchResult]):
    """Pretty print search results"""
    if not results:
        print("   No results found")
        return
    
    # Group by type
    structs = [r for r in results if r.result_type == 'struct']
    procs = [r for r in results if r.result_type == 'procedure']
    symbols = [r for r in results if r.result_type == 'symbol']
    
    if structs:
        print("\n🏗️ STRUCTS:")
        for r in structs:
            print(f"   🎯 {r.name}")
            print(f"      {r.file}:L{r.line} | {r.text}")
    
    if symbols:
        print("\n📦 SYMBOLS:")
        for r in symbols:
            print(f"   🔹 {r.name}")
            print(f"      {r.file}:L{r.line} | {r.text} | {r.context}")
    
    if procs:
        print("\n📋 PROCEDURES:")
        for r in procs:
            print(f"   📍 {r.name}")
            print(f"      {r.file}:L{r.line} | {r.context}")
            code_preview = r.text.replace('\n', ' ')[:100]
            print(f"      Code: {code_preview}...")


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
    if trace.get('return_type'):
        print(f"   Returns: {trace['return_type']}")
    if trace.get('attributes'):
        print(f"   Attributes: {', '.join(trace['attributes'])}")
    
    print(f"\n   ⬆️ Called by ({len(trace['direct_callers'])}):")
    for caller in trace['direct_callers'][:10]:
        print(f"      • {caller}")
    if len(trace['direct_callers']) > 10:
        print(f"      ... and {len(trace['direct_callers']) - 10} more")
    
    print(f"\n   ⬇️ Calls ({len(trace['direct_callees'])}):")
    for callee in trace['direct_callees'][:10]:
        print(f"      • {callee}")
    if len(trace['direct_callees']) > 10:
        print(f"      ... and {len(trace['direct_callees']) - 10} more")
    
    if trace['all_callers']:
        print(f"\n   🌐 Impact: {trace['impact_count']} procedures in call chain")


def print_usage_result(usages: List[Dict], suggestions: List[str], symbol: str):
    """Pretty print usage results"""
    if suggestions:
        print(f"   ❌ Symbol '{symbol}' not found")
        print(f"   💡 Did you mean: {', '.join(suggestions[:5])}")
        return
    
    if not usages:
        print(f"   No usages found for '{symbol}'")
        return
    
    # Group by file
    by_file = defaultdict(list)
    for u in usages:
        by_file[u['file']].append(u)
    
    print(f"\n📊 Usage of '{symbol}' ({len(usages)} references):")
    for file, refs in sorted(by_file.items()):
        print(f"\n   📁 {file}:")
        for ref in refs:
            proc_type = ref.get('type', 'PROC')
            print(f"      • {ref['procedure']} ({proc_type}) L{ref['line']}")


def print_explain_result(explain: Dict):
    """Pretty print explain result"""
    if 'error' in explain:
        print(f"   ❌ {explain['error']}")
        if explain.get('suggestions'):
            print(f"   💡 Did you mean: {', '.join(explain['suggestions'][:5])}")
        return
    
    proc = explain['procedure']
    
    print(f"\n📖 EXPLAIN: {proc['name']}")
    print("=" * 60)
    
    # Basic info
    print(f"\n📍 Location: {proc['file']}:L{proc['start_line']}-{proc['end_line']}")
    print(f"   Type: {proc['proc_type']}")
    if proc.get('return_type'):
        print(f"   Returns: {proc['return_type']}")
    if proc.get('attributes'):
        print(f"   Attributes: {', '.join(proc['attributes'])}")
    if proc.get('parameters'):
        print(f"   Parameters: {', '.join(proc['parameters'])}")
    
    # Show actual code (prefer raw_code with comments)
    code = proc.get('raw_code', proc.get('code', ''))
    if code:
        lines = code.splitlines()
        print(f"\n📜 CODE ({len(lines)} lines):")
        print("-" * 60)
        # Show up to 50 lines
        for i, line in enumerate(lines[:50]):
            print(f"   {line}")
        if len(lines) > 50:
            print(f"   ... ({len(lines) - 50} more lines)")
        print("-" * 60)
    
    # Business rules
    rules = explain.get('business_rules', [])
    if rules:
        print(f"\n📝 Business Rules ({len(rules)}):")
        for r in rules[:10]:
            print(f"   [{r['rule_type']}] L{r['line']}: {r['source_code'][:80]}")
    
    # Data references
    data_refs = explain.get('data_refs', [])
    if data_refs:
        print(f"\n🔗 Data References ({len(data_refs)}):")
        print(f"   {', '.join(data_refs[:20])}")
        if len(data_refs) > 20:
            print(f"   ... and {len(data_refs) - 20} more")
    
    # Call info
    call_info = explain.get('call_info', {})
    if call_info and 'error' not in call_info:
        print(f"\n📞 Calls: {', '.join(call_info.get('direct_callees', [])[:10])}")
        print(f"   Called by: {', '.join(call_info.get('direct_callers', [])[:10])}")


def print_struct_result(struct: Dict):
    """Pretty print struct result"""
    if 'error' in struct:
        print(f"   ❌ {struct['error']}")
        if struct.get('suggestions'):
            print(f"   💡 Did you mean: {', '.join(struct['suggestions'][:5])}")
        return
    
    inline_tag = " (inline parameter)" if struct.get('inline') else ""
    print(f"\n🏗️ STRUCT: {struct['name']}{inline_tag}")
    print(f"   File: {struct['file']}:L{struct['start_line']}-{struct['end_line']}")
    
    fields = struct.get('fields', [])
    print(f"\n   Fields ({len(fields)}):")
    for f in fields:
        ptr_tag = " (pointer)" if f.get('is_pointer') else ""
        bounds = f" [{f.get('array_bounds')}]" if f.get('array_bounds') else ""
        print(f"      • {f['name']}: {f['data_type']}{bounds}{ptr_tag}")


def print_list_result(items: List[Dict], item_type: str):
    """Pretty print list result"""
    if not items:
        print(f"   No {item_type} found")
        return
    
    print(f"\n📋 {item_type.upper()} ({len(items)}):")
    for item in items[:50]:
        if 'type' in item and item.get('type') in ('PROC', 'SUBPROC'):  # Procedure
            print(f"   • {item['name']} ({item['file']}) - {item.get('type', 'PROC')}")
        elif 'proc_type' in item:  # Procedure from direct record
            print(f"   • {item['name']} ({item['file']}) - {item.get('proc_type', 'PROC')}")
        elif 'rule_type' in item:  # Business rule
            print(f"   [{item['rule_type']}] {item['procedure']} L{item.get('line', '?')}")
        else:  # Symbol (define/literal)
            defn = item.get('definition', '')[:60]
            line_info = f":L{item['line']}" if 'line' in item else ""
            print(f"   • {item['name']} ({item['file']}{line_info})")
            if defn:
                print(f"     {defn}")
    
    if len(items) > 50:
        print(f"\n   ... and {len(items) - 50} more")


def print_ask_result(result: Dict):
    """Pretty print ASK (business logic question) result"""
    if 'answer' in result and 'suggestions' in result:
        print(f"\n❓ {result.get('answer', 'No answer found')}")
        if result.get('suggestions'):
            print(f"   💡 Related: {', '.join(result['suggestions'][:5])}")
        return
    
    print(f"\n" + "=" * 70)
    print(f"📖 ANSWER: {result.get('question', '')}")
    print("=" * 70)
    
    # FOCUSED ANSWER - Show the most relevant finding first
    focused = result.get('focused_answer')
    if focused and focused.get('score', 0) >= 5:
        print(f"\n🎯 MOST RELEVANT: Step {focused['step_num']} - {focused['description']}")
        print("-" * 50)
        
        if focused.get('calls'):
            print(f"   📞 Calls: {', '.join(focused['calls'][:4])}")
        
        if focused.get('errors'):
            error_label = "When this fails:" if 'fail' in result.get('question', '').lower() or 'invalid' in result.get('question', '').lower() else "Possible outcomes:"
            print(f"\n   ⚠️  {error_label}")
            for e in focused['errors'][:5]:
                etype = "🚫" if e['type'] == 'ERROR' else "⚡"
                print(f"      {etype} {e['code']}: {e['message'][:55]}")
        
        if focused.get('explanation'):
            print(f"\n   💡 {focused['explanation']}")
        
        if focused.get('outcome'):
            print(f"   → Result: {focused['outcome']}")
        
        print("-" * 50)
    
    # Summary header
    proc_type = result.get('proc_type', 'PROC')
    params = result.get('parameters', [])
    file_info = result.get('file', '')
    
    print(f"\n📋 {result.get('primary_procedure', 'Unknown')} ({file_info})")
    if proc_type == 'SUBPROC':
        print(f"   Type: SUBPROC (local procedure)")
    if params:
        print(f"   Parameters: {', '.join(params)}")
    print("-" * 70)
    
    summary = result.get('summary', '')
    if summary:
        print(f"\n{summary}")
    
    # Logic conditions (for SUBPROCs and small procedures)
    logic = result.get('logic', [])
    if logic:
        print(f"\n\n🔀 DECISION LOGIC ({len(logic)} items):")
        print("-" * 50)
        for i, item in enumerate(logic, 1):
            cond = item.get('condition', '')
            action = item.get('action', '')
            if cond == 'Assignment':
                print(f"\n   {i}. {action}")
            elif cond.startswith('CASE '):
                print(f"\n   {i}. {cond}")
                print(f"      → {action}")
            else:
                print(f"\n   {i}. IF {cond}")
                print(f"      THEN → {action}")
    
    # Validation Steps
    steps = result.get('steps', [])
    if steps:
        print(f"\n\n🔢 VALIDATION STEPS ({len(steps)}):")
        print("-" * 50)
        for step in steps:
            step_num = step.get('step_num', '?')
            desc = step.get('description', 'No description')
            print(f"\n   Step {step_num}: {desc}")
            
            if step.get('calls'):
                calls_str = ', '.join(step['calls'][:6])
                if len(step['calls']) > 6:
                    calls_str += f" (+{len(step['calls'])-6} more)"
                print(f"      📞 Calls: {calls_str}")
            if step.get('errors'):
                errors_str = ', '.join(step['errors'][:6])
                if len(step['errors']) > 6:
                    errors_str += f" (+{len(step['errors'])-6} more)"
                print(f"      ⚠️  Error codes: {errors_str}")
    
    # Error Codes
    error_codes = result.get('error_codes', [])
    if error_codes:
        print(f"\n\n⚠️  ERROR CODES ({len(error_codes)}):")
        print("-" * 40)
        for ec in error_codes[:15]:
            etype = "🚫" if ec['type'] == 'ERROR' else "⚡"
            print(f"   {etype} {ec['code']}: {ec['message'][:60]}")
        if len(error_codes) > 15:
            print(f"   ... and {len(error_codes) - 15} more")
    
    # Sub-procedures
    sub_procs = result.get('sub_procedures', [])
    if sub_procs:
        print(f"\n\n📞 SUB-PROCEDURES CALLED ({len(sub_procs)}):")
        print("-" * 40)
        # Group by function
        for proc in sub_procs[:15]:
            print(f"   • {proc}")
        if len(sub_procs) > 15:
            print(f"   ... and {len(sub_procs) - 15} more")
    
    # Data Used
    data_used = result.get('data_used', [])
    if data_used:
        print(f"\n\n🔗 KEY DATA FIELDS ({len(data_used)}):")
        print("-" * 40)
        # Show in columns
        cols = 4
        for i in range(0, min(20, len(data_used)), cols):
            row = data_used[i:i+cols]
            print(f"   {', '.join(row)}")
        if len(data_used) > 20:
            print(f"   ... and {len(data_used) - 20} more")
    
    # Related procedures
    related = result.get('related_procedures', [])
    if related:
        print(f"\n\n🔗 RELATED PROCEDURES:")
        print(f"   {', '.join(related)}")
    
    # For small procedures without steps, offer to show code
    if not steps and not error_codes:
        print(f"\n\n💡 TIP: Use 'EXPLAIN {result.get('primary_procedure', '')}' to see full code")
    
    print("\n" + "=" * 70)


def interactive_search(searcher: TalSearcherV2):
    """Interactive search REPL"""
    print("\n" + "=" * 60)
    print("TAL Intelligence System v2.0")
    print("=" * 60)
    print("\nCommands:")
    print("  <query>                - Search procedures and symbols")
    print("  ASK <question>         - Explain business logic (or end with ?)")
    print("  TRACE <proc>           - Show call graph")
    print("  USAGE <symbol>         - Find symbol references")
    print("  EXPLAIN <proc>         - Full procedure context")
    print("  STRUCT <name>          - Show struct definition")
    print("  LIST PROC [pattern]    - List procedures")
    print("  LIST DEFINE [pattern]  - List DEFINE macros")
    print("  LIST LITERAL [pattern] - List LITERAL constants")
    print("  LIST RULES [type]      - List business rules")
    print("  HELP                   - Show this help")
    print("  QUIT                   - Exit")
    print("=" * 60)
    
    while True:
        try:
            query = input("\n❓ Search: ").strip()
            
            if not query:
                continue
            
            query_upper = query.upper()
            
            # QUIT
            if query_upper in ('QUIT', 'EXIT', 'Q'):
                print("👋 Goodbye!")
                break
            
            # HELP
            if query_upper == 'HELP':
                print("\nCommands:")
                print("  <query>                - Search procedures and symbols")
                print("  ASK <question>         - Explain business logic (or end with ?)")
                print("  TRACE <proc>           - Show call graph")
                print("  USAGE <symbol>         - Find symbol references")
                print("  EXPLAIN <proc>         - Full procedure context")
                print("  STRUCT <name>          - Show struct definition")
                print("  LIST PROC [pattern]    - List procedures")
                print("  LIST DEFINE [pattern]  - List DEFINE macros")
                print("  LIST LITERAL [pattern] - List LITERAL constants")
                print("  LIST RULES [type]      - List business rules")
                print("\nTip: End any query with ? for business logic explanation")
                continue
            
            # ASK command or question ending with ?
            if query_upper.startswith('ASK ') or query.endswith('?'):
                question = query[4:].strip() if query_upper.startswith('ASK ') else query.rstrip('?').strip()
                # Check for DEBUG flag
                debug_mode = False
                if question.upper().startswith('DEBUG '):
                    debug_mode = True
                    question = question[6:].strip()
                result = searcher.ask_question(question, debug=debug_mode)
                print_ask_result(result)
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
                print_explain_result(result)
                continue
            
            # STRUCT
            if query_upper.startswith('STRUCT '):
                struct_name = query[7:].strip()
                result = searcher.get_struct(struct_name)
                print_struct_result(result)
                continue
            
            # LIST commands
            if query_upper.startswith('LIST '):
                rest = query[5:].strip()
                parts = rest.split(None, 1)
                list_type = parts[0].upper() if parts else ''
                pattern = parts[1] if len(parts) > 1 else None
                
                if list_type in ('PROC', 'PROCS', 'PROCEDURE', 'PROCEDURES'):
                    items = searcher.list_procedures(pattern)
                    print_list_result(items, 'Procedures')
                elif list_type in ('DEFINE', 'DEFINES'):
                    items = searcher.list_defines(pattern)
                    print_list_result(items, 'Defines')
                elif list_type in ('LITERAL', 'LITERALS'):
                    items = searcher.list_literals(pattern)
                    print_list_result(items, 'Literals')
                elif list_type in ('RULE', 'RULES'):
                    items = searcher.list_business_rules(pattern)
                    print_list_result(items, 'Business Rules')
                else:
                    print(f"   Unknown list type: {list_type}")
                    print("   Valid types: PROC, DEFINE, LITERAL, RULES")
                continue
            
            # Default: hybrid search
            print(f"\n🔍 Search: '{query}'")
            print("=" * 60)
            results = searcher.search(query)
            print_search_results(results)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"   ❌ Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Search TAL codebase (v2)")
    parser.add_argument("--db-path", default="./tal_index", help="Path to index")
    args = parser.parse_args()
    
    searcher = TalSearcherV2(db_path=args.db_path)
    try:
        interactive_search(searcher)
    finally:
        searcher.close()


if __name__ == "__main__":
    main()
