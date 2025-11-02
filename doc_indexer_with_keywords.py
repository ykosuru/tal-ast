"""
Wire Processing Business Capability Indexer with Query Expansion
LIGHTWEIGHT VERSION - No spaCy dependency required
FIXED: Now properly extracts domain terms like 'credit', 'party', 'determination'
"""

import os
import json
import math
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Any, Optional, Tuple
import hashlib
import re

# PDF processing
import PyPDF2
import pdfplumber
from PIL import Image

# OCR for images
try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("Warning: pytesseract not available, image OCR disabled")

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


class QueryExpander:
    """
    Query expansion using LLM to improve BM25 search results
    Handles typos, synonyms, and domain-specific expansions
    """
    
    def __init__(
        self,
        taxonomy: BusinessCapabilityTaxonomy,
        expansion_level: str = "medium",
        max_expansions: int = 5
    ):
        """
        Initialize Query Expander
        
        Args:
            taxonomy: Business capability taxonomy
            expansion_level: "basic", "medium", or "advanced"
            max_expansions: Maximum number of query variations to generate
        """
        self.taxonomy = taxonomy
        self.expansion_level = expansion_level
        self.max_expansions = max_expansions
        
        # Build reverse lookup: keyword -> capabilities
        self.keyword_to_capabilities = defaultdict(list)
        for capability, keywords in taxonomy.CAPABILITIES.items():
            for kw in keywords:
                self.keyword_to_capabilities[kw.lower()].append(capability)
    
    def call_llm(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Stub method for LLM calls - to be implemented by user
        
        Args:
            prompt: The prompt to send to LLM
            system_prompt: Optional system prompt for context
            
        Returns:
            LLM response as string
        """
        # TODO: Implement actual LLM call
        print(f"[STUB] call_llm() called with prompt: {prompt[:100]}...")
        return ""
    
    def expand_query(
        self,
        query: str,
        detected_capabilities: Optional[List[Tuple[str, float]]] = None
    ) -> Dict[str, Any]:
        """
        Expand query using multi-level expansion strategy
        """
        result = {
            "original_query": query,
            "expanded_queries": [query],
            "expanded_terms": [],
            "capabilities_context": [],
            "expansion_metadata": {
                "level": self.expansion_level,
                "methods_used": []
            }
        }
        
        # Level 1: Basic expansion (rule-based synonyms)
        if self.expansion_level in ["basic", "medium", "advanced"]:
            basic_expansions = self._basic_expansion(query)
            result["expanded_queries"].extend(basic_expansions)
            result["expansion_metadata"]["methods_used"].append("basic_synonyms")
        
        # Level 2: Domain-specific expansion (using capabilities)
        if self.expansion_level in ["medium", "advanced"]:
            domain_expansions = self._domain_expansion(
                query, 
                detected_capabilities
            )
            result["expanded_queries"].extend(domain_expansions["queries"])
            result["expanded_terms"].extend(domain_expansions["terms"])
            result["capabilities_context"] = domain_expansions["capabilities"]
            result["expansion_metadata"]["methods_used"].append("domain_capabilities")
        
        # Level 3: LLM-based expansion (adaptive)
        if self.expansion_level == "advanced":
            llm_expansions = self._llm_expansion(
                query,
                detected_capabilities,
                result["capabilities_context"]
            )
            result["expanded_queries"].extend(llm_expansions["queries"])
            result["expanded_terms"].extend(llm_expansions["terms"])
            result["expansion_metadata"]["methods_used"].append("llm_expansion")
            result["expansion_metadata"]["llm_response"] = llm_expansions.get("llm_response", "")
        
        # Deduplicate and limit
        result["expanded_queries"] = list(dict.fromkeys(result["expanded_queries"]))[:self.max_expansions]
        result["expanded_terms"] = list(dict.fromkeys(result["expanded_terms"]))
        
        return result
    
    def _basic_expansion(self, query: str) -> List[str]:
        """Level 1: Basic rule-based expansion using synonyms"""
        expansions = []
        query_lower = query.lower()
        
        # Apply synonym expansions
        for term, synonyms in self.taxonomy.SYNONYMS.items():
            if term in query_lower:
                for synonym in synonyms[:2]:
                    expanded = query_lower.replace(term, synonym)
                    if expanded != query_lower:
                        expansions.append(expanded)
        
        # Common typo corrections
        typo_corrections = {
            "ofac": ["ofac screening", "sanctions"],
            "bic": ["swift code", "bank identifier"],
            "iban": ["international bank account"],
            "stp": ["straight through processing"],
            "aml": ["anti money laundering"],
            "pacs": ["iso20022 payment"],
            "fedwire": ["federal reserve wire"],
        }
        
        for typo, corrections in typo_corrections.items():
            if typo in query_lower:
                for correction in corrections[:2]:
                    expansions.append(f"{query_lower} {correction}")
        
        return expansions[:3]
    
    def _domain_expansion(
        self,
        query: str,
        detected_capabilities: Optional[List[Tuple[str, float]]]
    ) -> Dict[str, Any]:
        """Level 2: Domain-specific expansion using business capabilities"""
        expansions = {
            "queries": [],
            "terms": [],
            "capabilities": []
        }
        
        if not detected_capabilities:
            return expansions
        
        # Use top capabilities (score > 0.3)
        relevant_capabilities = [
            cap for cap, score in detected_capabilities 
            if score > 0.3
        ][:3]
        
        expansions["capabilities"] = relevant_capabilities
        
        # For each capability, add related keywords
        for capability in relevant_capabilities:
            related_keywords = self.taxonomy.get_related_capabilities(capability)
            
            # Add top related terms
            for kw in related_keywords[:5]:
                expansions["terms"].append(kw)
                
                # Create combined queries
                combined = f"{query} {kw}"
                expansions["queries"].append(combined)
        
        return expansions
    
    def _llm_expansion(
        self,
        query: str,
        detected_capabilities: Optional[List[Tuple[str, float]]],
        capabilities_context: List[str]
    ) -> Dict[str, Any]:
        """Level 3: LLM-based adaptive expansion"""
        expansions = {
            "queries": [],
            "terms": [],
            "llm_response": ""
        }
        
        # Build context for LLM
        capability_info = ""
        if capabilities_context:
            capability_info = f"\n\nRelevant business capabilities:\n- " + "\n- ".join(capabilities_context)
        
        domain_terms = []
        if detected_capabilities:
            for cap, _ in detected_capabilities[:3]:
                terms = self.taxonomy.get_related_capabilities(cap)
                domain_terms.extend(terms[:5])
        
        domain_context = ""
        if domain_terms:
            domain_context = f"\n\nDomain-specific terms to consider:\n- " + "\n- ".join(set(domain_terms))
        
        system_prompt = """You are an expert in wire processing, payment systems, and financial transaction processing. 
Your task is to expand search queries with relevant synonyms, related terms, and corrections for better document retrieval."""

        user_prompt = f"""Expand this search query for a wire processing document search system:

Original query: "{query}"
{capability_info}
{domain_context}

Provide:
1. 3-5 expanded query variations that include synonyms and related terms
2. 5-10 relevant individual terms that should boost matching documents

Format your response as JSON:
{{
    "expanded_queries": ["variation 1", "variation 2", ...],
    "expanded_terms": ["term1", "term2", ...],
    "reasoning": "brief explanation"
}}"""

        try:
            llm_response = self.call_llm(user_prompt, system_prompt)
            expansions["llm_response"] = llm_response
            
            if llm_response:
                try:
                    parsed = json.loads(llm_response)
                    expansions["queries"] = parsed.get("expanded_queries", [])
                    expansions["expanded_terms"] = parsed.get("expanded_terms", [])
                except json.JSONDecodeError:
                    print("Warning: Could not parse LLM response as JSON")
        
        except Exception as e:
            print(f"Warning: LLM expansion failed: {e}")
        
        return expansions


class ImprovedQueryProcessor:
    """
    Pre-process queries to extract core terms and handle natural language
    Ensures "how do I implement X" returns same results as "X"
    FIXES: Issue where similar queries return different documents
    """
    
    def __init__(self):
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
        
        # Words that suggest action/question but should be removed for search
        self.action_indicators = {
            'how', 'implement', 'create', 'build', 'setup', 'configure',
            'explain', 'describe', 'show', 'tell', 'need', 'want', 'help'
        }
    
    def extract_core_terms(self, query: str) -> Dict[str, Any]:
        """
        Extract core searchable terms from query
        
        Returns:
            {
                'core_terms': ['credit', 'party', 'determination'],
                'original_query': 'how do I implement credit party determination',
                'cleaned_query': 'credit party determination',
                'is_question': True,
                'action_type': 'implement'
            }
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
            
            # Skip stopwords
            if token in self.stopwords:
                continue
            
            core_terms.append(token)
        
        # Reconstruct cleaned query
        cleaned_query = ' '.join(core_terms)
        
        return {
            'core_terms': core_terms,
            'original_query': original,
            'cleaned_query': cleaned_query,
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
    Extract keywords using regex patterns only - NO spaCy required
    FIXED: Now properly extracts domain terms like 'credit', 'party', 'determination'
    """
    
    def __init__(self):
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
        
        # Important domain-specific terms (ALWAYS extract these even if lowercase)
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
            'service', 'api', 'integration', 'interface', 'channel'
        }
        
        # Common stopwords to exclude (EXPANDED to filter generic business terms)
        self.stopwords = {
            # Basic stopwords
            'process', 'system', 'data', 'information',
            'general', 'related', 'based', 'using', 'including', 'provides',
            'allows', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
            'to', 'for', 'of', 'with', 'from', 'by', 'as', 'is', 'was', 'are',
            'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'should', 'could', 'may', 'might', 'must', 'can',
            'this', 'that', 'these', 'those', 'it', 'its', 'they', 'their', 'them',
            
            # Generic business terms that pollute results
            'business', 'approved', 'technical', 'approval', 'requirement',
            'requirements', 'document', 'documentation', 'section', 'page',
            'table', 'figure', 'example', 'note', 'reference', 'description',
            'overview', 'summary', 'details', 'detail', 'status', 'type',
            'item', 'value', 'field', 'name', 'number', 'date', 'time',
            'version', 'change', 'update', 'new', 'old', 'current', 'previous',
            'next', 'first', 'last', 'high', 'low', 'medium', 'level', 'phase',
            'step', 'procedure', 'method', 'function', 'operation', 'activity',
            'action', 'result', 'output', 'input', 'parameter', 'attribute',
            'property', 'component', 'element', 'module', 'application',
            'user', 'source', 'target', 'object', 'entity', 'record', 'entry',
            'list', 'set', 'group', 'class', 'category', 'code', 'identifier',
            'key', 'index', 'flag', 'option', 'setting', 'configuration',
            'setup', 'installation', 'deployment', 'implementation', 'definition',
            'specification', 'standard', 'policy', 'rule', 'guideline', 'principle',
            'practice', 'pattern', 'model', 'template', 'structure', 'schema',
            'design', 'architecture', 'framework', 'platform', 'technology',
            'tool', 'utility', 'helper', 'support', 'control', 'manage'
        }
    
    def extract(self, text: str, max_keywords: int = 20) -> List[Tuple[str, float]]:
        """
        Extract keywords with confidence scores using regex only
        Returns: List of (keyword, confidence) tuples
        """
        if not text or len(text.strip()) < 10:
            return []
        
        keyword_scores = defaultdict(float)
        text_lower = text.lower()
        
        # 1. Domain-specific patterns (highest confidence)
        for pattern, pattern_type in self.patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                keyword_scores[match.lower()] += 3.0
        
        # 2. Important domain terms (NEW: extract these even if lowercase)
        for term in self.important_terms:
            if term in text_lower:
                count = text_lower.count(term)
                # BOOSTED: Important terms get highest priority
                keyword_scores[term] += 3.0 * math.log1p(count)
        
        # 3. Capitalized words (medium-high confidence, but LOWER than important terms)
        capitalized_words = re.findall(r'\b[A-Z][a-z]{2,}\b', text)
        for word in capitalized_words:
            word_lower = word.lower()
            # Only add if NOT stopword and NOT already in important_terms
            if (word_lower not in self.stopwords and 
                word_lower not in self.important_terms):
                keyword_scores[word_lower] += 1.5
        
        # 4. Multi-word capitalized phrases (high confidence)
        capitalized_phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}\b', text)
        for phrase in capitalized_phrases:
            if len(phrase) > 5:
                keyword_scores[phrase.lower()] += 2.0
        
        # 5. Business capability keywords (boost known terms)
        capability_keywords = BusinessCapabilityTaxonomy.get_all_keywords()
        for kw in capability_keywords:
            if kw in text_lower:
                count = text_lower.count(kw)
                keyword_scores[kw] += 2.0 * math.log1p(count)
        
        # 6. Extract individual words from multi-word capability keywords (NEW FIX)
        for kw in capability_keywords:
            words = kw.split()
            if len(words) > 1:  # Multi-word phrase
                for word in words:
                    if (len(word) > 3 and 
                        word not in self.stopwords and 
                        word in text_lower):
                        count = text_lower.count(word)
                        keyword_scores[word] += 1.5 * math.log1p(count)
        
        # 7. Hyphenated technical terms
        hyphenated = re.findall(r'\b[a-z]+-[a-z]+\b', text_lower)
        for term in hyphenated:
            if len(term) > 5:
                keyword_scores[term] += 1.5
        
        # 8. Acronyms (3-5 capital letters)
        acronyms = re.findall(r'\b[A-Z]{3,5}\b', text)
        for acronym in acronyms:
            keyword_scores[acronym.lower()] += 1.8
        
        # Filter and normalize
        filtered = []
        for kw, score in keyword_scores.items():
            # Filter out stopwords and very short terms
            if (len(kw) > 2 and 
                kw not in self.stopwords and
                not all(c.isdigit() for c in kw)):
                filtered.append((kw, score))
        
        # Sort by score and return top N
        filtered.sort(key=lambda x: x[1], reverse=True)
        return filtered[:max_keywords]


class CapabilityMapper:
    """Map documents to business capabilities"""
    
    def __init__(self):
        self.taxonomy = BusinessCapabilityTaxonomy()
        self.capability_keywords = {}
        
        # Build keyword to capability mapping
        for capability, keywords in self.taxonomy.CAPABILITIES.items():
            for kw in keywords:
                if kw not in self.capability_keywords:
                    self.capability_keywords[kw] = []
                self.capability_keywords[kw].append(capability)
    
    def map_to_capabilities(
        self, 
        keywords: List[Tuple[str, float]], 
        text: str
    ) -> List[Tuple[str, float]]:
        """Map extracted keywords to business capabilities"""
        capability_scores = defaultdict(float)
        text_lower = text.lower()
        
        # Score each capability based on keyword matches
        for keyword, kw_score in keywords:
            if keyword in self.capability_keywords:
                for capability in self.capability_keywords[keyword]:
                    capability_scores[capability] += kw_score
            
            expanded = self.taxonomy.expand_with_synonyms(keyword)
            for exp_kw in expanded:
                if exp_kw in self.capability_keywords:
                    for capability in self.capability_keywords[exp_kw]:
                        capability_scores[capability] += kw_score * 0.5
        
        # Direct capability keyword matching in text
        for capability, keywords_list in self.taxonomy.CAPABILITIES.items():
            for kw in keywords_list:
                if kw in text_lower:
                    count = text_lower.count(kw)
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


class WireProcessingIndexer:
    """Fast PDF indexer for wire processing documents - NO SPACY VERSION"""
    
    def __init__(
        self,
        pdf_folder: str,
        index_path: str = "./wire_index",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        enable_ocr: bool = False,
        use_embeddings: bool = False  # Default False for lite version
    ):
        self.pdf_folder = Path(pdf_folder)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.enable_ocr = enable_ocr and OCR_AVAILABLE
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        self.use_embeddings = use_embeddings and EMBEDDINGS_AVAILABLE and FAISS_AVAILABLE
        
        print("Initializing keyword extractor (regex-based, FIXED for domain terms)...")
        self.keyword_extractor = FastKeywordExtractor()
        
        print("Initializing capability mapper...")
        self.capability_mapper = CapabilityMapper()
        
        if self.use_embeddings:
            print(f"Loading embedding model: {embedding_model}")
            self.embedder = SentenceTransformer(embedding_model)
            self.embedding_dim = self.embedder.get_sentence_embedding_dimension()
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        else:
            print("⚡ Embeddings disabled - using pure keyword/BM25 search (faster!)")
            self.embedder = None
            self.embedding_dim = None
            self.faiss_index = None
        
        self.metadata_store = []
        self.document_store = []
        self.keyword_doc_counts = Counter()
        self.capability_doc_counts = Counter()
        self.total_chunks = 0
    
    def extract_text_from_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract text, tables, and images from PDF"""
        content = {
            "text": "",
            "tables": [],
            "images": []
        }
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text() or ""
                    content["text"] += page_text + "\n"
                    
                    tables = page.extract_tables()
                    if tables:
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
        
        return content
    
    def _table_to_text(self, table: List[List]) -> str:
        """Convert table structure to text"""
        if not table:
            return ""
        
        text_rows = []
        for row in table:
            cleaned_row = [str(cell) if cell else "" for cell in row]
            text_rows.append(" | ".join(cleaned_row))
        
        return "\n".join(text_rows)
    
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
        """Main indexing pipeline"""
        print(f"Scanning PDF folder: {self.pdf_folder}")
        pdf_files = list(self.pdf_folder.glob("**/*.pdf"))
        print(f"Found {len(pdf_files)} PDF files")
        
        all_chunks = []
        
        print("\n=== Phase 1: Extracting Content ===")
        for idx, pdf_path in enumerate(pdf_files, 1):
            print(f"Processing [{idx}/{len(pdf_files)}]: {pdf_path.name}")
            
            content = self.extract_text_from_pdf(pdf_path)
            if not content["text"].strip():
                print(f"  ⚠ No text extracted, skipping")
                continue
            
            print(f"  Extracting keywords...")
            keywords = self.keyword_extractor.extract(content["text"])
            print(f"  Found {len(keywords)} keywords")
            if keywords:
                print(f"  Top keywords: {', '.join([kw for kw, _ in keywords[:5]])}")
            
            print(f"  Mapping to capabilities...")
            capabilities = self.capability_mapper.map_to_capabilities(
                keywords, 
                content["text"]
            )
            print(f"  Mapped to {len(capabilities)} capabilities")
            
            if capabilities:
                top_3 = capabilities[:3]
                for cap, score in top_3:
                    print(f"    - {cap}: {score:.2f}")
            
            doc_metadata = {
                "source_file": str(pdf_path.name),
                "source_path": str(pdf_path),
                "file_size": pdf_path.stat().st_size,
                "has_tables": len(content["tables"]) > 0,
                "table_count": len(content["tables"])
            }
            
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
        print(f"\nTotal chunks created: {self.total_chunks}")
        
        print("\n=== Phase 2: Computing TF-IDF Scores ===")
        idf_scores = {}
        for keyword, doc_count in self.keyword_doc_counts.items():
            if doc_count > 0:
                idf = math.log(self.total_chunks / doc_count)
                idf_scores[keyword] = idf
        
        print(f"Computed IDF for {len(idf_scores)} unique keywords")
        
        for chunk in all_chunks:
            keyword_weights = {}
            for kw in chunk["keywords"]:
                if kw in idf_scores:
                    keyword_weights[kw] = idf_scores[kw]
            chunk["keyword_weights"] = keyword_weights
            
            sorted_kw = sorted(
                keyword_weights.items(),
                key=lambda x: x[1],
                reverse=True
            )
            chunk["top_keywords"] = [kw for kw, _ in sorted_kw[:5]]
        
        if self.use_embeddings:
            print("\n=== Phase 3: Generating Embeddings ===")
            for i in range(0, len(all_chunks), batch_size):
                batch = all_chunks[i:i + batch_size]
                texts = [chunk["text"] for chunk in batch]
                embeddings = self.embedder.encode(texts, show_progress_bar=False)
                faiss.normalize_L2(embeddings)
                self.faiss_index.add(embeddings.astype('float32'))
                
                for chunk in batch:
                    top_capabilities = chunk["capabilities"][:3]
                    metadata = {
                        "source_file": chunk["source_file"],
                        "chunk_index": chunk["chunk_index"],
                        "has_tables": chunk["has_tables"],
                        "keywords": chunk["keywords"][:10],
                        "top_keywords": chunk["top_keywords"],
                        "capabilities": top_capabilities,
                        "primary_capability": top_capabilities[0] if top_capabilities else "unknown",
                        "capability_count": len(chunk["capabilities"]),
                        "capability_scores": chunk["capability_scores"]
                    }
                    self.metadata_store.append(metadata)
                    self.document_store.append(chunk["text"])
                
                print(f"  Indexed batch {i//batch_size + 1}/{(len(all_chunks)-1)//batch_size + 1}")
        else:
            print("\n=== Phase 3: Storing Metadata (No Embeddings) ===")
            for chunk in all_chunks:
                top_capabilities = chunk["capabilities"][:3]
                metadata = {
                    "source_file": chunk["source_file"],
                    "chunk_index": chunk["chunk_index"],
                    "has_tables": chunk["has_tables"],
                    "keywords": chunk["keywords"][:10],
                    "top_keywords": chunk["top_keywords"],
                    "capabilities": top_capabilities,
                    "primary_capability": top_capabilities[0] if top_capabilities else "unknown",
                    "capability_count": len(chunk["capabilities"]),
                    "capability_scores": chunk["capability_scores"]
                }
                self.metadata_store.append(metadata)
                self.document_store.append(chunk["text"])
            
            print(f"✓ Stored {len(all_chunks)} chunks (keyword search only)")
        
        print("\n=== Saving Index to Disk ===")
        
        if self.use_embeddings:
            faiss_index_path = self.index_path / "faiss.index"
            faiss.write_index(self.faiss_index, str(faiss_index_path))
            print(f"✓ Saved FAISS index to: {faiss_index_path}")
        
        metadata_path = self.index_path / "metadata.pkl"
        documents_path = self.index_path / "documents.pkl"
        
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata_store, f)
        print(f"✓ Saved metadata to: {metadata_path}")
        
        with open(documents_path, 'wb') as f:
            pickle.dump(self.document_store, f)
        print(f"✓ Saved documents to: {documents_path}")
        
        stats_path = self.index_path / "stats.json"
        stats = {
            "total_chunks": len(all_chunks),
            "total_keywords": len(idf_scores),
            "total_capabilities": len(self.capability_doc_counts),
            "pdf_files": len(pdf_files),
            "idf_scores": idf_scores,
            "capability_distribution": dict(self.capability_doc_counts),
            "embedding_dim": self.embedding_dim,
            "use_embeddings": self.use_embeddings
        }
        
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"✓ Saved statistics to: {stats_path}")
        
        return stats
    
    def get_statistics(self):
        """Display indexing statistics"""
        print("\n=== Indexing Statistics ===")
        print(f"Total chunks: {self.total_chunks}")
        print(f"Unique keywords: {len(self.keyword_doc_counts)}")
        print(f"Unique capabilities: {len(self.capability_doc_counts)}")
        
        print("\n=== Top 10 Capabilities by Document Count ===")
        for cap, count in self.capability_doc_counts.most_common(10):
            print(f"  {cap:40s} - {count:4d} chunks")
        
        print("\n=== Top 10 Keywords by Document Count ===")
        for kw, count in self.keyword_doc_counts.most_common(10):
            idf = math.log(self.total_chunks / count) if count > 0 else 0
            print(f"  {kw:30s} - {count:4d} chunks (IDF: {idf:.3f})")


class WireProcessingSearcher:
    """Fast search using BM25 with enhanced query processing - NO SPACY VERSION"""
    
    def __init__(
        self, 
        index_path: str = "./wire_index",
        enable_query_expansion: bool = True,
        expansion_level: str = "medium",
        enable_query_preprocessing: bool = True
    ):
        self.index_path = Path(index_path)
        self.enable_query_expansion = enable_query_expansion
        self.enable_query_preprocessing = enable_query_preprocessing
        
        stats_path = self.index_path / "stats.json"
        with open(stats_path, 'r') as f:
            self.stats = json.load(f)
        
        self.use_embeddings = self.stats.get('use_embeddings', False)
        
        metadata_path = self.index_path / "metadata.pkl"
        documents_path = self.index_path / "documents.pkl"
        
        with open(metadata_path, 'rb') as f:
            self.metadata_store = pickle.load(f)
        
        with open(documents_path, 'rb') as f:
            self.document_store = pickle.load(f)
        
        if self.use_embeddings and EMBEDDINGS_AVAILABLE and FAISS_AVAILABLE:
            faiss_index_path = self.index_path / "faiss.index"
            if faiss_index_path.exists():
                self.faiss_index = faiss.read_index(str(faiss_index_path))
                self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
                print("✓ Loaded FAISS index for semantic search")
            else:
                self.use_embeddings = False
                self.faiss_index = None
                self.embedder = None
        else:
            print("⚡ Using keyword/BM25 search (no embeddings)")
            self.faiss_index = None
            self.embedder = None
        
        tokenized_docs = [doc.lower().split() for doc in self.document_store]
        self.bm25 = BM25Okapi(tokenized_docs)
        
        # Enhanced query processing
        self.query_processor = ImprovedQueryProcessor()
        self.keyword_extractor = FastKeywordExtractor()
        self.capability_mapper = CapabilityMapper()
        
        if self.enable_query_expansion:
            self.query_expander = QueryExpander(
                taxonomy=BusinessCapabilityTaxonomy,
                expansion_level=expansion_level,
                max_expansions=5
            )
            print(f"✓ Query expansion enabled (level: {expansion_level})")
        else:
            self.query_expander = None
        
        if self.enable_query_preprocessing:
            print(f"✓ Enhanced query preprocessing enabled (removes stopwords)")
    
    def search(
        self,
        query: str,
        top_k: int = 20,
        capability_filter: Optional[List[str]] = None,
        min_capability_score: float = 0.3,
        use_query_expansion: Optional[bool] = None,
        verbose: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Enhanced search with query preprocessing and expansion
        Handles natural language queries consistently
        """
        if verbose:
            print(f"\n{'='*80}")
            print(f"Query: {query}")
            print(f"{'='*80}")
        
        # Step 1: Preprocess query (remove stopwords, extract core terms)
        original_query = query
        search_query = query
        core_terms = []
        phrases = []
        
        if self.enable_query_preprocessing:
            query_analysis = self.query_processor.extract_core_terms(query)
            search_query = query_analysis['cleaned_query']
            core_terms = query_analysis['core_terms']
            
            if verbose:
                print(f"\nQuery Preprocessing:")
                print(f"  Original: {query}")
                print(f"  Cleaned: {search_query}")
                print(f"  Core terms: {core_terms}")
                print(f"  Stopwords removed: {query_analysis['stopwords_removed']}")
            
            # Extract phrases
            phrases = self.query_processor.extract_phrases(query)
            if phrases and verbose:
                print(f"  Phrases: {phrases}")
        
        # Step 2: Extract keywords and capabilities
        query_keywords = self.keyword_extractor.extract(search_query)
        query_capabilities = self.capability_mapper.map_to_capabilities(
            query_keywords,
            search_query
        )
        
        if verbose:
            print(f"\nKeywords: {[kw for kw, _ in query_keywords[:5]]}")
            if query_capabilities:
                print(f"Capabilities: {[cap for cap, _ in query_capabilities[:3]]}")
        
        # Step 3: Query expansion
        expanded_info = None
        if use_query_expansion is None:
            use_query_expansion = self.enable_query_expansion
        
        if use_query_expansion and self.query_expander:
            if verbose:
                print(f"\nQuery Expansion:")
            
            expanded_info = self.query_expander.expand_query(
                search_query,  # Use cleaned query
                query_capabilities
            )
            
            if verbose:
                print(f"  Expanded queries: {expanded_info['expanded_queries'][:3]}")
                if expanded_info['expanded_terms']:
                    print(f"  Expanded terms: {expanded_info['expanded_terms'][:5]}")
        
        # Step 4: Auto-select capabilities
        if not capability_filter and query_capabilities:
            capability_filter = [cap for cap, score in query_capabilities 
                               if score >= min_capability_score][:3]
            if verbose and capability_filter:
                print(f"  Auto-selected capabilities: {capability_filter}")
        
        # Step 5: Execute search
        results = self._enhanced_bm25_search(
            search_query=search_query,
            core_terms=core_terms,
            phrases=phrases,
            query_capabilities=query_capabilities,
            capability_filter=capability_filter,
            expanded_info=expanded_info,
            top_k=top_k
        )
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"Found {len(results)} results")
            print(f"{'='*80}\n")
        
        return results
    
    def _enhanced_bm25_search(
        self,
        search_query: str,
        core_terms: List[str],
        phrases: List[str],
        query_capabilities: List[Tuple[str, float]],
        capability_filter: Optional[List[str]],
        expanded_info: Optional[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Enhanced BM25 search with better scoring"""
        
        # Collect all query variants
        query_variants = [search_query]
        
        if expanded_info:
            query_variants.extend(expanded_info.get('expanded_queries', []))
            if expanded_info.get('expanded_terms'):
                for term in expanded_info['expanded_terms'][:5]:
                    query_variants.append(term)
        
        # Add core terms as individual queries
        for term in core_terms:
            if len(term) > 3:  # Only substantial terms
                query_variants.append(term)
        
        # Aggregate scores across all variants
        aggregated_scores = np.zeros(len(self.document_store))
        
        for variant_idx, variant in enumerate(query_variants):
            query_tokens = variant.lower().split()
            variant_scores = self.bm25.get_scores(query_tokens)
            
            # Smart weighting:
            # - Cleaned query: 1.0 (highest)
            # - Core terms: 0.8
            # - Expanded queries: 0.5
            # - Expanded terms: 0.3
            if variant_idx == 0:
                weight = 1.0  # Cleaned query
            elif variant in core_terms:
                weight = 0.8  # Core terms
            elif expanded_info and variant in expanded_info.get('expanded_queries', []):
                weight = 0.5  # Expanded queries
            else:
                weight = 0.3  # Expanded terms
            
            aggregated_scores += variant_scores * weight
        
        # Add phrase matching bonus
        if phrases:
            phrase_bonuses = self._compute_phrase_bonuses(phrases)
            aggregated_scores += phrase_bonuses * 0.3
        
        # Get top candidates
        search_k = min(1000, top_k * 20) if capability_filter else top_k
        top_indices = np.argsort(aggregated_scores)[::-1][:search_k]
        
        # Format results
        formatted_results = []
        for idx in top_indices:
            if aggregated_scores[idx] < 0.01:  # Skip very low scores
                continue
            
            metadata = self.metadata_store[idx]
            
            # Filter by capability
            if capability_filter:
                if metadata['primary_capability'] not in capability_filter:
                    if not any(cap in capability_filter for cap in metadata['capabilities']):
                        continue
            
            # Compute capability overlap
            if query_capabilities:
                query_caps = set([cap for cap, _ in query_capabilities[:5]])
                doc_caps = set(metadata['capabilities'])
                overlap = len(query_caps & doc_caps)
                capability_overlap = overlap / len(query_caps) if query_caps else 0
            else:
                capability_overlap = 1.0
            
            # Normalize score
            max_score = aggregated_scores.max() if aggregated_scores.max() > 0 else 1.0
            normalized_score = aggregated_scores[idx] / max_score
            
            # Check for exact phrase matches
            doc_text_lower = self.document_store[idx].lower()
            phrase_match_bonus = 0.0
            for phrase in phrases:
                if phrase.lower() in doc_text_lower:
                    phrase_match_bonus += 0.1
            
            formatted_results.append({
                "text": self.document_store[idx],
                "source_file": metadata['source_file'],
                "chunk_index": metadata['chunk_index'],
                "bm25_score": aggregated_scores[idx],
                "normalized_score": normalized_score,
                "capabilities": metadata['capabilities'],
                "keywords": metadata['keywords'],
                "capability_overlap": capability_overlap,
                "primary_capability": metadata['primary_capability'],
                "phrase_match_bonus": phrase_match_bonus
            })
        
        # Re-rank by combined score
        for result in formatted_results:
            result['combined_score'] = (
                result['normalized_score'] * 0.5 +
                result['capability_overlap'] * 0.3 +
                result['phrase_match_bonus'] * 0.2
            )
        
        formatted_results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return formatted_results[:top_k]
    
    def _compute_phrase_bonuses(self, phrases: List[str]) -> np.ndarray:
        """Compute bonus scores for documents containing exact phrases"""
        bonuses = np.zeros(len(self.document_store))
        
        for phrase in phrases:
            phrase_lower = phrase.lower()
            for idx, doc in enumerate(self.document_store):
                if phrase_lower in doc.lower():
                    count = doc.lower().count(phrase_lower)
                    bonuses[idx] += np.log1p(count)
        
        return bonuses
    
    def search_by_capability(
        self,
        capability: str,
        top_k: int = 20
    ) -> List[Dict[str, Any]]:
        """Search documents by specific capability"""
        formatted = []
        
        for idx, metadata in enumerate(self.metadata_store):
            if metadata['primary_capability'] == capability or capability in metadata['capabilities']:
                formatted.append({
                    "text": self.document_store[idx],
                    "source_file": metadata['source_file'],
                    "chunk_index": metadata['chunk_index'],
                    "capabilities": metadata['capabilities'],
                    "keywords": metadata['keywords'],
                    "primary_capability": metadata['primary_capability']
                })
                
                if len(formatted) >= top_k:
                    break
        
        return formatted


def main():
    """Example usage - NO SPACY VERSION - FIXED KEYWORD EXTRACTION"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Wire Processing Indexer - Lightweight (No spaCy required) - FIXED"
    )
    parser.add_argument("--pdf-folder", required=True, help="Path to PDF folder")
    parser.add_argument("--index-path", default="./wire_index", help="Index storage path")
    parser.add_argument("--action", choices=["index", "search", "stats"], default="index")
    parser.add_argument("--query", help="Search query")
    parser.add_argument("--capability", help="Filter by capability")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results")
    parser.add_argument("--disable-expansion", action="store_true",
                       help="Disable query expansion")
    parser.add_argument("--expansion-level", default="medium",
                       choices=["basic", "medium", "advanced"],
                       help="Query expansion level")
    parser.add_argument("--verbose", action="store_true",
                       help="Show detailed search process")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Wire Processing Indexer - LITE VERSION (No spaCy) - FIXED")
    print("=" * 70)
    print()
    
    if args.action == "index":
        indexer = WireProcessingIndexer(
            pdf_folder=args.pdf_folder,
            index_path=args.index_path,
            use_embeddings=False  # Lite version
        )
        
        stats = indexer.index_pdfs()
        indexer.get_statistics()
        
    elif args.action == "search":
        if not args.query:
            print("Error: --query required for search")
            return
        
        searcher = WireProcessingSearcher(
            index_path=args.index_path,
            enable_query_expansion=not args.disable_expansion,
            expansion_level=args.expansion_level
        )
        
        if args.capability:
            results = searcher.search_by_capability(args.capability, args.top_k)
        else:
            results = searcher.search(
                args.query, 
                top_k=args.top_k,
                verbose=args.verbose
            )
        
        print(f"\n{'='*80}")
        print(f"Found {len(results)} results")
        print(f"{'='*80}\n")
        
        for i, result in enumerate(results, 1):
            print(f"[{i}] Score: {result.get('combined_score', 0):.3f}")
            print(f"Source: {result['source_file']}")
            print(f"Primary Capability: {result.get('primary_capability', 'N/A')}")
            print(f"All Capabilities: {', '.join(result['capabilities'][:3])}")
            print(f"Keywords: {', '.join(result['keywords'][:5])}")
            print(f"Text: {result['text'][:200]}...")
            print("-" * 80)
    
    elif args.action == "stats":
        searcher = WireProcessingSearcher(index_path=args.index_path)
        print("\n=== Capability Distribution ===")
        cap_dist = searcher.stats.get('capability_distribution', {})
        sorted_caps = sorted(cap_dist.items(), key=lambda x: x[1], reverse=True)
        for cap, count in sorted_caps:
            print(f"  {cap:40s} - {count:4d} chunks")


if __name__ == "__main__":
    main()

print("\n" + "=" * 70)
print("✓ Wire Processing Indexer - ENHANCED LITE VERSION - FIXED")
print("  • NO spaCy required (regex-based keyword extraction)")
print("  • FIXED: Now extracts 'credit', 'party', 'determination' etc.")
print("  • Enhanced query preprocessing (removes stopwords)")
print("  • Consistent results for similar queries")
print("  • Query expansion enabled (basic/medium/advanced)")
print("=" * 70)
