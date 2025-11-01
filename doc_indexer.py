"""
Wire Processing Business Capability Indexer with Query Expansion
Fast, non-LLM approach with optional LLM-based query expansion for improved BM25 search
Processes PDFs (text, tables, images) and maps to business capabilities
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

# NLP
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

# Vector embeddings
import numpy as np
from sentence_transformers import SentenceTransformer

# Vector database (FAISS)
import faiss
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
            
        Example implementation:
            def call_llm(self, prompt, system_prompt=None):
                # Using OpenAI
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": system_prompt or "You are a helpful assistant"},
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.choices[0].message.content
                
                # Or using Anthropic Claude
                response = anthropic.Anthropic().messages.create(
                    model="claude-3-sonnet-20240229",
                    system=system_prompt or "",
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
        """
        # TODO: Implement actual LLM call
        # For now, return empty string to maintain compatibility
        print(f"[STUB] call_llm() called with prompt: {prompt[:100]}...")
        return ""
    
    def expand_query(
        self,
        query: str,
        detected_capabilities: Optional[List[Tuple[str, float]]] = None
    ) -> Dict[str, Any]:
        """
        Expand query using multi-level expansion strategy
        
        Args:
            query: Original search query
            detected_capabilities: List of (capability, score) tuples from query analysis
            
        Returns:
            Dictionary with:
                - original_query: Original query
                - expanded_queries: List of expanded query variations
                - expanded_terms: List of individual expanded terms
                - capabilities_context: Relevant capabilities identified
                - expansion_metadata: Details about expansion process
        """
        result = {
            "original_query": query,
            "expanded_queries": [query],  # Always include original
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
        """
        Level 1: Basic rule-based expansion using synonyms
        Handles common typos and variations
        """
        expansions = []
        query_lower = query.lower()
        
        # Apply synonym expansions
        for term, synonyms in self.taxonomy.SYNONYMS.items():
            if term in query_lower:
                for synonym in synonyms[:2]:  # Limit to 2 synonyms per term
                    expanded = query_lower.replace(term, synonym)
                    if expanded != query_lower:
                        expansions.append(expanded)
        
        # Common typo corrections (wire processing specific)
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
        
        return expansions[:3]  # Limit basic expansions
    
    def _domain_expansion(
        self,
        query: str,
        detected_capabilities: Optional[List[Tuple[str, float]]]
    ) -> Dict[str, Any]:
        """
        Level 2: Domain-specific expansion using business capabilities
        Leverages the capabilities[] array to add related terms
        """
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
        """
        Level 3: LLM-based adaptive expansion
        Uses LLM to generate contextually relevant expansions
        """
        expansions = {
            "queries": [],
            "terms": [],
            "llm_response": ""
        }
        
        # Build context for LLM
        capability_info = ""
        if capabilities_context:
            capability_info = f"\n\nRelevant business capabilities:\n- " + "\n- ".join(capabilities_context)
        
        # Build domain vocabulary context
        domain_terms = []
        if detected_capabilities:
            for cap, _ in detected_capabilities[:3]:
                terms = self.taxonomy.get_related_capabilities(cap)
                domain_terms.extend(terms[:5])
        
        domain_context = ""
        if domain_terms:
            domain_context = f"\n\nDomain-specific terms to consider:\n- " + "\n- ".join(set(domain_terms))
        
        # Construct LLM prompt
        system_prompt = """You are an expert in wire processing, payment systems, and financial transaction processing. 
Your task is to expand search queries with relevant synonyms, related terms, and corrections for better document retrieval.

Focus on:
- Wire transfer and payment processing terminology
- ISO 20022 standards (pacs, pain, camt messages)
- SWIFT, FedWire, CHIPS networks
- Sanctions screening (OFAC)
- Payment validation and repair
- Business capabilities in payment processing"""

        user_prompt = f"""Expand this search query for a wire processing document search system:

Original query: "{query}"
{capability_info}
{domain_context}

Provide:
1. 3-5 expanded query variations that include synonyms and related terms
2. 5-10 relevant individual terms that should boost matching documents
3. Correct any potential typos or abbreviations

Format your response as JSON:
{{
    "expanded_queries": ["variation 1", "variation 2", ...],
    "expanded_terms": ["term1", "term2", ...],
    "reasoning": "brief explanation"
}}"""

        try:
            # Call LLM (stub for now)
            llm_response = self.call_llm(user_prompt, system_prompt)
            expansions["llm_response"] = llm_response
            
            # Parse JSON response
            if llm_response:
                try:
                    parsed = json.loads(llm_response)
                    expansions["queries"] = parsed.get("expanded_queries", [])
                    expansions["terms"] = parsed.get("expanded_terms", [])
                except json.JSONDecodeError:
                    print("Warning: Could not parse LLM response as JSON")
        
        except Exception as e:
            print(f"Warning: LLM expansion failed: {e}")
        
        return expansions
    
    def detect_query_complexity(self, query: str) -> str:
        """
        Detect query complexity to determine expansion strategy
        
        Returns: "simple", "medium", or "complex"
        """
        words = query.split()
        
        # Simple: 1-3 words, common terms
        if len(words) <= 3:
            return "simple"
        
        # Complex: technical terms, specific codes, multi-part
        technical_patterns = [
            r'ISO[\s-]?\d+',
            r'pacs\.\d+',
            r'[A-Z]{6}[A-Z0-9]{2}',
            r'ACE[\s-]?\d+',
        ]
        
        for pattern in technical_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return "complex"
        
        # Medium: everything else
        return "medium"


class FastKeywordExtractor:
    """Extract keywords without LLM using NER, patterns, and TF-IDF"""
    
    def __init__(self):
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            print("Downloading spaCy model...")
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        
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
        
        # Common stopwords to exclude
        self.stopwords = {
            'process', 'system', 'data', 'information', 'service',
            'management', 'processing', 'services', 'general', 'related',
            'based', 'using', 'including', 'provides', 'allows'
        }
    
    def extract(self, text: str, max_keywords: int = 20) -> List[Tuple[str, float]]:
        """
        Extract keywords with confidence scores
        Returns: List of (keyword, confidence) tuples
        """
        if not text or len(text.strip()) < 10:
            return []
        
        doc = self.nlp(text[:5000])  # Limit for speed
        keyword_scores = defaultdict(float)
        
        # 1. Domain-specific patterns (highest confidence)
        for pattern, pattern_type in self.patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                keyword_scores[match.lower()] += 3.0
        
        # 2. Named entities (high confidence)
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PRODUCT', 'GPE', 'LAW', 'MONEY']:
                keyword_scores[ent.text.lower()] += 2.5
        
        # 3. Noun phrases (medium confidence)
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) <= 3 and chunk.root.pos_ in ['NOUN', 'PROPN']:
                keyword_scores[chunk.text.lower()] += 1.5
        
        # 4. Capitalized terms (medium confidence)
        for token in doc:
            if (token.is_alpha and token.text[0].isupper() and 
                len(token.text) > 2 and not token.is_stop):
                keyword_scores[token.text.lower()] += 1.0
        
        # 5. Business capability keywords (boost known terms)
        capability_keywords = BusinessCapabilityTaxonomy.get_all_keywords()
        text_lower = text.lower()
        for kw in capability_keywords:
            if kw in text_lower:
                # Count occurrences
                count = text_lower.count(kw)
                keyword_scores[kw] += 2.0 * math.log1p(count)
        
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
        """
        Map extracted keywords to business capabilities
        Returns: List of (capability_name, confidence_score) tuples
        """
        capability_scores = defaultdict(float)
        text_lower = text.lower()
        
        # Score each capability based on keyword matches
        for keyword, kw_score in keywords:
            # Direct matches
            if keyword in self.capability_keywords:
                for capability in self.capability_keywords[keyword]:
                    capability_scores[capability] += kw_score
            
            # Expanded matches (synonyms)
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
    """Fast PDF indexer for wire processing documents"""
    
    def __init__(
        self,
        pdf_folder: str,
        index_path: str = "./wire_index",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        enable_ocr: bool = False,
        use_embeddings: bool = True
    ):
        """
        Initialize Wire Processing Indexer
        
        Args:
            pdf_folder: Path to folder containing PDFs
            index_path: Path to store index
            embedding_model: Sentence transformer model name (only used if use_embeddings=True)
            chunk_size: Words per chunk
            chunk_overlap: Overlap between chunks
            enable_ocr: Enable OCR for images
            use_embeddings: If False, use pure keyword/BM25 search (faster, no model download)
        """
        self.pdf_folder = Path(pdf_folder)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.enable_ocr = enable_ocr and OCR_AVAILABLE
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        self.use_embeddings = use_embeddings
        
        # Initialize keyword extractor
        print("Initializing keyword extractor...")
        self.keyword_extractor = FastKeywordExtractor()
        
        print("Initializing capability mapper...")
        self.capability_mapper = CapabilityMapper()
        
        # Initialize embeddings only if needed
        if self.use_embeddings:
            print(f"Loading embedding model: {embedding_model}")
            self.embedder = SentenceTransformer(embedding_model)
            self.embedding_dim = self.embedder.get_sentence_embedding_dimension()
            
            # Initialize FAISS index
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        else:
            print("⚡ Embeddings disabled - using pure keyword/BM25 search (faster!)")
            self.embedder = None
            self.embedding_dim = None
            self.faiss_index = None
        
        # Metadata storage (works for both modes)
        self.metadata_store = []
        self.document_store = []
        
        # Statistics
        self.keyword_doc_counts = Counter()
        self.capability_doc_counts = Counter()
        self.total_chunks = 0
    
    def extract_text_from_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Extract text, tables, and images from PDF
        Returns structured content
        """
        content = {
            "text": "",
            "tables": [],
            "images": []
        }
        
        try:
            # Use pdfplumber for better table extraction
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    # Extract text
                    page_text = page.extract_text() or ""
                    content["text"] += page_text + "\n"
                    
                    # Extract tables
                    tables = page.extract_tables()
                    if tables:
                        for table in tables:
                            # Convert table to text representation
                            table_text = self._table_to_text(table)
                            content["tables"].append({
                                "page": page_num + 1,
                                "text": table_text
                            })
                            content["text"] += f"\n[TABLE]\n{table_text}\n[/TABLE]\n"
            
            # Extract images if OCR enabled
            if self.enable_ocr:
                content["images"] = self._extract_images_with_ocr(pdf_path)
                for img in content["images"]:
                    content["text"] += f"\n[IMAGE]\n{img['text']}\n[/IMAGE]\n"
        
        except Exception as e:
            print(f"Error extracting from {pdf_path}: {e}")
            # Fallback to PyPDF2
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
            # Filter out None values
            cleaned_row = [str(cell) if cell else "" for cell in row]
            text_rows.append(" | ".join(cleaned_row))
        
        return "\n".join(text_rows)
    
    def _extract_images_with_ocr(self, pdf_path: Path) -> List[Dict]:
        """Extract images and perform OCR"""
        images = []
        # Implementation would use pdf2image + pytesseract
        # Skipping for brevity - can add if needed
        return images
    
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
        """Main indexing pipeline - no LLM needed"""
        print(f"Scanning PDF folder: {self.pdf_folder}")
        pdf_files = list(self.pdf_folder.glob("**/*.pdf"))
        print(f"Found {len(pdf_files)} PDF files")
        
        all_chunks = []
        
        # Phase 1: Extract and analyze documents
        print("\n=== Phase 1: Extracting Content ===")
        for idx, pdf_path in enumerate(pdf_files, 1):
            print(f"Processing [{idx}/{len(pdf_files)}]: {pdf_path.name}")
            
            # Extract content (text, tables, images)
            content = self.extract_text_from_pdf(pdf_path)
            if not content["text"].strip():
                print(f"  ⚠ No text extracted, skipping")
                continue
            
            # Extract keywords (no LLM)
            print(f"  Extracting keywords...")
            keywords = self.keyword_extractor.extract(content["text"])
            print(f"  Found {len(keywords)} keywords")
            
            # Map to business capabilities
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
            
            # Create chunks
            doc_metadata = {
                "source_file": str(pdf_path.name),
                "source_path": str(pdf_path),
                "file_size": pdf_path.stat().st_size,
                "has_tables": len(content["tables"]) > 0,
                "table_count": len(content["tables"])
            }
            
            chunks = self.chunk_text(content["text"], doc_metadata)
            
            # Add keywords and capabilities to each chunk
            for chunk in chunks:
                chunk["keywords"] = [kw for kw, score in keywords]
                chunk["keyword_scores"] = dict(keywords)
                chunk["capabilities"] = [cap for cap, score in capabilities]
                chunk["capability_scores"] = dict(capabilities)
                
                # Update statistics
                self.keyword_doc_counts.update(set(chunk["keywords"]))
                self.capability_doc_counts.update(set(chunk["capabilities"]))
            
            all_chunks.extend(chunks)
            print(f"  Created {len(chunks)} chunks")
        
        self.total_chunks = len(all_chunks)
        print(f"\nTotal chunks created: {self.total_chunks}")
        
        # Phase 2: Compute IDF scores
        print("\n=== Phase 2: Computing TF-IDF Scores ===")
        idf_scores = {}
        for keyword, doc_count in self.keyword_doc_counts.items():
            if doc_count > 0:
                idf = math.log(self.total_chunks / doc_count)
                idf_scores[keyword] = idf
        
        print(f"Computed IDF for {len(idf_scores)} unique keywords")
        
        # Add IDF weights to chunks
        for chunk in all_chunks:
            keyword_weights = {}
            for kw in chunk["keywords"]:
                if kw in idf_scores:
                    keyword_weights[kw] = idf_scores[kw]
            chunk["keyword_weights"] = keyword_weights
            
            # Top weighted keywords
            sorted_kw = sorted(
                keyword_weights.items(),
                key=lambda x: x[1],
                reverse=True
            )
            chunk["top_keywords"] = [kw for kw, _ in sorted_kw[:5]]
        
        # Phase 3: Generate embeddings and store (optional)
        if self.use_embeddings:
            print("\n=== Phase 3: Generating Embeddings and Indexing ===")
            for i in range(0, len(all_chunks), batch_size):
                batch = all_chunks[i:i + batch_size]
                
                # Generate embeddings
                texts = [chunk["text"] for chunk in batch]
                embeddings = self.embedder.encode(texts, show_progress_bar=False)
                
                # Normalize embeddings for cosine similarity with inner product
                faiss.normalize_L2(embeddings)
                
                # Add to FAISS index
                self.faiss_index.add(embeddings.astype('float32'))
                
                # Store metadata and documents
                for chunk in batch:
                    # Top 3 capabilities only (for faster filtering)
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
            
            print(f"\n✓ Indexing complete!")
            
            # Save FAISS index
            print("\n=== Saving Index to Disk ===")
            faiss_index_path = self.index_path / "faiss.index"
            faiss.write_index(self.faiss_index, str(faiss_index_path))
            print(f"✓ Saved FAISS index to: {faiss_index_path}")
        else:
            print("\n=== Phase 3: Storing Metadata (No Embeddings) ===")
            # Store metadata and documents without embeddings
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
        
        # Save metadata and documents (both modes)
        metadata_path = self.index_path / "metadata.pkl"
        documents_path = self.index_path / "documents.pkl"
        
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata_store, f)
        print(f"✓ Saved metadata to: {metadata_path}")
        
        with open(documents_path, 'wb') as f:
            pickle.dump(self.document_store, f)
        print(f"✓ Saved documents to: {documents_path}")
        
        # Save statistics
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
    """Fast search using capability filtering + FAISS/BM25 search with query expansion"""
    
    def __init__(
        self, 
        index_path: str = "./wire_index",
        enable_query_expansion: bool = True,
        expansion_level: str = "medium"
    ):
        """
        Initialize searcher with optional query expansion
        
        Args:
            index_path: Path to index files
            enable_query_expansion: Enable LLM-based query expansion
            expansion_level: "basic", "medium", or "advanced"
        """
        self.index_path = Path(index_path)
        self.enable_query_expansion = enable_query_expansion
        
        # Load statistics first to check mode
        stats_path = self.index_path / "stats.json"
        with open(stats_path, 'r') as f:
            self.stats = json.load(f)
        
        self.use_embeddings = self.stats.get('use_embeddings', True)
        
        # Load metadata and documents
        metadata_path = self.index_path / "metadata.pkl"
        documents_path = self.index_path / "documents.pkl"
        
        with open(metadata_path, 'rb') as f:
            self.metadata_store = pickle.load(f)
        
        with open(documents_path, 'rb') as f:
            self.document_store = pickle.load(f)
        
        # Load FAISS index only if using embeddings
        if self.use_embeddings:
            faiss_index_path = self.index_path / "faiss.index"
            if faiss_index_path.exists():
                self.faiss_index = faiss.read_index(str(faiss_index_path))
                self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
                print("✓ Loaded FAISS index for semantic search")
            else:
                print("⚠ FAISS index not found, falling back to keyword search")
                self.use_embeddings = False
                self.faiss_index = None
                self.embedder = None
        else:
            print("⚡ Using keyword/BM25 search (no embeddings)")
            self.faiss_index = None
            self.embedder = None
        
        # Initialize BM25 for keyword search (always available)
        tokenized_docs = [doc.lower().split() for doc in self.document_store]
        self.bm25 = BM25Okapi(tokenized_docs)
        
        self.keyword_extractor = FastKeywordExtractor()
        self.capability_mapper = CapabilityMapper()
        
        # Initialize query expander if enabled
        if self.enable_query_expansion:
            self.query_expander = QueryExpander(
                taxonomy=BusinessCapabilityTaxonomy,
                expansion_level=expansion_level,
                max_expansions=5
            )
            print(f"✓ Query expansion enabled (level: {expansion_level})")
        else:
            self.query_expander = None
            print("⚠ Query expansion disabled")
    
    def search(
        self,
        query: str,
        top_k: int = 20,
        capability_filter: Optional[List[str]] = None,
        min_capability_score: float = 0.3,
        use_query_expansion: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """
        Fast search using capability-based filtering with optional query expansion
        
        Args:
            query: Search query
            top_k: Number of results
            capability_filter: Filter by specific capabilities
            min_capability_score: Minimum capability match score (0-1)
            use_query_expansion: Override instance setting for this query
        """
        # Extract keywords and map to capabilities from query
        query_keywords = self.keyword_extractor.extract(query)
        query_capabilities = self.capability_mapper.map_to_capabilities(
            query_keywords,
            query
        )
        
        print(f"\nQuery Analysis:")
        print(f"  Original query: {query}")
        print(f"  Keywords: {[kw for kw, _ in query_keywords[:5]]}")
        print(f"  Capabilities: {[cap for cap, _ in query_capabilities[:3]]}")
        
        # Query expansion (if enabled)
        expanded_info = None
        if use_query_expansion is None:
            use_query_expansion = self.enable_query_expansion
        
        if use_query_expansion and self.query_expander:
            print(f"  Expanding query...")
            expanded_info = self.query_expander.expand_query(query, query_capabilities)
            
            print(f"  Expanded queries ({len(expanded_info['expanded_queries'])}):")
            for eq in expanded_info['expanded_queries'][:3]:
                print(f"    - {eq}")
            
            if expanded_info['expanded_terms']:
                print(f"  Expanded terms: {expanded_info['expanded_terms'][:5]}")
            
            if expanded_info['capabilities_context']:
                print(f"  Capability context: {expanded_info['capabilities_context'][:3]}")
        
        # Use top capabilities for filtering if not specified
        if not capability_filter and query_capabilities:
            capability_filter = [cap for cap, score in query_capabilities 
                               if score >= min_capability_score][:3]
            print(f"  Auto-selected capabilities: {capability_filter}")
        
        if self.use_embeddings and self.faiss_index is not None:
            # Semantic search with FAISS
            return self._search_with_faiss(
                query, query_capabilities, capability_filter, top_k, expanded_info
            )
        else:
            # Keyword search with BM25 (with expansion)
            return self._search_with_bm25(
                query, query_capabilities, capability_filter, top_k, expanded_info
            )
    
    def _search_with_bm25(
        self,
        query: str,
        query_capabilities: List[Tuple[str, float]],
        capability_filter: Optional[List[str]],
        top_k: int,
        expanded_info: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search using BM25 keyword search with query expansion
        This is the key improvement - multiple query variations
        """
        print("  Using keyword search (BM25) with query expansion")
        
        # Collect all query variations
        query_variants = [query]
        if expanded_info:
            query_variants.extend(expanded_info.get('expanded_queries', []))
        
        # Also add expanded terms as individual queries
        if expanded_info and expanded_info.get('expanded_terms'):
            for term in expanded_info['expanded_terms'][:5]:
                query_variants.append(term)
        
        print(f"  Searching with {len(query_variants)} query variations")
        
        # Aggregate scores across all query variants
        aggregated_scores = np.zeros(len(self.document_store))
        
        for variant_idx, variant in enumerate(query_variants):
            # Tokenize query variant
            query_tokens = variant.lower().split()
            
            # Get BM25 scores for this variant
            variant_scores = self.bm25.get_scores(query_tokens)
            
            # Weight: Original query gets full weight, expansions get reduced weight
            weight = 1.0 if variant_idx == 0 else 0.5
            aggregated_scores += variant_scores * weight
        
        # Get top candidates
        search_k = min(1000, top_k * 20) if capability_filter else top_k
        top_indices = np.argsort(aggregated_scores)[::-1][:search_k]
        
        # Format results
        formatted_results = []
        for idx in top_indices:
            metadata = self.metadata_store[idx]
            
            # Filter by capability if specified
            if capability_filter:
                if metadata['primary_capability'] not in capability_filter:
                    if not any(cap in capability_filter for cap in metadata['capabilities']):
                        continue
            
            # Compute capability overlap score
            if query_capabilities:
                query_caps = set([cap for cap, _ in query_capabilities[:5]])
                doc_caps = set(metadata['capabilities'])
                overlap = len(query_caps & doc_caps)
                capability_overlap = overlap / len(query_caps) if query_caps else 0
            else:
                capability_overlap = 1.0
            
            # Normalize BM25 score to 0-1 range
            max_score = aggregated_scores.max() if aggregated_scores.max() > 0 else 1.0
            normalized_score = aggregated_scores[idx] / max_score
            
            formatted_results.append({
                "text": self.document_store[idx],
                "source_file": metadata['source_file'],
                "chunk_index": metadata['chunk_index'],
                "bm25_score": aggregated_scores[idx],
                "normalized_score": normalized_score,
                "capabilities": metadata['capabilities'],
                "keywords": metadata['keywords'],
                "capability_overlap": capability_overlap,
                "primary_capability": metadata['primary_capability']
            })
        
        # Re-rank by combined score
        for result in formatted_results:
            result['combined_score'] = (
                result['normalized_score'] * 0.6 +
                result['capability_overlap'] * 0.4
            )
        
        formatted_results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return formatted_results[:top_k]
    
    def _search_with_faiss(
        self,
        query: str,
        query_capabilities: List[Tuple[str, float]],
        capability_filter: Optional[List[str]],
        top_k: int,
        expanded_info: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search using FAISS semantic search (with optional expansion)"""
        print("  Using semantic search (FAISS)")
        
        # For FAISS, we'll just use the original query
        # (expansion is more beneficial for keyword search)
        query_embedding = self.embedder.encode([query])
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search FAISS index
        search_k = min(1000, top_k * 20) if capability_filter else top_k
        distances, indices = self.faiss_index.search(
            query_embedding.astype('float32'), 
            search_k
        )
        
        # Format results
        formatted_results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
            
            metadata = self.metadata_store[idx]
            
            # Filter by capability if specified
            if capability_filter:
                if metadata['primary_capability'] not in capability_filter:
                    if not any(cap in capability_filter for cap in metadata['capabilities']):
                        continue
            
            # Compute capability overlap score
            if query_capabilities:
                query_caps = set([cap for cap, _ in query_capabilities[:5]])
                doc_caps = set(metadata['capabilities'])
                overlap = len(query_caps & doc_caps)
                capability_overlap = overlap / len(query_caps) if query_caps else 0
            else:
                capability_overlap = 1.0
            
            # Convert distance to similarity
            similarity = distance
            
            formatted_results.append({
                "text": self.document_store[idx],
                "source_file": metadata['source_file'],
                "chunk_index": metadata['chunk_index'],
                "distance": 1 - similarity,
                "similarity": similarity,
                "capabilities": metadata['capabilities'],
                "keywords": metadata['keywords'],
                "capability_overlap": capability_overlap,
                "primary_capability": metadata['primary_capability']
            })
        
        # Re-rank by combined score
        for result in formatted_results:
            result['combined_score'] = (
                result['similarity'] * 0.6 +
                result['capability_overlap'] * 0.4
            )
        
        formatted_results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return formatted_results[:top_k]
    
    def search_by_capability(
        self,
        capability: str,
        top_k: int = 20
    ) -> List[Dict[str, Any]]:
        """Search documents by specific capability"""
        formatted = []
        
        # Iterate through metadata to find matching capabilities
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
    
    def get_capability_statistics(self):
        """Show capability distribution"""
        print("\n=== Capability Distribution ===")
        cap_dist = self.stats.get('capability_distribution', {})
        
        sorted_caps = sorted(cap_dist.items(), key=lambda x: x[1], reverse=True)
        for cap, count in sorted_caps:
            print(f"  {cap:40s} - {count:4d} chunks")


def main():
    """Example usage with query expansion"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Index wire processing PDFs with query expansion support"
    )
    parser.add_argument("--pdf-folder", required=True, help="Path to PDF folder")
    parser.add_argument("--index-path", default="./wire_index", help="Index storage path")
    parser.add_argument("--action", choices=["index", "search", "stats"], default="index")
    parser.add_argument("--query", help="Search query")
    parser.add_argument("--capability", help="Filter by capability")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results")
    parser.add_argument("--enable-ocr", action="store_true", help="Enable OCR for images")
    parser.add_argument("--no-embeddings", action="store_true", 
                       help="Disable embeddings (use pure keyword/BM25 search - faster)")
    parser.add_argument("--disable-expansion", action="store_true",
                       help="Disable query expansion")
    parser.add_argument("--expansion-level", default="medium",
                       choices=["basic", "medium", "advanced"],
                       help="Query expansion level")
    
    args = parser.parse_args()
    
    if args.action == "index":
        indexer = WireProcessingIndexer(
            pdf_folder=args.pdf_folder,
            index_path=args.index_path,
            enable_ocr=args.enable_ocr,
            use_embeddings=not args.no_embeddings
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
            results = searcher.search(args.query, top_k=args.top_k)
        
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
        searcher.get_capability_statistics()


if __name__ == "__main__":
    main()
