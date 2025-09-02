#!/usr/bin/env python3

"""
TAL Code to Business Capability Linking System
=============================================

This system analyzes TAL (Tandem Application Language) payments functionality and maps them to 
standardized business capabilities defined in a keywords.json taxonomy. It bridges the gap 
between low-level code implementation and high-level business functions.

CONCEPT OVERVIEW:
The linking process works in multiple stages:

1. PARSING: Extract TAL procedures, function calls, and code patterns
2. KEYWORD MATCHING: Match code elements against taxonomy keywords using TF-IDF and fuzzy matching
3. SEMANTIC ANALYSIS: Use LLM to understand business intent beyond keyword matching
4. CAPABILITY MAPPING: Assign specific business capabilities with confidence scores
5. DOMAIN CLASSIFICATION: Group capabilities into business domains (payment-processing, compliance, etc.)

BUSINESS VALUE:
- Convert technical TAL code into business capability inventory
- Identify which regulatory/compliance capabilities are implemented  
- Understand system capabilities across large codebases
- Find gaps between required and implemented capabilities
- Support modernization and compliance initiatives

ARCHITECTURE:
- BusinessCapabilityExtractor: Manages the capability taxonomy from keywords.json
- EnhancedProcedureParser: Extracts and analyzes TAL procedure structures
- DynamicCapabilityMatcher: Core matching engine using multiple analysis methods
- LLMProvider: Optional semantic analysis using GPT-4 for complex mappings

The system is designed to be fully data-driven - all mappings derive from keywords.json
rather than hard-coded rules, making it adaptable to different banking domains.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
import re
import difflib
from dataclasses import dataclass, field
import requests
import time
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import defaultdict, Counter
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BusinessCapabilityMatch:
    """
    Represents the result of mapping a TAL procedure to business capabilities.
    """
    procedure_name: str
    source_file: str
    business_capabilities: List[str]
    primary_keyword_matches: List[str]
    related_keyword_matches: List[str]
    fuzzy_matches: List[str]
    semantic_matches: List[str]
    confidence_score: float
    domain_context: str
    procedural_patterns: List[str]
    reasoning: str
    validation_score: float
    evidence_strength: Dict[str, float]
    llm_analysis: Optional[Dict] = field(default_factory=dict)

class BusinessCapabilityExtractor:
    """
    Manages the business capability taxonomy loaded from keywords.json.
    
    This class builds the core data structures that drive the entire linking process:
    - Capability to domain mappings
    - Domain to capability relationships  
    - Capability specificity scoring (how unique a capability is)
    
    The taxonomy validation helps identify potential issues with over-assigned
    capabilities that appear in too many domains.
    """
    
    def __init__(self, keywords_data: List[Dict]):
        """
        Initialize the capability extractor with keywords.json data.
        
        Args:
            keywords_data: List of dictionaries from keywords.json, each containing:
                - keywords: Primary keywords for matching
                - related_keywords: Secondary keywords for matching
                - business_capability: List of capabilities this entry defines
                - metadata: Domain classification
        """
        self.keywords_data = keywords_data
        
        # Core mapping structures - these drive all capability detection
        self.capability_to_domain = {}      # capability_name -> [domain1, domain2, ...]
        self.domain_to_capabilities = {}    # domain_name -> [cap1, cap2, ...]
        self.all_capabilities = set()       # All unique capability names
        self.capability_hierarchy = {}      # For future hierarchical relationships
        
        # Build all the core mappings from the taxonomy
        self._build_capability_mappings()
        
        # Validate taxonomy structure and warn about potential issues
        self._validate_taxonomy()
    
    def _build_capability_mappings(self):
        """
        Build the core data structures from keywords.json.
        
        This creates bidirectional mappings between capabilities and domains,
        which are essential for:
        - Finding all capabilities in a domain
        - Determining which domains contain a capability
        - Calculating capability specificity scores
        """
        for entry in self.keywords_data:
            domain = entry.get('metadata', 'unknown')
            capabilities = entry.get('business_capability', [])
            
            # Map each capability to its domains (for specificity calculation)
            for capability in capabilities:
                self.all_capabilities.add(capability)
                if capability not in self.capability_to_domain:
                    self.capability_to_domain[capability] = []
                self.capability_to_domain[capability].append(domain)
            
            # Map each domain to its capabilities (for domain analysis)
            if domain not in self.domain_to_capabilities:
                self.domain_to_capabilities[domain] = []
            self.domain_to_capabilities[domain].extend(capabilities)
    
    def _validate_taxonomy(self):
        """
        Validate the taxonomy structure and identify potential issues.
        
        This helps maintain taxonomy quality by identifying:
        - Orphaned capabilities (no domain assignments)
        - Over-assigned capabilities (appear in too many domains)
        
        Over-assignment often indicates capabilities that are too generic
        and may dilute the precision of the matching algorithm.
        """
        issues = []
        
        # Check for orphaned capabilities (shouldn't happen with proper taxonomy)
        orphaned_caps = [cap for cap in self.all_capabilities 
                        if len(self.capability_to_domain[cap]) == 0]
        if orphaned_caps:
            issues.append(f"Orphaned capabilities: {orphaned_caps[:5]}")
        
        # Check for over-assigned capabilities (potential precision issues)
        overassigned_caps = [cap for cap in self.all_capabilities 
                           if len(self.capability_to_domain[cap]) > 3]
        if overassigned_caps:
            issues.append(f"Over-assigned capabilities (>3 domains): {overassigned_caps[:5]}")
        
        # Log validation results for taxonomy maintenance
        if issues:
            logger.warning(f"Taxonomy validation issues: {'; '.join(issues)}")
        else:
            logger.info("Taxonomy validation passed")
    
    def get_capability_specificity_score(self, capability: str) -> float:
        """
        Calculate how specific/unique a capability is across domains.
        
        More specific capabilities (appearing in fewer domains) get higher scores.
        This is used to prioritize specific capabilities over generic ones.
        
        Args:
            capability: The capability name to score
            
        Returns:
            Float between 0.0-1.0, where 1.0 is most specific
        """
        if capability not in self.capability_to_domain:
            return 0.0
        
        domain_count = len(self.capability_to_domain[capability])
        # Inverse relationship: fewer domains = more specific = higher score
        return 1.0 / (1.0 + domain_count)
    
    def get_capabilities_for_domain(self, domain: str) -> List[str]:
        """Get all business capabilities defined for a specific domain."""
        return self.domain_to_capabilities.get(domain, [])
    
    def get_domains_for_capability(self, capability: str) -> List[str]:
        """Get all domains that contain a specific capability."""
        return self.capability_to_domain.get(capability, [])
    
    def get_all_capabilities(self) -> Set[str]:
        """Get all unique business capabilities in the taxonomy."""
        return self.all_capabilities.copy()

class EnhancedProcedureParser:
    """
    Parses TAL source code to extract procedure definitions and analyze their structure.
    
    This parser handles the complexities of TAL syntax including:
    - Nested BEGIN/END blocks
    - Various procedure declaration formats (PROC, SUBPROC, FUNCTION)
    - Function call extraction
    - Utility function detection
    - Complexity scoring for better capability matching
    
    The parser is designed to be robust against syntax variations and provides
    rich metadata about each procedure for the matching algorithms.
    """
    
    def __init__(self):
        """Initialize the TAL parser with regex patterns for procedure detection."""
        
        # Enhanced regex patterns for different TAL procedure types
        # These cover the most common TAL procedure declaration formats
        self.proc_patterns = [
            re.compile(r'(?i)^\s*(?:INT\s+)?PROC\s+([\w_]+)\s*\((.*?)\)\s*;', re.MULTILINE | re.DOTALL),
            re.compile(r'(?i)^\s*(?:INT\s+)?SUBPROC\s+([\w_]+)\s*\((.*?)\)\s*;', re.MULTILINE | re.DOTALL),
            re.compile(r'(?i)^\s*(?:INT\s+)?PROCEDURE\s+([\w_]+)\s*\((.*?)\)\s*;', re.MULTILINE | re.DOTALL),
            re.compile(r'(?i)^\s*(?:INT\s+)?FUNCTION\s+([\w_]+)\s*\((.*?)\)\s*;', re.MULTILINE | re.DOTALL)
        ]
        
        # Patterns for extracting procedure body content
        self.body_pattern = re.compile(r'(?i)BEGIN\s*(.*?)\s*END\s*;', re.DOTALL)
        self.nested_begin_pattern = re.compile(r'(?i)BEGIN', re.MULTILINE)
        self.nested_end_pattern = re.compile(r'(?i)END', re.MULTILINE)
        
        # Patterns for finding function calls within procedures
        self.call_patterns = [
            re.compile(r'(?i)CALL\s+([\w_]+)', re.MULTILINE),        # Explicit CALL statements
            re.compile(r'(?i)([\w_]+)\s*\(', re.MULTILINE),          # Function calls with parentheses
        ]
        
        # Dynamic utility function detection (no hard-coded lists)
        self.utility_indicators = ['error', 'warning', 'debug', 'trace', 'log', 'format', 'normalize']
        
        # Comment pattern for TAL (line comments starting with -- or !)
        self.comment_pattern = re.compile(r'(?:--|!)[^\n]*', re.MULTILINE)
        
        # Pattern for better boundary detection (e.g., RETURN; as potential end)
        self.return_pattern = re.compile(r'(?i)return\s*;', re.MULTILINE)
    
    def _extract_comments(self, content: str) -> List[str]:
        """
        Extract all comments from the TAL content.
        
        Collects line comments starting with -- or ! for potential additional context.
        """
        comments = self.comment_pattern.findall(content)
        return [comment.strip() for comment in comments if comment.strip()]
    
    def _strip_comments(self, content: str) -> str:
        """
        Remove comments from the TAL content to improve parsing accuracy.
        
        Strips line comments starting with -- or ! before procedure extraction.
        """
        return self.comment_pattern.sub('', content)
    
    def _extract_balanced_body(self, content: str, start_pos: int) -> Tuple[str, int]:
        """
        Extract procedure body handling nested BEGIN/END blocks.
        
        TAL procedures can have complex nested structures, so we need to
        properly balance BEGIN/END pairs to extract the complete procedure body.
        """
        remaining = content[start_pos:]
        
        # Look for the first BEGIN to start the procedure body
        begin_match = self.nested_begin_pattern.search(remaining)
        if not begin_match:
            # No explicit BEGIN/END structure, use heuristic extraction
            return self._extract_implicit_body(remaining), len(remaining)
        
        # Balance nested BEGIN/END pairs to find the complete procedure
        begin_pos = begin_match.start()
        current_pos = begin_pos + len("BEGIN")
        depth = 1  # Track nesting depth
        
        # Walk through the content balancing BEGIN/END pairs
        while depth > 0 and current_pos < len(remaining):
            next_begin = self.nested_begin_pattern.search(remaining, current_pos)
            next_end = self.nested_end_pattern.search(remaining, current_pos)
            
            if next_end is None:
                # No more END statements found, break out
                break
            
            # Process the next keyword (BEGIN or END)
            if next_begin and next_begin.start() < next_end.start():
                # Found nested BEGIN, increase depth
                depth += 1
                current_pos = next_begin.end()
            else:
                # Found END, decrease depth
                depth -= 1
                current_pos = next_end.end()
                if depth == 0:
                    # Found the matching END for our procedure
                    body_content = remaining[begin_match.end():next_end.start()]
                    return body_content.strip(), start_pos + current_pos
        
        # Fallback if balancing failed
        return self._extract_implicit_body(remaining), len(remaining)
    
    def _extract_implicit_body(self, remaining: str) -> str:
        """
        Extract procedure body when no explicit BEGIN/END structure exists.
        
        This handles cases where procedures don't follow strict BEGIN/END
        patterns by looking for the next procedure or taking
        a reasonable chunk of code.
        """
        next_proc_pos = len(remaining)
        
        # Find the start of the next procedure to bound this one
        for pattern in self.proc_patterns:
            next_match = pattern.search(remaining, 1)  # Skip current position
            if next_match:
                next_proc_pos = min(next_proc_pos, next_match.start())
        
        # New: Look for RETURN; as an alternative boundary if no next procedure
        return_match = self.return_pattern.search(remaining)
        if return_match:
            body_end = min(next_proc_pos, return_match.end())
        else:
            body_end = next_proc_pos
        
        # Extract content up to next boundary or reasonable limit
        body_end = min(body_end, 3000)  # Cap at 3000 chars for performance
        return remaining[:body_end]
    
    def _is_utility_function(self, proc_name: str, proc_body: str) -> bool:
        """
        Determine if a procedure is a utility function using dynamic detection.
        
        Utility functions typically provide support services (logging, formatting, etc.)
        rather than core business functionality. Identifying them helps focus
        capability mapping on business-relevant procedures.
        """
        name_lower = proc_name.lower()
        body_lower = proc_body.lower()
        
        # Check for common utility prefixes in procedure names
        utility_prefixes = ['add_', 'log_', 'format_', 'normalize_', 'get_', 'set_', 'init_']
        if any(name_lower.startswith(prefix) for prefix in utility_prefixes):
            return True
        
        # Check for utility suffixes in procedure names
        if any(name_lower.endswith(suffix) for suffix in self.utility_indicators):
            return True
        
        # Analyze body content for utility characteristics
        utility_indicators_in_body = sum(1 for indicator in self.utility_indicators if indicator in body_lower)
        simple_operations = ['return', 'printf', 'sprintf', 'strlen', 'strcmp']
        simple_count = sum(1 for op in simple_operations if op in body_lower)
        
        # Short procedures with many utility operations are likely utilities
        if len(proc_body) < 500 and (utility_indicators_in_body >= 2 or simple_count >= 2):
            return True
        
        return False
    
    def extract_procedures(self, content: str, source_file: str) -> List[Dict]:
        """
        Main procedure extraction method.
        
        Processes TAL source code to extract all procedure definitions along with
        rich metadata for capability matching. Handles deduplication and provides
        statistics for monitoring the parsing process.
        """
        # New: Extract all comments from the full content
        all_comments = self._extract_comments(content)
        
        # New: Strip comments from content to improve parsing accuracy
        clean_content = self._strip_comments(content)
        
        procedures = []
        found_procedures = set()  # Global deduplication
        
        # Statistics for monitoring and debugging
        extraction_stats = {
            'total_found': 0,     # Total procedure declarations found
            'duplicates': 0,      # Duplicate procedures skipped
            'utilities': 0,       # Utility functions identified
            'parsed': 0          # Successfully parsed procedures
        }
        
        # Process each procedure pattern type
        for pattern in self.proc_patterns:
            matches = list(pattern.finditer(clean_content))
            extraction_stats['total_found'] += len(matches)
            
            for match in matches:
                proc_name = match.group(1).strip()
                
                # Skip duplicates (can occur across files in large repos)
                if proc_name in found_procedures:
                    extraction_stats['duplicates'] += 1
                    continue
                found_procedures.add(proc_name)
                
                # Extract procedure parameters
                parameters = match.group(2).strip() if match.group(2) else ""
                
                # Extract procedure body using balanced parsing
                body_content, body_end = self._extract_balanced_body(clean_content, match.end())
                
                # Extract function calls from the procedure body
                function_calls = set()
                for call_pattern in self.call_patterns:
                    calls = call_pattern.findall(body_content)
                    function_calls.update(calls)
                
                # Clean up function calls (remove self-references and short names)
                function_calls.discard(proc_name)  # Remove self-references
                function_calls = [call for call in function_calls if len(call) > 2]
                
                # Determine if this is a utility function
                is_utility = self._is_utility_function(proc_name, body_content)
                if is_utility:
                    extraction_stats['utilities'] += 1
                
                # New: Extract comments specific to this procedure (from start to end)
                proc_start = match.start()
                proc_end = match.end() + body_end
                proc_comments = self._extract_comments(content[proc_start:proc_end])
                
                # Create comprehensive procedure metadata
                procedures.append({
                    'name': proc_name,
                    'source_file': source_file,
                    'parameters': parameters,
                    'body': body_content,
                    'function_calls': sorted(list(function_calls)),
                    'full_context': clean_content[max(0, match.start() - 200):match.end() + min(len(body_content), 2000)],
                    'is_utility': is_utility,
                    'body_length': len(body_content),
                    'complexity_score': self._calculate_complexity_score(body_content, function_calls),
                    'comments': proc_comments  # New: Extracted comments for the procedure
                })
                extraction_stats['parsed'] += 1
        
        # Log extraction statistics for monitoring
        logger.info(f"Extraction stats for {Path(source_file).name}: {extraction_stats}")
        return procedures
    
    def _calculate_complexity_score(self, body: str, function_calls: List[str]) -> float:
        """
        Calculate a complexity score for the procedure.
        
        This score is used by the matching algorithm to:
        - Determine appropriate confidence thresholds
        - Limit capability assignments based on procedure sophistication
        - Identify procedures likely to implement multiple capabilities
        """
        score = 0.0
        
        # Length-based complexity (normalized)
        score += min(len(body) / 1000.0, 3.0)
        
        # Function call complexity
        score += len(function_calls) * 0.3
        
        # Control flow complexity (increased weight for loops/conditions)
        control_keywords = ['if', 'while', 'for', 'case', 'loop']
        for keyword in control_keywords:
            score += body.lower().count(keyword) * 0.2
        
        return min(score, 10.0)  # Cap at 10.0 for normalization

class DynamicCapabilityMatcher:
    """
    Core matching engine that maps TAL procedures to business capabilities.
    
    This is the heart of the linking system, implementing a multi-stage analysis:
    
    1. DIRECT NAME MATCHING: Semantic alignment between procedure names and capabilities
    2. CODE ANALYSIS: Pattern matching within procedure bodies using keywords
    3. LLM SEMANTIC BRIDGE: Advanced semantic understanding via GPT-4
    4. VALIDATION & COMBINATION: Intelligent merging with confidence scoring
    
    The matcher is fully data-driven - all logic derives from keywords.json
    rather than hard-coded rules, making it adaptable to different domains.
    """
    
    def __init__(self, keywords_data: List[Dict], fuzzy_threshold: float = 0.85):
        """Initialize the capability matcher."""
        self.keywords_data = keywords_data
        self.fuzzy_threshold = fuzzy_threshold
        self.capability_extractor = BusinessCapabilityExtractor(keywords_data)
        self.llm_provider = None  # Set via set_llm_provider() if LLM analysis desired
        
        # Build dynamic mappings from keywords.json - these drive all matching logic
        self.domain_keywords = {}               # domain -> [primary_keywords]
        self.domain_related_keywords = {}       # domain -> [related_keywords]
        self.keyword_importance_scores = {}     # keyword -> importance_score
        self.capability_keyword_map = {}        # capability -> {primary: [], related: []}
        
        # Build all all the core data structures from the taxonomy
        self._build_dynamic_mappings()
        
        # Initialize all TF-IDF for semantic similarity scoring
        self._init_enhanced_tfidf()
        
        # Dynamic false positive detection patterns
        self.validation_cache = {}
        self.false_positive_patterns = self._build_dynamic_false_positive_patterns()
    
    def _build_dynamic_mappings(self):
        """
        Build all core data structures from keywords.json.
        
        This creates the mappings that drive the entire matching process:
        - Domain to keyword relationships
        - Capability to keyword associations
        - Keyword importance scoring (based on inverse frequency)
        """
        keyword_frequency = Counter()  # Track keyword frequency for importance scoring
        
        # Process each entry in the taxonomy
        for entry in self.keywords_data:
            domain = entry.get('metadata', 'unknown')
            primary_kw = [kw.strip().lower() for kw in entry.get('keywords', '').split(',') if kw.strip()]
            related_kw = [kw.strip().lower() for kw in entry.get('related_keywords', '').split(',') if kw.strip()]
            capabilities = entry.get('business_capability', [])
            
            # Build domain-keyword mappings
            self.domain_keywords[domain] = primary_kw
            self.domain_related_keywords[domain] = related_kw
            
            # Build capability-keyword mappings (crucial for direct name matching)
            for capability in capabilities:
                if capability not in self.capability_keyword_map:
                    self.capability_keyword_map[capability] = {'primary': [], 'related': []}
                self.capability_keyword_map[capability]['primary'].extend(primary_kw)
                self.capability_keyword_map[capability]['related'].extend(related_kw)
            
            # Count keyword frequency for importance scoring
            for kw in primary_kw + related_kw:
                keyword_frequency[kw] += 1
        
        # Calculate keyword importance scores (inverse frequency)
        # Rare keywords get higher importance scores, common ones get lower scores
        max_freq = max(keyword_frequency.values()) if keyword_frequency else 1
        for keyword, freq in keyword_frequency.items():
            self.keyword_importance_scores[keyword] = max_freq / freq
        
        logger.info(f"Built dynamic mappings: {len(self.domain_keywords)} domains, "
                   f"{len(self.capability_keyword_map)} capabilities")
    
    def _build_dynamic_false_positive_patterns(self) -> List[re.Pattern]:
        """
        Build patterns to detect likely false positives dynamically from the data.
        
        Rather than hard-coding utility patterns, this examines the taxonomy
        to identify terms that typically indicate utility functions.
        """
        patterns = []
        
        # Extract utility-like terms from the taxonomy itself
        utility_terms = set()
        for entry in self.keywords_data:
            domain = entry.get('metadata', 'unknown')
            keywords = entry.get('keywords', '') + ',' + entry.get('related_keywords', '')
            for word in re.findall(r'\b\w+\b', keywords.lower()):
                # Common utility indicators found in banking taxonomies
                if word in ['error', 'warning', 'debug', 'trace', 'log', 'format', 'normalize']:
                    utility_terms.add(word)
        
        # Create regex patterns for utility detection
        if utility_terms:
            patterns.append(re.compile(r'(?i)^(add_|get_|set_|init_|log_|format_|normalize_)'))
            utility_pattern = '|'.join(utility_terms)
            patterns.append(re.compile(f'(?i)({utility_pattern})$'))
        
        return patterns
    
    def _init_enhanced_tfidf(self):
        """Initialize TF-IDF vectorizer for semantic similarity scoring."""
        try:
            self.tfidf_docs = []
            self.tfidf_metadata = []
            
            # Create a document for each taxonomy entry
            for entry in self.keywords_data:
                # Combine all textual information into a single document
                doc_parts = [
                    entry.get('keywords', ''),
                    entry.get('related_keywords', ''),
                    entry.get('description', ''),
                    ' '.join(entry.get('business_capability', []))  # Include capability names
                ]
                doc_text = ' '.join(part for part in doc_parts if part)
                
                self.tfidf_docs.append(doc_text)
                self.tfidf_metadata.append({
                    'domain': entry.get('metadata', 'unknown'),
                    'capabilities': entry.get('business_capability', []),
                    'primary_keywords': [kw.strip().lower() for kw in entry.get('keywords', '').split(',') if kw.strip()]
                })
            
            # Initialize the TF-IDF vectorizer with appropriate parameters
            if self.tfidf_docs:
                self.tfidf_vectorizer = TfidfVectorizer(
                    stop_words='english',    # Remove common English words
                    max_features=8000,       # Limit vocabulary size for performance
                    ngram_range=(1, 3),      # Include unigrams, bigrams, and trigrams
                    min_df=1,                # Don't ignore rare terms (important for banking domain)
                    max_df=0.8               # Ignore very common terms
                )
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.tfidf_docs)
                logger.info(f"Enhanced TF-IDF initialized with {len(self.tfidf_docs)} documents")
            else:
                self.tfidf_vectorizer = None
                self.tfidf_matrix = None
                logger.warning("No TF-IDF documents available")
                
        except Exception as e:
            logger.error(f"TF-IDF initialization failed: {e}")
            self.tfidf_vectorizer = None
            self.tfidf_matrix = None
    
    def set_llm_provider(self, llm_provider):
        """Set the LLM provider for semantic analysis (optional component)."""
        self.llm_provider = llm_provider
    
    def match_procedure_to_capabilities(self, procedure: Dict, min_confidence: float = 0.2) -> BusinessCapabilityMatch:
        """
        Main matching method - maps a TAL procedure to business capabilities.
        
        This orchestrates the multi-stage analysis process:
        1. Early filtering for utility functions
        2. Direct name-based capability detection
        3. Code content analysis using keyword patterns
        4. Optional LLM semantic analysis
        5. Validation and combination of results
        6. Confidence scoring and domain classification
        """
        # Extract procedure metadata for analysis
        proc_name = procedure['name']
        proc_body = procedure['body'].lower()
        function_calls = procedure.get('function_calls', [])

        full_context = procedure.get('full_context', '').lower()
        is_utility = procedure.get('is_utility', False)
        complexity_score = procedure.get('complexity_score', 0.0)
        
        # EARLY FILTERING: Skip detailed analysis for simple utility functions
        if is_utility and complexity_score < 2.0:
            return self._create_minimal_match(procedure, "Utility function - minimal capabilities")
        
        # FALSE POSITIVE DETECTION: Check for patterns likely to produce noise
        if self._is_likely_false_positive(proc_name, proc_body):
            return self._create_minimal_match(procedure, "Likely false positive pattern detected")
        
        # STAGE 1: DIRECT NAME MATCHING
        # Semantic alignment between procedure names and capability names
        direct_capabilities = self._find_direct_name_matches(proc_name)
        
        # STAGE 2: CODE CONTENT ANALYSIS
        # Analyze procedure body and function calls for capability indicators
        code_detected_capabilities = self._analyze_code_dynamically(proc_name, proc_body, full_context, function_calls)
        
        # STAGE 3: LLM SEMANTIC ANALYSIS (Optional)
        # Use advanced AI to find capabilities missed by algorithmic approaches
        llm_capabilities = []
        llm_reasoning = ""
        llm_confidence = 0.0
        llm_details = {}
        
        if self.llm_provider and self.llm_provider.is_available() and complexity_score >= 0.5:
            llm_result = self._enhanced_llm_semantic_bridge(procedure, direct_capabilities + code_detected_capabilities)
            if isinstance(llm_result, dict):
                llm_capabilities = llm_result.get('additional_capabilities', [])
                llm_reasoning = llm_result.get('reasoning', '')
                llm_confidence = llm_result.get('confidence', 0.0)
                llm_details = {
                    'capability_details': llm_result.get('capability_details', []),
                    'primary_domain': llm_result.get('primary_domain', 'unknown')
                }
        
        # STAGE 4: VALIDATION AND COMBINATION
        # Intelligently combine results from all analysis stages
        all_capabilities = self._validate_and_combine_capabilities(
            direct_capabilities, code_detected_capabilities, llm_capabilities, 
            proc_name, proc_body, complexity_score
        )
        
        # STAGE 5: CONFIDENCE CALCULATION
        # Calculate detailed confidence scores based on evidence strength
        confidence_details = self._calculate_enhanced_confidence(
            proc_name, proc_body, direct_capabilities, code_detected_capabilities, 
            llm_capabilities, llm_confidence, complexity_score
        )
        
        # STAGE 6: DOMAIN CLASSIFICATION
        # Determine the primary business domain for this procedure
        best_domain = self._find_enhanced_domain_match(proc_name, proc_body, list(all_capabilities))
        
        # STAGE 7: KEYWORD MATCH ANALYSIS
        # Get detailed breakdown of keyword matches for transparency
        keyword_matches = self._get_enhanced_keyword_matches(proc_name + " " + proc_body)
        
        # STAGE 8: VALIDATION SCORING
        # Calculate quality score for the overall mapping
        validation_score = self._calculate_validation_score(
            proc_name, list(all_capabilities), keyword_matches, complexity_score
        )
        
        # STAGE 9: PATTERN DETECTION
        # Identify procedural patterns for additional context
        procedural_patterns = self._detect_enhanced_procedural_patterns(proc_name, proc_body)
        
        # Create comprehensive results object
        return BusinessCapabilityMatch(
            procedure_name=proc_name,
            source_file=procedure['source_file'],
            business_capabilities=list(all_capabilities),
            primary_keyword_matches=keyword_matches.get('primary', []),
            related_keyword_matches=keyword_matches.get('related', []),
            fuzzy_matches=keyword_matches.get('fuzzy', []),
            semantic_matches=llm_capabilities,
            confidence_score=round(confidence_details['total'], 3),
            domain_context=best_domain,
            procedural_patterns=procedural_patterns,
            reasoning=llm_reasoning or self._build_reasoning_explanation(
                direct_capabilities, code_detected_capabilities, llm_capabilities
            ),
            validation_score=validation_score,
            evidence_strength=confidence_details,
            llm_analysis=llm_details  # Add llm analysis
        )
    
    def _find_direct_name_matches(self, proc_name: str) -> List[str]:
        """
        Find direct semantic matches between procedure names and capabilities.
        
        This stage looks for alignment between procedure naming conventions
        and capability names, using both exact matches and semantic similarity.
        """
        name_lower = proc_name.lower()
        direct_caps = []
        
        # Iterate through all capabilities and their keyword associations
        for capability, keyword_map in self.capability_keyword_map.items():
            match_score = 0.0
            
            # KEYWORD MATCHING: Check primary keywords first (highest confidence)
            for keyword in keyword_map['primary']:
                if self._keyword_matches_procedure_name(keyword, name_lower):
                    importance = self.keyword_importance_scores.get(keyword, 1.0)
                    match_score += importance * 2.0  # High weight for primary keywords
            
            # Check related keywords (medium confidence)
            for keyword in keyword_map['related']:
                if self._keyword_matches_procedure_name(keyword, name_lower):
                    importance = self.keyword_importance_scores.get(keyword, 1.0)
                    match_score += importance * 1.0  # Lower weight for related keywords
            
            # SEMANTIC ALIGNMENT: Direct capability name to procedure name matching
            cap_words = capability.lower().replace(' ', '_').split('_')
            proc_words = name_lower.replace('_', ' ').split()
            
            # Score semantic alignment between capability and procedure names
            for cap_word in cap_words:
                if len(cap_word) > 2:  # Skip very short words
                    for proc_word in proc_words:
                        if cap_word == proc_word:  # Exact word match
                            match_score += 3.0
                        elif cap_word in proc_word or proc_word in cap_word:  # Partial match
                            match_score += 1.5
            
            # Include capability if match score exceeds threshold
            if match_score >= 2.0:
                direct_caps.append((capability, match_score))
        
        # Sort by match score (highest confidence first) and return capability names
        direct_caps.sort(key=lambda x: x[1], reverse=True)
        return [cap for cap, score in direct_caps]
    
    def _keyword_matches_procedure_name(self, keyword: str, proc_name_lower: str) -> bool:
        """Sophisticated keyword matching against procedure names."""
        # STRATEGY 1: Direct substring match (highest confidence)
        if keyword in proc_name_lower:
            return True
        
        # STRATEGY 2: Word-level matching for multi-word keywords
        keyword_words = keyword.replace('_', ' ').split()
        proc_words = proc_name_lower.replace('_', ' ').split()
        
        # For multi-word keywords, check if all words appear in procedure words
        if len(keyword_words) > 1:
            return all(any(kw in pw or pw in kw for pw in proc_words) for kw in keyword_words)
        
        # STRATEGY 3: Single word partial matching (for morphological variations)
        if len(keyword_words) == 1 and len(keyword_words[0]) > 3:
            return any(keyword_words[0] in pw or pw in keyword_words[0] for pw in proc_words)
        
        return False
    
    def _analyze_code_dynamically(self, proc_name: str, proc_body: str, 
                                full_context: str, function_calls: List[str]) -> List[str]:
        """
        Analyze procedure code content for capability indicators.
        
        This stage examines the actual code implementation to find evidence
        of business capabilities through keyword presence and function calls.
        """
        capabilities = []
        
        # EARLY FILTERING: Skip detailed analysis for simple utility functions
        if len(proc_body) < 200 and any(proc_name.lower().startswith(prefix) 
                                       for prefix in ['add_', 'log_', 'format_', 'get_', 'set_']):
            return capabilities
        
        # Combine all available text for analysis, emphasizing function calls
        all_text = f"{proc_body} {full_context} {' '.join(function_calls)}".lower()
        
        # Iterate through each taxonomy entry to find matching capabilities
        for entry in self.keywords_data:
            business_caps = entry.get('business_capability', [])
            if not business_caps:
                continue  # Skip entries without capabilities
            
            # Extract keywords for this domain
            primary_keywords = [kw.strip().lower() for kw in entry.get('keywords', '').split(',') if kw.strip()]
            related_keywords = [kw.strip().lower() for kw in entry.get('related_keywords', '').split(',') if kw.strip()]
            
            # SCORING: Calculate match strength for this domain
            match_score = 0.0
            found_keywords = []
            
            # Score primary keyword matches (high confidence indicators)
            for kw in primary_keywords:
                if kw in all_text:
                    importance = self.keyword_importance_scores.get(kw, 1.0)
                    
                    # FUNCTION CALL BOOST: Keywords in function calls get higher scores
                    multiplier = 3.0 if any(kw in call.lower() for call in function_calls) else 1.0
                    match_score += importance * multiplier
                    found_keywords.append(kw)
            
            # Score related keyword matches (medium confidence indicators)
            for kw in related_keywords:
                if kw in all_text:
                    importance = self.keyword_importance_scores.get(kw, 1.0)
                    
                    # Function call boost for related keywords too, but less weight
                    multiplier = 2.0 if any(kw in call.lower() for call in function_calls) else 0.5
                    match_score += importance * multiplier * 0.5  # Lower base weight
                    found_keywords.append(kw)
            
            # CAPABILITY FILTERING: Only include capabilities with sufficient evidence
            if match_score >= 2.0:  # Increased threshold for quality
                relevant_caps = self._filter_capabilities_by_relevance(
                    business_caps, found_keywords, proc_name, all_text, match_score
                )
                capabilities.extend(relevant_caps)
        
        return list(set(capabilities))  # Remove duplicates
    
    def _filter_capabilities_by_relevance(self, capabilities: List[str], found_keywords: List[str], 
                                        proc_name: str, all_text: str, match_score: float) -> List[str]:
        """
        Filter domain capabilities to only the most relevant ones for this procedure.
        
        When a domain matches a procedure, not all all capabilities in that domain
        are necessarily relevant. This method scores each capability based on
        its specific relevance to the procedure.
        """
        relevant_caps = []
        
        for capability in capabilities:
            relevance_score = 0.0
            cap_words = capability.lower().replace(' ', '_').split('_')
            
            # FACTOR 1: Capability name alignment with procedure name (highest weight)
            proc_words = proc_name.lower().replace('_', ' ').split()
            for cap_word in cap_words:
                if len(cap_word) > 2:
                    for proc_word in proc_words:
                        if cap_word == proc_word:  # Exact word match
                            relevance_score += 4.0
                        elif cap_word in proc_word or proc_word in cap_word:  # Partial match
                            relevance_score += 2.0
            
            # FACTOR 2: Keyword-capability semantic relationships
            for keyword in found_keywords:
                keyword_words = keyword.replace(' ', '_').split('_')
                for kw_word in keyword_words:
                    if len(kw_word) > 2:
                        for cap_word in cap_words:
                            if kw_word == cap_word:
                                relevance_score += 1.5
                            elif kw_word in cap_word or cap_word in kw_word:
                                relevance_score += 1.0
            
            # FACTOR 3: Capability specificity boost (prefer specific capabilities)
            specificity = self.capability_extractor.get_capability_specificity_score(capability)
            relevance_score += specificity * 3.0
            
            # DYNAMIC THRESHOLDS: Adjust threshold based on overall match strength
            # not sure we need this..
            base_threshold = 3.0
            if match_score > 8.0:
                threshold = base_threshold * 0.6
            elif match_score > 5.0:
                threshold = base_threshold * 0.8
            else:
                threshold = base_threshold * 1.2
            
            # Include capability if it meets the relevance threshold
            if relevance_score >= threshold:
                relevant_caps.append((capability, relevance_score))
        
        # Sort by relevance score and return capability names
        relevant_caps.sort(key=lambda x: x[1], reverse=True)
        return [cap for cap, score in relevant_caps]
    
    def _enhanced_llm_semantic_bridge(self, procedure: Dict, already_detected: List[str]) -> Dict:
        """Enhanced semantic analysis using LLM with optimized token usage."""
        if not self.llm_provider or not self.llm_provider.is_available():
            return {}
        
        # TOKEN OPTIMIZATION: Create focused keyword entries based on procedure context
        proc_text = f"{procedure['name']} {procedure.get('full_context', '')[:1000]}".lower()
        
        # SMART FILTERING: Select only relevant taxonomy entries to reduce payload size
        relevant_entries = []
        for entry in self.keywords_data:
            # Extract all keywords from this entry
            all_keywords = (entry.get('keywords', '') + ',' + entry.get('related_keywords', '')).lower()
            keyword_list = [kw.strip() for kw in all_keywords.split(',') if kw.strip()]
            
            # Include entry if it has keyword matches OR undetected capabilities
            has_keyword_match = any(kw in proc_text for kw in keyword_list if len(kw) > 3)
            has_undetected_caps = any(cap not in already_detected 
                                    for cap in entry.get('business_capability', []))
            
            if has_keyword_match or has_undetected_caps:
                relevant_entries.append(entry)
        
        # Limit to top 15 most relevant entries to stay under token limits
        relevant_entries = relevant_entries[:15]
        
        # Create condensed taxonomy with only essential fields
        focused_taxonomy = []
        for entry in relevant_entries:
            focused_taxonomy.append({
                "keywords": entry.get('keywords', ''),
                "related_keywords": entry.get('related_keywords', ''),
                "business_capability": entry.get('business_capability', []),
                "metadata": entry.get('metadata', '')
            })
        
        # Truncate procedure context for token efficiency
        max_code_length = 2000
        code_context = procedure['full_context'][:max_code_length]
        if len(procedure['full_context']) > max_code_length:
            code_context += "\n... [truncated]"
        
        # Dynamically build evidence patterns from the taxonomy
        evidence_patterns = []
        for entry in focused_taxonomy:
            keywords = entry.get('keywords', '') + ', ' + entry.get('related_keywords', '')
            capabilities = entry.get('business_capability', [])
            if keywords and capabilities:
                # Create dynamic pattern from actual taxonomy data
                key_terms = [kw.strip() for kw in keywords.split(',') if kw.strip()][:3]  # Top 3 terms
                cap_list = capabilities[:3]  # Top 3 capabilities
                if key_terms and cap_list:
                    pattern = f"- {', '.join(key_terms)} → {', '.join(cap_list)}"
                    evidence_patterns.append(pattern)
        
        # Get all unique business capabilities from the taxonomy
        all_capabilities = set()
        for entry in focused_taxonomy:
            all_capabilities.update(entry.get('business_capability', []))
        
        evidence_section = '\n'.join(evidence_patterns) if evidence_patterns else "Look for keywords and patterns in the taxonomy above"
        
        # STRUCTURED PROMPT: Clear instructions for payment domain analysis
        prompt = f"""Analyze this TAL banking procedure to identify which payment domain business capabilities it implements.

    PROCEDURE DETAILS:
    Name: {procedure['name']}
    Function Calls: {', '.join(procedure.get('function_calls', [])[:8])}

    Code Context:
    {code_context}

    BUSINESS CAPABILITIES TAXONOMY:
    {json.dumps(focused_taxonomy, indent=1)}

    ALL AVAILABLE CAPABILITIES:
    {sorted(list(all_capabilities))}

    ALREADY DETECTED: {', '.join(already_detected) if already_detected else 'None'}

    ANALYSIS TASK:
    1. Read the TAL procedure code carefully
    2. Identify what banking/payment operations this procedure performs
    3. Match these operations to business capabilities from the ALL AVAILABLE CAPABILITIES list above
    4. Use the taxonomy entries to understand which keywords relate to which capabilities
    5. Focus on capabilities NOT already detected that have clear evidence in the code

    EVIDENCE PATTERNS FROM TAXONOMY:
    {evidence_section}

    Respond with JSON only:
    {{
        "capabilities": [
            {{
                "name": "Exact business capability name from ALL AVAILABLE CAPABILITIES list",
                "confidence": 0.85,
                "evidence": "Specific code evidence (function calls, validation logic, etc.)"
            }}
        ],
        "reasoning": "Brief explanation of why these capabilities match the procedure's actual functionality",
        "primary_domain": "Best matching metadata domain from taxonomy"
    }}"""
        
        system_prompt = """You are a payments domain expert analyzing TAL banking procedures.

    Your task: Map TAL procedures to specific business capabilities based on what the code actually does.

    CRITICAL RULES:
    1. Only suggest capabilities that appear in the provided taxonomy
    2. Base suggestions on actual code evidence, not procedure names, including function calls and comments
    3. Focus on what the procedure validates, processes, or manages
    4. Provide confidence scores between 0.4-0.95 (be conservative)
    5. Give specific code evidence for each capability

    OUTPUT: Valid JSON only, no explanatory text outside the JSON."""
        
        try:
            # Make the LLM API call
            result = self.llm_provider.call_chat_completion(prompt, system_prompt)
            
            if isinstance(result, dict) and 'capabilities' in result:
                # VALIDATION: Extract and validate returned capabilities
                valid_capabilities = []
                capability_details = []
                
                for mapping in result.get('capabilities', []):
                    capability = mapping.get('name', '')
                    confidence = mapping.get('confidence', 0.0)
                    evidence = mapping.get('evidence', '')
                    
                    # Ensure capability exists in our taxonomy and meets confidence threshold
                    all_capabilities = self.capability_extractor.get_all_capabilities()
                    if capability in all_capabilities and confidence >= 0.25:
                        valid_capabilities.append(capability)
                        capability_details.append({
                            'capability': capability,
                            'confidence': confidence,
                            'evidence': evidence
                        })
                    else:
                        logger.debug(f"LLM suggested invalid/low-confidence capability: {capability} (conf: {confidence})")
                
                logger.info(f"LLM found {len(valid_capabilities)} valid capabilities for {procedure['name']}")
                
                return {
                    'additional_capabilities': valid_capabilities,
                    'reasoning': result.get('reasoning', 'LLM analysis of procedure functionality'),
                    'confidence': sum(d['confidence'] for d in capability_details) / len(capability_details) if capability_details else 0.0,
                    'capability_details': capability_details,
                    'primary_domain': result.get('primary_domain', 'unknown')
                }
            elif isinstance(result, dict):
                # Handle API errors gracefully
                if 'error' in result:
                    logger.warning(f"LLM API error for {procedure['name']}: {result['error']}")
                else:
                    logger.info(f"LLM returned dict for {procedure['name']} with keys: {list(result.keys())}")
                return {}
            else:
                logger.warning(f"LLM returned unexpected format for {procedure['name']}: {type(result)}")
                return {}
        except Exception as e:
            logger.error(f"LLM semantic bridge failed for {procedure['name']}: {e}")
            return {}
    
    def _is_likely_false_positive(self, proc_name: str, proc_body: str) -> bool:
        """Detect procedures likely to produce false positive capability matches."""
        # Check against dynamically built false positive patterns
        for pattern in self.false_positive_patterns:
            if pattern.search(proc_name):
                return True
        
        # HEURISTIC 1: Very short procedures with simple return statements
        if len(proc_body) < 100 and 'return' in proc_body:
            return True
        
        # HEURISTIC 2: Procedures with too many generic keywords (potential noise)
        generic_count = 0
        for entry in self.keywords_data:
            keywords = (entry.get('keywords', '') + ',' + entry.get('related_keywords', '')).lower()
            generic_terms = ['data', 'process', 'handle', 'manage', 'system']
            for term in generic_terms:
                if term in keywords and term in proc_body:
                    generic_count += 1
        
        if generic_count > 3 and len(proc_body) < 500:
            return True
        
        return False
    
    def _create_minimal_match(self, procedure: Dict, reason: str) -> BusinessCapabilityMatch:
        """Create a minimal match result for utility functions or false positives."""
        return BusinessCapabilityMatch(
            procedure_name=procedure['name'],
            source_file=procedure['source_file'],
            business_capabilities=[],
            primary_keyword_matches=[],
            related_keyword_matches=[],
            fuzzy_matches=[],
            semantic_matches=[],
            confidence_score=0.0,
            domain_context='utility',
            procedural_patterns=[],
            reasoning=reason,
            validation_score=0.0,
            evidence_strength={'direct': 0.0, 'code': 0.0, 'llm': 0.0, 'total': 0.0}
        )
    
    def _validate_and_combine_capabilities(self, direct_caps: List[str], code_caps: List[str], 
                                         llm_caps: List[str], proc_name: str, proc_body: str, 
                                         complexity_score: float) -> Set[str]:
        """Intelligently combine capabilities from different analysis stages."""
        validated_caps = set(direct_caps)  # Start with highest-confidence matches
        
        # Add code-detected capabilities with validation
        for cap in code_caps:
            specificity = self.capability_extractor.get_capability_specificity_score(cap)
            # More specific capabilities get lower thresholds
            threshold = 0.3 if specificity > 0.5 else 0.6
            
            if complexity_score >= threshold:
                validated_caps.add(cap)
        
        # Add LLM capabilities with stricter validation
        for cap in llm_caps:
            if complexity_score >= 1.0:  # LLM capabilities need higher complexity
                validated_caps.add(cap)
        
        # Remove overly generic capabilities for simple procedures
        if complexity_score < 2.0:
            # Dynamically identify generic capabilities based on domain frequency
            generic_caps = set()
            for cap in validated_caps:
                domains = self.capability_extractor.get_domains_for_capability(cap)
                if len(domains) > 5:  # Capabilities in many domains are generic
                    generic_caps.add(cap)
            validated_caps -= generic_caps
        
        # Limit total capabilities based on procedure complexity
        max_caps = max(3, int(complexity_score))
        if len(validated_caps) > max_caps:
            # Sort by specificity score (more specific capabilities prioritized)
            cap_specificity = [(cap, self.capability_extractor.get_capability_specificity_score(cap)) 
                             for cap in validated_caps]
            cap_specificity.sort(key=lambda x: x[1], reverse=True)
            validated_caps = set([cap for cap, _ in cap_specificity[:max_caps]])
        
        return validated_caps
    
    def _calculate_enhanced_confidence(self, proc_name: str, proc_body: str, 
                                     direct_caps: List[str], code_caps: List[str], 
                                     llm_caps: List[str], llm_confidence: float,
                                     complexity_score: float) -> Dict[str, float]:
        """Calculate detailed confidence scores with breakdown by evidence source."""
        confidence_details = {
            'direct': 0.0,              # Confidence from direct name matching
            'code': 0.0,                # Confidence from code analysis  
            'llm': 0.0,                 # Confidence from LLM analysis
            'complexity_bonus': 0.0,    # Bonus for high complexity procedures
            'specificity_bonus': 0.0,   # Bonus for specific capabilities
            'total': 0.0                # Combined total confidence
        }
        
        # Direct mapping confidence (highest weight due to high precision)
        if direct_caps:
            confidence_details['direct'] = min(len(direct_caps) * 0.4, 0.8)
        
        # Code analysis confidence weighted by capability specificity
        if code_caps:
            specificity_sum = sum(self.capability_extractor.get_capability_specificity_score(cap) 
                                for cap in code_caps)
            confidence_details['code'] = min(specificity_sum * 0.3, 0.6)
        
        # LLM confidence (variable based on LLM's own confidence assessment)
        if llm_caps and llm_confidence > 0:
            confidence_details['llm'] = min(len(llm_caps) * llm_confidence * 0.2, 0.4)
        
        # Complexity bonus for sophisticated procedures
        if complexity_score > 3.0:
            confidence_details['complexity_bonus'] = min((complexity_score - 3.0) * 0.1, 0.2)
        
        # Calculate total with diminishing returns
        base_confidence = confidence_details['direct'] + confidence_details['code'] + confidence_details['llm']
        bonuses = confidence_details['complexity_bonus'] + confidence_details['specificity_bonus']
        
        confidence_details['total'] = min(base_confidence + bonuses, 1.0)
        
        return confidence_details
    
    def _calculate_validation_score(self, proc_name: str, capabilities: List[str], 
                                   keyword_matches: Dict, complexity_score: float) -> float:
        """Calculate a validation quality score for the mapping."""
        score = 0.0
        
        # Factor 1: Keyword match quality
        primary_matches = len(keyword_matches.get('primary', []))
        related_matches = len(keyword_matches.get('related', []))
        
        score += primary_matches * 0.3
        score += related_matches * 0.1
        
        # Factor 2: Capability-name alignment
        proc_words = set(proc_name.lower().replace('_', ' ').split())
        cap_alignment = 0
        
        for cap in capabilities:
            cap_words = set(cap.lower().replace('_', ' ').split())
            alignment = len(proc_words.intersection(cap_words))
            cap_alignment += alignment
        
        score += cap_alignment * 0.2
        
        # Factor 3: Complexity appropriateness (right number of capabilities for complexity)
        expected_caps = max(1, int(complexity_score / 2))
        actual_caps = len(capabilities)
        
        if actual_caps <= expected_caps:
            score += 0.3  # Bonus for appropriate capability count
        else:
            score -= (actual_caps - expected_caps) * 0.1  # Penalty for over-assignment
        
        return min(score, 1.0)
    
    def _get_enhanced_keyword_matches(self, searchable_text: str) -> Dict[str, List[str]]:
        """Get detailed keyword matches with importance weighting for reporting."""
        matches = {'primary': [], 'related': [], 'fuzzy': [], 'important': []}
        
        for entry in self.keywords_data:
            primary_kw = [kw.strip().lower() for kw in entry.get('keywords', '').split(',') if kw.strip()]
            related_kw = [kw.strip().lower() for kw in entry.get('related_keywords', '').split(',') if kw.strip()]
            
            primary_found = [kw for kw in primary_kw if kw in searchable_text.lower()]
            related_found = [kw for kw in related_kw if kw in searchable_text.lower()]
            
            matches['primary'].extend(primary_found)
            matches['related'].extend(related_found)
            
            # Track high-importance matches for additional insight
            for kw in primary_found + related_found:
                importance = self.keyword_importance_scores.get(kw, 1.0)
                if importance > 2.0:  # High importance threshold
                    matches['important'].append(kw)
        
        # Remove duplicates and add fuzzy matches
        matches['primary'] = list(set(matches['primary']))
        matches['related'] = list(set(matches['related']))
        matches['important'] = list(set(matches['important']))
        matches['fuzzy'] = self._find_enhanced_fuzzy_matches(searchable_text)
        
        return matches
    
    def _find_enhanced_fuzzy_matches(self, text: str) -> List[str]:
        """Enhanced fuzzy matching with better filtering and validation."""
        words = re.findall(r'\w{3,}', text.lower())  # Only words with 3+ characters
        fuzzy_matches = []
        
        # Get all unique keywords for fuzzy matching
        all_keywords = set()
        for entry in self.keywords_data:
            keywords = entry.get('keywords', '') + ',' + entry.get('related_keywords', '')
            for kw in keywords.split(','):
                kw = kw.strip().lower()
                if len(kw) > 4:  # Only fuzzy match longer keywords
                    all_keywords.add(kw)
        
        # Find fuzzy matches with additional validation
        for keyword in all_keywords:
            close_matches = difflib.get_close_matches(keyword, words, n=1, cutoff=self.fuzzy_threshold)
            if close_matches:
                matched_word = close_matches[0]
                # Additional validation: check length similarity
                length_ratio = min(len(keyword), len(matched_word)) / max(len(keyword), len(matched_word))
                if length_ratio > 0.7:  # Length similarity check
                    fuzzy_matches.append(keyword)
        
        return fuzzy_matches
    
    def _find_enhanced_domain_match(self, proc_name: str, proc_body: str, 
                                  capabilities: List[str]) -> str:
        """Enhanced domain matching with multiple factors and specificity weighting."""
        
        domain_scores = {}
        proc_text = f"{proc_name} {proc_body}".lower()
        
        for entry in self.keywords_data:
            domain = entry.get('metadata', 'unknown')
            domain_capabilities = entry.get('business_capability', [])
            
            if not domain_capabilities:
                continue
            
            score = 0.0
            
            # Factor 1: Capability overlap with specificity weighting
            capability_overlap = len(set(capabilities) & set(domain_capabilities))
            if capability_overlap > 0:
                # Bonus for having the most specific capabilities
                specificity_bonus = sum(self.capability_extractor.get_capability_specificity_score(cap) 
                                      for cap in capabilities if cap in domain_capabilities)
                score += capability_overlap * 3.0 + specificity_bonus
            
            # Factor 2: Direct domain-procedure name alignment  
            domain_name_parts = domain.lower().replace('-', ' ').split()
            proc_name_words = proc_name.lower().replace('_', ' ').split()
            
            for domain_part in domain_name_parts:
                if len(domain_part) > 3:  # Avoid short words like 'mt'
                    for proc_word in proc_name_words:
                        if domain_part == proc_word:
                            score += 4.0  # High bonus for exact domain-name match
                        elif domain_part in proc_word or proc_word in domain_part:
                            score += 2.0
            
            # Factor 3: Keyword matches with importance weighting
            primary_kw = [kw.strip().lower() for kw in entry.get('keywords', '').split(',') if kw.strip()]
            related_kw = [kw.strip().lower() for kw in entry.get('related_keywords', '').split(',') if kw.strip()]
            
            for kw in primary_kw:
                if kw in proc_text:
                    importance = self.keyword_importance_scores.get(kw, 1.0)
                    score += importance * 1.5
            
            for kw in related_kw:
                if kw in proc_text:
                    importance = self.keyword_importance_scores.get(kw, 1.0)
                    score += importance * 0.5
            
            if score > 0:
                domain_scores[domain] = score
        
        # Return the highest-scoring domain
        if domain_scores:
            best_domain = max(domain_scores.keys(), key=lambda d: domain_scores[d])
            return best_domain
        
        return 'unknown'
    
    def _detect_enhanced_procedural_patterns(self, proc_name: str, proc_body: str) -> List[str]:
        """Detect procedural patterns using keywords.json data dynamically."""
        patterns = []
        name_lower = proc_name.lower()
        body_lower = proc_body.lower()
        
        # Build pattern definitions dynamically from keywords.json
        pattern_keywords = {}
        for entry in self.keywords_data:
            domain = entry.get('metadata', 'unknown')
            keywords = entry.get('keywords', '') + ',' + entry.get('related_keywords', '')
            
            # Extract pattern-relevant keywords from the taxonomy
            for keyword in keywords.split(','):
                keyword = keyword.strip().lower()
                if any(pattern_word in keyword for pattern_word in 
                      ['validate', 'check', 'verify', 'screen', 'process', 'handle', 
                       'authorization', 'compliance', 'fraud', 'liquidity']):
                    if domain not in pattern_keywords:
                        pattern_keywords[domain] = []
                    pattern_keywords[domain].append(keyword)
        
        # Detect patterns based on keyword presence
        for domain, keywords in pattern_keywords.items():
            for keyword in keywords:
                if keyword in name_lower or keyword in body_lower:
                    pattern_name = f"{domain.replace('-', '_')}_logic"
                    if pattern_name not in patterns:
                        patterns.append(pattern_name)
        
        return patterns
    
    def _build_reasoning_explanation(self, direct_caps: List[str], code_caps: List[str], 
                                   llm_caps: List[str]) -> str:
        """Build human-readable explanation for capability mapping decisions."""
        parts = []
        
        if direct_caps:
            parts.append(f"Direct name mapping: {len(direct_caps)} capabilities")
        
        if code_caps:
            parts.append(f"Code analysis: {len(code_caps)} capabilities")
        
        if llm_caps:
            parts.append(f"LLM semantic analysis: {len(llm_caps)} capabilities")
        
        if not parts:
            return "No significant capability matches found"
        
        return "; ".join(parts)

class LLMProvider:
    """
    Handles communication with OpenAI's GPT-4 API for semantic analysis.
    
    This class provides:
    - API communication with proper error handling
    - Response caching to avoid duplicate API calls
    - Token optimization and request management
    - JSON parsing and validation
    
    The LLM component is optional - the system works without it using
    only algorithmic approaches for capability detection.
    """
    
    def __init__(self, model: str = "gpt-4", api_key: str = None):
        """Initialize the LLM provider."""
        self.model = model
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.base_url = "https://api.openai.com/v1"
        
        # Caching to avoid duplicate API calls for the same analysis
        self.cache = {}
        self.request_count = 0  # Track usage for monitoring
    
    def is_available(self) -> bool:
        """Check if the LLM provider is properly configured and available."""
        return bool(self.api_key)
    
    def call_chat_completion(self, prompt: str, system_prompt: str = None) -> Dict:
        """Make an OpenAI API call with caching and error handling."""
        if not self.is_available():
            return {"error": "API not available - check OPENAI_API_KEY environment variable"}
        
        # CACHING: Check if we've already analyzed this exact request
        cache_key = hash(f"{prompt}_{system_prompt}")
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            # Prepare the API request
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Build the message array
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            # API request payload
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": 500,        # Limit for cost control
                "temperature": 0.1        # Low temperature for consistent results
            }
            
            # Make the API call
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60  # 60 second timeout
            )
            
            self.request_count += 1  # Track usage
            
            # Process the response
            if response.status_code == 200:
                result = response.json()
                if 'choices' in result and result['choices']:
                    content = result['choices'][0]['message']['content']
                    try:
                        # Parse the JSON response
                        parsed_result = json.loads(content)
                        self.cache[cache_key] = parsed_result  # Cache successful results
                        return parsed_result
                    except json.JSONDecodeError:
                        # Return error with raw content for debugging
                        return {"error": "Invalid JSON response", "raw_content": content}
            
            return {"error": f"API error: {response.status_code}"}
            
        except Exception as e:
            return {"error": f"Request failed: {str(e)}"}

def main():
    """
    Main execution function - orchestrates the entire analysis process.
    
    This function:
    1. Parses command line arguments
    2. Loads and validates the keywords.json taxonomy
    3. Initializes all system components
    4. Processes TAL source files
    5. Generates comprehensive output with statistics
    6. Provides monitoring and debugging information
    
    The system is designed to handle both individual files and large
    repository-scale analysis with appropriate performance monitoring.
    """
    # COMMAND LINE INTERFACE: Define all available options
    parser = argparse.ArgumentParser(
        description="Dynamic TAL Procedure to Business Capability Mapper",
        epilog="""
        This system analyzes TAL payments/wire processing procedures and maps them to business capabilities
        defined in a keywords.json taxonomy. It uses multiple analysis techniques including
        keyword matching, semantic analysis, and optional AI-powered understanding.

        Example usage:
        python <this>.py /path/to/dir -k keywords.json --use-llm
        python <this>.py single_file.tal -k keywords.json --min-confidence 0.7
                """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("input_path", help="Input TAL file or directory to analyze")
    parser.add_argument("-k", "--keywords", required=True, 
                       help="Keywords JSON file containing business capability taxonomy")
    parser.add_argument("-o", "--output", default="procedure_capability_mapping.json", 
                       help="Output file for mapping results (default: procedure_capability_mapping.json)")
    parser.add_argument("--min-confidence", type=float, default=0.2, 
                       help="Minimum confidence threshold for capability inclusion (default: 0.2)")
    parser.add_argument("--use-llm", action="store_true", 
                       help="Enable LLM semantic analysis using OpenAI GPT-4 (requires OPENAI_API_KEY)")
    parser.add_argument("--llm-model", default="gpt-4", 
                       help="OpenAI model to use for semantic analysis (default: gpt-4)")
    parser.add_argument("--fuzzy-threshold", type=float, default=0.85, 
                       help="Fuzzy string matching threshold (default: 0.85)")
    
    args = parser.parse_args()
    
    # TAXONOMY LOADING: Load and validate the keywords.json file
    try:
        with open(args.keywords, 'r', encoding='utf-8') as f:
            keywords_data = json.load(f)
        logger.info(f"Loaded taxonomy with {len(keywords_data)} entries from {args.keywords}")
    except Exception as e:
        print(f"Error loading keywords file: {e}")
        return 1
    
    # COMPONENT INITIALIZATION: Set up all analysis components
    procedure_parser = EnhancedProcedureParser()
    capability_matcher = DynamicCapabilityMatcher(keywords_data, args.fuzzy_threshold)
    
    # LLM SETUP: Initialize optional LLM component
    llm_provider = None
    if args.use_llm:
        llm_provider = LLMProvider(model=args.llm_model)
        if llm_provider.is_available():
            capability_matcher.set_llm_provider(llm_provider)
            print(f"LLM provider initialized: {args.llm_model}")
        else:
            print("Warning: LLM requested but API key not found. Set OPENAI_API_KEY environment variable.")
            print("Continuing with keyword-based analysis only.")
    
    # FILE DISCOVERY: Find all TAL files to process
    input_path = Path(args.input_path)
    files_to_process = []
    
    if input_path.is_file():
        files_to_process = [input_path]
    elif input_path.is_dir():
        # Search for common TAL file extensions
        for pattern in ['*.tal', '*.TAL', '*.ast', '*.AST']:
            files_to_process.extend(input_path.rglob(pattern))
        files_to_process = list(set(files_to_process))  # Remove duplicates
    else:
        print(f"Error: Input path {input_path} not found")
        return 1
    
    print(f"Processing {len(files_to_process)} files...")
    
    # PROCEDURE EXTRACTION: Parse all TAL files to extract procedures
    all_procedures = []
    global_procedure_names = set()  # Global deduplication across all files
    
    for file_path in files_to_process:
        try:
            # Read file with error handling for encoding issues
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            procedures = procedure_parser.extract_procedures(content, str(file_path))
            
            # Global deduplication to handle procedures appearing in multiple files
            unique_procedures = []
            for proc in procedures:
                if proc['name'] not in global_procedure_names:
                    unique_procedures.append(proc)
                    global_procedure_names.add(proc['name'])
                else:
                    print(f"  Skipping duplicate procedure {proc['name']} from {file_path.name}")
            
            all_procedures.extend(unique_procedures)
            print(f"Extracted {len(unique_procedures)} unique procedures from {file_path.name}")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    print(f"Total unique procedures found: {len(all_procedures)}")
    
    # CAPABILITY MAPPING: Analyze each procedure for business capabilities
    capability_mappings = []
    for i, procedure in enumerate(all_procedures, 1):
        print(f"Analyzing procedure {i}/{len(all_procedures)}: {procedure['name']}")
        
        # Perform the multi-stage capability analysis
        capability_match = capability_matcher.match_procedure_to_capabilities(
            procedure, args.min_confidence
        )
        
        # Process results and provide user feedback
        if capability_match.business_capabilities:
            print(f"  → Mapped to {len(capability_match.business_capabilities)} capabilities")
            print(f"  → Confidence: {capability_match.confidence_score:.3f}")
            print(f"  → Domain: {capability_match.domain_context}")
            
            # Show top capabilities for user feedback
            top_caps = capability_match.business_capabilities[:4]
            if top_caps:
                print(f"  → Top capabilities: {', '.join(top_caps)}")
                if len(capability_match.business_capabilities) > 4:
                    print(f"    ... and {len(capability_match.business_capabilities)-4} more")
            
            # Extract LLM analysis details 
            llm_analysis = capability_match.llm_analysis
            
            # Build comprehensive mapping record
            capability_mappings.append({
                "procedure_name": capability_match.procedure_name,
                "source_file": capability_match.source_file,
                "business_capabilities": capability_match.business_capabilities,
                "confidence_score": capability_match.confidence_score,
                "domain_context": capability_match.domain_context,
                "keyword_matches": {
                    "primary": capability_match.primary_keyword_matches,
                    "related": capability_match.related_keyword_matches,
                    "fuzzy": capability_match.fuzzy_matches,
                    "semantic": capability_match.semantic_matches
                },
                "procedural_patterns": capability_match.procedural_patterns,
                "reasoning": capability_match.reasoning,
                "validation_score": capability_match.validation_score,
                "evidence_strength": capability_match.evidence_strength,
                "llm_analysis": llm_analysis
            })
        else:
            print(f"  → No capabilities mapped (confidence too low)")
    
    # STATISTICS GENERATION: Calculate comprehensive coverage metrics
    total_capabilities = capability_matcher.capability_extractor.get_all_capabilities()
    mapped_capabilities = set()
    for mapping in capability_mappings:
        mapped_capabilities.update(mapping["business_capabilities"])
    
    # BUILD OUTPUT STRUCTURE: Comprehensive results with metadata
    output_data = {
        "metadata": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "input_path": str(args.input_path),
            "keywords_file": args.keywords,
            "total_files_processed": len(files_to_process),
            "total_procedures_found": len(all_procedures),
            "procedures_mapped": len(capability_mappings),
            "min_confidence_threshold": args.min_confidence,
            "llm_enabled": args.use_llm and (llm_provider and llm_provider.is_available()),
            "total_business_capabilities": len(total_capabilities),
            "mapped_business_capabilities": len(mapped_capabilities),
            "coverage_percentage": round((len(mapped_capabilities) / len(total_capabilities)) * 100, 1) if total_capabilities else 0,
            "dynamic_mapping": True,
            "hard_coded_mappings": False,
            "system_version": "1.0 - Enhanced Dynamic Capability Mapper"
        },
        "business_capability_summary": {
            "all_capabilities": sorted(list(total_capabilities)),
            "mapped_capabilities": sorted(list(mapped_capabilities)),
            "unmapped_capabilities": sorted(list(total_capabilities - mapped_capabilities))
        },
        "procedure_mappings": sorted(capability_mappings, key=lambda x: x["confidence_score"], reverse=True)
    }
    
    # Add LLM usage statistics if available
    if llm_provider:
        output_data["llm_usage"] = {
            "requests_made": llm_provider.request_count,
            "cache_size": len(llm_provider.cache),
            "model_used": llm_provider.model
        }
    
    # RESULTS EXPORT: Write comprehensive output file
    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\nMapping results saved to: {args.output}")
    except Exception as e:
        print(f"Error saving output file: {e}")
        return 1
    
    # FINAL REPORTING: User-friendly summary
    print(f"\n{'='*60}")
    print("DYNAMIC PROCEDURE TO BUSINESS CAPABILITY MAPPING COMPLETE")
    print(f"{'='*60}")
    print(f"Files processed: {len(files_to_process)}")
    print(f"Procedures found: {len(all_procedures)}")
    print(f"Procedures mapped: {len(capability_mappings)}")
    print(f"Business capabilities covered: {len(mapped_capabilities)}/{len(total_capabilities)} ({output_data['metadata']['coverage_percentage']}%)")
    print(f"Dynamic mapping (no hard-coded rules): ✓")
    
    # Show top results for immediate feedback
    if capability_mappings:
        print(f"\nTop procedure mappings:")
        for i, mapping in enumerate(capability_mappings[:5], 1):
            caps_str = ', '.join(mapping['business_capabilities'][:3])
            if len(mapping['business_capabilities']) > 3:
                caps_str += f" (+{len(mapping['business_capabilities'])-3} more)"
            print(f"{i}. {mapping['procedure_name']} → {caps_str} (conf: {mapping['confidence_score']:.3f})")
    
    # Show unmapped capabilities for taxonomy improvement guidance
    if output_data['business_capability_summary']['unmapped_capabilities']:
        print(f"\nUnmapped capabilities: {len(output_data['business_capability_summary']['unmapped_capabilities'])}")
        for cap in output_data['business_capability_summary']['unmapped_capabilities'][:5]:
            print(f"  - {cap}")
        if len(output_data['business_capability_summary']['unmapped_capabilities']) > 5:
            print(f"  ... and {len(output_data['business_capability_summary']['unmapped_capabilities'])-5} more")
    
    return 0

if __name__ == "__main__":
    exit(main())

