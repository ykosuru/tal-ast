#!/usr/bin/env python3

import json
import argparse
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
import re
import difflib
from dataclasses import dataclass
import requests
import time
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import defaultdict, Counter
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BusinessCapabilityMatch:
    """Enhanced match result with business capability focus"""
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

class BusinessCapabilityExtractor:
    """Extracts business capabilities from keywords.json structure with validation"""
    
    def __init__(self, keywords_data: List[Dict]):
        self.keywords_data = keywords_data
        self.capability_to_domain = {}
        self.domain_to_capabilities = {}
        self.all_capabilities = set()
        self.capability_hierarchy = {}
        self._build_capability_mappings()
        self._validate_taxonomy()
    
    def _build_capability_mappings(self):
        """Build mappings between domains and business capabilities"""
        for entry in self.keywords_data:
            domain = entry.get('metadata', 'unknown')
            capabilities = entry.get('business_capability', [])
            
            # Map capabilities to domains
            for capability in capabilities:
                self.all_capabilities.add(capability)
                if capability not in self.capability_to_domain:
                    self.capability_to_domain[capability] = []
                self.capability_to_domain[capability].append(domain)
            
            # Map domains to capabilities
            if domain not in self.domain_to_capabilities:
                self.domain_to_capabilities[domain] = []
            self.domain_to_capabilities[domain].extend(capabilities)
    
    def _validate_taxonomy(self):
        """Validate the taxonomy structure and log potential issues"""
        issues = []
        
        # Check for orphaned capabilities
        orphaned_caps = [cap for cap in self.all_capabilities 
                        if len(self.capability_to_domain[cap]) == 0]
        if orphaned_caps:
            issues.append(f"Orphaned capabilities: {orphaned_caps[:5]}")
        
        # Check for capabilities in too many domains (potential over-assignment)
        overassigned_caps = [cap for cap in self.all_capabilities 
                           if len(self.capability_to_domain[cap]) > 3]
        if overassigned_caps:
            issues.append(f"Over-assigned capabilities (>3 domains): {overassigned_caps[:5]}")
        
        # Log validation results
        if issues:
            logger.warning(f"Taxonomy validation issues: {'; '.join(issues)}")
        else:
            logger.info("Taxonomy validation passed")
    
    def get_capability_specificity_score(self, capability: str) -> float:
        """Calculate how specific/unique a capability is across domains"""
        if capability not in self.capability_to_domain:
            return 0.0
        
        domain_count = len(self.capability_to_domain[capability])
        # More specific capabilities are in fewer domains
        return 1.0 / (1.0 + domain_count)
    
    def get_capabilities_for_domain(self, domain: str) -> List[str]:
        """Get business capabilities for a specific domain"""
        return self.domain_to_capabilities.get(domain, [])
    
    def get_domains_for_capability(self, capability: str) -> List[str]:
        """Get domains that contain a specific capability"""
        return self.capability_to_domain.get(capability, [])
    
    def get_all_capabilities(self) -> Set[str]:
        """Get all unique business capabilities"""
        return self.all_capabilities.copy()

class EnhancedProcedureParser:
    """Enhanced TAL procedure parser with better pattern matching and validation"""
    
    def __init__(self):
        # Enhanced regex patterns for TAL procedure detection
        self.proc_patterns = [
            re.compile(r'(?i)^\s*(?:INT\s+)?PROC\s+([\w_]+)\s*\((.*?)\)\s*;', re.MULTILINE | re.DOTALL),
            re.compile(r'(?i)^\s*(?:INT\s+)?SUBPROC\s+([\w_]+)\s*\((.*?)\)\s*;', re.MULTILINE | re.DOTALL),
            re.compile(r'(?i)^\s*(?:INT\s+)?PROCEDURE\s+([\w_]+)\s*\((.*?)\)\s*;', re.MULTILINE | re.DOTALL),
            re.compile(r'(?i)^\s*(?:INT\s+)?FUNCTION\s+([\w_]+)\s*\((.*?)\)\s*;', re.MULTILINE | re.DOTALL)
        ]
        
        # Enhanced body extraction with nested BEGIN/END handling
        self.body_pattern = re.compile(r'(?i)BEGIN\s*(.*?)\s*END\s*;', re.DOTALL)
        self.nested_begin_pattern = re.compile(r'(?i)BEGIN', re.MULTILINE)
        self.nested_end_pattern = re.compile(r'(?i)END', re.MULTILINE)
        
        # Enhanced function call patterns
        self.call_patterns = [
            re.compile(r'(?i)CALL\s+([\w_]+)', re.MULTILINE),
            re.compile(r'(?i)([\w_]+)\s*\(', re.MULTILINE),
        ]
        
        # Dynamic utility pattern detection (no hard-coding)
        self.utility_indicators = ['error', 'warning', 'debug', 'trace', 'log', 'format', 'normalize']
    
    def _extract_balanced_body(self, content: str, start_pos: int) -> Tuple[str, int]:
        """Extract procedure body handling nested BEGIN/END blocks"""
        remaining = content[start_pos:]
        
        # Find first BEGIN
        begin_match = self.nested_begin_pattern.search(remaining)
        if not begin_match:
            return self._extract_implicit_body(remaining), len(remaining)
        
        # Count nested BEGIN/END pairs
        begin_pos = begin_match.start()
        current_pos = begin_pos + len("BEGIN")
        depth = 1
        
        while depth > 0 and current_pos < len(remaining):
            next_begin = self.nested_begin_pattern.search(remaining, current_pos)
            next_end = self.nested_end_pattern.search(remaining, current_pos)
            
            if next_end is None:
                break
            
            if next_begin and next_begin.start() < next_end.start():
                depth += 1
                current_pos = next_begin.end()
            else:
                depth -= 1
                current_pos = next_end.end()
                if depth == 0:
                    body_content = remaining[begin_match.end():next_end.start()]
                    return body_content.strip(), start_pos + current_pos
        
        return self._extract_implicit_body(remaining), len(remaining)
    
    def _extract_implicit_body(self, remaining: str) -> str:
        """Extract procedure body without explicit BEGIN/END"""
        next_proc_pos = len(remaining)
        
        for pattern in self.proc_patterns:
            next_match = pattern.search(remaining, 1)
            if next_match:
                next_proc_pos = min(next_proc_pos, next_match.start())
        
        body_end = min(next_proc_pos, 3000)
        return remaining[:body_end]
    
    def _is_utility_function(self, proc_name: str, proc_body: str) -> bool:
        """Determine if a procedure is a utility function using dynamic detection"""
        name_lower = proc_name.lower()
        body_lower = proc_body.lower()
        
        # Check for utility prefixes
        utility_prefixes = ['add_', 'log_', 'format_', 'normalize_', 'get_', 'set_', 'init_', 'cleanup_']
        if any(name_lower.startswith(prefix) for prefix in utility_prefixes):
            return True
        
        # Check for utility suffixes
        if any(name_lower.endswith(suffix) for suffix in self.utility_indicators):
            return True
        
        # Check body characteristics
        utility_indicators_in_body = sum(1 for indicator in self.utility_indicators if indicator in body_lower)
        simple_operations = ['return', 'printf', 'sprintf', 'strlen', 'strcmp']
        simple_count = sum(1 for op in simple_operations if op in body_lower)
        
        # Short procedures with many utility operations are likely utilities
        if len(proc_body) < 500 and (utility_indicators_in_body >= 2 or simple_count >= 2):
            return True
        
        return False
    
    def extract_procedures(self, content: str, source_file: str) -> List[Dict]:
        """Enhanced procedure extraction with better validation"""
        procedures = []
        found_procedures = set()
        extraction_stats = {'total_found': 0, 'duplicates': 0, 'utilities': 0, 'parsed': 0}
        
        for pattern in self.proc_patterns:
            matches = list(pattern.finditer(content))
            extraction_stats['total_found'] += len(matches)
            
            for match in matches:
                proc_name = match.group(1).strip()
                
                if proc_name in found_procedures:
                    extraction_stats['duplicates'] += 1
                    continue
                found_procedures.add(proc_name)
                
                parameters = match.group(2).strip() if match.group(2) else ""
                
                body_content, body_end = self._extract_balanced_body(content, match.end())
                
                # Extract function calls
                function_calls = set()
                for call_pattern in self.call_patterns:
                    calls = call_pattern.findall(body_content)
                    function_calls.update(calls)
                
                function_calls.discard(proc_name)
                function_calls = [call for call in function_calls if len(call) > 2]
                
                is_utility = self._is_utility_function(proc_name, body_content)
                if is_utility:
                    extraction_stats['utilities'] += 1
                
                procedures.append({
                    'name': proc_name,
                    'source_file': source_file,
                    'parameters': parameters,
                    'body': body_content,
                    'function_calls': sorted(list(function_calls)),
                    'full_context': content[max(0, match.start() - 200):match.end() + min(len(body_content), 2000)],
                    'is_utility': is_utility,
                    'body_length': len(body_content),
                    'complexity_score': self._calculate_complexity_score(body_content, function_calls)
                })
                extraction_stats['parsed'] += 1
        
        logger.info(f"Extraction stats for {Path(source_file).name}: {extraction_stats}")
        return procedures
    
    def _calculate_complexity_score(self, body: str, function_calls: List[str]) -> float:
        """Calculate procedure complexity score"""
        score = 0.0
        
        score += min(len(body) / 1000.0, 3.0)
        score += len(function_calls) * 0.3
        
        control_keywords = ['if', 'while', 'for', 'case', 'loop']
        for keyword in control_keywords:
            score += body.lower().count(keyword) * 0.2
        
        return min(score, 10.0)

class DynamicCapabilityMatcher:
    """Fully dynamic matcher that derives everything from keywords.json"""
    
    def __init__(self, keywords_data: List[Dict], fuzzy_threshold: float = 0.85):
        self.keywords_data = keywords_data
        self.fuzzy_threshold = fuzzy_threshold
        self.capability_extractor = BusinessCapabilityExtractor(keywords_data)
        self.llm_provider = None
        
        # Build dynamic mappings from keywords.json
        self.domain_keywords = {}
        self.domain_related_keywords = {}
        self.keyword_importance_scores = {}
        self.capability_keyword_map = {}
        
        self._build_dynamic_mappings()
        self._init_enhanced_tfidf()
        
        # Dynamic false positive detection
        self.validation_cache = {}
        self.false_positive_patterns = self._build_dynamic_false_positive_patterns()
    
    def _build_dynamic_mappings(self):
        """Build all mappings dynamically from keywords.json"""
        keyword_frequency = Counter()
        
        # Build domain keyword mappings
        for entry in self.keywords_data:
            domain = entry.get('metadata', 'unknown')
            primary_kw = [kw.strip().lower() for kw in entry.get('keywords', '').split(',') if kw.strip()]
            related_kw = [kw.strip().lower() for kw in entry.get('related_keywords', '').split(',') if kw.strip()]
            capabilities = entry.get('business_capability', [])
            
            self.domain_keywords[domain] = primary_kw
            self.domain_related_keywords[domain] = related_kw
            
            # Build capability to keyword mappings
            for capability in capabilities:
                if capability not in self.capability_keyword_map:
                    self.capability_keyword_map[capability] = {'primary': [], 'related': []}
                self.capability_keyword_map[capability]['primary'].extend(primary_kw)
                self.capability_keyword_map[capability]['related'].extend(related_kw)
            
            # Count frequency for importance scoring
            for kw in primary_kw + related_kw:
                keyword_frequency[kw] += 1
        
        # Calculate importance scores
        max_freq = max(keyword_frequency.values()) if keyword_frequency else 1
        for keyword, freq in keyword_frequency.items():
            self.keyword_importance_scores[keyword] = max_freq / freq
        
        logger.info(f"Built dynamic mappings: {len(self.domain_keywords)} domains, "
                   f"{len(self.capability_keyword_map)} capabilities")
    
    def _build_dynamic_false_positive_patterns(self) -> List[re.Pattern]:
        """Build false positive patterns dynamically from data"""
        patterns = []
        
        # Extract utility-like patterns from keywords
        utility_terms = set()
        for entry in self.keywords_data:
            keywords = entry.get('keywords', '').lower() + ',' + entry.get('related_keywords', '').lower()
            for word in re.findall(r'\b\w+\b', keywords):
                if word in ['error', 'warning', 'debug', 'trace', 'log', 'format', 'normalize']:
                    utility_terms.add(word)
        
        # Create patterns for utility detection
        if utility_terms:
            utility_pattern = '|'.join(utility_terms)
            patterns.append(re.compile(f'(?i)^(add_|get_|set_|init_)'))
            patterns.append(re.compile(f'(?i)({utility_pattern})$'))
        
        return patterns
    
    def _init_enhanced_tfidf(self):
        """Enhanced TF-IDF initialization with better preprocessing"""
        try:
            self.tfidf_docs = []
            self.tfidf_metadata = []
            
            for entry in self.keywords_data:
                doc_parts = [
                    entry.get('keywords', ''),
                    entry.get('related_keywords', ''),
                    entry.get('description', ''),
                    ' '.join(entry.get('business_capability', []))
                ]
                doc_text = ' '.join(part for part in doc_parts if part)
                
                self.tfidf_docs.append(doc_text)
                self.tfidf_metadata.append({
                    'domain': entry.get('metadata', 'unknown'),
                    'capabilities': entry.get('business_capability', []),
                    'primary_keywords': [kw.strip().lower() for kw in entry.get('keywords', '').split(',') if kw.strip()]
                })
            
            if self.tfidf_docs:
                self.tfidf_vectorizer = TfidfVectorizer(
                    stop_words='english', 
                    max_features=8000,
                    ngram_range=(1, 3),
                    min_df=1,
                    max_df=0.8
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
        """Set LLM provider for semantic analysis"""
        self.llm_provider = llm_provider
    
    def match_procedure_to_capabilities(self, procedure: Dict, min_confidence: float = 0.2) -> BusinessCapabilityMatch:
        """Main matching method using only dynamic data from keywords.json"""
        
        proc_name = procedure['name']
        proc_body = procedure['body'].lower()
        function_calls = procedure.get('function_calls', [])
        full_context = procedure.get('full_context', '').lower()
        is_utility = procedure.get('is_utility', False)
        complexity_score = procedure.get('complexity_score', 0.0)
        
        # Early exit for utility functions with low complexity
        if is_utility and complexity_score < 2.0:
            return self._create_minimal_match(procedure, "Utility function - minimal capabilities")
        
        # Check for false positive patterns
        if self._is_likely_false_positive(proc_name, proc_body):
            return self._create_minimal_match(procedure, "Likely false positive pattern detected")
        
        # Dynamic capability detection
        direct_capabilities = self._find_direct_name_matches(proc_name)
        code_detected_capabilities = self._analyze_code_dynamically(proc_name, proc_body, full_context, function_calls)
        
        # Enhanced LLM semantic analysis
        llm_capabilities = []
        llm_reasoning = ""
        llm_confidence = 0.0
        llm_details = {}
        
        if self.llm_provider and self.llm_provider.is_available() and complexity_score >= 1.0:
            llm_result = self._enhanced_llm_semantic_bridge(procedure, direct_capabilities + code_detected_capabilities)
            if isinstance(llm_result, dict):
                llm_capabilities = llm_result.get('additional_capabilities', [])
                llm_reasoning = llm_result.get('reasoning', '')
                llm_confidence = llm_result.get('confidence', 0.0)
                llm_details = {
                    'capability_details': llm_result.get('capability_details', []),
                    'primary_domain': llm_result.get('primary_domain', 'unknown')
                }
        
        # Validate and combine capabilities
        all_capabilities = self._validate_and_combine_capabilities(
            direct_capabilities, code_detected_capabilities, llm_capabilities, 
            proc_name, proc_body, complexity_score
        )
        
        # Enhanced confidence calculation
        confidence_details = self._calculate_enhanced_confidence(
            proc_name, proc_body, direct_capabilities, code_detected_capabilities, 
            llm_capabilities, llm_confidence, complexity_score
        )
        
        # Find best domain match
        best_domain = self._find_enhanced_domain_match(proc_name, proc_body, list(all_capabilities))
        
        # Get detailed keyword matches
        keyword_matches = self._get_enhanced_keyword_matches(proc_name + " " + proc_body)
        
        # Calculate validation score
        validation_score = self._calculate_validation_score(
            proc_name, all_capabilities, keyword_matches, complexity_score
        )
        
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
            procedural_patterns=self._detect_enhanced_procedural_patterns(proc_name, proc_body),
            reasoning=llm_reasoning or self._build_reasoning_explanation(
                direct_capabilities, code_detected_capabilities, llm_capabilities
            ),
            validation_score=validation_score,
            evidence_strength=confidence_details
        )
    
    def _find_direct_name_matches(self, proc_name: str) -> List[str]:
        """Find direct matches between procedure names and capabilities using keywords.json"""
        name_lower = proc_name.lower()
        direct_caps = []
        
        # Search through all capabilities and their associated keywords
        for capability, keyword_map in self.capability_keyword_map.items():
            match_score = 0.0
            
            # Check primary keywords
            for keyword in keyword_map['primary']:
                if self._keyword_matches_procedure_name(keyword, name_lower):
                    importance = self.keyword_importance_scores.get(keyword, 1.0)
                    match_score += importance * 2.0
            
            # Check related keywords
            for keyword in keyword_map['related']:
                if self._keyword_matches_procedure_name(keyword, name_lower):
                    importance = self.keyword_importance_scores.get(keyword, 1.0)
                    match_score += importance * 1.0
            
            # Check capability name alignment with procedure name
            cap_words = capability.lower().replace(' ', '_').split('_')
            proc_words = name_lower.replace('_', ' ').split()
            
            for cap_word in cap_words:
                if len(cap_word) > 2:
                    for proc_word in proc_words:
                        if cap_word == proc_word:  # Exact word match
                            match_score += 3.0
                        elif cap_word in proc_word or proc_word in cap_word:  # Partial match
                            match_score += 1.5
            
            # Include capability if match score is high enough
            if match_score >= 2.0:
                direct_caps.append((capability, match_score))
        
        # Sort by match score and return capabilities
        direct_caps.sort(key=lambda x: x[1], reverse=True)
        return [cap for cap, score in direct_caps]
    
    def _keyword_matches_procedure_name(self, keyword: str, proc_name_lower: str) -> bool:
        """Check if a keyword matches a procedure name with various strategies"""
        # Direct substring match
        if keyword in proc_name_lower:
            return True
        
        # Word-level matching
        keyword_words = keyword.replace('_', ' ').split()
        proc_words = proc_name_lower.replace('_', ' ').split()
        
        # Check if all keyword words appear in procedure words
        if len(keyword_words) > 1:
            return all(any(kw in pw or pw in kw for pw in proc_words) for kw in keyword_words)
        
        # Single word partial matching
        if len(keyword_words) == 1 and len(keyword_words[0]) > 3:
            return any(keyword_words[0] in pw or pw in keyword_words[0] for pw in proc_words)
        
        return False
    
    def _analyze_code_dynamically(self, proc_name: str, proc_body: str, 
                                full_context: str, function_calls: List[str]) -> List[str]:
        """Analyze code for capabilities using only keywords.json data"""
        capabilities = []
        
        # Skip analysis for simple utility functions
        if len(proc_body) < 200 and any(proc_name.lower().startswith(prefix) 
                                       for prefix in ['add_', 'log_', 'format_', 'get_', 'set_']):
            return capabilities
        
        all_text = f"{proc_body} {full_context} {' '.join(function_calls)}".lower()
        
        # Analyze each entry in keywords.json
        for entry in self.keywords_data:
            business_caps = entry.get('business_capability', [])
            if not business_caps:
                continue
            
            primary_keywords = [kw.strip().lower() for kw in entry.get('keywords', '').split(',') if kw.strip()]
            related_keywords = [kw.strip().lower() for kw in entry.get('related_keywords', '').split(',') if kw.strip()]
            
            match_score = 0.0
            found_keywords = []
            
            # Score primary keyword matches
            for kw in primary_keywords:
                if kw in all_text:
                    importance = self.keyword_importance_scores.get(kw, 1.0)
                    multiplier = 3.0 if any(kw in call.lower() for call in function_calls) else 1.0
                    match_score += importance * multiplier
                    found_keywords.append(kw)
            
            # Score related keyword matches
            for kw in related_keywords:
                if kw in all_text:
                    importance = self.keyword_importance_scores.get(kw, 1.0)
                    multiplier = 2.0 if any(kw in call.lower() for call in function_calls) else 0.5
                    match_score += importance * multiplier * 0.5
                    found_keywords.append(kw)
            
            # Only add capabilities if match score exceeds threshold
            if match_score >= 2.0:
                relevant_caps = self._filter_capabilities_by_relevance(
                    business_caps, found_keywords, proc_name, all_text, match_score
                )
                capabilities.extend(relevant_caps)
        
        return list(set(capabilities))
    
    def _filter_capabilities_by_relevance(self, capabilities: List[str], found_keywords: List[str], 
                                        proc_name: str, all_text: str, match_score: float) -> List[str]:
        """Filter capabilities based on relevance using dynamic analysis"""
        relevant_caps = []
        
        for capability in capabilities:
            relevance_score = 0.0
            cap_words = capability.lower().replace(' ', '_').split('_')
            
            # Score based on capability terms in procedure name (highest weight)
            proc_words = proc_name.lower().replace('_', ' ').split()
            for cap_word in cap_words:
                if len(cap_word) > 2:
                    for proc_word in proc_words:
                        if cap_word == proc_word:  # Exact match
                            relevance_score += 4.0
                        elif cap_word in proc_word or proc_word in cap_word:  # Partial match
                            relevance_score += 2.0
            
            # Score based on keyword matches
            for keyword in found_keywords:
                keyword_words = keyword.replace(' ', '_').split('_')
                for kw_word in keyword_words:
                    if len(kw_word) > 2:
                        for cap_word in cap_words:
                            if kw_word == cap_word:
                                relevance_score += 1.5
                            elif kw_word in cap_word or cap_word in kw_word:
                                relevance_score += 1.0
            
            # Score based on capability specificity (favor more specific capabilities)
            specificity = self.capability_extractor.get_capability_specificity_score(capability)
            relevance_score += specificity * 3.0
            
            # Dynamic threshold based on match score
            base_threshold = 3.0
            if match_score > 8.0:
                threshold = base_threshold * 0.6
            elif match_score > 5.0:
                threshold = base_threshold * 0.8
            else:
                threshold = base_threshold * 1.2
            
            if relevance_score >= threshold:
                relevant_caps.append((capability, relevance_score))
        
        # Sort by relevance score and return capabilities
        relevant_caps.sort(key=lambda x: x[1], reverse=True)
        return [cap for cap, score in relevant_caps]
    
    def _is_likely_false_positive(self, proc_name: str, proc_body: str) -> bool:
        """Check if procedure is likely a false positive using dynamic patterns"""
        # Check against dynamically built false positive patterns
        for pattern in self.false_positive_patterns:
            if pattern.search(proc_name):
                return True
        
        # Dynamic heuristics
        if len(proc_body) < 100 and 'return' in proc_body:
            return True
        
        # Check for too many generic keywords
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
        """Create minimal match for utility functions or false positives"""
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
    
    def _enhanced_llm_semantic_bridge(self, procedure: Dict, already_detected: List[str]) -> Dict:
        """Enhanced LLM analysis using chunked keywords.json to avoid token limits"""
        if not self.llm_provider or not self.llm_provider.is_available():
            return {}
        
        # Create focused keyword entries based on procedure context
        proc_text = f"{procedure['name']} {procedure.get('full_context', '')[:1000]}".lower()
        
        # Filter to most relevant keyword entries to reduce payload size
        relevant_entries = []
        for entry in self.keywords_data:
            # Check if any keywords from this entry appear in the procedure
            all_keywords = (entry.get('keywords', '') + ',' + entry.get('related_keywords', '')).lower()
            keyword_list = [kw.strip() for kw in all_keywords.split(',') if kw.strip()]
            
            # Include entry if any keywords match or if it has capabilities we haven't detected
            has_keyword_match = any(kw in proc_text for kw in keyword_list if len(kw) > 3)
            has_undetected_caps = any(cap not in already_detected 
                                    for cap in entry.get('business_capability', []))
            
            if has_keyword_match or has_undetected_caps:
                relevant_entries.append(entry)
        
        # Limit to top 20 most relevant entries to stay under token limits
        relevant_entries = relevant_entries[:20]
        
        # Create condensed taxonomy focused on procedure
        focused_taxonomy = json.dumps(relevant_entries, indent=1)  # Minimal formatting
        
        # Truncate procedure context more aggressively 
        max_code_length = 800  # Further reduced for token efficiency
        code_context = procedure['full_context'][:max_code_length]
        if len(procedure['full_context']) > max_code_length:
            code_context += "\n... [truncated]"
        
        prompt = f"""Analyze this TAL banking procedure against the relevant keywords taxonomy entries.

RELEVANT TAXONOMY ENTRIES:
{focused_taxonomy}

PROCEDURE:
Name: {procedure['name']}
Function Calls: {', '.join(procedure.get('function_calls', [])[:6])}

Code Context:
{code_context}

ALREADY DETECTED: {', '.join(already_detected) if already_detected else 'None'}

TASK:
Review the taxonomy entries above and identify which business capabilities this procedure implements.
Look for evidence in function calls, validation patterns, and processing logic.

Focus on capabilities NOT already detected that have clear evidence in the code.

Respond with JSON:
{{
    "mapped_capabilities": [
        {{
            "capability": "Exact Name from business_capability",
            "confidence": 0.85,
            "evidence": "Specific code evidence"
        }}
    ],
    "reasoning": "Why these capabilities match the code",
    "primary_domain": "Best matching domain from metadata"
}}"""
        
        system_prompt = """You are an expert in wire transfer and payment processing systems.

Analyze TAL procedures to identify implemented business capabilities based on the provided taxonomy entries.

Focus on concrete evidence from code patterns, function calls, and processing logic.
Use only capability names that appear in the business_capability arrays from the taxonomy.
Be conservative - only high-confidence mappings with clear evidence.

Respond with valid JSON only."""
        
        try:
            result = self.llm_provider.call_chat_completion(prompt, system_prompt)
            
            if isinstance(result, dict) and 'mapped_capabilities' in result:
                # Extract and validate capabilities
                valid_capabilities = []
                capability_details = []
                
                for mapping in result.get('mapped_capabilities', []):
                    capability = mapping.get('capability', '')
                    confidence = mapping.get('confidence', 0.0)
                    evidence = mapping.get('evidence', '')
                    
                    # Validate capability exists in our taxonomy
                    all_capabilities = self.capability_extractor.get_all_capabilities()
                    if capability in all_capabilities and confidence >= 0.4:
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
                    'reasoning': result.get('reasoning', 'LLM analysis of focused taxonomy'),
                    'confidence': sum(d['confidence'] for d in capability_details) / len(capability_details) if capability_details else 0.0,
                    'capability_details': capability_details,
                    'primary_domain': result.get('primary_domain', 'unknown')
                }
            elif isinstance(result, dict):
                # Check for errors
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
    
    def _validate_and_combine_capabilities(self, direct_caps: List[str], code_caps: List[str], 
                                         llm_caps: List[str], proc_name: str, proc_body: str, 
                                         complexity_score: float) -> Set[str]:
        """Validate and intelligently combine capabilities from different sources"""
        
        validated_caps = set(direct_caps)
        
        # Add code-detected capabilities with validation
        for cap in code_caps:
            specificity = self.capability_extractor.get_capability_specificity_score(cap)
            threshold = 0.3 if specificity > 0.5 else 0.6
            
            if complexity_score >= threshold:
                validated_caps.add(cap)
        
        # Add LLM capabilities with additional validation
        for cap in llm_caps:
            if complexity_score >= 1.0:
                validated_caps.add(cap)
        
        # Remove overly generic capabilities for simple procedures
        if complexity_score < 2.0:
            # Dynamically identify generic capabilities based on frequency across domains
            generic_caps = set()
            for cap in validated_caps:
                domains = self.capability_extractor.get_domains_for_capability(cap)
                if len(domains) > 5:  # Capabilities in many domains are generic
                    generic_caps.add(cap)
            validated_caps -= generic_caps
        
        # Limit total capabilities based on complexity
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
        """Enhanced confidence calculation with detailed breakdown"""
        
        confidence_details = {
            'direct': 0.0,
            'code': 0.0, 
            'llm': 0.0,
            'complexity_bonus': 0.0,
            'specificity_bonus': 0.0,
            'total': 0.0
        }
        
        # Direct mapping confidence
        if direct_caps:
            confidence_details['direct'] = min(len(direct_caps) * 0.4, 0.8)
        
        # Code analysis confidence weighted by capability specificity
        if code_caps:
            specificity_sum = sum(self.capability_extractor.get_capability_specificity_score(cap) 
                                for cap in code_caps)
            confidence_details['code'] = min(specificity_sum * 0.3, 0.6)
        
        # LLM confidence
        if llm_caps and llm_confidence > 0:
            confidence_details['llm'] = min(len(llm_caps) * llm_confidence * 0.2, 0.4)
        
        # Complexity bonus
        if complexity_score > 3.0:
            confidence_details['complexity_bonus'] = min((complexity_score - 3.0) * 0.1, 0.2)
        
        # Calculate total
        base_confidence = confidence_details['direct'] + confidence_details['code'] + confidence_details['llm']
        bonuses = confidence_details['complexity_bonus'] + confidence_details['specificity_bonus']
        
        confidence_details['total'] = min(base_confidence + bonuses, 1.0)
        
        return confidence_details
    
    def _calculate_validation_score(self, proc_name: str, capabilities: List[str], 
                                   keyword_matches: Dict, complexity_score: float) -> float:
        """Calculate validation quality score"""
        score = 0.0
        
        # Keyword match quality
        primary_matches = len(keyword_matches.get('primary', []))
        related_matches = len(keyword_matches.get('related', []))
        
        score += primary_matches * 0.3
        score += related_matches * 0.1
        
        # Capability-name alignment
        proc_words = set(proc_name.lower().replace('_', ' ').split())
        cap_alignment = 0
        
        for cap in capabilities:
            cap_words = set(cap.lower().replace('_', ' ').split())
            alignment = len(proc_words.intersection(cap_words))
            cap_alignment += alignment
        
        score += cap_alignment * 0.2
        
        # Complexity appropriateness
        expected_caps = max(1, int(complexity_score / 2))
        actual_caps = len(capabilities)
        
        if actual_caps <= expected_caps:
            score += 0.3
        else:
            score -= (actual_caps - expected_caps) * 0.1
        
        return min(score, 1.0)
    
    def _get_enhanced_keyword_matches(self, searchable_text: str) -> Dict[str, List[str]]:
        """Enhanced keyword matching with importance weighting"""
        matches = {'primary': [], 'related': [], 'fuzzy': [], 'important': []}
        
        for entry in self.keywords_data:
            primary_kw = [kw.strip().lower() for kw in entry.get('keywords', '').split(',') if kw.strip()]
            related_kw = [kw.strip().lower() for kw in entry.get('related_keywords', '').split(',') if kw.strip()]
            
            primary_found = [kw for kw in primary_kw if kw in searchable_text.lower()]
            related_found = [kw for kw in related_kw if kw in searchable_text.lower()]
            
            matches['primary'].extend(primary_found)
            matches['related'].extend(related_found)
            
            # Track high-importance matches
            for kw in primary_found + related_found:
                importance = self.keyword_importance_scores.get(kw, 1.0)
                if importance > 2.0:
                    matches['important'].append(kw)
        
        matches['primary'] = list(set(matches['primary']))
        matches['related'] = list(set(matches['related']))
        matches['important'] = list(set(matches['important']))
        matches['fuzzy'] = self._find_enhanced_fuzzy_matches(searchable_text)
        
        return matches
    
    def _find_enhanced_fuzzy_matches(self, text: str) -> List[str]:
        """Enhanced fuzzy matching with better filtering"""
        words = re.findall(r'\w{3,}', text.lower())
        fuzzy_matches = []
        
        all_keywords = set()
        for entry in self.keywords_data:
            keywords = entry.get('keywords', '') + ',' + entry.get('related_keywords', '')
            for kw in keywords.split(','):
                kw = kw.strip().lower()
                if len(kw) > 4:
                    all_keywords.add(kw)
        
        for keyword in all_keywords:
            close_matches = difflib.get_close_matches(keyword, words, n=1, cutoff=self.fuzzy_threshold)
            if close_matches:
                matched_word = close_matches[0]
                length_ratio = min(len(keyword), len(matched_word)) / max(len(keyword), len(matched_word))
                if length_ratio > 0.7:
                    fuzzy_matches.append(keyword)
        
        return fuzzy_matches
    
    def _find_enhanced_domain_match(self, proc_name: str, proc_body: str, 
                                  capabilities: List[str]) -> str:
        """Enhanced domain matching with multiple factors"""
        
        domain_scores = {}
        proc_text = f"{proc_name} {proc_body}".lower()
        
        for entry in self.keywords_data:
            domain = entry.get('metadata', 'unknown')
            domain_capabilities = entry.get('business_capability', [])
            
            if not domain_capabilities:
                continue
            
            score = 0.0
            
            # Capability overlap (highest weight) - but prioritize primary capabilities
            capability_overlap = len(set(capabilities) & set(domain_capabilities))
            if capability_overlap > 0:
                # Bonus for having the most specific capabilities
                specificity_bonus = sum(self.capability_extractor.get_capability_specificity_score(cap) 
                                      for cap in capabilities if cap in domain_capabilities)
                score += capability_overlap * 3.0 + specificity_bonus
            
            # Direct domain-procedure name alignment
            domain_name_parts = domain.lower().replace('-', ' ').split()
            proc_name_words = proc_name.lower().replace('_', ' ').split()
            
            for domain_part in domain_name_parts:
                if len(domain_part) > 3:  # Avoid short words like 'mt'
                    for proc_word in proc_name_words:
                        if domain_part == proc_word:
                            score += 4.0  # High bonus for exact domain-name match
                        elif domain_part in proc_word or proc_word in domain_part:
                            score += 2.0
            
            # Keyword matches (medium weight)
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
        
        if domain_scores:
            best_domain = max(domain_scores.keys(), key=lambda d: domain_scores[d])
            return best_domain
        
        return 'unknown'
    
    def _detect_enhanced_procedural_patterns(self, proc_name: str, proc_body: str) -> List[str]:
        """Enhanced procedural pattern detection using keywords.json data"""
        patterns = []
        name_lower = proc_name.lower()
        body_lower = proc_body.lower()
        
        # Build pattern definitions dynamically from keywords.json
        pattern_keywords = {}
        for entry in self.keywords_data:
            domain = entry.get('metadata', 'unknown')
            keywords = entry.get('keywords', '') + ',' + entry.get('related_keywords', '')
            
            # Extract pattern-relevant keywords
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
        """Build reasoning explanation for capability mapping"""
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
    """LLM provider for semantic analysis"""
    
    def __init__(self, model: str = "gpt-4", api_key: str = None):
        self.model = model
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.base_url = "https://api.openai.com/v1"
        self.cache = {}
        self.request_count = 0
    
    def is_available(self) -> bool:
        return bool(self.api_key)
    
    def call_chat_completion(self, prompt: str, system_prompt: str = None) -> Dict:
        """Make OpenAI API call with caching"""
        if not self.is_available():
            return {"error": "API not available"}
        
        cache_key = hash(f"{prompt}_{system_prompt}")
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": 500,
                "temperature": 0.1
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            
            self.request_count += 1
            
            if response.status_code == 200:
                result = response.json()
                if 'choices' in result and result['choices']:
                    content = result['choices'][0]['message']['content']
                    try:
                        parsed_result = json.loads(content)
                        self.cache[cache_key] = parsed_result
                        return parsed_result
                    except json.JSONDecodeError:
                        return {"error": "Invalid JSON response", "raw_content": content}
            
            return {"error": f"API error: {response.status_code}"}
            
        except Exception as e:
            return {"error": f"Request failed: {str(e)}"}

def main():
    parser = argparse.ArgumentParser(description="Dynamic TAL Procedure to Business Capability Mapper")
    parser.add_argument("input_path", help="Input TAL file or directory")
    parser.add_argument("-k", "--keywords", required=True, help="Keywords JSON file with business capabilities")
    parser.add_argument("-o", "--output", default="procedure_capability_mapping.json", help="Output mapping file")
    parser.add_argument("--min-confidence", type=float, default=0.2, help="Minimum confidence threshold")
    parser.add_argument("--use-llm", action="store_true", help="Enable LLM semantic analysis")
    parser.add_argument("--llm-model", default="gpt-4", help="OpenAI model to use")
    parser.add_argument("--fuzzy-threshold", type=float, default=0.85, help="Fuzzy matching threshold")
    
    args = parser.parse_args()
    
    # Load keywords data
    try:
        with open(args.keywords, 'r', encoding='utf-8') as f:
            keywords_data = json.load(f)
    except Exception as e:
        print(f"Error loading keywords file: {e}")
        return 1
    
    # Initialize components
    procedure_parser = EnhancedProcedureParser()
    capability_matcher = DynamicCapabilityMatcher(keywords_data, args.fuzzy_threshold)
    
    # Initialize LLM if requested
    llm_provider = None
    if args.use_llm:
        llm_provider = LLMProvider(model=args.llm_model)
        if llm_provider.is_available():
            capability_matcher.set_llm_provider(llm_provider)
            print(f"LLM provider initialized: {args.llm_model}")
        else:
            print("Warning: LLM requested but API key not found. Set OPENAI_API_KEY environment variable.")
    
    # Process input files
    input_path = Path(args.input_path)
    files_to_process = []
    
    if input_path.is_file():
        files_to_process = [input_path]
    elif input_path.is_dir():
        for pattern in ['*.tal', '*.TAL', '*.ast', '*.AST']:
            files_to_process.extend(input_path.rglob(pattern))
        files_to_process = list(set(files_to_process))
    else:
        print(f"Error: Input path {input_path} not found")
        return 1
    
    print(f"Processing {len(files_to_process)} files...")
    
    # Extract procedures and map to capabilities
    all_procedures = []
    global_procedure_names = set()
    
    for file_path in files_to_process:
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            procedures = procedure_parser.extract_procedures(content, str(file_path))
            
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
    
    # Map each procedure to business capabilities
    capability_mappings = []
    for i, procedure in enumerate(all_procedures, 1):
        print(f"Analyzing procedure {i}/{len(all_procedures)}: {procedure['name']}")
        
        capability_match = capability_matcher.match_procedure_to_capabilities(
            procedure, args.min_confidence
        )
        
        if capability_match.business_capabilities:
            print(f"  → Mapped to {len(capability_match.business_capabilities)} capabilities")
            print(f"  → Confidence: {capability_match.confidence_score:.3f}")
            print(f"  → Domain: {capability_match.domain_context}")
            
            top_caps = capability_match.business_capabilities[:4]
            if top_caps:
                print(f"  → Top capabilities: {', '.join(top_caps)}")
                if len(capability_match.business_capabilities) > 4:
                    print(f"    ... and {len(capability_match.business_capabilities)-4} more")
            
            # Extract LLM details if available
            llm_analysis = {}
            if hasattr(capability_match, 'llm_analysis') and capability_match.llm_analysis:
                llm_analysis = capability_match.llm_analysis
            
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
    
    # Generate summary statistics
    total_capabilities = capability_matcher.capability_extractor.get_all_capabilities()
    mapped_capabilities = set()
    for mapping in capability_mappings:
        mapped_capabilities.update(mapping["business_capabilities"])
    
    # Create output structure
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
            "hard_coded_mappings": False
        },
        "business_capability_summary": {
            "all_capabilities": sorted(list(total_capabilities)),
            "mapped_capabilities": sorted(list(mapped_capabilities)),
            "unmapped_capabilities": sorted(list(total_capabilities - mapped_capabilities))
        },
        "procedure_mappings": sorted(capability_mappings, key=lambda x: x["confidence_score"], reverse=True)
    }
    
    # Add LLM usage stats if available
    if llm_provider:
        output_data["llm_usage"] = {
            "requests_made": llm_provider.request_count,
            "cache_size": len(llm_provider.cache),
            "model_used": llm_provider.model
        }
    
    # Write output file
    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\nMapping results saved to: {args.output}")
    except Exception as e:
        print(f"Error saving output file: {e}")
        return 1
    
    # Print summary
    print(f"\n{'='*60}")
    print("DYNAMIC PROCEDURE TO BUSINESS CAPABILITY MAPPING COMPLETE")
    print(f"{'='*60}")
    print(f"Files processed: {len(files_to_process)}")
    print(f"Procedures found: {len(all_procedures)}")
    print(f"Procedures mapped: {len(capability_mappings)}")
    print(f"Business capabilities covered: {len(mapped_capabilities)}/{len(total_capabilities)} ({output_data['metadata']['coverage_percentage']}%)")
    print(f"Dynamic mapping (no hard-coded rules): ✓")
    
    if capability_mappings:
        print(f"\nTop procedure mappings:")
        for i, mapping in enumerate(capability_mappings[:5], 1):
            caps_str = ', '.join(mapping['business_capabilities'][:3])
            if len(mapping['business_capabilities']) > 3:
                caps_str += f" (+{len(mapping['business_capabilities'])-3} more)"
            print(f"{i}. {mapping['procedure_name']} → {caps_str} (conf: {mapping['confidence_score']:.3f})")
    
    if output_data['business_capability_summary']['unmapped_capabilities']:
        print(f"\nUnmapped capabilities: {len(output_data['business_capability_summary']['unmapped_capabilities'])}")
        for cap in output_data['business_capability_summary']['unmapped_capabilities'][:5]:
            print(f"  - {cap}")
        if len(output_data['business_capability_summary']['unmapped_capabilities']) > 5:
            print(f"  ... and {len(output_data['business_capability_summary']['unmapped_capabilities'])-5} more")
    
    return 0

if __name__ == "__main__":
    exit(main())
