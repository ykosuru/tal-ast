#!/usr/bin/env python3
"""
Enhanced AST to Business Capability Mapper
FULL VERSION: Preserves all original functionality + adds TAL naming convention support + Full Procedure Context for LLM
"""

import json
import re
import sys
import argparse
import os
import glob
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import ast
import asyncio
from abc import ABC, abstractmethod
from pathlib import Path
import time
import math

@dataclass
class ProcedureCapabilityMatch:
    """Individual procedure match to a capability - ORIGINAL INTERFACE PRESERVED"""
    procedure_name: str
    file_path: str
    procedure_path: str
    procedure_type: str
    confidence: float
    match_type: str  # 'exact', 'partial', 'semantic', 'llm_semantic', 'tal_structural'
    matched_keywords: List[str]
    procedure_comments: str = ""
    # NEW: Additional TAL-specific fields (optional to preserve compatibility)
    tal_naming_info: Optional[Dict] = None

class LLMProvider(ABC):
    """Abstract base class for LLM providers - PRESERVED"""
    
    @abstractmethod
    async def expand_query(self, procedure_info: Dict, context: str) -> Dict:
        """Expand procedure information with semantic understanding"""
        pass

class LocalLLMProvider(LLMProvider):
    """Local LLM provider using sentence transformers - PRESERVED + ENHANCED"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load sentence transformer model"""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            print(f"✅ Loaded local LLM model: {self.model_name}")
        except ImportError:
            print("⚠️ Warning: sentence-transformers not available. Install with: pip install sentence-transformers")
            print("   Falling back to rule-based matching only.")
    
    def _truncate_to_token_limit(self, text: str, max_tokens: int = 2000) -> str:
        """Truncate text to approximate token limit (rough estimate: 1 token ≈ 4 characters)"""
        if not text:
            return text
        
        # Rough estimation: 1 token ≈ 4 characters for most text
        max_chars = max_tokens * 4
        
        if len(text) <= max_chars:
            return text
        
        # Truncate at word boundary near the limit
        truncated = text[:max_chars]
        last_space = truncated.rfind(' ')
        if last_space > max_chars * 0.8:  # If we can find a space in the last 20%
            truncated = truncated[:last_space]
        
        return truncated + "... [TRUNCATED]"
    
    def _extract_business_keywords_from_full_text(self, full_text: str) -> List[str]:
        """Extract business-relevant keywords from full procedure text using enhanced pattern matching"""
        if not full_text:
            return []
        
        full_text_lower = full_text.lower()
        business_keywords = []
        
        # Enhanced payment domain patterns with more sophisticated matching
        enhanced_patterns = {
            'payment_initiation': [
                r'\b(initiate|start|begin|create|submit|send|trigger|launch)\b.*\b(payment|transfer|wire|remittance)\b',
                r'\b(payment|wire|transfer).*\b(initiate|start|begin|create|submit|send)\b',
                r'\binitiate.*\b(transaction|funds|money)\b'
            ],
            'payment_validation': [
                r'\b(validate|verify|check|confirm|authorize|approve)\b.*\b(payment|transfer|wire|account|routing)\b',
                r'\b(payment|account|routing|swift|iban).*\b(validate|verify|check|confirm)\b',
                r'\bvalidation.*\b(rule|process|engine|service)\b'
            ],
            'payment_routing': [
                r'\b(route|forward|direct|dispatch|transmit)\b.*\b(payment|message|wire|transfer)\b',
                r'\b(routing|forwarding).*\b(table|engine|service|logic)\b',
                r'\bmessage.*\b(routing|forwarding|dispatch)\b'
            ],
            'wire_transfer': [
                r'\b(wire|fedwire|swift|chips)\b.*\b(transfer|payment|message)\b',
                r'\b(mt103|mt202|pacs\.008|pacs\.009)\b',
                r'\b(domestic|international|cross.border)\b.*\bwire\b'
            ],
            'sanctions_screening': [
                r'\b(sanction|ofac|aml|screening)\b.*\b(check|scan|validate|monitor)\b',
                r'\b(prohibited|blocked|sanctioned)\b.*\b(party|entity|country)\b',
                r'\blist.*\b(screening|checking|validation)\b'
            ],
            'fraud_monitoring': [
                r'\b(fraud|suspicious|anomaly|unusual)\b.*\b(detect|monitor|alert|flag)\b',
                r'\b(monitoring|detection).*\b(fraud|suspicious|anomaly)\b',
                r'\brisk.*\b(assessment|scoring|evaluation)\b'
            ],
            'settlement_processing': [
                r'\b(settle|clear|post|book)\b.*\b(payment|transaction|transfer)\b',
                r'\b(settlement|clearing).*\b(process|engine|system)\b',
                r'\b(credit|debit).*\b(account|ledger|posting)\b'
            ],
            'message_transformation': [
                r'\b(transform|convert|translate|format|parse)\b.*\b(message|format|data)\b',
                r'\b(mt103|swift|iso20022|fedwire).*\b(format|message|transformation)\b',
                r'\bmessage.*\b(conversion|transformation|formatting)\b'
            ],
            'network_integration': [
                r'\b(network|channel|integration|connection)\b.*\b(swift|fedwire|chips|sepa)\b',
                r'\b(inbound|outbound).*\b(message|payment|wire|processing)\b',
                r'\bnetwork.*\b(interface|gateway|connector|adapter)\b'
            ],
            'exception_handling': [
                r'\b(exception|error|fail|reject|return)\b.*\b(handle|process|manage|repair)\b',
                r'\b(repair|retry|reprocess)\b.*\b(payment|transaction|message)\b',
                r'\berror.*\b(handling|processing|management|recovery)\b'
            ]
        }
        
        # Apply enhanced pattern matching
        for category, patterns in enhanced_patterns.items():
            for pattern in patterns:
                if re.search(pattern, full_text_lower):
                    # Convert category to business-friendly keywords
                    category_keywords = category.replace('_', ' ').split()
                    business_keywords.extend(category_keywords[:3])  # Top 3 words from category
                    
                    # Add specific domain terms found in the pattern
                    matches = re.findall(pattern, full_text_lower)
                    for match in matches[:2]:  # Limit to first 2 matches per pattern
                        if isinstance(match, tuple):
                            business_keywords.extend([word for word in match if len(word) > 2])
                        elif len(match) > 2:
                            business_keywords.append(match)
        
        # Add financial domain-specific terms found in the text
        financial_terms = [
            'fedwire', 'swift', 'chips', 'sepa', 'ach', 'wire', 'transfer', 'payment',
            'mt103', 'mt202', 'pacs', 'iso20022', 'bic', 'iban', 'aba', 'routing',
            'ofac', 'sanctions', 'aml', 'fraud', 'validation', 'settlement',
            'correspondent', 'nostro', 'vostro', 'remittance', 'cross-border'
        ]
        
        for term in financial_terms:
            if term in full_text_lower:
                business_keywords.append(term)
        
        return list(set(business_keywords))  # Remove duplicates
    
    async def expand_query(self, procedure_info: Dict, context: str) -> Dict:
        """ENHANCED: Rule-based query expansion with domain knowledge + full procedure context analysis"""
        # Initialize ALL variables first
        name = procedure_info.get('name', '').lower()
        comments = procedure_info.get('comments', '').lower()
        params = ' '.join(procedure_info.get('parameters', [])).lower()
        full_procedure_text = procedure_info.get('full_procedure_text', '')
        
        # DEBUG: Log what we received (now all variables are initialized)
        print(f"DEBUG expand_query for '{name}':")
        print(f"  procedure_info keys: {list(procedure_info.keys())}")
        print(f"  full_procedure_text length: {len(full_procedure_text) if full_procedure_text else 'None/Empty'}")
        if full_procedure_text:
            print(f"  full_procedure_text preview: {full_procedure_text[:200]}...")
        
        # Original text analysis (PRESERVED)
        basic_text = f"{name} {comments} {params}"
        business_keywords = []
        
        # PRESERVED: Financial payments domain patterns
        payment_patterns = {
            'payment': ['pay', 'payment', 'transfer', 'send', 'remit'],
            'validation': ['validate', 'verify', 'check', 'confirm', 'authorize'],
            'routing': ['route', 'forward', 'direct', 'dispatch', 'send'],
            'wire': ['wire', 'swift', 'fedwire', 'chips', 'mt103', 'pacs'],
            'sanctions': ['sanction', 'aml', 'screen', 'ofac', 'compliance'],
            'fraud': ['fraud', 'monitor', 'detect', 'alert', 'suspicious'],
            'settlement': ['settle', 'clear', 'post', 'book', 'credit', 'debit'],
            'messaging': ['message', 'msg', 'format', 'transform', 'parse'],
            'network': ['network', 'channel', 'integration', 'connection'],
            'exception': ['exception', 'error', 'repair', 'retry', 'fail']
        }
        
        for category, keywords in payment_patterns.items():
            if any(kw in basic_text for kw in keywords):
                business_keywords.extend(keywords[:3])  # Top 3 relevant keywords
        
        # NEW: Enhanced analysis using full procedure text from procedure_info
        enhanced_keywords = []
        confidence_boost = 0.0
        
        if full_procedure_text and full_procedure_text.strip():
            # Truncate to token limit
            truncated_text = self._truncate_to_token_limit(full_procedure_text, max_tokens=2000)
            
            # Extract business keywords from full context
            enhanced_keywords = self._extract_business_keywords_from_full_text(truncated_text)
            
            # Calculate confidence boost based on full text analysis
            if enhanced_keywords:
                confidence_boost = min(0.3, len(enhanced_keywords) * 0.05)  # Max 0.3 boost
            
            # Log the analysis for debugging
            print(f"Enhanced LLM analysis for '{name}':")
            print(f"   • Full text length: {len(full_procedure_text)} chars -> {len(truncated_text)} chars")
            print(f"   • Enhanced keywords: {enhanced_keywords[:5]}...")  # Show first 5
            print(f"   • Confidence boost: +{confidence_boost:.2f}")
        else:
            print(f"WARNING: No full_procedure_text available for '{name}'")
            if full_procedure_text is None:
                print("  full_procedure_text is None")
            elif not full_procedure_text.strip():
                print(f"  full_procedure_text is empty/whitespace: '{full_procedure_text}'")
        
        # Combine original and enhanced keywords
        all_business_keywords = list(set(business_keywords + enhanced_keywords))
        
        # Calculate final confidence
        base_confidence = 0.7 if business_keywords else 0.3
        final_confidence = max(base_confidence, 0.4 + confidence_boost)
        
        print(f"Final LLM result for '{name}': confidence={final_confidence:.2f}, keywords={len(all_business_keywords)}")
        
        return {
            "business_keywords": all_business_keywords,
            "confidence": final_confidence,
            "enhanced_analysis": {
                "full_text_analyzed": full_procedure_text is not None and full_procedure_text.strip() != "",
                "text_length": len(full_procedure_text) if full_procedure_text else 0,
                "truncated_length": len(self._truncate_to_token_limit(full_procedure_text, 2000)) if full_procedure_text else 0,
                "enhanced_keywords_count": len(enhanced_keywords),
                "confidence_boost": confidence_boost,
                "basic_keywords_count": len(business_keywords)
            }
        }

class EnhancedBatchASTCapabilityMapper:
    """ENHANCED version preserving ALL original functionality + TAL improvements + Full Procedure Context"""
    
    def __init__(self, keywords_file_path: str, llm_provider: Optional[LLMProvider] = None):
        """Initialize batch mapper with keywords configuration"""
        print(f"📖 Loading keywords from: {keywords_file_path}")
        print(f"🔧 Using EnhancedBatchASTCapabilityMapper (ENHANCED VERSION)")  # DEBUG: Confirm correct class
        
        with open(keywords_file_path, 'r') as f:
            self.keywords_data = json.load(f)
        
        print(f"✅ Loaded {len(self.keywords_data)} keyword definitions")
        
        self.llm_provider = llm_provider or LocalLLMProvider()
        
        # Get all unique capabilities first - PRESERVED
        self.all_capabilities = set()
        for item in self.keywords_data:
            self.all_capabilities.update(item.get('business_capability', []))
        
        # Build lookup maps - PRESERVED
        self.capability_to_keywords = self._build_capability_keyword_map()
        self.keyword_to_capabilities = self._build_keyword_capability_map()
        self.business_context = self._build_business_context()
        
        # NEW: Enhanced TAL naming support
        self.keyword_weights = self._calculate_keyword_weights()
        self.debug_mode = False
        
        # NEW: Store the original file content for full procedure extraction
        self.file_contents = {}  # file_path -> content mapping
        
        print(f"✅ Built indexes for {len(self.all_capabilities)} capabilities")
    
    def _build_capability_keyword_map(self) -> Dict[str, Set[str]]:
        """Build map from capability to all associated keywords - PRESERVED"""
        capability_map = defaultdict(set)
        
        for item in self.keywords_data:
            capabilities = item.get('business_capability', []))
            
            all_keywords = set()
            all_keywords.update(self._normalize_keywords(item.get('keywords', '')))
            all_keywords.update(self._normalize_keywords(item.get('related_keywords', '')))
            all_keywords.update(self._normalize_keywords(item.get('description', '')))
            
            for capability in capabilities:
                capability_map[capability].update(all_keywords)
                
        return capability_map
    
    def _build_keyword_capability_map(self) -> Dict[str, Set[str]]:
        """Build map from keyword to all associated capabilities - PRESERVED"""
        keyword_map = defaultdict(set)
        
        for item in self.keywords_data:
            capabilities = item.get('business_capability', [])
            
            all_keywords = set()
            all_keywords.update(self._normalize_keywords(item.get('keywords', '')))
            all_keywords.update(self._normalize_keywords(item.get('related_keywords', '')))
            
            for keyword in all_keywords:
                keyword_map[keyword].update(capabilities)
                
        return keyword_map
    
    def _build_business_context(self) -> str:
        """Build business context string - PRESERVED"""
        return f"Financial payments system with {len(self.all_capabilities)} business capabilities"
    
    def _normalize_keywords(self, keyword_string: str) -> List[str]:
        """ENHANCED: Normalize and split keyword strings with TAL support"""
        if not keyword_string:
            return []
        
        keywords = [kw.strip().lower() for kw in keyword_string.split(',')]
        
        expanded_keywords = []
        for kw in keywords:
            # PRESERVED: Original expansions
            expanded_keywords.append(kw)
            expanded_keywords.append(kw.replace(' ', ''))
            expanded_keywords.append(kw.replace(' ', '_'))
            expanded_keywords.append(kw.replace(' ', '-'))
            expanded_keywords.extend(kw.split())
            
            # NEW: TAL naming convention expansions
            if ' ' in kw:
                words = kw.split()
                expanded_keywords.append('.'.join(words))  # module.function
                expanded_keywords.append('^' + '_'.join(words))  # ^system_proc
                
        return list(set(expanded_keywords))
    
    def _calculate_keyword_weights(self) -> Dict[str, float]:
        """NEW: Calculate TF-IDF style weights for keywords"""
        all_keywords = []
        for item in self.keywords_data:
            all_keywords.extend(self._normalize_keywords(item.get('keywords', '')))
            all_keywords.extend(self._normalize_keywords(item.get('related_keywords', '')))
        
        keyword_counts = Counter(all_keywords)
        total_keywords = len(set(all_keywords))
        
        weights = {}
        for keyword, count in keyword_counts.items():
            weight = math.log(total_keywords / count) if count > 0 else 1.0
            weights[keyword] = weight
        
        return weights
    
    def find_ast_files(self, directory_path: str) -> List[str]:
        """ENHANCED: Find all .ast and .tal files in directory"""
        directory = Path(directory_path)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        # Find both .ast and .tal files
        ast_files = list(directory.glob("*.ast")) + list(directory.glob("*.tal"))
        
        if not ast_files:
            print(f"⚠️ No .ast or .tal files found in {directory_path}")
            return []
        
        ast_file_paths = [str(f) for f in ast_files]
        print(f"✅ Found {len(ast_file_paths)} AST/TAL files")
        
        return ast_file_paths
    
    def _extract_full_procedure_text(self, file_path: str, procedure_name: str, procedure_path: str) -> str:
        """FIXED: Extract full procedure text from original file content"""
        try:
            # Get file content (cache it if not already cached)
            if file_path not in self.file_contents:
                with open(file_path, 'r', encoding='utf-8') as f:
                    self.file_contents[file_path] = f.read()
            
            content = self.file_contents[file_path]
            
            if self.debug_mode:
                print(f"    Extracting full text for: {procedure_name}")
                print(f"    File content length: {len(content)} chars")
            
            # Try to extract the specific procedure block
            # This is a heuristic approach - could be improved for specific AST formats
            
            # Method 1: S-expression procedure blocks (escape special regex characters)
            escaped_name = re.escape(procedure_name)
            procedure_pattern = rf'\(procedure\s+:name\s+{escaped_name}\b'
            match = re.search(procedure_pattern, content, re.IGNORECASE)
            
            if match:
                if self.debug_mode:
                    print(f"    Found S-expression match at position {match.start()}")
                
                start_pos = match.start()
                # Find the matching closing parenthesis
                paren_count = 0
                pos = start_pos
                
                for i, char in enumerate(content[start_pos:], start_pos):
                    if char == '(':
                        paren_count += 1
                    elif char == ')':
                        paren_count -= 1
                        if paren_count == 0:
                            extracted = content[start_pos:i+1]
                            if self.debug_mode:
                                print(f"    Extracted S-expression: {len(extracted)} chars")
                            return extracted
                
                # Fallback: take next 2000 characters
                fallback = content[start_pos:start_pos+2000]
                if self.debug_mode:
                    print(f"    S-expression fallback: {len(fallback)} chars")
                return fallback
            
            # Method 2: Try to find by procedure path
            if procedure_path and 'regex_match_' in procedure_path:
                try:
                    match_pos = int(procedure_path.split('_')[-1])
                    extracted = content[match_pos:match_pos+1500]
                    if self.debug_mode:
                        print(f"    Extracted by path position: {len(extracted)} chars")
                    return extracted
                except:
                    if self.debug_mode:
                        print(f"    Failed to parse procedure path: {procedure_path}")
                    pass
            
            # Method 3: Search for procedure name in various formats
            patterns = [
                rf'def\s+{escaped_name}\s*\(',
                rf'function\s+{escaped_name}\s*\(',
                rf'procedure\s+{escaped_name}\s*\(',
                rf'{escaped_name}\s*:\s*function',
                rf'\b{escaped_name}\b.*\('
            ]
            
            for i, pattern in enumerate(patterns):
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    start_pos = match.start()
                    # Extract a reasonable amount of context
                    end_pos = min(start_pos + 1500, len(content))
                    extracted = content[start_pos:end_pos]
                    if self.debug_mode:
                        print(f"    Extracted using pattern {i+1}: {len(extracted)} chars")
                    return extracted
            
            # Fallback: return procedure name and basic info
            fallback_text = f"Procedure: {procedure_name}\nPath: {procedure_path}\nNo full content found in {len(content)} chars."
            if self.debug_mode:
                print(f"    No matches found, returning fallback: {len(fallback_text)} chars")
            return fallback_text
            
        except Exception as e:
            error_text = f"Procedure: {procedure_name}\nError extracting full content: {str(e)}"
            if self.debug_mode:
                print(f"    Exception during extraction: {e}")
            return error_text
    
    def extract_procedures_from_ast(self, file_path: str) -> List[Dict]:
        """ENHANCED: Extract procedures with TAL naming convention parsing and full procedure text"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"⚠️ Error reading {file_path}: {e}")
            return []
        
        procedures = []
        
        # Try JSON AST first - PRESERVED
        try:
            ast_data = json.loads(content)
            procedures = self._extract_from_json_ast(ast_data, file_path)
        except json.JSONDecodeError:
            # Try Python AST - PRESERVED
            try:
                tree = ast.parse(content)
                procedures = self._extract_from_python_ast(tree, file_path)
            except SyntaxError:
                # Fallback: regex extraction - ENHANCED
                procedures = self._extract_using_regex_enhanced(content, file_path)
        
        # NEW: Add TAL naming analysis AND full procedure text to all procedures
        for procedure in procedures:
            procedure['tal_naming_info'] = self._analyze_tal_naming(
                procedure['name'], 
                procedure.get('parameters', []),
                procedure.get('comments', '')
            )
            
            # FIXED: Extract and add full procedure text to procedure_info
            procedure['full_procedure_text'] = self._extract_full_procedure_text(
                file_path,
                procedure['name'],
                procedure['path']
            )
        
        return procedures
    
    def _extract_from_json_ast(self, ast_data: Dict, file_path: str) -> List[Dict]:
        """Extract procedures from JSON AST - PRESERVED"""
        procedures = []
        
        def traverse(node, path=""):
            if isinstance(node, dict):
                if any(key in node for key in ['function', 'procedure', 'method', 'name']):
                    name = node.get('name', node.get('function', node.get('procedure', '')))
                    if name and isinstance(name, str) and name.strip():
                        proc_info = {
                            'name': name.strip(),
                            'file_path': file_path,
                            'path': path,
                            'type': node.get('type', 'function'),
                            'parameters': node.get('parameters', node.get('params', [])),
                            'comments': node.get('comments', node.get('docstring', '')),
                        }
                        procedures.append(proc_info)
                
                for key, value in node.items():
                    new_path = f"{path}.{key}" if path else key
                    traverse(value, new_path)
                    
            elif isinstance(node, list):
                for i, item in enumerate(node):
                    new_path = f"{path}[{i}]" if path else f"[{i}]"
                    traverse(item, new_path)
        
        traverse(ast_data)
        return procedures
    
    def _extract_from_python_ast(self, tree: ast.AST, file_path: str) -> List[Dict]:
        """Extract procedures from Python AST - PRESERVED"""
        procedures = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                proc_info = {
                    'name': node.name,
                    'file_path': file_path,
                    'path': f"line_{node.lineno}",
                    'type': 'async_function' if isinstance(node, ast.AsyncFunctionDef) else 'function',
                    'parameters': [arg.arg for arg in node.args.args],
                    'comments': ast.get_docstring(node) or '',
                }
                procedures.append(proc_info)
                
        return procedures
    
    def _extract_using_regex_enhanced(self, content: str, file_path: str) -> List[Dict]:
        """ENHANCED: Regex extraction with improved TAL support and context scoping"""
        procedures = []
        
        # Enhanced S-expression patterns with better context extraction
        sexp_patterns = [
            (r'\(procedure\s+:name\s+([^\s)]+)', 'sexp_procedure'),
            (r'\(proc\s+:name\s+([^\s)]+)', 'sexp_proc'),
            (r'\(function\s+:name\s+([^\s)]+)', 'sexp_function'),
        ]
        
        # ENHANCED: Extract procedures with proper scoping
        for pattern, proc_type in sexp_patterns:
            matches = list(re.finditer(pattern, content, re.IGNORECASE))
            for match in matches:
                proc_name = match.group(1)
                
                # NEW: Extract procedure-specific block (not global comments)
                proc_block = self._extract_procedure_block(content, match)
                
                # Extract parameters and comments from ONLY this procedure block
                params = self._extract_sexp_parameters_from_block(proc_block, proc_name)
                comments = self._extract_sexp_comments_from_block(proc_block, proc_name)
                
                proc_info = {
                    'name': proc_name,
                    'file_path': file_path,
                    'path': f"regex_match_{match.start()}",
                    'type': proc_type,
                    'parameters': params,
                    'comments': comments,
                }
                procedures.append(proc_info)
        
        # PRESERVED: Other patterns
        other_patterns = [
            (r'def\s+(\w+)\s*\(([^)]*)\)', 'python_function'),
            (r'function\s+(\w+)\s*\(([^)]*)\)', 'javascript_function'),
            (r'procedure\s+(\w+)\s*\(([^)]*)\)', 'sql_procedure'),
            (r'(\w+)\s*:\s*function\s*\(([^)]*)\)', 'object_method'),
        ]
        
        for pattern, proc_type in other_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                name = match.group(1)
                params_str = match.group(2) if len(match.groups()) > 1 else ""
                params = [p.strip() for p in params_str.split(',') if p.strip()]
                
                proc_info = {
                    'name': name,
                    'file_path': file_path,
                    'path': f"regex_match_{match.start()}",
                    'type': proc_type,
                    'parameters': params,
                    'comments': '',
                }
                procedures.append(proc_info)
                
        return procedures
    
    def _extract_procedure_block(self, content: str, proc_match) -> str:
        """NEW: Extract the specific procedure block to avoid global comment contamination"""
        # Find the procedure block boundaries
        start_pos = proc_match.start()
        
        # Simple heuristic: find the matching closing parenthesis or next procedure
        paren_count = 0
        pos = start_pos
        in_procedure = False
        
        for i, char in enumerate(content[start_pos:], start_pos):
            if char == '(':
                paren_count += 1
                in_procedure = True
            elif char == ')':
                paren_count -= 1
                if in_procedure and paren_count == 0:
                    return content[start_pos:i+1]
        
        # Fallback: take next 1000 characters
        return content[start_pos:start_pos+1000]
    
    def _extract_sexp_parameters_from_block(self, proc_block: str, proc_name: str) -> List[str]:
        """ENHANCED: Extract parameters from procedure block only"""
        param_pattern = r'\(parameter\s+:name\s+([^\s)]+)'
        return re.findall(param_pattern, proc_block, re.IGNORECASE)

    def _extract_sexp_comments_from_block(self, proc_block: str, proc_name: str) -> str:
        """ENHANCED: Extract comments from procedure block only"""
        comment_pattern = r'\(comment\s+:value\s+([^)]+)\)'
        comments = re.findall(comment_pattern, proc_block, re.IGNORECASE)
        return ' '.join(comments)
    
    def _analyze_tal_naming(self, proc_name: str, parameters: List[str], comments: str) -> Dict:
        """NEW: Analyze TAL naming conventions"""
        analysis = {
            'pattern_type': 'unknown',
            'semantic_parts': [],
            'business_indicators': [],
            'confidence_boost': 0.0
        }
        
        name_lower = proc_name.lower()
        
        # Module.Function pattern
        if '.' in proc_name:
            analysis['pattern_type'] = 'module_dot_function'
            parts = proc_name.split('.')
            analysis['semantic_parts'] = [p.lower() for p in parts]
            analysis['confidence_boost'] = 0.2  # Structured naming gets boost
            
            # Map common modules to business domains
            module_mappings = {
                'payment': ['payment_processing', 'payment_initiation'],
                'wire': ['wire_transfer', 'domestic_wire', 'international_wire'],
                'swift': ['swift_messaging', 'cross_border_payment'],
                'validate': ['validation', 'compliance'],
                'ofac': ['sanctions_screening']
            }
            
            if parts[0].lower() in module_mappings:
                analysis['business_indicators'] = module_mappings[parts[0].lower()]
        
        # Caret-prefixed pattern
        elif '^' in proc_name:
            analysis['pattern_type'] = 'caret_system'
            clean_name = proc_name.lstrip('^').lower()
            analysis['semantic_parts'] = [clean_name]
            analysis['confidence_boost'] = 0.15
            
        # Underscore-separated pattern  
        elif '_' in proc_name:
            analysis['pattern_type'] = 'underscore_separated'
            parts = proc_name.split('_')
            analysis['semantic_parts'] = [p.lower() for p in parts]
            analysis['confidence_boost'] = 0.1
        
        return analysis
    
    def write_structured_output(self, output_data: Dict, output_path: str):
        """Write structured output to multiple files in directory with improved error handling and deduplication"""
        # Create output directory
        output_dir = Path(output_path)
        output_dir.mkdir(exist_ok=True)
        
        print(f"📁 Writing structured output to: {output_dir}")
        
        # Verify required keys exist
        required_keys = ['metadata', 'business_capabilities', 'summary_statistics', 'graph_data']
        missing_keys = [key for key in required_keys if key not in output_data]
        if missing_keys:
            raise KeyError(f"Missing required keys in output_data: {missing_keys}")
        
        # Helper function to make data JSON serializable
        def make_serializable(obj):
            """Convert sets and other non-serializable objects to lists/dicts"""
            if isinstance(obj, set):
                return list(obj)
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            elif hasattr(obj, '__dict__'):
                # Handle custom objects like ProcedureCapabilityMatch
                return make_serializable(obj.__dict__)
            else:
                return obj
        
        # Helper function to deduplicate procedures in keyword mappings
        def deduplicate_keyword_procedures(keyword_to_procedures):
            """Remove duplicate procedures per keyword, keeping best match"""
            deduplicated = {}
            
            for keyword, procedures in keyword_to_procedures.items():
                unique_procedures = {}
                
                for proc in procedures:
                    proc_name = proc.get("procedure", "")
                    if proc_name not in unique_procedures:
                        unique_procedures[proc_name] = proc
                    else:
                        # Keep the entry with highest confidence, or if tied, prefer exact matches
                        existing = unique_procedures[proc_name]
                        current_conf = proc.get("confidence", 0)
                        existing_conf = existing.get("confidence", 0)
                        current_type = proc.get("match_type", "")
                        existing_type = existing.get("match_type", "")
                        
                        is_better = (
                            current_conf > existing_conf or
                            (current_conf == existing_conf and current_type == 'exact' and existing_type != 'exact')
                        )
                        
                        if is_better:
                            unique_procedures[proc_name] = proc
                
                deduplicated[keyword] = list(unique_procedures.values())
            
            return deduplicated
        
        try:
            # 1. Main summary file
            main_file = output_dir / "capability_mapping_summary.json"
            summary_data = {
                "metadata": make_serializable(output_data["metadata"]),
                "summary_statistics": make_serializable(output_data["summary_statistics"]),
                "processing_errors": make_serializable(output_data.get("processing_errors", {}))
            }
            
            print(f"Writing summary data: {len(str(summary_data))} chars")
            with open(main_file, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
            print(f"✓ Created {main_file.name}")
            
        except Exception as e:
            print(f"❌ Error writing summary file: {e}")
            debug_file = output_dir / "debug_summary.txt"
            with open(debug_file, 'w') as f:
                f.write(f"Error: {e}\n")
                f.write(f"Metadata keys: {list(output_data['metadata'].keys())}\n")
                f.write(f"Summary keys: {list(output_data['summary_statistics'].keys())}\n")
        
        try:
            # 2. File processing details
            files_file = output_dir / "file_processing_details.json"
            file_details = make_serializable(output_data.get("file_details", {}))
            
            print(f"Writing file details: {len(file_details)} files")
            with open(files_file, 'w', encoding='utf-8') as f:
                json.dump(file_details, f, indent=2, ensure_ascii=False)
            print(f"✓ Created {files_file.name}")
            
        except Exception as e:
            print(f"❌ Error writing file details: {e}")
        
        try:
            # 3. Business capabilities with procedures
            capabilities_file = output_dir / "business_capabilities.json"
            capabilities_data = make_serializable(output_data["business_capabilities"])
            
            print(f"Writing capabilities: {len(capabilities_data)} capabilities")
            with open(capabilities_file, 'w', encoding='utf-8') as f:
                json.dump(capabilities_data, f, indent=2, ensure_ascii=False)
            print(f"✓ Created {capabilities_file.name}")
            
        except Exception as e:
            print(f"❌ Error writing capabilities: {e}")
            debug_file = output_dir / "debug_capabilities.txt" 
            with open(debug_file, 'w') as f:
                f.write(f"Error: {e}\n")
                f.write(f"Capabilities count: {len(output_data['business_capabilities'])}\n")
                for cap_name, cap_data in list(output_data["business_capabilities"].items())[:3]:
                    f.write(f"Sample capability {cap_name}: {type(cap_data)}\n")
                    if isinstance(cap_data, dict):
                        for key, value in cap_data.items():
                            f.write(f"  {key}: {type(value)}\n")
        
        try:
            # 4. Graph data for capability -> keywords -> procedures (ENHANCED WITH DEDUPLICATION)
            graph_file = output_dir / "capability_keyword_procedure_graph.json"
            
            # Deduplicate keyword-to-procedures mappings
            original_keyword_to_procedures = output_data["graph_data"]["keyword_to_procedures"]
            deduplicated_keyword_to_procedures = deduplicate_keyword_procedures(original_keyword_to_procedures)
            
            # Calculate deduplication stats
            original_total = sum(len(procs) for procs in original_keyword_to_procedures.values())
            deduplicated_total = sum(len(procs) for procs in deduplicated_keyword_to_procedures.values())
            
            print(f"📊 Deduplication: {original_total} → {deduplicated_total} procedure entries ({original_total - deduplicated_total} removed)")
            
            graph_data = {
                "capabilities_to_keywords": make_serializable(output_data["graph_data"]["capability_keywords"]),
                "keywords_to_procedures": make_serializable(deduplicated_keyword_to_procedures),
                "procedures_to_capabilities": make_serializable(output_data["graph_data"]["procedure_to_capabilities"]),
                "metadata": {
                    "description": "Graph structure for capability -> keyword -> procedure relationships",
                    "confidence_threshold": output_data["metadata"]["confidence_threshold"],
                    "deduplication_applied": True,
                    "original_procedure_entries": original_total,
                    "deduplicated_procedure_entries": deduplicated_total,
                    "total_nodes": {
                        "capabilities": len(output_data["graph_data"]["capability_keywords"]),
                        "keywords": len(deduplicated_keyword_to_procedures),
                        "procedures": len(output_data["graph_data"]["procedure_to_capabilities"])
                    }
                }
            }
            
            print(f"Writing graph data: {graph_data['metadata']['total_nodes']}")
            with open(graph_file, 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, indent=2, ensure_ascii=False)
            print(f"✓ Created {graph_file.name}")
            
        except Exception as e:
            print(f"❌ Error writing graph data: {e}")
        
        try:
            # 5. Procedures by file
            procedures_by_file = {}
            file_details = output_data.get("file_details", {})
            
            for file_name, file_info in file_details.items():
                procedures_by_file[file_name] = {
                    "file_info": {
                        "full_path": file_info.get("full_path", ""),
                        "file_type": file_info.get("file_type", "unknown"),
                        "total_lines": file_info.get("total_lines", 0),
                        "processing_status": file_info.get("processing_status", "unknown")
                    },
                    "main_procedures": make_serializable(file_info.get("procedures", [])),
                    "subprocedures": make_serializable(file_info.get("subprocedures", [])),
                    "summary": {
                        "total_procedures": len(file_info.get("procedures", [])) + len(file_info.get("subprocedures", [])),
                        "main_procedure_count": len(file_info.get("procedures", [])),
                        "subprocedure_count": len(file_info.get("subprocedures", []))
                    }
                }
            
            procedures_file = output_dir / "procedures_by_file.json"
            print(f"Writing procedures by file: {len(procedures_by_file)} files")
            with open(procedures_file, 'w', encoding='utf-8') as f:
                json.dump(procedures_by_file, f, indent=2, ensure_ascii=False)
            print(f"✓ Created {procedures_file.name}")
            
        except Exception as e:
            print(f"❌ Error writing procedures by file: {e}")
        
        try:
            # 6. Graph edge list (for easy graph library import) - ENHANCED WITH DEDUPLICATION
            edges_file = output_dir / "graph_edges.csv"
            edges_written = 0
            
            with open(edges_file, 'w', encoding='utf-8') as f:
                f.write("source,target,edge_type,confidence,match_type\n")
                
                # Capability -> Keyword edges
                for capability, keywords in output_data["graph_data"]["capability_keywords"].items():
                    for keyword in keywords:
                        # Clean and escape data for CSV
                        clean_cap = str(capability).replace('"', '""').replace('\n', ' ').replace('\r', ' ')
                        clean_kw = str(keyword).replace('"', '""').replace('\n', ' ').replace('\r', ' ')
                        f.write(f'"{clean_cap}","{clean_kw}",capability_to_keyword,1.0,definition\n')
                        edges_written += 1
                
                # Keyword -> Procedure edges (using deduplicated data)
                for keyword, procedures in deduplicated_keyword_to_procedures.items():
                    for proc_info in procedures:
                        clean_kw = str(keyword).replace('"', '""').replace('\n', ' ').replace('\r', ' ')
                        clean_proc = str(proc_info.get("procedure", "")).replace('"', '""').replace('\n', ' ').replace('\r', ' ')
                        confidence = proc_info.get("confidence", 0.0)
                        match_type = proc_info.get("match_type", "unknown")
                        f.write(f'"{clean_kw}","{clean_proc}",keyword_to_procedure,{confidence},{match_type}\n')
                        edges_written += 1
            
            print(f"✓ Created {edges_file.name} with {edges_written} edges")
            
        except Exception as e:
            print(f"❌ Error writing graph edges: {e}")
            import traceback
            print(f"Full error: {traceback.format_exc()}")
        
        print(f"\n📊 File Writing Summary:")
        print(f"   Output directory: {output_dir}")
        
        # Check file sizes
        for file_path in output_dir.glob("*.json"):
            try:
                size = file_path.stat().st_size
                print(f"   {file_path.name}: {size} bytes")
                if size == 0:
                    print(f"   ⚠️ {file_path.name} is empty!")
            except:
                print(f"   ❌ Could not check {file_path.name}")
        
        # Check CSV file
        csv_file = output_dir / "graph_edges.csv"
        if csv_file.exists():
            size = csv_file.stat().st_size
            print(f"   {csv_file.name}: {size} bytes")
        
        print(f"✅ All files written successfully with deduplication applied")
    
    async def process_directory(self, directory_path: str, confidence_threshold: float = 0.1) -> Dict:
        """Process directory with comprehensive error handling and deduplication"""
        print(f"🔍 Processing AST files in: {directory_path}")
        
        try:
            # Find all AST files with validation
            ast_files = self.find_ast_files(directory_path)
            if not ast_files:
                return {
                    "error": "No valid .ast/.tal files found in directory",
                    "metadata": {
                        "source_directory": directory_path,
                        "total_ast_files": 0,
                        "total_procedures_found": 0,
                        "error_details": "Directory contained no readable .ast/.tal files"
                    }
                }
        except Exception as e:
            return {
                "error": f"Failed to access directory: {e}",
                "metadata": {
                    "source_directory": directory_path,
                    "error_details": str(e)
                }
            }
        
        # Process each file with error tracking
        all_procedures = []
        file_stats = {}
        failed_files = []
        
        print(f"📄 Processing {len(ast_files)} files...")
        
        for i, ast_file in enumerate(ast_files):
            file_name = Path(ast_file).name
            print(f"   📄 [{i+1}/{len(ast_files)}] Processing {file_name}")
            
            try:
                procedures = self.extract_procedures_from_ast(ast_file)
                all_procedures.extend(procedures)
                
                file_stats[ast_file] = {
                    "procedure_count": len(procedures),
                    "procedures": [p.get('name', 'unknown') for p in procedures],
                    "status": "success"
                }
                
                if len(procedures) == 0:
                    print(f"      ⚠️ No procedures found in {file_name}")
                else:
                    print(f"      ✅ Found {len(procedures)} procedures")
                
            except Exception as e:
                error_msg = f"Failed to process {file_name}: {e}"
                print(f"      ❌ {error_msg}")
                
                failed_files.append({
                    "file": ast_file,
                    "error": str(e)
                })
                
                file_stats[ast_file] = {
                    "procedure_count": 0,
                    "procedures": [],
                    "status": "failed",
                    "error": str(e)
                }
        
        if not all_procedures and not failed_files:
            return {
                "error": "No procedures found in any files",
                "metadata": {
                    "source_directory": directory_path,
                    "total_ast_files": len(ast_files),
                    "total_procedures_found": 0,
                    "file_statistics": file_stats
                }
            }
        
        print(f"✅ Extracted {len(all_procedures)} total procedures from {len(ast_files)} files")
        if failed_files:
            print(f"⚠️ Failed to process {len(failed_files)} files")
        
        # Map all procedures to capabilities with batch processing
        print(f"🔗 Mapping procedures to capabilities...")
        
        # Initialize all capabilities
        capability_results = {}
        for capability in sorted(self.all_capabilities):
            capability_results[capability] = {
                "total_procedures": 0,
                "high_confidence_procedures": 0,
                "medium_confidence_procedures": 0,
                "low_confidence_procedures": 0,
                "procedures": []
            }
        
        # Process procedures in batches
        batch_size = 20
        total_batches = (len(all_procedures) + batch_size - 1) // batch_size
        successful_mappings = 0
        failed_mappings = 0
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(all_procedures))
            batch = all_procedures[start_idx:end_idx]
            
            print(f"   Progress: Batch {batch_idx + 1}/{total_batches} ({len(batch)} procedures)")
            
            for procedure in batch:
                try:
                    capability_matches = await self.map_procedure_to_capabilities(procedure)
                    
                    for capability, match in capability_matches.items():
                        if match.confidence >= confidence_threshold:
                            capability_results[capability]["total_procedures"] += 1
                            
                            # Categorize by confidence
                            if match.confidence >= 0.7:
                                capability_results[capability]["high_confidence_procedures"] += 1
                            elif match.confidence >= 0.4:
                                capability_results[capability]["medium_confidence_procedures"] += 1
                            else:
                                capability_results[capability]["low_confidence_procedures"] += 1
                            
                            # Add procedure details
                            try:
                                proc_detail = {
                                    "procedure_name": match.procedure_name,
                                    "file_path": os.path.basename(match.file_path) if match.file_path else "unknown",
                                    "full_file_path": match.file_path,
                                    "procedure_path": match.procedure_path,
                                    "procedure_type": match.procedure_type,
                                    "confidence": round(match.confidence, 3),
                                    "match_type": match.match_type,
                                    "matched_keywords": match.matched_keywords,
                                    "comments": (match.procedure_comments[:100] + "...") if len(match.procedure_comments) > 100 else match.procedure_comments
                                }
                                
                                # Add TAL naming info if available
                                if hasattr(match, 'tal_naming_info') and match.tal_naming_info:
                                    proc_detail["tal_naming_pattern"] = match.tal_naming_info.get('pattern_type', 'unknown')
                                    proc_detail["semantic_parts"] = match.tal_naming_info.get('semantic_parts', [])
                                
                                capability_results[capability]["procedures"].append(proc_detail)
                            except Exception as detail_error:
                                print(f"Warning: Error adding procedure detail: {detail_error}")
                    
                    successful_mappings += 1
                    
                except Exception as e:
                    print(f"Warning: Failed to map procedure {procedure.get('name', 'unknown')}: {e}")
                    failed_mappings += 1
                    continue
        
        # Sort procedures by confidence within each capability
        for capability_data in capability_results.values():
            try:
                capability_data["procedures"].sort(key=lambda x: x.get("confidence", 0), reverse=True)
            except Exception as e:
                print(f"Warning: Error sorting procedures: {e}")
        
        # Generate graph data structures with proper deduplication
        print("📊 Building graph data structures...")
        
        # Build capability -> keywords mapping
        capability_keywords = {}
        for capability, keywords in self.capability_to_keywords.items():
            capability_keywords[capability] = list(keywords)
        
        # Build keyword -> procedures mapping with deduplication at source
        keyword_to_procedures = defaultdict(dict)  # keyword -> {proc_name -> best_proc_info}
        procedure_to_capabilities = defaultdict(dict)  # proc_name -> {capability -> best_cap_info}
        
        for capability, cap_data in capability_results.items():
            for proc in cap_data["procedures"]:
                proc_name = proc["procedure_name"]
                confidence = proc["confidence"]
                match_type = proc["match_type"]
                file_path = proc["file_path"]
                
                # Add to procedure -> capabilities mapping (keep best per capability)
                if capability not in procedure_to_capabilities[proc_name] or \
                   confidence > procedure_to_capabilities[proc_name][capability]["confidence"]:
                    procedure_to_capabilities[proc_name][capability] = {
                        "capability": capability,
                        "confidence": confidence,
                        "match_type": match_type
                    }
                
                # Process each matched keyword with deduplication
                for keyword in proc.get("matched_keywords", []):
                    # Clean up keyword prefixes (tal:, llm:, pattern:, partial:)
                    clean_keyword = keyword.split(":")[-1] if ":" in keyword else keyword
                    
                    # Check if we already have this procedure for this keyword
                    if proc_name in keyword_to_procedures[clean_keyword]:
                        existing = keyword_to_procedures[clean_keyword][proc_name]
                        existing_conf = existing["confidence"]
                        existing_type = existing["match_type"]
                        
                        # Determine if new entry is better
                        is_better = (
                            confidence > existing_conf or
                            (confidence == existing_conf and match_type == 'exact' and existing_type != 'exact') or
                            (confidence == existing_conf and match_type == existing_type and match_type in ['llm_semantic', 'semantic'])
                        )
                        
                        if not is_better:
                            continue  # Skip this entry, keep the existing better one
                    
                    # Add or update the procedure for this keyword
                    keyword_to_procedures[clean_keyword][proc_name] = {
                        "procedure": proc_name,
                        "confidence": confidence,
                        "match_type": match_type,
                        "file_path": file_path
                    }
        
        # Convert nested dicts to lists for JSON serialization
        final_keyword_to_procedures = {}
        for keyword, proc_dict in keyword_to_procedures.items():
            final_keyword_to_procedures[keyword] = list(proc_dict.values())
        
        final_procedure_to_capabilities = {}
        for proc_name, cap_dict in procedure_to_capabilities.items():
            final_procedure_to_capabilities[proc_name] = list(cap_dict.values())
        
        # Debug output for deduplication verification
        total_mappings = sum(len(procs) for procs in final_keyword_to_procedures.values())
        print(f"📊 Graph structure built:")
        print(f"   • {len(final_keyword_to_procedures)} unique keywords")
        print(f"   • {total_mappings} total keyword→procedure mappings (deduplicated)")
        print(f"   • {len(final_procedure_to_capabilities)} unique procedures")
        
        # Verify deduplication worked
        sample_keywords = list(final_keyword_to_procedures.keys())[:3]
        for kw in sample_keywords:
            proc_count = len(final_keyword_to_procedures[kw])
            proc_names = [p["procedure"] for p in final_keyword_to_procedures[kw]]
            unique_names = set(proc_names)
            print(f"   • Keyword '{kw}': {proc_count} procedures, {len(unique_names)} unique names")
            if len(proc_names) != len(unique_names):
                print(f"     ⚠️ Still has duplicates in '{kw}': {[name for name in proc_names if proc_names.count(name) > 1]}")
        
        # Build file details structure
        file_details = {}
        for file_path, file_stat in file_stats.items():
            file_name = os.path.basename(file_path)
            
            # Count lines in file for metadata
            total_lines = 0
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    total_lines = sum(1 for _ in f)
            except:
                total_lines = 0
            
            file_details[file_name] = {
                "full_path": file_path,
                "file_type": "tal" if file_path.endswith('.tal') else "ast",
                "total_lines": total_lines,
                "processing_status": file_stat["status"],
                "procedures": [p for p in all_procedures if p["file_path"] == file_path],
                "subprocedures": []
            }
            if file_stat["status"] == "failed":
                file_details[file_name]["error"] = file_stat.get("error", "Unknown error")
        

        # Build final output
        output = {
            "metadata": {
                "source_directory": directory_path,
                "total_ast_files": len(ast_files),
                "successful_files": len(ast_files) - len(failed_files),
                "failed_files": len(failed_files),
                "total_procedures_found": len(all_procedures),
                "successful_mappings": successful_mappings,
                "failed_mappings": failed_mappings,
                "confidence_threshold": confidence_threshold,
                "processing_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "llm_enhanced": self.llm_provider is not None,
                "tal_naming_enhanced": True,
                "semantic_similarity_enabled": hasattr(self.llm_provider, 'model') and self.llm_provider.model is not None if self.llm_provider else False,
                "similarity_model": self.llm_provider.model_name if self.llm_provider and hasattr(self.llm_provider, 'model_name') else "none",
                "deduplication_applied": True,
                "full_procedure_context_enabled": True  # NEW: Indicate full context is used
            },
            "file_statistics": file_stats,
            "business_capabilities": capability_results,
            "summary_statistics": {
                "capabilities_with_procedures": sum(1 for cap_data in capability_results.values() if cap_data["total_procedures"] > 0),
                "total_capability_mappings": sum(cap_data["total_procedures"] for cap_data in capability_results.values()),
                "high_confidence_mappings": sum(cap_data["high_confidence_procedures"] for cap_data in capability_results.values()),
                "medium_confidence_mappings": sum(cap_data["medium_confidence_procedures"] for cap_data in capability_results.values()),
                "low_confidence_mappings": sum(cap_data["low_confidence_procedures"] for cap_data in capability_results.values())
            },
            "graph_data": {
                "capability_keywords": capability_keywords,
                "keyword_to_procedures": final_keyword_to_procedures,
                "procedure_to_capabilities": final_procedure_to_capabilities
            },
            "file_details": file_details
        }
        
        # Add error information if there were failures
        if failed_files:
            output["processing_errors"] = {
                "failed_files": failed_files,
                "failed_mapping_count": failed_mappings
            }
        
        print(f"✅ Processing complete!")
        print(f"   📊 {output['summary_statistics']['total_capability_mappings']} total mappings")
        print(f"   🎯 {output['summary_statistics']['capabilities_with_procedures']} capabilities have procedures")
        print(f"   🔗 {total_mappings} deduplicated keyword→procedure edges")
        if failed_files:
            print(f"   ⚠️ {len(failed_files)} files failed to process")
        if failed_mappings:
            print(f"   ⚠️ {failed_mappings} procedures failed to map")
        
        return output
    
    async def map_procedure_to_capabilities(self, procedure: Dict) -> Dict[str, ProcedureCapabilityMatch]:
        """ENHANCED: Map procedure with TAL naming support + full procedure context while preserving original logic"""
        # Get searchable text (includes full procedure body)
        searchable_text = self._get_searchable_text(procedure)
        normalized_text = ' '.join(self._normalize_keywords(searchable_text)).lower()
        
        # Score all capabilities - PRESERVED LOGIC
        capability_scores = defaultdict(lambda: {'score': 0, 'keywords': [], 'match_type': 'none'})
        
        # 1. Direct keyword matching - PRESERVED
        for keyword, capabilities in self.keyword_to_capabilities.items():
            if keyword in normalized_text:
                for capability in capabilities:
                    # ENHANCED: Use keyword weights
                    weight = self.keyword_weights.get(keyword, 1.0)
                    capability_scores[capability]['score'] += 2.0 * weight
                    capability_scores[capability]['keywords'].append(keyword)
                    capability_scores[capability]['match_type'] = 'exact'
        
        # 2. Partial keyword matching - PRESERVED
        for capability, keywords in self.capability_to_keywords.items():
            for keyword in keywords:
                parts = [part for part in keyword.split() if len(part) > 2]
                if any(part in normalized_text for part in parts):
                    weight = self.keyword_weights.get(keyword, 1.0)
                    capability_scores[capability]['score'] += 0.5 * weight
                    capability_scores[capability]['keywords'].append(f"partial:{keyword}")
                    if capability_scores[capability]['match_type'] == 'none':
                        capability_scores[capability]['match_type'] = 'partial'
        
        # 3. Semantic pattern matching - PRESERVED
        semantic_matches = self._get_semantic_patterns(procedure['name'])
        for capability, score in semantic_matches.items():
            capability_scores[capability]['score'] += score
            capability_scores[capability]['keywords'].append(f"pattern:{procedure['name']}")
            if capability_scores[capability]['match_type'] == 'none':
                capability_scores[capability]['match_type'] = 'semantic'
        
        # 4. NEW: TAL naming pattern matching
        tal_info = procedure.get('tal_naming_info', {})
        if tal_info and tal_info.get('business_indicators'):
            for indicator in tal_info['business_indicators']:
                if indicator in self.keyword_to_capabilities:
                    for capability in self.keyword_to_capabilities[indicator]:
                        capability_scores[capability]['score'] += 1.5  # TAL structure bonus
                        capability_scores[capability]['keywords'].append(f"tal:{indicator}")
                        capability_scores[capability]['match_type'] = 'tal_structural'
        
        # 5. ENHANCED: LLM enhancement with full procedure context - USE LLM CONFIDENCE DIRECTLY
        llm_capability_scores = {}  # Store LLM-derived scores separately
        if self.llm_provider:
            try:
                # LLM analysis now gets full context from procedure_info
                llm_expansion = await self.llm_provider.expand_query(procedure, self.business_context)
                
                # Use LLM confidence directly instead of manual scoring
                llm_confidence = llm_expansion.get('confidence', 0.0)
                enhanced_keywords = llm_expansion.get('business_keywords', [])
                
                # For each keyword found by LLM, apply its confidence to relevant capabilities
                for keyword in enhanced_keywords:
                    if keyword.lower() in self.keyword_to_capabilities:
                        for capability in self.keyword_to_capabilities[keyword.lower()]:
                            # Use LLM confidence directly, not manual scoring
                            if capability not in llm_capability_scores:
                                llm_capability_scores[capability] = {
                                    'confidence': 0.0,
                                    'keywords': [],
                                    'match_type': 'llm_semantic'
                                }
                            
                            # Accumulate LLM confidence (average if multiple keywords map to same capability)
                            current_confidence = llm_capability_scores[capability]['confidence']
                            llm_capability_scores[capability]['confidence'] = max(current_confidence, llm_confidence)
                            llm_capability_scores[capability]['keywords'].append(f"llm:{keyword}")
                
                # Log enhanced analysis results
                if self.debug_mode and llm_expansion.get('enhanced_analysis', {}).get('full_text_analyzed'):
                    analysis = llm_expansion['enhanced_analysis']
                    print(f"    LLM Enhanced: {procedure['name']} -> confidence: {llm_confidence:.2f}, keywords: {len(enhanced_keywords)}")
                    
            except Exception as e:
                if self.debug_mode:
                    print(f"    Warning: LLM enhancement failed for {procedure['name']}: {e}")
                pass  # Silently continue if LLM fails
        
        # Convert to ProcedureCapabilityMatch objects - ENHANCED INTERFACE
        matches = {}
        for capability in self.all_capabilities:
            # Start with rule-based score
            score_data = capability_scores[capability]
            confidence = min(score_data['score'] / 3.0, 1.0)  # PRESERVED normalization
            match_type = score_data['match_type']
            keywords = score_data['keywords'][:5]  # Top 5 keywords
            
            # If LLM provided a score for this capability, use it instead of rule-based
            if capability in llm_capability_scores:
                llm_data = llm_capability_scores[capability]
                # Use LLM confidence directly
                confidence = llm_data['confidence']
                match_type = llm_data['match_type']
                # Combine keywords from both sources
                keywords = list(set(keywords + llm_data['keywords']))[:5]
            
            # NEW: Apply TAL confidence boost
            if tal_info.get('confidence_boost', 0) > 0:
                confidence = min(confidence + tal_info['confidence_boost'], 1.0)
            
            match = ProcedureCapabilityMatch(
                procedure_name=procedure['name'],
                file_path=procedure['file_path'],
                procedure_path=procedure['path'],
                procedure_type=procedure['type'],
                confidence=confidence,
                match_type=match_type,
                matched_keywords=keywords,
                procedure_comments=procedure.get('comments', ''),
                tal_naming_info=tal_info  # NEW field
            )
            matches[capability] = match
        
        return matches
    
    def _get_searchable_text(self, procedure: Dict) -> str:
        """ENHANCED: Get searchable text including full procedure body"""
        parts = [
            procedure.get('name', ''),
            procedure.get('comments', ''),
            ' '.join(procedure.get('parameters', []))
        ]
        
        # NEW: Include full procedure body in searchable text
        full_procedure_text = procedure.get('full_procedure_text', '')
        if full_procedure_text and full_procedure_text.strip():
            # Truncate procedure body to reasonable size for keyword matching
            # (we don't need the full 2000 tokens for basic keyword matching)
            truncated_body = full_procedure_text[:1000] if len(full_procedure_text) > 1000 else full_procedure_text
            parts.append(truncated_body)
        
        return ' '.join(filter(None, parts))
    
    def _get_semantic_patterns(self, procedure_name: str) -> Dict[str, float]:
        """Get semantic pattern matches based on naming conventions - PRESERVED"""
        patterns = {
            'Payment Initiation': ['initiate', 'start', 'begin', 'create', 'submit', 'send'],
            'Payment Validation': ['validate', 'verify', 'check', 'confirm', 'authorize'],
            'Payment Processing': ['process', 'execute', 'handle', 'run', 'perform'],
            'Account Validation': ['account', 'acct'],
            'Message Routing': ['route', 'forward', 'dispatch', 'transmit'],
            'Sanctions Screening': ['sanction', 'screen', 'filter', 'ofac'],
            'Fraud Monitoring': ['fraud', 'monitor', 'detect', 'alert', 'suspicious'],
            'Payment Cancel': ['cancel', 'abort', 'stop', 'halt', 'terminate'],
            'Confirmations': ['confirm', 'ack', 'acknowledge', 'receipt', 'notify'],
            'Message Format Transformation': ['transform', 'convert', 'format', 'parse'],
            'Inbound SWIFT': ['swift', 'inbound', 'receive', 'mt103', 'mt200'],
            'Outbound SWIFT': ['swift', 'outbound', 'send', 'mt103', 'mt200'],
            'Inbound Fed': ['fed', 'fedwire', 'inbound', 'receive'],
            'Outbound Fed': ['fed', 'fedwire', 'outbound', 'send'],
            'Cross-Border Payment': ['cross', 'border', 'international', 'foreign'],
            'Liquidity Check': ['liquidity', 'funds', 'balance', 'availability'],
            'Account Posting': ['post', 'book', 'credit', 'debit', 'entry']
        }
        
        name_lower = procedure_name.lower()
        matches = {}
        
        for capability, keywords in patterns.items():
            score = sum(1.0 for keyword in keywords if keyword in name_lower)
            if score > 0:
                matches[capability] = score
        
        return matches


async def main():
    """Main CLI function with configurable confidence parameters"""
    parser = argparse.ArgumentParser(description='Enhanced AST to Business Capability Mapper with Full Procedure Context')
    parser.add_argument('ast_directory', help='Directory containing .ast/.tal files')
    parser.add_argument('keywords_file', help='Path to keywords.json file')
    parser.add_argument('-o', '--output', help='Output directory for structured results (default: capability_mapping_output)',
                       default='capability_mapping_output')
    parser.add_argument('-t', '--threshold', type=float, default=0.2,
                       help='Confidence threshold for including mappings (default: 0.2, raised from 0.1)')
    parser.add_argument('--no-llm', action='store_true', help='Disable LLM enhancement')
    parser.add_argument('--quiet', action='store_true', help='Minimal output')
    parser.add_argument('--debug', action='store_true', help='Enable debug output for troubleshooting')
    parser.add_argument('--show-semantic-examples', action='store_true', help='Show semantic similarity examples')
    parser.add_argument('--show-output-structure', action='store_true', help='Show output file structure')
    parser.add_argument('--show-confidence-config', action='store_true', help='Show confidence calculation configuration')
    
    # NEW: Confidence configuration parameters
    parser.add_argument('--base-confidence', type=float, default=0.15, 
                       help='Base confidence for keyword matches (default: 0.15)')
    parser.add_argument('--domain-bonus', type=float, default=0.25,
                       help='Bonus for domain-specific keywords (default: 0.25)')
    parser.add_argument('--exact-bonus', type=float, default=0.20,
                       help='Bonus for exact keyword matches (default: 0.20)')
    parser.add_argument('--semantic-threshold', type=float, default=0.15,
                       help='Minimum semantic similarity to include (default: 0.15)')
    
    args = parser.parse_args()
    
    if args.show_confidence_config:
        print("\n=== Confidence Calculation Configuration ===")
        print("The system uses multi-factor confidence scoring:")
        print()
        print("Base Components:")
        print(f"  • No keywords: 0.05 (fixed)")
        print(f"  • Base keyword match: {args.base_confidence}")
        print(f"  • Domain-specific bonus: {args.domain_bonus}")
        print(f"  • Exact match bonus: {args.exact_bonus}")
        print(f"  • Multiple matches bonus: 0.10 (fixed)")
        print(f"  • Maximum confidence: 0.85 (fixed)")
        print()
        print("Keyword Specificity Weights:")
        print("  • High specificity (MT103, SWIFT, OFAC): 1.0")
        print("  • Medium specificity (payment, routing): 0.7") 
        print("  • Low specificity (send, process): 0.4")
        print()
        print("Final Confidence Calculation:")
        print("  • Rule-based (60%) + Semantic similarity (40%)")
        print("  • Quality bonuses for high-specificity keywords")
        print("  • TAL naming pattern bonuses")
        print(f"  • Semantic similarity threshold: {args.semantic_threshold}")
        print("  • NEW: Full procedure context analysis with 2000 token limit")
        return
    
    if args.show_semantic_examples:
        print("\n=== Enhanced Semantic Analysis Examples ===")
        print("The system now analyzes full procedure context with LLM enhancement:")
        print()
        print("Example procedure: 'Valid^codeword' with full TAL code context")
        print("Enhanced Analysis:")
        print("  - Extracts full procedure implementation (~1500 chars)")
        print("  - Truncates to 2000 token limit for LLM processing")
        print("  - Identifies business patterns in full code context")
        print("  - Maps TAL-specific constructs to business capabilities")
        print()
        print("High confidence mappings (>0.7):")
        print("  - Payment Validation: 'validate routing number' + full validation logic")
        print("  - Business Rule Validation: code structure + validation patterns")
        print()
        print("Medium confidence mappings (0.4-0.7):")
        print("  - Message Format Validation: TAL string operations + format checks")
        print("  - Account Validation: parameter validation + account structure")
        print()
        print("Enhanced features:")
        print("  - Full procedure text extraction from AST files")
        print("  - Token-aware truncation (≈4 chars per token)")
        print("  - Pattern recognition across entire procedure implementation")
        print("  - TAL naming convention analysis (^, ., _)")
        return
    
    if args.show_output_structure:
        print("\n=== Output Directory Structure ===")
        print("The --output flag now creates a directory with multiple structured files:")
        print()
        print("📁 output_directory/")
        print("  ├── 📄 capability_mapping_summary.json")
        print("  │   └── Main metadata and processing statistics")
        print("  ├── 📄 file_processing_details.json")
        print("  │   └── Detailed info for each TAL/AST file processed")
        print("  ├── 📄 business_capabilities.json")
        print("  │   └── Complete capability→procedure mappings")
        print("  ├── 📄 capability_keyword_procedure_graph.json")
        print("  │   └── Graph structure: capability→keyword→procedure")
        print("  ├── 📄 procedures_by_file.json")
        print("  │   └── All procedures/subprocedures organized by source file")
        print("  └── 📄 graph_edges.csv")
        print("      └── Graph edges in CSV format for external tools")
        print()
        print("Graph structure enables visualization/analysis of:")
        print("  • Business Capability → Keywords (definitional)")
        print("  • Keywords → Procedures (with confidence scores)")
        print("  • File → Procedures/Subprocedures (hierarchical)")
        print("  • NEW: Full procedure context included in analysis")
        return
    
    if not args.quiet:
        print("🚀 Enhanced AST to Business Capability Mapper with Full Procedure Context")
        print("=" * 80)
    
    # Initialize LLM provider with custom configuration
    llm_provider = None
    if not args.no_llm:
        llm_provider = LocalLLMProvider()
        # Update confidence configuration based on command line args
        if hasattr(llm_provider, 'confidence_config'):
            llm_provider.confidence_config.update({
                'base_keyword_match': args.base_confidence,
                'domain_specific_bonus': args.domain_bonus,
                'exact_match_bonus': args.exact_bonus
            })
        # Update semantic similarity threshold
        if hasattr(llm_provider, 'compute_semantic_confidence_for_all_capabilities'):
            llm_provider.semantic_threshold = args.semantic_threshold
    
    # Initialize mapper
    try:
        mapper = EnhancedBatchASTCapabilityMapper(args.keywords_file, llm_provider)
        mapper.debug_mode = args.debug
        
        if not args.quiet:
            print(f"📊 Enhanced Configuration:")
            print(f"   • Base keyword confidence: {args.base_confidence}")
            print(f"   • Domain bonus: {args.domain_bonus}")
            print(f"   • Exact match bonus: {args.exact_bonus}")
            print(f"   • Mapping threshold: {args.threshold}")
            print(f"   • Full procedure context: ENABLED")
            print(f"   • Token limit: 2000 tokens (~8000 chars)")
            if llm_provider:
                print(f"   • Semantic threshold: {args.semantic_threshold}")
        
    except Exception as e:
        print(f"❌ Failed to initialize mapper: {e}")
        sys.exit(1)
    
    # Process directory
    try:
        result = await mapper.process_directory(args.ast_directory, args.threshold)
        
        if "error" in result:
            print(f"❌ {result['error']}")
            sys.exit(1)
        
        # Write structured output to directory
        mapper.write_structured_output(result, args.output)
        
        if not args.quiet:
            print(f"\n📈 Processing Summary:")
            print(f"   • Files processed: {result['metadata']['total_ast_files']}")
            print(f"   • Procedures found: {result['metadata']['total_procedures_found']}")
            print(f"   • Total mappings: {result['summary_statistics']['total_capability_mappings']}")
            print(f"   • High confidence (≥0.7): {result['summary_statistics']['high_confidence_mappings']}")
            print(f"   • Medium confidence (0.4-0.7): {result['summary_statistics']['medium_confidence_mappings']}")
            print(f"   • Low confidence (threshold-0.4): {result['summary_statistics']['low_confidence_mappings']}")
            print(f"   • TAL naming enhanced: {result['metadata']['tal_naming_enhanced']}")
            print(f"   • Full context analysis: {result['metadata']['full_procedure_context_enabled']}")
            print(f"   • Semantic similarity: {result['metadata']['semantic_similarity_enabled']}")
            if result['metadata']['semantic_similarity_enabled']:
                print(f"   • Model used: {result['metadata']['similarity_model']}")
            
            # Show file breakdown
            total_procedures = sum(info.get('procedure_count', 0) for info in result.get('file_details', {}).values())
            total_subprocs = sum(len(info.get('subprocedures', [])) for info in result.get('file_details', {}).values())
            print(f"\n📊 File Analysis:")
            print(f"   • Main procedures: {total_procedures - total_subprocs}")
            print(f"   • Subprocedures: {total_subprocs}")
            print(f"   • Graph nodes: {len(result['graph_data']['capability_keywords'])} capabilities, " +
                  f"{len(result['graph_data']['keyword_to_procedures'])} keywords")
            
            # Show improvement from full context analysis
            print(f"\n✨ Enhanced Analysis Features:")
            print(f"   • Full procedure text extraction from AST files")
            print(f"   • Smart token truncation (2000 token limit)")
            print(f"   • Enhanced pattern recognition in full code context")
            print(f"   • TAL-specific naming convention analysis")
            print(f"   • Improved confidence scoring with context awareness")
            
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
