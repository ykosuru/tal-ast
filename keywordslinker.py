#!/usr/bin/env python3
"""
AST to Business Capability Mapper
Processes directory of AST files and maps to business capabilities
Output: JSON organized by business capability with procedures and confidence scores
"""

import json
import re
import sys
import argparse
import os
import glob
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import ast
import asyncio
from abc import ABC, abstractmethod
from pathlib import Path
import time

@dataclass
class ProcedureCapabilityMatch:
    """Individual procedure match to a capability"""
    procedure_name: str
    file_path: str
    procedure_path: str
    procedure_type: str
    confidence: float
    match_type: str  # 'exact', 'partial', 'semantic', 'llm_semantic'
    matched_keywords: List[str]
    procedure_comments: str = ""

class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    async def expand_query(self, procedure_info: Dict, context: str) -> Dict:
        """Expand procedure information with semantic understanding"""
        pass

class LocalLLMProvider(LLMProvider):
    """Local LLM provider using sentence transformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load sentence transformer model"""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            print(f"✓ Loaded local LLM model: {self.model_name}")
        except ImportError:
            print("⚠️  Warning: sentence-transformers not available. Install with: pip install sentence-transformers")
            print("   Falling back to rule-based matching only.")
    
    async def expand_query(self, procedure_info: Dict, context: str) -> Dict:
        """Rule-based query expansion with domain knowledge"""
        name = procedure_info.get('name', '').lower()
        comments = procedure_info.get('comments', '').lower()
        params = ' '.join(procedure_info.get('parameters', [])).lower()
        
        full_text = f"{name} {comments} {params}"
        
        business_keywords = []
        processes = []
        
        # Financial payments domain patterns
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
            if any(kw in full_text for kw in keywords):
                business_keywords.extend(keywords[:3])  # Top 3 relevant keywords
        
        return {
            "business_keywords": list(set(business_keywords)),
            "confidence": 0.7 if business_keywords else 0.3
        }

class BatchASTCapabilityMapper:
    def __init__(self, keywords_file_path: str, llm_provider: Optional[LLMProvider] = None):
        """Initialize batch mapper with keywords configuration"""
        print(f"📖 Loading keywords from: {keywords_file_path}")
        
        with open(keywords_file_path, 'r') as f:
            self.keywords_data = json.load(f)
        
        print(f"✓ Loaded {len(self.keywords_data)} keyword definitions")
        
        self.llm_provider = llm_provider or LocalLLMProvider()
        
        # Get all unique capabilities first
        self.all_capabilities = set()
        for item in self.keywords_data:
            self.all_capabilities.update(item.get('business_capability', []))
        
        # Build lookup maps
        self.capability_to_keywords = self._build_capability_keyword_map()
        self.keyword_to_capabilities = self._build_keyword_capability_map()
        self.business_context = self._build_business_context()
        
        print(f"✓ Built indexes for {len(self.all_capabilities)} capabilities")
    
    def _build_capability_keyword_map(self) -> Dict[str, Set[str]]:
        """Build map from capability to all associated keywords"""
        capability_map = defaultdict(set)
        
        for item in self.keywords_data:
            capabilities = item.get('business_capability', [])
            
            all_keywords = set()
            all_keywords.update(self._normalize_keywords(item.get('keywords', '')))
            all_keywords.update(self._normalize_keywords(item.get('related_keywords', '')))
            all_keywords.update(self._normalize_keywords(item.get('description', '')))
            
            for capability in capabilities:
                capability_map[capability].update(all_keywords)
                
        return capability_map
    
    def _build_keyword_capability_map(self) -> Dict[str, Set[str]]:
        """Build map from keyword to all associated capabilities"""
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
        """Build business context string"""
        return f"Financial payments system with {len(self.all_capabilities)} business capabilities"
    
    def _normalize_keywords(self, keyword_string: str) -> List[str]:
        """Normalize and split keyword strings"""
        if not keyword_string:
            return []
        
        keywords = [kw.strip().lower() for kw in keyword_string.split(',')]
        
        expanded_keywords = []
        for kw in keywords:
            expanded_keywords.append(kw)
            expanded_keywords.append(kw.replace(' ', ''))
            expanded_keywords.append(kw.replace(' ', '_'))
            expanded_keywords.append(kw.replace(' ', '-'))
            expanded_keywords.extend(kw.split())
            
        return list(set(expanded_keywords))
    
    def find_ast_files(self, directory_path: str) -> List[str]:
        """Find all .ast files in directory"""
        directory = Path(directory_path)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        # Find .ast files
        ast_files = list(directory.glob("*.ast"))
        
        if not ast_files:
            print(f"⚠️  No .ast files found in {directory_path}")
            return []
        
        ast_file_paths = [str(f) for f in ast_files]
        print(f"✓ Found {len(ast_file_paths)} .ast files")
        
        return ast_file_paths
    
    def extract_procedures_from_ast(self, file_path: str) -> List[Dict]:
        """Extract procedures from a single AST file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"⚠️  Error reading {file_path}: {e}")
            return []
        
        procedures = []
        
        # Try JSON AST first
        try:
            ast_data = json.loads(content)
            procedures = self._extract_from_json_ast(ast_data, file_path)
        except json.JSONDecodeError:
            # Try Python AST
            try:
                tree = ast.parse(content)
                procedures = self._extract_from_python_ast(tree, file_path)
            except SyntaxError:
                # Fallback: regex extraction
                procedures = self._extract_using_regex(content, file_path)
        
        return procedures
    
    def _extract_from_json_ast(self, ast_data: Dict, file_path: str) -> List[Dict]:
        """Extract procedures from JSON AST"""
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
        """Extract procedures from Python AST"""
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
    
    def _extract_using_regex(self, content: str, file_path: str) -> List[Dict]:
        """Enhanced regex extraction with TAL AST support"""
        procedures = []
        
        patterns = [
            # S-expression patterns (for your TAL AST format) - ADD THESE!
            (r'\(procedure\s+:name\s+([^\s)]+)', 'sexp_procedure'),
            (r'\(proc\s+:name\s+([^\s)]+)', 'sexp_proc'),
            (r'\(function\s+:name\s+([^\s)]+)', 'sexp_function'),
            
            # Original patterns
            (r'def\s+(\w+)\s*\(([^)]*)\)', 'python_function'),
            (r'function\s+(\w+)\s*\(([^)]*)\)', 'javascript_function'),
            (r'procedure\s+(\w+)\s*\(([^)]*)\)', 'sql_procedure'),
            (r'(\w+)\s*:\s*function\s*\(([^)]*)\)', 'object_method'),
        ]
        
        for pattern, proc_type in patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                name = match.group(1)
                params_str = match.group(2) if len(match.groups()) > 1 else ""
                params = [p.strip() for p in params_str.split(',') if p.strip()]
                
                # For S-expression procedures, extract parameters differently
                if proc_type.startswith('sexp_'):
                    params = self._extract_sexp_parameters(content, name)
                    comments = self._extract_sexp_comments(content, name)
                else:
                    comments = ''
                
                proc_info = {
                    'name': name,
                    'file_path': file_path,
                    'path': f"regex_match_{match.start()}",
                    'type': proc_type,
                    'parameters': params,
                    'comments': comments,
                }
                procedures.append(proc_info)
                
        return procedures

    def _extract_sexp_parameters(self, content: str, proc_name: str) -> List[str]:
        """Extract parameters from S-expression procedure"""
        proc_pattern = rf'\(procedure\s+:name\s+{re.escape(proc_name)}.*?\(parameters.*?\)'
        proc_match = re.search(proc_pattern, content, re.DOTALL | re.IGNORECASE)
        
        if proc_match:
            proc_block = proc_match.group(0)
            param_pattern = r'\(parameter\s+:name\s+([^\s)]+)'
            return re.findall(param_pattern, proc_block, re.IGNORECASE)
        return []

    def _extract_sexp_comments(self, content: str, proc_name: str) -> str:
        """Extract comments from S-expression procedure"""
        comment_pattern = r'\(comment\s+:value\s+([^)]+)\)'
        comments = re.findall(comment_pattern, content, re.IGNORECASE)
        return ' '.join(comments)
    
    async def map_procedure_to_capabilities(self, procedure: Dict) -> Dict[str, ProcedureCapabilityMatch]:
        """Map procedure to all capabilities with confidence scores"""
        # Get searchable text
        searchable_text = self._get_searchable_text(procedure)
        normalized_text = ' '.join(self._normalize_keywords(searchable_text)).lower()
        
        # Score all capabilities
        capability_scores = defaultdict(lambda: {'score': 0, 'keywords': [], 'match_type': 'none'})
        
        # 1. Direct keyword matching
        for keyword, capabilities in self.keyword_to_capabilities.items():
            if keyword in normalized_text:
                for capability in capabilities:
                    capability_scores[capability]['score'] += 2.0
                    capability_scores[capability]['keywords'].append(keyword)
                    capability_scores[capability]['match_type'] = 'exact'
        
        # 2. Partial keyword matching
        for capability, keywords in self.capability_to_keywords.items():
            for keyword in keywords:
                parts = [part for part in keyword.split() if len(part) > 2]
                if any(part in normalized_text for part in parts):
                    capability_scores[capability]['score'] += 0.5
                    capability_scores[capability]['keywords'].append(f"partial:{keyword}")
                    if capability_scores[capability]['match_type'] == 'none':
                        capability_scores[capability]['match_type'] = 'partial'
        
        # 3. Semantic pattern matching
        semantic_matches = self._get_semantic_patterns(procedure['name'])
        for capability, score in semantic_matches.items():
            capability_scores[capability]['score'] += score
            capability_scores[capability]['keywords'].append(f"pattern:{procedure['name']}")
            if capability_scores[capability]['match_type'] == 'none':
                capability_scores[capability]['match_type'] = 'semantic'
        
        # 4. LLM enhancement
        if self.llm_provider:
            try:
                llm_expansion = await self.llm_provider.expand_query(procedure, self.business_context)
                for keyword in llm_expansion.get('business_keywords', []):
                    if keyword.lower() in self.keyword_to_capabilities:
                        for capability in self.keyword_to_capabilities[keyword.lower()]:
                            capability_scores[capability]['score'] += 1.0
                            capability_scores[capability]['keywords'].append(f"llm:{keyword}")
                            capability_scores[capability]['match_type'] = 'llm_semantic'
            except Exception:
                pass  # Silently continue if LLM fails
        
        # Convert to ProcedureCapabilityMatch objects
        matches = {}
        for capability in self.all_capabilities:
            score_data = capability_scores[capability]
            confidence = min(score_data['score'] / 3.0, 1.0)  # Normalize to 0-1
            
            match = ProcedureCapabilityMatch(
                procedure_name=procedure['name'],
                file_path=procedure['file_path'],
                procedure_path=procedure['path'],
                procedure_type=procedure['type'],
                confidence=confidence,
                match_type=score_data['match_type'],
                matched_keywords=score_data['keywords'][:5],  # Top 5 keywords
                procedure_comments=procedure.get('comments', '')
            )
            matches[capability] = match
        
        return matches
    
    def _get_searchable_text(self, procedure: Dict) -> str:
        """Get searchable text from procedure"""
        parts = [
            procedure.get('name', ''),
            procedure.get('comments', ''),
            ' '.join(procedure.get('parameters', []))
        ]
        return ' '.join(filter(None, parts))
    
    def _get_semantic_patterns(self, procedure_name: str) -> Dict[str, float]:
        """Get semantic pattern matches based on naming conventions"""
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
    
    async def process_directory(self, directory_path: str, confidence_threshold: float = 0.1) -> Dict:
        """Process all AST files in directory with comprehensive error handling"""
        print(f"🔍 Processing AST files in: {directory_path}")
        
        try:
            # Find all AST files with validation
            ast_files = self.find_ast_files(directory_path)
            if not ast_files:
                return {
                    "error": "No valid .ast files found in directory",
                    "metadata": {
                        "source_directory": directory_path,
                        "total_ast_files": 0,
                        "total_procedures_found": 0,
                        "error_details": "Directory contained no readable .ast files"
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
        
        print(f"📁 Processing {len(ast_files)} files...")
        
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
                    print(f"      ⚠️  No procedures found in {file_name}")
                else:
                    print(f"      ✓ Found {len(procedures)} procedures")
                
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
        
        print(f"✓ Extracted {len(all_procedures)} total procedures from {len(ast_files)} files")
        if failed_files:
            print(f"⚠️  Failed to process {len(failed_files)} files")
        
        # Map all procedures to capabilities with batch processing
        print(f"🔄 Mapping procedures to capabilities...")
        
        # Organize results by capability
        capability_results = {}
        
        # Initialize all capabilities
        for capability in sorted(self.all_capabilities):
            capability_results[capability] = {
                "total_procedures": 0,
                "high_confidence_procedures": 0,
                "medium_confidence_procedures": 0,
                "low_confidence_procedures": 0,
                "procedures": []
            }
        
        # Process procedures in batches for better performance and error handling
        batch_size = 20  # Smaller batches for better error isolation
        total_batches = (len(all_procedures) + batch_size - 1) // batch_size
        successful_mappings = 0
        failed_mappings = 0
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(all_procedures))
            batch = all_procedures[start_idx:end_idx]
            
            print(f"   Progress: Batch {batch_idx + 1}/{total_batches} ({len(batch)} procedures)")
            
            # Process batch with individual error handling
            for procedure in batch:
                try:
                    capability_matches = await self.map_procedure_to_capabilities(procedure)
                    
                    # Add to capability results
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
        
        # Build final output with comprehensive error tracking
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
                "llm_enhanced": self.llm_provider is not None
            },
            "file_statistics": file_stats,
            "business_capabilities": capability_results,
            "summary_statistics": {
                "capabilities_with_procedures": sum(1 for cap_data in capability_results.values() if cap_data["total_procedures"] > 0),
                "total_capability_mappings": sum(cap_data["total_procedures"] for cap_data in capability_results.values()),
                "high_confidence_mappings": sum(cap_data["high_confidence_procedures"] for cap_data in capability_results.values()),
                "medium_confidence_mappings": sum(cap_data["medium_confidence_procedures"] for cap_data in capability_results.values()),
                "low_confidence_mappings": sum(cap_data["low_confidence_procedures"] for cap_data in capability_results.values())
            }
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
        if failed_files:
            print(f"   ⚠️  {len(failed_files)} files failed to process")
        if failed_mappings:
            print(f"   ⚠️  {failed_mappings} procedures failed to map")
        
        return output

async def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(description='Batch process AST files to map business capabilities')
    parser.add_argument('ast_directory', help='Directory containing .ast files')
    parser.add_argument('keywords_file', help='Path to keywords.json file')
    parser.add_argument('-o', '--output', help='Output JSON file (default: capability_mapping.json)',
                       default='capability_mapping.json')
    parser.add_argument('-t', '--threshold', type=float, default=0.1,
                       help='Confidence threshold for including mappings (default: 0.1)')
    parser.add_argument('--no-llm', action='store_true', help='Disable LLM enhancement')
    parser.add_argument('--quiet', action='store_true', help='Minimal output')
    parser.add_argument('--debug', action='store_true', help='Enable debug output for troubleshooting')
    
    args = parser.parse_args()
    
    if not args.quiet:
        print("🚀 Batch AST to Business Capability Mapper")
        print("=" * 45)
    
    # Initialize LLM provider
    llm_provider = None if args.no_llm else LocalLLMProvider()
    
    # Initialize mapper
    try:
        mapper = BatchASTCapabilityMapper(args.keywords_file, llm_provider)
        mapper.debug_mode = args.debug  # Enable debug mode if requested
    except Exception as e:
        print(f"❌ Failed to initialize mapper: {e}")
        sys.exit(1)
    
    # Process directory
    try:
        result = await mapper.process_directory(args.ast_directory, args.threshold)
        
        if "error" in result:
            print(f"❌ {result['error']}")
            sys.exit(1)
        
        # Write output
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        if not args.quiet:
            print(f"💾 Output written to: {args.output}")
            print(f"📈 Summary:")
            print(f"   • Files processed: {result['metadata']['total_ast_files']}")
            print(f"   • Procedures found: {result['metadata']['total_procedures_found']}")
            print(f"   • Total mappings: {result['summary_statistics']['total_capability_mappings']}")
            print(f"   • High confidence: {result['summary_statistics']['high_confidence_mappings']}")
            
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())

