#!/usr/bin/env python3
"""
Clean AST to Business Capability Mapper with Full Procedure Context
"""

import json
import re
import sys
import argparse
import os
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict, Counter
import asyncio
from abc import ABC, abstractmethod
from pathlib import Path
import time
import math

@dataclass
class ProcedureCapabilityMatch:
    procedure_name: str
    file_path: str
    procedure_path: str
    procedure_type: str
    confidence: float
    match_type: str
    matched_keywords: List[str]
    procedure_comments: str = ""

class LLMProvider(ABC):
    @abstractmethod
    async def expand_query(self, procedure_info: Dict, context: str) -> Dict:
        """Expand procedure information with semantic understanding"""
        pass

class LocalLLMProvider(LLMProvider):
    def __init__(self):
        print("Initialized LocalLLMProvider (add your LLM integration here)")
    
    async def expand_query(self, procedure_info: Dict, context: str) -> Dict:
        """Enhanced query expansion with full procedure context"""
        name = procedure_info.get('name', '')
        full_procedure_text = procedure_info.get('full_procedure_text', '')
        
        print(f"LLM analyzing procedure: {name}")
        print(f"Full procedure text length: {len(full_procedure_text)} chars")
        
        if full_procedure_text:
            print(f"Procedure preview: {full_procedure_text[:200]}...")
            
            # TODO: Replace this with your actual LLM API call
            # Example: response = openai.chat.completions.create(...)
            
            # For now, do enhanced pattern matching on full text
            business_keywords = self._extract_keywords_from_full_text(full_procedure_text)
            confidence = 0.8 if business_keywords else 0.3
            
            print(f"Extracted keywords: {business_keywords}")
            return {
                "business_keywords": business_keywords,
                "confidence": confidence
            }
        else:
            print("WARNING: No full procedure text available")
            return {
                "business_keywords": [],
                "confidence": 0.3
            }
    
    def _extract_keywords_from_full_text(self, text: str) -> List[str]:
        """Extract business keywords from full procedure text"""
        keywords = []
        text_lower = text.lower()
        
        # Financial domain patterns
        patterns = {
            'validation': r'\b(validat|verify|check|confirm|authoriz)\w*',
            'payment': r'\b(payment|transfer|wire|remit)\w*',
            'routing': r'\b(rout|forward|dispatch)\w*',
            'wire': r'\b(wire|swift|fedwire|chips|mt103)\w*',
            'sanctions': r'\b(sanction|ofac|aml|screen)\w*',
            'fraud': r'\b(fraud|monitor|detect|alert)\w*',
            'settlement': r'\b(settl|clear|post|book)\w*',
            'account': r'\b(account|acct|credit|debit)\w*'
        }
        
        for category, pattern in patterns.items():
            if re.search(pattern, text_lower):
                keywords.append(category)
        
        return keywords

class ASTCapabilityMapper:
    def __init__(self, keywords_file_path: str, llm_provider: Optional[LLMProvider] = None):
        print(f"Loading keywords from: {keywords_file_path}")
        
        with open(keywords_file_path, 'r') as f:
            self.keywords_data = json.load(f)
        
        print(f"Loaded {len(self.keywords_data)} keyword definitions")
        
        self.llm_provider = llm_provider or LocalLLMProvider()
        
        # Build capability mappings
        self.all_capabilities = set()
        for item in self.keywords_data:
            self.all_capabilities.update(item.get('business_capability', []))
        
        self.capability_to_keywords = self._build_capability_keyword_map()
        self.keyword_to_capabilities = self._build_keyword_capability_map()
        
        print(f"Built indexes for {len(self.all_capabilities)} capabilities")
    
    def _build_capability_keyword_map(self) -> Dict[str, Set[str]]:
        capability_map = defaultdict(set)
        
        for item in self.keywords_data:
            capabilities = item.get('business_capability', [])
            
            all_keywords = set()
            all_keywords.update(self._normalize_keywords(item.get('keywords', '')))
            all_keywords.update(self._normalize_keywords(item.get('related_keywords', '')))
            
            for capability in capabilities:
                capability_map[capability].update(all_keywords)
                
        return capability_map
    
    def _build_keyword_capability_map(self) -> Dict[str, Set[str]]:
        keyword_map = defaultdict(set)
        
        for item in self.keywords_data:
            capabilities = item.get('business_capability', [])
            
            all_keywords = set()
            all_keywords.update(self._normalize_keywords(item.get('keywords', '')))
            all_keywords.update(self._normalize_keywords(item.get('related_keywords', '')))
            
            for keyword in all_keywords:
                keyword_map[keyword].update(capabilities)
                
        return keyword_map
    
    def _normalize_keywords(self, keyword_string: str) -> List[str]:
        if not keyword_string:
            return []
        
        keywords = [kw.strip().lower() for kw in keyword_string.split(',')]
        
        expanded_keywords = []
        for kw in keywords:
            expanded_keywords.append(kw)
            expanded_keywords.append(kw.replace(' ', ''))
            expanded_keywords.append(kw.replace(' ', '_'))
            expanded_keywords.extend(kw.split())
            
        return list(set(expanded_keywords))
    
    def find_ast_files(self, directory_path: str) -> List[str]:
        directory = Path(directory_path)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        ast_files = list(directory.glob("*.ast")) + list(directory.glob("*.tal"))
        
        if not ast_files:
            print(f"No .ast or .tal files found in {directory_path}")
            return []
        
        ast_file_paths = [str(f) for f in ast_files]
        print(f"Found {len(ast_file_paths)} AST/TAL files")
        
        return ast_file_paths
    
    def extract_procedures_from_ast(self, file_path: str) -> List[Dict]:
        """Extract procedures and their full text from AST file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return []
        
        procedures = []
        
        # Try S-expression parsing first (for TAL AST files)
        sexp_procedures = self._extract_sexp_procedures(content, file_path)
        if sexp_procedures:
            procedures.extend(sexp_procedures)
        else:
            # Fallback to other formats
            procedures.extend(self._extract_other_procedures(content, file_path))
        
        print(f"Extracted {len(procedures)} procedures from {Path(file_path).name}")
        return procedures
    
    def _extract_sexp_procedures(self, content: str, file_path: str) -> List[Dict]:
        """Extract procedures from S-expression format (TAL AST)"""
        procedures = []
        
        # Find all procedure definitions
        procedure_pattern = r'\(procedure\s+:name\s+([^\s)]+)'
        matches = list(re.finditer(procedure_pattern, content, re.IGNORECASE))
        
        for match in matches:
            proc_name = match.group(1)
            start_pos = match.start()
            
            # Extract full procedure block using parenthesis matching
            full_procedure_text = self._extract_procedure_block(content, start_pos)
            
            # Extract parameters from the procedure block
            params = self._extract_parameters(full_procedure_text)
            
            # Extract comments from the procedure block
            comments = self._extract_comments(full_procedure_text)
            
            procedure = {
                'name': proc_name,
                'file_path': file_path,
                'path': f"sexp_{start_pos}",
                'type': 'tal_procedure',
                'parameters': params,
                'comments': comments,
                'full_procedure_text': full_procedure_text  # KEY: Include full text
            }
            
            procedures.append(procedure)
            print(f"  Found procedure: {proc_name} ({len(full_procedure_text)} chars)")
        
        return procedures
    
    def _extract_procedure_block(self, content: str, start_pos: int) -> str:
        """Extract complete procedure block using parenthesis matching"""
        paren_count = 0
        
        for i, char in enumerate(content[start_pos:], start_pos):
            if char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
                if paren_count == 0:
                    return content[start_pos:i+1]
        
        # Fallback: take reasonable chunk
        return content[start_pos:start_pos+2000]
    
    def _extract_parameters(self, procedure_text: str) -> List[str]:
        """Extract parameter names from procedure text"""
        param_pattern = r'\(parameter\s+:name\s+([^\s)]+)'
        return re.findall(param_pattern, procedure_text, re.IGNORECASE)
    
    def _extract_comments(self, procedure_text: str) -> str:
        """Extract comments from procedure text"""
        comment_pattern = r'\(comment\s+:value\s+([^)]+)\)'
        comments = re.findall(comment_pattern, procedure_text, re.IGNORECASE)
        return ' '.join(comments)
    
    def _extract_other_procedures(self, content: str, file_path: str) -> List[Dict]:
        """Extract procedures from other formats (Python, JS, etc.)"""
        procedures = []
        
        patterns = [
            (r'def\s+(\w+)\s*\(([^)]*)\)', 'python_function'),
            (r'function\s+(\w+)\s*\(([^)]*)\)', 'javascript_function'),
            (r'procedure\s+(\w+)\s*\(([^)]*)\)', 'sql_procedure'),
        ]
        
        for pattern, proc_type in patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                name = match.group(1)
                params_str = match.group(2) if len(match.groups()) > 1 else ""
                params = [p.strip() for p in params_str.split(',') if p.strip()]
                
                # Extract reasonable context around the procedure
                start_pos = match.start()
                end_pos = min(start_pos + 1500, len(content))
                full_text = content[start_pos:end_pos]
                
                procedure = {
                    'name': name,
                    'file_path': file_path,
                    'path': f"regex_{start_pos}",
                    'type': proc_type,
                    'parameters': params,
                    'comments': '',
                    'full_procedure_text': full_text
                }
                procedures.append(procedure)
        
        return procedures
    
    async def map_procedure_to_capabilities(self, procedure: Dict) -> Dict[str, ProcedureCapabilityMatch]:
        """Map procedure to business capabilities using rule-based + LLM analysis"""
        
        # Get searchable text (includes full procedure body)
        searchable_text = self._get_searchable_text(procedure)
        normalized_text = ' '.join(self._normalize_keywords(searchable_text)).lower()
        
        # Rule-based scoring
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
        
        # 3. LLM enhancement
        llm_capability_scores = {}
        if self.llm_provider:
            try:
                llm_result = await self.llm_provider.expand_query(procedure, "financial payments")
                llm_confidence = llm_result.get('confidence', 0.0)
                llm_keywords = llm_result.get('business_keywords', [])
                
                # Map LLM keywords to capabilities
                for keyword in llm_keywords:
                    if keyword.lower() in self.keyword_to_capabilities:
                        for capability in self.keyword_to_capabilities[keyword.lower()]:
                            if capability not in llm_capability_scores:
                                llm_capability_scores[capability] = {
                                    'confidence': llm_confidence,
                                    'keywords': [],
                                    'match_type': 'llm_semantic'
                                }
                            llm_capability_scores[capability]['keywords'].append(f"llm:{keyword}")
                
            except Exception as e:
                print(f"LLM analysis failed for {procedure['name']}: {e}")
        
        # Create final matches
        matches = {}
        for capability in self.all_capabilities:
            # Start with rule-based score
            score_data = capability_scores[capability]
            confidence = min(score_data['score'] / 3.0, 1.0)
            match_type = score_data['match_type']
            keywords = score_data['keywords'][:5]
            
            # Use LLM confidence if available
            if capability in llm_capability_scores:
                llm_data = llm_capability_scores[capability]
                confidence = llm_data['confidence']
                match_type = llm_data['match_type']
                keywords = list(set(keywords + llm_data['keywords']))[:5]
            
            match = ProcedureCapabilityMatch(
                procedure_name=procedure['name'],
                file_path=procedure['file_path'],
                procedure_path=procedure['path'],
                procedure_type=procedure['type'],
                confidence=confidence,
                match_type=match_type,
                matched_keywords=keywords,
                procedure_comments=procedure.get('comments', '')
            )
            matches[capability] = match
        
        return matches
    
    def _get_searchable_text(self, procedure: Dict) -> str:
        """Get searchable text including full procedure body"""
        parts = [
            procedure.get('name', ''),
            procedure.get('comments', ''),
            ' '.join(procedure.get('parameters', []))
        ]
        
        # Include full procedure body
        full_text = procedure.get('full_procedure_text', '')
        if full_text:
            # Use first 1000 chars for keyword matching
            truncated_body = full_text[:1000] if len(full_text) > 1000 else full_text
            parts.append(truncated_body)
        
        return ' '.join(filter(None, parts))
    
    async def process_directory(self, directory_path: str, confidence_threshold: float = 0.2) -> Dict:
        """Process all AST files in directory"""
        print(f"Processing AST files in: {directory_path}")
        
        # Find AST files
        ast_files = self.find_ast_files(directory_path)
        if not ast_files:
            return {"error": "No AST files found"}
        
        # Extract all procedures
        all_procedures = []
        for ast_file in ast_files:
            procedures = self.extract_procedures_from_ast(ast_file)
            all_procedures.extend(procedures)
        
        print(f"Total procedures extracted: {len(all_procedures)}")
        
        # Map procedures to capabilities
        capability_results = {cap: {"procedures": [], "total_procedures": 0} 
                            for cap in self.all_capabilities}
        
        for procedure in all_procedures:
            print(f"\nAnalyzing: {procedure['name']}")
            matches = await self.map_procedure_to_capabilities(procedure)
            
            for capability, match in matches.items():
                if match.confidence >= confidence_threshold:
                    capability_results[capability]["procedures"].append({
                        "procedure_name": match.procedure_name,
                        "file_path": os.path.basename(match.file_path),
                        "confidence": round(match.confidence, 3),
                        "match_type": match.match_type,
                        "matched_keywords": match.matched_keywords
                    })
                    capability_results[capability]["total_procedures"] += 1
        
        # Sort by confidence
        for cap_data in capability_results.values():
            cap_data["procedures"].sort(key=lambda x: x["confidence"], reverse=True)
        
        return {
            "metadata": {
                "total_files": len(ast_files),
                "total_procedures": len(all_procedures),
                "confidence_threshold": confidence_threshold,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "capabilities": capability_results
        }
    
    def write_results(self, results: Dict, output_path: str):
        """Write results to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results written to: {output_path}")

async def main():
    parser = argparse.ArgumentParser(description='Clean AST to Business Capability Mapper')
    parser.add_argument('ast_directory', help='Directory containing .ast/.tal files')
    parser.add_argument('keywords_file', help='Path to keywords.json file')
    parser.add_argument('-o', '--output', help='Output JSON file', default='capability_mapping.json')
    parser.add_argument('-t', '--threshold', type=float, default=0.2, help='Confidence threshold')
    
    args = parser.parse_args()
    
    try:
        # Initialize mapper
        mapper = ASTCapabilityMapper(args.keywords_file)
        
        # Process directory
        results = await mapper.process_directory(args.ast_directory, args.threshold)
        
        if "error" in results:
            print(f"Error: {results['error']}")
            sys.exit(1)
        
        # Write results
        mapper.write_results(results, args.output)
        
        # Summary
        print(f"\nSummary:")
        print(f"Files processed: {results['metadata']['total_files']}")
        print(f"Procedures found: {results['metadata']['total_procedures']}")
        
        capabilities_with_procedures = sum(1 for cap in results['capabilities'].values() 
                                         if cap['total_procedures'] > 0)
        print(f"Capabilities with procedures: {capabilities_with_procedures}")
        
    except Exception as e:
        print(f"Processing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
