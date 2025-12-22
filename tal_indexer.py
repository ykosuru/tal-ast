#!/usr/bin/env python3
"""
TAL Code Intelligence Indexer v2.0
===================================
Enhanced with patterns from ANTLR-based Java parser

For HPE NonStop Tandem TAL (Transaction Application Language)

Two-Pass Architecture:
1. Extract all symbols (STRUCTs, DEFINEs, LITERALs, variables)
2. Extract procedures and find references with global context

Embedding Options (all optional):
- Set EMBEDDING_API_URL for OpenAI-compatible API
- Or run without embeddings (index still works, just no vector search)

Usage:
    python tal_indexer_v2.py --code-dir /path/to/tal --db-path ./tal_index
    #YK123
"""

import re
import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Set
from collections import defaultdict

# Optional: requests for OpenAI-compatible API
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# Optional: HuggingFace transformers
HAS_TRANSFORMERS = False
try:
    import torch
    from transformers import AutoModel, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    pass

# Optional: Qdrant
HAS_QDRANT = False
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    HAS_QDRANT = True
except ImportError:
    pass


# ============================================================================
# TAL Domain Knowledge
# ============================================================================

TAL_KEYWORDS = {
    # Control flow
    "IF", "THEN", "ELSE", "DO", "WHILE", "UNTIL", "FOR", "CASE", "OF",
    "BEGIN", "END", "RETURN", "CALL", "GOTO", "SCAN", "RSCAN",
    "OTHERWISE", "DOWNTO", "BY", "TO",
    
    # Declarations
    "PROC", "SUBPROC", "STRUCT", "DEFINE", "LITERAL", "BLOCK",
    "INT", "FIXED", "REAL", "STRING", "UNSIGNED", "EXTDECS",
    "FORWARD", "EXTERNAL", "PRIV", "RESIDENT", "EXTENSIBLE",
    "VARIABLE", "MAIN", "INTERRUPT", "CALLABLE", "SHARED", "REENTRANT",
    "BYTE", "CHAR", "TIMESTAMP", "EXTADDR", "SGADDR", "BOOLEAN",
    
    # Operators and built-ins
    "AND", "OR", "NOT", "XOR", "LOR", "LAND", "MOD",
    "BYTES", "WORDS", "ELEMENTS", "BITLENGTH",
    "MOVE", "MOVL", "MOVR", "BITDEPOSIT", "ASSIGN",
    
    # I/O and system
    "READ", "WRITE", "WRITEREAD", "REPLY", "AWAITIO",
    "FILE_OPEN_", "FILE_CLOSE_", "FILE_GETINFO_",
    "PROCESS_CREATE_", "PROCESS_STOP_", "PROCESS_GETINFO_",
    
    # Memory
    "USE", "DROP", "STACK", "STORE",
    
    # Compiler directives (without ?)
    "SOURCE", "SEARCH", "SECTION", "PAGE", "LIST", "NOLIST",
    "CODE", "NOCODE", "CROSSREF", "NOCROSSREF", "SYMBOLS",
    
    # Special values
    "TRUE", "FALSE", "NIL"
}

SYSTEM_PROCEDURES = {
    "WRITEREAD", "READ", "WRITE", "OPEN", "CLOSE",
    "POSITION", "SETMODE", "GETINFO", "PUTINFO", "CONTROL", "NOWAIT",
    "AWAITIO", "FILE_OPEN_", "FILE_CLOSE_", "PROCESS_STOP_",
    "MOVE", "SCAN", "RSCAN", "FILL", "MOVL", "MOVR", "FILEOP"
}

BUSINESS_KEYWORDS = {
    "TRANSACTION", "COMMIT", "ABORT", "BACKOUT", "TMF",
    "VALIDATE", "PROCESS", "CALCULATE", "UPDATE", "INSERT", "DELETE",
    "RETRIEVE", "FETCH", "STORE", "CHECK", "VERIFY", "FORMAT",
    "INITIALIZE", "SETUP", "CLEANUP", "ERROR", "HANDLE", "LOG",
    "SEND", "RECEIVE", "REQUEST", "RESPONSE", "MESSAGE",
    "OPEN", "CLOSE", "READ", "WRITE", "POSITION", "CONTROL"
}


# ============================================================================
# TAL Indexer
# ============================================================================

class TalIndexerV2:
    """
    Production-grade TAL code indexer with patterns from ANTLR Java parser.
    Works with or without embeddings.
    """
    
    def __init__(self, db_path: str = "./tal_index", collection_name: str = "tal"):
        print("🚀 Initializing TAL Indexer v2.0 (Enhanced)...")
        
        # Vector DB path
        self.db_path = db_path
        os.makedirs(db_path, exist_ok=True)
        
        # Collections
        self.logic_coll = f"{collection_name}_logic"
        self.symbol_coll = f"{collection_name}_symbols"
        
        # Initialize embedding model (optional)
        self.model = None
        self.tokenizer = None
        self.qdrant = None
        self.has_embeddings = False
        
        self._init_embeddings()
        
        # Data stores
        self.procedures: List[Dict] = []
        self.symbols: List[Dict] = []
        self.structs: List[Dict] = []
        self.business_rules: List[Dict] = []
        self.all_symbol_names: Set[str] = set()
        self.all_struct_names: Set[str] = set()  # Track struct names separately
        self.all_proc_names: Set[str] = set()
        self.call_graph: Dict[str, Dict] = {}
        self.include_content: Dict[str, str] = {}
        self.data_usage_index: Dict[str, List[Dict]] = defaultdict(list)
        self.preprocessor_defines: Dict[str, str] = {}
        
        # Compile regex patterns
        self._compile_patterns()
        
        # Statistics
        self.stats = defaultdict(int)
    
    def _init_embeddings(self):
        """Initialize embeddings if available"""
        
        # Option 1: OpenAI-compatible API
        api_url = os.environ.get('EMBEDDING_API_URL')
        if api_url and HAS_REQUESTS:
            self.embedding_provider = 'openai_compatible'
            self.api_url = api_url.rstrip('/')
            self.api_key = os.environ.get('EMBEDDING_API_KEY', '')
            self.embedding_model = os.environ.get('EMBEDDING_MODEL', 'text-embedding-ada-002')
            self.embedding_dim = int(os.environ.get('EMBEDDING_DIMENSION', '1536'))
            self.has_embeddings = True
            print(f"   📡 Using OpenAI-compatible embeddings: {self.api_url}")
            self._init_qdrant()
            return
        
        # Option 2: HuggingFace transformers
        if HAS_TRANSFORMERS:
            try:
                model_name = 'sentence-transformers/all-MiniLM-L6-v2'
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name)
                self.model.eval()
                self.embedding_provider = 'transformers'
                self.embedding_dim = 384
                self.has_embeddings = True
                print("   🤗 Using HuggingFace transformers embeddings")
                self._init_qdrant()
                return
            except Exception as e:
                print(f"   ⚠️  Could not load transformers: {e}")
        
        # Option 3: No embeddings
        self.embedding_provider = None
        self.has_embeddings = False
        print("   📚 Indexing without embeddings (JSON only)")
    
    def _init_qdrant(self):
        """Initialize Qdrant vector store"""
        if not HAS_QDRANT:
            print("   ⚠️  Qdrant not installed, skipping vector index")
            self.has_embeddings = False
            return
        
        try:
            self.qdrant = QdrantClient(path=self.db_path)
            
            for coll in [self.logic_coll, self.symbol_coll]:
                if self.qdrant.collection_exists(coll):
                    self.qdrant.delete_collection(coll)
                self.qdrant.create_collection(
                    coll,
                    vectors_config=VectorParams(size=self.embedding_dim, distance=Distance.COSINE)
                )
        except Exception as e:
            print(f"   ⚠️  Could not initialize Qdrant: {e}")
            self.qdrant = None
            self.has_embeddings = False
    
    def _compile_patterns(self):
        """Pre-compile regex patterns"""
        
        # ===== PREPROCESSOR DIRECTIVES =====
        self.re_include = re.compile(
            r'\?(?:SOURCE|SEARCH)\s+([^\s,;!]+)',
            re.IGNORECASE
        )
        
        self.re_define_directive = re.compile(
            r'\?DEFINE\s+([A-Za-z_][A-Za-z0-9_^]*)\s*=?\s*(.*)',
            re.IGNORECASE
        )
        
        # ===== PROCEDURE PATTERNS =====
        # Typed procedure: TYPE PROC name(params) [attrs];
        self.re_typed_proc = re.compile(
            r'^\s*([A-Z_][A-Z0-9_]*(?:\([^)]*\))?)\s+PROC\s+([A-Za-z_][A-Za-z0-9_^]*)\s*\(([^)]*)\)([^;]*);',
            re.IGNORECASE | re.MULTILINE
        )
        
        # Untyped procedure: PROC name(params) [attrs];
        self.re_untyped_proc = re.compile(
            r'^\s*PROC\s+([A-Za-z_][A-Za-z0-9_^]*)\s*\(([^)]*)\)([^;]*);',
            re.IGNORECASE | re.MULTILINE
        )
        
        # ===== DATA DECLARATION PATTERNS =====
        # Standard: TYPE name
        self.re_standard_decl = re.compile(
            r'\b(INT(?:\(\d+\))?|STRING(?:\(\d+\))?|REAL(?:\(\d+\))?|FIXED(?:\([^)]*\))?|BYTE|CHAR|TIMESTAMP|UNSIGNED(?:\(\d+\))?|EXTADDR|SGADDR|BOOLEAN)\s+([A-Za-z_][A-Za-z0-9_^]*)',
            re.IGNORECASE
        )
        
        # Pointer: TYPE .name
        self.re_pointer_decl = re.compile(
            r'\b(INT(?:\(\d+\))?|STRING(?:\(\d+\))?|REAL(?:\(\d+\))?|FIXED(?:\([^)]*\))?|BYTE|CHAR)\s+\.([A-Za-z_][A-Za-z0-9_^]*)',
            re.IGNORECASE
        )
        
        # Array: TYPE name[bounds] or TYPE .name[bounds]
        self.re_array_decl = re.compile(
            r'\b(INT|STRING|REAL|FIXED|BYTE|CHAR)(?:\([^)]*\))?\s+([.*]?)([A-Za-z_][A-Za-z0-9_^]*)\s*\[([^\]]+)\]',
            re.IGNORECASE
        )
        
        # STRUCT definition: STRUCT name;
        self.re_struct_def = re.compile(
            r'\bSTRUCT\s+([A-Za-z_][A-Za-z0-9_^]*)\s*;',
            re.IGNORECASE
        )
        
        # STRUCT instance: STRUCT .name (template)
        self.re_struct_instance = re.compile(
            r'\bSTRUCT\s+\.?([A-Za-z_][A-Za-z0-9_^]*)\s*\(([A-Za-z_][A-Za-z0-9_^]*)\)',
            re.IGNORECASE
        )
        
        # DEFINE: DEFINE name = value
        self.re_define = re.compile(
            r'\bDEFINE\s+([A-Za-z_][A-Za-z0-9_^]*)\s*(?:\([^)]*\))?\s*=',
            re.IGNORECASE
        )
        
        # LITERAL: LITERAL name = value
        self.re_literal = re.compile(
            r'\bLITERAL\s+([A-Za-z_][A-Za-z0-9_^]*)\s*=',
            re.IGNORECASE
        )
        
        # ===== CALL PATTERNS =====
        self.re_call = re.compile(
            r'\bCALL\s+([A-Za-z_][A-Za-z0-9_^]*)',
            re.IGNORECASE
        )
        
        self.re_func_call = re.compile(
            r'\b([A-Za-z_][A-Za-z0-9_^]*)\s*\([^)]*\)',
            re.IGNORECASE
        )
        
        self.re_system_call = re.compile(
            r'\$([A-Za-z_][A-Za-z0-9_]*)',
            re.IGNORECASE
        )
        
        # Comments: -- to end of line, or ! to end of line
        self.re_comment = re.compile(r'(?:--|!).*$', re.MULTILINE)
        
        # END statement
        self.re_end = re.compile(r'\bEND\s*;', re.IGNORECASE)
        
        # ===== BUSINESS RULE PATTERNS =====
        self.re_if_then = re.compile(r'\bIF\s+(.+?)\s+THEN\b', re.IGNORECASE)
        self.re_validation = re.compile(r'\b(VALIDATE|CHECK|VERIFY)\b', re.IGNORECASE)
        self.re_error = re.compile(r'\b(ERROR|ABEND|TRAP)\b', re.IGNORECASE)
    
    def _normalize_tal(self, content: str) -> str:
        """Normalize TAL source code"""
        # Remove comments
        content = self.re_comment.sub('', content)
        
        # Normalize line continuations (& at end of line)
        content = re.sub(r'&\s*\n', ' ', content)
        
        # Join lines that are clearly continuations (parameter lists spanning lines)
        content = re.sub(r',\s*\n\s+', ', ', content)
        
        return content
    
    def _extract_preprocessor(self, content: str, filename: str):
        """Extract preprocessor directives"""
        
        # ?DEFINE directives
        for match in self.re_define_directive.finditer(content):
            name = match.group(1).upper()
            value = match.group(2).strip() if match.group(2) else ""
            self.preprocessor_defines[name] = value
            self.stats['preprocessor_defines'] += 1
        
        # ?SOURCE/SEARCH includes
        for match in self.re_include.finditer(content):
            self.stats['includes'] += 1
    
    def _extract_structs(self, content: str, filename: str) -> List[Dict]:
        """Extract STRUCT definitions including inline parameter structs"""
        structs = []
        lines = content.splitlines()
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Pattern 1: Inline parameter struct - STRUCT NAME(*); BEGIN...END;
            inline_match = re.search(
                r'\bSTRUCT\s+([A-Za-z_][A-Za-z0-9_^]*)\s*\(\s*\*\s*\)\s*;',
                line, re.IGNORECASE
            )
            if inline_match:
                struct_name = inline_match.group(1).upper()
                start_line = i + 1
                
                # Find BEGIN...END block
                struct_content = [line]
                in_struct = False
                end_line = start_line
                
                for j in range(i + 1, min(i + 150, len(lines))):
                    check_line = lines[j].strip().upper()
                    if not check_line or check_line.startswith('!'):
                        struct_content.append(lines[j])
                        continue
                    if 'BEGIN' in check_line:
                        in_struct = True
                    struct_content.append(lines[j])
                    if in_struct and self.re_end.search(lines[j]):
                        end_line = j + 1
                        break
                
                if in_struct:
                    fields = self._extract_struct_fields('\n'.join(struct_content), filename, struct_name)
                    
                    struct = {
                        'name': struct_name,
                        'file': filename,
                        'start_line': start_line,
                        'end_line': end_line,
                        'fields': fields,
                        'inline': True
                    }
                    structs.append(struct)
                    self.all_symbol_names.add(struct_name)
                    self.all_struct_names.add(struct_name)  # Track struct names
                    self.stats['inline_structs'] += 1
                    
                    i = end_line
                    continue
            
            # Pattern 2: Standard STRUCT definition - STRUCT name;
            struct_match = self.re_struct_def.search(line)
            if struct_match:
                struct_name = struct_match.group(1).upper()
                start_line = i + 1
                
                # Find BEGIN...END block
                struct_content = []
                in_struct = False
                end_line = start_line
                
                for j in range(i, min(i + 150, len(lines))):
                    check_line = lines[j].strip().upper()
                    if 'BEGIN' in check_line:
                        in_struct = True
                    struct_content.append(lines[j])
                    if in_struct and self.re_end.search(lines[j]):
                        end_line = j + 1
                        break
                
                if in_struct:
                    fields = self._extract_struct_fields('\n'.join(struct_content), filename, struct_name)
                    
                    struct = {
                        'name': struct_name,
                        'file': filename,
                        'start_line': start_line,
                        'end_line': end_line,
                        'fields': fields,
                        'inline': False
                    }
                    structs.append(struct)
                    self.all_symbol_names.add(struct_name)
                    self.all_struct_names.add(struct_name)  # Track struct names
                    self.stats['structs'] += 1
                    
                    i = end_line
                    continue
            
            i += 1
        
        return structs
    
    def _extract_struct_fields(self, content: str, filename: str, struct_name: str) -> List[Dict]:
        """Extract fields from a STRUCT definition"""
        fields = []
        lines = content.splitlines()
        in_body = False
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if 'BEGIN' in line_stripped.upper():
                in_body = True
                continue
            if not in_body:
                continue
            if self.re_end.search(line):
                break
            
            # Skip comments and empty lines
            if not line_stripped or line_stripped.startswith('!'):
                continue
            
            # Look for field declarations with array bounds
            array_match = self.re_array_decl.search(line)
            if array_match:
                data_type = array_match.group(1).upper()
                prefix = array_match.group(2)
                name = array_match.group(3).upper()
                bounds = array_match.group(4)
                
                if name not in TAL_KEYWORDS:
                    field = {
                        'name': name,
                        'data_type': f'{data_type} ARRAY',
                        'file': filename,
                        'line': i + 1,
                        'section': f'STRUCT_{struct_name}',
                        'parent_struct': struct_name,
                        'array_bounds': bounds,
                        'is_pointer': prefix == '.'
                    }
                    fields.append(field)
                    self.all_symbol_names.add(name)
                continue
            
            # Look for standard field declarations
            for match in self.re_standard_decl.finditer(line):
                data_type = match.group(1).upper()
                name = match.group(2).upper()
                
                if name not in TAL_KEYWORDS:
                    field = {
                        'name': name,
                        'data_type': data_type,
                        'file': filename,
                        'line': i + 1,
                        'section': f'STRUCT_{struct_name}',
                        'parent_struct': struct_name
                    }
                    fields.append(field)
                    self.all_symbol_names.add(name)
        
        return fields
    
    def _extract_all_symbols(self, content: str, filename: str) -> List[Dict]:
        """Extract all symbol definitions from TAL content"""
        symbols = []
        lines = content.splitlines()
        
        for i, line in enumerate(lines):
            line_num = i + 1
            
            # Skip preprocessor and comments
            if line.strip().startswith('?') or line.strip().startswith('!'):
                continue
            
            # Skip procedure declarations for symbol extraction
            if re.search(r'\b(PROC|SUBPROC)\b', line, re.IGNORECASE):
                continue
            
            # Skip STRUCT(*) inline declarations (handled separately)
            if re.search(r'STRUCT\s+\w+\s*\(\s*\*\s*\)', line, re.IGNORECASE):
                continue
            
            # DEFINE
            for match in self.re_define.finditer(line):
                name = match.group(1).upper()
                if name not in TAL_KEYWORDS and name not in self.all_symbol_names:
                    symbols.append({
                        'name': name,
                        'data_type': 'DEFINE',
                        'file': filename,
                        'line': line_num,
                        'section': 'DEFINES',
                        'definition': line.strip()
                    })
                    self.all_symbol_names.add(name)
                    self.stats['defines'] += 1
            
            # LITERAL
            for match in self.re_literal.finditer(line):
                name = match.group(1).upper()
                if name not in TAL_KEYWORDS and name not in self.all_symbol_names:
                    symbols.append({
                        'name': name,
                        'data_type': 'LITERAL',
                        'file': filename,
                        'line': line_num,
                        'section': 'LITERALS',
                        'definition': line.strip()
                    })
                    self.all_symbol_names.add(name)
                    self.stats['literals'] += 1
            
            # Arrays (check before standard to capture array syntax)
            for match in self.re_array_decl.finditer(line):
                data_type = match.group(1).upper()
                prefix = match.group(2)
                name = match.group(3).upper()
                bounds = match.group(4)
                
                if name not in TAL_KEYWORDS and name not in self.all_symbol_names:
                    is_pointer = prefix == '.'
                    full_name = ('.' + name) if is_pointer else name
                    symbols.append({
                        'name': full_name,
                        'data_type': f'{data_type} ARRAY',
                        'file': filename,
                        'line': line_num,
                        'section': 'ARRAYS',
                        'is_pointer': is_pointer,
                        'is_array': True,
                        'array_bounds': bounds,
                        'definition': line.strip()
                    })
                    self.all_symbol_names.add(name)
                    self.stats['arrays'] += 1
            
            # Pointers (non-array)
            for match in self.re_pointer_decl.finditer(line):
                data_type = match.group(1).upper()
                name = match.group(2).upper()
                
                if name not in TAL_KEYWORDS and name not in self.all_symbol_names:
                    # Skip if already captured as array
                    if not re.search(rf'\b{name}\s*\[', line, re.IGNORECASE):
                        symbols.append({
                            'name': '.' + name,
                            'data_type': f'{data_type} POINTER',
                            'file': filename,
                            'line': line_num,
                            'section': 'POINTERS',
                            'is_pointer': True,
                            'definition': line.strip()
                        })
                        self.all_symbol_names.add(name)
                        self.stats['pointers'] += 1
            
            # STRUCT instances
            for match in self.re_struct_instance.finditer(line):
                name = match.group(1).upper()
                template = match.group(2).upper()
                
                if name not in TAL_KEYWORDS and name not in self.all_symbol_names:
                    # Skip if it's a wildcard struct parameter
                    if template != '*':
                        symbols.append({
                            'name': name,
                            'data_type': f'STRUCT({template})',
                            'file': filename,
                            'line': line_num,
                            'section': 'STRUCT_INSTANCES',
                            'parent_struct': template,
                            'definition': line.strip()
                        })
                        self.all_symbol_names.add(name)
                        self.stats['struct_instances'] += 1
            
            # Standard declarations (catch remaining)
            for match in self.re_standard_decl.finditer(line):
                data_type = match.group(1).upper()
                name = match.group(2).upper()
                
                # Skip if already captured or is keyword
                if name in TAL_KEYWORDS or name in self.all_symbol_names:
                    continue
                
                symbols.append({
                    'name': name,
                    'data_type': data_type,
                    'file': filename,
                    'line': line_num,
                    'section': 'VARIABLES',
                    'definition': line.strip()
                })
                self.all_symbol_names.add(name)
                self.stats['variables'] += 1
        
        return symbols
    
    def _extract_procedures(self, content: str, raw_content: str, filename: str) -> List[Dict]:
        """Extract procedures and subprocedures with their code"""
        procedures = []
        
        # Use RAW content for procedure detection to maintain correct line numbers
        # Normalized content may have different line counts due to comment removal
        raw_lines = raw_content.splitlines()
        
        # Find all procedure starts in RAW content
        proc_starts = []
        
        for i, line in enumerate(raw_lines):
            line_upper = line.upper()
            
            # Skip comments
            if line.strip().startswith('!'):
                continue
            
            # Skip FORWARD declarations
            if re.search(r'^\s*FORWARD\b', line, re.IGNORECASE):
                continue
            
            # Typed PROC
            match = self.re_typed_proc.search(line)
            if match:
                proc_starts.append({
                    'line': i,
                    'name': match.group(2).upper(),
                    'return_type': match.group(1),
                    'params': match.group(3),
                    'attrs': match.group(4),
                    'type': 'PROC'
                })
                continue
            
            # Untyped PROC
            match = self.re_untyped_proc.search(line)
            if match:
                proc_starts.append({
                    'line': i,
                    'name': match.group(1).upper(),
                    'return_type': None,
                    'params': match.group(2),
                    'attrs': match.group(3),
                    'type': 'PROC'
                })
                continue
            
            # SUBPROC - handle multi-line parameter lists
            if re.search(r'^\s*SUBPROC\s+', line, re.IGNORECASE):
                name_match = re.search(r'SUBPROC\s+([A-Za-z_][A-Za-z0-9_^]*)', line, re.IGNORECASE)
                if name_match:
                    subproc_name = name_match.group(1).upper()
                    
                    # Find parameters - might span multiple lines
                    params = ''
                    if '(' in line:
                        if ')' in line:
                            # Complete on single line
                            param_match = re.search(r'\(([^)]*)\)', line)
                            if param_match:
                                params = param_match.group(1)
                        else:
                            # Multi-line parameters
                            param_start = line.find('(')
                            param_text = line[param_start + 1:]
                            for j in range(i + 1, min(i + 15, len(raw_lines))):
                                if ')' in raw_lines[j]:
                                    param_text += ' ' + raw_lines[j][:raw_lines[j].find(')')]
                                    break
                                else:
                                    param_text += ' ' + raw_lines[j]
                            params = param_text.strip()
                    
                    proc_starts.append({
                        'line': i,
                        'name': subproc_name,
                        'return_type': None,
                        'params': params,
                        'attrs': '',
                        'type': 'SUBPROC'
                    })
        
        # Extract procedure bodies from RAW content
        for proc_info in proc_starts:
            start_line = proc_info['line']
            is_subproc = proc_info['type'] == 'SUBPROC'
            
            # Find END of this procedure using RAW lines
            end_line = self._find_procedure_end(raw_lines, start_line, is_subproc)
            
            # Extract raw code (with comments) for display and step extraction
            raw_code = '\n'.join(raw_lines[start_line:end_line + 1])
            
            # Create normalized code (without comments) for analysis
            normalized_code = self._normalize_procedure_code(raw_code)
            
            # Parse parameters
            params = []
            if proc_info['params']:
                params = [p.strip() for p in proc_info['params'].split(',') if p.strip()]
            
            # Parse attributes
            attrs = []
            attr_str = proc_info['attrs'] if proc_info['attrs'] else ''
            for attr in ['MAIN', 'INTERRUPT', 'RESIDENT', 'CALLABLE', 'PRIV',
                         'VARIABLE', 'EXTENSIBLE', 'SHARED', 'REENTRANT']:
                if attr in attr_str.upper():
                    attrs.append(attr)
            
            # Extract validation steps from raw code (with comments)
            validation_steps = self._extract_validation_steps(raw_code)
            
            procedure = {
                'name': proc_info['name'],
                'proc_type': proc_info['type'],
                'return_type': proc_info['return_type'],
                'parameters': params,
                'attributes': attrs,
                'file': filename,
                'start_line': start_line + 1,
                'end_line': end_line + 1,
                'code': normalized_code,
                'raw_code': raw_code,  # Store raw code for ASK queries
                'validation_steps': validation_steps,  # Pre-extracted steps
                'calls': [],
                'data_refs': []
            }
            
            procedures.append(procedure)
            self.all_proc_names.add(proc_info['name'])
            self.stats['procedures'] += 1
        
        return procedures
    
    def _normalize_procedure_code(self, code: str) -> str:
        """Remove comments from procedure code while preserving structure"""
        lines = code.splitlines()
        normalized = []
        for line in lines:
            # Remove inline comments (! to end of line)
            if '!' in line:
                # Be careful not to remove ! inside strings
                in_string = False
                new_line = []
                for i, char in enumerate(line):
                    if char == '"' and (i == 0 or line[i-1] != '\\'):
                        in_string = not in_string
                    if char == '!' and not in_string:
                        break
                    new_line.append(char)
                line = ''.join(new_line)
            
            # Keep non-empty lines
            if line.strip():
                normalized.append(line)
        
        return '\n'.join(normalized)
    
    def _extract_validation_steps(self, raw_code: str) -> List[Dict]:
        """Extract validation steps from raw code with comments"""
        lines = raw_code.splitlines()
        steps = []
        current_step = None
        
        for line in lines:
            line_stripped = line.strip()
            
            # Pattern: ! STEP N: DESCRIPTION or variants
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
            
            # Collect details for current step
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
                    if call_name not in current_step['calls'] and call_name != 'ADD_VALIDATION_ERROR':
                        current_step['calls'].append(call_name)
        
        if current_step:
            steps.append(current_step)
        
        return steps
    
    def _find_procedure_end(self, lines: List[str], start_line: int, is_subproc: bool = False) -> int:
        """Find the END statement for a procedure, skipping struct BEGIN/END blocks"""
        # Strategy: Track nested BEGIN/END but skip struct blocks
        # For SUBPROCs: Stop at next SUBPROC definition or when depth returns to 0
        
        depth = 0
        in_struct_block = False
        struct_depth = 0
        found_proc_begin = False
        
        for i in range(start_line, len(lines)):
            line = lines[i]
            line_upper = line.upper()
            line_stripped = line_upper.strip()
            
            # For SUBPROCs, stop if we hit another SUBPROC definition
            if is_subproc and i > start_line:
                if re.search(r'^\s*SUBPROC\s+[A-Za-z_]', line_upper):
                    return i - 1
            
            # Check if entering a struct definition (inline or standard)
            if re.search(r'\bSTRUCT\s+[A-Za-z_]\w*\s*(\(\s*\*\s*\))?\s*;', line_upper):
                in_struct_block = True
                struct_depth = 0
                continue
            
            # Track struct BEGIN/END separately
            if in_struct_block:
                if re.search(r'\bBEGIN\b', line_upper):
                    struct_depth += 1
                if re.search(r'\bEND\s*;', line_upper):
                    struct_depth -= 1
                    if struct_depth <= 0:
                        in_struct_block = False
                continue
            
            # Track procedure BEGIN/END (not in struct block)
            begin_matches = len(re.findall(r'\bBEGIN\b', line_upper))
            end_matches = len(re.findall(r'\bEND\b', line_upper))
            
            if begin_matches > 0:
                found_proc_begin = True
                depth += begin_matches
            
            depth -= end_matches
            
            # Procedure ends when we've seen at least one BEGIN and depth returns to 0
            if found_proc_begin and depth <= 0:
                return i
            
            # For SUBPROCs without BEGIN, look for END; on its own
            if is_subproc and not found_proc_begin and re.match(r'^\s*END\s*;', line_stripped):
                return i
        
        # Fallback: look for END; near end of file or before next PROC/SUBPROC
        for i in range(start_line + 1, len(lines)):
            line_upper = lines[i].upper().strip()
            # Found next procedure definition
            if re.search(r'^\s*(INT|STRING|FIXED|REAL)?\s*PROC\s+', line_upper):
                return i - 1
            if re.search(r'^\s*SUBPROC\s+', line_upper):
                return i - 1
        
        return len(lines) - 1
    
    def _find_data_references(self, code: str) -> List[str]:
        """Find references to known symbols in code"""
        refs = []
        words = set(re.findall(r'\b[A-Z_][A-Z0-9_^]*\b', code.upper()))
        
        for word in words:
            if word in self.all_symbol_names and word not in TAL_KEYWORDS:
                refs.append(word)
        
        return refs
    
    def _extract_calls(self, code: str, proc_name: str = None) -> List[Dict]:
        """Extract procedure calls from code"""
        calls = []
        
        # Build exclusion set for things that are NOT procedures
        exclude = set(TAL_KEYWORDS)
        exclude.update(self.all_struct_names)  # Struct names aren't callable
        # Don't exclude all_symbol_names - some might be procedure names
        if proc_name:
            exclude.add(proc_name.upper())
        
        # CALL statements - these are definite procedure calls
        for match in self.re_call.finditer(code):
            target = match.group(1).upper()
            # Only exclude keywords and struct names, allow everything else
            if target not in exclude:
                calls.append({'target': target, 'type': 'CALL'})
        
        # Function-style calls - be more selective
        for match in self.re_func_call.finditer(code):
            target = match.group(1).upper()
            # Skip keywords, struct names, system procedures
            if target in exclude:
                continue
            if target in SYSTEM_PROCEDURES:
                continue
            # Skip data type names used in declarations/casts
            if target in {'INT', 'STRING', 'FIXED', 'REAL', 'UNSIGNED', 'BYTE', 'CHAR'}:
                continue
            # Only include if it's a known procedure (to avoid false positives)
            if target in self.all_proc_names:
                calls.append({'target': target, 'type': 'FUNC_CALL'})
        
        # System calls ($procedure_name)
        for match in self.re_system_call.finditer(code):
            target = '$' + match.group(1).upper()
            calls.append({'target': target, 'type': 'SYSTEM_CALL'})
        
        # Deduplicate
        seen = set()
        unique_calls = []
        for call in calls:
            if call['target'] not in seen:
                seen.add(call['target'])
                unique_calls.append(call)
        
        return unique_calls
    
    def _extract_business_rules(self, code: str, filename: str,
                                proc_name: str, start_line: int) -> List[Dict]:
        """Extract business rules from code"""
        rules = []
        lines = code.splitlines()
        
        for i, line in enumerate(lines):
            line_num = start_line + i
            
            # Conditional logic
            if self.re_if_then.search(line):
                rules.append({
                    'rule_type': 'CONDITIONAL_LOGIC',
                    'description': 'Conditional business logic',
                    'file': filename,
                    'line': line_num,
                    'procedure': proc_name,
                    'source_code': line.strip()[:200]
                })
            
            # Validation patterns
            if self.re_validation.search(line):
                rules.append({
                    'rule_type': 'VALIDATION',
                    'description': 'Data validation logic',
                    'file': filename,
                    'line': line_num,
                    'procedure': proc_name,
                    'source_code': line.strip()[:200]
                })
            
            # Error handling
            if self.re_error.search(line):
                rules.append({
                    'rule_type': 'ERROR_HANDLING',
                    'description': 'Error handling logic',
                    'file': filename,
                    'line': line_num,
                    'procedure': proc_name,
                    'source_code': line.strip()[:200]
                })
            
            # SQL/Database operations
            if re.search(r'\b(SELECT|INSERT|UPDATE|DELETE)\b', line, re.IGNORECASE):
                rules.append({
                    'rule_type': 'DATA_ACCESS',
                    'description': 'Database operation',
                    'file': filename,
                    'line': line_num,
                    'procedure': proc_name,
                    'source_code': line.strip()[:200]
                })
        
        return rules
    
    def _create_augmented_text(self, item: Dict, item_type: str) -> str:
        """Create enriched text for semantic embedding"""
        parts = []
        
        if item_type == 'procedure':
            parts.append(f"TAL Procedure: {item['name']}")
            parts.append(f"Type: {item.get('proc_type', 'PROC')}")
            parts.append(f"File: {item['file']}")
            
            if item.get('return_type'):
                parts.append(f"Returns: {item['return_type']}")
            
            if item.get('attributes'):
                parts.append(f"Attributes: {', '.join(item['attributes'])}")
            
            # Infer purpose from name
            name = item['name']
            for keyword in BUSINESS_KEYWORDS:
                if keyword in name:
                    parts.append(f"Business Function: {keyword}")
                    break
            
            if item.get('data_refs'):
                parts.append(f"Uses: {', '.join(item['data_refs'][:15])}")
            
            if item.get('calls'):
                call_targets = [c['target'] for c in item['calls'][:10]]
                parts.append(f"Calls: {', '.join(call_targets)}")
            
            # Include code preview
            code = item.get('code', '')
            parts.append(f"Code: {code[:500]}")
            
        elif item_type == 'symbol':
            parts.append(f"TAL Symbol: {item['name']}")
            parts.append(f"Type: {item.get('data_type', 'UNKNOWN')}")
            parts.append(f"Section: {item.get('section', 'GLOBAL')}")
            
            if item.get('parent_struct'):
                parts.append(f"Parent: {item['parent_struct']}")
            
            if item.get('is_pointer'):
                parts.append("Pointer: Yes")
            
            if item.get('is_array'):
                parts.append(f"Array: {item.get('array_bounds', 'Yes')}")
            
            parts.append(f"File: {item['file']}")
        
        return ' | '.join(parts)
    
    def _embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Embed texts in batches using available provider"""
        if not self.has_embeddings or not texts:
            return []
        
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            if self.embedding_provider == 'openai_compatible':
                embeddings = self._embed_openai(batch)
            elif self.embedding_provider == 'transformers':
                embeddings = self._embed_transformers(batch)
            else:
                return []
            
            all_embeddings.extend(embeddings)
        
        return all_embeddings
    
    def _embed_openai(self, texts: List[str]) -> List[List[float]]:
        """Embed using OpenAI-compatible API"""
        try:
            headers = {'Content-Type': 'application/json'}
            if self.api_key:
                headers['Authorization'] = f'Bearer {self.api_key}'
            
            response = requests.post(
                f"{self.api_url}/embeddings",
                headers=headers,
                json={
                    'model': self.embedding_model,
                    'input': texts
                },
                timeout=60
            )
            response.raise_for_status()
            
            data = response.json()
            embeddings = [item['embedding'] for item in sorted(data['data'], key=lambda x: x['index'])]
            return embeddings
        
        except Exception as e:
            print(f"   ⚠️  Embedding API error: {e}")
            return [[0.0] * self.embedding_dim] * len(texts)  # Return zeros
    
    def _embed_transformers(self, texts: List[str]) -> List[List[float]]:
        """Embed using HuggingFace transformers"""
        try:
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=512
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)
                return embeddings.cpu().numpy().tolist()
        
        except Exception as e:
            print(f"   ⚠️  Transformers embedding error: {e}")
            return [[0.0] * self.embedding_dim] * len(texts)
    
    def index_directory(self, root_dir: str):
        """Index TAL codebase with two-pass architecture"""
        root = Path(root_dir)
        
        # TAL files typically have .TAL or .TXT extension
        extensions = {'.tal', '.txt', '.h', '.inc', '.ext'}
        
        all_files = [
            p for p in root.rglob('*')
            if p.suffix.lower() in extensions and p.is_file()
        ]
        
        # Try to identify include files vs main programs
        include_patterns = ['EXTDEC', 'DEFINE', 'STRUCT', 'LITERAL', 'INC', 'HDR', 'TYPEDEF']
        includes = []
        programs = []
        
        for f in all_files:
            name_upper = f.stem.upper()
            if any(pat in name_upper for pat in include_patterns):
                includes.append(f)
            else:
                programs.append(f)
        
        # If no clear separation, treat all as programs
        if not includes:
            programs = all_files
        
        print(f"📦 Found {len(programs)} programs, {len(includes)} include files")
        
        # ===================================================================
        # PASS 1: Extract all symbols from ALL files
        # ===================================================================
        print("\n📋 Pass 1: Building global symbol table...")
        
        for fpath in includes + programs:
            try:
                with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                    raw = f.read()
                content = self._normalize_tal(raw)
                
                # Cache include content
                if fpath in includes:
                    self.include_content[fpath.stem.upper()] = content
                
                # Extract preprocessor
                self._extract_preprocessor(raw, fpath.name)
                
                # Extract structs (including inline)
                file_structs = self._extract_structs(content, fpath.name)
                self.structs.extend(file_structs)
                
                # Extract symbols
                file_symbols = self._extract_all_symbols(content, fpath.name)
                self.symbols.extend(file_symbols)
                
            except Exception as e:
                print(f"  ⚠️ Error in {fpath.name}: {e}")
        
        print(f"   Found {len(self.symbols)} symbols, {len(self.all_symbol_names)} unique names")
        print(f"   Found {len(self.structs)} structs")
        
        # ===================================================================
        # PASS 2: Extract procedures and find references
        # ===================================================================
        print("\n🔍 Pass 2: Extracting procedures and references...")
        
        for fpath in programs:
            try:
                with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                    raw = f.read()
                content = self._normalize_tal(raw)
                
                # Extract procedures (pass both normalized and raw content)
                file_procedures = self._extract_procedures(content, raw, fpath.name)
                
                for proc in file_procedures:
                    # Find data references using global symbol table
                    data_refs = self._find_data_references(proc['code'])
                    proc['data_refs'] = data_refs
                    
                    # Update data usage index
                    for ref in data_refs:
                        self.data_usage_index[ref].append({
                            'procedure': proc['name'],
                            'file': fpath.name,
                            'line': proc['start_line'],
                            'type': proc['proc_type']
                        })
                    
                    # Extract calls (pass proc name to exclude self-references)
                    calls = self._extract_calls(proc['code'], proc['name'])
                    proc['calls'] = calls
                    
                    # Extract business rules
                    rules = self._extract_business_rules(
                        proc['code'], fpath.name,
                        proc['name'], proc['start_line']
                    )
                    self.business_rules.extend(rules)
                    
                    # Create augmented text
                    proc['augmented_text'] = self._create_augmented_text(proc, 'procedure')
                    
                    self.procedures.append(proc)
                    
                    # Update call graph
                    proc_name = proc['name']
                    self.call_graph.setdefault(proc_name, {
                        'file': fpath.name,
                        'type': proc['proc_type'],
                        'targets': [],
                        'callers': []
                    })
                    
                    for call in calls:
                        target = call['target']
                        if target not in self.call_graph[proc_name]['targets']:
                            self.call_graph[proc_name]['targets'].append(target)
                        
                        self.call_graph.setdefault(target, {
                            'file': None,
                            'type': 'external',
                            'targets': [],
                            'callers': []
                        })
                        if proc_name not in self.call_graph[target]['callers']:
                            self.call_graph[target]['callers'].append(proc_name)
                
            except Exception as e:
                print(f"  ⚠️ Error in {fpath.name}: {e}")
        
        print(f"   Extracted {len(self.procedures)} procedures")
        print(f"   Found {len(self.business_rules)} business rules")
        print(f"   Built usage index for {len(self.data_usage_index)} symbols")
        
        # ===================================================================
        # PASS 3: Persist to vector DB and JSON
        # ===================================================================
        print("\n💾 Persisting indexes...")
        
        # Add augmented text to symbols
        for sym in self.symbols:
            sym['augmented_text'] = self._create_augmented_text(sym, 'symbol')
        
        self._persist_to_vector_db()
        self._save_json_stores()
        
        self._print_stats()
    
    def _persist_to_vector_db(self):
        """Store embeddings in Qdrant (if available)"""
        if not self.has_embeddings or not self.qdrant:
            print("   ℹ️  Skipping vector embeddings (not configured)")
            return
        
        if self.procedures:
            print(f"   Embedding {len(self.procedures)} procedures...")
            texts = [p['augmented_text'] for p in self.procedures]
            embeddings = self._embed_batch(texts)
            
            if embeddings and HAS_QDRANT:
                points = [
                    PointStruct(id=i, vector=emb, payload=proc)
                    for i, (proc, emb) in enumerate(zip(self.procedures, embeddings))
                ]
                self.qdrant.upsert(self.logic_coll, points)
        
        if self.symbols:
            print(f"   Embedding {len(self.symbols)} symbols...")
            texts = [s['augmented_text'] for s in self.symbols]
            embeddings = self._embed_batch(texts)
            
            if embeddings and HAS_QDRANT:
                points = [
                    PointStruct(id=i + 1000000, vector=emb, payload=sym)
                    for i, (sym, emb) in enumerate(zip(self.symbols, embeddings))
                ]
                self.qdrant.upsert(self.symbol_coll, points)
    
    def _save_json_stores(self):
        """Save all data to JSON"""
        stores = {
            'procedures.json': self.procedures,
            'symbols.json': self.symbols,
            'structs.json': self.structs,
            'call_graph.json': self.call_graph,
            'data_usage.json': dict(self.data_usage_index),
            'business_rules.json': self.business_rules,
            'includes.json': {k: v[:200] + '...' for k, v in self.include_content.items()},
            'preprocessor.json': self.preprocessor_defines
        }
        
        for filename, data in stores.items():
            with open(os.path.join(self.db_path, filename), 'w') as f:
                json.dump(data, f, indent=2)
        
        print(f"   Saved to {self.db_path}/")
    
    def _print_stats(self):
        """Print indexing statistics"""
        total_calls = sum(len(p.get('calls', [])) for p in self.procedures)
        total_steps = sum(len(p.get('validation_steps', [])) for p in self.procedures)
        
        print(f"\n✅ Indexing complete!")
        print(f"   📋 {len(self.procedures)} procedures")
        print(f"   🏷️  {len(self.symbols)} symbols")
        print(f"   🏗️  {len(self.structs)} structs ({self.stats.get('inline_structs', 0)} inline)")
        print(f"   📊 {len(self.data_usage_index)} symbols with usage tracking")
        print(f"   📝 {len(self.business_rules)} business rules")
        print(f"   📞 {total_calls} procedure calls extracted")
        print(f"   🔢 {total_steps} validation steps extracted")
        print(f"\n   Breakdown:")
        for key, value in sorted(self.stats.items()):
            print(f"     {key}: {value}")
    
    def close(self):
        """Close database connection"""
        if self.qdrant:
            try:
                self.qdrant.close()
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description="Index TAL codebase (Enhanced v2)")
    parser.add_argument("--code-dir", required=True, help="Directory containing TAL files")
    parser.add_argument("--db-path", default="./tal_index", help="Path to store index")
    args = parser.parse_args()
    
    indexer = TalIndexerV2(db_path=args.db_path)
    try:
        indexer.index_directory(args.code_dir)
    finally:
        indexer.close()


if __name__ == "__main__":
    main()
