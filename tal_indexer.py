#!/usr/bin/env python3
"""
TAL Intelligence System - Enhanced Indexer v3.0

Combines robust AST structures from tal_proc_parser with comprehensive
logic extraction from tal_indexer_v2.

Key Enhancements:
- Proper AST with TALNode structure
- Strong typing with TALType enum
- Scoped SymbolTable with enter/exit scope
- Precise SourceLocation tracking
- Procedure attributes (MAIN, FORWARD, EXTERNAL, VARIABLE, EXTENSIBLE)
- Rich ParseError with suggestions
- S-expression output for debugging

Architecture:
- Pass 1: Build global symbol table (STRUCTs, DEFINEs, LITERALs)
- Pass 2: Parse procedures with full body analysis
- Pass 3: Build call graph and extract business logic
#YK123
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict

# Optional: Vector embeddings
try:
    from sentence_transformers import SentenceTransformer
    from qdrant_client import QdrantClient
    from qdrant_client.models import VectorParams, Distance, PointStruct
    HAS_VECTORS = True
except ImportError:
    HAS_VECTORS = False
    print("Note: sentence-transformers/qdrant not installed. Vector search disabled.")


# =============================================================================
# TYPE SYSTEM
# =============================================================================

class TALType(Enum):
    """TAL data types with size specifications"""
    INT = "INT"
    INT16 = "INT(16)"
    INT32 = "INT(32)"
    INT64 = "INT(64)"
    STRING = "STRING"
    REAL = "REAL"
    REAL32 = "REAL(32)"
    REAL64 = "REAL(64)"
    FIXED = "FIXED"
    UNSIGNED = "UNSIGNED"
    UNSIGNED16 = "UNSIGNED(16)"
    UNSIGNED32 = "UNSIGNED(32)"
    BYTE = "BYTE"
    CHAR = "CHAR"
    STRUCT = "STRUCT"
    POINTER = "POINTER"
    LITERAL = "LITERAL"
    DEFINE = "DEFINE"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def from_string(cls, type_str: str) -> 'TALType':
        """Parse TAL type string into TALType enum"""
        if not type_str:
            return cls.UNKNOWN
        
        type_str = type_str.upper().strip()
        
        # Handle parameterized integer types
        if type_str.startswith("INT("):
            if "32" in type_str:
                return cls.INT32
            elif "64" in type_str:
                return cls.INT64
            elif "16" in type_str:
                return cls.INT16
            return cls.INT
        
        # Handle parameterized real types
        if type_str.startswith("REAL("):
            if "64" in type_str:
                return cls.REAL64
            elif "32" in type_str:
                return cls.REAL32
            return cls.REAL
        
        # Handle unsigned types
        if type_str.startswith("UNSIGNED("):
            if "32" in type_str:
                return cls.UNSIGNED32
            elif "16" in type_str:
                return cls.UNSIGNED16
            return cls.UNSIGNED
        
        # Handle simple types
        type_map = {
            'INT': cls.INT,
            'STRING': cls.STRING,
            'REAL': cls.REAL,
            'FIXED': cls.FIXED,
            'UNSIGNED': cls.UNSIGNED,
            'BYTE': cls.BYTE,
            'CHAR': cls.CHAR,
            'STRUCT': cls.STRUCT,
            'LITERAL': cls.LITERAL,
            'DEFINE': cls.DEFINE,
        }
        return type_map.get(type_str, cls.UNKNOWN)


class ProcType(Enum):
    """Procedure types"""
    PROC = "PROC"
    SUBPROC = "SUBPROC"


class ErrorSeverity(Enum):
    """Error severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    FATAL = "fatal"


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class SourceLocation:
    """Source code location for error reporting"""
    filename: str = ""
    line: int = 0
    column: int = 0
    end_line: int = 0
    
    def __str__(self):
        return f"{self.filename}:{self.line}"
    
    def to_dict(self) -> Dict:
        return {'file': self.filename, 'line': self.line, 'end_line': self.end_line}


@dataclass
class ParseError:
    """Rich parsing error with context"""
    message: str
    location: SourceLocation
    severity: ErrorSeverity = ErrorSeverity.ERROR
    context_lines: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    error_code: str = ""
    
    def __str__(self):
        result = f"{self.severity.value.upper()}: {self.message} at {self.location}"
        if self.suggestions:
            result += f" (Suggestions: {', '.join(self.suggestions)})"
        return result


@dataclass
class Symbol:
    """Symbol table entry"""
    name: str
    symbol_type: TALType
    location: SourceLocation
    scope: str = "global"
    is_pointer: bool = False
    is_array: bool = False
    array_size: int = 0
    struct_name: Optional[str] = None
    value: Optional[str] = None  # For LITERAL/DEFINE
    definition: Optional[str] = None  # Full definition text
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'type': self.symbol_type.value,
            'file': self.location.filename,
            'line': self.location.line,
            'scope': self.scope,
            'is_pointer': self.is_pointer,
            'is_array': self.is_array,
            'array_size': self.array_size,
            'struct_name': self.struct_name,
            'value': self.value,
            'definition': self.definition,
        }


@dataclass
class StructField:
    """Field within a STRUCT"""
    name: str
    field_type: TALType
    offset: int = 0
    size: int = 0
    is_pointer: bool = False
    is_array: bool = False
    array_size: int = 0
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'type': self.field_type.value,
            'offset': self.offset,
            'is_pointer': self.is_pointer,
            'is_array': self.is_array,
            'array_size': self.array_size,
        }


@dataclass
class StructDef:
    """Structure definition"""
    name: str
    location: SourceLocation
    fields: List[StructField] = field(default_factory=list)
    is_template: bool = False  # STRUCT (*) reference template
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'file': self.location.filename,
            'line': self.location.line,
            'fields': [f.to_dict() for f in self.fields],
            'is_template': self.is_template,
        }


@dataclass
class ValidationStep:
    """Validation step extracted from comments"""
    step_num: int
    description: str
    calls: List[str] = field(default_factory=list)
    error_codes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'step_num': self.step_num,
            'description': self.description,
            'calls': self.calls,
            'errors': self.error_codes,
        }


@dataclass
class ErrorCode:
    """Error/warning code with message"""
    code: str
    message: str
    error_type: str = "ERROR"  # ERROR or WARNING
    
    def to_dict(self) -> Dict:
        return {
            'code': self.code,
            'message': self.message,
            'type': self.error_type,
        }


@dataclass
class BusinessRule:
    """Extracted business rule"""
    rule_type: str  # VALIDATION, THRESHOLD, COMPLIANCE, etc.
    source_code: str
    procedure: str
    location: SourceLocation
    
    def to_dict(self) -> Dict:
        return {
            'rule_type': self.rule_type,
            'source_code': self.source_code,
            'procedure': self.procedure,
            'file': self.location.filename,
            'line': self.location.line,
        }


@dataclass
class ProcedureCall:
    """Record of a procedure call"""
    target: str
    location: SourceLocation
    context: str = ""  # Surrounding code context
    
    def to_dict(self) -> Dict:
        return {
            'target': self.target,
            'line': self.location.line,
            'context': self.context,
        }


@dataclass 
class TALProcedure:
    """Complete procedure representation"""
    name: str
    proc_type: ProcType
    location: SourceLocation
    
    # Signature
    return_type: Optional[TALType] = None
    parameters: List[Symbol] = field(default_factory=list)
    
    # Attributes
    is_main: bool = False
    is_forward: bool = False
    is_external: bool = False
    is_variable: bool = False
    is_extensible: bool = False
    
    # Code content
    raw_code: str = ""  # With comments
    normalized_code: str = ""  # Comments removed
    
    # Analysis results
    calls: List[ProcedureCall] = field(default_factory=list)
    data_refs: List[str] = field(default_factory=list)
    validation_steps: List[ValidationStep] = field(default_factory=list)
    error_codes: List[ErrorCode] = field(default_factory=list)
    business_rules: List[BusinessRule] = field(default_factory=list)
    local_symbols: List[Symbol] = field(default_factory=list)
    
    # For SUBPROCs
    parent_proc: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'proc_type': self.proc_type.value,
            'file': self.location.filename,
            'line': self.location.line,
            'end_line': self.location.end_line,
            'return_type': self.return_type.value if self.return_type else None,
            'parameters': [p.name for p in self.parameters],
            'parameter_types': {p.name: p.symbol_type.value for p in self.parameters},
            'is_main': self.is_main,
            'is_forward': self.is_forward,
            'is_external': self.is_external,
            'is_variable': self.is_variable,
            'is_extensible': self.is_extensible,
            'raw_code': self.raw_code,
            'code': self.normalized_code,
            'calls': [c.to_dict() for c in self.calls],
            'data_refs': self.data_refs,
            'validation_steps': [s.to_dict() for s in self.validation_steps],
            'error_codes': [e.to_dict() for e in self.error_codes],
            'business_rules': [r.to_dict() for r in self.business_rules],
            'parent_proc': self.parent_proc,
        }
    
    def to_sexp(self, indent: int = 0) -> str:
        """Convert to S-expression for debugging"""
        spaces = "  " * indent
        result = f"{spaces}(procedure :name {self.name} :type {self.proc_type.value}"
        
        if self.return_type:
            result += f" :returns {self.return_type.value}"
        
        if self.parameters:
            result += f"\n{spaces}  (parameters"
            for p in self.parameters:
                result += f"\n{spaces}    (param :name {p.name} :type {p.symbol_type.value})"
            result += ")"
        
        if self.calls:
            result += f"\n{spaces}  (calls"
            for c in self.calls[:10]:
                result += f"\n{spaces}    {c.target}"
            if len(self.calls) > 10:
                result += f"\n{spaces}    ... and {len(self.calls) - 10} more"
            result += ")"
        
        if self.validation_steps:
            result += f"\n{spaces}  (steps {len(self.validation_steps)})"
        
        if self.error_codes:
            result += f"\n{spaces}  (errors {len(self.error_codes)})"
        
        result += ")"
        return result


# =============================================================================
# SYMBOL TABLE
# =============================================================================

class SymbolTable:
    """Scope-aware symbol table"""
    
    def __init__(self):
        self.scopes: Dict[str, Dict[str, Symbol]] = {"global": {}}
        self.current_scope = "global"
        self.scope_stack: List[str] = ["global"]
        self.structs: Dict[str, StructDef] = {}
        
    def enter_scope(self, scope_name: str):
        """Enter a new scope (e.g., procedure)"""
        self.scope_stack.append(scope_name)
        self.current_scope = scope_name
        if scope_name not in self.scopes:
            self.scopes[scope_name] = {}
    
    def exit_scope(self):
        """Exit current scope"""
        if len(self.scope_stack) > 1:
            self.scope_stack.pop()
            self.current_scope = self.scope_stack[-1]
    
    def declare(self, symbol: Symbol) -> Optional[ParseError]:
        """Declare a symbol in current scope"""
        symbol.scope = self.current_scope
        
        if self.current_scope not in self.scopes:
            self.scopes[self.current_scope] = {}
        
        # Allow redeclaration in different scopes
        self.scopes[self.current_scope][symbol.name.upper()] = symbol
        return None
    
    def lookup(self, name: str) -> Optional[Symbol]:
        """Look up symbol in scope chain"""
        name_upper = name.upper()
        for scope in reversed(self.scope_stack):
            if scope in self.scopes and name_upper in self.scopes[scope]:
                return self.scopes[scope][name_upper]
        return None
    
    def declare_struct(self, struct: StructDef):
        """Register a struct definition"""
        self.structs[struct.name.upper()] = struct
    
    def lookup_struct(self, name: str) -> Optional[StructDef]:
        """Look up struct definition"""
        return self.structs.get(name.upper())
    
    def get_all_global_symbols(self) -> List[Symbol]:
        """Get all global scope symbols"""
        return list(self.scopes.get("global", {}).values())
    
    def get_all_symbols(self) -> List[Symbol]:
        """Get all symbols from all scopes"""
        all_symbols = []
        for scope_symbols in self.scopes.values():
            all_symbols.extend(scope_symbols.values())
        return all_symbols


# =============================================================================
# TAL INDEXER
# =============================================================================

class TALIndexer:
    """
    Enhanced TAL code indexer with proper AST structures.
    
    Three-pass architecture:
    1. Build global symbol table (STRUCTs, DEFINEs, LITERALs)
    2. Parse procedures with full body analysis
    3. Build call graph and cross-references
    """
    
    # File extensions to process
    EXTENSIONS = {'.tal', '.txt', '.h', '.inc', '.ext'}
    
    # Patterns for procedure detection
    PROC_PATTERN = re.compile(
        r'^\s*(?:(INT(?:\([^)]*\))?|REAL(?:\([^)]*\))?|STRING|FIXED|UNSIGNED(?:\([^)]*\))?)\s+)?'
        r'(PROC|SUBPROC)\s+([A-Za-z_][A-Za-z0-9_^]*)',
        re.IGNORECASE | re.MULTILINE
    )
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.symbol_table = SymbolTable()
        self.procedures: List[TALProcedure] = []
        self.business_rules: List[BusinessRule] = []
        self.call_graph: Dict[str, Dict] = {}
        self.data_usage: Dict[str, List[Dict]] = defaultdict(list)
        self.errors: List[ParseError] = []
        
        # Ensure output directory exists
        os.makedirs(db_path, exist_ok=True)
    
    def index_directory(self, code_dir: str):
        """Main entry point - index all TAL files in directory"""
        root = Path(code_dir)
        
        # Find all TAL files
        files = [
            p for p in root.rglob('*')
            if p.suffix.lower() in self.EXTENSIONS and p.is_file()
        ]
        
        if not files:
            print(f"No TAL files found in {code_dir}")
            return
        
        print(f"\n📂 Found {len(files)} TAL file(s)")
        
        # Pass 1: Build global symbol table
        print("\n📋 Pass 1: Building global symbol table...")
        for filepath in files:
            self._pass1_symbols(filepath)
        
        symbols = self.symbol_table.get_all_global_symbols()
        structs = list(self.symbol_table.structs.values())
        print(f"   Found {len(symbols)} symbols, {len(structs)} structs")
        
        # Pass 2: Parse procedures
        print("\n📋 Pass 2: Parsing procedures...")
        for filepath in files:
            self._pass2_procedures(filepath)
        
        print(f"   Extracted {len(self.procedures)} procedures")
        
        # Pass 3: Build call graph and cross-references
        print("\n📋 Pass 3: Building call graph...")
        self._pass3_call_graph()
        print(f"   Built call graph with {len(self.call_graph)} entries")
        
        # Save results
        self._save_index()
        
        # Create vector embeddings if available
        if HAS_VECTORS:
            self._create_embeddings()
        
        # Report errors
        if self.errors:
            print(f"\n⚠️  {len(self.errors)} parsing issue(s):")
            for err in self.errors[:10]:
                print(f"   {err}")
            if len(self.errors) > 10:
                print(f"   ... and {len(self.errors) - 10} more")
        
        print("\n✅ Indexing complete!")
    
    # =========================================================================
    # PASS 1: Symbol Collection
    # =========================================================================
    
    def _pass1_symbols(self, filepath: Path):
        """First pass: extract global symbols"""
        try:
            content = filepath.read_text(encoding='utf-8', errors='replace')
        except Exception as e:
            self.errors.append(ParseError(
                f"Could not read file: {e}",
                SourceLocation(str(filepath)),
                ErrorSeverity.ERROR
            ))
            return
        
        filename = filepath.name
        lines = content.split('\n')
        
        # Extract STRUCTs
        self._extract_structs(lines, filename)
        
        # Extract DEFINEs
        self._extract_defines(lines, filename)
        
        # Extract LITERALs
        self._extract_literals(lines, filename)
        
        # Extract global variables
        self._extract_globals(lines, filename)
    
    def _extract_structs(self, lines: List[str], filename: str):
        """Extract STRUCT definitions"""
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Match: STRUCT name; or STRUCT name(*);
            match = re.match(
                r'^\s*STRUCT\s+([A-Za-z_][A-Za-z0-9_]*)\s*(\(\s*\*\s*\))?\s*;',
                line, re.IGNORECASE
            )
            
            if match:
                struct_name = match.group(1)
                is_template = match.group(2) is not None
                
                location = SourceLocation(filename, i + 1)
                struct_def = StructDef(struct_name, location, is_template=is_template)
                
                # Parse struct body
                if not is_template:
                    fields, end_line = self._parse_struct_body(lines, i + 1, filename)
                    struct_def.fields = fields
                    location.end_line = end_line
                
                self.symbol_table.declare_struct(struct_def)
            
            i += 1
    
    def _parse_struct_body(self, lines: List[str], start: int, filename: str) -> Tuple[List[StructField], int]:
        """Parse fields within a STRUCT BEGIN...END block"""
        fields = []
        i = start
        in_struct = False
        depth = 0
        
        while i < len(lines):
            line = lines[i].strip()
            
            # Remove comments
            comment_pos = line.find('!')
            if comment_pos >= 0:
                line = line[:comment_pos].strip()
            
            if not line:
                i += 1
                continue
            
            # Look for BEGIN
            if re.search(r'\bBEGIN\b', line, re.IGNORECASE):
                in_struct = True
                depth += 1
                i += 1
                continue
            
            # Look for END
            if re.search(r'\bEND\b', line, re.IGNORECASE):
                depth -= 1
                if depth <= 0:
                    return fields, i + 1
                i += 1
                continue
            
            # Parse field declarations inside struct
            if in_struct and depth == 1:
                field = self._parse_field_declaration(line, filename, i + 1)
                if field:
                    fields.append(field)
            
            i += 1
        
        return fields, i
    
    def _parse_field_declaration(self, line: str, filename: str, line_num: int) -> Optional[StructField]:
        """Parse a single field declaration"""
        # Match: TYPE name[size]; or TYPE .name;
        match = re.match(
            r'(INT(?:\([^)]*\))?|REAL(?:\([^)]*\))?|STRING|FIXED|UNSIGNED(?:\([^)]*\))?|STRUCT)\s+'
            r'(\.)?([A-Za-z_][A-Za-z0-9_]*)(?:\[(\d+)\])?',
            line, re.IGNORECASE
        )
        
        if match:
            field_type = TALType.from_string(match.group(1))
            is_pointer = match.group(2) is not None
            name = match.group(3)
            array_size = int(match.group(4)) if match.group(4) else 0
            
            return StructField(
                name=name,
                field_type=field_type,
                is_pointer=is_pointer,
                is_array=array_size > 0,
                array_size=array_size
            )
        
        return None
    
    def _extract_defines(self, lines: List[str], filename: str):
        """Extract DEFINE macros"""
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Match: DEFINE name = value; or DEFINE name(params) = value;
            match = re.match(
                r'^\s*DEFINE\s+([A-Za-z_][A-Za-z0-9_]*)(?:\([^)]*\))?\s*=\s*(.+?)\s*[;#]',
                line, re.IGNORECASE
            )
            
            if match:
                name = match.group(1)
                value = match.group(2).strip()
                
                symbol = Symbol(
                    name=name,
                    symbol_type=TALType.DEFINE,
                    location=SourceLocation(filename, i + 1),
                    value=value,
                    definition=line.strip()
                )
                self.symbol_table.declare(symbol)
            
            i += 1
    
    def _extract_literals(self, lines: List[str], filename: str):
        """Extract LITERAL constants"""
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Match: LITERAL name = value; or LITERAL name = value,
            match = re.match(
                r'^\s*LITERAL\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^,;]+)',
                line, re.IGNORECASE
            )
            
            if match:
                name = match.group(1)
                value = match.group(2).strip()
                
                symbol = Symbol(
                    name=name,
                    symbol_type=TALType.LITERAL,
                    location=SourceLocation(filename, i + 1),
                    value=value,
                    definition=f"LITERAL {name} = {value}"
                )
                self.symbol_table.declare(symbol)
            
            # Also match continuation literals: name = value,
            elif '=' in line and not line.strip().startswith('!'):
                cont_match = re.match(
                    r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([^,;]+)',
                    line, re.IGNORECASE
                )
                if cont_match:
                    # Check if previous non-empty line was LITERAL
                    for j in range(i - 1, max(0, i - 5), -1):
                        prev = lines[j].strip()
                        if prev and not prev.startswith('!'):
                            if re.search(r'\bLITERAL\b', prev, re.IGNORECASE) or prev.endswith(','):
                                name = cont_match.group(1)
                                value = cont_match.group(2).strip()
                                symbol = Symbol(
                                    name=name,
                                    symbol_type=TALType.LITERAL,
                                    location=SourceLocation(filename, i + 1),
                                    value=value,
                                    definition=f"LITERAL {name} = {value}"
                                )
                                self.symbol_table.declare(symbol)
                            break
            
            i += 1
    
    def _extract_globals(self, lines: List[str], filename: str):
        """Extract global variable declarations"""
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Skip if inside a procedure (simple heuristic)
            # This will be refined in pass 2
            
            # Match: INT name; or STRING name[size];
            match = re.match(
                r'^\s*(INT(?:\([^)]*\))?|REAL(?:\([^)]*\))?|STRING|FIXED|UNSIGNED(?:\([^)]*\))?)\s+'
                r'(\.)?([A-Za-z_][A-Za-z0-9_]*)(?:\[(\d+)\])?',
                line, re.IGNORECASE
            )
            
            if match:
                # Skip if this looks like it's inside a procedure
                if not self._is_inside_procedure(lines, i):
                    var_type = TALType.from_string(match.group(1))
                    is_pointer = match.group(2) is not None
                    name = match.group(3)
                    array_size = int(match.group(4)) if match.group(4) else 0
                    
                    symbol = Symbol(
                        name=name,
                        symbol_type=var_type,
                        location=SourceLocation(filename, i + 1),
                        is_pointer=is_pointer,
                        is_array=array_size > 0,
                        array_size=array_size
                    )
                    self.symbol_table.declare(symbol)
            
            i += 1
    
    def _is_inside_procedure(self, lines: List[str], current_line: int) -> bool:
        """Check if current line is inside a procedure"""
        depth = 0
        for i in range(current_line - 1, -1, -1):
            line = lines[i]
            
            # Count BEGIN/END
            if re.search(r'\bEND\b', line, re.IGNORECASE):
                depth += 1
            if re.search(r'\bBEGIN\b', line, re.IGNORECASE):
                depth -= 1
            
            # Found PROC/SUBPROC before balancing
            if depth < 0 and re.search(r'\b(PROC|SUBPROC)\b', line, re.IGNORECASE):
                return True
        
        return False
    
    # =========================================================================
    # PASS 2: Procedure Parsing
    # =========================================================================
    
    def _pass2_procedures(self, filepath: Path):
        """Second pass: parse procedure declarations and bodies"""
        try:
            content = filepath.read_text(encoding='utf-8', errors='replace')
        except Exception as e:
            return
        
        filename = filepath.name
        lines = content.split('\n')
        
        # Find all procedure declarations
        proc_locations = self._find_procedure_locations(lines, filename)
        
        # Parse each procedure
        for proc_start, proc_type, proc_name, return_type in proc_locations:
            proc = self._parse_procedure(lines, proc_start, proc_type, proc_name, 
                                        return_type, filename)
            if proc:
                self.procedures.append(proc)
                self.business_rules.extend(proc.business_rules)
    
    def _find_procedure_locations(self, lines: List[str], filename: str) -> List[Tuple[int, ProcType, str, Optional[TALType]]]:
        """Find all procedure declaration start lines"""
        locations = []
        
        for i, line in enumerate(lines):
            # Remove comments
            comment_pos = line.find('!')
            if comment_pos >= 0:
                code = line[:comment_pos]
            else:
                code = line
            
            match = self.PROC_PATTERN.search(code)
            if match:
                return_type_str = match.group(1)
                proc_type_str = match.group(2).upper()
                proc_name = match.group(3)
                
                return_type = TALType.from_string(return_type_str) if return_type_str else None
                proc_type = ProcType.PROC if proc_type_str == 'PROC' else ProcType.SUBPROC
                
                locations.append((i, proc_type, proc_name, return_type))
        
        return locations
    
    def _parse_procedure(self, lines: List[str], start_line: int, proc_type: ProcType,
                        proc_name: str, return_type: Optional[TALType], 
                        filename: str) -> Optional[TALProcedure]:
        """Parse a complete procedure"""
        location = SourceLocation(filename, start_line + 1)
        
        # Create procedure object
        proc = TALProcedure(
            name=proc_name,
            proc_type=proc_type,
            location=location,
            return_type=return_type
        )
        
        # Find procedure end
        end_line = self._find_procedure_end(lines, start_line, proc_type == ProcType.SUBPROC)
        location.end_line = end_line + 1
        
        # Extract raw code
        raw_code_lines = lines[start_line:end_line + 1]
        proc.raw_code = '\n'.join(raw_code_lines)
        
        # Normalize code (remove comments)
        proc.normalized_code = self._normalize_code(proc.raw_code)
        
        # Parse procedure signature (parameters and attributes)
        self._parse_signature(proc, lines, start_line)
        
        # Extract procedure body content
        self._analyze_procedure_body(proc)
        
        return proc
    
    def _find_procedure_end(self, lines: List[str], start_line: int, is_subproc: bool) -> int:
        """Find the end line of a procedure"""
        depth = 0
        found_begin = False
        in_struct = False
        struct_depth = 0
        
        for i in range(start_line, len(lines)):
            line = lines[i]
            
            # Remove comments
            comment_pos = line.find('!')
            if comment_pos >= 0:
                code = line[:comment_pos]
            else:
                code = line
            
            code_upper = code.upper()
            
            # For SUBPROCs, stop at next PROC/SUBPROC definition
            if is_subproc and i > start_line:
                if re.search(r'\b(PROC|SUBPROC)\s+[A-Za-z_]', code, re.IGNORECASE):
                    return i - 1
            
            # Track STRUCT blocks (they have their own BEGIN/END)
            if re.search(r'\bSTRUCT\s+[A-Za-z_][A-Za-z0-9_]*\s*(\(\s*\*\s*\))?\s*;', code, re.IGNORECASE):
                in_struct = True
                struct_depth = 0
                continue
            
            if in_struct:
                if 'BEGIN' in code_upper:
                    struct_depth += 1
                if 'END' in code_upper:
                    struct_depth -= 1
                    if struct_depth <= 0:
                        in_struct = False
                continue
            
            # Track procedure BEGIN/END
            if 'BEGIN' in code_upper:
                begin_count = len(re.findall(r'\bBEGIN\b', code_upper))
                depth += begin_count
                found_begin = True
            
            if 'END' in code_upper:
                end_count = len(re.findall(r'\bEND\b', code_upper))
                depth -= end_count
            
            # Procedure ends when depth returns to 0
            if found_begin and depth <= 0:
                return i
        
        return len(lines) - 1
    
    def _normalize_code(self, raw_code: str) -> str:
        """Remove comments from code"""
        lines = raw_code.split('\n')
        normalized = []
        
        for line in lines:
            # Remove inline comments (but preserve strings)
            result = []
            in_string = False
            i = 0
            while i < len(line):
                char = line[i]
                if char == '"' and (i == 0 or line[i-1] != '\\'):
                    in_string = not in_string
                    result.append(char)
                elif char == '!' and not in_string:
                    break  # Rest is comment
                else:
                    result.append(char)
                i += 1
            
            normalized_line = ''.join(result).rstrip()
            if normalized_line:
                normalized.append(normalized_line)
        
        return '\n'.join(normalized)
    
    def _parse_signature(self, proc: TALProcedure, lines: List[str], start_line: int):
        """Parse procedure signature (parameters and attributes)"""
        # Collect declaration lines until semicolon
        decl_lines = []
        for i in range(start_line, min(start_line + 20, len(lines))):
            line = lines[i]
            decl_lines.append(line)
            
            # Remove comments and check for semicolon
            comment_pos = line.find('!')
            if comment_pos >= 0:
                code = line[:comment_pos]
            else:
                code = line
            
            if ';' in code:
                break
        
        declaration = '\n'.join(decl_lines)
        declaration_upper = declaration.upper()
        
        # Check for attributes
        proc.is_main = 'MAIN' in declaration_upper
        proc.is_forward = 'FORWARD' in declaration_upper
        proc.is_external = 'EXTERNAL' in declaration_upper
        proc.is_variable = 'VARIABLE' in declaration_upper
        proc.is_extensible = 'EXTENSIBLE' in declaration_upper
        
        # Extract parameters
        paren_match = re.search(r'\(([^)]*)\)', declaration)
        if paren_match:
            param_str = paren_match.group(1)
            param_names = [p.strip() for p in param_str.split(',') if p.strip()]
            
            for param_name in param_names:
                is_pointer = param_name.startswith('.')
                clean_name = param_name.lstrip('.')
                
                param = Symbol(
                    name=clean_name,
                    symbol_type=TALType.UNKNOWN,
                    location=proc.location,
                    is_pointer=is_pointer
                )
                proc.parameters.append(param)
        
        # Parse parameter type declarations after signature
        self._parse_parameter_types(proc, lines, start_line)
    
    def _parse_parameter_types(self, proc: TALProcedure, lines: List[str], start_line: int):
        """Parse parameter type declarations after procedure signature"""
        # Look for type declarations after PROC line, before BEGIN
        param_names = {p.name.upper() for p in proc.parameters}
        
        for i in range(start_line + 1, min(start_line + 30, len(lines))):
            line = lines[i]
            
            # Stop at BEGIN
            if re.search(r'\bBEGIN\b', line, re.IGNORECASE):
                break
            
            # Remove comments
            comment_pos = line.find('!')
            if comment_pos >= 0:
                code = line[:comment_pos]
            else:
                code = line
            
            # Match type declarations
            match = re.match(
                r'^\s*(INT(?:\([^)]*\))?|REAL(?:\([^)]*\))?|STRING|FIXED|UNSIGNED(?:\([^)]*\))?|STRUCT)\s+(.+)',
                code, re.IGNORECASE
            )
            
            if match:
                type_str = match.group(1)
                var_part = match.group(2).rstrip(';').strip()
                data_type = TALType.from_string(type_str)
                
                # Parse variable names
                for var in var_part.split(','):
                    var = var.strip()
                    is_pointer = var.startswith('.')
                    var_name = var.lstrip('.').split('[')[0].split('(')[0].strip()
                    
                    # Update parameter type if it matches
                    if var_name.upper() in param_names:
                        for param in proc.parameters:
                            if param.name.upper() == var_name.upper():
                                param.symbol_type = data_type
                                param.is_pointer = is_pointer
    
    def _analyze_procedure_body(self, proc: TALProcedure):
        """Analyze procedure body for calls, data refs, rules, etc."""
        code = proc.normalized_code
        raw_code = proc.raw_code
        
        # Extract procedure calls
        proc.calls = self._extract_calls(code, proc)
        
        # Extract data references
        proc.data_refs = self._extract_data_refs(code)
        
        # Extract validation steps from comments
        proc.validation_steps = self._extract_validation_steps(raw_code)
        
        # Extract error codes
        proc.error_codes = self._extract_error_codes(raw_code)
        
        # Map error codes to steps
        self._map_errors_to_steps(proc)
        
        # Extract business rules
        proc.business_rules = self._extract_business_rules(code, proc)
    
    def _extract_calls(self, code: str, proc: TALProcedure) -> List[ProcedureCall]:
        """Extract procedure calls from code"""
        calls = []
        seen = set()
        
        # Pattern: CALL procname or CALL procname(
        call_pattern = re.compile(
            r'\bCALL\s+([A-Za-z_][A-Za-z0-9_^]*)',
            re.IGNORECASE
        )
        
        for i, line in enumerate(code.split('\n')):
            for match in call_pattern.finditer(line):
                target = match.group(1).upper()
                
                # Skip duplicates and self-calls
                if target not in seen and target != proc.name.upper():
                    seen.add(target)
                    
                    call = ProcedureCall(
                        target=target,
                        location=SourceLocation(proc.location.filename, 
                                               proc.location.line + i),
                        context=line.strip()[:100]
                    )
                    calls.append(call)
        
        return calls
    
    def _extract_data_refs(self, code: str) -> List[str]:
        """Extract references to known data symbols"""
        refs = set()
        
        # Get all known symbols
        known_symbols = {s.name.upper() for s in self.symbol_table.get_all_symbols()}
        known_structs = {s.upper() for s in self.symbol_table.structs.keys()}
        
        # Find identifiers in code
        identifiers = re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*)\b', code)
        
        for ident in identifiers:
            ident_upper = ident.upper()
            if ident_upper in known_symbols or ident_upper in known_structs:
                refs.add(ident_upper)
        
        return sorted(refs)
    
    def _extract_validation_steps(self, raw_code: str) -> List[ValidationStep]:
        """Extract validation steps from code comments"""
        steps = []
        
        # Pattern: ! STEP N: description or !* STEP N: description
        step_pattern = re.compile(
            r'!\s*=*\s*\*?\s*STEP\s+(\d+)\s*[:\-]\s*(.+?)(?:\s*=*\s*)?$',
            re.IGNORECASE | re.MULTILINE
        )
        
        lines = raw_code.split('\n')
        
        for i, line in enumerate(lines):
            match = step_pattern.search(line)
            if match:
                step_num = int(match.group(1))
                description = match.group(2).strip().rstrip('=').strip()
                
                # Find calls and errors within this step
                step_calls = []
                step_errors = []
                
                # Look ahead until next step or end
                for j in range(i + 1, len(lines)):
                    next_line = lines[j]
                    
                    # Stop at next step
                    if step_pattern.search(next_line):
                        break
                    
                    # Find calls
                    call_matches = re.findall(r'\bCALL\s+([A-Za-z_][A-Za-z0-9_^]*)', 
                                             next_line, re.IGNORECASE)
                    step_calls.extend(call_matches)
                    
                    # Find error codes
                    error_matches = re.findall(r'"([A-Z]{2,4}\d{2,4})"', next_line)
                    step_errors.extend(error_matches)
                
                step = ValidationStep(
                    step_num=step_num,
                    description=description,
                    calls=list(dict.fromkeys(step_calls)),  # Remove duplicates, keep order
                    error_codes=list(dict.fromkeys(step_errors))
                )
                steps.append(step)
        
        # Sort by step number
        steps.sort(key=lambda s: s.step_num)
        
        return steps
    
    def _extract_error_codes(self, raw_code: str) -> List[ErrorCode]:
        """Extract error/warning codes with messages"""
        errors = []
        seen = set()
        
        # Pattern: ADD_VALIDATION_ERROR("CODE", "message")
        error_pattern = re.compile(
            r'ADD_VALIDATION_ERROR\s*\(\s*"([A-Z]{2,4}\d{2,4})"\s*,\s*"([^"]+)"',
            re.IGNORECASE
        )
        
        # Pattern: ADD_VALIDATION_WARNING("CODE", "message")
        warning_pattern = re.compile(
            r'ADD_VALIDATION_WARNING\s*\(\s*"([A-Z]{2,4}\d{2,4})"\s*,\s*"([^"]+)"',
            re.IGNORECASE
        )
        
        for match in error_pattern.finditer(raw_code):
            code = match.group(1)
            if code not in seen:
                seen.add(code)
                errors.append(ErrorCode(code, match.group(2), "ERROR"))
        
        for match in warning_pattern.finditer(raw_code):
            code = match.group(1)
            if code not in seen:
                seen.add(code)
                errors.append(ErrorCode(code, match.group(2), "WARNING"))
        
        return errors
    
    def _map_errors_to_steps(self, proc: TALProcedure):
        """Map error codes to their validation steps"""
        error_to_step: Dict[str, int] = {}
        
        for step in proc.validation_steps:
            for code in step.error_codes:
                error_to_step[code] = step.step_num
        
        # Add step info to error codes
        for error in proc.error_codes:
            if error.code in error_to_step:
                # Could add step_num attribute here if needed
                pass
    
    def _extract_business_rules(self, code: str, proc: TALProcedure) -> List[BusinessRule]:
        """Extract business rules from code"""
        rules = []
        lines = code.split('\n')
        
        for i, line in enumerate(lines):
            line_upper = line.upper()
            location = SourceLocation(proc.location.filename, proc.location.line + i)
            
            # Validation rules (IF with error)
            if 'IF' in line_upper and ('ERROR' in line_upper or 'VALID' in line_upper):
                rules.append(BusinessRule(
                    rule_type='VALIDATION',
                    source_code=line.strip()[:200],
                    procedure=proc.name,
                    location=location
                ))
            
            # Threshold checks
            elif re.search(r'>\s*(THRESHOLD|LIMIT|MAX|MIN)', line_upper):
                rules.append(BusinessRule(
                    rule_type='THRESHOLD',
                    source_code=line.strip()[:200],
                    procedure=proc.name,
                    location=location
                ))
            
            # Compliance checks
            elif any(kw in line_upper for kw in ['OFAC', 'AML', 'KYC', 'SANCTION', 'COMPLIANCE']):
                rules.append(BusinessRule(
                    rule_type='COMPLIANCE',
                    source_code=line.strip()[:200],
                    procedure=proc.name,
                    location=location
                ))
            
            # Amount checks
            elif re.search(r'AMOUNT\s*[<>=]', line_upper):
                rules.append(BusinessRule(
                    rule_type='AMOUNT_CHECK',
                    source_code=line.strip()[:200],
                    procedure=proc.name,
                    location=location
                ))
        
        return rules
    
    # =========================================================================
    # PASS 3: Call Graph
    # =========================================================================
    
    def _pass3_call_graph(self):
        """Build complete call graph"""
        # Initialize call graph entries
        for proc in self.procedures:
            name = proc.name.upper()
            self.call_graph[name] = {
                'file': proc.location.filename,
                'proc_type': proc.proc_type.value,
                'callees': [c.target for c in proc.calls],
                'callers': []
            }
        
        # Build reverse index (callers)
        for proc in self.procedures:
            caller_name = proc.name.upper()
            for call in proc.calls:
                callee_name = call.target.upper()
                if callee_name in self.call_graph:
                    if caller_name not in self.call_graph[callee_name]['callers']:
                        self.call_graph[callee_name]['callers'].append(caller_name)
        
        # Build data usage index
        for proc in self.procedures:
            for data_ref in proc.data_refs:
                self.data_usage[data_ref].append({
                    'procedure': proc.name,
                    'file': proc.location.filename,
                    'proc_type': proc.proc_type.value
                })
    
    # =========================================================================
    # SAVE INDEX
    # =========================================================================
    
    def _save_index(self):
        """Save all index data to JSON files"""
        print("\n📁 Saving index...")
        
        # Save procedures
        proc_data = [p.to_dict() for p in self.procedures]
        self._save_json('procedures.json', proc_data)
        
        # Save symbols
        symbol_data = [s.to_dict() for s in self.symbol_table.get_all_global_symbols()]
        self._save_json('symbols.json', symbol_data)
        
        # Save structs
        struct_data = [s.to_dict() for s in self.symbol_table.structs.values()]
        self._save_json('structs.json', struct_data)
        
        # Save call graph
        self._save_json('call_graph.json', self.call_graph)
        
        # Save data usage
        self._save_json('data_usage.json', dict(self.data_usage))
        
        # Save business rules
        rule_data = [r.to_dict() for r in self.business_rules]
        self._save_json('business_rules.json', rule_data)
        
        print(f"   Saved {len(proc_data)} procedures")
        print(f"   Saved {len(symbol_data)} symbols")
        print(f"   Saved {len(struct_data)} structs")
        print(f"   Saved {len(self.call_graph)} call graph entries")
        print(f"   Saved {len(rule_data)} business rules")
    
    def _save_json(self, filename: str, data: Any):
        """Save data to JSON file"""
        path = os.path.join(self.db_path, filename)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    # =========================================================================
    # VECTOR EMBEDDINGS
    # =========================================================================
    
    def _create_embeddings(self):
        """Create vector embeddings for semantic search"""
        print("\n🔢 Creating vector embeddings...")
        
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            client = QdrantClient(path=self.db_path)
            
            # Create procedures collection
            collection_name = "tal_logic"
            
            try:
                client.delete_collection(collection_name)
            except:
                pass
            
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
            
            # Create embeddings for procedures
            points = []
            for i, proc in enumerate(self.procedures):
                # Create searchable text
                text_parts = [
                    f"procedure {proc.name}",
                    f"type {proc.proc_type.value}",
                ]
                
                if proc.parameters:
                    text_parts.append(f"parameters {' '.join(p.name for p in proc.parameters)}")
                
                for step in proc.validation_steps:
                    text_parts.append(f"step {step.step_num} {step.description}")
                
                text = ' '.join(text_parts)
                embedding = model.encode(text).tolist()
                
                points.append(PointStruct(
                    id=i,
                    vector=embedding,
                    payload={
                        'name': proc.name,
                        'file': proc.location.filename,
                        'proc_type': proc.proc_type.value,
                        'text': text[:500]
                    }
                ))
            
            if points:
                client.upsert(collection_name=collection_name, points=points)
                print(f"   Created {len(points)} procedure embeddings")
            
        except Exception as e:
            print(f"   Warning: Could not create embeddings: {e}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='TAL Intelligence System - Enhanced Indexer v3.0',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tal_indexer_v3.py --code-dir ./tal_source --db-path ./tal_index
  python tal_indexer_v3.py -c /path/to/code -d /path/to/index

Features:
  - Proper AST with TALNode structure
  - Strong typing with TALType enum
  - Scoped symbol table
  - Validation step extraction
  - Error code mapping
  - Call graph analysis
  - Business rule detection
        """
    )
    
    parser.add_argument('-c', '--code-dir', required=True,
                       help='Directory containing TAL source files')
    parser.add_argument('-d', '--db-path', required=True,
                       help='Output directory for index files')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("TAL Intelligence System - Enhanced Indexer v3.0")
    print("=" * 60)
    
    indexer = TALIndexer(args.db_path)
    indexer.index_directory(args.code_dir)


if __name__ == '__main__':
    main()
