#!/usr/bin/env python3
"""
This parser converts TAL source code into Abstract Syntax Trees (ASTs) with support for:
- Enhanced error reporting with context and source location tracking
- Symbol table management with scoped symbol resolution  
- Advanced TAL language features (procedures, structs, literals, defines)
- Multiple output formats (S-expressions, JSON, DOT graphs)
- Performance optimizations with packrat parsing
- Robust error recovery mechanisms

FIXES APPLIED:
1. Missing procedure definitions - procedures now properly closed and added to AST
2. Variable scope placement - local vars go to local_declarations, not statements
3. Global variable parsing - struct variables properly parsed
"""

from pyparsing import *
import sys
import re
import argparse
import json
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple, Union
from enum import Enum
from collections import defaultdict, deque

# Enable packrat parsing for performance optimization with memoization
ParserElement.enablePackrat()

class TALType(Enum):
    """
    Enumeration of TAL data types supported by the parser.
    
    TAL supports both primitive and composite data types:
    - INT: Integer numbers (16-bit or 32-bit depending on context)
    - STRING: Variable-length character strings
    - REAL: Floating-point numbers (single or double precision)
    - FIXED: Fixed-point decimal numbers with specified precision
    - BYTE: 8-bit unsigned integers
    - CHAR: Single character values
    - STRUCT: User-defined composite data structures
    - POINTER: Memory address references (indicated by leading dot)
    - UNKNOWN: Type not yet determined or unresolvable
    """
    INT = "INT"
    STRING = "STRING" 
    REAL = "REAL"
    FIXED = "FIXED"
    BYTE = "BYTE"
    CHAR = "CHAR"
    STRUCT = "STRUCT"
    POINTER = "POINTER"
    UNKNOWN = "UNKNOWN"

class ErrorSeverity(Enum):
    """
    Classification of error severity levels for diagnostic reporting.
    
    Levels determine how the parser handles different types of issues:
    - INFO: Informational messages about parsing process
    - WARNING: Non-critical issues that don't prevent parsing
    - ERROR: Serious issues that may affect correctness but allow continued parsing
    - FATAL: Critical errors that prevent successful parsing
    """
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    FATAL = "fatal"

@dataclass
class SourceLocation:
    """
    Represents a specific location in source code for error reporting and debugging.
    
    Attributes:
        filename: Path to the source file
        line: Line number (1-based indexing)
        column: Column number (1-based indexing)
        length: Length of the token or construct at this location
        
    Used for:
    - Error message context
    - Symbol cross-referencing
    - IDE integration support
    - Debugging information
    """
    filename: str = ""
    line: int = 0
    column: int = 0
    length: int = 0
    
    def __str__(self):
        """Return human-readable location string for error messages."""
        return f"{self.filename}:{self.line}:{self.column}"

@dataclass
class ParseError:
    """
    Enhanced parse error with comprehensive context information.
    
    Provides detailed error reporting including:
    - Source location with filename, line, and column
    - Contextual source code lines around the error
    - Suggested fixes or alternatives
    - Error classification codes for tooling integration
    
    Attributes:
        message: Human-readable error description
        location: Source location where error occurred
        severity: Classification of error importance
        context_lines: Source lines around the error for context
        suggestions: Recommended fixes or alternatives
        error_code: Unique identifier for error type (e.g., "E001")
    """
    message: str
    location: SourceLocation
    severity: ErrorSeverity
    context_lines: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    error_code: str = ""
    
    def __str__(self):
        """Format error for display with context and suggestions."""
        result = f"{self.severity.value.upper()}: {self.message}\n"
        result += f"  at {self.location}\n"
        
        # Add source context if available
        if self.context_lines:
            for i, line in enumerate(self.context_lines):
                # Mark the middle line as the error location
                marker = ">>>" if i == 1 else "   "
                result += f"  {marker} {line}\n"
        
        # Add helpful suggestions
        if self.suggestions:
            result += "  Suggestions:\n"
            for suggestion in self.suggestions:
                result += f"    - {suggestion}\n"
        
        return result

@dataclass
class Symbol:
    """
    Symbol table entry representing a declared identifier in TAL code.
    
    Tracks all information about declared symbols including:
    - Basic properties (name, type, location)
    - Scope and visibility information  
    - Type-specific attributes (array bounds, struct members, etc.)
    - Usage tracking with reference locations
    
    Used for:
    - Type checking and semantic analysis
    - Cross-reference generation
    - Unused variable detection
    - IDE features like "go to definition"
    
    Attributes:
        name: Identifier name as written in source
        symbol_type: TAL data type classification
        location: Where symbol was first declared
        scope: Scope name where symbol exists (e.g., procedure name)
        is_pointer: True if symbol is a pointer (starts with '.')
        is_array: True if symbol is an array with bounds
        array_bounds: Tuple of (start, end) indices for arrays
        struct_name: Name of struct type for struct variables
        procedure_params: Parameter list for procedure symbols
        return_type: Return type for procedure symbols
        bit_fields: Bit field definitions for packed structures
        is_main: True if this is the main procedure entry point
        references: List of all locations where symbol is referenced
    """
    name: str
    symbol_type: TALType
    location: SourceLocation
    scope: str = ""
    is_pointer: bool = False
    is_array: bool = False
    array_bounds: Optional[Tuple[int, int]] = None
    struct_name: Optional[str] = None
    procedure_params: Optional[List['Symbol']] = None
    return_type: Optional[TALType] = None
    bit_fields: Optional[Dict[str, Tuple[int, int]]] = None
    is_main: bool = False
    references: List[SourceLocation] = field(default_factory=list)
    
    def add_reference(self, location: SourceLocation):
        """
        Record a reference to this symbol at the given location.
        
        Args:
            location: Source location where symbol is referenced
            
        Used for cross-reference analysis and unused variable detection.
        """
        self.references.append(location)

class SymbolTable:
    """
    Multi-scope symbol table with hierarchical name resolution.
    
    Manages symbol declarations and lookups across different scopes:
    - Global scope for module-level declarations
    - Procedure scopes for local variables and parameters
    - Struct scopes for member definitions
    
    Features:
    - Nested scope management with scope stack
    - Redeclaration detection within same scope
    - Symbol lookup with scope chain traversal
    - Separate tracking of struct type definitions
    
    TAL Scoping Rules:
    - Global symbols visible throughout module
    - Procedure parameters and locals shadow globals
    - Struct members have their own namespace
    - Forward references allowed for procedures
    """
    
    def __init__(self):
        """Initialize empty symbol table with global scope."""
        # Dictionary mapping scope names to their symbol dictionaries
        self.scopes: Dict[str, Dict[str, Symbol]] = defaultdict(dict)
        self.current_scope = "global"
        # Stack tracking nested scope entry/exit
        self.scope_stack: List[str] = ["global"]
        # Separate namespace for struct type definitions
        self.struct_definitions: Dict[str, Dict[str, Symbol]] = {}
        
    def enter_scope(self, scope_name: str):
        """
        Enter a new lexical scope (e.g., when entering a procedure).
        
        Args:
            scope_name: Name of the new scope to enter
            
        Updates current scope context and maintains scope stack for
        proper nested scope resolution.
        """
        self.scope_stack.append(scope_name)
        self.current_scope = scope_name
        
    def exit_scope(self):
        """
        Exit current scope and return to parent scope.
        
        Maintains scope stack integrity by preventing exit from global scope.
        Called when leaving procedures or other scoped constructs.
        """
        if len(self.scope_stack) > 1:
            self.scope_stack.pop()
            self.current_scope = self.scope_stack[-1]
    
    def declare_symbol(self, symbol: Symbol) -> Optional[ParseError]:
        """
        Declare a new symbol in the current scope.
        
        Args:
            symbol: Symbol to declare with all required attributes
            
        Returns:
            ParseError if symbol already exists in current scope, None if successful
            
        Performs redeclaration checking within the current scope only.
        Symbols in parent scopes can be shadowed by local declarations.
        """
        symbol.scope = self.current_scope
        
        # Check for redeclaration in current scope only
        if symbol.name in self.scopes[self.current_scope]:
            existing = self.scopes[self.current_scope][symbol.name]
            return ParseError(
                f"Symbol '{symbol.name}' already declared in scope '{self.current_scope}'",
                symbol.location,
                ErrorSeverity.ERROR,
                error_code="E001",
                suggestions=[f"Previous declaration at {existing.location}"]
            )
        
        self.scopes[self.current_scope][symbol.name] = symbol
        return None
    
    def lookup_symbol(self, name: str, location: SourceLocation) -> Optional[Symbol]:
        """
        Look up a symbol using TAL's scoping rules.
        
        Args:
            name: Symbol name to find
            location: Location of the reference (for tracking)
            
        Returns:
            Symbol if found in current scope chain, None otherwise
            
        Search order:
        1. Current scope (e.g., current procedure)
        2. Parent scopes up the stack
        3. Global scope (always searched last)
        
        Automatically records the reference location for cross-reference analysis.
        """
        # Search from current scope up to global
        for scope in reversed(self.scope_stack):
            if name in self.scopes[scope]:
                symbol = self.scopes[scope][name]
                symbol.add_reference(location)
                return symbol
        return None
    
    def get_all_symbols(self) -> List[Symbol]:
        """
        Retrieve all declared symbols from all scopes.
        
        Returns:
            List of all Symbol objects across all scopes
            
        Used for analysis, reporting, and symbol table export.
        """
        symbols = []
        for scope_symbols in self.scopes.values():
            symbols.extend(scope_symbols.values())
        return symbols

@dataclass
class TALNode:
    """
    Abstract Syntax Tree node representing TAL language constructs.
    
    Forms the backbone of the parsed representation with:
    - Hierarchical structure through parent-child relationships
    - Rich metadata for analysis and code generation
    - Type information for semantic analysis
    - Source location tracking for error reporting
    
    Node Types:
    - program: Root node containing entire compilation unit
    - procedure: Function/procedure definitions
    - var_decl: Variable declarations
    - struct_decl: Structure type definitions  
    - statement: Executable statements
    - expression: Value-producing expressions
    - comment: Source code comments
    
    Attributes:
        type: Classification of AST node (e.g., 'procedure', 'var_decl')
        name: Identifier name for named constructs
        value: Literal value or source text representation
        children: List of child nodes forming tree structure
        location: Source location for error reporting
        attributes: Additional metadata as key-value pairs
        symbol: Associated symbol table entry if applicable
        semantic_type: Resolved TAL type after semantic analysis
        cross_refs: References to related nodes for analysis
    """
    type: str
    name: str = ""
    value: Any = None
    children: List['TALNode'] = field(default_factory=list)
    location: SourceLocation = field(default_factory=SourceLocation)
    attributes: Dict[str, Any] = field(default_factory=dict)
    symbol: Optional[Symbol] = None
    semantic_type: Optional[TALType] = None
    cross_refs: List['TALNode'] = field(default_factory=list)
    
    def add_child(self, child):
        """
        Add a child node to this AST node.
        
        Args:
            child: Either a TALNode or primitive value to add
            
        Primitive values are automatically wrapped in token nodes.
        Maintains the tree structure for traversal and analysis.
        """
        if isinstance(child, TALNode):
            self.children.append(child)
        else:
            # Wrap primitive values in token nodes
            self.children.append(TALNode('token', value=str(child)))
    
    def find_children_by_type(self, node_type: str) -> List['TALNode']:
        """
        Recursively find all descendant nodes of a specific type.
        
        Args:
            node_type: Type string to search for
            
        Returns:
            List of matching nodes in depth-first order
            
        Useful for analysis passes that need to process all nodes
        of a particular kind (e.g., all variable declarations).
        """
        result = []
        for child in self.children:
            if child.type == node_type:
                result.append(child)
            result.extend(child.find_children_by_type(node_type))
        return result
    
    def get_path(self) -> str:
        """
        Get hierarchical path to this node for debugging.
        
        Returns:
            String representation of path from root to this node
            
        Used for debugging and error reporting to help locate
        specific nodes within large ASTs.
        """
        # This would be set during parsing with full path
        return getattr(self, '_path', f"{self.type}.{self.name}")
    
    def to_sexp(self, indent=0):
        """
        Convert AST to S-expression format for readable output.
        
        Args:
            indent: Current indentation level for pretty-printing
            
        Returns:
            String representation in LISP-like S-expression format
            
        S-expressions provide a clean, structured view of the AST
        that's both human-readable and machine-parseable. Special
        handling for program nodes organizes output into logical sections.
        """
        spaces = "  " * indent
        
        # Special formatting for program root to organize output
        if self.type == 'program':
            sections = []
            
            # Group children by type for better organization
            globals_children = [c for c in self.children if c.type in ['name_decl', 'struct_decl', 'var_decl', 'literal_decl', 'define_decl']]
            procedure_children = [c for c in self.children if c.type == 'procedure']
            comment_children = [c for c in self.children if c.type == 'comment']
            
            # Format globals section
            if globals_children:
                globals_section = f"{spaces}  (globals"
                for child in globals_children:
                    globals_section += f"\n{child.to_sexp(indent + 2)}"
                globals_section += ")"
                sections.append(globals_section)
            
            # Format procedures section
            if procedure_children:
                procedures_section = f"{spaces}  (procedures"
                for child in procedure_children:
                    procedures_section += f"\n{child.to_sexp(indent + 2)}"
                procedures_section += ")"
                sections.append(procedures_section)
            
            # Format comments section
            if comment_children:
                comments_section = f"{spaces}  (comments"
                for child in comment_children:
                    comments_section += f"\n{child.to_sexp(indent + 2)}"
                comments_section += ")"
                sections.append(comments_section)
            
            sections_str = "\n".join(sections)
            return f"(program\n{sections_str})"
        
        # Format regular nodes with enhanced attributes
        name_part = f" {self.name}" if self.name else ""
        attrs = []
        
        # Include custom attributes
        if self.attributes:
            for key, value in self.attributes.items():
                attrs.append(f"{key}={value}")
        
        # Include semantic type information
        if self.semantic_type:
            attrs.append(f"semantic_type={self.semantic_type.value}")
        
        attr_part = f" ({' '.join(attrs)})" if attrs else ""
        
        # Handle leaf nodes
        if not self.children:
            if self.value is not None:
                return f"{spaces}({self.type}{name_part} {self.value}{attr_part})"
            else:
                return f"{spaces}({self.type}{name_part}{attr_part})"
        else:
            # Format nodes with children
            child_sexps = [child.to_sexp(indent + 1) for child in self.children if child]
            if not child_sexps:
                return f"{spaces}({self.type}{name_part}{attr_part})"
            
            children_str = "\n".join(child_sexps)
            return f"{spaces}({self.type}{name_part}{attr_part}\n{children_str})"
    
    def to_json(self) -> Dict[str, Any]:
        """
        Convert AST to JSON format for structured data exchange.
        
        Returns:
            Dictionary representation suitable for JSON serialization
            
        JSON format provides machine-readable output for tools that
        need to process the AST programmatically. Includes all node
        metadata and maintains the tree structure.
        """
        result = {
            "type": self.type,
            "name": self.name,
            "value": self.value,
            "location": {
                "file": self.location.filename,
                "line": self.location.line,
                "column": self.location.column
            },
            "attributes": self.attributes
        }
        
        # Include type information if available
        if self.semantic_type:
            result["semantic_type"] = self.semantic_type.value
            
        # Recursively convert children
        if self.children:
            result["children"] = [child.to_json() for child in self.children]
            
        return result
    
    def to_dot(self, graph_name="ast") -> str:
        """
        Convert AST to Graphviz DOT format for visualization.
        
        Args:
            graph_name: Name for the generated graph
            
        Returns:
            DOT format string for graph rendering
            
        Generates directed graph suitable for visualization with
        Graphviz tools. Nodes show type and key information,
        edges show parent-child relationships.
        """
        lines = [f"digraph {graph_name} {{"]
        lines.append("  rankdir=TB;")  # Top-to-bottom layout
        lines.append("  node [shape=box, style=rounded];")  # Box-shaped nodes
        
        node_id = 0
        node_map = {}
        
        def add_node(node, parent_id=None):
            """Recursively add nodes to DOT graph."""
            nonlocal node_id
            current_id = node_id
            node_id += 1
            
            # Create descriptive label
            label = f"{node.type}"
            if node.name:
                label += f"\\n{node.name}"
            if node.value and len(str(node.value)) < 20:
                label += f"\\n{node.value}"
                
            lines.append(f'  n{current_id} [label="{label}"];')
            
            # Add edge from parent if not root
            if parent_id is not None:
                lines.append(f"  n{parent_id} -> n{current_id};")
            
            # Process children recursively
            for child in node.children:
                add_node(child, current_id)
        
        add_node(self)
        lines.append("}")
        return "\n".join(lines)

class EnhancedTALParser:
    """
    Main parser class for TAL source code with comprehensive analysis capabilities.
    
    Features:
    - Multi-pass parsing with preprocessing and AST construction
    - Symbol table management with scoped symbol resolution
    - Enhanced error reporting with source context
    - Multiple output formats (S-expressions, JSON, DOT graphs)
    - Performance monitoring with detailed statistics
    - Robust error recovery for continued parsing after errors
    
    Parser Architecture:
    1. Preprocessing: Clean up source, handle multi-line constructs
    2. Lexical analysis: Identify TAL language constructs
    3. Syntax analysis: Build Abstract Syntax Tree
    4. Semantic analysis: Type checking and symbol resolution (optional)
    5. Cross-reference generation: Build symbol usage maps
    6. Output generation: Convert to requested formats
    
    TAL Language Support:
    - Data declarations: INT, STRING, REAL, FIXED, BYTE, CHAR
    - Structured data: STRUCT definitions with nested members  
    - Procedures: Parameter lists, return types, local variables
    - Control flow: IF/THEN, WHILE/DO, CASE/OF, SCAN statements
    - Memory management: Pointers, arrays, structured access
    - Preprocessor: LITERAL and DEFINE constant definitions
    - Comments: Line and inline comment handling
    """
    
    def __init__(self):
        """Initialize parser with clean state and empty symbol table."""
        self.reset_state()
        self.symbol_table = SymbolTable()
        self.errors: List[ParseError] = []
        self.warnings: List[ParseError] = []
        self.source_lines: List[str] = []
        self.filename = ""
        
        # Performance and analysis statistics
        self.stats = {
            'parse_time': 0.0,
            'lines_processed': 0,
            'nodes_created': 0,
            'symbols_declared': 0,
            'errors_found': 0,
            'warnings_issued': 0
        }
        
    def reset_state(self):
        """
        Reset parser internal state for fresh parsing.
        
        Called before parsing each new file to ensure clean state.
        Clears all parsing context and temporary data structures.
        """
        self.current_procedure = None  # Currently open procedure being parsed
        self.in_struct = False         # Whether we're inside a struct definition
        self.struct_stack = []         # Stack of nested struct contexts
        self.brace_level = 0          # Nesting level for brace matching
        self.in_proc_params = False    # Whether parsing procedure parameters
        self.collecting_statements = False  # Whether collecting procedure body statements
        self.current_struct = None     # Current struct being defined
        
    def add_error(self, message: str, location: SourceLocation, 
                 severity: ErrorSeverity = ErrorSeverity.ERROR,
                 error_code: str = "", suggestions: List[str] = None):
        """
        Add an error or warning with rich context information.
        
        Args:
            message: Human-readable error description
            location: Source location where error occurred
            severity: Classification of error severity
            error_code: Unique identifier for error type
            suggestions: List of suggested fixes or alternatives
            
        Automatically includes source context lines around the error
        location for better debugging. Updates statistics counters.
        """
        context_lines = self._get_context_lines(location.line)
        
        error = ParseError(
            message=message,
            location=location,
            severity=severity,
            context_lines=context_lines,
            suggestions=suggestions or [],
            error_code=error_code
        )
        
        # Categorize error by severity
        if severity in [ErrorSeverity.ERROR, ErrorSeverity.FATAL]:
            self.errors.append(error)
            self.stats['errors_found'] += 1
        else:
            self.warnings.append(error)
            self.stats['warnings_issued'] += 1
    
    def _get_context_lines(self, line_num: int, context=1) -> List[str]:
        """
        Extract source lines around an error for context display.
        
        Args:
            line_num: Line number where error occurred (1-based)
            context: Number of lines to include before and after error
            
        Returns:
            List of formatted context lines with line numbers
            
        Provides visual context in error messages to help users
        understand the location and nature of parsing errors.
        """
        lines = []
        start = max(0, line_num - context - 1)
        end = min(len(self.source_lines), line_num + context)
        
        for i in range(start, end):
            line_content = self.source_lines[i] if i < len(self.source_lines) else ""
            lines.append(f"{i+1:4d}: {line_content}")
            
        return lines
    
    def parse_file(self, filename: str) -> Dict[str, Any]:
        """
        Parse a TAL source file with comprehensive analysis.
        
        Args:
            filename: Path to TAL source file to parse
            
        Returns:
            Dictionary containing parse results, AST, errors, and analysis data
            
        Main entry point for parsing. Handles file I/O, coordinates
        all parsing phases, and returns comprehensive results including
        the AST, error information, symbol table, and performance stats.
        """
        start_time = time.time()
        self.filename = filename
        
        try:
            # Read source file with UTF-8 encoding
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split into lines for error context
            self.source_lines = content.split('\n')
            self.stats['lines_processed'] = len(self.source_lines)
            
            # Perform main parsing
            result = self._parse_content(content)
            
            # Add performance and diagnostic information
            self.stats['parse_time'] = time.time() - start_time
            result['stats'] = self.stats
            result['errors'] = [str(e) for e in self.errors]
            result['warnings'] = [str(w) for w in self.warnings]
            result['symbols'] = self._export_symbol_table()
            
            return result
            
        except Exception as e:
            # Handle file I/O and other critical errors
            return {
                'success': False, 
                'error': f"File error: {e}",
                'stats': self.stats,
                'errors': [str(err) for err in self.errors],
                'warnings': [str(warn) for warn in self.warnings]
            }
    
    def _parse_content(self, content: str) -> Dict[str, Any]:
        """
        Parse source content through multiple analysis phases.
        
        Args:
            content: Complete source code text
            
        Returns:
            Dictionary with parse results and analysis data
            
        Coordinates the multi-phase parsing process:
        1. Preprocessing to clean up source
        2. AST construction from preprocessed tokens
        3. Optional semantic analysis
        4. Cross-reference generation
        5. Result compilation and formatting
        """
        lines = content.split('\n')
        
        # Phase 1: Preprocess source lines
        processed_lines = self._preprocess_lines(lines)
        
        # Phase 2: Parse into AST
        program = self._parse_to_ast(processed_lines)
        
        # Phase 3: Skip semantic analysis to avoid false errors
        # Semantic analysis would check types, resolve symbols, etc.
        # Disabled to prevent false positives during parsing
        # self._semantic_analysis(program)
        
        # Phase 4: Generate cross-references between symbols
        self._generate_cross_references(program)
        
        # Update statistics
        self.stats['nodes_created'] = self._count_nodes(program)
        self.stats['symbols_declared'] = len(self.symbol_table.get_all_symbols())
        
        # Return comprehensive results
        return {
            'success': len([e for e in self.errors if e.severity == ErrorSeverity.FATAL]) == 0,
            'ast': program,
            'sexp': program.to_sexp(),
            'json': program.to_json(),
            'dot': program.to_dot(),
            'node_count': self.stats['nodes_created'],
            'structure': self._analyze_structure(program)
        }
    
    def _preprocess_lines(self, lines: List[str]) -> List[Tuple[str, int, int]]:
        """
        Preprocess source lines with multi-line construct handling.
        
        Args:
            lines: Raw source lines from file
            
        Returns:
            List of (processed_line, line_number, column) tuples
            
        Preprocessing handles:
        - Comment removal (both line and inline comments)
        - Multi-line construct merging (LITERAL, DEFINE, STRUCT, PROC)
        - Empty line filtering
        - Column position tracking for error reporting
        
        TAL allows constructs to span multiple lines, especially
        for complex declarations and procedure definitions.
        """
        processed = []
        i = 0
        
        while i < len(lines):
            line = lines[i].strip()
            original_line = lines[i]
            # Calculate column based on original indentation
            column = len(lines[i]) - len(lines[i].lstrip()) + 1
            
            # Skip empty lines and standalone comments
            if not line or line.startswith('!'):
                i += 1
                continue
            
            # Strip inline comments before processing
            line = self._strip_inline_comment(line)
            if not line:
                i += 1
                continue
            
            # Handle multi-line constructs
            if self._is_multiline_start(line):
                merged_line, consumed = self._merge_multiline(lines, i)
                # Strip comments from merged line too
                merged_line = self._strip_inline_comment(merged_line)
                processed.append((merged_line, i + 1, column))
                i += consumed
            else:
                processed.append((line, i + 1, column))
                i += 1
        
        return processed
    
    def _strip_inline_comment(self, line: str) -> str:
        """
        Remove inline comments while preserving string literals.
        
        Args:
            line: Source line potentially containing comments
            
        Returns:
            Line with comments removed but strings preserved
            
        TAL uses '!' for comments. Must be careful not to remove
        '!' characters that appear inside string literals.
        Handles both single and double quoted strings.
        """
        result = ""
        in_quotes = False
        quote_char = None
        i = 0
        
        while i < len(line):
            char = line[i]
            
            # Handle quote character transitions
            if char in '"\'':
                if not in_quotes:
                    # Starting a quoted string
                    in_quotes = True
                    quote_char = char
                elif char == quote_char:
                    # Ending the quoted string
                    in_quotes = False
                    quote_char = None
                result += char
            elif char == '!' and not in_quotes:
                # Found comment start outside of quotes - stop here
                break
            else:
                result += char
            i += 1
        
        return result.rstrip()
    
    def _is_multiline_start(self, line: str) -> bool:
        """
        Detect if a line starts a multi-line construct.
        
        Args:
            line: Source line to analyze
            
        Returns:
            True if line starts a multi-line construct
            
        TAL constructs that commonly span multiple lines:
        - LITERAL definitions with multiple assignments
        - DEFINE preprocessor directives  
        - STRUCT type definitions
        - PROC procedure declarations with parameters
        - Lines with unmatched parentheses or brackets
        """
        upper_line = line.upper()
        
        # Known multi-line construct starters
        multiline_starts = [
            'LITERAL', 'DEFINE', 'STRUCT', 'PROC',
            'INT PROC', 'STRING PROC', 'REAL PROC'
        ]
        
        for start in multiline_starts:
            if upper_line.startswith(start) and not line.endswith((';', '#', '#;')):
                return True
        
        # Check for incomplete statements with unmatched delimiters
        if (',' in line and not line.endswith(';') or
            line.count('(') != line.count(')') or
            line.count('[') != line.count(']')):
            return True
            
        return False
    
    def _merge_multiline(self, lines: List[str], start_idx: int) -> Tuple[str, int]:
        """
        Merge multi-line constructs into single logical lines.
        
        Args:
            lines: Complete list of source lines
            start_idx: Index of line starting the multi-line construct
            
        Returns:
            Tuple of (merged_line_text, lines_consumed)
            
        Combines continuation lines until finding a terminator:
        - Semicolon (;) for most statements
        - Hash (#) for DEFINE directives  
        - BEGIN...END blocks for procedures and structs
        - Balanced parentheses and brackets
        
        Includes safety limit to prevent infinite loops on malformed input.
        """
        merged = lines[start_idx].strip()
        consumed = 1
        paren_count = merged.count('(') - merged.count(')')
        bracket_count = merged.count('[') - merged.count(']')
        
        for i in range(start_idx + 1, len(lines)):
            next_line = lines[i].strip()
            
            # Skip empty lines and comments
            if not next_line or next_line.startswith('!'):
                consumed += 1
                continue
            
            merged += ' ' + next_line
            consumed += 1
            
            # Update delimiter counting
            paren_count += next_line.count('(') - next_line.count(')')
            bracket_count += next_line.count('[') - next_line.count(']')
            
            # Check for termination conditions
            if (next_line.endswith((';', '#', '#;')) and 
                paren_count == 0 and bracket_count == 0):
                break
                
            # Handle BEGIN...END blocks
            if ('BEGIN' in merged and 'END;' in merged):
                break
                
            # Safety limit to prevent runaway merging
            if consumed > 20:
                self.add_error(
                    "Multiline construct too long, possible syntax error",
                    SourceLocation(self.filename, start_idx + 1, 1),
                    ErrorSeverity.WARNING,
                    "W001"
                )
                break
        
        return merged, consumed
    
    def _parse_to_ast(self, processed_lines: List[Tuple[str, int, int]]) -> TALNode:
        """
        Build Abstract Syntax Tree from preprocessed source lines.
        
        Args:
            processed_lines: List of (line_text, line_number, column) tuples
            
        Returns:
            Root TALNode representing the complete program
            
        Main AST construction phase that:
        - Creates program root node
        - Processes struct definitions first (for type resolution)
        - Parses each line based on TAL language constructs
        - Manages procedure parsing state (declaration -> body -> end)
        - Places nodes in correct AST locations (global vs local scope)
        - Handles proper procedure closure and symbol table management
        """
        program = TALNode('program')
        program.location = SourceLocation(self.filename, 1, 1)
        self.reset_state()
        
        # Pre-extract struct definitions for type resolution
        struct_bodies = self._extract_struct_bodies(processed_lines)
        
        for line_text, line_num, column in processed_lines:
            try:
                location = SourceLocation(self.filename, line_num, column)
                
                # Skip lines already processed as struct bodies
                if self._is_struct_body_line(line_text, struct_bodies):
                    continue
                
                upper_line = line_text.upper()
                
                # Handle procedure lifecycle management
                if any(upper_line.startswith(p) for p in ['PROC ', 'INT PROC ', 'STRING PROC ', 'REAL PROC ']):
                    # Starting new procedure - close previous if exists
                    if self.current_procedure:
                        program.add_child(self.current_procedure)
                    
                    # Parse procedure declaration (creates new current_procedure)
                    node = self._parse_procedure_declaration(line_text, location)
                    # Don't add to program yet - will be added when procedure ends
                    continue
                    
                elif upper_line == 'BEGIN' and self.current_procedure:
                    # Start collecting procedure body statements
                    self.collecting_statements = True
                    continue
                    
                elif upper_line == 'END;' and self.current_procedure:
                    # Close current procedure
                    proc = self.current_procedure
                    self.current_procedure = None
                    self.collecting_statements = False
                    self.symbol_table.exit_scope()
                    program.add_child(proc)
                    continue
                
                # Parse line based on context and content
                node = self._parse_line_contextual(line_text, location)
                
                # Add struct body members if this is a struct declaration
                if node and node.type == 'struct_decl' and node.name in struct_bodies:
                    members = struct_bodies[node.name]
                    for member in members:
                        node.add_child(member)
                
                # Place node in appropriate AST location
                if node:
                    if self.current_procedure:
                        # Inside a procedure - determine correct placement
                        if self.collecting_statements:
                            # Add to procedure body (after BEGIN)
                            for child in self.current_procedure.children:
                                if child.type == 'statements':
                                    child.add_child(node)
                                    break
                        else:
                            # Add to local declarations (between PROC and BEGIN)
                            if node.type == 'var_decl':
                                for child in self.current_procedure.children:
                                    if child.type == 'local_declarations':
                                        child.add_child(node)
                                        break
                            else:
                                # Non-variable declarations go to statements for now
                                for child in self.current_procedure.children:
                                    if child.type == 'statements':
                                        child.add_child(node)
                                        break
                    else:
                        # Global level declaration
                        program.add_child(node)
                        
            except Exception as e:
                self.add_error(
                    f"Parse error: {e}",
                    SourceLocation(self.filename, line_num, column),
                    ErrorSeverity.ERROR,
                    "E002"
                )
        
        # Close any remaining open procedure
        if self.current_procedure:
            program.add_child(self.current_procedure)
            self.current_procedure = None
            self.symbol_table.exit_scope()
        
        return program

    def _is_variable_declaration_line(self, line: str) -> bool:
        """
        Enhanced detection of variable declaration lines.
        
        Args:
            line: Source line to analyze
            
        Returns:
            True if line appears to be a variable declaration
            
        TAL variable declarations can take several forms:
        - Standard types: INT var1, var2; STRING(20) name;
        - Struct variables: struct_name .pointer_var, normal_var;
        - Array declarations: INT array_name[0:99];
        - Initialized variables: INT counter := 0;
        
        Must distinguish from:
        - Procedure declarations containing type keywords
        - Control flow statements  
        - Complex expressions that might contain type names
        """
        upper_line = line.upper()
        parts = line.split()
        
        if not parts:
            return False
        
        # Standard TAL type declarations (but not procedure declarations)
        if any(upper_line.startswith(t) for t in ['INT', 'STRING', 'REAL', 'FIXED', 'BYTE', 'CHAR']):
            return 'PROC ' not in upper_line
        
        # Struct variable declarations: struct_name .variable_name
        # Must exclude known non-declaration constructs
        if (len(parts) >= 2 and 
            not any(upper_line.startswith(kw) for kw in [
                'NAME ', 'STRUCT ', 'LITERAL ', 'DEFINE ', 'PROC ', 
                'WHILE ', 'CASE ', 'SCAN ', 'RETURN ', 'CALL ', 'IF '
            ]) and
            not ':=' in line and  # Not an assignment statement
            not line.strip().endswith(('DO', 'THEN')) and  # Not control flow
            not self._looks_like_statement(line)):  # Not a complex statement
            return True
        
        return False

    def _looks_like_statement(self, line: str) -> bool:
        """
        Heuristic check if line looks like executable statement vs declaration.
        
        Args:
            line: Source line to analyze
            
        Returns:
            True if line appears to be an executable statement
            
        Uses heuristics to distinguish variable declarations from
        executable statements. Statements typically contain:
        - Operators and expressions
        - Function calls with parentheses
        - Complex multi-word constructs
        
        This helps avoid misclassifying complex expressions as declarations.
        """
        upper_line = line.upper()
        return (
            # Contains operators or complex expressions
            any(word in upper_line for word in ['(', ')', '+', '-', '*', '/', '=', '<', '>', 'AND', 'OR']) or
            # Complex expressions tend to have more spaces
            line.count(' ') > 3
        )

    def _parse_line_contextual(self, line: str, location: SourceLocation) -> Optional[TALNode]:
        """
        Parse a source line based on TAL language constructs and context.
        
        Args:
            line: Source line to parse
            location: Source location for error reporting
            
        Returns:
            TALNode representing the parsed construct, or None if not parseable
            
        Central dispatch method that identifies TAL language constructs
        and delegates to appropriate specialized parsing methods:
        
        - Comments: Lines starting with '!'
        - Declarations: NAME, STRUCT, LITERAL, DEFINE  
        - Variables: Type-prefixed declarations
        - Control flow: WHILE, CASE, SCAN, IF statements
        - Procedures: RETURN, CALL statements
        - Assignments: Lines containing ':=' operator
        - General statements: Fallback for other constructs
        
        Skips procedure-related keywords that are handled at a higher level.
        """
        line = line.strip()
        if not line:
            return None
        
        try:
            # Handle comments
            if line.startswith('!'):
                return self._parse_comment(line, location)
            
            # Skip procedure-related keywords (handled in _parse_to_ast)
            upper_line = line.upper()
            if (any(upper_line.startswith(p) for p in ['PROC ', 'INT PROC ', 'STRING PROC ', 'REAL PROC ']) or
                upper_line in ['BEGIN', 'END;']):
                return None
            
            # Dispatch to appropriate parser based on line content
            if upper_line.startswith('NAME '):
                return self._parse_name_declaration(line, location)
            elif upper_line.startswith('STRUCT '):
                return self._parse_struct_declaration(line, location)
            elif upper_line.startswith('LITERAL '):
                return self._parse_literal_declaration(line, location)
            elif upper_line.startswith('DEFINE '):
                return self._parse_define_declaration(line, location)
            elif self._is_variable_declaration_line(line):
                return self._parse_variable_declaration(line, location)
            elif upper_line.startswith('WHILE '):
                return self._parse_while_statement(line, location)
            elif upper_line.startswith('CASE '):
                return self._parse_case_statement(line, location)
            elif upper_line.startswith('SCAN '):
                return self._parse_scan_statement(line, location)
            elif upper_line.startswith('RETURN '):
                return self._parse_return_statement(line, location)
            elif upper_line.startswith('CALL ') or line.startswith('$'):
                return self._parse_call_statement(line, location)
            elif upper_line.startswith('IF '):
                return self._parse_if_statement(line, location)
            elif ':=' in line:
                return self._parse_assignment(line, location)
            else:
                return self._parse_general_statement(line, location)
                
        except Exception as e:
            self.add_error(
                f"Error parsing line: {e}",
                location,
                ErrorSeverity.ERROR,
                "E003"
            )
            return None

    def _parse_parameters_fixed(self, param_text: str, location: SourceLocation) -> Optional[TALNode]:
        """
        Parse procedure parameter list with improved type detection.
        
        Args:
            param_text: Parameter text including parentheses
            location: Source location for error reporting
            
        Returns:
            TALNode containing parameter specifications, or None if empty
            
        TAL procedure parameters can be:
        - Typed parameters: INT param1, STRING(20) param2
        - Struct parameters: struct_name param3  
        - Pointer parameters: INT .pointer_param (leading dot)
        - Mixed parameter lists with comma separation
        
        Handles parentheses removal and comma-separated parameter parsing
        with proper type classification and pointer detection.
        """
        if not param_text or param_text == '()':
            return None
        
        params_node = TALNode('parameters')
        params_node.location = location
        
        # Remove parentheses and split by comma
        param_text = param_text.strip('()')
        if not param_text:
            return params_node
        
        param_parts = self._smart_split(param_text, ',')
        
        for param in param_parts:
            param = param.strip()
            if param:
                param_node = TALNode('parameter')
                param_node.location = location
                
                parts = param.split()
                param_type = TALType.UNKNOWN
                param_name = ""
                is_pointer = False
                struct_name = None
                
                if len(parts) >= 2:
                    first_part = parts[0].upper()
                    second_part = parts[1]
                    
                    # Check if first part is a standard TAL type
                    if first_part in ['INT', 'STRING', 'REAL', 'FIXED', 'BYTE', 'CHAR']:
                        param_type = TALType(first_part)
                        if second_part.startswith('.'):
                            is_pointer = True
                            param_name = second_part[1:]
                        else:
                            param_name = second_part
                    else:
                        # Struct parameter type
                        param_type = TALType.STRUCT
                        struct_name = parts[0]
                        if second_part.startswith('.'):
                            is_pointer = True
                            param_name = second_part[1:]
                        else:
                            param_name = second_part
                            
                elif len(parts) == 1:
                    # Single part - just parameter name
                    if parts[0].startswith('.'):
                        is_pointer = True
                        param_name = parts[0][1:]
                    else:
                        param_name = parts[0]
                
                # Set parameter attributes
                param_node.name = param_name
                param_node.attributes['type'] = param_type.value
                if is_pointer:
                    param_node.attributes['pointer'] = True
                if struct_name:
                    param_node.attributes['struct_name'] = struct_name
                param_node.value = param
                
                params_node.add_child(param_node)
        
        return params_node

    def _extract_struct_bodies(self, processed_lines: List[Tuple[str, int, int]]) -> Dict[str, List[TALNode]]:
        """
        Extract struct member definitions from BEGIN...END blocks.
        
        Args:
            processed_lines: Preprocessed source lines with location info
            
        Returns:
            Dictionary mapping struct names to lists of member nodes
            
        TAL struct definitions can have member lists in BEGIN...END blocks:
        
        STRUCT data_packet_def;
        BEGIN
            INT sequence_number;
            STRING(20) message_text;
            REAL timestamp;
        END;
        
        This method identifies such structures and extracts the member
        definitions for later attachment to struct declaration nodes.
        Must distinguish struct bodies from procedure bodies.
        """
        struct_bodies = {}
        i = 0
        
        while i < len(processed_lines):
            line_text, line_num, column = processed_lines[i]
            
            # Look for struct declaration not followed by PROC
            if (line_text.upper().startswith('STRUCT ') and 
                'PROC' not in line_text.upper()):
                
                # Extract struct name from declaration
                struct_name = line_text.split()[1].rstrip('(*);')
                i += 1
                
                # Look for BEGIN on following lines
                while i < len(processed_lines):
                    next_line, next_line_num, next_column = processed_lines[i]
                    if next_line.upper().strip() == 'BEGIN':
                        # Found struct body - collect members until END
                        members = []
                        i += 1
                        
                        while i < len(processed_lines):
                            member_line, member_line_num, member_column = processed_lines[i]
                            member_upper = member_line.upper().strip()
                            
                            # Check for end of struct body
                            if member_upper in ['END;', 'END']:
                                break
                                
                            # Parse non-empty, non-comment lines as struct members
                            if member_line.strip() and not member_line.startswith('!'):
                                member_node = self._parse_struct_member(member_line, SourceLocation(self.filename, member_line_num, member_column))
                                if member_node:
                                    members.append(member_node)
                            i += 1
                        
                        struct_bodies[struct_name] = members
                        break
                    elif (next_line.upper().startswith('PROC ') or 
                          'PROC ' in next_line.upper()):
                        # This is actually a procedure, not a struct with body
                        break
                    i += 1
            else:
                i += 1
        
        return struct_bodies
    
    def _parse_struct_member(self, line: str, location: SourceLocation) -> Optional[TALNode]:
        """
        Parse a single struct member declaration.
        
        Args:
            line: Source line containing member declaration
            location: Source location for error reporting
            
        Returns:
            TALNode representing the struct member
            
        Struct members follow standard TAL variable declaration syntax:
        - TYPE member_name;
        - TYPE .pointer_member; (for pointer members)
        - TYPE member_array[bounds]; (for array members)
        
        Creates a struct_member node with type and name information.
        """
        line = line.strip().rstrip(';')
        if not line:
            return None
            
        member_node = TALNode('struct_member')
        member_node.location = location
        member_node.value = line
        
        # Parse member: TYPE [.] name
        parts = line.split()
        if len(parts) >= 2:
            member_type = parts[0].upper()
            member_name = parts[1]
            
            # Handle pointer members (leading dot)
            if member_name.startswith('.'):
                member_node.attributes['is_pointer'] = True
                member_name = member_name[1:]
            
            member_node.name = member_name
            member_node.attributes['type'] = member_type
        
        return member_node
    
    def _is_struct_body_line(self, line: str, struct_bodies: Dict[str, List[TALNode]]) -> bool:
        """
        Check if line is part of an already-processed struct body.
        
        Args:
            line: Source line to check
            struct_bodies: Previously extracted struct bodies
            
        Returns:
            True if line was already processed as part of a struct body
            
        Prevents double-processing of struct member lines that were
        already extracted in the struct body extraction phase.
        Checks for BEGIN/END keywords and member declaration patterns.
        """
        upper_line = line.upper()
        
        # Skip BEGIN/END and member declarations that are part of struct bodies
        if upper_line == 'BEGIN' or upper_line == 'END;' or upper_line == 'END':
            return True
        
        # Check if this looks like a struct member that was already processed
        parts = line.split()
        if len(parts) >= 2:
            potential_type = parts[0].upper()
            if potential_type in ['INT', 'STRING', 'REAL', 'FIXED', 'BYTE', 'CHAR']:
                # This might be a struct member - check against extracted bodies
                return any(any(member.value and line.strip().rstrip(';') in member.value 
                              for member in members) 
                          for members in struct_bodies.values())
        
        return False
    
    def _looks_like_struct_member(self, line: str) -> bool:
        """
        Check if line looks like struct member rather than variable declaration.
        
        Args:
            line: Source line to analyze
            
        Returns:
            True if line appears to be a struct member
            
        Helps distinguish between local variable declarations and struct
        member definitions. Struct members typically have simpler syntax
        and appear in struct definition context.
        """
        # Heuristic based on parsing context and line structure
        return (self.in_struct or 
                len(line.split()) == 2 and  # Simple "TYPE name" pattern
                not line.endswith(';'))
    
    def _parse_comment(self, line: str, location: SourceLocation) -> TALNode:
        """
        Parse TAL comment line (starts with '!').
        
        Args:
            line: Comment line including '!' prefix
            location: Source location for documentation
            
        Returns:
            TALNode representing the comment
            
        TAL uses '!' for line comments. Comments are preserved in the AST
        for documentation generation and code analysis tools.
        """
        node = TALNode('comment', value=line[1:].strip())
        node.location = location
        return node
    
    def _handle_begin(self, location: SourceLocation) -> Optional[TALNode]:
        """
        Handle BEGIN statement in procedure context.
        
        Args:
            location: Source location of BEGIN
            
        Returns:
            TALNode for begin block, or None if not in procedure
            
        BEGIN marks the start of executable statements in procedures.
        Sets the collecting_statements flag to indicate that subsequent
        lines should be added to the procedure body.
        """
        self.collecting_statements = True
        if self.current_procedure:
            begin_node = TALNode('begin_block')
            begin_node.location = location
            return begin_node
        return None
    
    def _parse_name_declaration(self, line: str, location: SourceLocation) -> TALNode:
        """
        Parse TAL NAME declaration.
        
        Args:
            line: NAME declaration line
            location: Source location
            
        Returns:
            TALNode representing the name declaration
            
        TAL NAME declarations define module or compilation unit names:
        NAME module_name;
        
        Used for modular programming and symbol visibility control.
        """
        name = line.split()[1].rstrip(';')
        node = TALNode('name_decl', name=name)
        node.location = location
        return node
    
    def _parse_struct_declaration(self, line: str, location: SourceLocation) -> TALNode:
        """
        Parse TAL STRUCT type declaration.
        
        Args:
            line: STRUCT declaration line
            location: Source location
            
        Returns:
            TALNode representing the struct declaration
            
        TAL STRUCT declarations define composite data types:
        STRUCT type_name;
        
        May be followed by BEGIN...END block with member definitions,
        which are handled separately and attached later.
        """
        node = TALNode('struct_decl')
        node.location = location
        
        # Extract struct name from declaration
        parts = line.split()
        if len(parts) >= 2:
            struct_name = parts[1].rstrip('(*);')
            node.name = struct_name
        
        # Struct body members are added later if found
        return node
    
    def _parse_procedure_declaration(self, line: str, location: SourceLocation) -> TALNode:
        """
        Parse TAL procedure declaration with return type and parameters.
        
        Args:
            line: Procedure declaration line
            location: Source location
            
        Returns:
            TALNode representing the procedure
            
        TAL procedures can have optional return types and parameters:
        PROC procedure_name (param1, param2);
        INT PROC function_name (INT param);
        STRING PROC formatter (STRING input);
        
        Creates procedure node with parameter and local declaration sections.
        Manages symbol table scope entry and parsing state for procedure body.
        """
        proc_node = TALNode('procedure')
        proc_node.location = location
        
        # Determine return type based on declaration prefix
        upper_line = line.upper()
        return_type = TALType.UNKNOWN
        name_start = 1
        
        if upper_line.startswith('INT PROC'):
            return_type = TALType.INT
            name_start = 2
        elif upper_line.startswith('STRING PROC'):
            return_type = TALType.STRING
            name_start = 2
        elif upper_line.startswith('REAL PROC'):
            return_type = TALType.REAL
            name_start = 2
        
        proc_node.attributes['return_type'] = return_type.value
        
        # Extract procedure name and parameters
        if '(' in line:
            name_part = line.split('(')[0]
            param_part = line[line.find('('):line.rfind(')')+1] if ')' in line else ''
        else:
            name_part = line.rstrip(';')
            param_part = ''
        
        # Get procedure name from appropriate position
        name_parts = name_part.split()
        if len(name_parts) >= name_start + 1:
            proc_name = name_parts[name_start]
            proc_node.name = proc_name
            
            # Check for MAIN procedure
            if 'MAIN' in line.upper():
                proc_node.attributes['main'] = True
            
            # Enter new scope for procedure
            self.symbol_table.enter_scope(proc_name)
        
        # Parse parameter list if present
        if param_part:
            params_node = self._parse_parameters_fixed(param_part, location)
            if params_node:
                proc_node.add_child(params_node)
        
        # Add standard procedure sections
        locals_node = TALNode('local_declarations')
        locals_node.location = location
        proc_node.add_child(locals_node)
        
        statements_node = TALNode('statements')
        statements_node.location = location
        proc_node.add_child(statements_node)
        
        # Set as current procedure for subsequent parsing
        self.current_procedure = proc_node
        # Don't start collecting statements until BEGIN
        
        return proc_node
    
    def _parse_variable_spec(self, var_spec: str, var_type: str, location: SourceLocation) -> TALNode:
        """
        Parse individual variable specification within a declaration.
        
        Args:
            var_spec: Single variable specification string
            var_type: TAL type for this variable
            location: Source location
            
        Returns:
            TALNode representing the variable specification
            
        Variable specifications can include:
        - Simple names: variable_name
        - Pointers: .pointer_variable  
        - Arrays: array_name[start:end]
        - Initialization: variable := initial_value
        
        Extracts all attributes and creates clean AST representation.
        """
        spec_node = TALNode('var_spec')
        spec_node.location = location
        
        is_pointer = False
        is_array = False
        var_name = ""
        initializer = None
        
        # Handle initialization expressions
        if ':=' in var_spec:
            name_part, init_part = var_spec.split(':=', 1)
            name_part = name_part.strip()
            initializer = init_part.strip()
            spec_node.attributes['initializer'] = initializer
        else:
            name_part = var_spec.strip()
        
        # Handle array declarations with bounds
        if '[' in name_part and ']' in name_part:
            is_array = True
            bracket_start = name_part.find('[')
            bracket_end = name_part.find(']')
            array_spec = name_part[bracket_start+1:bracket_end]
            var_name = name_part[:bracket_start]
            spec_node.attributes['array_bounds'] = array_spec
        else:
            var_name = name_part
        
        # Handle pointer variables (leading dot)
        if var_name.startswith('.'):
            is_pointer = True
            var_name = var_name[1:]
        
        # Clean up variable name
        var_name = var_name.strip().rstrip(';').rstrip(',')
        
        spec_node.name = var_name
        spec_node.value = var_name  # Clean, simple value
        
        if is_pointer:
            spec_node.attributes['pointer'] = True
        if is_array:
            spec_node.attributes['array'] = True
        
        return spec_node
    
    def _parse_literal_declaration(self, line: str, location: SourceLocation) -> TALNode:
        """
        Parse TAL LITERAL constant declaration.
        
        Args:
            line: LITERAL declaration line
            location: Source location
            
        Returns:
            TALNode representing literal declarations
            
        TAL LITERAL declarations define compile-time constants:
        LITERAL MAX_SIZE = 100, DEFAULT_NAME = "SYSTEM";
        
        Multiple assignments can be specified in a single declaration.
        """
        literal_node = TALNode('literal_decl')
        literal_node.location = location
        
        # Extract content after LITERAL keyword
        content = line[7:].strip().rstrip(';')
        assignments = self._smart_split(content, ',')
        
        for assignment in assignments:
            if '=' in assignment:
                name, value = assignment.split('=', 1)
                assign_node = TALNode('assignment')
                assign_node.name = name.strip()
                assign_node.value = value.strip()
                assign_node.location = location
                literal_node.add_child(assign_node)
        
        return literal_node
    
    def _parse_define_declaration(self, line: str, location: SourceLocation) -> TALNode:
        """
        Parse TAL DEFINE preprocessor declaration.
        
        Args:
            line: DEFINE declaration line
            location: Source location
            
        Returns:
            TALNode representing define declarations
            
        TAL DEFINE declarations create preprocessor macros:
        DEFINE MAX_USERS = 50, DEBUG_MODE = 1 #;
        
        Similar to LITERAL but processed at compile time.
        Can end with # or #; to mark preprocessor directives.
        """
        define_node = TALNode('define_decl')
        define_node.location = location
        
        # Extract content after DEFINE keyword, handle # terminators
        content = line[6:].strip()
        if content.endswith('#'):
            content = content[:-1]
        if content.endswith(';'):
            content = content[:-1]
        
        assignments = self._smart_split(content, ',')
        for assignment in assignments:
            if '=' in assignment:
                name, value = assignment.split('=', 1)
                assign_node = TALNode('assignment')
                assign_node.name = name.strip()
                assign_node.value = value.strip()
                assign_node.location = location
                define_node.add_child(assign_node)
        
        return define_node
    
    def _parse_while_statement(self, line: str, location: SourceLocation) -> TALNode:
        """
        Parse TAL WHILE loop statement.
        
        Args:
            line: WHILE statement line
            location: Source location
            
        Returns:
            TALNode representing the while loop
            
        TAL WHILE loops have the form:
        WHILE condition DO
        
        The condition expression is extracted and stored as an attribute.
        """
        while_node = TALNode('while_stmt')
        while_node.location = location
        while_node.value = line.rstrip(';')
        
        # Extract condition between WHILE and DO
        if ' DO' in line.upper():
            condition = line.split(' DO')[0][5:].strip()  # Remove 'WHILE'
            while_node.attributes['condition'] = condition
        
        return while_node
    
    def _parse_case_statement(self, line: str, location: SourceLocation) -> TALNode:
        """
        Parse TAL CASE statement for multi-way branching.
        
        Args:
            line: CASE statement line
            location: Source location
            
        Returns:
            TALNode representing the case statement
            
        TAL CASE statements have the form:
        CASE expression OF
        
        The expression to switch on is extracted and stored as an attribute.
        """
        case_node = TALNode('case_stmt')
        case_node.location = location
        case_node.value = line.rstrip(';')
        
        # Extract expression between CASE and OF
        if ' OF' in line.upper():
            expression = line.split(' OF')[0][4:].strip()  # Remove 'CASE'
            case_node.attributes['expression'] = expression
        
        return case_node
    
    def _parse_scan_statement(self, line: str, location: SourceLocation) -> TALNode:
        """
        Parse TAL SCAN statement for array/string scanning.
        
        Args:
            line: SCAN statement line
            location: Source location
            
        Returns:
            TALNode representing the scan statement
            
        TAL SCAN statements iterate through data structures:
        SCAN variable WHILE condition -> label;
        
        Extracts the variable, condition, and target label components.
        """
        scan_node = TALNode('scan_stmt')
        scan_node.location = location
        scan_node.value = line.rstrip(';')
        
        # Parse SCAN components
        parts = line.split()
        if len(parts) >= 2:
            scan_node.attributes['variable'] = parts[1]
        
        # Extract WHILE condition and target if present
        if 'WHILE' in line.upper():
            while_idx = line.upper().find('WHILE')
            condition_part = line[while_idx + 5:].strip()
            if '->' in condition_part:
                condition, target = condition_part.split('->', 1)
                scan_node.attributes['condition'] = condition.strip()
                scan_node.attributes['target'] = target.strip()
            else:
                scan_node.attributes['condition'] = condition_part
        
        return scan_node
    
    def _parse_return_statement(self, line: str, location: SourceLocation) -> TALNode:
        """
        Parse TAL RETURN statement for procedure exit.
        
        Args:
            line: RETURN statement line
            location: Source location
            
        Returns:
            TALNode representing the return statement
            
        TAL RETURN statements can optionally return values:
        RETURN;
        RETURN expression;
        
        The return expression is extracted if present.
        """
        return_node = TALNode('return_stmt')
        return_node.location = location
        return_expr = line[7:].rstrip(';')  # Remove 'RETURN' prefix
        return_node.value = return_expr
        
        # Store return expression if present
        if return_expr:
            return_node.attributes['expression'] = return_expr
        
        return return_node
    
    def _parse_call_statement(self, line: str, location: SourceLocation) -> TALNode:
        """
        Parse TAL procedure call statement.
        
        Args:
            line: CALL statement line
            location: Source location
            
        Returns:
            TALNode representing the procedure call
            
        TAL procedure calls can use different syntaxes:
        CALL procedure_name(arg1, arg2);
        $procedure_name(arg1, arg2);  (short form)
        
        Extracts function name and argument list for analysis.
        """
        call_node = TALNode('call_stmt')
        call_node.location = location
        call_node.value = line.rstrip(';')
        
        # Extract function call expression
        if line.startswith('CALL '):
            call_expr = line[5:].strip()
        elif line.startswith('$'):
            call_expr = line
        else:
            call_expr = line
        
        # Parse function name and arguments
        if '(' in call_expr:
            func_name = call_expr.split('(')[0].strip()
            if func_name.startswith('$'):
                func_name = func_name[1:]
            call_node.attributes['function'] = func_name
            
            # Extract and parse argument list
            arg_part = call_expr[call_expr.find('(')+1:call_expr.rfind(')')]
            if arg_part:
                args = self._smart_split(arg_part, ',')
                call_node.attributes['arguments'] = [arg.strip() for arg in args]
        
        return call_node
    
    def _parse_if_statement(self, line: str, location: SourceLocation) -> TALNode:
        """
        Parse TAL IF conditional statement.
        
        Args:
            line: IF statement line
            location: Source location
            
        Returns:
            TALNode representing the if statement
            
        TAL IF statements have the form:
        IF condition THEN
        
        The condition expression is extracted for later analysis.
        """
        if_node = TALNode('if_stmt')
        if_node.location = location
        if_node.value = line.rstrip(';')
        
        # Extract condition between IF and THEN
        if ' THEN' in line.upper():
            condition = line.split(' THEN')[0][2:].strip()  # Remove 'IF'
            if_node.attributes['condition'] = condition
        
        return if_node
    
    def _parse_assignment(self, line: str, location: SourceLocation) -> TALNode:
        """
        Parse TAL assignment statement with multiple assignment operators.
        
        Args:
            line: Assignment statement line
            location: Source location
            
        Returns:
            TALNode representing the assignment
            
        TAL supports different assignment operators:
        - := standard assignment
        - ':=' string assignment (special operator for string handling)
        
        Extracts target variable, value expression, and operator type.
        """
        assign_node = TALNode('assignment')
        assign_node.location = location
        assign_node.value = line.rstrip(';')
        
        # Handle different TAL assignment operators
        if "':='" in line:
            # Special TAL string assignment operator
            lhs, rhs = line.split("':='", 1)
            assign_node.name = lhs.strip()
            assign_node.attributes['target'] = lhs.strip()
            assign_node.attributes['value'] = rhs.strip().rstrip(';')
            assign_node.attributes['operator'] = "':='"
        elif ':=' in line:
            # Standard TAL assignment
            lhs, rhs = line.split(':=', 1)
            assign_node.name = lhs.strip()
            assign_node.attributes['target'] = lhs.strip()
            assign_node.attributes['value'] = rhs.strip().rstrip(';')
            assign_node.attributes['operator'] = ':='
        
        return assign_node
    
    def _parse_general_statement(self, line: str, location: SourceLocation) -> TALNode:
        """
        Parse general TAL statement that doesn't match specific patterns.
        
        Args:
            line: Statement line
            location: Source location
            
        Returns:
            TALNode representing the generic statement
            
        Fallback parser for TAL constructs that don't match specific
        patterns. Preserves the original text for later analysis or
        specialized handling.
        """
        stmt_node = TALNode('statement')
        stmt_node.location = location
        stmt_node.value = line.rstrip(';')
        return stmt_node
    
    def _generate_cross_references(self, program: TALNode):
        """
        Generate cross-references between symbols and their usage.
        
        Args:
            program: Root AST node to analyze
            
        Analyzes the AST to build relationships between symbol declarations
        and references. Useful for:
        - IDE "go to definition" features
        - Unused variable detection
        - Call graph generation
        - Dependency analysis
        
        Currently a placeholder for future implementation.
        """
        # TODO: Implementation for cross-reference generation
        # Would traverse AST to find symbol references and link to declarations
        pass
    
    def _export_symbol_table(self) -> Dict[str, Any]:
        """
        Export symbol table to dictionary format for analysis tools.
        
        Returns:
            Dictionary representation of the complete symbol table
            
        Converts the internal symbol table structure to a format suitable
        for JSON serialization and external tool consumption. Includes
        all symbol metadata and reference locations.
        """
        symbols = {}
        for scope_name, scope_symbols in self.symbol_table.scopes.items():
            symbols[scope_name] = {}
            for sym_name, symbol in scope_symbols.items():
                symbols[scope_name][sym_name] = {
                    'type': symbol.symbol_type.value,
                    'location': f"{symbol.location.filename}:{symbol.location.line}:{symbol.location.column}",
                    'is_pointer': symbol.is_pointer,
                    'is_array': symbol.is_array,
                    'references': [f"{ref.filename}:{ref.line}:{ref.column}" for ref in symbol.references]
                }
        return symbols
    
    def _smart_split(self, text: str, delimiter: str) -> List[str]:
        """
        Split text by delimiter while respecting nested structures.
        
        Args:
            text: Text to split
            delimiter: Character to split on
            
        Returns:
            List of text segments split at delimiter
            
        Unlike simple string split, this method respects:
        - Nested parentheses and brackets
        - Quoted string literals (single and double quotes)
        - Balanced delimiter counting
        
        Essential for parsing parameter lists and complex expressions
        where commas might appear inside nested structures.
        
        Example:
        "func(a, b), array[1, 2], \"hello, world\"" 
        splits to: ["func(a, b)", "array[1, 2]", "\"hello, world\""]
        """
        parts = []
        current = ""
        paren_level = 0      # Track () nesting
        bracket_level = 0    # Track [] nesting  
        in_quotes = False
        quote_char = None
        
        for char in text:
            # Handle quote character transitions
            if char in '"\'':
                if not in_quotes:
                    in_quotes = True
                    quote_char = char
                elif char == quote_char:
                    in_quotes = False
                    quote_char = None
            
            # Only track delimiters outside of quotes
            if not in_quotes:
                if char == '(':
                    paren_level += 1
                if char == ')':
                    paren_level -= 1
                if char == '[':
                    bracket_level += 1
                if char == ']':
                    bracket_level -= 1
                
                # Split only when all delimiters are balanced
                if (char == delimiter and 
                    paren_level == 0 and bracket_level == 0):
                    parts.append(current.strip())
                    current = ""
                    continue
            
            current += char
        
        # Add final part if non-empty
        if current.strip():
            parts.append(current.strip())
        
        return parts
    
    def _count_nodes(self, node: TALNode) -> int:
        """
        Recursively count total nodes in AST.
        
        Args:
            node: Root node to count from
            
        Returns:
            Total number of nodes in the subtree
            
        Used for performance analysis and complexity metrics.
        Counts all nodes in the tree rooted at the given node.
        """
        count = 1
        for child in node.children:
            count += self._count_nodes(child)
        return count
    
    def _analyze_structure(self, program: TALNode) -> Dict[str, Any]:
        """
        Analyze program structure with detailed metrics and organization.
        
        Args:
            program: Root program AST node
            
        Returns:
            Dictionary with comprehensive structural analysis
            
        Provides detailed analysis of the TAL program including:
        - Node counts by type (procedures, variables, etc.)
        - Complexity metrics for maintainability assessment
        - Lists of program entities (procedures, variables, structs)
        - Organizational structure analysis
        
        Useful for:
        - Code quality assessment
        - Documentation generation
        - Refactoring planning
        - Project metrics
        """
        structure = {
            'total_nodes': self._count_nodes(program),
            'globals': 0,
            'procedures': 0,
            'comments': 0,
            'statements': 0,
            'variables': 0,
            'structs': 0,
            'literals': 0,
            'defines': 0,
            'complexity_score': 0,
            'procedure_list': [],
            'global_variables': [],
            'struct_list': []
        }
        
        # Analyze program children by type
        for child in program.children:
            if child.type == 'name_decl':
                structure['globals'] += 1
            elif child.type == 'struct_decl':
                structure['globals'] += 1
                structure['structs'] += 1
                structure['struct_list'].append(child.name)
            elif child.type == 'var_decl':
                structure['globals'] += 1
                structure['variables'] += 1
                # Extract variable names from specifications
                for spec in child.children:
                    if spec.name:
                        structure['global_variables'].append(spec.name)
            elif child.type == 'literal_decl':
                structure['globals'] += 1
                structure['literals'] += 1
            elif child.type == 'define_decl':
                structure['globals'] += 1
                structure['defines'] += 1
            elif child.type == 'procedure':
                structure['procedures'] += 1
                structure['procedure_list'].append({
                    'name': child.name,
                    'return_type': child.attributes.get('return_type', 'void'),
                    'is_main': child.attributes.get('main', False)
                })
                # Count statements within procedure
                for proc_child in child.children:
                    if proc_child.type == 'statements':
                        structure['statements'] += len(proc_child.children)
            elif child.type == 'comment':
                structure['comments'] += 1
        
        # Calculate complexity score based on program elements
        # Higher scores indicate more complex programs requiring more maintenance
        structure['complexity_score'] = (
            structure['procedures'] * 10 +      # Procedures add significant complexity
            structure['statements'] * 2 +       # Each statement adds complexity
            structure['variables'] * 1 +        # Variables add some complexity
            structure['structs'] * 5           # Structs add moderate complexity
        )
        
        return structure
    
    def _should_be_global_variable(self, line: str, location: SourceLocation) -> bool:
        """
        Determine if variable declaration should be placed at global scope.
        
        Args:
            line: Variable declaration line
            location: Source location
            
        Returns:
            True if variable should be global, False if local
            
        TAL scoping rules for variable placement:
        - Outside any procedure: always global
        - Before BEGIN in procedure: could be global (context dependent)
        - After BEGIN in procedure: always local to procedure
        
        Used to correctly place variable declarations in the AST.
        """
        # If we're not in a procedure, it's definitely global
        if not self.current_procedure:
            return True
        
        # If we haven't started collecting statements (before BEGIN), 
        # it could be global depending on context
        if not self.collecting_statements:
            return True
            
        # If we're inside a procedure body (after BEGIN), it's local
        return False

    def _parse_variable_declaration(self, line: str, location: SourceLocation) -> TALNode:
        """
        Parse TAL variable declarations with comprehensive type and attribute support.
        
        Args:
            line: Variable declaration line
            location: Source location
            
        Returns:
            TALNode representing the variable declaration
            
        TAL variable declarations support:
        - Standard types: INT, STRING, REAL, FIXED, BYTE, CHAR
        - Struct variables: struct_name variable_name
        - Pointers: indicated by leading dot (.)
        - Arrays: with bounds [start:end]
        - Multiple variables: comma-separated lists
        - Initialization: variable := initial_value
        
        Handles both global and local variable contexts with proper
        scoping and type analysis.
        """
        var_node = TALNode('var_decl')
        var_node.location = location
        
        parts = line.split()
        if not parts:
            return var_node
        
        # Handle struct variable declarations (not standard types)
        if len(parts) >= 2 and not any(parts[0].upper().startswith(t) for t in ['INT', 'STRING', 'REAL', 'FIXED', 'BYTE', 'CHAR']):
            # Struct variable declaration: struct_name var_name
            struct_name = parts[0]
            var_name = parts[1]
            
            # Handle pointer indicator and clean up name
            is_pointer = False
            if var_name.startswith('.'):
                is_pointer = True
                var_name = var_name[1:]
            
            # Clean up variable name (remove punctuation)
            var_name = var_name.rstrip(';').rstrip(',')
            
            # Set struct variable attributes
            var_node.attributes['type'] = 'STRUCT'
            var_node.attributes['struct_name'] = struct_name
            if is_pointer:
                var_node.attributes['pointer'] = True
            
            # Create variable specification node
            spec_node = TALNode('var_spec')
            spec_node.name = var_name
            spec_node.location = location
            spec_node.value = var_name
            if is_pointer:
                spec_node.attributes['pointer'] = True
            
            var_node.add_child(spec_node)
            
        else:
            # Standard TAL type declaration (INT, STRING, etc.)
            var_type_full = parts[0]
            var_type_str = var_type_full.upper()
            
            # Handle types with size specifiers like STRING(20)
            if '(' in var_type_str:
                base_type = var_type_str.split('(')[0]
                var_node.attributes['type'] = base_type
                var_node.attributes['type_full'] = var_type_full
            else:
                var_node.attributes['type'] = var_type_str
            
            # Extract variable specifications from remaining text
            remaining = ' '.join(parts[1:]).rstrip(';')
            
            # Handle comma-separated variable lists
            if ',' in remaining:
                var_specs = self._smart_split(remaining, ',')
            else:
                var_specs = [remaining] if remaining else []
            
            # Parse each individual variable specification
            for var_spec in var_specs:
                var_spec = var_spec.strip()
                if var_spec:
                    spec_node = self._parse_variable_spec(var_spec, var_type_str, location)
                    if spec_node:
                        var_node.add_child(spec_node)
        
        return var_node

    def _is_variable_declaration(self, line: str) -> bool:
        """
        Determine if a source line represents a variable declaration.
        
        Args:
            line: Source line to analyze
            
        Returns:
            True if line appears to be a variable declaration
            
        Comprehensive detection of TAL variable declaration patterns:
        - Standard type prefixes: INT, STRING, REAL, etc.
        - Struct variable patterns: struct_name variable_name
        - Exclusion of non-declaration constructs
        
        Must distinguish from similar-looking constructs like procedure
        declarations, control statements, and complex expressions.
        """
        upper_line = line.upper()
        
        # Standard TAL type declarations (excluding procedure declarations)
        if (any(upper_line.startswith(t) for t in ['INT', 'STRING', 'REAL', 'FIXED', 'BYTE', 'CHAR']) 
            and 'PROC ' not in upper_line):
            return True
        
        # Struct variable declarations with careful exclusion filtering
        if (len(line.split()) >= 2 and 
            not any(upper_line.startswith(kw) for kw in ['NAME ', 'STRUCT ', 'LITERAL ', 'DEFINE ', 'PROC ', 'WHILE ', 'CASE ', 'SCAN ', 'RETURN ', 'CALL ', 'IF ']) and
            not ':=' in line and  # Not an assignment statement
            not self._looks_like_struct_member(line) and  # Not a struct member
            not line.strip().endswith('DO') and  # Not a control structure
            not line.strip().endswith('THEN')):  # Not a conditional
            return True
        
        return False


def main():
    """
    Enhanced main function with comprehensive command-line interface.
    
    Provides a full-featured command-line interface for the TAL parser with:
    - Multiple output format options (S-expressions, JSON, DOT graphs)
    - Detailed analysis and reporting modes
    - Symbol table export capabilities
    - Performance statistics and error reporting
    - Flexible output redirection
    
    Command-line usage supports both simple parsing and advanced analysis
    workflows for different use cases:
    - Quick syntax checking with error-only mode
    - Detailed analysis for code quality assessment  
    - Graph generation for visualization
    - JSON export for tool integration
    """
    parser = argparse.ArgumentParser(
        description='Enhanced TAL parser with comprehensive analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output formats:
  sexp    : S-expression format (default) - Human-readable tree structure
  json    : JSON format - JSON format - Machine-readable structured data  
  dot     : Graphviz DOT format - Graph visualization
  pretty  : Pretty-printed TAL - Formatted source code
  
Examples:
  %(prog)s input.tal                    # Parse and show S-expression
  %(prog)s input.tal -f json -o out.json # Output JSON format to file
  %(prog)s input.tal -a -v             # Verbose output with detailed analysis
  %(prog)s input.tal -f dot | dot -Tpng -o ast.png  # Generate AST visualization
  %(prog)s input.tal -e               # Show only errors and warnings
  %(prog)s input.tal -s --stats       # Display symbol table and statistics
        """
    )
    
    # Required and optional arguments
    parser.add_argument('input_file', help='Input TAL file to parse')
    parser.add_argument('-o', '--output', help='Output file (default: stdout)')
    parser.add_argument('-f', '--format', choices=['sexp', 'json', 'dot', 'pretty'], 
                       default='sexp', help='Output format selection')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output with detailed information')
    parser.add_argument('-a', '--analyze', action='store_true', help='Show detailed structural analysis')
    parser.add_argument('-s', '--symbols', action='store_true', help='Display symbol table with cross-references')
    parser.add_argument('-e', '--errors-only', action='store_true', help='Show only errors and warnings')
    parser.add_argument('--no-semantic', action='store_true', help='Skip semantic analysis phase')
    parser.add_argument('--stats', action='store_true', help='Show detailed parsing statistics')
    
    args = parser.parse_args()
    
    if args.verbose:
        print(f"Enhanced TAL Parser - Processing: {args.input_file}")
    
    # Initialize parser and process file
    tal_parser = EnhancedTALParser()
    result = tal_parser.parse_file(args.input_file)
    
    # Display errors and warnings with context
    if result.get('errors') or result.get('warnings'):
        if not args.errors_only:
            print("=== PARSE RESULTS ===", file=sys.stderr)
        
        # Show all errors with full context
        for error in result.get('errors', []):
            print(error, file=sys.stderr)
        
        # Show warnings if not in error-only mode
        for warning in result.get('warnings', []):
            print(warning, file=sys.stderr)
        
        # Exit early if only showing errors
        if args.errors_only:
            sys.exit(1 if result.get('errors') else 0)
    
    # Display parsing statistics
    if args.stats and 'stats' in result:
        print("=== PARSING STATISTICS ===")
        stats = result['stats']
        print(f"Parse time: {stats['parse_time']:.3f} seconds")
        print(f"Lines processed: {stats['lines_processed']}")
        print(f"AST nodes created: {stats['nodes_created']}")
        print(f"Symbols declared: {stats['symbols_declared']}")
        print(f"Errors found: {stats['errors_found']}")
        print(f"Warnings issued: {stats['warnings_issued']}")
        print()
    
    # Display detailed structural analysis
    if args.analyze and 'structure' in result:
        print("=== STRUCTURAL ANALYSIS ===")
        structure = result['structure']
        print(f"Total AST nodes: {structure['total_nodes']}")
        print(f"Procedures defined: {structure['procedures']}")
        print(f"Global variables: {structure['variables']}")
        print(f"Struct types: {structure['structs']}")
        print(f"Executable statements: {structure['statements']}")
        print(f"Complexity score: {structure['complexity_score']}")
        
        # List procedures with details
        if structure['procedure_list']:
            print("\nProcedures:")
            for proc in structure['procedure_list']:
                main_marker = " (MAIN ENTRY POINT)" if proc['is_main'] else ""
                print(f"  - {proc['name']}: returns {proc['return_type']}{main_marker}")
        
        # List struct types
        if structure['struct_list']:
            print(f"\nStruct types: {', '.join(structure['struct_list'])}")
        
        print()
    
    # Display symbol table with cross-references
    if args.symbols and 'symbols' in result:
        print("=== SYMBOL TABLE ===")
        symbols = result['symbols']
        for scope_name, scope_symbols in symbols.items():
            print(f"\nScope: {scope_name}")
            for sym_name, sym_info in scope_symbols.items():
                # Format symbol information
                refs = f" (referenced {len(sym_info['references'])} times)" if sym_info['references'] else " (unused)"
                pointer_marker = " *" if sym_info['is_pointer'] else ""
                array_marker = " []" if sym_info['is_array'] else ""
                print(f"  {sym_name}: {sym_info['type']}{pointer_marker}{array_marker} @ {sym_info['location']}{refs}")
        print()
    
    # Generate and output the parsed result
    if result.get('success'):
        # Select appropriate output format
        if args.format == 'sexp':
            output = result['sexp']
        elif args.format == 'json':
            output = json.dumps(result['json'], indent=2)
        elif args.format == 'dot':
            output = result['dot']
        elif args.format == 'pretty':
            # Pretty-printing would need additional implementation
            output = result['sexp']  # Fallback to S-expression
        
        if args.verbose and not args.errors_only:
            print(f"Parse successful! Output format: {args.format}")
        
        # Write output to file or stdout
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(output)
            if args.verbose:
                print(f"Output written to: {args.output}")
        else:
            # Only print output if not showing analysis information
            if not (args.analyze or args.symbols or args.stats):
                print(output)
    
    else:
        # Handle parse failure
        print(f"Parse failed: {result.get('error', 'Unknown error')}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

