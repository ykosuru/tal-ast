"""
TAL Procedure Parser - Foundation parsing for Transaction Application Language (TAL)

This module provides the foundational parsing capabilities for TAL procedure declarations
and basic language constructs. It serves as the base layer for the enhanced TAL parser,
focusing on reliable procedure boundary detection and parameter parsing.

Key Features:
- Robust procedure declaration parsing with multi-line support
- Parameter type analysis and symbol table management
- TAL-specific type system support
- Error reporting with source location tracking
- S-expression AST generation for interoperability

Architecture:
- Uses regex-based parsing for procedure declarations
- Maintains symbol tables for scope management
- Generates structured AST nodes for further processing
- Provides foundation for rich parsing enhancement
"""

import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

class TALType(Enum):
    """
    Enumeration of TAL data types.
    
    TAL supports various numeric and string types with optional size specifications.
    This enum captures the core TAL type system for semantic analysis.
    """
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
    BYTE = "BYTE"
    CHAR = "CHAR"
    STRUCT = "STRUCT"
    POINTER = "POINTER"
    UNKNOWN = "UNKNOWN"

class ErrorSeverity(Enum):
    """
    Error severity levels for parsing diagnostics.
    
    Provides a hierarchy of error types to enable appropriate handling
    and user feedback during the parsing process.
    """
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    FATAL = "fatal"

@dataclass
class SourceLocation:
    """
    Represents a location in the source code for error reporting and debugging.
    
    Tracks file position information to provide precise error messages
    and enable IDE integration for syntax highlighting and navigation.
    
    Attributes:
        filename: Name of the source file
        line: Line number (1-based)
        column: Column number (1-based)
        length: Length of the token/construct at this location
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
    Represents a parsing error with context and suggestions.
    
    Provides comprehensive error information including location, severity,
    context lines, and potential fixes to aid in debugging and correction.
    
    Attributes:
        message: Human-readable error description
        location: Where the error occurred in the source
        severity: How serious the error is
        context_lines: Surrounding source lines for context
        suggestions: Potential fixes or workarounds
        error_code: Machine-readable error identifier
    """
    message: str
    location: SourceLocation
    severity: ErrorSeverity
    context_lines: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    error_code: str = ""
    
    def __str__(self):
        """Format error for display with context and suggestions."""
        result = f"{self.severity.value.upper()}: {self.message}\n  at {self.location}"
        if self.context_lines:
            for i, line in enumerate(self.context_lines):
                marker = ">>>" if i == 1 else "   "
                result += f"\n  {marker} {line}"
        if self.suggestions:
            result += "\n  Suggestions:"
            for suggestion in self.suggestions:
                result += f"\n    - {suggestion}"
        return result

@dataclass
class Symbol:
    """
    Represents a symbol (variable, parameter, procedure) in the symbol table.
    
    Maintains comprehensive information about declared symbols including
    type information, scope, and usage tracking for semantic analysis.
    
    Attributes:
        name: Symbol identifier
        symbol_type: TAL data type
        location: Where symbol was declared
        scope: Scope where symbol is valid
        is_pointer: Whether symbol is a pointer type
        is_array: Whether symbol is an array
        array_bounds: Array size limits (if applicable)
        struct_name: Name of struct type (if applicable)
        procedure_params: Parameter list (for procedures)
        return_type: Return type (for procedures)
        references: List of locations where symbol is used
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
    references: List[SourceLocation] = field(default_factory=list)

@dataclass
class TALNode:
    """
    Represents a node in the Abstract Syntax Tree (AST).
    
    The fundamental building block for representing parsed TAL code structure.
    Each node can contain child nodes to form a hierarchical representation
    of the program structure.
    
    Attributes:
        type: Node type (e.g., 'procedure', 'assignment', 'expression')
        name: Node identifier (if applicable)
        value: Node value/content
        children: List of child nodes
        location: Source location for this node
        attributes: Additional metadata about the node
        symbol: Associated symbol table entry
        semantic_type: Resolved TAL type for type checking
    """
    type: str
    name: str = ""
    value: Any = None
    children: List['TALNode'] = field(default_factory=list)
    location: SourceLocation = field(default_factory=SourceLocation)
    attributes: Dict[str, Any] = field(default_factory=dict)
    symbol: Optional[Symbol] = None
    semantic_type: Optional[TALType] = None
    
    def add_child(self, child):
        """
        Add a child node to this node.
        
        Accepts either TALNode instances or primitive values that will be
        converted to token nodes automatically.
        
        Args:
            child: TALNode instance or primitive value to add as child
        """
        if isinstance(child, TALNode):
            self.children.append(child)
        else:
            # Convert primitive values to token nodes
            self.children.append(TALNode('token', value=str(child)))
    
    def find_children_by_type(self, node_type: str) -> List['TALNode']:
        """
        Find all direct children of a specific type.
        
        Searches only immediate children, not recursive descendants.
        Useful for finding specific constructs like parameters or statements.
        
        Args:
            node_type: Type of nodes to find
            
        Returns:
            List of matching child nodes
        """
        return [child for child in self.children if child.type == node_type]
    
    def find_child_by_name(self, name: str) -> Optional['TALNode']:
        """
        Find first child with specific name.
        
        Searches immediate children for a node with the given name.
        Returns the first match found.
        
        Args:
            name: Name to search for
            
        Returns:
            First matching child node or None if not found
        """
        for child in self.children:
            if child.name == name:
                return child
        return None
    
    
    def to_sexp(self, indent: int = 0) -> str:
        """
        Convert AST node to S-expression format for debugging and interoperability.
        
        Generates a human-readable tree representation that can be easily
        parsed by other tools or used for debugging and visualization.
        
        Args:
            indent: Current indentation level for pretty printing
            
        Returns:
            S-expression string representation of the node and its children
        """
        spaces = "  " * indent
        result = f"{spaces}({self.type}"
        
        # Add node name if present
        if self.name:
            result += f" :name {self.name}"
        
        # Add node value if present
        if self.value is not None:
            result += f" :value {self.value}"
        
        # Add attributes if present (formatted for readability)
        if self.attributes:
            attrs_str = "{"
            for key, value in self.attributes.items():
                if isinstance(value, str):
                    attrs_str += f"'{key}': '{value}', "
                else:
                    attrs_str += f"'{key}': {value}, "
            attrs_str = attrs_str.rstrip(", ") + "}"
            result += f" :attrs {attrs_str}"
        
        # Add children with proper indentation
        if self.children:
            result += "\n"
            for child in self.children:
                result += child.to_sexp(indent + 1) + "\n"
            result += f"{spaces})"
        else:
            result += ")"
        
        return result

class SymbolTable:
    """
    Manages symbol declarations and lookups across different scopes.
    
    Provides scope-aware symbol management for TAL parsing, handling
    global scope, procedure scopes, and nested block scopes. Tracks
    symbol declarations and enables proper scoping rules enforcement.
    
    Attributes:
        scopes: Dictionary mapping scope names to symbol dictionaries
        current_scope: Name of the currently active scope
        scope_stack: Stack of active scopes for nested scope handling
        struct_definitions: Global registry of struct type definitions
    """
    
    def __init__(self):
        """Initialize symbol table with global scope."""
        self.scopes: Dict[str, Dict[str, Symbol]] = {"global": {}}
        self.current_scope = "global"
        self.scope_stack: List[str] = ["global"]
        self.struct_definitions: Dict[str, Dict[str, Symbol]] = {}
        
    def enter_scope(self, scope_name: str):
        """
        Enter a new scope (e.g., when parsing a procedure).
        
        Pushes the new scope onto the scope stack and makes it current.
        Creates the scope if it doesn't already exist.
        
        Args:
            scope_name: Name of the scope to enter
        """
        self.scope_stack.append(scope_name)
        self.current_scope = scope_name
        if scope_name not in self.scopes:
            self.scopes[scope_name] = {}
        
    def exit_scope(self):
        """
        Exit the current scope and return to the previous scope.
        
        Pops the current scope from the stack. The global scope
        cannot be exited (always remains as the base scope).
        """
        if len(self.scope_stack) > 1:
            self.scope_stack.pop()
            self.current_scope = self.scope_stack[-1]
    
    def declare_symbol(self, symbol: Symbol) -> Optional[ParseError]:
        """
        Declare a symbol in the current scope.
        
        Adds a symbol to the current scope, checking for duplicate declarations.
        Returns an error if the symbol is already declared in the current scope.
        
        Args:
            symbol: Symbol to declare
            
        Returns:
            ParseError if symbol already exists, None if successful
        """
        symbol.scope = self.current_scope
        if self.current_scope not in self.scopes:
            self.scopes[self.current_scope] = {}
        
        # Check for duplicate declaration in current scope
        if symbol.name in self.scopes[self.current_scope]:
            existing = self.scopes[self.current_scope][symbol.name]
            return ParseError(
                f"Symbol '{symbol.name}' already declared in scope '{self.current_scope}'",
                symbol.location,
                ErrorSeverity.ERROR,
                error_code="E001",
                suggestions=[f"Previous declaration at {existing.location}"]
            )
        
        # Add symbol to current scope
        self.scopes[self.current_scope][symbol.name] = symbol
        return None
    
    def lookup_symbol(self, name: str) -> Optional[Symbol]:
        """
        Lookup symbol in current scope chain.
        
        Searches for a symbol starting from the current scope and walking
        up the scope chain to find the most recent declaration.
        
        Args:
            name: Symbol name to look up
            
        Returns:
            Symbol if found, None otherwise
        """
        # Search from most recent scope to global scope
        for scope in reversed(self.scope_stack):
            if scope in self.scopes and name in self.scopes[scope]:
                return self.scopes[scope][name]
        return None

@dataclass
class ProcedureInfo:
    """
    Metadata about a TAL procedure declaration.
    
    Stores comprehensive information about procedure signatures, attributes,
    and declaration characteristics for use in semantic analysis and
    code generation.
    
    Attributes:
        name: Procedure name
        return_type: Return data type (None for void procedures)
        parameters: List of parameter symbols
        attributes: Procedure attributes (MAIN, FORWARD, etc.)
        is_forward: Whether this is a forward declaration
        is_external: Whether this is an external procedure reference
        location: Source location of the procedure declaration
    """
    name: str
    return_type: Optional[TALType] = None
    parameters: List[Symbol] = field(default_factory=list)
    attributes: List[str] = field(default_factory=list)
    is_forward: bool = False
    is_external: bool = False
    location: SourceLocation = field(default_factory=SourceLocation)

def parse_tal_type(type_str: str) -> TALType:
    """
    Parse TAL type string into TALType enum.
    
    Converts string representations of TAL types (including parameterized
    types like INT(32)) into the corresponding TALType enumeration value.
    Handles both simple and complex type specifications.
    
    Args:
        type_str: String representation of TAL type
        
    Returns:
        Corresponding TALType enum value, UNKNOWN if not recognized
    """
    type_str = type_str.upper().strip()
    
    # Handle parameterized integer types
    if type_str.startswith("INT("):
        if "32" in type_str:
            return TALType.INT32
        elif "64" in type_str:
            return TALType.INT64
        elif "16" in type_str:
            return TALType.INT16
        else:
            return TALType.INT
    
    # Handle parameterized real types
    elif type_str.startswith("REAL("):
        if "64" in type_str:
            return TALType.REAL64
        elif "32" in type_str:
            return TALType.REAL32
        else:
            return TALType.REAL
    
    # Handle unsigned types
    elif type_str.startswith("UNSIGNED"):
        return TALType.UNSIGNED
    
    # Handle simple types by direct enum lookup
    try:
        return TALType(type_str)
    except ValueError:
        return TALType.UNKNOWN

def find_procedure_declarations(tal_code: str) -> List[Tuple[int, str, str]]:
    """
    Find all procedure declarations in TAL code.
    
    Scans the TAL source code to locate procedure declarations, handling
    multi-line declarations properly by scanning until the terminating
    semicolon. This is critical for TAL parsing since procedure declarations
    can span multiple lines with parameter lists and type declarations.
    
    The function handles:
    - Multi-line procedure declarations
    - Inline comments
    - Various procedure return types
    - TAL-specific identifier syntax (including ^ characters)
    
    Args:
        tal_code: Complete TAL source code as string
        
    Returns:
        List of tuples containing (line_number, proc_name, full_declaration)
        where line_number is 1-based, proc_name is the procedure identifier,
        and full_declaration is the complete multi-line declaration text
    """
    lines = tal_code.split('\n')
    procedures = []
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip empty lines and comment-only lines
        if not line or line.startswith('!'):
            i += 1
            continue
        
        # Remove inline comments to get clean code for analysis
        comment_pos = line.find('!')
        if comment_pos >= 0:
            code_part = line[:comment_pos].strip()
        else:
            code_part = line
        
        # Skip lines with no actual code content
        if not code_part:
            i += 1
            continue
        
        # Look for procedure declaration using comprehensive regex
        # Matches optional return type followed by PROC and procedure name
        # Handles TAL-specific naming with ^ characters
        proc_match = re.search(
            r'\b(?:(INT(?:\([^)]*\))?|REAL(?:\([^)]*\))?|STRING|FIXED|UNSIGNED(?:\([^)]*\))?)\s+)?'
            r'PROC\s+([a-zA-Z_][a-zA-Z0-9_^]*)', 
            code_part, 
            re.IGNORECASE
        )
        
        if proc_match:
            start_line = i + 1  # Convert to 1-based line numbering
            proc_name = proc_match.group(2)
            
            # Collect all lines until we find the terminating semicolon
            # This is essential for TAL because procedure declarations
            # can span many lines with parameter lists and type declarations
            declaration_lines = []
            j = i
            found_semicolon = False
            
            while j < len(lines) and not found_semicolon:
                current_line = lines[j]
                declaration_lines.append(current_line)
                
                # Check if this line contains a semicolon outside of comments
                comment_pos = current_line.find('!')
                if comment_pos >= 0:
                    code_part = current_line[:comment_pos]
                else:
                    code_part = current_line
                
                # Found the terminating semicolon
                if ';' in code_part:
                    found_semicolon = True
                
                j += 1
            
            # Join all declaration lines and add to results
            full_declaration = '\n'.join(declaration_lines)
            procedures.append((start_line, proc_name, full_declaration))
            
            # Continue parsing from after the semicolon
            i = j
        else:
            i += 1
    
    return procedures


def extract_parameters_from_declaration(declaration: str) -> List[str]:
    """
    Extract parameter names from a procedure declaration.
    
    Parses the parameter list from a TAL procedure declaration, handling
    multi-line declarations and proper parentheses matching. Cleans up
    comments and normalizes whitespace for reliable parsing.
    
    Args:
        declaration: Complete procedure declaration text
        
    Returns:
        List of parameter name strings (may include type prefixes)
    """
    # Remove all comments from the declaration for clean parsing
    clean_decl = re.sub(r'!.*$', '', declaration, flags=re.MULTILINE)
    
    # Normalize all whitespace to single spaces
    clean_decl = ' '.join(clean_decl.split())
    
    # Find the parameter list within parentheses
    # Uses simple regex since we've already cleaned the declaration
    paren_match = re.search(r'\(([^)]*)\)', clean_decl)
    if not paren_match:
        return []  # No parameter list found
    
    param_string = paren_match.group(1).strip()
    if not param_string:
        return []  # Empty parameter list
    
    # Split parameters on commas and clean up each parameter name
    parameters = []
    for param in param_string.split(','):
        param = param.strip()
        if param:  # Skip empty parameters
            parameters.append(param)
    
    return parameters

def parse_procedure_declaration(proc_name: str, declaration: str, start_line: int, filename: str, symbol_table: SymbolTable) -> Tuple[Optional[TALNode], List[ParseError]]:
    """
    Parse a single procedure declaration into a TALNode.
    
    Converts a complete TAL procedure declaration into a structured AST node
    with full semantic information including return types, parameters, and
    procedure attributes. Handles both simple and complex declarations.
    
    The function:
    1. Creates the main procedure node with metadata
    2. Extracts and parses the return type
    3. Processes the parameter list
    4. Identifies procedure attributes (MAIN, FORWARD, etc.)
    5. Parses parameter type declarations that follow the main declaration
    6. Updates the symbol table with parameter information
    7. Creates standard procedure sections for non-forward procedures
    
    Args:
        proc_name: Name of the procedure to parse
        declaration: Complete multi-line declaration text
        start_line: Starting line number (1-based) for error reporting
        filename: Source filename for error reporting
        symbol_table: Symbol table to update with procedure and parameter symbols
        
    Returns:
        Tuple of (TALNode for the procedure, list of parsing errors)
        TALNode will be None if parsing fails completely
    """
    errors = []
    location = SourceLocation(filename, start_line, 1)
    
    # Create the main procedure node
    proc_node = TALNode('procedure', name=proc_name, location=location)
    
    # Enter procedure scope for parameter declarations
    symbol_table.enter_scope(proc_name)
    
    # Extract return type from declaration
    # Looks for type keyword before PROC
    return_type_match = re.search(
        r'\b(INT(?:\([^)]*\))?|REAL(?:\([^)]*\))?|STRING|FIXED|UNSIGNED(?:\([^)]*\))?)\s+PROC', 
        declaration, 
        re.IGNORECASE
    )
    if return_type_match:
        return_type = parse_tal_type(return_type_match.group(1))
        proc_node.attributes['return_type'] = return_type.value
    
    # Extract and parse parameter list
    parameters = extract_parameters_from_declaration(declaration)
    if parameters:
        # Create parameters container node
        params_node = TALNode('parameters', location=location)
        
        # Process each parameter
        for param_name in parameters:
            param_name = param_name.strip()
            if param_name:
                # Create parameter node with unknown type initially
                param_node = TALNode('parameter', name=param_name, location=location)
                param_node.attributes['type'] = TALType.UNKNOWN.value
                params_node.add_child(param_node)
                
                # Add parameter to symbol table
                # Handle pointer parameters (names starting with .)
                clean_name = param_name.lstrip('.')
                symbol = Symbol(
                    name=clean_name,
                    symbol_type=TALType.UNKNOWN,
                    location=location,
                    is_pointer=param_name.startswith('.')
                )
                error = symbol_table.declare_symbol(symbol)
                if error:
                    errors.append(error)
        
        proc_node.add_child(params_node)
    
    # Check for procedure attributes in the declaration
    attributes = []
    if re.search(r'\bMAIN\b', declaration, re.IGNORECASE):
        attributes.append('MAIN')
        proc_node.attributes['is_main'] = True
    if re.search(r'\bFORWARD\b', declaration, re.IGNORECASE):
        attributes.append('FORWARD')
        proc_node.attributes['is_forward'] = True
    if re.search(r'\bEXTERNAL\b', declaration, re.IGNORECASE):
        attributes.append('EXTERNAL')
        proc_node.attributes['is_external'] = True
    if re.search(r'\bVARIABLE\b', declaration, re.IGNORECASE):
        attributes.append('VARIABLE')
    if re.search(r'\bEXTENSIBLE\b', declaration, re.IGNORECASE):
        attributes.append('EXTENSIBLE')
    
    if attributes:
        proc_node.attributes['attributes'] = attributes
    
    # Parse parameter type declarations that follow the main procedure line
    # These specify the actual types of the parameters
    decl_lines = declaration.split('\n')
    param_type_lines = []
    
    # Skip the main procedure declaration line and collect type declarations
    for i, line in enumerate(decl_lines[1:], 1):
        line = line.strip()
        # Skip comments and BEGIN statements
        if line and not line.startswith('!') and not re.search(r'\bBEGIN\b', line, re.IGNORECASE):
            # Remove trailing semicolon for processing
            if line.endswith(';'):
                line = line[:-1].strip()
            if line:
                param_type_lines.append(line)
    
    # Process parameter type declarations
    for line in param_type_lines:
        # Match type declarations like "INT x, y;" or "STRING .buffer;"
        type_match = re.match(
            r'(INT(?:\([^)]+\))?|REAL(?:\([^)]+\))?|STRING|FIXED|UNSIGNED(?:\([^)]+\))?|STRUCT)\s+(.+)', 
            line, 
            re.IGNORECASE
        )
        
        if type_match:
            data_type = parse_tal_type(type_match.group(1))
            var_names = type_match.group(2)
            
            # Parse variable names from the declaration
            for var in var_names.split(','):
                var = var.strip()
                is_pointer = var.startswith('.')
                var_name = var.lstrip('.')
                
                # Update parameter type in the AST
                params_node = None
                for child in proc_node.children:
                    if child.type == 'parameters':
                        params_node = child
                        break
                
                if params_node:
                    # Find and update the corresponding parameter node
                    for param_node in params_node.children:
                        if param_node.type == 'parameter':
                            # Match parameter names with or without dots
                            if (param_node.name == var_name or 
                                param_node.name == var or
                                param_node.name.lstrip('.') == var_name):
                                param_node.attributes['type'] = data_type.value
                                param_node.semantic_type = data_type
                                if is_pointer:
                                    param_node.attributes['pointer'] = True
                
                # Update symbol table with type information
                symbol = symbol_table.lookup_symbol(var_name)
                if symbol:
                    symbol.symbol_type = data_type
                    symbol.is_pointer = is_pointer
    
    # Add standard procedure sections for procedures with bodies
    # FORWARD and EXTERNAL procedures don't have bodies
    if not proc_node.attributes.get('is_forward') and not proc_node.attributes.get('is_external'):
        # Add local declarations section (for variables declared in the procedure)
        proc_node.add_child(TALNode('local_declarations', location=location))
        # Add statements section (for executable code)
        proc_node.add_child(TALNode('statements', location=location))
    
    return proc_node, errors

def parse_multiple_procedures(tal_code: str, filename: str, symbol_table: SymbolTable) -> Tuple[List[TALNode], List[ParseError]]:
    """
    Parse multiple TAL procedures from a code block.
    
    The main entry point for parsing TAL procedure declarations from source code.
    Orchestrates the parsing of all procedures found in the code and manages
    symbol table scope transitions.
    
    Process:
    1. Find all procedure declarations in the source
    2. Parse each declaration individually
    3. Manage symbol table scopes
    4. Collect all errors for reporting
    5. Return structured results
    
    Args:
        tal_code: Complete TAL source code
        filename: Source filename for error reporting
        symbol_table: Symbol table to populate during parsing
        
    Returns:
        Tuple of (list of procedure TALNodes, list of parsing errors)
    """
    procedures = []
    errors = []
    
    # Find all procedure declarations in the source code
    proc_declarations = find_procedure_declarations(tal_code)
    
    # Parse each procedure declaration
    for start_line, proc_name, declaration in proc_declarations:
        proc_node, proc_errors = parse_procedure_declaration(
            proc_name, declaration, start_line, filename, symbol_table
        )
        
        # Collect successful parse results
        if proc_node:
            procedures.append(proc_node)
        
        # Collect all errors for reporting
        errors.extend(proc_errors)
        
        # Exit the procedure scope to return to global scope
        symbol_table.exit_scope()
    
    return procedures, errors

def main():
    """
    Main function to process TAL file from command line argument.
    
    Provides a command-line interface for testing the TAL procedure parser.
    Reads a TAL source file, parses all procedures, and displays results
    including AST structure and any parsing errors.
    
    Usage: python tal_proc_parser.py <tal_file>
    
    The function:
    1. Validates command line arguments
    2. Checks file existence and readability
    3. Parses the TAL source code
    4. Displays parsing results and errors
    5. Provides summary statistics
    
    Exit codes:
        0: Success
        1: Error (file not found, parsing error, etc.)
    """
    import sys
    import os
    
    # Validate command line arguments
    if len(sys.argv) != 2:
        print("Usage: python tal_parser.py <tal_file>")
        print("Example: python tal_parser.py input.tal")
        sys.exit(1)
    
    tal_file = sys.argv[1]
    
    # Check if the specified file exists
    if not os.path.exists(tal_file):
        print(f"Error: File '{tal_file}' not found")
        sys.exit(1)
    
    try:
        # Read the TAL source file
        with open(tal_file, 'r', encoding='utf-8') as f:
            tal_code = f.read()
        
        # Display parsing header
        print(f"Parsing TAL file: {tal_file}")
        print("=" * 60)
        
        # Create symbol table and parse all procedures
        symbol_table = SymbolTable()
        procedures, errors = parse_multiple_procedures(tal_code, tal_file, symbol_table)
        
        # Display any parsing errors found
        if errors:
            print(f"\nFound {len(errors)} error(s):")
            print("-" * 40)
            for error in errors:
                print(f"{error}\n")
        
        # Display parsed procedures with their AST structure
        if procedures:
            print(f"\nFound {len(procedures)} procedure(s):")
            print("-" * 40)
            for i, proc in enumerate(procedures, 1):
                print(f"\nProcedure {i}: {proc.name}")
                print(proc.to_sexp())
                print()
        
        # Handle case where no procedures were found
        if not procedures and not errors:
            print("No procedures found in the file.")
        
        # Display parsing summary statistics
        print("\nParsing Summary:")
        print(f"  Procedures found: {len(procedures)}")
        print(f"  Errors: {len(errors)}")
        print(f"  Warnings: {len([e for e in errors if e.severity == ErrorSeverity.WARNING])}")
        
    except FileNotFoundError:
        print(f"Error: Could not open file '{tal_file}'")
        sys.exit(1)
    except UnicodeDecodeError:
        print(f"Error: Could not decode file '{tal_file}' as UTF-8")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
