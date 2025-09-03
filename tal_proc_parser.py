import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

class TALType(Enum):
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
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    FATAL = "fatal"

@dataclass
class SourceLocation:
    filename: str = ""
    line: int = 0
    column: int = 0
    length: int = 0
    
    def __str__(self):
        return f"{self.filename}:{self.line}:{self.column}"

@dataclass
class ParseError:
    message: str
    location: SourceLocation
    severity: ErrorSeverity
    context_lines: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    error_code: str = ""
    
    def __str__(self):
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
    type: str
    name: str = ""
    value: Any = None
    children: List['TALNode'] = field(default_factory=list)
    location: SourceLocation = field(default_factory=SourceLocation)
    attributes: Dict[str, Any] = field(default_factory=dict)
    symbol: Optional[Symbol] = None
    semantic_type: Optional[TALType] = None
    
    def add_child(self, child):
        if isinstance(child, TALNode):
            self.children.append(child)
        else:
            self.children.append(TALNode('token', value=str(child)))
    
    def find_children_by_type(self, node_type: str) -> List['TALNode']:
        """Find all direct children of a specific type"""
        return [child for child in self.children if child.type == node_type]
    
    def find_child_by_name(self, name: str) -> Optional['TALNode']:
        """Find first child with specific name"""
        for child in self.children:
            if child.name == name:
                return child
        return None
    
    def to_sexp(self, indent: int = 0) -> str:
        """Convert to S-expression format for debugging"""
        spaces = "  " * indent
        result = f"{spaces}({self.type}"
        
        if self.name:
            result += f" :name {self.name}"
        if self.value is not None:
            result += f" :value {self.value}"
        if self.attributes:
            # Format attributes more clearly
            attrs_str = "{"
            for key, value in self.attributes.items():
                if isinstance(value, str):
                    attrs_str += f"'{key}': '{value}', "
                else:
                    attrs_str += f"'{key}': {value}, "
            attrs_str = attrs_str.rstrip(", ") + "}"
            result += f" :attrs {attrs_str}"
        
        if self.children:
            result += "\n"
            for child in self.children:
                result += child.to_sexp(indent + 1) + "\n"
            result += f"{spaces})"
        else:
            result += ")"
        
        return result

class SymbolTable:
    def __init__(self):
        self.scopes: Dict[str, Dict[str, Symbol]] = {"global": {}}
        self.current_scope = "global"
        self.scope_stack: List[str] = ["global"]
        self.struct_definitions: Dict[str, Dict[str, Symbol]] = {}
        
    def enter_scope(self, scope_name: str):
        """Enter a new scope"""
        self.scope_stack.append(scope_name)
        self.current_scope = scope_name
        if scope_name not in self.scopes:
            self.scopes[scope_name] = {}
        
    def exit_scope(self):
        """Exit current scope"""
        if len(self.scope_stack) > 1:
            self.scope_stack.pop()
            self.current_scope = self.scope_stack[-1]
    
    def declare_symbol(self, symbol: Symbol) -> Optional[ParseError]:
        """Declare a symbol in current scope"""
        symbol.scope = self.current_scope
        if self.current_scope not in self.scopes:
            self.scopes[self.current_scope] = {}
        
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
    
    def lookup_symbol(self, name: str) -> Optional[Symbol]:
        """Lookup symbol in current scope chain"""
        for scope in reversed(self.scope_stack):
            if scope in self.scopes and name in self.scopes[scope]:
                return self.scopes[scope][name]
        return None

@dataclass
class ProcedureInfo:
    name: str
    return_type: Optional[TALType] = None
    parameters: List[Symbol] = field(default_factory=list)
    attributes: List[str] = field(default_factory=list)
    is_forward: bool = False
    is_external: bool = False
    location: SourceLocation = field(default_factory=SourceLocation)

def parse_tal_type(type_str: str) -> TALType:
    """Parse TAL type string into TALType enum"""
    type_str = type_str.upper().strip()
    
    # Handle parameterized types
    if type_str.startswith("INT("):
        if "32" in type_str:
            return TALType.INT32
        elif "64" in type_str:
            return TALType.INT64
        elif "16" in type_str:
            return TALType.INT16
        else:
            return TALType.INT
    elif type_str.startswith("REAL("):
        if "64" in type_str:
            return TALType.REAL64
        elif "32" in type_str:
            return TALType.REAL32
        else:
            return TALType.REAL
    elif type_str.startswith("UNSIGNED"):
        return TALType.UNSIGNED
    
    # Handle simple types
    try:
        return TALType(type_str)
    except ValueError:
        return TALType.UNKNOWN

def find_procedure_declarations(tal_code: str) -> List[Tuple[int, str, str]]:
    """
    Find all procedure declarations in TAL code.
    Returns list of (line_number, proc_name, full_declaration) tuples.
    """
    lines = tal_code.split('\n')
    procedures = []
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip empty lines
        if not line:
            i += 1
            continue
        
        # Skip comment lines (lines that start with !)
        if line.startswith('!'):
            i += 1
            continue
        
        # Check if line contains a comment and extract the non-comment part
        comment_pos = line.find('!')
        if comment_pos >= 0:
            # Only consider the part before the comment
            code_part = line[:comment_pos].strip()
        else:
            code_part = line
        
        # Skip if the entire line is a comment or empty after removing comments
        if not code_part:
            i += 1
            continue
        
        # Look for procedure declaration pattern (case insensitive)
        # Only search in the non-comment part of the line
        proc_match = re.search(r'\b(?:(INT(?:\([^)]*\))?|REAL(?:\([^)]*\))?|STRING|FIXED|UNSIGNED(?:\([^)]*\))?)\s+)?PROC\s+([a-zA-Z_][a-zA-Z0-9_^]*)', code_part, re.IGNORECASE)
        
        if proc_match:
            start_line = i + 1  # 1-based line numbering
            return_type = proc_match.group(1) if proc_match.group(1) else None
            proc_name = proc_match.group(2)
            
            # Collect the full procedure declaration
            declaration_lines = [lines[i]]
            j = i + 1
            
            # Continue collecting lines until we find a complete declaration
            # (ends with semicolon) or hit BEGIN
            while j < len(lines):
                current_line = lines[j].strip()
                declaration_lines.append(lines[j])
                
                # Check if this line completes the declaration
                if current_line.endswith(';'):
                    break
                
                # Stop if we hit BEGIN (procedure body starts)
                if re.search(r'\bBEGIN\b', current_line, re.IGNORECASE):
                    break
                
                j += 1
            
            full_declaration = '\n'.join(declaration_lines)
            procedures.append((start_line, proc_name, full_declaration))
            i = j + 1
        else:
            i += 1
    
    return procedures

def extract_parameters_from_declaration(declaration: str) -> List[str]:
    """
    Extract parameter names from a procedure declaration.
    Handles multi-line declarations and proper parentheses matching.
    """
    # Remove comments and clean up the declaration
    clean_decl = re.sub(r'!.*$', '', declaration, flags=re.MULTILINE)
    clean_decl = ' '.join(clean_decl.split())  # Normalize whitespace
    
    # Find the parameter list in parentheses
    paren_match = re.search(r'\(([^)]*)\)', clean_decl)
    if not paren_match:
        return []
    
    param_string = paren_match.group(1).strip()
    if not param_string:
        return []
    
    # Split on commas and clean up parameter names
    parameters = []
    for param in param_string.split(','):
        param = param.strip()
        if param:
            parameters.append(param)
    
    return parameters

def parse_procedure_declaration(proc_name: str, declaration: str, start_line: int, filename: str, symbol_table: SymbolTable) -> Tuple[Optional[TALNode], List[ParseError]]:
    """
    Parse a single procedure declaration into a TALNode.
    """
    errors = []
    location = SourceLocation(filename, start_line, 1)
    
    # Create procedure node
    proc_node = TALNode('procedure', name=proc_name, location=location)
    symbol_table.enter_scope(proc_name)
    
    # Extract return type
    return_type_match = re.search(r'\b(INT(?:\([^)]*\))?|REAL(?:\([^)]*\))?|STRING|FIXED|UNSIGNED(?:\([^)]*\))?)\s+PROC', declaration, re.IGNORECASE)
    if return_type_match:
        return_type = parse_tal_type(return_type_match.group(1))
        proc_node.attributes['return_type'] = return_type.value
    
    # Extract and parse parameters
    parameters = extract_parameters_from_declaration(declaration)
    if parameters:
        params_node = TALNode('parameters', location=location)
        
        for param_name in parameters:
            param_name = param_name.strip()
            if param_name:
                param_node = TALNode('parameter', name=param_name, location=location)
                param_node.attributes['type'] = TALType.UNKNOWN.value
                params_node.add_child(param_node)
                
                # Add to symbol table
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
    
    # Check for procedure attributes
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
    
    # Parse parameter type declarations that follow the procedure declaration
    decl_lines = declaration.split('\n')
    param_type_lines = []
    
    # Skip the main procedure declaration line and collect parameter type declarations
    for i, line in enumerate(decl_lines[1:], 1):
        line = line.strip()
        if line and not line.startswith('!') and not re.search(r'\bBEGIN\b', line, re.IGNORECASE):
            if line.endswith(';'):
                line = line[:-1].strip()
            if line:
                param_type_lines.append(line)
    
    # Parse parameter type declarations
    for line in param_type_lines:
        # Match type declarations like "INT x, y;" or "STRING .buffer;"
        type_match = re.match(r'(INT(?:\([^)]+\))?|REAL(?:\([^)]+\))?|STRING|FIXED|UNSIGNED(?:\([^)]+\))?|STRUCT)\s+(.+)', line, re.IGNORECASE)
        
        if type_match:
            data_type = parse_tal_type(type_match.group(1))
            var_names = type_match.group(2)
            
            # Parse variable names
            for var in var_names.split(','):
                var = var.strip()
                is_pointer = var.startswith('.')
                var_name = var.lstrip('.')
                
                # Update parameter type in AST
                params_node = None
                for child in proc_node.children:
                    if child.type == 'parameters':
                        params_node = child
                        break
                
                if params_node:
                    for param_node in params_node.children:
                        if param_node.type == 'parameter':
                            # Match parameter names (with or without dots)
                            if (param_node.name == var_name or 
                                param_node.name == var or
                                param_node.name.lstrip('.') == var_name):
                                param_node.attributes['type'] = data_type.value
                                param_node.semantic_type = data_type
                                if is_pointer:
                                    param_node.attributes['pointer'] = True
                
                # Update symbol table
                symbol = symbol_table.lookup_symbol(var_name)
                if symbol:
                    symbol.symbol_type = data_type
                    symbol.is_pointer = is_pointer
    
    # Add standard procedure sections for non-forward/external procedures
    if not proc_node.attributes.get('is_forward') and not proc_node.attributes.get('is_external'):
        proc_node.add_child(TALNode('local_declarations', location=location))
        proc_node.add_child(TALNode('statements', location=location))
    
    return proc_node, errors

def parse_multiple_procedures(tal_code: str, filename: str, symbol_table: SymbolTable) -> Tuple[List[TALNode], List[ParseError]]:
    """
    Parse multiple TAL procedures from a code block.
    """
    procedures = []
    errors = []
    
    # Find all procedure declarations
    proc_declarations = find_procedure_declarations(tal_code)
    
    for start_line, proc_name, declaration in proc_declarations:
        proc_node, proc_errors = parse_procedure_declaration(
            proc_name, declaration, start_line, filename, symbol_table
        )
        
        if proc_node:
            procedures.append(proc_node)
        errors.extend(proc_errors)
        
        # Exit the procedure scope
        symbol_table.exit_scope()
    
    return procedures, errors

def main():
    """Main function to process TAL file from command line argument"""
    import sys
    import os
    
    if len(sys.argv) != 2:
        print("Usage: python tal_parser.py <tal_file>")
        print("Example: python tal_parser.py input.tal")
        sys.exit(1)
    
    tal_file = sys.argv[1]
    
    # Check if file exists
    if not os.path.exists(tal_file):
        print(f"Error: File '{tal_file}' not found")
        sys.exit(1)
    
    try:
        # Read the TAL file
        with open(tal_file, 'r', encoding='utf-8') as f:
            tal_code = f.read()
        
        print(f"Parsing TAL file: {tal_file}")
        print("=" * 60)
        
        # Create symbol table and parse procedures
        symbol_table = SymbolTable()
        procedures, errors = parse_multiple_procedures(tal_code, tal_file, symbol_table)
        
        # Display results
        if errors:
            print(f"\nFound {len(errors)} error(s):")
            print("-" * 40)
            for error in errors:
                print(f"{error}\n")
        
        if procedures:
            print(f"\nFound {len(procedures)} procedure(s):")
            print("-" * 40)
            for i, proc in enumerate(procedures, 1):
                print(f"\nProcedure {i}: {proc.name}")
                print(proc.to_sexp())
                print()
        
        if not procedures and not errors:
            print("No procedures found in the file.")
        
        # Summary
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
