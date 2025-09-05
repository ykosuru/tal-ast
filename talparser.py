"""
Enhanced TAL Parser - A comprehensive parser for Transaction Application Language (TAL)

APPROACH OVERVIEW:
=================

This parser implements a hybrid two-stage parsing approach that combines the strengths of
two different parsing strategies:

1. FOUNDATION PARSING (tal_proc_parser):
   - Uses the existing tal_proc_parser module as a foundation
   - Handles procedure detection and basic structural parsing
   - Provides reliable procedure boundary detection and parameter parsing
   - Creates the basic AST skeleton with procedures, parameters, and declarations

2. ENHANCEMENT PARSING (rich AST generation):
   - Adds detailed parsing for procedure bodies, expressions, and statements
   - Implements comprehensive system function detection and validation
   - Provides detailed analysis of TAL-specific constructs (bit fields, operators)
   - Generates rich metadata and attributes for all AST nodes

DESIGN:
=======

- Modular Design: Separate classes for system functions, operators, and main parser
- Error Resilience: Continues parsing even when encountering errors
- Comprehensive Coverage: Handles all major TAL constructs including:
  * Procedure declarations (PROC, INT PROC, etc.)
  * System functions ($LEN, $OCCURS, $DBL, etc.)
  * Control flow (IF, WHILE, FOR, CASE, SCAN)
  * TAL-specific features (bit fields, pointer operations)
  * Compiler directives (?PAGE, ?SECTION, etc.)
  * Global declarations (STRUCT, TEMPLATE, LITERAL, etc.)

- Rich Metadata: Each AST node contains detailed attributes for analysis
- Validation: Built-in validation for system function usage and argument counts
- Extensibility: Easy to add new system functions and language constructs

PARSING FLOW:
============

1. File Reading → Content Preprocessing
2. Foundation Parsing → Basic procedure structure via tal_proc_parser
3. Line Range Calculation → Determine procedure boundaries
4. Enhancement Phase → Rich parsing of procedure bodies and global content
5. AST Assembly → Combine enhanced procedures with global content
6. Validation → Check system function usage and generate warnings
7. Output Generation → Produce S-expressions, JSON, or analysis reports

This approach ensures both reliability (from the proven tal_proc_parser) and
comprehensiveness (from the enhanced parsing logic).
"""

import re
import argparse
import os
import json
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum

# Import tal_proc_parser module to parse PROCs
import tal_proc_parser

class TALSystemFunctions:
    """
    Comprehensive registry of TAL system functions with parameter counts and return types.
    
    This class maintains a complete catalog of all known TAL system functions,
    organized by category for easier maintenance and lookup. Each function entry
    includes parameter count, return type, and description for validation and
    documentation purposes.
    """
    
    # String manipulation and information functions
    STRING_FUNCTIONS = {
        '$LEN': {'params': 1, 'returns': 'INT', 'description': 'Length of string'},
        '$OCCURS': {'params': 1, 'returns': 'INT', 'description': 'Number of occurrences'},
        '$TYPE': {'params': 1, 'returns': 'INT', 'description': 'Data type'},
        '$BITLENGTH': {'params': 1, 'returns': 'INT', 'description': 'Bit length'},
        '$BYTELENGTH': {'params': 1, 'returns': 'INT', 'description': 'Byte length'},
        '$OFFSET': {'params': 1, 'returns': 'INT', 'description': 'Offset in structure'},
    }
    
    # Numeric conversion and casting functions  
    NUMERIC_FUNCTIONS = {
        '$DBL': {'params': 1, 'returns': 'REAL*8', 'description': 'Convert to double'},
        '$FIX': {'params': 1, 'returns': 'FIXED', 'description': 'Convert to fixed'},
        '$FLOAT': {'params': 1, 'returns': 'REAL', 'description': 'Convert to float'},
        '$IFIX': {'params': 1, 'returns': 'INT', 'description': 'Convert to integer'},
        '$SNGL': {'params': 1, 'returns': 'REAL', 'description': 'Convert to single precision'},
        '$UDBL': {'params': 1, 'returns': 'REAL*8', 'description': 'Unsigned to double'},
        '$UFLOAT': {'params': 1, 'returns': 'REAL', 'description': 'Unsigned to float'},
    }
    
    # Bitwise operation functions
    BIT_FUNCTIONS = {
        '$BITAND': {'params': 2, 'returns': 'INT', 'description': 'Bitwise AND'},
        '$BITOR': {'params': 2, 'returns': 'INT', 'description': 'Bitwise OR'},
        '$BITXOR': {'params': 2, 'returns': 'INT', 'description': 'Bitwise XOR'},
        '$BITNOT': {'params': 1, 'returns': 'INT', 'description': 'Bitwise NOT'},
        '$SHIFTL': {'params': 2, 'returns': 'INT', 'description': 'Shift left'},
        '$SHIFTR': {'params': 2, 'returns': 'INT', 'description': 'Shift right'},
        '$ROTATEL': {'params': 2, 'returns': 'INT', 'description': 'Rotate left'},
        '$ROTATER': {'params': 2, 'returns': 'INT', 'description': 'Rotate right'},
    }
    
    # Array manipulation and boundary functions
    ARRAY_FUNCTIONS = {
        '$HIGH': {'params': 1, 'returns': 'INT', 'description': 'High bound of array'},
        '$BOUNDS': {'params': 1, 'returns': 'INT', 'description': 'Array bounds'},
        '$DIMENSION': {'params': 1, 'returns': 'INT', 'description': 'Array dimension'},
    }
    
    # Mathematical computation functions
    MATH_FUNCTIONS = {
        '$ABS': {'params': 1, 'returns': 'NUMERIC', 'description': 'Absolute value'},
        '$MAX': {'params': 2, 'returns': 'NUMERIC', 'description': 'Maximum value'},
        '$MIN': {'params': 2, 'returns': 'NUMERIC', 'description': 'Minimum value'},
        '$SQRT': {'params': 1, 'returns': 'REAL', 'description': 'Square root'},
        '$EXP': {'params': 1, 'returns': 'REAL', 'description': 'Exponential'},
        '$LOG': {'params': 1, 'returns': 'REAL', 'description': 'Natural logarithm'},
        '$SIN': {'params': 1, 'returns': 'REAL', 'description': 'Sine'},
        '$COS': {'params': 1, 'returns': 'REAL', 'description': 'Cosine'},
        '$TAN': {'params': 1, 'returns': 'REAL', 'description': 'Tangent'},
        '$ATAN': {'params': 1, 'returns': 'REAL', 'description': 'Arctangent'},
    }
    
    # System state and process information functions
    SYSTEM_FUNCTIONS = {
        '$AXADR': {'params': 1, 'returns': 'INT', 'description': 'Address of parameter'},
        '$PARAM': {'params': 1, 'returns': 'ANY', 'description': 'Parameter value'},
        '$SPECIAL': {'params': -1, 'returns': 'ANY', 'description': 'Special system function'},
        '$CARRY': {'params': 0, 'returns': 'INT', 'description': 'Carry flag'},
        '$OVERFLOW': {'params': 0, 'returns': 'INT', 'description': 'Overflow flag'},
        '$READCLOCK': {'params': 0, 'returns': 'FIXED', 'description': 'Read system clock'},
        '$MYGMOM': {'params': 0, 'returns': 'INT', 'description': 'Current GMOM'},
        '$MYSEGMENT': {'params': 0, 'returns': 'INT', 'description': 'Current segment'},
    }
    
    # Input/output and file operation functions
    IO_FUNCTIONS = {
        '$RECEIVEINFO': {'params': 0, 'returns': 'INT', 'description': 'Receive information'},
        '$FILENAME_TO_FILE_': {'params': 2, 'returns': 'INT', 'description': 'Convert filename'},
        '$FILE_TO_FILENAME_': {'params': 2, 'returns': 'INT', 'description': 'Convert to filename'},
        '$DISPLAY': {'params': -1, 'returns': 'VOID', 'description': 'Display output (variable args)'},
    }
    
    @classmethod
    def get_all_functions(cls) -> Dict[str, Dict[str, Any]]:
        """
        Combine all system function categories into a single registry.
        
        Returns:
            Dict containing all system functions with their metadata
        """
        all_functions = {}
        all_functions.update(cls.STRING_FUNCTIONS)
        all_functions.update(cls.NUMERIC_FUNCTIONS)
        all_functions.update(cls.BIT_FUNCTIONS)
        all_functions.update(cls.ARRAY_FUNCTIONS)
        all_functions.update(cls.MATH_FUNCTIONS)
        all_functions.update(cls.SYSTEM_FUNCTIONS)
        all_functions.update(cls.IO_FUNCTIONS)
        return all_functions
    
    @classmethod
    def is_system_function(cls, name: str) -> bool:
        """
        Check if a given name represents a known TAL system function.
        
        Args:
            name: Function name to check (with or without $ prefix)
            
        Returns:
            True if the name is a recognized system function
        """
        name_upper = name.upper()
        if not name_upper.startswith('$'):
            name_upper = '$' + name_upper
        return name_upper in cls.get_all_functions()
    
    @classmethod
    def get_function_info(cls, name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve detailed information about a system function.
        
        Args:
            name: Function name to look up (with or without $ prefix)
            
        Returns:
            Dictionary with function metadata or None if not found
        """
        name_upper = name.upper()
        if not name_upper.startswith('$'):
            name_upper = '$' + name_upper
        return cls.get_all_functions().get(name_upper)

class TALOperators:
    """
    Registry of TAL operators organized by type for expression parsing and validation.
    
    This class categorizes all TAL operators to enable proper parsing, precedence
    handling, and semantic analysis of expressions.
    """
    
    ARITHMETIC_OPERATORS = {'+', '-', '*', '/', 'MOD', '**'}
    COMPARISON_OPERATORS = {'=', '<>', '<', '>', '<=', '>=', 'LT', 'LE', 'GT', 'GE', 'EQ', 'NE'}
    LOGICAL_OPERATORS = {'AND', 'OR', 'NOT', 'XOR'}
    ASSIGNMENT_OPERATORS = {':=', "':='"}
    ADDRESSING_OPERATORS = {'@', '.', '->', '[', ']'}
    
    @classmethod
    def is_operator(cls, token: str) -> bool:
        """
        Determine if a token represents a TAL operator.
        
        Args:
            token: Token to check
            
        Returns:
            True if token is a recognized operator
        """
        token_upper = token.upper()
        return (token in cls.ARITHMETIC_OPERATORS or 
                token in cls.COMPARISON_OPERATORS or
                token_upper in cls.LOGICAL_OPERATORS or
                token in cls.ASSIGNMENT_OPERATORS or
                token in cls.ADDRESSING_OPERATORS)
    
    @classmethod
    def get_operator_type(cls, token: str) -> str:
        """
        Classify an operator by its functional category.
        
        Args:
            token: Operator token to classify
            
        Returns:
            String describing the operator category
        """
        token_upper = token.upper()
        if token in cls.ARITHMETIC_OPERATORS:
            return 'arithmetic'
        elif token in cls.COMPARISON_OPERATORS or token_upper in cls.COMPARISON_OPERATORS:
            return 'comparison'
        elif token_upper in cls.LOGICAL_OPERATORS:
            return 'logical'
        elif token in cls.ASSIGNMENT_OPERATORS:
            return 'assignment'
        elif token in cls.ADDRESSING_OPERATORS:
            return 'addressing'
        return 'unknown'

class EnhancedTALParser:
    """
    Enhanced TAL parser implementing a hybrid two-stage parsing approach.
    
    This parser combines the reliable procedure detection from tal_proc_parser
    with comprehensive rich parsing for detailed AST generation. It handles
    all major TAL language constructs including procedures, system functions,
    control flow, and TAL-specific features like bit fields.
    
    Architecture:
    1. Foundation parsing using tal_proc_parser for reliable structure
    2. Enhanced parsing for detailed AST nodes and metadata
    3. Comprehensive validation and error reporting
    4. Rich output generation (S-expressions, JSON, analysis)
    """
    
    def __init__(self):
        """
        Initialize the enhanced TAL parser with all necessary components.
        
        Sets up error tracking, symbol tables, and parsing utilities.
        """
        self.debug_mode = False
        self.filename = ""
        self.source_lines = []
        self.symbol_table = tal_proc_parser.SymbolTable()
        self.errors = []
        self.warnings = []
        self.system_functions = TALSystemFunctions()
        self.operators = TALOperators()
        
    
    def parse_file(self, filename: str) -> Dict[str, Any]:
        """
        Main entry point for parsing a TAL file.
        
        This method orchestrates the entire parsing process:
        1. Reads the source file
        2. Applies the hybrid parsing approach
        3. Generates comprehensive results
        
        Args:
            filename: Path to the TAL source file
            
        Returns:
            Dictionary containing parse results, AST, and analysis data
        """
        self.filename = filename
        
        try:
            # Read and preprocess the source file
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
            
            self.source_lines = content.split('\n')
            
            # Apply the enhanced hybrid parsing approach
            result = self._parse_using_tal_proc_parser_enhanced(content)
            return result
            
        except Exception as e:
            return {
                'success': False, 
                'error': f"File error: {e}",
                'errors': [str(err) for err in self.errors],
                'warnings': [str(warn) for warn in self.warnings]
            }
    
    def _parse_using_tal_proc_parser_enhanced(self, content: str) -> Dict[str, Any]:
        """
        Implement the hybrid parsing approach combining tal_proc_parser with rich enhancement.
        
        This is the core of the enhanced parsing strategy:
        1. Use tal_proc_parser for reliable procedure detection
        2. Calculate precise procedure line ranges
        3. Apply rich parsing to procedure bodies
        4. Parse global content with comprehensive analysis
        5. Assemble complete program AST
        
        Args:
            content: Complete source file content
            
        Returns:
            Comprehensive parsing results with enhanced AST
        """
        lines = content.split('\n')
        
        # Step 1: Foundation parsing - use tal_proc_parser for reliable procedure detection
        procedures, proc_errors = tal_proc_parser.parse_multiple_procedures(
            content, self.filename, self.symbol_table
        )
        self.errors.extend(proc_errors)
        
        # Step 2: Calculate precise procedure line ranges for body parsing
        proc_declarations = tal_proc_parser.find_procedure_declarations(content)
        proc_line_ranges = self._calculate_simple_line_ranges(proc_declarations, lines)
        
        # Step 3: Enhancement phase - apply rich parsing to each procedure
        enhanced_procedures = []
        all_proc_lines = set()
        
        for proc_node in procedures:
            # Find the line range for this procedure
            line_range = proc_line_ranges.get(proc_node.name)
            if line_range:
                # Track all lines that belong to procedures
                all_proc_lines.update(range(line_range['start'], line_range['end'] + 1))
                
                # Apply rich parsing enhancement to procedure body
                enhanced_proc = self._enhance_procedure_with_rich_parsing(proc_node, lines, line_range)
                enhanced_procedures.append(enhanced_proc)
            else:
                # Keep procedure as-is if no range found
                enhanced_procedures.append(proc_node)
        
        # Step 4: Create comprehensive program AST
        program = tal_proc_parser.TALNode('program')
        program.location = tal_proc_parser.SourceLocation(self.filename, 1, 1)
        
        # Step 5: Add global content with rich parsing (everything not in procedures)
        self._add_global_content_rich(lines, program, all_proc_lines)
        
        # Step 6: Integrate enhanced procedures into program AST
        for proc in enhanced_procedures:
            program.add_child(proc)
        
        # Step 7: Generate comprehensive results
        return {
            'success': len([e for e in self.errors if e.severity == tal_proc_parser.ErrorSeverity.FATAL]) == 0,
            'ast': program,
            'sexp': program.to_sexp(),
            'json': self._ast_to_json(program),
            'node_count': self._count_nodes(program),
            'structure': self._analyze_structure(program),
            'system_functions_used': self._collect_system_functions(program),
            'procedures': self._extract_procedure_info(enhanced_procedures)
        }
    

    def _has_bit_field_syntax(self, expr: str) -> bool:
        """
        Detect TAL bit field syntax in expressions.
        
        TAL supports bit field access using syntax like <start:end> to extract
        specific bit ranges from variables.
        
        Args:
            expr: Expression string to check
            
        Returns:
            True if expression contains bit field syntax
        """
        import re
        return re.search(r'<\d+:\d+>', expr) is not None

    def _parse_bit_field_expression(self, expr: str) -> Optional[Dict[str, Any]]:
        """
        Parse TAL bit field expressions into structured components.
        
        Extracts the base expression and bit range from TAL bit field syntax
        like variable.field.<0:7> into analyzable components.
        
        Args:
            expr: Expression containing bit field syntax
            
        Returns:
            Dictionary with parsed bit field components or None
        """
        import re
        bit_field_match = re.search(r'<(\d+):(\d+)>', expr)
        if bit_field_match:
            start_bit = int(bit_field_match.group(1))
            end_bit = int(bit_field_match.group(2))
            base_expr = expr[:bit_field_match.start()]
            
            return {
                'base_expression': base_expr.strip(),
                'start_bit': start_bit,
                'end_bit': end_bit,
                'bit_range': f"{start_bit}:{end_bit}"
            }
        return None

    def _calculate_simple_line_ranges(self, proc_declarations: List[Tuple[int, str, str]], lines: List[str]) -> Dict[str, Dict[str, int]]:
        """
        Calculate accurate line ranges for procedures, handling multi-line declarations.
        
        This function determines exactly where each procedure starts and ends,
        accounting for multi-line procedure declarations that may span several lines.
        Critical for avoiding duplicate parsing of procedure content.
        
        Args:
            proc_declarations: List of (line_num, proc_name, declaration_text) tuples
            lines: All source code lines
            
        Returns:
            Dictionary mapping procedure names to their line ranges
        """
        ranges = {}
        
        for i, (start_line, proc_name, declaration) in enumerate(proc_declarations):
            # Calculate how many lines the declaration actually spans
            declaration_line_count = declaration.count('\n') + 1
            
            # Find the actual end of the declaration
            declaration_end_line = start_line + declaration_line_count - 1
            
            # Determine where this procedure ends (start of next procedure or end of file)
            if i + 1 < len(proc_declarations):
                end_line = proc_declarations[i + 1][0] - 1
            else:
                end_line = len(lines)
            
            ranges[proc_name] = {
                'start': start_line,
                'declaration_end': declaration_end_line,  # Track declaration boundary
                'end': end_line
            }
        
        return ranges
    
    def _enhance_procedure_with_rich_parsing(self, proc_node: tal_proc_parser.TALNode, lines: List[str], line_range: Dict[str, int]) -> tal_proc_parser.TALNode:
        """
        Apply rich parsing enhancement to a procedure's body content.
        
        This method takes a basic procedure node from tal_proc_parser and enhances
        it with detailed parsing of the procedure body, including:
        - Local variable declarations
        - Statement parsing with system function detection
        - Control flow analysis
        - Expression parsing
        
        Args:
            proc_node: Basic procedure node from tal_proc_parser
            lines: All source code lines
            line_range: Dictionary with start/end line numbers for this procedure
            
        Returns:
            Enhanced procedure node with rich AST content
        """
        
        # Skip enhancement for FORWARD/EXTERNAL procedures (no body to parse)
        if proc_node.attributes.get('is_forward') or proc_node.attributes.get('is_external'):
            return proc_node
        
        # Locate existing structure nodes within the procedure
        local_decls_node = None
        statements_node = None
        
        for child in proc_node.children:
            if child.type == 'local_declarations':
                local_decls_node = child
            elif child.type == 'statements':
                statements_node = child
        
        # Clear existing content to avoid duplicates during enhancement
        if local_decls_node:
            local_decls_node.children = []
        if statements_node:
            statements_node.children = []
        
        # CRITICAL: Start parsing AFTER the procedure declaration ends
        # This prevents re-parsing the procedure declaration as body content
        declaration_end_line = line_range.get('declaration_end', line_range['start'])
        
        # Extract procedure body lines (after declaration, before next procedure)
        proc_lines = lines[declaration_end_line:line_range['end']]
        start_line_for_parsing = declaration_end_line + 1
        
        if self.debug_mode: 
            print(f"DEBUG: Procedure body parsing from line {start_line_for_parsing} to {line_range['end']}")
            print(f"DEBUG: Declaration ended at line {declaration_end_line}")
        
        # Apply rich parsing to the procedure body
        self._parse_procedure_body_rich(proc_lines, local_decls_node, statements_node, start_line_for_parsing)
        
        return proc_node

    
    def _parse_procedure_body_rich_old(self, proc_lines: List[str], local_decls_node: tal_proc_parser.TALNode, statements_node: tal_proc_parser.TALNode, start_line_num: int):
        """
        Parse procedure body lines with comprehensive AST generation.
        
        This method processes each line of a procedure body, determining whether
        it represents a local declaration or an executable statement, and creates
        appropriate rich AST nodes with detailed metadata.
        
        Args:
            proc_lines: Lines of code within the procedure body
            local_decls_node: AST node for local declarations
            statements_node: AST node for executable statements
            start_line_num: Starting line number for location tracking
        """
        in_statements = False
        found_begin = False
        
        for line_idx, line_text in enumerate(proc_lines):
            line_stripped = line_text.strip()
            
            # Skip empty lines
            if not line_stripped:
                continue
            
            # Handle comment lines (starting with !)
            if line_stripped.startswith('!'):
                location = tal_proc_parser.SourceLocation(self.filename, start_line_num + line_idx, 1)
                comment_node = tal_proc_parser.TALNode('comment', value=line_stripped[1:].strip(), location=location)
                if in_statements:
                    statements_node.add_child(comment_node)
                else:
                    local_decls_node.add_child(comment_node)
                continue
            
            # Extract and handle inline comments
            comment_pos = line_stripped.find('!')
            if comment_pos >= 0:
                code_part = line_stripped[:comment_pos].strip()
                # Create separate node for inline comment
                inline_comment = line_stripped[comment_pos+1:].strip()
                if inline_comment:
                    location = tal_proc_parser.SourceLocation(self.filename, start_line_num + line_idx, comment_pos + 1)
                    comment_node = tal_proc_parser.TALNode('comment', value=inline_comment, location=location)
                    if in_statements:
                        statements_node.add_child(comment_node)
                    else:
                        local_decls_node.add_child(comment_node)
            else:
                code_part = line_stripped
            
            # Skip lines with no executable content
            if not code_part:
                continue
            
            location = tal_proc_parser.SourceLocation(self.filename, start_line_num + line_idx, 1)
            
            # Skip the PROC declaration line (should already be handled)
            if re.search(r'\bPROC\b', code_part, re.IGNORECASE):
                continue
            
            # Detect BEGIN - switches from declarations to statements
            if re.search(r'\bBEGIN\b', code_part, re.IGNORECASE):
                found_begin = True
                in_statements = True
                continue
            
            # Detect END - stops parsing procedure body
            if re.search(r'\bEND\b', code_part, re.IGNORECASE) and found_begin:
                break
            
            # Parse the line using comprehensive rich parsing methods
            try:
                node = self._parse_body_line_comprehensive(code_part, location, in_statements)
                if node:
                    # Determine whether this belongs in declarations or statements
                    if in_statements or self._is_statement_node(node):
                        statements_node.add_child(node)
                    else:
                        local_decls_node.add_child(node)
            except Exception as e:
                # Log parsing errors but continue processing
                self.errors.append(tal_proc_parser.ParseError(
                    f"Error parsing line: {e}",
                    location,
                    tal_proc_parser.ErrorSeverity.WARNING,
                    error_code="E100"
                ))

    def _parse_procedure_body_rich(self, proc_lines: List[str], local_decls_node: tal_proc_parser.TALNode, statements_node: tal_proc_parser.TALNode, start_line_num: int):
        """
        Parse procedure body lines with comprehensive AST generation.
        Now handles multi-line statements like IF conditions.
        """
        in_statements = False
        found_begin = False
        
        i = 0
        while i < len(proc_lines):
            line_text = proc_lines[i]
            line_stripped = line_text.strip()
            
            # Skip empty lines
            if not line_stripped:
                i += 1
                continue
            
            # Handle comment lines (starting with !)
            if line_stripped.startswith('!'):
                location = tal_proc_parser.SourceLocation(self.filename, start_line_num + i, 1)
                comment_node = tal_proc_parser.TALNode('comment', value=line_stripped[1:].strip(), location=location)
                if in_statements:
                    statements_node.add_child(comment_node)
                else:
                    local_decls_node.add_child(comment_node)
                i += 1
                continue
            
            # Extract and handle inline comments
            comment_pos = line_stripped.find('!')
            if comment_pos >= 0:
                code_part = line_stripped[:comment_pos].strip()
                # Create separate node for inline comment
                inline_comment = line_stripped[comment_pos+1:].strip()
                if inline_comment:
                    location = tal_proc_parser.SourceLocation(self.filename, start_line_num + i, comment_pos + 1)
                    comment_node = tal_proc_parser.TALNode('comment', value=inline_comment, location=location)
                    if in_statements:
                        statements_node.add_child(comment_node)
                    else:
                        local_decls_node.add_child(comment_node)
            else:
                code_part = line_stripped
            
            # Skip lines with no executable content
            if not code_part:
                i += 1
                continue
            
            location = tal_proc_parser.SourceLocation(self.filename, start_line_num + i, 1)
            
            # Skip the PROC declaration line (should already be handled)
            if re.search(r'\bPROC\b', code_part, re.IGNORECASE):
                i += 1
                continue
            
            # Detect BEGIN - switches from declarations to statements
            if re.search(r'\bBEGIN\b', code_part, re.IGNORECASE):
                found_begin = True
                in_statements = True
                i += 1
                continue
            
            # Detect END - stops parsing procedure body
            if re.search(r'\bEND\b', code_part, re.IGNORECASE) and found_begin:
                break
            
            # Check for multi-line statements
            if self._is_multiline_statement_start(code_part):
                complete_statement, lines_consumed = self._parse_multiline_statement(proc_lines, i)
                location = tal_proc_parser.SourceLocation(self.filename, start_line_num + i, 1)
                
                try:
                    node = self._parse_body_line_comprehensive(complete_statement, location, in_statements)
                    if node:
                        if in_statements or self._is_statement_node(node):
                            statements_node.add_child(node)
                        else:
                            local_decls_node.add_child(node)
                except Exception as e:
                    self.errors.append(tal_proc_parser.ParseError(
                        f"Error parsing multi-line statement: {e}",
                        location,
                        tal_proc_parser.ErrorSeverity.WARNING,
                        error_code="E100"
                    ))
                i += lines_consumed
                continue
            
            # Parse single-line constructs
            try:
                node = self._parse_body_line_comprehensive(code_part, location, in_statements)
                if node:
                    if in_statements or self._is_statement_node(node):
                        statements_node.add_child(node)
                    else:
                        local_decls_node.add_child(node)
            except Exception as e:
                self.errors.append(tal_proc_parser.ParseError(
                    f"Error parsing line: {e}",
                    location,
                    tal_proc_parser.ErrorSeverity.WARNING,
                    error_code="E100"
                ))
            
            i += 1

    def _is_multiline_statement_start(self, line: str) -> bool:
        """
        Check if a line starts a statement that might span multiple lines.
        """
        upper_line = line.upper().strip()
        
        # Statements that commonly span multiple lines
        multiline_starters = ['IF ', 'WHILE ', 'FOR ', 'CASE ', 'SCAN ', 'RSCAN ']
        
        if any(upper_line.startswith(starter) for starter in multiline_starters):
            return True
        
        # Check for unbalanced parentheses indicating continuation
        if self._has_unbalanced_parentheses(line):
            return True
        
        return False

    def _parse_multiline_statement(self, proc_lines: List[str], start_index: int) -> Tuple[str, int]:
        """
        Parse statements that span multiple lines by checking for balanced parentheses
        and statement completion.
        """
        if start_index >= len(proc_lines):
            return "", 1
        
        current_statement = proc_lines[start_index].strip()
        lines_consumed = 1
        
        # Remove any inline comments for parsing logic
        if '!' in current_statement:
            current_statement = current_statement[:current_statement.find('!')].strip()
        
        # Continue until we have balanced parentheses and logical completion
        while start_index + lines_consumed < len(proc_lines):
            next_line = proc_lines[start_index + lines_consumed].strip()
            
            # Skip empty lines
            if not next_line:
                lines_consumed += 1
                continue
            
            # Remove inline comments
            if '!' in next_line:
                next_line = next_line[:next_line.find('!')].strip()
            
            if not next_line:
                lines_consumed += 1
                continue
            
            # Stop if we hit a clear new statement
            if self._is_new_statement_start(next_line):
                break
            
            # Add the continuation line
            current_statement += ' ' + next_line
            lines_consumed += 1
            
            # Check if statement is now complete
            if self._is_statement_complete(current_statement):
                break
        
        return current_statement, lines_consumed

    def _has_unbalanced_parentheses(self, text: str) -> bool:
        """
        Check if text has unbalanced parentheses, indicating it continues on next line.
        Properly handles nested parentheses and quoted strings.
        """
        paren_count = 0
        in_quotes = False
        quote_char = None
        
        for i, char in enumerate(text):
            # Handle quotes
            if char in '"\'':
                if not in_quotes:
                    in_quotes = True
                    quote_char = char
                elif char == quote_char and (i == 0 or text[i-1] != '\\'):
                    in_quotes = False
                    quote_char = None
            
            # Count parentheses only when not in quotes
            elif not in_quotes:
                if char == '(':
                    paren_count += 1
                elif char == ')':
                    paren_count -= 1
        
        return paren_count != 0

    def _is_statement_complete(self, statement: str) -> bool:
        """
        Check if a statement appears to be complete based on TAL syntax.
        """
        upper_stmt = statement.upper().strip()
        
        # IF statements need THEN
        if upper_stmt.startswith('IF '):
            return ' THEN' in upper_stmt and self._has_balanced_parentheses(statement)
        
        # WHILE statements need DO
        elif upper_stmt.startswith('WHILE '):
            return ' DO' in upper_stmt and self._has_balanced_parentheses(statement)
        
        # FOR statements need DO
        elif upper_stmt.startswith('FOR '):
            return ' DO' in upper_stmt and self._has_balanced_parentheses(statement)
        
        # CASE statements need OF
        elif upper_stmt.startswith('CASE '):
            return ' OF' in upper_stmt and self._has_balanced_parentheses(statement)
        
        # General check for balanced parentheses
        else:
            return self._has_balanced_parentheses(statement)

    def _has_balanced_parentheses(self, text: str) -> bool:
        """Check if all parentheses are balanced."""
        return not self._has_unbalanced_parentheses(text)

    def _is_new_statement_start(self, line: str) -> bool:
        """
        Check if a line starts a new statement rather than continuing the previous one.
        """
        upper_line = line.upper().strip()
        
        # Clear statement starters
        statement_starters = [
            'IF ', 'WHILE ', 'FOR ', 'CASE ', 'SCAN ', 'RSCAN ', 'CALL ', 'RETURN',
            'GOTO ', 'STOP', 'ABORT', 'ASSERT '
        ]
        
        if any(upper_line.startswith(starter) for starter in statement_starters):
            return True
        
        # Assignment statements
        if ':=' in line:
            return True
        
        # Label definitions
        if line.strip().endswith(':') and ':=' not in line:
            return True
        
        # Variable declarations
        type_keywords = ['INT ', 'STRING ', 'REAL ', 'FIXED ', 'BYTE ', 'CHAR ', 'UNSIGNED ']
        if any(upper_line.startswith(kw) for kw in type_keywords):
            return True
        
        return False
    
    def _parse_body_line_comprehensive(self, line: str, location: tal_proc_parser.SourceLocation, in_statements: bool) -> Optional[tal_proc_parser.TALNode]:
        """
        Parse individual lines with comprehensive pattern recognition.
        
        This method applies pattern matching to identify and parse different
        types of TAL constructs, creating appropriate AST nodes with rich
        metadata for each construct type.
        
        Args:
            line: Source code line to parse
            location: Source location for error reporting
            in_statements: Whether we're in the statements section (after BEGIN)
            
        Returns:
            Parsed AST node or None if line should be skipped
        """
        line = line.strip()
        if not line:
            return None
        
        upper_line = line.upper()
        
        # Variable declarations (before BEGIN or explicitly typed)
        if not in_statements and self._is_variable_declaration(line):
            return self._parse_variable_declaration_comprehensive(line, location)
        
        # Assignment statements (with := or ':=' operators)
        elif ':=' in line or "':='" in line:
            return self._parse_assignment_comprehensive(line, location)
        
        # Control flow statements
        elif upper_line.startswith('IF '):
            return self._parse_if_statement_comprehensive(line, location)
        elif upper_line.startswith('WHILE '):
            return self._parse_while_statement_comprehensive(line, location)
        elif upper_line.startswith('FOR '):
            return self._parse_for_statement_comprehensive(line, location)
        elif upper_line.startswith('CASE '):
            return self._parse_case_statement_comprehensive(line, location)
        elif upper_line.startswith('SCAN '):
            return self._parse_scan_statement_comprehensive(line, location)
        elif upper_line.startswith('RSCAN '):
            return self._parse_rscan_statement_comprehensive(line, location)
        elif upper_line.startswith('GOTO '):
            return self._parse_goto_statement_comprehensive(line, location)
        
        # Procedure calls and returns
        elif upper_line.startswith('CALL ') or line.startswith('$'):
            return self._parse_call_statement_comprehensive(line, location)
        elif upper_line.startswith('RETURN'):
            return self._parse_return_statement_comprehensive(line, location)
        
        # TAL-specific statements
        elif upper_line.startswith('INTERRUPT '):
            return self._parse_interrupt_statement_comprehensive(line, location)
        elif upper_line.startswith('ASSERT '):
            return self._parse_assert_statement_comprehensive(line, location)
        elif upper_line.startswith('STOP'):
            return self._parse_stop_statement_comprehensive(line, location)
        elif upper_line.startswith('ABORT'):
            return self._parse_abort_statement_comprehensive(line, location)
        
        # Label definitions
        elif self._is_label(line):
            return self._parse_label_comprehensive(line, location)
        
        # General statements (fallback with rich analysis)
        else:
            return self._parse_general_statement_comprehensive(line, location)
    
    def _is_variable_declaration(self, line: str) -> bool:
        """
        Determine if a line represents a variable declaration.
        
        Checks for TAL data type keywords and declaration patterns to identify
        variable declarations vs. other statements.
        
        Args:
            line: Source code line to analyze
            
        Returns:
            True if line appears to be a variable declaration
        """
        upper_line = line.upper()
        tal_types = ['INT', 'STRING', 'REAL', 'FIXED', 'BYTE', 'CHAR', 'UNSIGNED', 'STRUCT']
        
        # Direct type keyword match
        if any(upper_line.startswith(t) for t in tal_types):
            return True
        
        # Heuristic: identifier followed by type (but not control flow)
        parts = line.split()
        if (len(parts) >= 2 and 
            not any(upper_line.startswith(kw) for kw in ['IF ', 'WHILE ', 'CASE ', 'CALL ', 'RETURN ']) and
            not ':=' in line):
            return True
        
        return False
    
    def _is_statement_node(self, node: tal_proc_parser.TALNode) -> bool:
        """
        Classify whether an AST node represents an executable statement.
        
        This helps determine proper placement within the procedure AST structure.
        
        Args:
            node: AST node to classify
            
        Returns:
            True if node represents an executable statement
        """
        statement_types = ['assignment', 'call_stmt', 'return_stmt', 'if_stmt', 'while_stmt', 'for_stmt', 'case_stmt', 'scan_stmt', 'rscan_stmt', 'goto_stmt', 'statement', 'interrupt_stmt', 'assert_stmt', 'stop_stmt', 'abort_stmt', 'label']
        return node.type in statement_types
    
    def _is_label(self, line: str) -> bool:
        """
        Identify TAL label definitions.
        
        Labels in TAL end with a colon and mark jump targets for GOTO statements.
        
        Args:
            line: Source code line to check
            
        Returns:
            True if line appears to be a label definition
        """
        stripped = line.strip()
        if stripped.endswith(':') and not any(op in stripped for op in [':=', "':='"]):
            # Ensure it's not a case label or other construct
            label_name = stripped[:-1].strip()
            # Valid label names are typically identifiers
            if re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', label_name):
                return True
        return False
    
    def _parse_variable_declaration_comprehensive(self, line: str, location: tal_proc_parser.SourceLocation) -> tal_proc_parser.TALNode:
        """
        Parse variable declarations with comprehensive type and attribute analysis.
        
        Handles various TAL variable declaration formats including arrays,
        pointers, initializers, and multiple variable declarations.
        
        Args:
            line: Variable declaration line
            location: Source location for error reporting
            
        Returns:
            AST node representing the variable declaration
        """
        var_node = tal_proc_parser.TALNode('var_decl', location=location, value=line.rstrip(';'))
        
        parts = line.split()
        if len(parts) >= 2:
            var_type = parts[0].upper()
            var_node.attributes['type'] = var_type
            
            # Parse variable specifications (may be multiple, comma-separated)
            remaining = ' '.join(parts[1:]).rstrip(';')
            var_specs = self._smart_split(remaining, ',')
            
            for var_spec in var_specs:
                var_spec = var_spec.strip()
                if var_spec:
                    spec_node = self._parse_variable_spec_comprehensive(var_spec, var_type, location)
                    if spec_node:
                        var_node.add_child(spec_node)
        
        return var_node
    
    def _parse_variable_spec_comprehensive(self, var_spec: str, var_type: str, location: tal_proc_parser.SourceLocation) -> tal_proc_parser.TALNode:
        """
        Parse individual variable specifications within declarations.
        
        Extracts variable name, array bounds, pointer indicators, and initializers
        from variable specification strings.
        
        Args:
            var_spec: Individual variable specification (e.g., "array[0:9]" or ".pointer")
            var_type: Data type of the variable
            location: Source location
            
        Returns:
            AST node for the variable specification
        """
        spec_node = tal_proc_parser.TALNode('var_spec', location=location)
        
        is_pointer = False
        is_array = False
        var_name = ""
        initializer = None
        
        # Handle initialization (var := value)
        if ':=' in var_spec:
            name_part, init_part = var_spec.split(':=', 1)
            name_part = name_part.strip()
            initializer = init_part.strip()
            spec_node.attributes['initializer'] = initializer
        else:
            name_part = var_spec.strip()
        
        # Handle array declarations (var[bounds])
        if '[' in name_part and ']' in name_part:
            is_array = True
            bracket_start = name_part.find('[')
            bracket_end = name_part.find(']')
            array_spec = name_part[bracket_start+1:bracket_end]
            var_name = name_part[:bracket_start]
            spec_node.attributes['array_bounds'] = array_spec
        else:
            var_name = name_part
        
        # Handle pointer declarations (.var)
        if var_name.startswith('.'):
            is_pointer = True
            var_name = var_name[1:]
        
        var_name = var_name.strip().rstrip(';').rstrip(',')
        
        spec_node.name = var_name
        spec_node.value = var_name
        
        if is_pointer:
            spec_node.attributes['pointer'] = True
        if is_array:
            spec_node.attributes['array'] = True
        
        return spec_node
    
    def _parse_assignment_comprehensive(self, line: str, location: tal_proc_parser.SourceLocation) -> tal_proc_parser.TALNode:
        """
        Parse assignment statements with operator and bit field detection.
        
        Handles both standard assignments (:=) and string assignments (':='),
        including TAL-specific bit field assignments.
        
        Args:
            line: Assignment statement line
            location: Source location
            
        Returns:
            AST node for the assignment
        """
        assign_node = tal_proc_parser.TALNode('assignment', location=location, value=line.rstrip(';'))
        
        # Determine assignment operator type
        if "':='" in line:
            target, value = line.split("':='", 1)
            assign_node.attributes['operator'] = "':='"
            assign_node.attributes['assignment_type'] = 'string_move'
        elif ':=' in line:
            target, value = line.split(':=', 1)
            assign_node.attributes['operator'] = ':='
            assign_node.attributes['assignment_type'] = 'standard'
        else:
            return assign_node
        
        target = target.strip()
        value = value.strip().rstrip(';')
        
        assign_node.name = target
        assign_node.attributes['target'] = target
        assign_node.attributes['value'] = value
        
        # Check for TAL bit field syntax in assignment target
        if self._has_bit_field_syntax(target):
            assign_node.attributes['bit_field_assignment'] = True
            bit_field_info = self._parse_bit_field_expression(target)
            if bit_field_info:
                assign_node.attributes['bit_field'] = bit_field_info
        
        # Parse the assignment value for system functions and operators
        value_expr = self._parse_expression_comprehensive(value, location)
        if value_expr:
            assign_node.add_child(value_expr)
        
        return assign_node
    
    def _parse_expression_comprehensive(self, expr: str, location: tal_proc_parser.SourceLocation) -> Optional[tal_proc_parser.TALNode]:
        """
        Parse expressions with system function and operator detection.
        
        Analyzes expressions to identify and catalog system functions, operators,
        and other language constructs for comprehensive AST representation.
        
        Args:
            expr: Expression string to parse
            location: Source location
            
        Returns:
            AST node representing the expression
        """
        expr = expr.strip()
        if not expr:
            return None
        
        expr_node = tal_proc_parser.TALNode('expression', value=expr, location=location)
        
        # Extract and create nodes for system functions
        system_funcs = self._extract_system_functions_from_expr(expr)
        for func_name, func_info in system_funcs:
            func_node = tal_proc_parser.TALNode('system_function', name=func_name, location=location)
            func_node.attributes.update(func_info)
            expr_node.add_child(func_node)
        
        # Extract and create nodes for operators
        operators = self._extract_operators_from_expr(expr)
        for op_name, op_type in operators:
            op_node = tal_proc_parser.TALNode('operator', name=op_name, location=location)
            op_node.attributes['operator_type'] = op_type
            expr_node.add_child(op_node)
        
        return expr_node
    
    def _extract_system_functions_from_expr(self, expr: str) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Extract system function calls from expression strings.
        
        Uses pattern matching to identify system function calls (starting with $)
        and retrieves their metadata from the system function registry.
        
        Args:
            expr: Expression string to analyze
            
        Returns:
            List of (function_name, function_info) tuples
        """
        functions = []
        
        # Pattern to match system functions: $FUNCTION_NAME
        pattern = r'\$[A-Z_][A-Z0-9_]*'
        matches = re.findall(pattern, expr.upper())
        
        for match in matches:
            func_info = self.system_functions.get_function_info(match)
            if func_info:
                functions.append((match, func_info))
            else:
                # Record unknown system functions for analysis
                functions.append((match, {
                    'params': -1, 
                    'returns': 'UNKNOWN', 
                    'description': 'Unknown system function',
                    'unknown': True
                }))
        
        return functions
    
    def _extract_operators_from_expr(self, expr: str) -> List[Tuple[str, str]]:
        """
        Extract operators from expression strings.
        
        Tokenizes expressions and identifies TAL operators, classifying them
        by type for semantic analysis.
        
        Args:
            expr: Expression string to analyze
            
        Returns:
            List of (operator, operator_type) tuples
        """
        operators = []
        
        # Simple tokenization to identify operators
        tokens = re.findall(r'\w+|[^\w\s]', expr.upper())
        
        for token in tokens:
            if self.operators.is_operator(token):
                op_type = self.operators.get_operator_type(token)
                operators.append((token, op_type))
        
        return operators
    
    def _parse_if_statement_comprehensive(self, line: str, location: tal_proc_parser.SourceLocation) -> tal_proc_parser.TALNode:
        """
        Parse IF statements with condition analysis.
        
        Extracts the conditional expression and parses it for system functions
        and operators.
        
        Args:
            line: IF statement line
            location: Source location
            
        Returns:
            AST node for the IF statement
        """
        if_node = tal_proc_parser.TALNode('if_stmt', location=location, value=line.rstrip(';'))
        
        if ' THEN' in line.upper():
            condition = line.split(' THEN')[0][2:].strip()  # Remove 'IF'
            if_node.attributes['condition'] = condition
            
            # Parse condition for system functions and operators
            condition_expr = self._parse_expression_comprehensive(condition, location)
            if condition_expr:
                if_node.add_child(condition_expr)
        
        return if_node
    
    def _parse_while_statement_comprehensive(self, line: str, location: tal_proc_parser.SourceLocation) -> tal_proc_parser.TALNode:
        """
        Parse WHILE statements with loop condition analysis.
        
        Args:
            line: WHILE statement line
            location: Source location
            
        Returns:
            AST node for the WHILE statement
        """
        while_node = tal_proc_parser.TALNode('while_stmt', location=location, value=line.rstrip(';'))
        
        if ' DO' in line.upper():
            condition = line.split(' DO')[0][5:].strip()  # Remove 'WHILE'
            while_node.attributes['condition'] = condition
            
            # Parse condition for system functions and operators
            condition_expr = self._parse_expression_comprehensive(condition, location)
            if condition_expr:
                while_node.add_child(condition_expr)
        
        return while_node
    
    def _parse_for_statement_comprehensive(self, line: str, location: tal_proc_parser.SourceLocation) -> tal_proc_parser.TALNode:
        """
        Parse FOR loop statements.
        
        Args:
            line: FOR statement line
            location: Source location
            
        Returns:
            AST node for the FOR statement
        """
        for_node = tal_proc_parser.TALNode('for_stmt', location=location, value=line.rstrip(';'))
        
        # Extract FOR loop specification
        for_content = line[3:].strip()  # Remove 'FOR'
        if ' DO' in for_content.upper():
            loop_spec, rest = for_content.split(' DO', 1)
            for_node.attributes['loop_spec'] = loop_spec.strip()
        
        return for_node
    
    def _parse_case_statement_comprehensive(self, line: str, location: tal_proc_parser.SourceLocation) -> tal_proc_parser.TALNode:
        """
        Parse CASE statements with expression analysis.
        
        Args:
            line: CASE statement line
            location: Source location
            
        Returns:
            AST node for the CASE statement
        """
        case_node = tal_proc_parser.TALNode('case_stmt', location=location, value=line.rstrip(';'))
        
        if ' OF' in line.upper():
            expression = line.split(' OF')[0][4:].strip()  # Remove 'CASE'
            case_node.attributes['expression'] = expression
            
            # Parse expression for system functions
            expr_node = self._parse_expression_comprehensive(expression, location)
            if expr_node:
                case_node.add_child(expr_node)
        
        return case_node
    
    def _parse_scan_statement_comprehensive(self, line: str, location: tal_proc_parser.SourceLocation) -> tal_proc_parser.TALNode:
        """
        Parse TAL SCAN statements (string scanning construct).
        
        SCAN is a TAL-specific construct for iterating through strings or arrays.
        
        Args:
            line: SCAN statement line
            location: Source location
            
        Returns:
            AST node for the SCAN statement
        """
        scan_node = tal_proc_parser.TALNode('scan_stmt', location=location, value=line.rstrip(';'))
        
        parts = line.split()
        if len(parts) >= 2:
            scan_node.attributes['variable'] = parts[1]
        
        if 'WHILE' in line.upper():
            while_idx = line.upper().find('WHILE')
            condition_part = line[while_idx + 5:].strip()
            if '->' in condition_part:
                condition, target = condition_part.split('->', 1)
                scan_node.attributes['condition'] = condition.strip()
                scan_node.attributes['target'] = target.strip()
        
        return scan_node
    
    def _parse_rscan_statement_comprehensive(self, line: str, location: tal_proc_parser.SourceLocation) -> tal_proc_parser.TALNode:
        """
        Parse TAL RSCAN statements (reverse string scanning).
        
        RSCAN is similar to SCAN but operates in reverse direction.
        
        Args:
            line: RSCAN statement line
            location: Source location
            
        Returns:
            AST node for the RSCAN statement
        """
        rscan_node = tal_proc_parser.TALNode('rscan_stmt', location=location, value=line.rstrip(';'))
        
        parts = line.split()
        if len(parts) >= 2:
            rscan_node.attributes['variable'] = parts[1]
        
        if 'WHILE' in line.upper():
            while_idx = line.upper().find('WHILE')
            condition_part = line[while_idx + 5:].strip()
            if '->' in condition_part:
                condition, target = condition_part.split('->', 1)
                rscan_node.attributes['condition'] = condition.strip()
                rscan_node.attributes['target'] = target.strip()
        
        return rscan_node
    
    def _parse_goto_statement_comprehensive(self, line: str, location: tal_proc_parser.SourceLocation) -> tal_proc_parser.TALNode:
        """
        Parse GOTO statements with target label extraction.
        
        Args:
            line: GOTO statement line
            location: Source location
            
        Returns:
            AST node for the GOTO statement
        """
        goto_node = tal_proc_parser.TALNode('goto_stmt', location=location, value=line.rstrip(';'))
        
        parts = line.split()
        if len(parts) >= 2:
            target = parts[1].rstrip(';')
            goto_node.attributes['target'] = target
        
        return goto_node
    
    def _parse_call_statement_comprehensive(self, line: str, location: tal_proc_parser.SourceLocation) -> tal_proc_parser.TALNode:
        """
        Parse procedure calls and system function calls with enhanced validation.
        
        This method handles both regular procedure calls (CALL proc) and system
        function calls (starting with $), providing argument validation for
        system functions.
        
        Args:
            line: Call statement line
            location: Source location
            
        Returns:
            AST node for the call statement
        """
        call_node = tal_proc_parser.TALNode('call_stmt', location=location, value=line.rstrip(';'))
        
        # Extract the function call expression
        if line.startswith('CALL '):
            call_expr = line[5:].strip()
        elif line.startswith('$'):
            call_expr = line
        else:
            call_expr = line
        
        func_name = ""
        if '(' in call_expr:
            func_name = call_expr.split('(')[0].strip()
            if func_name.startswith('$'):
                # System function call
                original_func_name = func_name
                func_name = func_name[1:]  # Remove $ for attribute storage
                call_node.attributes['function'] = func_name
                call_node.attributes['original_name'] = original_func_name
                
                # Validate against system function registry
                func_info = self.system_functions.get_function_info(original_func_name)
                if func_info:
                    call_node.attributes['system_function'] = True
                    call_node.attributes.update(func_info)
                    call_node.type = 'system_function_call'
                else:
                    call_node.attributes['system_function'] = True
                    call_node.attributes['unknown_system_function'] = True
                    self.warnings.append(tal_proc_parser.ParseError(
                        f"Unknown system function: {original_func_name}",
                        location,
                        tal_proc_parser.ErrorSeverity.WARNING,
                        error_code="W301"
                    ))
            else:
                # Regular procedure call
                call_node.attributes['function'] = func_name
            
            # Parse function arguments
            arg_part = call_expr[call_expr.find('(')+1:call_expr.rfind(')')]
            if arg_part:
                args = self._smart_split(arg_part, ',')
                call_node.attributes['arguments'] = [arg.strip() for arg in args]
                call_node.attributes['argument_count'] = len(args)
                
                # Validate argument count for known system functions
                if (call_node.attributes.get('system_function') and 
                    func_info and 
                    func_info.get('params', -1) >= 0 and
                    len(args) != func_info['params']):
                    self.warnings.append(tal_proc_parser.ParseError(
                        f"Function {original_func_name} expects {func_info['params']} arguments, got {len(args)}",
                        location,
                        tal_proc_parser.ErrorSeverity.WARNING,
                        error_code="W302"
                    ))
        else:
            # Function call without parentheses (parameter-less)
            if call_expr.startswith('$'):
                func_name = call_expr
                call_node.attributes['function'] = func_name[1:]
                call_node.attributes['original_name'] = func_name
                call_node.attributes['system_function'] = True
                
                func_info = self.system_functions.get_function_info(func_name)
                if func_info:
                    call_node.attributes.update(func_info)
                    call_node.type = 'system_function_call'
        
        return call_node
    
    def _parse_return_statement_comprehensive(self, line: str, location: tal_proc_parser.SourceLocation) -> tal_proc_parser.TALNode:
        """
        Parse RETURN statements with expression and bit field analysis.
        
        Args:
            line: RETURN statement line
            location: Source location
            
        Returns:
            AST node for the RETURN statement
        """
        return_node = tal_proc_parser.TALNode('return_stmt', location=location)
        return_expr = line[6:].rstrip(';') if line.upper().startswith('RETURN') else line
        return_node.value = return_expr
        
        if return_expr:
            return_node.attributes['expression'] = return_expr
            
            # Check for TAL bit field syntax in return expression
            if self._has_bit_field_syntax(return_expr):
                return_node.attributes['bit_field_return'] = True
                bit_field_info = self._parse_bit_field_expression(return_expr)
                if bit_field_info:
                    return_node.attributes['bit_field'] = bit_field_info
            
            # Parse return expression for system functions and operators
            expr_node = self._parse_expression_comprehensive(return_expr, location)
            if expr_node:
                return_node.add_child(expr_node)
        
        return return_node
    
    def _parse_interrupt_statement_comprehensive(self, line: str, location: tal_proc_parser.SourceLocation) -> tal_proc_parser.TALNode:
        """
        Parse TAL INTERRUPT statements.
        
        INTERRUPT is a TAL-specific statement for interrupt handling.
        
        Args:
            line: INTERRUPT statement line
            location: Source location
            
        Returns:
            AST node for the INTERRUPT statement
        """
        interrupt_node = tal_proc_parser.TALNode('interrupt_stmt', location=location, value=line.rstrip(';'))
        
        parts = line.split()
        if len(parts) >= 2:
            interrupt_node.attributes['interrupt_number'] = parts[1].rstrip(';')
        
        return interrupt_node
    
    def _parse_assert_statement_comprehensive(self, line: str, location: tal_proc_parser.SourceLocation) -> tal_proc_parser.TALNode:
        """
        Parse ASSERT statements with condition analysis.
        
        Args:
            line: ASSERT statement line
            location: Source location
            
        Returns:
            AST node for the ASSERT statement
        """
        assert_node = tal_proc_parser.TALNode('assert_stmt', location=location, value=line.rstrip(';'))
        
        condition = line[6:].strip().rstrip(';')  # Remove 'ASSERT'
        if condition:
            assert_node.attributes['condition'] = condition
            # Parse condition for system functions
            condition_expr = self._parse_expression_comprehensive(condition, location)
            if condition_expr:
                assert_node.add_child(condition_expr)
        
        return assert_node
    
    def _parse_stop_statement_comprehensive(self, line: str, location: tal_proc_parser.SourceLocation) -> tal_proc_parser.TALNode:
        """
        Parse STOP statements with optional stop codes.
        
        Args:
            line: STOP statement line
            location: Source location
            
        Returns:
            AST node for the STOP statement
        """
        stop_node = tal_proc_parser.TALNode('stop_stmt', location=location, value=line.rstrip(';'))
        
        # STOP might have an optional code
        if len(line.strip()) > 4:
            code = line[4:].strip().rstrip(';')
            if code:
                stop_node.attributes['stop_code'] = code
        
        return stop_node
    
    def _parse_abort_statement_comprehensive(self, line: str, location: tal_proc_parser.SourceLocation) -> tal_proc_parser.TALNode:
        """
        Parse ABORT statements with optional abort codes.
        
        Args:
            line: ABORT statement line
            location: Source location
            
        Returns:
            AST node for the ABORT statement
        """
        abort_node = tal_proc_parser.TALNode('abort_stmt', location=location, value=line.rstrip(';'))
        
        # ABORT might have an optional code
        if len(line.strip()) > 5:
            code = line[5:].strip().rstrip(';')
            if code:
                abort_node.attributes['abort_code'] = code
        
        return abort_node
    
    def _parse_label_comprehensive(self, line: str, location: tal_proc_parser.SourceLocation) -> tal_proc_parser.TALNode:
        """
        Parse label definitions for GOTO targets.
        
        Args:
            line: Label definition line
            location: Source location
            
        Returns:
            AST node for the label
        """
        label_name = line.strip().rstrip(':')
        label_node = tal_proc_parser.TALNode('label', name=label_name, location=location, value=label_name)
        label_node.attributes['label_name'] = label_name
        
        return label_node
    

    def _parse_general_statement_comprehensive(self, line: str, location: tal_proc_parser.SourceLocation) -> tal_proc_parser.TALNode:
        """
        Parse general statements with comprehensive analysis and classification.
        
        This is the fallback parser for statements that don't match specific
        patterns. It applies various heuristics to classify and analyze the
        statement content.
        
        Args:
            line: Statement line to parse
            location: Source location
            
        Returns:
            AST node for the general statement
        """
        stmt_node = tal_proc_parser.TALNode('statement', location=location, value=line.rstrip(';'))
        
        # Debug input processing if enabled
        if self.debug_mode:
            print(f"DEBUG INPUT: Raw line = '{line}'")
            print(f"DEBUG INPUT: After rstrip(';') = '{line.rstrip(';')}'")

        # Check if this should actually be parsed as a variable declaration
        if self._is_variable_declaration(line):
            if self.debug_mode:
                print(f"DEBUG: Line identified as variable declaration")
            return self._parse_variable_declaration_comprehensive(line, location)
        else:
            if self.debug_mode:
                print(f"DEBUG: Line processed as general statement")
        
        # Classify the statement type using various heuristics
        upper_line = line.upper().strip()
        
        # Extract system functions if present
        if '$' in line:
            system_funcs = self._extract_system_functions_from_expr(line)
            for func_name, func_info in system_funcs:
                func_node = tal_proc_parser.TALNode('system_function', name=func_name, location=location)
                func_node.attributes.update(func_info)
                stmt_node.add_child(func_node)
        
        # Extract operators if present
        operators = self._extract_operators_from_expr(line)
        for op_name, op_type in operators:
            op_node = tal_proc_parser.TALNode('operator', name=op_name, location=location)
            op_node.attributes['operator_type'] = op_type
            stmt_node.add_child(op_node)
        
        # Enhanced classification of potential statement types
        if any(keyword in upper_line for keyword in ['DROP', 'ADD', 'SUB', 'MOVE']):
            stmt_node.attributes['potential_type'] = 'tal_instruction'
        elif re.search(r'^[A-Z_][A-Z0-9_\^]*$', upper_line.rstrip(';')):
            stmt_node.attributes['potential_type'] = 'identifier_reference'
        elif any(char in line for char in '()[]'):
            stmt_node.attributes['potential_type'] = 'complex_expression'
        elif self._has_bit_field_syntax(line):
            stmt_node.attributes['potential_type'] = 'bit_field_operation'
            bit_field_info = self._parse_bit_field_expression(line)
            if bit_field_info:
                stmt_node.attributes['bit_field'] = bit_field_info
        elif '.' in line and not line.startswith('!'):
            stmt_node.attributes['potential_type'] = 'pointer_or_field_access'
        elif '@' in line:
            stmt_node.attributes['potential_type'] = 'address_operation'
        
        # Debug output after processing (if enabled)
        if self.debug_mode:
            print(f"DEBUG: Final node children count: {len(stmt_node.children)}")
            print(f"DEBUG: Final node attributes: {stmt_node.attributes}")
            
            # Show S-expression construction process
            attrs_str = str(stmt_node.attributes)
            print(f"DEBUG ATTRS: attrs_str = '{attrs_str}'")
            
            debug_result = f"(statement :value {stmt_node.value}"
            if stmt_node.attributes:
                debug_result += f" :attrs {attrs_str}"
            print(f"DEBUG RESULT: after attrs = '{debug_result}'")
            
            final_result = stmt_node.to_sexp()
            print(f"DEBUG FINAL: final result = '{final_result}'")
            print(f"DEBUG: Created statement node: {stmt_node.to_sexp()}")
            
            # Debug problematic patterns
            stripped_line = line.strip()
            if (stripped_line.endswith(');') and 
                not ' ' in stripped_line.replace(');', '').replace('^', '').replace('_', '') and
                len(stripped_line) > 3):
                print(f"DEBUG: Processing potential procedure continuation line: '{line}'")
                print(f"DEBUG: Node structure: type={stmt_node.type}, value='{stmt_node.value}'")
                print(f"DEBUG: Attributes: {stmt_node.attributes}")
                
                manual_sexp = f"(statement :value {stmt_node.value})"
                if stmt_node.attributes:
                    manual_sexp = f"(statement :value {stmt_node.value} :attrs {stmt_node.attributes})"
                print(f"DEBUG: Expected S-exp format: {manual_sexp}")
        
        return stmt_node

    
    def _smart_split(self, text: str, delimiter: str) -> List[str]:
        """
        Split text while respecting nested structures like parentheses and quotes.
        
        This utility function properly handles splitting complex expressions
        that contain nested parentheses, brackets, or quoted strings, ensuring
        that splits only occur at the appropriate delimiter positions.
        
        Args:
            text: Text to split
            delimiter: Character to split on
            
        Returns:
            List of split text parts
        """
        parts = []
        current = ""
        paren_level = 0
        bracket_level = 0
        in_quotes = False
        quote_char = None
        
        for char in text:
            # Handle quote state changes
            if char in '"\'':
                if not in_quotes:
                    in_quotes = True
                    quote_char = char
                elif char == quote_char:
                    in_quotes = False
                    quote_char = None
            
            # Track nesting levels only when not in quotes
            if not in_quotes:
                if char == '(':
                    paren_level += 1
                if char == ')':
                    paren_level -= 1
                if char == '[':
                    bracket_level += 1
                if char == ']':
                    bracket_level -= 1
                
                # Split only when not nested and not in quotes
                if (char == delimiter and 
                    paren_level == 0 and bracket_level == 0):
                    parts.append(current.strip())
                    current = ""
                    continue
            
            current += char
        
        if current.strip():
            parts.append(current.strip())
        
        return parts
    
    def _add_global_content_rich(self, lines: List[str], program: tal_proc_parser.TALNode, proc_lines: set):
        """
        Add global content (non-procedure code) with comprehensive parsing.
        
        This method processes all lines that are not part of procedure bodies,
        applying rich parsing to global declarations, directives, and other
        top-level constructs.
        
        Args:
            lines: All source code lines
            program: Program AST node to add content to
            proc_lines: Set of line numbers that belong to procedures
        """
        
        if self.debug_mode:
            print(f"DEBUG: Initial proc_lines = {sorted(proc_lines)}")
        
        # Expand proc_lines to include all procedure declaration lines
        proc_declarations = tal_proc_parser.find_procedure_declarations('\n'.join(lines))
        if self.debug_mode:
            print(f"DEBUG: Found {len(proc_declarations)} procedure declarations")
        
        for start_line, proc_name, declaration in proc_declarations:
            declaration_line_count = declaration.count('\n') + 1
            if self.debug_mode:
                print(f"DEBUG: Procedure {proc_name} starts at line {start_line}, spans {declaration_line_count} lines")
                print(f"DEBUG: Declaration:\n{declaration}")
            
            # Mark all declaration lines as procedure content
            for line_num in range(start_line, start_line + declaration_line_count):
                proc_lines.add(line_num)
                if self.debug_mode:
                    print(f"DEBUG: Added line {line_num} to proc_lines")
        
        if self.debug_mode:
            print(f"DEBUG: Final proc_lines = {sorted(proc_lines)}")
        
        # Process each line that's not part of a procedure
        for i, line in enumerate(lines, 1):
            if i in proc_lines:
                if self.debug_mode:
                    print(f"DEBUG: Skipping line {i} (in proc_lines): '{line.strip()}'")
                continue
    
            line_stripped = line.strip()
            if not line_stripped:
                continue
            
            location = tal_proc_parser.SourceLocation(self.filename, i, 1)
            
            # Handle global comments
            if line_stripped.startswith('!'):
                comment_node = tal_proc_parser.TALNode('comment', value=line_stripped[1:].strip(), location=location)
                program.add_child(comment_node)
                continue
            
            # Parse global lines with comprehensive analysis
            try:
                node = self._parse_global_line_comprehensive(line_stripped, location)
                if node:
                    program.add_child(node)
            except Exception as e:
                self.errors.append(tal_proc_parser.ParseError(
                    f"Error parsing global line: {e}",
                    location,
                    tal_proc_parser.ErrorSeverity.WARNING,
                    error_code="E200"
                ))
    
    def _parse_global_line_comprehensive(self, line: str, location: tal_proc_parser.SourceLocation) -> Optional[tal_proc_parser.TALNode]:
        """
        Parse global-level constructs with comprehensive pattern matching.
        
        This method handles all top-level TAL constructs including compiler
        directives, module declarations, data structures, and global variables.
        
        Args:
            line: Global source line to parse
            location: Source location for error reporting
            
        Returns:
            AST node for the global construct or None to skip
        """
        line = line.strip()
        if not line:
            return None
        
        upper_line = line.upper()
        
        # Handle procedure continuation lines that weren't properly filtered
        if (line.strip().endswith(');') and 
            not any(upper_line.startswith(keyword) for keyword in ['PROC ', 'INT PROC ', 'STRING PROC ', 'REAL PROC ']) and
            not upper_line.startswith('?') and
            '(' not in line and
            not any(upper_line.startswith(keyword) for keyword in ['CALL ', 'IF ', 'WHILE ', 'FOR ', 'CASE '])):
            if self.debug_mode: 
                print(f"DEBUG: Skipping procedure continuation line: '{line}'")
            return None
        
        # Handle parameter continuation lines (indented with commas)
        if (line.strip().endswith(',') and 
            not upper_line.startswith('?') and
            not ':=' in line and
            not any(upper_line.startswith(keyword) for keyword in ['STRING ', 'INT ', 'REAL ', 'STRUCT ', 'LITERAL ', 'CONST ']) and
            len(line) - len(line.lstrip()) > 10):
            print(f"DEBUG: Skipping parameter continuation line: '{line}'")
            return None
        
        # Handle isolated parameter names that end with );
        stripped = line.strip()
        if (stripped.endswith(');') and 
            not ' ' in stripped.replace(');', '').replace('^', '').replace('_', '') and
            not any(c in stripped for c in ['(', '[', ':', '=', '<', '>', '+', '-', '*', '/'])):
            print(f"DEBUG: Skipping isolated parameter ending: '{line}'")
            return None
        
        # Compiler directives
        if upper_line.startswith('?PAGE '):
            return self._parse_page_directive_comprehensive(line, location)
        elif upper_line.startswith('?SECTION '):
            return self._parse_section_directive_comprehensive(line, location)
        elif upper_line.startswith('?SOURCE '):
            return self._parse_source_directive_comprehensive(line, location)
        elif upper_line.startswith('?NOLIST'):
            return tal_proc_parser.TALNode('nolist_directive', value=line, location=location)
        elif upper_line.startswith('?LIST'):
            return tal_proc_parser.TALNode('list_directive', value=line, location=location)
        elif upper_line.startswith('?SYMBOLS'):
            return tal_proc_parser.TALNode('symbols_directive', value=line, location=location)
        elif upper_line.startswith('?NOSYMBOLS'):
            return tal_proc_parser.TALNode('nosymbols_directive', value=line, location=location)
        elif upper_line.startswith('?SAVE'):
            return tal_proc_parser.TALNode('save_directive', value=line, location=location)
        elif upper_line.startswith('?RESTORE'):
            return tal_proc_parser.TALNode('restore_directive', value=line, location=location)
        elif upper_line.startswith('?HEAP'):
            return tal_proc_parser.TALNode('heap_directive', value=line, location=location)
        elif upper_line.startswith('?STACK'):
            return tal_proc_parser.TALNode('stack_directive', value=line, location=location)
        
        # Module declarations
        elif upper_line.startswith('NAME '):
            name = line.split()[1].rstrip(';')
            node = tal_proc_parser.TALNode('name_decl', name=name, location=location)
            node.attributes['module_name'] = name
            return node
            
        # Structure definitions
        elif upper_line.startswith('STRUCT '):
            return self._parse_struct_declaration_comprehensive(line, location)
            
        # Template definitions
        elif upper_line.startswith('TEMPLATE '):
            return self._parse_template_declaration_comprehensive(line, location)
            
        # Subtype definitions
        elif upper_line.startswith('SUBTYPE '):
            return self._parse_subtype_declaration_comprehensive(line, location)
            
        # External declarations
        elif upper_line.startswith('EXTERNAL '):
            return self._parse_external_declaration_comprehensive(line, location)
            
        # Forward declarations
        elif upper_line.startswith('FORWARD '):
            return self._parse_forward_declaration_comprehensive(line, location)
            
        # Constants and literals
        elif upper_line.startswith('LITERAL '):
            return self._parse_literal_declaration_comprehensive(line, location)
        elif upper_line.startswith('DEFINE '):
            return self._parse_define_declaration_comprehensive(line, location)
        elif upper_line.startswith('CONST '):
            return self._parse_const_declaration_comprehensive(line, location)
            
        # Equates
        elif upper_line.startswith('EQU '):
            return self._parse_equ_declaration_comprehensive(line, location)
            
        # Variable declarations
        elif self._is_global_variable_declaration(line):
            return self._parse_global_variable_declaration_comprehensive(line, location)
            
        # Use/Include statements
        elif upper_line.startswith('USE '):
            return self._parse_use_statement_comprehensive(line, location)
        elif upper_line.startswith('INCLUDE '):
            return self._parse_include_statement_comprehensive(line, location)
            
        # Conditional compilation
        elif upper_line.startswith('?IF '):
            return self._parse_conditional_compilation_comprehensive(line, location)
        elif upper_line.startswith('?ENDIF'):
            return tal_proc_parser.TALNode('endif_directive', value=line, location=location)
        elif upper_line.startswith('?ELSE'):
            return tal_proc_parser.TALNode('else_directive', value=line, location=location)
            
        else:
            # Enhanced fallback with system function detection
            global_node = tal_proc_parser.TALNode('global_statement', value=line, location=location)
            
            # Analyze for system functions in global context
            if '$' in line:
                system_funcs = self._extract_system_functions_from_expr(line)
                for func_name, func_info in system_funcs:
                    func_node = tal_proc_parser.TALNode('system_function', name=func_name, location=location)
                    func_node.attributes.update(func_info)
                    global_node.add_child(func_node)
            
            return global_node
    
    # Comprehensive parsers for specific global constructs
    
    def _parse_page_directive_comprehensive(self, line: str, location: tal_proc_parser.SourceLocation) -> tal_proc_parser.TALNode:
        """
        Parse ?PAGE compiler directive with title extraction.
        
        ?PAGE directives control page formatting in compiler listings.
        
        Args:
            line: ?PAGE directive line
            location: Source location
            
        Returns:
            AST node for the page directive
        """
        page_node = tal_proc_parser.TALNode('page_directive', value=line, location=location)
        
        # Extract page title if present
        if len(line) > 6:
            title_part = line[6:].strip()
            if title_part.startswith('"') and title_part.endswith('"'):
                page_node.attributes['title'] = title_part[1:-1]
            else:
                page_node.attributes['title'] = title_part
        
        return page_node
    
    def _parse_section_directive_comprehensive(self, line: str, location: tal_proc_parser.SourceLocation) -> tal_proc_parser.TALNode:
        """
        Parse ?SECTION compiler directive.
        
        ?SECTION directives organize code into logical sections.
        
        Args:
            line: ?SECTION directive line
            location: Source location
            
        Returns:
            AST node for the section directive
        """
        section_node = tal_proc_parser.TALNode('section_directive', value=line, location=location)
        
        parts = line.split()
        if len(parts) > 1:
            section_node.attributes['section_name'] = parts[1]
        
        return section_node
    
    def _parse_source_directive_comprehensive(self, line: str, location: tal_proc_parser.SourceLocation) -> tal_proc_parser.TALNode:
        """
        Parse ?SOURCE compiler directive for file inclusion.
        
        Args:
            line: ?SOURCE directive line
            location: Source location
            
        Returns:
            AST node for the source directive
        """
        source_node = tal_proc_parser.TALNode('source_directive', value=line, location=location)

        if len(line) > 7:
            content = line[7:].strip()
            if content.startswith('='):
                content = content[1:].strip()

            if '(' in content:
                base_name = content[:content.find('(')].strip()
                source_node.attributes['filename'] = base_name
            else:
                source_node.attributes['filename'] = content

        return source_node
    
    def _parse_struct_declaration_comprehensive(self, line: str, location: tal_proc_parser.SourceLocation) -> tal_proc_parser.TALNode:
        """
        Parse STRUCT declarations with template parameter support.
        
        STRUCT defines user-defined data types in TAL.
        
        Args:
            line: STRUCT declaration line
            location: Source location
            
        Returns:
            AST node for the struct declaration
        """
        struct_node = tal_proc_parser.TALNode('struct_decl', location=location)
        
        # Extract struct name and attributes
        parts = line.split()
        if len(parts) >= 2:
            struct_name = parts[1].rstrip('(*);')
            struct_node.name = struct_name
            struct_node.attributes['struct_name'] = struct_name
        
        # Check for template parameters
        if '(' in line and ')' in line:
            param_start = line.find('(')
            param_end = line.find(')')
            params = line[param_start+1:param_end]
            if params.strip():
                struct_node.attributes['template_params'] = params.strip()
        
        return struct_node
    
    def _parse_template_declaration_comprehensive(self, line: str, location: tal_proc_parser.SourceLocation) -> tal_proc_parser.TALNode:
        """
        Parse TEMPLATE declarations for generic programming.
        
        Args:
            line: TEMPLATE declaration line
            location: Source location
            
        Returns:
            AST node for the template declaration
        """
        template_node = tal_proc_parser.TALNode('template_decl', location=location, value=line)
        
        parts = line.split()
        if len(parts) >= 2:
            template_node.name = parts[1].rstrip(';')
            template_node.attributes['template_name'] = parts[1].rstrip(';')
        
        return template_node
    
    def _parse_subtype_declaration_comprehensive(self, line: str, location: tal_proc_parser.SourceLocation) -> tal_proc_parser.TALNode:
        """
        Parse SUBTYPE declarations for type aliases.
        
        Args:
            line: SUBTYPE declaration line
            location: Source location
            
        Returns:
            AST node for the subtype declaration
        """
        subtype_node = tal_proc_parser.TALNode('subtype_decl', location=location, value=line)
        
        # Parse SUBTYPE name = base_type;
        if '=' in line:
            name_part, type_part = line.split('=', 1)
            subtype_name = name_part.split()[1].strip()
            base_type = type_part.strip().rstrip(';')
            
            subtype_node.name = subtype_name
            subtype_node.attributes['subtype_name'] = subtype_name
            subtype_node.attributes['base_type'] = base_type
        
        return subtype_node
    
    def _parse_external_declaration_comprehensive(self, line: str, location: tal_proc_parser.SourceLocation) -> tal_proc_parser.TALNode:
        """
        Parse EXTERNAL declarations for external symbol references.
        
        Args:
            line: EXTERNAL declaration line
            location: Source location
            
        Returns:
            AST node for the external declaration
        """
        external_node = tal_proc_parser.TALNode('external_decl', location=location, value=line)
        
        # Extract external symbol information
        content = line[8:].strip().rstrip(';')  # Remove 'EXTERNAL'
        external_node.attributes['external_name'] = content
        
        return external_node
    
    def _parse_forward_declaration_comprehensive(self, line: str, location: tal_proc_parser.SourceLocation) -> tal_proc_parser.TALNode:
        """
        Parse FORWARD declarations for forward procedure references.
        
        Args:
            line: FORWARD declaration line
            location: Source location
            
        Returns:
            AST node for the forward declaration
        """
        forward_node = tal_proc_parser.TALNode('forward_decl', location=location, value=line)
        
        content = line[7:].strip().rstrip(';')  # Remove 'FORWARD'
        forward_node.attributes['forward_name'] = content
        
        return forward_node
    
    def _parse_const_declaration_comprehensive(self, line: str, location: tal_proc_parser.SourceLocation) -> tal_proc_parser.TALNode:
        """
        Parse CONST declarations for named constants.
        
        Args:
            line: CONST declaration line
            location: Source location
            
        Returns:
            AST node for the const declaration
        """
        const_node = tal_proc_parser.TALNode('const_decl', location=location, value=line)
        
        # Parse CONST name = value;
        if '=' in line:
            name_part, value_part = line.split('=', 1)
            const_name = name_part.split()[1].strip()
            const_value = value_part.strip().rstrip(';')
            
            const_node.name = const_name
            const_node.attributes['const_name'] = const_name
            const_node.attributes['const_value'] = const_value
        
        return const_node
    
    def _parse_equ_declaration_comprehensive(self, line: str, location: tal_proc_parser.SourceLocation) -> tal_proc_parser.TALNode:
        """
        Parse EQU declarations for symbolic constants.
        
        Args:
            line: EQU declaration line
            location: Source location
            
        Returns:
            AST node for the equate declaration
        """
        equ_node = tal_proc_parser.TALNode('equ_decl', location=location, value=line)
        
        parts = line.split()
        if len(parts) >= 3:  # EQU name value
            equ_node.name = parts[1]
            equ_node.attributes['equ_name'] = parts[1]
            equ_node.attributes['equ_value'] = parts[2].rstrip(';')
        
        return equ_node
    
    def _parse_use_statement_comprehensive(self, line: str, location: tal_proc_parser.SourceLocation) -> tal_proc_parser.TALNode:
        """
        Parse USE statements for module inclusion.
        
        Args:
            line: USE statement line
            location: Source location
            
        Returns:
            AST node for the use statement
        """
        use_node = tal_proc_parser.TALNode('use_stmt', location=location, value=line)
        
        module_name = line[3:].strip().rstrip(';')  # Remove 'USE'
        use_node.attributes['module_name'] = module_name
        
        return use_node
    
    def _parse_include_statement_comprehensive(self, line: str, location: tal_proc_parser.SourceLocation) -> tal_proc_parser.TALNode:
        """
        Parse INCLUDE statements for file inclusion.
        
        Args:
            line: INCLUDE statement line
            location: Source location
            
        Returns:
            AST node for the include statement
        """
        include_node = tal_proc_parser.TALNode('include_stmt', location=location, value=line)
        
        filename = line[7:].strip().rstrip(';')  # Remove 'INCLUDE'
        include_node.attributes['filename'] = filename
        
        return include_node
    
    def _parse_conditional_compilation_comprehensive(self, line: str, location: tal_proc_parser.SourceLocation) -> tal_proc_parser.TALNode:
        """
        Parse conditional compilation directives like ?IF.
        
        Args:
            line: Conditional compilation line
            location: Source location
            
        Returns:
            AST node for the conditional compilation directive
        """
        if_node = tal_proc_parser.TALNode('conditional_compilation', location=location, value=line)
        
        condition = line[3:].strip()  # Remove '?IF'
        if_node.attributes['condition'] = condition
        
        return if_node
    
    def _parse_literal_declaration_comprehensive(self, line: str, location: tal_proc_parser.SourceLocation) -> tal_proc_parser.TALNode:
        """
        Parse LITERAL declarations with multiple assignments.
        
        LITERAL creates compile-time string constants.
        
        Args:
            line: LITERAL declaration line
            location: Source location
            
        Returns:
            AST node for the literal declaration
        """
        literal_node = tal_proc_parser.TALNode('literal_decl', location=location)
        content = line[7:].strip().rstrip(';')
        assignments = self._smart_split(content, ',')
        
        for assignment in assignments:
            if '=' in assignment:
                name, value = assignment.split('=', 1)
                assign_node = tal_proc_parser.TALNode('assignment', name=name.strip(), value=value.strip(), location=location)
                literal_node.add_child(assign_node)
        
        return literal_node
    
    def _parse_define_declaration_comprehensive(self, line: str, location: tal_proc_parser.SourceLocation) -> tal_proc_parser.TALNode:
        """
        Parse DEFINE declarations for preprocessor-style definitions.
        
        Args:
            line: DEFINE declaration line
            location: Source location
            
        Returns:
            AST node for the define declaration
        """
        define_node = tal_proc_parser.TALNode('define_decl', location=location)
        content = line[6:].strip()
        if content.endswith('#'):
            content = content[:-1]
        if content.endswith(';'):
            content = content[:-1]
        
        assignments = self._smart_split(content, ',')
        for assignment in assignments:
            if '=' in assignment:
                name, value = assignment.split('=', 1)
                assign_node = tal_proc_parser.TALNode('assignment', name=name.strip(), value=value.strip(), location=location)
                define_node.add_child(assign_node)
        
        return define_node
    
    def _is_global_variable_declaration(self, line: str) -> bool:
        """
        Identify global variable declarations (not within procedures).
        
        Args:
            line: Source code line to check
            
        Returns:
            True if line appears to be a global variable declaration
        """
        upper_line = line.upper()
        tal_types = ['INT', 'STRING', 'REAL', 'FIXED', 'BYTE', 'CHAR', 'UNSIGNED', 'STRUCT']
        
        if any(upper_line.startswith(t) for t in tal_types) and 'PROC ' not in upper_line:
            return True
        
        return False
    
    def _parse_global_variable_declaration_comprehensive(self, line: str, location: tal_proc_parser.SourceLocation) -> tal_proc_parser.TALNode:
        """
        Parse global variable declarations with full type analysis.
        
        Args:
            line: Global variable declaration line
            location: Source location
            
        Returns:
            AST node for the global variable declaration
        """
        var_node = tal_proc_parser.TALNode('var_decl', location=location, value=line.rstrip(';'))
        
        parts = line.split()
        if len(parts) >= 2:
            var_type = parts[0].upper()
            var_node.attributes['type'] = var_type
            
            remaining = ' '.join(parts[1:]).rstrip(';')
            var_specs = self._smart_split(remaining, ',')
            
            for var_spec in var_specs:
                var_spec = var_spec.strip()
                if var_spec:
                    spec_node = self._parse_variable_spec_comprehensive(var_spec, var_type, location)
                    if spec_node:
                        var_node.add_child(spec_node)
        
        return var_node
    
    def _collect_system_functions(self, node: tal_proc_parser.TALNode) -> Dict[str, Any]:
        """
        Recursively collect all system functions used throughout the program.
        
        This method traverses the entire AST to catalog system function usage,
        providing statistics and location information for analysis.
        
        Args:
            node: Root AST node to search from
            
        Returns:
            Dictionary of system functions with usage statistics
        """
        system_funcs = {}
        
        def collect_recursive(n):
            if n.type in ['system_function_call', 'system_function']:
                func_name = n.attributes.get('original_name', n.name)
                if func_name not in system_funcs:
                    system_funcs[func_name] = {
                        'count': 0,
                        'locations': [],
                        'info': n.attributes.copy()
                    }
                system_funcs[func_name]['count'] += 1
                if n.location:
                    system_funcs[func_name]['locations'].append({
                        'line': n.location.line,
                        'column': n.location.column
                    })
            
            for child in n.children:
                collect_recursive(child)
        
        collect_recursive(node)
        return system_funcs
    
    def _extract_procedure_info(self, procedures: List[tal_proc_parser.TALNode]) -> Dict[str, Any]:
        """
        Extract comprehensive information about all procedures in the program.
        
        This method analyzes each procedure to extract metadata including
        parameters, local variables, statement counts, and system function usage.
        
        Args:
            procedures: List of procedure AST nodes
            
        Returns:
            Dictionary mapping procedure names to their detailed information
        """
        proc_info = {}
        
        for proc in procedures:
            info = {
                'name': proc.name,
                'return_type': proc.attributes.get('return_type', 'void'),
                'is_main': proc.attributes.get('is_main', False),
                'is_forward': proc.attributes.get('is_forward', False),
                'is_external': proc.attributes.get('is_external', False),
                'parameters': [],
                'local_variables': [],
                'statements_count': 0,
                'system_functions_used': []
            }
            
            # Extract parameter information
            params_node = proc.find_child_by_name('parameters')
            if params_node:
                for param in params_node.children:
                    param_info = {
                        'name': param.name,
                        'type': param.attributes.get('type', 'UNKNOWN'),
                        'is_pointer': param.attributes.get('pointer', False),
                        'is_array': param.attributes.get('array', False)
                    }
                    info['parameters'].append(param_info)
            
            # Extract local variable information
            local_decls = proc.find_child_by_name('local_declarations')
            if local_decls:
                for decl in local_decls.children:
                    if decl.type == 'var_decl':
                        for var_spec in decl.children:
                            if var_spec.type == 'var_spec':
                                var_info = {
                                    'name': var_spec.name,
                                    'type': decl.attributes.get('type', 'UNKNOWN'),
                                    'is_pointer': var_spec.attributes.get('pointer', False),
                                    'is_array': var_spec.attributes.get('array', False)
                                }
                                info['local_variables'].append(var_info)
            
            # Count statements and collect system functions
            statements = proc.find_child_by_name('statements')
            if statements:
                info['statements_count'] = len(statements.children)
                
                # Collect system functions used specifically in this procedure
                proc_system_funcs = self._collect_system_functions(proc)
                info['system_functions_used'] = list(proc_system_funcs.keys())
            
            proc_info[proc.name] = info
        
        return proc_info
    
    def _ast_to_json(self, node: tal_proc_parser.TALNode) -> Dict[str, Any]:
        """
        Convert the AST to JSON format for serialization and analysis.
        
        This method recursively converts the AST structure to a JSON-serializable
        dictionary, preserving all node information and metadata.
        
        Args:
            node: AST node to convert
            
        Returns:
            Dictionary representation of the AST node
        """
        result = {
            'type': node.type,
            'name': node.name,
            'value': node.value,
            'attributes': node.attributes.copy(),
            'children': []
        }
        
        if node.location:
            result['location'] = {
                'filename': node.location.filename,
                'line': node.location.line,
                'column': node.location.column
            }
        
        for child in node.children:
            result['children'].append(self._ast_to_json(child))
        
        return result
    
    def _count_nodes(self, node: tal_proc_parser.TALNode) -> int:
        """
        Count the total number of nodes in the AST.
        
        This provides a metric for AST complexity and parsing completeness.
        
        Args:
            node: Root node to count from
            
        Returns:
            Total number of AST nodes
        """
        count = 1
        for child in node.children:
            count += self._count_nodes(child)
        return count
    
    def _analyze_structure(self, program: tal_proc_parser.TALNode) -> Dict[str, Any]:
        """
        Analyze the overall program structure and generate comprehensive statistics.
        
        This method provides high-level metrics about the parsed program including
        procedure counts, variable declarations, system function usage, and more.
        
        Args:
            program: Root program AST node
            
        Returns:
            Dictionary containing structural analysis results
        """
        structure = {
            'total_nodes': self._count_nodes(program),
            'procedures': 0,
            'variables': 0,
            'statements': 0,
            'procedure_list': [],
            'system_functions_count': 0,
            'directives_count': 0,
            'global_variables': 0
        }
        
        # Analyze global elements
        for child in program.children:
            if child.type == 'procedure':
                structure['procedures'] += 1
                structure['procedure_list'].append({
                    'name': child.name,
                    'return_type': child.attributes.get('return_type', 'void'),
                    'is_main': child.attributes.get('is_main', False)
                })
                
                # Count statements and variables within procedures
                for proc_child in child.children:
                    if proc_child.type == 'statements':
                        structure['statements'] += len(proc_child.children)
                    elif proc_child.type == 'local_declarations':
                        structure['variables'] += len(proc_child.children)
            
            elif child.type in ['system_function_call', 'system_function']:
                structure['system_functions_count'] += 1
            
            elif child.type.endswith('_directive'):
                structure['directives_count'] += 1
            
            elif child.type == 'var_decl':
                structure['global_variables'] += 1
        
        return structure

def main():
    """
    Enhanced main function providing a comprehensive command-line interface.
    
    This function implements a full-featured CLI for the TAL parser with support
    for different output formats, verbose reporting, system function analysis,
    and optional LLM integration for code analysis.
    
    The CLI supports:
    - Multiple output formats (S-expressions, JSON)
    - Detailed system function reporting and validation
    - Verbose parsing statistics and error reporting
    - Extensible LLM analysis integration
    - Comprehensive error handling and user feedback
    
    Returns:
        Exit code (0 for success, 1 for error)
    """
    parser = argparse.ArgumentParser(
        description='Enhanced TAL parser with comprehensive system function support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input.tal                          # Parse and output to input.tal.ast
  %(prog)s input.tal -f json -o output.json   # Output JSON format to specific file
  %(prog)s input.tal -usellm                  # Parse and analyze with LLM
  %(prog)s input.tal -usellm -analysis security  # Security-focused LLM analysis
  %(prog)s input.tal --show-system-funcs      # Show detailed system function usage
        """
    )
    
    # Required arguments
    parser.add_argument('input_file', help='Input TAL file to parse')
    
    # Optional arguments
    parser.add_argument('-f', '--format', choices=['sexp', 'json'], 
                       default='sexp', help='Output format (default: sexp)')
    parser.add_argument('-o', '--output', help='Output file (default: input_file.tal.ast)')
    parser.add_argument('-usellm', '--use-llm', action='store_true', 
                       help='Analyze code with LLM after parsing')
    parser.add_argument('-analysis', '--analysis-type', 
                       choices=['general', 'security', 'performance', 'maintainability'],
                       default='general', help='Type of LLM analysis (default: general)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--show-system-funcs', action='store_true', 
                       help='Show detailed system function usage')
    parser.add_argument('--validate-system-funcs', action='store_true',
                       help='Validate system function usage and arguments')
    parser.add_argument('--llm-model', default='gpt-4', help='LLM model to use (default: gpt-4)')
    parser.add_argument('--llm-tokens', type=int, default=2000, help='Max tokens for LLM (default: 2000)')
    
    args = parser.parse_args()
    
    # Validate input file existence
    if not os.path.exists(args.input_file):
        print(f"Error: File '{args.input_file}' not found")
        return 1
    
    # Generate default output filename if not specified
    if not args.output:
        base_name = os.path.splitext(args.input_file)[0]
        extension = '.ast' if args.format == 'sexp' else '.json'
        args.output = base_name + '.tal' + extension
    
    # Display configuration if verbose
    if args.verbose:
        print(f"Enhanced TAL Parser v2.0 - Processing: {args.input_file}")
        print(f"Output format: {args.format}")
        print(f"Output file: {args.output}")
        if args.use_llm:
            print(f"LLM analysis: {args.analysis_type} using {args.llm_model}")
    
    try:
        # Initialize and run the enhanced TAL parser
        tal_parser = EnhancedTALParser()
        result = tal_parser.parse_file(args.input_file)
        
        print(f"Parsing TAL file: {args.input_file}")
        print("=" * 60)
        
        # Display parsing errors and warnings
        if tal_parser.errors:
            print(f"\nFound {len(tal_parser.errors)} error(s):")
            print("-" * 40)
            for error in tal_parser.errors:
                print(f"{error}\n")
        
        if tal_parser.warnings:
            print(f"\nFound {len(tal_parser.warnings)} warning(s):")
            print("-" * 40)
            for warning in tal_parser.warnings:
                print(f"{warning}\n")
        
        # Check for fatal parsing errors
        if not result.get('success'):
            print(f"Parse failed: {result.get('error', 'Unknown error')}")
            return 1
        
        # Extract procedure information for reporting
        procedures = []
        if result.get('ast'):
            for child in result['ast'].children:
                if child.type == 'procedure':
                    procedures.append(child)
        
        # Display procedure information
        if procedures:
            print(f"\nFound {len(procedures)} procedure(s):")
            print("-" * 40)
            for i, proc in enumerate(procedures, 1):
                print(f"\nProcedure {i}: {proc.name}")
                
                # Show procedure attributes
                attrs = []
                if proc.attributes.get('return_type'):
                    attrs.append(f"returns {proc.attributes['return_type']}")
                if proc.attributes.get('is_main'):
                    attrs.append("MAIN")
                if proc.attributes.get('is_forward'):
                    attrs.append("FORWARD")
                if proc.attributes.get('is_external'):
                    attrs.append("EXTERNAL")
                
                if attrs:
                    print(f"  Attributes: {', '.join(attrs)}")
                
                # Show detailed information in verbose mode
                if args.verbose:
                    params_node = proc.find_child_by_name('parameters')
                    if params_node and params_node.children:
                        print(f"  Parameters:")
                        for param in params_node.children:
                            param_info = param.name
                            if param.attributes.get('type') != 'UNKNOWN':
                                param_info += f" ({param.attributes['type']}"
                                if param.attributes.get('pointer'):
                                    param_info += " pointer"
                                param_info += ")"
                            print(f"    - {param_info}")
                    
                    # Show content counts
                    local_decls = proc.find_child_by_name('local_declarations')
                    statements = proc.find_child_by_name('statements')
                    
                    if local_decls:
                        print(f"  Local declarations: {len(local_decls.children)}")
                    if statements:
                        print(f"  Statements: {len(statements.children)}")
        
        # Display system function usage analysis
        if args.show_system_funcs or args.verbose:
            system_funcs = result.get('system_functions_used', {})
            if system_funcs:
                print(f"\nSystem Functions Used ({len(system_funcs)}):")
                print("-" * 40)
                for func_name, func_info in system_funcs.items():
                    count = func_info['count']
                    description = func_info['info'].get('description', 'Unknown function')
                    returns = func_info['info'].get('returns', 'Unknown')
                    
                    print(f"  {func_name}: used {count} time(s)")
                    print(f"    Description: {description}")
                    print(f"    Returns: {returns}")
                    
                    if func_info['info'].get('unknown'):
                        print(f"    ⚠️  Unknown system function")
                    
                    if args.verbose and func_info['locations']:
                        locations = func_info['locations'][:3]  # Show first 3 locations
                        loc_str = ', '.join([f"line {loc['line']}" for loc in locations])
                        if len(func_info['locations']) > 3:
                            loc_str += f" (+{len(func_info['locations']) - 3} more)"
                        print(f"    Locations: {loc_str}")
                    print()
        
        # System function validation reporting
        if args.validate_system_funcs:
            print("\nSystem Function Validation:")
            print("-" * 40)
            validation_warnings = []
            for error in tal_parser.warnings:
                if hasattr(error, 'error_code') and error.error_code.startswith('W3'):
                    validation_warnings.append(error)
            
            if validation_warnings:
                for warning in validation_warnings:
                    print(f"  ⚠️  {warning}")
            else:
                print("  ✅ No system function validation issues found")
        
        # Generate output content based on requested format
        if args.format == 'sexp':
            output_content = result['sexp']
        elif args.format == 'json':
            # Create comprehensive JSON output with all analysis data
            json_output = {
                'success': result['success'],
                'filename': args.input_file,
                'ast': result['json'],
                'procedures': result.get('procedures', {}),
                'system_functions_used': result.get('system_functions_used', {}),
                'statistics': {
                    'node_count': result['node_count'],
                    'procedure_count': len(procedures),
                    'error_count': len(tal_parser.errors),
                    'warning_count': len(tal_parser.warnings),
                    'system_functions_count': len(result.get('system_functions_used', {}))
                },
                'structure': result.get('structure', {})
            }
            output_content = json.dumps(json_output, indent=2)
        
        # Write output to file
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(output_content)
            
            if args.verbose:
                print(f"\nAST written to: {args.output}")
            else:
                print(f"\nOutput written to: {args.output}")
                
        except Exception as e:
            print(f"Error writing output file: {e}")
            return 1
        
        # LLM Analysis integration (placeholder for future implementation)
        if args.use_llm:
            print("\n" + "=" * 60)
            print("LLM ANALYSIS:")
            print("=" * 60)
            
            try:
                # Note: LLM analysis would need separate implementation
                # This provides the interface structure for future extension
                print(f"Analysis Type: {args.analysis_type}")
                print(f"Model: {args.llm_model}")
                print("LLM analysis feature requires separate implementation")
                
            except Exception as e:
                print(f"Error during LLM analysis: {e}")
        
        # Display comprehensive parsing summary
        print(f"\n" + "=" * 60)
        print("Parsing Summary:")
        print(f"  Total AST nodes: {result['node_count']}")
        print(f"  Procedures found: {len(procedures)}")
        print(f"  System functions used: {len(result.get('system_functions_used', {}))}")
        print(f"  Errors: {len(tal_parser.errors)}")
        print(f"  Warnings: {len(tal_parser.warnings)}")
        if args.use_llm:
            print(f"  LLM analysis: {args.analysis_type}")
        
        return 0
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
