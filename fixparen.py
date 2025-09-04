#!/usr/bin/env python3
"""
Enhanced TAL AST Node-Aware Parentheses Fixer

This program fixes unmatched parentheses by understanding all TAL AST node types
from the enhanced TAL parser. It recognizes the comprehensive set of node types
including procedures, system functions, control flow, declarations, and TAL-specific
constructs.

Usage:
    python fix_parentheses.py input.ast [output.ast]
"""

import sys
import argparse
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# Comprehensive AST node types from the enhanced TAL parser
AST_NODE_TYPES = {
    # Top-level program structure
    'program': {'can_have_children': True, 'is_container': True, 'priority': 1},
    'module': {'can_have_children': True, 'is_container': True, 'priority': 1},
    
    # Procedure-related nodes
    'procedure': {'can_have_children': True, 'is_container': True, 'priority': 2},
    'parameters': {'can_have_children': True, 'is_container': True, 'priority': 3},
    'parameter': {'can_have_children': False, 'is_container': False, 'priority': 4},
    'local_declarations': {'can_have_children': True, 'is_container': True, 'priority': 3},
    'statements': {'can_have_children': True, 'is_container': True, 'priority': 3},
    
    # Declaration nodes
    'var_decl': {'can_have_children': True, 'is_container': False, 'priority': 4},
    'var_spec': {'can_have_children': False, 'is_container': False, 'priority': 5},
    'const_decl': {'can_have_children': False, 'is_container': False, 'priority': 4},
    'literal_decl': {'can_have_children': True, 'is_container': False, 'priority': 4},
    'struct_decl': {'can_have_children': True, 'is_container': False, 'priority': 4},
    'template_decl': {'can_have_children': False, 'is_container': False, 'priority': 4},
    'subtype_decl': {'can_have_children': False, 'is_container': False, 'priority': 4},
    'external_decl': {'can_have_children': False, 'is_container': False, 'priority': 4},
    'forward_decl': {'can_have_children': False, 'is_container': False, 'priority': 4},
    'equ_decl': {'can_have_children': False, 'is_container': False, 'priority': 4},
    'define_decl': {'can_have_children': True, 'is_container': False, 'priority': 4},
    'name_decl': {'can_have_children': False, 'is_container': False, 'priority': 4},
    
    # Statement nodes
    'statement': {'can_have_children': True, 'is_container': False, 'priority': 5},
    'assignment': {'can_have_children': True, 'is_container': False, 'priority': 5},
    'call_stmt': {'can_have_children': True, 'is_container': False, 'priority': 5},
    'system_function_call': {'can_have_children': True, 'is_container': False, 'priority': 5},
    'return_stmt': {'can_have_children': True, 'is_container': False, 'priority': 5},
    'if_stmt': {'can_have_children': True, 'is_container': False, 'priority': 5},
    'while_stmt': {'can_have_children': True, 'is_container': False, 'priority': 5},
    'for_stmt': {'can_have_children': True, 'is_container': False, 'priority': 5},
    'case_stmt': {'can_have_children': True, 'is_container': False, 'priority': 5},
    'scan_stmt': {'can_have_children': False, 'is_container': False, 'priority': 5},
    'rscan_stmt': {'can_have_children': False, 'is_container': False, 'priority': 5},
    'goto_stmt': {'can_have_children': False, 'is_container': False, 'priority': 5},
    'interrupt_stmt': {'can_have_children': False, 'is_container': False, 'priority': 5},
    'assert_stmt': {'can_have_children': True, 'is_container': False, 'priority': 5},
    'stop_stmt': {'can_have_children': False, 'is_container': False, 'priority': 5},
    'abort_stmt': {'can_have_children': False, 'is_container': False, 'priority': 5},
    'label': {'can_have_children': False, 'is_container': False, 'priority': 5},
    
    # Expression and operator nodes
    'expression': {'can_have_children': True, 'is_container': False, 'priority': 6},
    'system_function': {'can_have_children': False, 'is_container': False, 'priority': 6},
    'operator': {'can_have_children': False, 'is_container': False, 'priority': 6},
    
    # Directive nodes
    'comment': {'can_have_children': False, 'is_container': False, 'priority': 7},
    'page_directive': {'can_have_children': False, 'is_container': False, 'priority': 4},
    'section_directive': {'can_have_children': False, 'is_container': False, 'priority': 4},
    'source_directive': {'can_have_children': False, 'is_container': False, 'priority': 4},
    'nolist_directive': {'can_have_children': False, 'is_container': False, 'priority': 4},
    'list_directive': {'can_have_children': False, 'is_container': False, 'priority': 4},
    'symbols_directive': {'can_have_children': False, 'is_container': False, 'priority': 4},
    'nosymbols_directive': {'can_have_children': False, 'is_container': False, 'priority': 4},
    'save_directive': {'can_have_children': False, 'is_container': False, 'priority': 4},
    'restore_directive': {'can_have_children': False, 'is_container': False, 'priority': 4},
    'heap_directive': {'can_have_children': False, 'is_container': False, 'priority': 4},
    'stack_directive': {'can_have_children': False, 'is_container': False, 'priority': 4},
    'conditional_compilation': {'can_have_children': False, 'is_container': False, 'priority': 4},
    'endif_directive': {'can_have_children': False, 'is_container': False, 'priority': 4},
    'else_directive': {'can_have_children': False, 'is_container': False, 'priority': 4},
    
    # Global statement nodes
    'global_statement': {'can_have_children': True, 'is_container': False, 'priority': 4},
    'use_stmt': {'can_have_children': False, 'is_container': False, 'priority': 4},
    'include_stmt': {'can_have_children': False, 'is_container': False, 'priority': 4},
}

@dataclass
class ASTNode:
    """Represents an AST node during parsing with enhanced TAL support."""
    node_type: str
    start_pos: int
    line_num: int
    depth: int
    is_complete: bool = False
    children: List['ASTNode'] = None
    expected_close_pos: Optional[int] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
    
    def can_have_children(self) -> bool:
        return AST_NODE_TYPES.get(self.node_type, {}).get('can_have_children', True)
    
    def is_container(self) -> bool:
        return AST_NODE_TYPES.get(self.node_type, {}).get('is_container', False)
    
    def get_priority(self) -> int:
        """Get closure priority - lower numbers should close before higher numbers."""
        return AST_NODE_TYPES.get(self.node_type, {}).get('priority', 10)

class EnhancedASTParser:
    """Enhanced parser that understands all TAL AST node types."""
    
    def __init__(self, content: str):
        self.content = content
        self.lines = content.split('\n')
        self.pos = 0
        self.line_num = 1
        self.column = 1
        self.stack = []  # Stack of incomplete ASTNode objects
        self.completed_nodes = []
        
    def current_char(self) -> Optional[str]:
        return self.content[self.pos] if self.pos < len(self.content) else None
    
    def advance(self):
        if self.pos < len(self.content):
            if self.content[self.pos] == '\n':
                self.line_num += 1
                self.column = 1
            else:
                self.column += 1
            self.pos += 1
    
    def skip_whitespace(self):
        while self.current_char() and self.current_char() in ' \t\n':
            self.advance()
    
    def read_identifier(self) -> str:
        """Read an identifier (node type)."""
        start_pos = self.pos
        while (self.current_char() and 
               self.current_char() not in '() \t\n:{}"\'^'):
            self.advance()
        return self.content[start_pos:self.pos]
    
    def skip_to_next_sexp_or_close(self):
        """Skip content until next '(' or ')'."""
        in_string = False
        in_attrs = False
        brace_depth = 0
        
        while self.current_char():
            char = self.current_char()
            
            if char == '"' and not in_attrs:
                in_string = not in_string
            elif char == '{' and not in_string:
                in_attrs = True
                brace_depth = 1
            elif char == '}' and in_attrs:
                brace_depth -= 1
                if brace_depth == 0:
                    in_attrs = False
            elif char in '()' and not in_string and not in_attrs:
                break
            
            self.advance()
    
    def parse(self) -> List[ASTNode]:
        """Parse the content and return incomplete nodes."""
        while self.pos < len(self.content):
            self.skip_whitespace()
            
            char = self.current_char()
            if char == '(':
                self.parse_node()
            elif char == ')':
                self.close_current_node()
            else:
                self.advance()
        
        return self.stack  # Return incomplete nodes
    
    def parse_node(self):
        """Parse an S-expression node with TAL node type recognition."""
        if self.current_char() != '(':
            return
        
        start_pos = self.pos
        start_line = self.line_num
        self.advance()  # Skip '('
        
        self.skip_whitespace()
        
        # Read node type
        node_type = self.read_identifier()
        
        # Validate node type against known TAL types
        if node_type not in AST_NODE_TYPES:
            # Add unknown node types with default properties
            AST_NODE_TYPES[node_type] = {
                'can_have_children': True, 
                'is_container': False, 
                'priority': 8
            }
        
        # Create node
        node = ASTNode(
            node_type=node_type,
            start_pos=start_pos,
            line_num=start_line,
            depth=len(self.stack)
        )
        
        # Add to parent if exists
        if self.stack:
            self.stack[-1].children.append(node)
        else:
            self.completed_nodes.append(node)
        
        # Push onto stack
        self.stack.append(node)
        
        # Skip to next meaningful content
        self.skip_to_next_sexp_or_close()
    
    def close_current_node(self):
        """Close the current node."""
        if self.current_char() == ')':
            self.advance()  # Skip ')'
            
            if self.stack:
                node = self.stack.pop()
                node.is_complete = True
        
        self.skip_whitespace()

def find_closure_points(content: str, incomplete_nodes: List[ASTNode]) -> List[Tuple[int, int, str]]:
    """
    Find appropriate closure points for incomplete nodes using TAL-specific logic.
    
    Returns list of (position, indent_level, node_type) for insertions.
    """
    if not incomplete_nodes:
        return []
    
    lines = content.split('\n')
    insertions = []
    
    # Sort nodes by priority (containers first), then by depth (deepest first), then by position
    sorted_nodes = sorted(incomplete_nodes, key=lambda n: (n.get_priority(), -n.depth, n.start_pos))
    
    for node in sorted_nodes:
        insert_pos, indent_level = find_node_closure_point(content, lines, node, incomplete_nodes)
        insertions.append((insert_pos, indent_level, node.node_type))
    
    return insertions

def find_node_closure_point(content: str, lines: List[str], node: ASTNode, all_incomplete: List[ASTNode]) -> Tuple[int, int]:
    """
    Find the best closure point for a specific TAL AST node.
    
    Returns (position, indent_level).
    """
    node_line_idx = node.line_num - 1  # Convert to 0-based
    
    # Determine closure strategy based on node type and characteristics
    if node.is_container():
        # Container nodes (program, procedure, parameters, etc.) should close after their logical content
        return find_container_closure_point(content, lines, node, node_line_idx)
    elif node.node_type in ['procedure', 'struct_decl', 'template_decl']:
        # Major structural nodes need special handling
        return find_structural_closure_point(content, lines, node, node_line_idx)
    elif node.node_type in ['parameters', 'local_declarations', 'statements']:
        # Procedure section nodes
        return find_section_closure_point(content, lines, node, node_line_idx)
    elif node.node_type.endswith('_stmt') or node.node_type in ['assignment', 'call_stmt', 'system_function_call']:
        # Statement nodes should close quickly
        return find_statement_closure_point(content, lines, node, node_line_idx)
    elif node.node_type.endswith('_directive'):
        # Directive nodes are typically single-line
        return find_directive_closure_point(content, lines, node, node_line_idx)
    else:
        # Default strategy for other nodes
        return find_simple_closure_point(content, lines, node, node_line_idx)

def find_container_closure_point(content: str, lines: List[str], node: ASTNode, node_line_idx: int) -> Tuple[int, int]:
    """Find closure point for container nodes (program, procedure, etc.)."""
    
    # For program nodes, close at the very end
    if node.node_type == 'program':
        return len(content), 0
    
    # Look for the next sibling node at the same level or higher
    for i in range(node_line_idx + 1, len(lines)):
        line = lines[i].strip()
        
        # Skip empty lines and comments
        if not line or line.startswith('!'):
            continue
        
        # Check if this line starts a new node at the same level as our node
        if line.startswith('('):
            # Get the indentation of this line
            line_indent = len(lines[i]) - len(lines[i].lstrip())
            
            # Get the indentation of our node's line
            if node_line_idx < len(lines):
                node_indent = len(lines[node_line_idx]) - len(lines[node_line_idx].lstrip())
            else:
                node_indent = 0
            
            # If this line has same or less indentation, it's a sibling or higher level
            if line_indent <= node_indent:
                # Insert before this line
                insert_pos = sum(len(lines[j]) + 1 for j in range(i))
                return insert_pos, node_indent
    
    # No sibling found, close at end of file
    return len(content), 0

def find_structural_closure_point(content: str, lines: List[str], node: ASTNode, node_line_idx: int) -> Tuple[int, int]:
    """Find closure point for structural nodes like procedures and structs."""
    
    # Look for END statement for procedures
    if node.node_type == 'procedure':
        for i in range(node_line_idx + 1, len(lines)):
            line = lines[i].strip().upper()
            if line.startswith('END') and not line.startswith('ENDIF'):
                # Close after the END statement
                insert_pos = sum(len(lines[j]) + 1 for j in range(i + 1))
                node_indent = len(lines[node_line_idx]) - len(lines[node_line_idx].lstrip()) if node_line_idx < len(lines) else 0
                return insert_pos, node_indent
    
    # For struct declarations, look for the end of the structure
    elif node.node_type == 'struct_decl':
        brace_count = 0
        in_struct = False
        for i in range(node_line_idx, len(lines)):
            line = lines[i]
            if 'STRUCT' in line.upper():
                in_struct = True
            if in_struct:
                brace_count += line.count('(')
                brace_count -= line.count(')')
                if brace_count == 0 and i > node_line_idx:
                    insert_pos = sum(len(lines[j]) + 1 for j in range(i + 1))
                    node_indent = len(lines[node_line_idx]) - len(lines[node_line_idx].lstrip()) if node_line_idx < len(lines) else 0
                    return insert_pos, node_indent
    
    # Default: close at a reasonable distance
    return find_container_closure_point(content, lines, node, node_line_idx)

def find_section_closure_point(content: str, lines: List[str], node: ASTNode, node_line_idx: int) -> Tuple[int, int]:
    """Find closure point for procedure section nodes (parameters, local_declarations, statements)."""
    
    # Look for the start of the next section or END
    section_keywords = ['BEGIN', 'END']
    if node.node_type == 'parameters':
        section_keywords = ['BEGIN', 'LOCAL', 'END']
    elif node.node_type == 'local_declarations':
        section_keywords = ['BEGIN', 'END']
    
    for i in range(node_line_idx + 1, len(lines)):
        line = lines[i].strip().upper()
        
        # Skip empty lines and comments
        if not line or line.startswith('!'):
            continue
        
        # Check for section transitions
        for keyword in section_keywords:
            if line.startswith(keyword):
                insert_pos = sum(len(lines[j]) + 1 for j in range(i))
                node_indent = len(lines[node_line_idx]) - len(lines[node_line_idx].lstrip()) if node_line_idx < len(lines) else 0
                return insert_pos, node_indent
        
        # Check for new procedure (same level)
        if line.startswith('(PROCEDURE') or 'PROC ' in line:
            insert_pos = sum(len(lines[j]) + 1 for j in range(i))
            node_indent = len(lines[node_line_idx]) - len(lines[node_line_idx].lstrip()) if node_line_idx < len(lines) else 0
            return insert_pos, node_indent
    
    # Default: close at end of file
    return len(content), 0

def find_statement_closure_point(content: str, lines: List[str], node: ASTNode, node_line_idx: int) -> Tuple[int, int]:
    """Find closure point for statement nodes."""
    
    # For statement nodes, look for the end of the current statement or the start of the next
    for i in range(node_line_idx + 1, min(node_line_idx + 5, len(lines))):
        line = lines[i].strip()
        
        # Skip empty lines
        if not line:
            continue
        
        # If we hit a comment, we can close before it
        if line.startswith('!'):
            insert_pos = sum(len(lines[j]) + 1 for j in range(i))
            node_indent = len(lines[node_line_idx]) - len(lines[node_line_idx].lstrip()) if node_line_idx < len(lines) else 0
            return insert_pos, node_indent
        
        # If we hit a new statement or declaration at same or higher level, close before it
        if (line.startswith('(') or 
            any(line.upper().startswith(kw) for kw in ['IF ', 'WHILE ', 'FOR ', 'CASE ', 'CALL ', 'RETURN ', 'BEGIN', 'END'])):
            
            line_indent = len(lines[i]) - len(lines[i].lstrip())
            node_indent = len(lines[node_line_idx]) - len(lines[node_line_idx].lstrip()) if node_line_idx < len(lines) else 0
            
            if line_indent <= node_indent + 2:  # Allow slight indentation
                insert_pos = sum(len(lines[j]) + 1 for j in range(i))
                return insert_pos, node_indent
    
    # Default: close after a few lines
    target_line = min(node_line_idx + 3, len(lines))
    insert_pos = sum(len(lines[j]) + 1 for j in range(target_line))
    node_indent = len(lines[node_line_idx]) - len(lines[node_line_idx].lstrip()) if node_line_idx < len(lines) else 0
    return insert_pos, node_indent

def find_directive_closure_point(content: str, lines: List[str], node: ASTNode, node_line_idx: int) -> Tuple[int, int]:
    """Find closure point for directive nodes (typically single-line)."""
    
    # Directives are usually single-line, close at end of current line
    if node_line_idx + 1 < len(lines):
        insert_pos = sum(len(lines[j]) + 1 for j in range(node_line_idx + 1))
    else:
        insert_pos = len(content)
    
    node_indent = len(lines[node_line_idx]) - len(lines[node_line_idx].lstrip()) if node_line_idx < len(lines) else 0
    return insert_pos, node_indent

def find_simple_closure_point(content: str, lines: List[str], node: ASTNode, node_line_idx: int) -> Tuple[int, int]:
    """Find closure point for simple nodes (expressions, etc.)."""
    
    # For simple nodes, look for the end of the current logical construct
    for i in range(node_line_idx + 1, min(node_line_idx + 8, len(lines))):
        line = lines[i].strip()
        
        # Skip empty lines
        if not line:
            continue
        
        # If we hit a comment, we can close before it
        if line.startswith('!'):
            insert_pos = sum(len(lines[j]) + 1 for j in range(i))
            node_indent = len(lines[node_line_idx]) - len(lines[node_line_idx].lstrip()) if node_line_idx < len(lines) else 0
            return insert_pos, node_indent
        
        # If we hit a new construct at same or higher level, close before it
        if line.startswith('('):
            line_indent = len(lines[i]) - len(lines[i].lstrip())
            node_indent = len(lines[node_line_idx]) - len(lines[node_line_idx].lstrip()) if node_line_idx < len(lines) else 0
            
            if line_indent <= node_indent:
                insert_pos = sum(len(lines[j]) + 1 for j in range(i))
                return insert_pos, node_indent
    
    # Default: close at a reasonable distance after the node
    target_line = min(node_line_idx + 4, len(lines))
    insert_pos = sum(len(lines[j]) + 1 for j in range(target_line))
    node_indent = len(lines[node_line_idx]) - len(lines[node_line_idx].lstrip()) if node_line_idx < len(lines) else 0
    return insert_pos, node_indent

def fix_parentheses_tal_aware(content: str, verbose: bool = False) -> str:
    """
    Fix parentheses using enhanced TAL AST node type awareness.
    """
    if verbose:
        print("Parsing TAL AST structure...")
    
    parser = EnhancedASTParser(content)
    incomplete_nodes = parser.parse()
    
    if verbose:
        print(f"Found {len(incomplete_nodes)} incomplete nodes:")
        for node in incomplete_nodes:
            priority = node.get_priority()
            print(f"  - {node.node_type} at line {node.line_num} (depth {node.depth}, priority {priority})")
    
    if not incomplete_nodes:
        if verbose:
            print("All nodes are complete!")
        return content
    
    # Find closure points
    insertions = find_closure_points(content, incomplete_nodes)
    
    if verbose:
        print(f"Planning {len(insertions)} closures:")
        for pos, indent, node_type in insertions:
            print(f"  - Close {node_type} at position {pos} with indent {indent}")
    
    # Apply insertions (from end to start to avoid position shifts)
    result = content
    for pos, indent_level, node_type in sorted(insertions, reverse=True):
        # Create closing parenthesis with proper indentation
        indent_str = ' ' * indent_level
        closing = f"{indent_str})"
        
        # Insert
        if pos >= len(result):
            # At end of file
            if not result.endswith('\n'):
                result += '\n'
            result += closing
        else:
            # Insert at position
            result = result[:pos] + closing + '\n' + result[pos:]
    
    if verbose:
        print(f"Added {len(insertions)} closing parentheses")
    
    return result

def main():
    """Main function with enhanced TAL support."""
    parser = argparse.ArgumentParser(
        description='Enhanced TAL AST node-aware parentheses fixer',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('input_file', help='Input TAL AST file to fix')
    parser.add_argument('output_file', nargs='?', help='Output file (default: overwrite input)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--check-only', action='store_true', help='Only analyze, do not fix')
    parser.add_argument('--show-node-types', action='store_true', help='Show all recognized node types')
    
    args = parser.parse_args()
    
    if args.show_node_types:
        print("Recognized TAL AST Node Types:")
        print("=" * 50)
        for node_type, properties in sorted(AST_NODE_TYPES.items()):
            container = "Container" if properties['is_container'] else "Simple"
            children = "Children" if properties['can_have_children'] else "No children"
            priority = properties['priority']
            print(f"{node_type:25} | {container:9} | {children:11} | Priority: {priority}")
        return 0
    
    # Read input file
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: Input file '{args.input_file}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading input file: {e}")
        sys.exit(1)
    
    if args.check_only:
        parser = EnhancedASTParser(content)
        incomplete_nodes = parser.parse()
        
        print(f"TAL AST Analysis of {args.input_file}:")
        print(f"  Incomplete nodes: {len(incomplete_nodes)}")
        
        if incomplete_nodes:
            print("  Incomplete nodes by type:")
            node_counts = {}
            for node in incomplete_nodes:
                node_counts[node.node_type] = node_counts.get(node.node_type, 0) + 1
            
            for node_type, count in sorted(node_counts.items()):
                print(f"    {node_type}: {count}")
            
            if args.verbose:
                print("  Details:")
                for node in incomplete_nodes:
                    priority = node.get_priority()
                    print(f"    - {node.node_type} at line {node.line_num} (depth {node.depth}, priority {priority})")
        else:
            print("  All nodes are complete!")
        
        sys.exit(0 if not incomplete_nodes else 1)
    
    # Fix the content
    fixed_content = fix_parentheses_tal_aware(content, args.verbose)
    
    # Determine output file
    output_file = args.output_file if args.output_file else args.input_file
    
    # Write fixed content
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        
        # Verify
        final_opens = fixed_content.count('(')
        final_closes = fixed_content.count(')')
        
        if final_opens == final_closes:
            print(f"Successfully fixed TAL AST structure in {output_file}")
        else:
            print(f"Warning: Still unbalanced - opens: {final_opens}, closes: {final_closes}")
            
    except Exception as e:
        print(f"Error writing output file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
