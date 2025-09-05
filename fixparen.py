#!/usr/bin/env python3
"""
Enhanced Hybrid AST Validator: Analyzes and fixes parentheses issues correctly.
"""

import sys
import argparse
import re
import os
from datetime import datetime

def get_all_node_types():
    """
    Return list of all possible AST node types.
    
    HARDCODED: This list is specific to TAL (Tandem Application Language) AST structure.
    """
    return [
        'program', 'comment', 'global_statement', 'source_directive', 'page_directive',
        'list_directive', 'nolist_directive', 'procedure', 'var_decl', 
        'struct_decl', 'const_decl', 'literal_decl', 'parameter', 'parameters',
        'local_declarations', 'statements', 'assignment', 'return_stmt', 
        'if_stmt', 'while_stmt', 'for_stmt', 'case_stmt', 'statement', 
        'var_spec', 'expression', 'operator', 'system_function'
    ]

def analyze_line_parentheses(line):
    """Analyze parentheses in a single line, respecting string boundaries."""
    opens = 0
    closes = 0
    in_string = False
    string_char = None
    
    for i, char in enumerate(line):
        if char in '"\'':
            if not in_string:
                in_string = True
                string_char = char
            elif char == string_char and (i == 0 or line[i-1] != '\\'):
                in_string = False
                string_char = None
        elif not in_string:
            if char == '(':
                opens += 1
            elif char == ')':
                closes += 1
    
    return opens, closes, opens - closes

def detect_node_starts(content):
    """Detect where AST nodes start in the content."""
    
    node_types = get_all_node_types()
    node_pattern = r'\((' + '|'.join(re.escape(nt) for nt in node_types) + r')\b'
    
    nodes = []
    lines = content.split('\n')
    
    for match in re.finditer(node_pattern, content):
        node_type = match.group(1)
        start_pos = match.start()
        
        # Calculate line number
        line_num = content[:start_pos].count('\n') + 1
        
        # Get the line content
        line_content = lines[line_num - 1].strip() if line_num <= len(lines) else ""
        
        # Analyze just this line
        opens, closes, balance = analyze_line_parentheses(line_content)
        
        nodes.append({
            'type': node_type,
            'line': line_num,
            'position': start_pos,
            'line_content': line_content,
            'line_opens': opens,
            'line_closes': closes,
            'line_balance': balance,
            'is_line_balanced': balance == 0
        })
    
    return nodes

def classify_issues(nodes, lines):
    """
    Classify parentheses issues to distinguish real problems from normal AST structure.
    
    FIXED: This function now correctly identifies when nodes should be balanced vs unbalanced.
    """
    real_issues = []
    likely_normal = []
    
    # HARDCODED: Define parent node types that should remain open for children
    parent_node_types = {
        'program', 'procedure', 'parameters', 'local_declarations', 'statements',
        'var_decl', 'assignment', 'case_stmt', 'if_stmt', 'while_stmt', 'for_stmt',
        'return_stmt', 'expression'
    }
    
    # HARDCODED: Define leaf node types that should be self-contained
    leaf_node_types = {
        'comment', 'nolist_directive', 'list_directive', 'page_directive',
        'parameter', 'var_spec', 'operator', 'system_function'
    }
    
    for node in nodes:
        if node['is_line_balanced']:
            # FIXED: Balanced nodes are generally OK and should not be "fixed"
            continue
            
        line_num = node['line']
        line_content = lines[line_num - 1] if line_num <= len(lines) else ""
        node_type = node['type']
        
        # FIXED: Check for leaf nodes first - these should always be balanced
        if node_type in leaf_node_types:
            # Comments and other leaf nodes should be self-contained and balanced
            # If they're unbalanced, it's definitely a real issue
            real_issues.append(node)
        
        # Check if this is a parent node that should remain open
        elif node_type in parent_node_types:
            # Parent nodes should typically have missing closes (remain open for children)
            # Only consider it a real issue if it has extra closes or seems complete
            if (node['line_balance'] < 0 or  # Extra closes - always wrong
                (':attrs' in line_content and line_content.rstrip().endswith(')'))):  # Complete with attrs
                real_issues.append(node)
            else:
                likely_normal.append(node)
        
        # HARDCODED: Special case for struct_decl
        elif node_type == 'struct_decl':
            # struct_decl should usually be self-contained if it has attrs
            if ':attrs' in line_content:
                real_issues.append(node)
            else:
                likely_normal.append(node)
        
        # HARDCODED: Special cases for global_statement and source_directive
        elif node_type in ['global_statement', 'source_directive']:
            has_attrs = ':attrs' in line_content
            has_malformed_attrs = ':attrs (' in line_content  # Should be :attrs {
            appears_incomplete = (
                line_content.rstrip().endswith(',') or
                not line_content.rstrip().endswith(')') or
                has_malformed_attrs
            )
            has_extra_parens = node['line_balance'] < 0
            
            # If it has extra closing parens, it's definitely a real issue
            if has_extra_parens:
                real_issues.append(node)
            # If it has attrs but appears malformed or incomplete, it's a real issue
            elif has_attrs and (appears_incomplete or has_malformed_attrs):
                real_issues.append(node)
            # If no attrs and unbalanced, likely multi-line
            elif not has_attrs:
                likely_normal.append(node)
            # If severely unbalanced (more than 1), likely a real issue
            elif abs(node['line_balance']) > 1:
                real_issues.append(node)
            else:
                likely_normal.append(node)
        
        else:
            # For unknown types, use heuristics but be conservative
            # FIXED: Don't assume everything needs fixing
            is_single_line_declaration = (
                ':attrs' in line_content and line_content.rstrip().endswith(')') or
                node_type == 'statement' and node['line_balance'] != 0
            )
            
            if is_single_line_declaration:
                real_issues.append(node)
            else:
                likely_normal.append(node)
    
    return real_issues, likely_normal

def get_line_analysis(content):
    """Get line-by-line parentheses analysis."""
    
    lines = content.split('\n')
    line_analysis = []
    
    for line_num, line in enumerate(lines, 1):
        opens, closes, balance = analyze_line_parentheses(line)
        
        line_analysis.append({
            'line_num': line_num,
            'content': line.strip(),
            'opens': opens,
            'closes': closes,
            'balance': balance,
            'is_balanced': balance == 0
        })
    
    return line_analysis

def fix_parentheses_issues(content, real_issues):
    """Fix parentheses issues in the content."""
    
    lines = content.split('\n')
    fixes_applied = []
    
    # Sort issues by line number in reverse order so line numbers stay valid during editing
    sorted_issues = sorted(real_issues, key=lambda x: x['line'], reverse=True)
    
    for issue in sorted_issues:
        line_num = issue['line']
        balance = issue['line_balance']
        
        if line_num > len(lines):
            continue
            
        original_line = lines[line_num - 1]
        
        if balance > 0:
            # Missing closing parentheses
            fixed_line = original_line + ')' * balance
            fix_type = f"Added {balance} closing parenthesis/parentheses"
        elif balance < 0:
            # Extra closing parentheses - remove from the end
            extra_closes = -balance
            temp_line = original_line.rstrip()
            while extra_closes > 0 and temp_line.endswith(')'):
                temp_line = temp_line[:-1].rstrip()
                extra_closes -= 1
            
            # Preserve any trailing whitespace structure
            trailing_whitespace = original_line[len(original_line.rstrip()):]
            fixed_line = temp_line + trailing_whitespace
            fix_type = f"Removed {-balance} extra closing parenthesis/parentheses"
        else:
            continue
        
        lines[line_num - 1] = fixed_line
        
        fixes_applied.append({
            'line': line_num,
            'node_type': issue['type'],
            'original': original_line.strip(),
            'fixed': fixed_line.strip(),
            'fix_type': fix_type,
            'balance_fixed': balance
        })
    
    return '\n'.join(lines), fixes_applied

def backup_file(filepath):
    """Create a backup of the original file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{filepath}.backup_{timestamp}"
    
    try:
        with open(filepath, 'r', encoding='utf-8') as original:
            content = original.read()
        
        with open(backup_path, 'w', encoding='utf-8') as backup:
            backup.write(content)
        
        return backup_path
    except Exception as e:
        print(f"Error creating backup: {e}")
        return None

def show_node_analysis(nodes, show_balanced=True):
    """Show analysis of detected AST nodes."""
    
    print("AST NODE ANALYSIS (LINE-BASED)")
    print("=" * 80)
    print(f"{'Line':<6} {'Type':<18} {'Opens':<6} {'Closes':<7} {'Balance':<8} {'Status'}")
    print("-" * 80)
    
    balanced_count = 0
    unbalanced_count = 0
    
    for node in nodes:
        if node['is_line_balanced']:
            balanced_count += 1
            status = "OK"
        else:
            unbalanced_count += 1
            if node['line_balance'] > 0:
                status = f"MISSING {node['line_balance']}"
            else:
                status = f"EXTRA {-node['line_balance']}"
        
        if show_balanced or not node['is_line_balanced']:
            print(f"{node['line']:<6} {node['type']:<18} {node['line_opens']:<6} {node['line_closes']:<7} {node['line_balance']:+8d} {status}")
    
    print("-" * 80)
    print(f"Balanced nodes: {balanced_count}")
    print(f"Unbalanced nodes: {unbalanced_count}")
    print(f"Total nodes analyzed: {len(nodes)}")

def show_classified_issues(real_issues, likely_normal):
    """Show classified issues."""
    
    print(f"\nISSUE CLASSIFICATION")
    print("=" * 60)
    print(f"Real Issues (need fixing): {len(real_issues)}")
    print(f"Likely Normal (multi-line AST): {len(likely_normal)}")
    
    if real_issues:
        print(f"\nREAL ISSUES THAT NEED FIXING:")
        print("-" * 40)
        for issue in real_issues:
            status = f"MISSING {issue['line_balance']}" if issue['line_balance'] > 0 else f"EXTRA {-issue['line_balance']}"
            print(f"Line {issue['line']}: {issue['type']} - {status}")
    
    if likely_normal and len(likely_normal) <= 10:
        print(f"\nLIKELY NORMAL (multi-line structures):")
        print("-" * 40)
        for issue in likely_normal:
            print(f"Line {issue['line']}: {issue['type']}")
    elif likely_normal:
        print(f"\nLIKELY NORMAL: {len(likely_normal)} multi-line AST structures")

def show_fixes_applied(fixes_applied):
    """Show what fixes were applied."""
    
    if not fixes_applied:
        print("\nNo fixes were applied.")
        return
    
    print(f"\nFIXES APPLIED ({len(fixes_applied)} changes)")
    print("=" * 80)
    
    for fix in fixes_applied:
        print(f"\nLine {fix['line']}: {fix['node_type']}")
        print(f"Fix: {fix['fix_type']}")
        print(f"Before: {fix['original']}")
        print(f"After:  {fix['fixed']}")

def show_line_analysis(line_analysis, unbalanced_only=False):
    """Show line-by-line analysis."""
    
    print("LINE-BY-LINE PARENTHESES ANALYSIS")
    print("=" * 70)
    print(f"{'Line':<6} {'Opens':<6} {'Closes':<7} {'Balance':<8} {'Content'}")
    print("-" * 70)
    
    total_unbalanced = 0
    
    for line_info in line_analysis:
        if unbalanced_only and line_info['is_balanced']:
            continue
        
        if not line_info['is_balanced']:
            total_unbalanced += 1
            status_marker = " *"
        else:
            status_marker = ""
        
        content_preview = line_info['content'][:45]
        if len(line_info['content']) > 45:
            content_preview += "..."
        
        print(f"{line_info['line_num']:<6} {line_info['opens']:<6} {line_info['closes']:<7} {line_info['balance']:+8d}{status_marker} {content_preview}")
    
    print("-" * 70)
    print(f"Unbalanced lines: {total_unbalanced}")

def show_summary(nodes, line_analysis, real_issues, likely_normal):
    """Show summary of analysis."""
    
    unbalanced_lines = [line for line in line_analysis if not line['is_balanced']]
    
    print(f"\nSUMMARY")
    print("=" * 40)
    print(f"Total AST nodes found: {len(nodes)}")
    print(f"Total lines analyzed: {len(line_analysis)}")
    print(f"Unbalanced lines: {len(unbalanced_lines)}")
    print(f"Real issues needing fixes: {len(real_issues)}")
    print(f"Likely normal multi-line: {len(likely_normal)}")

def main():
    parser = argparse.ArgumentParser(description='Enhanced AST parentheses validator and fixer')
    parser.add_argument('input_file', help='Input AST file')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--show-all', action='store_true', help='Show balanced nodes too')
    parser.add_argument('--line-by-line', action='store_true', help='Show line-by-line analysis')
    parser.add_argument('--unbalanced-only', action='store_true', help='Show only unbalanced items')
    parser.add_argument('--classify', action='store_true', help='Classify issues (real vs likely normal)')
    parser.add_argument('--summary', action='store_true', help='Show summary')
    parser.add_argument('--fix', action='store_true', help='Fix identified real issues')
    parser.add_argument('--output', help='Output file (default: overwrite input, use - for stdout)')
    parser.add_argument('--no-backup', action='store_true', help='Skip creating backup when fixing')
    args = parser.parse_args()

    # Read input
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return 1

    if args.verbose:
        print(f"Analyzing: {args.input_file}")
        print("=" * 50)

    # Get analysis
    nodes = detect_node_starts(content)
    line_analysis = get_line_analysis(content)
    lines = content.split('\n')
    real_issues, likely_normal = classify_issues(nodes, lines)
    
    if args.verbose:
        print(f"Found {len(nodes)} AST nodes")
        print(f"Identified {len(real_issues)} real issues")

    # Show requested analysis
    if args.line_by_line:
        show_line_analysis(line_analysis, args.unbalanced_only)
    elif args.classify:
        show_classified_issues(real_issues, likely_normal)
    elif args.summary:
        show_summary(nodes, line_analysis, real_issues, likely_normal)
    else:
        # Default: show node analysis with classification
        show_balanced = args.show_all and not args.unbalanced_only
        show_node_analysis(nodes, show_balanced)
        show_classified_issues(real_issues, likely_normal)
    
    # Apply fixes if requested
    if args.fix:
        if not real_issues:
            print("\nNo real issues to fix!")
            return 0
        
        print(f"\nAttempting to fix {len(real_issues)} real issues...")
        
        # Create backup unless disabled
        if not args.no_backup:
            backup_path = backup_file(args.input_file)
            if backup_path:
                print(f"Backup created: {backup_path}")
        
        # Apply fixes
        fixed_content, fixes_applied = fix_parentheses_issues(content, real_issues)
        
        # Determine output
        if args.output == '-':
            print("\n" + "="*60)
            print("FIXED CONTENT:")
            print("="*60)
            print(fixed_content)
        else:
            output_file = args.output or args.input_file
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(fixed_content)
                print(f"Fixed content written to: {output_file}")
            except Exception as e:
                print(f"Error writing fixed content: {e}")
                return 1
        
        show_fixes_applied(fixes_applied)
        print(f"\nSUCCESS: Fixed {len(fixes_applied)} real issues")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

class ASTNode:
    """
    Represents a single node in the Abstract Syntax Tree.
    
    Each node has a type, optional name, attributes, children, and original content.
    """
    
    def __init__(self, node_type: str, name: str = None, 
                 start_pos: int = 0, line_num: int = 0):
        self.node_type = node_type
        self.name = name
        self.attributes = {}
        self.children = []
        self.parent = None
        self.start_pos = start_pos
        self.end_pos = None
        self.line_num = line_num
        self.raw_content = ""
        
    def add_child(self, child: 'ASTNode'):
        """Add a child node and set its parent reference."""
        child.parent = self
        self.children.append(child)
        
    def find_children_by_type(self, node_type: str) -> List['ASTNode']:
        """Find all direct children of a specific type."""
        return [child for child in self.children if child.node_type == node_type]
    
    def find_descendants_by_type(self, node_type: str) -> List['ASTNode']:
        """Find all descendants (recursive) of a specific type."""
        results = []
        for child in self.children:
            if child.node_type == node_type:
                results.append(child)
            results.extend(child.find_descendants_by_type(node_type))
        return results
    
    def get_full_path(self) -> str:
        """Get the full path from root to this node."""
        path = []
        current = self
        while current:
            if current.name:
                path.append(f"{current.node_type}:{current.name}")
            else:
                path.append(current.node_type)
            current = current.parent
        return " -> ".join(reversed(path))
    
    def to_sexp(self, indent: int = 0) -> str:
        """
        Convert this node and its children back to S-expression format.
        
        Args:
            indent: Current indentation level for pretty printing
            
        Returns:
            String representation in S-expression format
        """
        indent_str = "  " * indent
        
        # Start with opening paren and node type
        result = f"({self.node_type}"
        
        # Add name if present
        if self.name:
            result += f" :name {self.name}"
        
        # Add other attributes
        for key, value in self.attributes.items():
            if key != 'name':  # Skip name as it's handled above
                if isinstance(value, str):
                    result += f" :{key} {value}"
                else:
                    result += f" :{key} {value}"
        
        # Add children
        if self.children:
            for child in self.children:
                result += "\n" + indent_str + "  " + child.to_sexp(indent + 1)
        
        result += ")"
        return result
    
    def __repr__(self):
        return f"ASTNode({self.node_type}, name={self.name}, children={len(self.children)})"


class ASTParser:
    """
    Parser that converts S-expression AST text into a tree of ASTNode objects.
    
    Handles nested parentheses, attributes, and maintains position information
    for each node.
    """
    
    def __init__(self, content: str):
        self.content = content
        self.pos = 0
        self.line_num = 1
        self.root = None
        
    def parse(self) -> ASTNode:
        """
        Parse the entire AST content into a tree structure.
        
        Returns:
            The root ASTNode of the parsed tree
        """
        self.root = self._parse_node()
        return self.root
    
    def _skip_whitespace(self):
        """Skip whitespace and update line number tracking."""
        while self.pos < len(self.content) and self.content[self.pos].isspace():
            if self.content[self.pos] == '\n':
                self.line_num += 1
            self.pos += 1
    
    def _peek_char(self) -> Optional[str]:
        """Peek at the current character without advancing position."""
        if self.pos >= len(self.content):
            return None
        return self.content[self.pos]
    
    def _consume_char(self) -> Optional[str]:
        """Consume and return the current character, advancing position."""
        if self.pos >= len(self.content):
            return None
        char = self.content[self.pos]
        self.pos += 1
        if char == '\n':
            self.line_num += 1
        return char
    
    def _parse_token(self) -> str:
        """
        Parse a single token (word, string, or complex value).
        
        Handles quoted strings, complex attribute values, and nested structures.
        """
        self._skip_whitespace()
        
        if self.pos >= len(self.content):
            return ""
        
        start_pos = self.pos
        
        # Handle quoted strings
        if self.content[self.pos] in '"\'':
            quote_char = self.content[self.pos]
            self.pos += 1
            while self.pos < len(self.content):
                if self.content[self.pos] == quote_char:
                    self.pos += 1
                    break
                elif self.content[self.pos] == '\\':
                    self.pos += 2  # Skip escaped character
                else:
                    self.pos += 1
            return self.content[start_pos:self.pos]
        
        # Handle complex values (like dictionaries or lists)
        if self.content[self.pos] in '{[':
            bracket_count = 0
            while self.pos < len(self.content):
                char = self.content[self.pos]
                if char in '{[(':
                    bracket_count += 1
                elif char in '}])':
                    bracket_count -= 1
                    if bracket_count == 0:
                        self.pos += 1
                        break
                self.pos += 1
            return self.content[start_pos:self.pos]
        
        # Handle regular tokens
        while (self.pos < len(self.content) and 
               not self.content[self.pos].isspace() and 
               self.content[self.pos] not in '()'):
            self.pos += 1
        
        return self.content[start_pos:self.pos]
    
    def _parse_node(self) -> Optional[ASTNode]:
        """
        Parse a single AST node and all its children.
        
        Returns:
            ASTNode object or None if no valid node found
        """
        self._skip_whitespace()
        
        # Must start with opening parenthesis
        if self.pos >= len(self.content) or self.content[self.pos] != '(':
            return None
        
        start_pos = self.pos
        start_line = self.line_num
        self._consume_char()  # Consume '('
        
        # Get node type
        node_type = self._parse_token()
        if not node_type:
            return None
        
        node = ASTNode(node_type, start_pos=start_pos, line_num=start_line)
        
        # Parse attributes and children
        while True:
            self._skip_whitespace()
            
            if self.pos >= len(self.content):
                break
                
            char = self._peek_char()
            
            if char == ')':
                # End of this node
                node.end_pos = self.pos
                self._consume_char()
                break
            elif char == '(':
                # Child node
                child = self._parse_node()
                if child:
                    node.add_child(child)
            else:
                # Attribute or value
                token = self._parse_token()
                if token.startswith(':'):
                    # This is an attribute
                    attr_name = token[1:]  # Remove ':'
                    attr_value = self._parse_token()
                    
                    # Special handling for :name attribute
                    if attr_name == 'name':
                        node.name = attr_value
                    
                    node.attributes[attr_name] = attr_value
        
        # Store raw content for this node
        if node.end_pos:
            node.raw_content = self.content[start_pos:node.end_pos + 1]
        
        return node


class ProcedureExtractor:
    """
    Utility class for extracting procedure information from AST trees.
    
    Provides methods to find procedures and extract their complete definitions.
    """
    
    def __init__(self, ast_root: ASTNode):
        self.ast_root = ast_root
        
    def get_all_procedures(self) -> List[ASTNode]:
        """
        Get all procedure nodes from the AST.
        
        Returns:
            List of ASTNode objects representing procedures
        """
        return self.ast_root.find_descendants_by_type('procedure')
    
    def get_procedure_info(self, procedure_node: ASTNode) -> Dict[str, Any]:
        """
        Extract detailed information about a procedure.
        
        Args:
            procedure_node: ASTNode representing a procedure
            
        Returns:
            Dictionary with procedure information including name, 
            parameters, local declarations, and statements
        """
        info = {
            'name': procedure_node.name,
            'attributes': procedure_node.attributes.copy(),
            'line_number': procedure_node.line_num,
            'parameters': [],
            'local_declarations': [],
            'statements': [],
            'sexp': procedure_node.to_sexp(),
            'raw_sexp': procedure_node.raw_content
        }
        
        # Extract parameters
        parameters_node = procedure_node.find_children_by_type('parameters')
        if parameters_node:
            params = parameters_node[0].find_children_by_type('parameter')
            for param in params:
                info['parameters'].append({
                    'name': param.name,
                    'attributes': param.attributes
                })
        
        # Extract local declarations
        local_decls_node = procedure_node.find_children_by_type('local_declarations')
        if local_decls_node:
            decls = local_decls_node[0].children
            for decl in decls:
                info['local_declarations'].append({
                    'type': decl.node_type,
                    'name': decl.name,
                    'attributes': decl.attributes
                })
        
        # Extract statements
        statements_node = procedure_node.find_children_by_type('statements')
        if statements_node:
            stmts = statements_node[0].children
            for stmt in stmts:
                info['statements'].append({
                    'type': stmt.node_type,
                    'name': stmt.name,
                    'attributes': stmt.attributes
                })
        
        return info
    
    def extract_procedure_sexps(self) -> Dict[str, str]:
        """
        Extract all procedures and their complete S-expressions.
        
        Returns:
            Dictionary mapping procedure names to their S-expression strings
        """
        procedures = self.get_all_procedures()
        result = {}
        
        for proc in procedures:
            proc_name = proc.name or f"unnamed_proc_line_{proc.line_num}"
            result[proc_name] = proc.raw_content or proc.to_sexp()
        
        return result
    
    def print_procedure_summary(self):
        """Print a summary of all procedures found in the AST."""
        procedures = self.get_all_procedures()
        
        print(f"PROCEDURE SUMMARY")
        print("=" * 60)
        print(f"Found {len(procedures)} procedures:")
        print()
        
        for i, proc in enumerate(procedures, 1):
            info = self.get_procedure_info(proc)
            print(f"{i}. Procedure: {info['name'] or 'unnamed'}")
            print(f"   Line: {info['line_number']}")
            print(f"   Parameters: {len(info['parameters'])}")
            print(f"   Local declarations: {len(info['local_declarations'])}")
            print(f"   Statements: {len(info['statements'])}")
            
            if info['attributes']:
                print(f"   Attributes: {info['attributes']}")
            
            print()


def main():
    parser = argparse.ArgumentParser(description='AST Tree Parser and Procedure Extractor')
    parser.add_argument('input_file', help='Input AST file')
    parser.add_argument('--procedures', action='store_true', help='Show procedure summary')
    parser.add_argument('--extract-sexps', action='store_true', help='Extract procedure S-expressions')
    parser.add_argument('--output-dir', help='Directory to save extracted procedures')
    parser.add_argument('--tree', action='store_true', help='Show AST tree structure')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()

    # Read input file
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return 1

    if args.verbose:
        print(f"Parsing AST from: {args.input_file}")
        print("=" * 50)

    # Parse AST
    try:
        ast_parser = ASTParser(content)
        root = ast_parser.parse()
        
        if not root:
            print("Error: Could not parse AST")
            return 1
            
    except Exception as e:
        print(f"Error parsing AST: {e}")
        return 1

    if args.verbose:
        print(f"Successfully parsed AST with root node: {root.node_type}")

    # Create procedure extractor
    extractor = ProcedureExtractor(root)

    # Show tree structure
    if args.tree:
        print("AST TREE STRUCTURE")
        print("=" * 40)
        print(root.to_sexp())
        print()

    # Show procedure summary
    if args.procedures:
        extractor.print_procedure_summary()

    # Extract S-expressions
    if args.extract_sexps:
        sexps = extractor.extract_procedure_sexps()
        
        print("EXTRACTED PROCEDURE S-EXPRESSIONS")
        print("=" * 60)
        
        for proc_name, sexp in sexps.items():
            print(f"\nProcedure: {proc_name}")
            print("-" * 40)
            print(sexp)
            
            # Save to file if output directory specified
            if args.output_dir:
                import os
                os.makedirs(args.output_dir, exist_ok=True)
                filename = f"{proc_name}.sexp"
                filepath = os.path.join(args.output_dir, filename)
                
                try:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(sexp)
                    print(f"Saved to: {filepath}")
                except Exception as e:
                    print(f"Error saving {filename}: {e}")

    # Default: show summary if no specific action requested
    if not any([args.procedures, args.extract_sexps, args.tree]):
        extractor.print_procedure_summary()

    return 0


if __name__ == "__main__":
    sys.exit(main())
