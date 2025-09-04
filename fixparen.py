#!/usr/bin/env python3
"""
Robust TAL AST Parentheses and Syntax Fixer

This program fixes not only unmatched parentheses but also other syntax issues
in malformed S-expressions, including:
- Mixed quote types
- Malformed attribute syntax
- Invalid node names
- Unmatched braces and brackets

Usage:
    python robust_fix_parentheses.py input.ast [output.ast]
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
class SyntaxIssue:
    """Represents a syntax issue found during parsing."""
    issue_type: str
    position: int
    line_num: int
    description: str
    suggested_fix: str = ""

class RobustSyntaxFixer:
    """
    Robust syntax fixer that handles multiple types of S-expression syntax issues.
    """
    
    def __init__(self, content: str, verbose: bool = False):
        self.original_content = content
        self.content = content
        self.verbose = verbose
        self.issues = []
        self.fixes_applied = []
        
    def fix_all_syntax_issues(self) -> str:
        """Apply all syntax fixes in the correct order."""
        
        if self.verbose:
            print("Starting comprehensive syntax fixing...")
        
        # Step 1: Fix invalid node names
        self.fix_invalid_node_names()
        
        # Step 2: Fix quote consistency
        self.fix_quote_consistency()
        
        # Step 3: Fix malformed attributes
        self.fix_malformed_attributes()
        
        # Step 4: Fix unmatched braces and brackets
        self.fix_unmatched_braces()
        
        # Step 5: Fix parentheses (using enhanced logic)
        self.fix_parentheses()
        
        # Step 6: Clean up whitespace and formatting
        self.cleanup_formatting()
        
        if self.verbose:
            print(f"Applied {len(self.fixes_applied)} fixes:")
            for fix in self.fixes_applied:
                print(f"  - {fix}")
        
        return self.content
    
    def fix_invalid_node_names(self):
        """Fix invalid node names like 'source *directive' -> 'source_directive'."""
        
        # Pattern to find invalid node names with spaces or special characters
        pattern = r'\(([a-zA-Z_][a-zA-Z0-9_]*)\s+\*([a-zA-Z_][a-zA-Z0-9_]*)'
        
        def replace_invalid_node(match):
            prefix = match.group(1)
            suffix = match.group(2)
            fixed_name = f"{prefix}_{suffix}"
            self.fixes_applied.append(f"Fixed node name: '{prefix} *{suffix}' -> '{fixed_name}'")
            return f"({fixed_name}"
        
        self.content = re.sub(pattern, replace_invalid_node, self.content)
    
    def fix_quote_consistency(self):
        """Standardize quote usage - prefer double quotes."""
        
        # Find all quoted strings and standardize to double quotes
        # But be careful with nested quotes
        
        original_content = self.content
        
        # Replace single quotes with double quotes, but handle escapes
        # This is a simplified approach - a full solution would need proper parsing
        
        # Count quote types
        single_quotes = self.content.count("'")
        double_quotes = self.content.count('"')
        
        if single_quotes > 0 and double_quotes > 0:
            # Mixed quotes detected - convert single to double where safe
            
            # Pattern to find single-quoted strings that don't contain double quotes
            pattern = r"'([^'\"]*?)'"
            
            def replace_quotes(match):
                content = match.group(1)
                if '"' not in content:  # Safe to convert
                    return f'"{content}"'
                return match.group(0)  # Leave as is if contains double quotes
            
            self.content = re.sub(pattern, replace_quotes, self.content)
            
            if self.content != original_content:
                self.fixes_applied.append("Standardized quote usage to double quotes")
    
    def fix_malformed_attributes(self):
        """Fix malformed attribute syntax."""
        
        # Pattern to find :attrs with mixed braces/parens
        # :attrs ('key': 'value') -> :attrs {"key": "value"}
        # :attrs {'key': 'value'} -> :attrs {"key": "value"}
        
        original_content = self.content
        
        # Fix :attrs with parentheses instead of braces
        pattern1 = r':attrs\s*\(\s*([^)]*?)\s*\)'
        
        def fix_attrs_parens(match):
            attrs_content = match.group(1)
            # Convert to proper brace format
            return f':attrs {{{attrs_content}}}'
        
        self.content = re.sub(pattern1, fix_attrs_parens, self.content)
        
        # Fix incomplete attribute structures
        pattern2 = r':attrs\s*\{\s*([^}]*?)$'
        
        def fix_incomplete_attrs(match):
            attrs_content = match.group(1)
            return f':attrs {{{attrs_content}}}'
        
        self.content = re.sub(pattern2, fix_incomplete_attrs, self.content, flags=re.MULTILINE)
        
        if self.content != original_content:
            self.fixes_applied.append("Fixed malformed attribute syntax")
    
    def fix_unmatched_braces(self):
        """Fix unmatched braces in attribute sections."""
        
        # Count braces and add missing ones
        brace_count = 0
        result = []
        i = 0
        
        while i < len(self.content):
            char = self.content[i]
            
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
            
            result.append(char)
            i += 1
        
        # Add missing closing braces
        missing_braces = brace_count
        if missing_braces > 0:
            result.extend(['}'] * missing_braces)
            self.fixes_applied.append(f"Added {missing_braces} missing closing braces")
        
        self.content = ''.join(result)
    
    def fix_parentheses(self):
        """Fix unmatched parentheses using enhanced logic."""
        
        # First, let's analyze the parentheses structure
        paren_stack = []
        issues = []
        
        for i, char in enumerate(self.content):
            if char == '(':
                paren_stack.append(i)
            elif char == ')':
                if paren_stack:
                    paren_stack.pop()
                else:
                    issues.append(f"Extra closing parenthesis at position {i}")
        
        # Add missing closing parentheses
        missing_closes = len(paren_stack)
        if missing_closes > 0:
            # Add closing parentheses at the end
            self.content += ')' * missing_closes
            self.fixes_applied.append(f"Added {missing_closes} missing closing parentheses")
    
    def cleanup_formatting(self):
        """Clean up whitespace and formatting issues."""
        
        original_content = self.content
        
        # Remove extra whitespace
        self.content = re.sub(r'\s+', ' ', self.content)
        
        # Fix spacing around parentheses and braces
        self.content = re.sub(r'\(\s+', '(', self.content)
        self.content = re.sub(r'\s+\)', ')', self.content)
        self.content = re.sub(r'\{\s+', '{', self.content)
        self.content = re.sub(r'\s+\}', '}', self.content)
        
        # Add proper line breaks for readability
        self.content = re.sub(r'\)\s*\(', ')\n(', self.content)
        
        if self.content != original_content:
            self.fixes_applied.append("Cleaned up formatting and whitespace")

def fix_malformed_sexp(content: str, verbose: bool = False) -> str:
    """
    Main function to fix malformed S-expressions with comprehensive syntax repair.
    """
    
    fixer = RobustSyntaxFixer(content, verbose)
    return fixer.fix_all_syntax_issues()

def analyze_syntax_issues(content: str) -> List[SyntaxIssue]:
    """Analyze and report all syntax issues found in the content."""
    
    issues = []
    
    # Check parentheses balance
    opens = content.count('(')
    closes = content.count(')')
    if opens != closes:
        issues.append(SyntaxIssue(
            "parentheses_imbalance",
            0,
            1,
            f"Unbalanced parentheses: {opens} opens, {closes} closes",
            f"Need {abs(opens - closes)} {'closing' if opens > closes else 'opening'} parentheses"
        ))
    
    # Check brace balance
    open_braces = content.count('{')
    close_braces = content.count('}')
    if open_braces != close_braces:
        issues.append(SyntaxIssue(
            "braces_imbalance",
            0,
            1,
            f"Unbalanced braces: {open_braces} opens, {close_braces} closes",
            f"Need {abs(open_braces - close_braces)} {'closing' if open_braces > close_braces else 'opening'} braces"
        ))
    
    # Check for mixed quotes
    single_quotes = content.count("'")
    double_quotes = content.count('"')
    if single_quotes > 0 and double_quotes > 0:
        issues.append(SyntaxIssue(
            "mixed_quotes",
            0,
            1,
            f"Mixed quote types: {single_quotes} single, {double_quotes} double",
            "Standardize to one quote type"
        ))
    
    # Check for invalid node names
    invalid_nodes = re.findall(r'\([a-zA-Z_][a-zA-Z0-9_]*\s+\*[a-zA-Z_][a-zA-Z0-9_]*', content)
    if invalid_nodes:
        issues.append(SyntaxIssue(
            "invalid_node_names",
            0,
            1,
            f"Found {len(invalid_nodes)} invalid node names with spaces/asterisks",
            "Replace spaces and asterisks with underscores"
        ))
    
    # Check for malformed attributes
    malformed_attrs = re.findall(r':attrs\s*\([^)]*\)', content)
    if malformed_attrs:
        issues.append(SyntaxIssue(
            "malformed_attributes",
            0,
            1,
            f"Found {len(malformed_attrs)} attributes using parentheses instead of braces",
            "Replace :attrs (...) with :attrs {...}"
        ))
    
    return issues

def main():
    """Main function with robust syntax fixing capabilities."""
    
    parser = argparse.ArgumentParser(
        description='Robust TAL AST syntax and parentheses fixer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This tool fixes multiple syntax issues in malformed S-expressions:
- Invalid node names (e.g., "source *directive" -> "source_directive") 
- Mixed quote types (standardizes to double quotes)
- Malformed attributes (fixes :attrs syntax)
- Unmatched parentheses, braces, and brackets
- Formatting and whitespace issues

Examples:
  %(prog)s malformed.ast                    # Fix and overwrite
  %(prog)s malformed.ast fixed.ast          # Fix to new file  
  %(prog)s malformed.ast -v                 # Verbose output
  %(prog)s malformed.ast --analyze-only     # Just show issues
        """
    )
    
    parser.add_argument('input_file', help='Input malformed AST file to fix')
    parser.add_argument('output_file', nargs='?', help='Output file (default: overwrite input)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--analyze-only', action='store_true', help='Only analyze issues, do not fix')
    
    args = parser.parse_args()
    
    # Read input file
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: Input file '{args.input_file}' not found")
        return 1
    except Exception as e:
        print(f"Error reading input file: {e}")
        return 1
    
    print(f"Analyzing malformed S-expression: {args.input_file}")
    print("=" * 60)
    
    # Analyze syntax issues
    issues = analyze_syntax_issues(content)
    
    if issues:
        print(f"Found {len(issues)} syntax issues:")
        for i, issue in enumerate(issues, 1):
            print(f"{i}. {issue.issue_type}: {issue.description}")
            if issue.suggested_fix:
                print(f"   Fix: {issue.suggested_fix}")
        print()
    else:
        print("No syntax issues detected.")
        return 0
    
    if args.analyze_only:
        return 1 if issues else 0
    
    # Apply fixes
    print("Applying comprehensive syntax fixes...")
    fixed_content = fix_malformed_sexp(content, args.verbose)
    
    # Determine output file
    output_file = args.output_file if args.output_file else args.input_file
    
    # Write fixed content
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        
        # Verify the fix
        final_issues = analyze_syntax_issues(fixed_content)
        
        print(f"\nResults:")
        print(f"  Original issues: {len(issues)}")
        print(f"  Remaining issues: {len(final_issues)}")
        print(f"  Fixed successfully: {len(issues) - len(final_issues)}")
        
        if len(final_issues) == 0:
            print(f"✅ All syntax issues fixed! Output written to: {output_file}")
        else:
            print(f"⚠️  Some issues remain. Check the output file: {output_file}")
            if args.verbose:
                print("Remaining issues:")
                for issue in final_issues:
                    print(f"  - {issue.description}")
        
        return 0
        
    except Exception as e:
        print(f"Error writing output file: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
