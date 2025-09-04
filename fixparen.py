#!/usr/bin/env python3
"""
Enhanced Hybrid AST Validator: Analyzes and optionally fixes parentheses issues.
"""

import sys
import argparse
import re
import os
from datetime import datetime

def get_all_node_types():
    """Return list of all possible AST node types."""
    return [
        'program', 'comment', 'global_statement', 'source_directive', 'page_directive',
        'list_directive', 'nolist_directive', 'procedure', 'var_decl', 
        'struct_decl', 'const_decl', 'literal_decl', 'parameter', 'parameters',
        'local_declarations', 'statements', 'assignment', 'return_stmt', 
        'if_stmt', 'while_stmt', 'for_stmt', 'case_stmt', 'statement', 
        'var_spec', 'expression', 'operator', 'system_function'
    ]

def analyze_line_parentheses(line):
    """Analyze parentheses in a single line."""
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

def classify_issues(nodes, lines):
    """Classify issues to distinguish real problems from normal AST structure."""
    
    real_issues = []
    likely_normal = []
    
    # Define parent node types that should remain open for children
    parent_node_types = {
        'program', 'procedure', 'parameters', 'local_declarations', 'statements',
        'var_decl', 'assignment', 'case_stmt', 'if_stmt', 'while_stmt', 'for_stmt',
        'return_stmt', 'expression'
    }
    
    # Define leaf node types that should be self-contained
    leaf_node_types = {
        'comment', 'nolist_directive', 'list_directive', 'page_directive',
        'parameter', 'var_spec', 'operator', 'system_function'
    }
    
    for node in nodes:
        if node['is_line_balanced']:
            continue
            
        line_num = node['line']
        line_content = lines[line_num - 1] if line_num <= len(lines) else ""
        node_type = node['type']
        
        # Check if this is a parent node that should remain open
        if node_type in parent_node_types:
            # Parent nodes should typically have missing closes (remain open for children)
            # Only consider it a real issue if it has extra closes or seems complete
            if (node['line_balance'] < 0 or  # Extra closes
                (':attrs' in line_content and line_content.rstrip().endswith(')'))):  # Complete with attrs
                real_issues.append(node)
            else:
                likely_normal.append(node)
        
        # Check if this is a leaf node that should be self-contained
        elif node_type in leaf_node_types:
            # Leaf nodes should be balanced on their own line
            real_issues.append(node)
        
        # Special cases for specific node types
        elif node_type == 'struct_decl':
            # struct_decl should usually be self-contained if it has attrs
            if ':attrs' in line_content:
                real_issues.append(node)
            else:
                likely_normal.append(node)
        
        elif node_type in ['global_statement', 'source_directive']:
            # These can be either self-contained or multi-line
            # Check for specific malformed patterns
            has_attrs = ':attrs' in line_content
            has_malformed_attrs = ':attrs (' in line_content  # Should be :attrs {
            appears_incomplete = (
                line_content.rstrip().endswith(',') or  # Ends with comma
                not line_content.rstrip().endswith(')') or  # Doesn't end with closing paren
                has_malformed_attrs
            )
            has_extra_parens = node['line_balance'] < 0  # More closes than opens
            
            # If it has extra closing parens, it's definitely a real issue
            if has_extra_parens:
                real_issues.append(node)
            # If it has attrs but appears malformed or incomplete, it's a real issue
            elif has_attrs and (appears_incomplete or has_malformed_attrs):
                real_issues.append(node)
            # If no attrs and unbalanced, likely multi-line
            elif not has_attrs:
                likely_normal.append(node)
            # If complex nesting but appears complete, check balance more carefully
            else:
                # If severely unbalanced (more than 1), likely a real issue
                if abs(node['line_balance']) > 1:
                    real_issues.append(node)
                else:
                    likely_normal.append(node)
        
        else:
            # For unknown types, use heuristics
            is_single_line_declaration = (
                # Has attributes and appears complete
                (':attrs' in line_content and line_content.rstrip().endswith(')')) or
                # Simple statement types
                node_type == 'statement' or
                # Contains both opening and closing but unbalanced
                ('(' in line_content and ')' in line_content and node['line_balance'] != 0)
            )
            
            if is_single_line_declaration:
                real_issues.append(node)
            else:
                likely_normal.append(node)
    
    return real_issues, likely_normal

def fix_parentheses_issues(content, real_issues):
    """
    Fix parentheses issues in the content by adding or removing parentheses.
    
    This function modifies lines to balance parentheses based on the detected issues.
    It processes issues in reverse line order to maintain line number validity during editing.
    
    HARDCODED FIXING LOGIC:
    - Positive balance: Add ')' characters to the end of the line
    - Negative balance: Remove ')' characters from the end of the line
    - Preserves trailing whitespace structure
    
    Args:
        content (str): Original file content
        real_issues (list): List of classified real issues to fix
        
    Returns:
        tuple: (fixed_content, fixes_applied) where fixes_applied contains details
    """
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
            # HARDCODED FIX: Missing closing parentheses - add them to the end
            fixed_line = original_line + ')' * balance
            fix_type = f"Added {balance} closing parenthesis/parentheses"
        elif balance < 0:
            # HARDCODED FIX: Extra closing parentheses - remove from the end
            extra_closes = -balance
            # Remove closing parens from the end, preserving structure
            temp_line = original_line.rstrip()
            while extra_closes > 0 and temp_line.endswith(')'):
                temp_line = temp_line[:-1].rstrip()
                extra_closes -= 1
            
            # Preserve any trailing whitespace structure
            trailing_whitespace = original_line[len(original_line.rstrip()):]
            fixed_line = temp_line + trailing_whitespace
            fix_type = f"Removed {-balance} extra closing parenthesis/parentheses"
        else:
            continue  # Shouldn't happen as we filtered balanced nodes
        
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
    print(f"Real Issues (likely problems): {len(real_issues)}")
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
        print(f"\nLIKELY NORMAL: {len(likely_normal)} multi-line AST structures (use --show-all to see details)")

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
    
    if real_issues:
        print(f"\nReal issues by type:")
        issue_types = {}
        for issue in real_issues:
            node_type = issue['type']
            issue_types[node_type] = issue_types.get(node_type, 0) + 1
        
        for node_type, count in sorted(issue_types.items()):
            print(f"  {node_type}: {count}")

def main():
    parser = argparse.ArgumentParser(description='Enhanced AST parentheses validator and fixer')
    parser.add_argument('input_file', help='Input AST file')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--show-all', action='store_true', help='Show balanced nodes too')
    parser.add_argument('--line-by-line', action='store_true', help='Show line-by-line analysis')
    parser.add_argument('--unbalanced-only', action='store_true', help='Show only unbalanced items')
    parser.add_argument('--classify', action='store_true', help='Classify issues (real vs likely normal)')
    parser.add_argument('--summary', action='store_true', help='Show summary')
    parser.add_argument('--fix', action='store_true', help='Fix identified issues')
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
        print(f"Analyzed {len(line_analysis)} lines")
        print(f"Identified {len(real_issues)} real issues")

    # Show requested analysis
    if args.line_by_line:
        show_line_analysis(line_analysis, args.unbalanced_only)
    elif args.classify:
        show_classified_issues(real_issues, likely_normal)
    elif args.summary:
        show_summary(nodes, line_analysis, real_issues, likely_normal)
    else:
        # Default: show node analysis
        show_balanced = args.show_all and not args.unbalanced_only
        show_node_analysis(nodes, show_balanced)
        
        if not args.unbalanced_only:
            show_classified_issues(real_issues, likely_normal)
    
    # Apply fixes if requested
    if args.fix:
        if not real_issues:
            print("\nNo real issues to fix!")
            return 0
        
        print(f"\nAttempting to fix {len(real_issues)} issues...")
        
        # Create backup unless disabled
        if not args.no_backup:
            backup_path = backup_file(args.input_file)
            if backup_path:
                print(f"Backup created: {backup_path}")
            else:
                print("Warning: Could not create backup!")
                response = input("Continue without backup? (y/N): ")
                if response.lower() != 'y':
                    return 1
        
        # Apply fixes
        fixed_content, fixes_applied = fix_parentheses_issues(content, real_issues)
        
        # Determine output
        if args.output == '-':
            # Output to stdout
            print("\n" + "="*60)
            print("FIXED CONTENT:")
            print("="*60)
            print(fixed_content)
        else:
            # Write to file
            output_file = args.output or args.input_file
            
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(fixed_content)
                print(f"Fixed content written to: {output_file}")
            except Exception as e:
                print(f"Error writing fixed content: {e}")
                return 1
        
        # Show what was fixed
        show_fixes_applied(fixes_applied)
        
        print(f"\nSUCCESS: Fixed {len(fixes_applied)} issues")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

def get_all_node_types():
    """Return list of all possible AST node types."""
    return [
        'program', 'comment', 'global_statement', 'source_directive', 'page_directive',
        'list_directive', 'nolist_directive', 'procedure', 'var_decl', 
        'struct_decl', 'const_decl', 'literal_decl', 'parameter', 'parameters',
        'local_declarations', 'statements', 'assignment', 'return_stmt', 
        'if_stmt', 'while_stmt', 'for_stmt', 'case_stmt', 'statement', 
        'var_spec', 'expression', 'operator', 'system_function'
    ]

def analyze_line_parentheses(line):
    """Analyze parentheses in a single line."""
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

def show_unbalanced_details(nodes, line_analysis):
    """Show detailed information about unbalanced nodes."""
    
    unbalanced_nodes = [node for node in nodes if not node['is_line_balanced']]
    
    if not unbalanced_nodes:
        print("\nNo unbalanced AST nodes found!")
        return
    
    print(f"\nDETAILED VIEW OF {len(unbalanced_nodes)} UNBALANCED AST NODES")
    print("=" * 80)
    
    for node in unbalanced_nodes:
        print(f"\nLine {node['line']}: {node['type']} (balance: {node['line_balance']:+d})")
        print("-" * 50)
        
        # Show the content
        content = node['line_content']
        if len(content) > 100:
            print(f"Content: {content[:100]}...")
        else:
            print(f"Content: {content}")
        
        # Show what's needed to fix
        if node['line_balance'] > 0:
            print(f"ISSUE: Missing {node['line_balance']} closing parenthesis/parentheses")
            print(f"FIX: Add {node['line_balance']} ')' at the end of this line")
        else:
            print(f"ISSUE: Extra {-node['line_balance']} closing parenthesis/parentheses")
            print(f"FIX: Remove {-node['line_balance']} ')' from this line")

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

def show_summary(nodes, line_analysis):
    """Show summary of analysis."""
    
    unbalanced_nodes = [node for node in nodes if not node['is_line_balanced']]
    unbalanced_lines = [line for line in line_analysis if not line['is_balanced']]
    
    print(f"\nSUMMARY")
    print("=" * 40)
    print(f"Total AST nodes found: {len(nodes)}")
    print(f"Unbalanced AST nodes: {len(unbalanced_nodes)}")
    print(f"Total lines analyzed: {len(line_analysis)}")
    print(f"Unbalanced lines: {len(unbalanced_lines)}")
    
    if unbalanced_nodes:
        print(f"\nUnbalanced AST nodes:")
        for node in unbalanced_nodes:
            print(f"  Line {node['line']}: {node['type']} (balance: {node['line_balance']:+d})")

def main():
    parser = argparse.ArgumentParser(description='Hybrid AST parentheses validator')
    parser.add_argument('input_file', help='Input AST file')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--show-all', action='store_true', help='Show balanced nodes too')
    parser.add_argument('--line-by-line', action='store_true', help='Show line-by-line analysis')
    parser.add_argument('--unbalanced-only', action='store_true', help='Show only unbalanced items')
    parser.add_argument('--nodes-only', action='store_true', help='Show only AST node analysis')
    parser.add_argument('--summary', action='store_true', help='Show summary')
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
    
    if args.verbose:
        print(f"Found {len(nodes)} AST nodes")
        print(f"Analyzed {len(line_analysis)} lines")

    # Show requested analysis
    if args.line_by_line:
        show_line_analysis(line_analysis, args.unbalanced_only)
    elif args.nodes_only:
        show_balanced = args.show_all and not args.unbalanced_only
        show_node_analysis(nodes, show_balanced)
        show_unbalanced_details(nodes, line_analysis)
    elif args.summary:
        show_summary(nodes, line_analysis)
    else:
        # Default: show node analysis
        show_balanced = args.show_all and not args.unbalanced_only
        show_node_analysis(nodes, show_balanced)
        show_unbalanced_details(nodes, line_analysis)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
