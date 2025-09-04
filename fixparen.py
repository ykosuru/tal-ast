#!/usr/bin/env python3
"""
Working TAL AST Fixer
"""

import sys
import argparse
import re

def find_malformed_sibling_patterns(content, verbose=False):
    """Find malformed sibling patterns that need fixing."""
    
    # Pattern: source_directive followed by global_statement with incomplete closing
    pattern = r'(\(source_directive\b[^)]*(?:\([^)]*\))*[^)]*)\s+(\(global_statement\b)'
    
    fixes = []
    matches = list(re.finditer(pattern, content))
    
    for match in matches:
        source_start = match.start(1)
        source_end = match.end(1)
        global_start = match.start(2)
        
        # Count how many opens we have in the source_directive content
        source_content = content[source_start:source_end]
        opens_in_source = source_content.count('(')
        closes_in_source = source_content.count(')')
        
        # We need enough closes to balance the source_directive
        needed_closes = opens_in_source - closes_in_source
        
        if needed_closes > 0:
            fixes.append({
                'position': global_start,
                'closes_needed': needed_closes,
                'description': f'Close source_directive before global_statement'
            })
            
            if verbose:
                print(f"Found malformed source_directive at {source_start}")
                print(f"  Opens in source: {opens_in_source}, Closes: {closes_in_source}")
                print(f"  Need to add {needed_closes} closes before global_statement at {global_start}")
    
    return fixes

def apply_fixes(content, fixes, verbose=False):
    """Apply the fixes by inserting closing parentheses."""
    
    if not fixes:
        return content
    
    # Sort fixes by position (right to left) to maintain correct positions
    fixes.sort(key=lambda x: x['position'], reverse=True)
    
    result = content
    for fix in fixes:
        pos = fix['position']
        closes = ')' * fix['closes_needed']
        result = result[:pos] + closes + result[pos:]
        
        if verbose:
            print(f"Applied fix at position {pos}: added {fix['closes_needed']} closes")
    
    return result

def balance_final_parentheses(content, verbose=False):
    """Add any remaining closing parentheses needed at the end."""
    
    opens = content.count('(')
    closes = content.count(')')
    
    if opens > closes:
        needed = opens - closes
        result = content + ')' * needed
        if verbose:
            print(f"Added {needed} final closing parentheses at end")
        return result
    
    return content

def fix_malformed_attributes(content, verbose=False):
    """Fix malformed attribute syntax."""
    
    original = content
    
    # Fix :attrs ( to :attrs {
    content = re.sub(r':attrs\s*\(\s*([^}]+)\}', r':attrs {\1}', content)
    content = re.sub(r':attrs\s*\(\s*([^)]+)\)', r':attrs {\1}', content)
    
    if verbose and content != original:
        print("Fixed malformed attribute syntax")
    
    return content

def main():
    parser = argparse.ArgumentParser(description='TAL AST parentheses fixer')
    parser.add_argument('input_file', help='Input AST file')
    parser.add_argument('-o', '--output', help='Output file')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()

    # Read input
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return 1

    if args.verbose:
        print(f"Processing: {args.input_file}")
        print("=" * 50)

    # Check initial state
    initial_opens = content.count('(')
    initial_closes = content.count(')')
    print(f"Initial: {initial_opens} opens, {initial_closes} closes")

    # Step 1: Fix malformed attributes
    if args.verbose:
        print("\nStep 1: Fixing malformed attributes...")
    content = fix_malformed_attributes(content, args.verbose)

    # Step 2: Find and fix malformed sibling patterns
    if args.verbose:
        print("\nStep 2: Finding malformed sibling patterns...")
    
    fixes = find_malformed_sibling_patterns(content, args.verbose)
    print(f"Found {len(fixes)} patterns to fix")

    # Step 3: Apply fixes
    if args.verbose:
        print("\nStep 3: Applying fixes...")
    fixed_content = apply_fixes(content, fixes, args.verbose)

    # Step 4: Balance any remaining parentheses
    if args.verbose:
        print("\nStep 4: Balancing remaining parentheses...")
    balanced_content = balance_final_parentheses(fixed_content, args.verbose)

    # Check final result
    final_opens = balanced_content.count('(')
    final_closes = balanced_content.count(')')
    print(f"\nFinal: {final_opens} opens, {final_closes} closes")
    
    if final_opens == final_closes:
        print("Successfully balanced!")
    else:
        print(f"Still unbalanced by {abs(final_opens - final_closes)}")

    # Write output
    output_file = args.output if args.output else f"{args.input_file}.fixed"
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(balanced_content)
        print(f"Fixed file written to: {output_file}")
        return 0
    except Exception as e:
        print(f"Error writing: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
