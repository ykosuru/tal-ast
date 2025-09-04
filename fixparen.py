#!/usr/bin/env python3
"""
Flattened Stack-Based S-Expression Fixer

Strategy:
1. Remove all newlines and carriage returns to flatten content
2. Process character by character with a stack
3. Push nodes when encountering opening parentheses
4. Pop and close nodes when encountering closing parentheses
5. Insert closing parentheses before sibling nodes begin

This approach eliminates multi-line parsing complexity and focuses on
pure structural S-expression balance.
"""

import sys
import argparse
import re
from typing import List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class StackNode:
    """Represents a node on the parsing stack."""
    node_type: str
    start_pos: int
    depth: int
    has_content: bool = False

class FlattenedSExpParser:
    """
    Simple stack-based parser that works on flattened (no newlines) content.
    """
    
    def __init__(self, content: str, verbose: bool = False):
        self.original_content = content
        self.verbose = verbose
        # Step 1: Flatten content by removing all newlines and normalizing whitespace
        self.flattened_content = self._flatten_content(content)
        self.stack = []
        self.incomplete_nodes = []
        
    def _flatten_content(self, content: str) -> str:
        """Remove all newlines and normalize whitespace."""
        # Remove newlines and carriage returns
        flattened = content.replace('\n', ' ').replace('\r', ' ')
        
        # Normalize multiple spaces to single spaces
        flattened = re.sub(r'\s+', ' ', flattened)
        
        # Clean up spaces around parentheses and braces
        flattened = re.sub(r'\s*\(\s*', '(', flattened)
        flattened = re.sub(r'\s*\)\s*', ')', flattened)
        flattened = re.sub(r'\s*\{\s*', '{', flattened)
        flattened = re.sub(r'\s*\}\s*', '}', flattened)
        
        return flattened.strip()
    
    def parse_and_fix(self) -> str:
        """Parse the flattened content and fix parentheses balance."""
        
        if self.verbose:
            print(f"Original length: {len(self.original_content)} chars")
            print(f"Flattened length: {len(self.flattened_content)} chars")
            print("Flattened content preview:")
            print(self.flattened_content[:200] + "..." if len(self.flattened_content) > 200 else self.flattened_content)
            print()
        
        result = []
        i = 0
        in_string = False
        string_char = None
        
        while i < len(self.flattened_content):
            char = self.flattened_content[i]
            
            # Handle string boundaries
            if char in '"\'':
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char:
                    in_string = False
                    string_char = None
                result.append(char)
                i += 1
                continue
            
            # If we're inside a string, just append and continue
            if in_string:
                result.append(char)
                i += 1
                continue
            
            # Handle opening parentheses
            if char == '(':
                # Before opening a new node, check if we need to close incomplete siblings
                self._close_incomplete_siblings_before_new_node(result, i)
                
                result.append(char)
                
                # Read the node type
                node_type = self._read_node_type(i + 1)
                
                # Create stack entry
                stack_node = StackNode(
                    node_type=node_type,
                    start_pos=len(result) - 1,
                    depth=len(self.stack)
                )
                
                self.stack.append(stack_node)
                
                if self.verbose:
                    print(f"Opened node: {node_type} at depth {stack_node.depth}")
                
                i += 1
                
            # Handle closing parentheses
            elif char == ')':
                if self.stack:
                    closed_node = self.stack.pop()
                    if self.verbose:
                        print(f"Closed node: {closed_node.node_type} at depth {closed_node.depth}")
                else:
                    if self.verbose:
                        print("Warning: Extra closing parenthesis")
                
                result.append(char)
                i += 1
                
            # Handle other characters
            else:
                # Mark current node as having content
                if self.stack and char.strip():
                    self.stack[-1].has_content = True
                
                result.append(char)
                i += 1
        
        # Close any remaining incomplete nodes
        while self.stack:
            closed_node = self.stack.pop()
            result.append(')')
            if self.verbose:
                print(f"Auto-closed incomplete node: {closed_node.node_type}")
        
        fixed_content = ''.join(result)
        
        # Add back some formatting for readability
        formatted_content = self._add_basic_formatting(fixed_content)
        
        return formatted_content
    
    def _close_incomplete_siblings_before_new_node(self, result: List[str], current_pos: int):
        """Close incomplete sibling nodes before starting a new node at the same level."""
        
        if not self.stack:
            return
        
        # Look ahead to see what kind of node is starting
        upcoming_node_type = self._read_node_type(current_pos + 1)
        
        # If we're about to start a new top-level node or sibling, close incomplete nodes
        current_depth = len(self.stack)
        
        # Close nodes that should be siblings of the upcoming node
        while (self.stack and 
               self._should_close_before_sibling(self.stack[-1], upcoming_node_type, current_depth)):
            
            closed_node = self.stack.pop()
            result.append(')')
            
            if self.verbose:
                print(f"Auto-closed sibling: {closed_node.node_type} before {upcoming_node_type}")
    
    def _should_close_before_sibling(self, current_node: StackNode, upcoming_node_type: str, upcoming_depth: int) -> bool:
        """Determine if a node should be closed before a sibling starts."""
        
        # Close if we're at the same depth (siblings)
        if current_node.depth >= upcoming_depth:
            return True
        
        # Close certain node types before others (based on AST structure knowledge)
        if self._are_sibling_node_types(current_node.node_type, upcoming_node_type):
            return True
        
        # Close nodes that typically don't have children when another node starts
        non_container_types = {
            'comment', 'var_spec', 'parameter', 'system_function', 'operator',
            'const_decl', 'equ_decl', 'use_stmt', 'include_stmt', 'label',
            'goto_stmt', 'scan_stmt', 'rscan_stmt', 'stop_stmt', 'abort_stmt'
        }
        
        if current_node.node_type in non_container_types:
            return True
        
        return False
    
    def _are_sibling_node_types(self, current_type: str, upcoming_type: str) -> bool:
        """Check if two node types are typically siblings in the AST."""
        
        # Directive types are usually siblings
        directive_types = {
            'page_directive', 'section_directive', 'source_directive', 
            'list_directive', 'nolist_directive', 'symbols_directive'
        }
        
        if current_type in directive_types and upcoming_type in directive_types:
            return True
        
        # Declaration types are usually siblings
        declaration_types = {
            'var_decl', 'const_decl', 'literal_decl', 'struct_decl',
            'template_decl', 'external_decl', 'forward_decl'
        }
        
        if current_type in declaration_types and upcoming_type in declaration_types:
            return True
        
        # Statement types are usually siblings
        statement_types = {
            'statement', 'assignment', 'call_stmt', 'return_stmt',
            'if_stmt', 'while_stmt', 'for_stmt', 'case_stmt'
        }
        
        if current_type in statement_types and upcoming_type in statement_types:
            return True
        
        # Global content types are siblings
        global_types = {'global_statement', 'use_stmt', 'include_stmt'}
        
        if current_type in global_types and upcoming_type in global_types:
            return True
        
        return False
    
    def _read_node_type(self, start_pos: int) -> str:
        """Read the node type starting from the given position."""
        
        if start_pos >= len(self.flattened_content):
            return "unknown"
        
        # Skip whitespace
        i = start_pos
        while i < len(self.flattened_content) and self.flattened_content[i].isspace():
            i += 1
        
        if i >= len(self.flattened_content):
            return "unknown"
        
        # Read identifier
        node_type = ""
        while (i < len(self.flattened_content) and 
               self.flattened_content[i] not in ' \t(){}:"\'' and
               not self.flattened_content[i].isspace()):
            node_type += self.flattened_content[i]
            i += 1
        
        return node_type if node_type else "unknown"
    
    def _add_basic_formatting(self, content: str) -> str:
        """Add basic formatting to make the output more readable."""
        
        # Add line breaks between top-level S-expressions
        formatted = re.sub(r'\)\s*\(', ')\n(', content)
        
        # Add some indentation for nested structures (basic)
        lines = formatted.split('\n')
        result_lines = []
        
        for line in lines:
            if line.strip().startswith('('):
                # Simple indentation based on nesting depth
                depth = 0
                indent = 0
                for char in line:
                    if char == '(':
                        depth += 1
                    elif char == ')':
                        depth -= 1
                        if depth == 0:
                            break
                
                # Basic indentation (could be improved)
                result_lines.append('  ' * indent + line.strip())
            else:
                result_lines.append(line)
        
        return '\n'.join(result_lines)

def main():
    """
    Main function for flattened S-expression fixing with comprehensive command-line interface.
    
    This function provides a command-line interface for the S-expression fixer, handling
    argument parsing, file I/O, error handling, and result reporting.
    
    The tool implements a three-step process:
    1. Read and validate input file
    2. Apply the flattened stack-based parsing and fixing algorithm
    3. Write results and provide verification feedback
    
    Returns:
        Exit code: 0 for success, 1 for failure
    """
    
    # Set up command-line argument parsing with detailed help
    parser = argparse.ArgumentParser(
        description='Flattened stack-based S-expression fixer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Algorithm Overview:
This tool uses a simplified approach that eliminates multi-line parsing complexity:

1. FLATTEN: Remove all newlines/carriage returns from input content
2. TOKENIZE: Process character by character with string boundary awareness  
3. STACK-PARSE: Use stack to track open nodes, push on '(', pop on ')'
4. SIBLING-CLOSE: Automatically close incomplete nodes before siblings begin
5. AUTO-COMPLETE: Close any remaining incomplete nodes at end of input
6. FORMAT: Add basic formatting for readability

Key Benefits:
- Eliminates confusion from multi-line structures and varied indentation
- Simple character-by-character processing is predictable and reliable  
- Stack-based parsing naturally handles arbitrary nesting depth
- Sibling detection prevents AST nodes from bleeding into each other
- Respects string boundaries to avoid breaking quoted content

Examples:
  %(prog)s input.ast                    # Fix and overwrite input file
  %(prog)s input.ast -o output.ast      # Fix to new output file
  %(prog)s input.ast -v                 # Show verbose parsing steps and decisions
        """
    )
    
    # Define command-line arguments
    parser.add_argument('input_file', 
                       help='Input S-expression file to be fixed')
    parser.add_argument('-o', '--output', 
                       help='Output file (default: overwrite input file)')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Show verbose parsing output including stack operations')
    
    # Parse command-line arguments
    args = parser.parse_args()
    
    # Step 1: Read and validate input file
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: File '{args.input_file}' not found")
        return 1
    except PermissionError:
        print(f"Error: Permission denied reading '{args.input_file}'")
        return 1
    except Exception as e:
        print(f"Error reading file: {e}")
        return 1
    
    # Provide initial status if verbose
    if args.verbose:
        print(f"Processing: {args.input_file}")
        print("=" * 50)
        
        # Show initial parentheses count for comparison
        initial_opens = content.count('(')
        initial_closes = content.count(')')
        print(f"Initial state: {initial_opens} opens, {initial_closes} closes")
        print(f"Imbalance: {initial_opens - initial_closes} missing closes")
        print()
    
    # Step 2: Create parser instance and apply fixing algorithm
    parser_instance = FlattenedSExpParser(content, args.verbose)
    fixed_content = parser_instance.parse_and_fix()
    
    # Step 3: Verify the fix by checking final parentheses balance
    final_opens = fixed_content.count('(')
    final_closes = fixed_content.count(')')
    
    print(f"Final parentheses balance: {final_opens} opens, {final_closes} closes")
    
    # Report success or remaining issues
    if final_opens == final_closes:
        print("✅ Successfully balanced all parentheses!")
    else:
        print(f"⚠️  Still unbalanced by {abs(final_opens - final_closes)} parentheses")
        print("This may indicate malformed quoted strings or other structural issues.")
        return 1
    
    # Step 4: Write results to output file
    output_file = args.output if args.output else args.input_file
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        
        print(f"Fixed content written to: {output_file}")
        
        # Provide summary statistics if verbose
        if args.verbose:
            print(f"\nSummary:")
            print(f"  Input file: {args.input_file}")
            print(f"  Output file: {output_file}")
            print(f"  Original size: {len(content)} characters")
            print(f"  Fixed size: {len(fixed_content)} characters")
            print(f"  Parentheses added: {final_closes - content.count(')')}")
        
        return 0
        
    except PermissionError:
        print(f"Error: Permission denied writing to '{output_file}'")
        return 1
    except Exception as e:
        print(f"Error writing output: {e}")
        return 1

# Entry point - run main function if script is executed directly
if __name__ == "__main__":
    exit(main())
