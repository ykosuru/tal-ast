#!/usr/bin/env python3
"""
LLM-Based Prompt Expander with One-Shot Learning
Refactored to use external call_llm function
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def call_llm(prompt: str) -> str:
    """
    Implement this function with your preferred LLM provider
    
    Args:
        prompt: The prompt to send to the LLM
        
    Returns:
        str: The LLM's response text
        
    Example implementations:
    
    # For Anthropic Claude:
    import anthropic
    client = anthropic.Anthropic(api_key="your-key")
    response = client.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text
    
    # For OpenAI GPT:
    import openai
    client = openai.OpenAI(api_key="your-key")
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000
    )
    return response.choices[0].message.content
    """
    """try:
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[...],
            max_tokens=2000,
            temperature=0.1
        )
        return response.choices[0].message.content.strip()
    except openai.AuthenticationError:
        # ... error handling

    return prompt
    """
    return prompt  # Placeholder implementation

class PaymentPromptExpander:
    """Payment systems prompt expander using LLM with one-shot learning"""
    
    def __init__(self):
        self.taxonomy_data = None
        self.one_shot_examples = []
    
    def add_one_shot_example(self, input_prompt: str, expected_output: str, description: str = ""):
        """Add a one-shot learning example"""
        example = {
            'input': input_prompt,
            'output': expected_output,
            'description': description,
            'added_at': datetime.now().isoformat()
        }
        self.one_shot_examples.append(example)
        logger.info(f"Added one-shot example: {description or input_prompt[:50]}")
    
    def load_one_shot_examples(self, examples_file: str = "one_shot_examples.json") -> bool:
        """Load one-shot examples from JSON file"""
        try:
            with open(examples_file, 'r', encoding='utf-8') as f:
                examples = json.load(f)
                self.one_shot_examples = examples
            logger.info(f"Loaded {len(self.one_shot_examples)} one-shot examples")
            return True
        except FileNotFoundError:
            logger.info(f"No existing examples file found: {examples_file}")
            return False
        except Exception as e:
            logger.error(f"Error loading examples: {e}")
            return False
    
    def save_one_shot_examples(self, examples_file: str = "one_shot_examples.json"):
        """Save one-shot examples to JSON file"""
        try:
            with open(examples_file, 'w', encoding='utf-8') as f:
                json.dump(self.one_shot_examples, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(self.one_shot_examples)} examples to {examples_file}")
        except Exception as e:
            logger.error(f"Error saving examples: {e}")
            raise
    
    def load_taxonomy(self, file_path: str = "keywords2.json") -> bool:
        """Load payment systems taxonomy from JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.taxonomy_data = json.load(f)
            logger.info(f"Successfully loaded {len(self.taxonomy_data)} taxonomy entries")
            return True
        except FileNotFoundError:
            logger.error(f"Taxonomy file not found: {file_path}")
            return False
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in taxonomy file: {e}")
            return False
        except Exception as e:
            logger.error(f"Error loading taxonomy: {e}")
            return False
    
    def _filter_relevant_taxonomy(self, query: str) -> List[Dict]:
        """Filter taxonomy entries relevant to the query"""
        if not self.taxonomy_data:
            return []
        
        query_lower = query.lower()
        relevant_entries = []
        
        for entry in self.taxonomy_data:
            keywords = entry.get('keywords', '').lower()
            related_keywords = entry.get('related_keywords', '').lower()
            capabilities = [cap.lower() for cap in entry.get('business_capability', [])]
            description = entry.get('description', '').lower()
            
            searchable_text = f"{keywords} {related_keywords} {' '.join(capabilities)} {description}"
            
            relevance_score = 0
            query_words = set(query_lower.split())
            
            for word in query_words:
                if word in searchable_text:
                    relevance_score += 1
            
            core_payment_terms = ['payment', 'wire', 'transfer', 'swift', 'fedwire', 'chips', 'charges', 'fees']
            if relevance_score > 0 or any(term in searchable_text for term in core_payment_terms):
                relevant_entries.append(entry)
        
        return relevant_entries[:50]
    
    def _build_prompt_expansion_request(self, user_query: str) -> str:
        """Build prompt for expanding the user's input query with one-shot learning examples"""
        
        relevant_entries = self._filter_relevant_taxonomy(user_query)
        
        one_shot_section = ""
        if self.one_shot_examples:
            one_shot_section = "\nONE-SHOT LEARNING EXAMPLES:\n"
            one_shot_section += "Learn from these examples of how to expand prompts in the payment systems domain:\n\n"
            
            for i, example in enumerate(self.one_shot_examples, 1):
                one_shot_section += f"EXAMPLE {i}:\n"
                one_shot_section += f"Input: \"{example['input']}\"\n"
                one_shot_section += f"Expected Output: \"{example['output']}\"\n"
                if example.get('description'):
                    one_shot_section += f"Description: {example['description']}\n"
                one_shot_section += "\n"
            
            one_shot_section += "Follow the same expansion style and depth shown in these examples.\n"
        
        prompt = f"""You are an expert Payment Systems Business Analyst. Your task is to EXPAND the user's input prompt with comprehensive context from the payment systems taxonomy.

USER'S ORIGINAL PROMPT: "{user_query}"

COMPREHENSIVE PAYMENT SYSTEMS TAXONOMY:
{json.dumps(relevant_entries, indent=2)}
{one_shot_section}
INSTRUCTIONS FOR PROMPT EXPANSION:

1. ANALYZE the user's original prompt to identify:
   - Primary business capability being requested
   - Payment domain (charges/pricing, cross-border, domestic, SWIFT, etc.)
   - Level of detail implied

2. EXPAND the original prompt by adding relevant context from the taxonomy:
   - Related business capabilities and processes
   - Specific systems and technologies involved
   - Integration points and dependencies
   - Regulatory and compliance considerations
   - Technical standards (Fed ISO, SWIFT MT, ISO 20022)

3. MAINTAIN the original intent while enriching with:
   - Specific terminology from the payment systems domain
   - Related workflows and operational processes
   - System integration requirements
   - Compliance and regulatory aspects

4. DOMAIN-SPECIFIC EXPANSION GUIDANCE:
   - CHARGES/PRICING: Include fee calculation, product codes, billing systems, Hogan integration
   - CROSS-BORDER: Add SWIFT messaging, correspondent banking, sanctions screening, Travel Rule
   - DOMESTIC: Include Fedwire, RTGS, ABA routing, same-day settlement
   - ISO 20022: Add message formats, migration considerations, structured data requirements

5. LEARNING FROM EXAMPLES:
   {"Apply the same expansion pattern, style, and depth demonstrated in the one-shot examples above." if self.one_shot_examples else "Use comprehensive domain expertise to create detailed expansions."}

OUTPUT FORMAT:
Return ONLY the expanded prompt - do not include explanations, just the enriched version of the user's original request that incorporates all relevant context from the payment systems taxonomy."""

        return prompt
    
    def expand_prompt(self, user_query: str) -> Dict[str, Any]:
        """Expand user's input prompt with comprehensive payment systems context"""
        if not self.taxonomy_data:
            raise ValueError("Taxonomy data not loaded. Call load_taxonomy() first.")
        
        prompt = self._build_prompt_expansion_request(user_query)
        
        try:
            expanded_prompt = call_llm(prompt).strip()
            
            return {
                'original_prompt': user_query,
                'expanded_prompt': expanded_prompt,
                'expansion_ratio': len(expanded_prompt.split()) / len(user_query.split()),
                'generated_at': datetime.now().isoformat(),
                'taxonomy_entries_used': len(self._filter_relevant_taxonomy(user_query)),
                'one_shot_examples_used': len(self.one_shot_examples)
            }
            
        except Exception as e:
            logger.error(f"Error expanding prompt: {e}")
            raise

def main():
    """Main application function with one-shot learning capabilities"""
    print("=" * 70)
    print("PROMPT EXPANDER WITH ONE-SHOT LEARNING")
    print("Transform simple queries into comprehensive payment systems prompts")
    print("=" * 70)
    
    try:
        # Initialize expander
        expander = PaymentPromptExpander()
        
        # Load taxonomy
        taxonomy_file = input("Taxonomy file path [keywords2.json]: ").strip() or "keywords2.json"
        if not expander.load_taxonomy(taxonomy_file):
            print("Failed to load taxonomy. Exiting.")
            return
        
        # Load existing one-shot examples
        expander.load_one_shot_examples()
        
        print(f"Loaded {len(expander.one_shot_examples)} one-shot training examples")
        print("Ready to expand prompts with payment systems context")
        
        print("\nExample input prompts:")
        print("  - create business requirements for payment charges")
        print("  - wire transfer processing")
        print("  - cross-border payments")
        
        print("\nCommands:")
        print("  'quit'     - Exit application")
        print("  'add'      - Add new one-shot training example")
        print("  'examples' - View current one-shot examples")
        print("  'save'     - Save one-shot examples to file")
        print("  'clear'    - Clear all one-shot examples")
        print("-" * 70)
        
        while True:
            try:
                query = input("\nEnter your prompt to expand (or command): ").strip()
                
                if not query:
                    continue
                
                if query.lower() == 'quit':
                    print("Goodbye!")
                    break
                
                if query.lower() == 'add':
                    print("\nAdding new one-shot training example:")
                    input_prompt = input("Input prompt: ").strip()
                    expected_output = input("Expected expanded output: ").strip()
                    description = input("Description (optional): ").strip()
                    
                    if input_prompt and expected_output:
                        expander.add_one_shot_example(input_prompt, expected_output, description)
                        print("Example added successfully!")
                    else:
                        print("Both input and output are required.")
                    continue
                
                if query.lower() == 'examples':
                    if expander.one_shot_examples:
                        print(f"\nCurrent one-shot examples ({len(expander.one_shot_examples)}):")
                        for i, example in enumerate(expander.one_shot_examples, 1):
                            print(f"\n--- Example {i} ---")
                            print(f"Input: {example['input']}")
                            print(f"Output: {example['output'][:100]}{'...' if len(example['output']) > 100 else ''}")
                            if example.get('description'):
                                print(f"Description: {example['description']}")
                    else:
                        print("No one-shot examples loaded.")
                    continue
                
                if query.lower() == 'save':
                    expander.save_one_shot_examples()
                    print("One-shot examples saved to file.")
                    continue
                
                if query.lower() == 'clear':
                    confirm = input("Clear all one-shot examples? (y/N): ").strip().lower()
                    if confirm == 'y':
                        expander.one_shot_examples = []
                        print("All examples cleared.")
                    continue
                
                # Expand prompt using one-shot learning
                print(f"\nExpanding prompt: '{query}'")
                print(f"Using {len(expander.one_shot_examples)} one-shot examples for learning...")
                
                result = expander.expand_prompt(query)
                
                print(f"\nORIGINAL PROMPT:")
                print(f"'{result['original_prompt']}'")
                print(f"\nEXPANDED PROMPT (with one-shot learning):")
                print("=" * 60)
                print(result['expanded_prompt'])
                print("=" * 60)
                print(f"\nExpansion ratio: {result['expansion_ratio']:.1f}x")
                print(f"Taxonomy entries analyzed: {result['taxonomy_entries_used']}")
                print(f"One-shot examples used: {result['one_shot_examples_used']}")
                
                # Offer to add this as a training example
                if len(expander.one_shot_examples) < 10:
                    add_example = input(f"\nAdd this as a one-shot training example? (y/N): ").strip().lower()
                    if add_example == 'y':
                        description = input("Description (optional): ").strip()
                        expander.add_one_shot_example(query, result['expanded_prompt'], description)
                        print("Added as training example!")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
                logger.exception("Application error")
                
    except Exception as e:
        print(f"Initialization error: {e}")

if __name__ == "__main__":
    print("SETUP INSTRUCTIONS:")
    print("1. Install OpenAI library: pip install openai")
    print("2. Set API key: export OPENAI_API_KEY='your-openai-api-key'")
    print("3. Ensure keywords2.json is in the current directory")
    print()
    
    main()

