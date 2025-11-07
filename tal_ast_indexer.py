"""
TAL Code Indexer - Robust indexing and search system for TAL AST

This module builds multiple indexes on top of TAL AST structures:
1. Structural Index: Procedures, functions, variables, calls
2. Semantic Index: Business capability mappings
3. TF-IDF Index: Keyword-based code search
4. Call Graph: Function dependencies and relationships

Key Features:
- Resilient to malformed AST nodes (handles missing brackets, incomplete nodes)
- Multi-level indexing for different query types
- Business capability mapping (TAL -> business functions)
- TF-IDF vector search for keyword matching
- Graph-based relationship tracking
"""

import re
import json
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import math
import pickle
from pathlib import Path

# Import the TAL parsers (assuming they're available)
try:
    import tal_proc_parser
except ImportError:
    print("Warning: tal_proc_parser not found, using fallback types")
    tal_proc_parser = None

@dataclass
class CodeElement:
    """
    Represents an indexed code element from the AST.
    Resilient to missing or malformed data.
    """
    element_type: str  # 'procedure', 'variable', 'call', 'system_function', etc.
    name: str
    file_path: str
    line_number: int = 0
    
    # Optional fields (may be None for malformed nodes)
    return_type: Optional[str] = None
    parameters: List[str] = field(default_factory=list)
    local_variables: List[str] = field(default_factory=list)
    calls: List[str] = field(default_factory=list)
    called_by: List[str] = field(default_factory=list)
    
    # Semantic mappings
    business_capabilities: Set[str] = field(default_factory=set)
    keywords: Set[str] = field(default_factory=set)
    
    # Raw content for TF-IDF
    content: str = ""
    
    # Source node reference (for debugging)
    source_node: Optional[Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'element_type': self.element_type,
            'name': self.name,
            'file_path': self.file_path,
            'line_number': self.line_number,
            'return_type': self.return_type,
            'parameters': self.parameters,
            'local_variables': self.local_variables,
            'calls': self.calls,
            'called_by': self.called_by,
            'business_capabilities': list(self.business_capabilities),
            'keywords': list(self.keywords),
            'content': self.content
        }

@dataclass
class BusinessCapability:
    """
    Represents a business capability that code implements.
    """
    capability_id: str
    name: str
    description: str
    keywords: Set[str] = field(default_factory=set)
    related_capabilities: Set[str] = field(default_factory=set)
    implementing_functions: Set[str] = field(default_factory=set)

class TALCodeIndexer:
    """
    Robust indexer for TAL code AST with error resilience.
    
    Builds multiple indexes:
    - Element index: All code elements by name
    - File index: Elements grouped by file
    - Type index: Elements grouped by type
    - Call graph: Function call relationships
    - Capability index: Business capability mappings
    - TF-IDF index: Document vectors for search
    """
    
    def __init__(self):
        """Initialize the indexer with empty indexes."""
        # Core indexes
        self.elements: Dict[str, CodeElement] = {}  # name -> element
        self.file_index: Dict[str, List[CodeElement]] = defaultdict(list)
        self.type_index: Dict[str, List[CodeElement]] = defaultdict(list)
        
        # Relationship indexes
        self.call_graph: Dict[str, Set[str]] = defaultdict(set)  # caller -> callees
        self.reverse_call_graph: Dict[str, Set[str]] = defaultdict(set)  # callee -> callers
        
        # Business capability mappings
        self.capabilities: Dict[str, BusinessCapability] = {}
        self.capability_patterns: Dict[str, List[str]] = {}  # keyword -> capability_ids
        
        # TF-IDF components
        self.vocabulary: Set[str] = set()
        self.document_frequencies: Counter = Counter()
        self.tf_idf_vectors: Dict[str, Dict[str, float]] = {}  # element_name -> {term: score}
        self.idf_scores: Dict[str, float] = {}
        
        # Statistics
        self.stats = {
            'total_elements': 0,
            'total_procedures': 0,
            'total_variables': 0,
            'total_calls': 0,
            'files_indexed': 0,
            'malformed_nodes_recovered': 0
        }
    
    def index_ast(self, ast_node, file_path: str) -> None:
        """
        Index a complete AST from a TAL file.
        
        Args:
            ast_node: Root AST node (TALNode from parser)
            file_path: Source file path for reference
        """
        print(f"Indexing file: {file_path}")
        
        try:
            # Traverse AST and extract elements
            self._traverse_ast(ast_node, file_path)
            
            # Build TF-IDF vectors for all elements
            self._build_tfidf_index()
            
            self.stats['files_indexed'] += 1
            print(f"  Indexed {len(self.elements)} total elements")
            print(f"  Recovered from {self.stats['malformed_nodes_recovered']} malformed nodes")
            
        except Exception as e:
            print(f"Error indexing {file_path}: {e}")
            # Don't fail completely - continue with partial index
    
    def _traverse_ast(self, node, file_path: str, parent_name: str = None) -> None:
        """
        Recursively traverse AST and extract indexable elements.
        Resilient to malformed nodes.
        
        Args:
            node: Current AST node
            file_path: Source file path
            parent_name: Name of parent element (for scoping)
        """
        if node is None:
            return
        
        try:
            # Extract node properties safely
            node_type = self._safe_get_attr(node, 'type', 'unknown')
            node_name = self._safe_get_attr(node, 'name', '')
            node_value = self._safe_get_attr(node, 'value', '')
            node_location = self._safe_get_attr(node, 'location', None)
            node_attributes = self._safe_get_attr(node, 'attributes', {})
            node_children = self._safe_get_attr(node, 'children', [])
            
            # Get line number safely
            line_num = 0
            if node_location and hasattr(node_location, 'line'):
                line_num = node_location.line
            
            # Process different node types
            if node_type == 'procedure':
                self._index_procedure(node, file_path, node_name, line_num, 
                                    node_attributes, node_children)
            
            elif node_type == 'subproc':
                self._index_subproc(node, file_path, node_name, line_num,
                                   node_attributes, node_children, parent_name)
            
            elif node_type in ['var_decl', 'variable']:
                self._index_variable(node, file_path, line_num, 
                                   node_attributes, parent_name)
            
            elif node_type in ['call_stmt', 'system_function_call']:
                self._index_call(node, file_path, line_num, 
                               node_attributes, parent_name)
            
            elif node_type == 'assignment':
                self._index_assignment(node, file_path, line_num,
                                     node_attributes, parent_name)
            
            # Recursively process children
            if node_children:
                for child in node_children:
                    try:
                        self._traverse_ast(child, file_path, 
                                         parent_name or node_name)
                    except Exception as e:
                        self.stats['malformed_nodes_recovered'] += 1
                        # Continue with next child
                        continue
                        
        except Exception as e:
            # Node is malformed, but continue processing
            self.stats['malformed_nodes_recovered'] += 1
    
    def _safe_get_attr(self, obj, attr: str, default=None):
        """Safely get attribute from object, return default if missing."""
        try:
            return getattr(obj, attr, default)
        except Exception:
            return default
    
    def _index_procedure(self, node, file_path: str, name: str, 
                        line_num: int, attributes: Dict, children: List) -> None:
        """Index a procedure/function."""
        if not name:
            name = f"unnamed_proc_{line_num}"
        
        element = CodeElement(
            element_type='procedure',
            name=name,
            file_path=file_path,
            line_number=line_num,
            return_type=attributes.get('return_type'),
            source_node=node
        )
        
        # Extract parameters
        for child in children:
            try:
                if self._safe_get_attr(child, 'type') == 'parameters':
                    for param in self._safe_get_attr(child, 'children', []):
                        param_name = self._safe_get_attr(param, 'name', '')
                        if param_name:
                            element.parameters.append(param_name)
            except Exception:
                continue
        
        # Build content for TF-IDF
        content_parts = [name]
        if element.return_type:
            content_parts.append(element.return_type)
        content_parts.extend(element.parameters)
        
        # Extract keywords from comments and attributes
        content_parts.extend(self._extract_keywords_from_node(node))
        
        element.content = ' '.join(content_parts)
        element.keywords = set(self._tokenize(element.content))
        
        # Store element
        self.elements[name] = element
        self.file_index[file_path].append(element)
        self.type_index['procedure'].append(element)
        self.stats['total_procedures'] += 1
        self.stats['total_elements'] += 1
    
    def _index_subproc(self, node, file_path: str, name: str, line_num: int,
                      attributes: Dict, children: List, parent_name: str) -> None:
        """Index a subprocedure."""
        if not name:
            name = f"unnamed_subproc_{line_num}"
        
        # Qualify name with parent if available
        if parent_name:
            qualified_name = f"{parent_name}.{name}"
        else:
            qualified_name = name
        
        element = CodeElement(
            element_type='subproc',
            name=qualified_name,
            file_path=file_path,
            line_number=line_num,
            return_type=attributes.get('return_type'),
            source_node=node
        )
        
        # Build content
        content_parts = [name, qualified_name]
        if element.return_type:
            content_parts.append(element.return_type)
        content_parts.extend(self._extract_keywords_from_node(node))
        
        element.content = ' '.join(content_parts)
        element.keywords = set(self._tokenize(element.content))
        
        self.elements[qualified_name] = element
        self.file_index[file_path].append(element)
        self.type_index['subproc'].append(element)
        self.stats['total_procedures'] += 1
        self.stats['total_elements'] += 1
    
    def _index_variable(self, node, file_path: str, line_num: int,
                       attributes: Dict, parent_name: str) -> None:
        """Index a variable declaration."""
        var_name = self._safe_get_attr(node, 'name', '')
        if not var_name:
            # Try to extract from value or children
            var_name = self._safe_get_attr(node, 'value', '')
        
        if not var_name or not var_name.strip():
            return  # Skip unnamed variables
        
        # Qualify with parent scope
        if parent_name:
            qualified_name = f"{parent_name}.{var_name}"
        else:
            qualified_name = var_name
        
        element = CodeElement(
            element_type='variable',
            name=qualified_name,
            file_path=file_path,
            line_number=line_num,
            return_type=attributes.get('type'),
            source_node=node
        )
        
        # Build content
        content_parts = [var_name, qualified_name]
        if element.return_type:
            content_parts.append(element.return_type)
        
        element.content = ' '.join(content_parts)
        element.keywords = set(self._tokenize(element.content))
        
        # Add to parent's local variables if applicable
        if parent_name and parent_name in self.elements:
            self.elements[parent_name].local_variables.append(var_name)
        
        self.elements[qualified_name] = element
        self.file_index[file_path].append(element)
        self.type_index['variable'].append(element)
        self.stats['total_variables'] += 1
        self.stats['total_elements'] += 1
    
    def _index_call(self, node, file_path: str, line_num: int,
                   attributes: Dict, parent_name: str) -> None:
        """Index a function call."""
        func_name = attributes.get('function', '')
        if not func_name:
            func_name = self._safe_get_attr(node, 'name', '')
        
        if not func_name:
            return
        
        # Record call relationship
        if parent_name:
            self.call_graph[parent_name].add(func_name)
            self.reverse_call_graph[func_name].add(parent_name)
            
            # Update caller's calls list
            if parent_name in self.elements:
                if func_name not in self.elements[parent_name].calls:
                    self.elements[parent_name].calls.append(func_name)
        
        self.stats['total_calls'] += 1
    
    def _index_assignment(self, node, file_path: str, line_num: int,
                         attributes: Dict, parent_name: str) -> None:
        """Index assignment statements (for variable usage tracking)."""
        target = attributes.get('target', '')
        value = attributes.get('value', '')
        
        if target and parent_name and parent_name in self.elements:
            # Add to parent's content for search
            parent_element = self.elements[parent_name]
            parent_element.content += f" {target} {value}"
    
    def _extract_keywords_from_node(self, node) -> List[str]:
        """
        Extract searchable keywords from node attributes and comments.
        """
        keywords = []
        
        try:
            # Get node value
            value = self._safe_get_attr(node, 'value', '')
            if value and isinstance(value, str):
                keywords.append(value)
            
            # Get attributes
            attributes = self._safe_get_attr(node, 'attributes', {})
            if isinstance(attributes, dict):
                for key, val in attributes.items():
                    if isinstance(val, str):
                        keywords.append(val)
            
            # Extract from comments
            children = self._safe_get_attr(node, 'children', [])
            for child in children:
                if self._safe_get_attr(child, 'type') == 'comment':
                    comment = self._safe_get_attr(child, 'value', '')
                    if comment:
                        keywords.append(comment)
        except Exception:
            pass
        
        return keywords
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for TF-IDF, preserving TAL-specific identifiers.
        """
        if not text:
            return []
        
        # Convert to lowercase
        text = text.lower()
        
        # Extract TAL identifiers (including $ for system functions)
        # Preserve underscores and ^ characters common in TAL
        tokens = re.findall(r'\$?[a-z_^][a-z0-9_^]*', text)
        
        # Filter out very short tokens and common stop words
        stop_words = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'of', 'to', 'in', 'for'}
        tokens = [t for t in tokens if len(t) > 1 and t not in stop_words]
        
        return tokens
    
    def _build_tfidf_index(self) -> None:
        """
        Build TF-IDF vectors for all indexed elements.
        """
        print("Building TF-IDF index...")
        
        # Build vocabulary and document frequencies
        self.vocabulary = set()
        doc_count = len(self.elements)
        
        if doc_count == 0:
            return
        
        # First pass: collect vocabulary and document frequencies
        for element in self.elements.values():
            tokens = self._tokenize(element.content)
            unique_tokens = set(tokens)
            
            self.vocabulary.update(unique_tokens)
            
            for token in unique_tokens:
                self.document_frequencies[token] += 1
        
        # Calculate IDF scores
        for term in self.vocabulary:
            df = self.document_frequencies[term]
            self.idf_scores[term] = math.log((doc_count + 1) / (df + 1)) + 1
        
        # Second pass: calculate TF-IDF vectors
        for name, element in self.elements.items():
            tokens = self._tokenize(element.content)
            term_freq = Counter(tokens)
            
            vector = {}
            for term, count in term_freq.items():
                tf = count / len(tokens) if tokens else 0
                vector[term] = tf * self.idf_scores.get(term, 0)
            
            self.tf_idf_vectors[name] = vector
        
        print(f"  Vocabulary size: {len(self.vocabulary)}")
        print(f"  Document vectors created: {len(self.tf_idf_vectors)}")
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float, CodeElement]]:
        """
        Search for code elements using TF-IDF similarity.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            
        Returns:
            List of (element_name, score, element) tuples sorted by relevance
        """
        # Tokenize query
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []
        
        # Build query vector
        query_tf = Counter(query_tokens)
        query_vector = {}
        for term, count in query_tf.items():
            if term in self.vocabulary:
                tf = count / len(query_tokens)
                query_vector[term] = tf * self.idf_scores[term]
        
        # Calculate cosine similarity with all documents
        scores = []
        for name, doc_vector in self.tf_idf_vectors.items():
            similarity = self._cosine_similarity(query_vector, doc_vector)
            if similarity > 0:
                scores.append((name, similarity, self.elements[name]))
        
        # Sort by score and return top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    
    def _cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """Calculate cosine similarity between two sparse vectors."""
        if not vec1 or not vec2:
            return 0.0
        
        # Calculate dot product
        common_terms = set(vec1.keys()) & set(vec2.keys())
        dot_product = sum(vec1[term] * vec2[term] for term in common_terms)
        
        # Calculate magnitudes
        mag1 = math.sqrt(sum(v * v for v in vec1.values()))
        mag2 = math.sqrt(sum(v * v for v in vec2.values()))
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        return dot_product / (mag1 * mag2)
    
    def add_business_capability(self, capability_id: str, name: str, 
                               description: str, keywords: List[str]) -> None:
        """
        Register a business capability with associated keywords.
        
        Args:
            capability_id: Unique identifier (e.g., 'payment.wire.initiation')
            name: Human-readable name
            description: Capability description
            keywords: Keywords that indicate this capability
        """
        capability = BusinessCapability(
            capability_id=capability_id,
            name=name,
            description=description,
            keywords=set(kw.lower() for kw in keywords)
        )
        
        self.capabilities[capability_id] = capability
        
        # Build reverse keyword index
        for keyword in capability.keywords:
            if keyword not in self.capability_patterns:
                self.capability_patterns[keyword] = []
            self.capability_patterns[keyword].append(capability_id)
    
    def map_capabilities(self) -> None:
        """
        Map code elements to business capabilities based on keywords.
        """
        print("Mapping business capabilities...")
        
        mapped_count = 0
        for name, element in self.elements.items():
            # Check element keywords against capability patterns
            for keyword in element.keywords:
                if keyword in self.capability_patterns:
                    for capability_id in self.capability_patterns[keyword]:
                        element.business_capabilities.add(capability_id)
                        self.capabilities[capability_id].implementing_functions.add(name)
                        mapped_count += 1
        
        print(f"  Created {mapped_count} capability mappings")
    
    def get_capability_functions(self, capability_id: str) -> List[CodeElement]:
        """
        Get all functions that implement a business capability.
        
        Args:
            capability_id: Capability identifier
            
        Returns:
            List of code elements implementing this capability
        """
        if capability_id not in self.capabilities:
            return []
        
        capability = self.capabilities[capability_id]
        return [self.elements[name] for name in capability.implementing_functions
                if name in self.elements]
    
    def get_function_capabilities(self, function_name: str) -> List[BusinessCapability]:
        """
        Get all business capabilities implemented by a function.
        
        Args:
            function_name: Function/procedure name
            
        Returns:
            List of business capabilities
        """
        if function_name not in self.elements:
            return []
        
        element = self.elements[function_name]
        return [self.capabilities[cap_id] for cap_id in element.business_capabilities
                if cap_id in self.capabilities]
    
    def get_call_chain(self, function_name: str, max_depth: int = 3) -> List[List[str]]:
        """
        Get call chains starting from a function.
        
        Args:
            function_name: Starting function
            max_depth: Maximum chain depth
            
        Returns:
            List of call chains (each chain is a list of function names)
        """
        chains = []
        
        def dfs(current: str, chain: List[str], depth: int):
            if depth > max_depth:
                return
            
            if current in self.call_graph:
                for callee in self.call_graph[current]:
                    new_chain = chain + [callee]
                    chains.append(new_chain)
                    dfs(callee, new_chain, depth + 1)
        
        dfs(function_name, [function_name], 0)
        return chains
    
    def get_callers(self, function_name: str) -> List[CodeElement]:
        """
        Get all functions that call a given function.
        
        Args:
            function_name: Function to find callers for
            
        Returns:
            List of caller code elements
        """
        if function_name not in self.reverse_call_graph:
            return []
        
        callers = []
        for caller_name in self.reverse_call_graph[function_name]:
            if caller_name in self.elements:
                callers.append(self.elements[caller_name])
        
        return callers
    
    def export_index(self, output_path: str) -> None:
        """
        Export the complete index to a file.
        
        Args:
            output_path: Path to save the index
        """
        print(f"Exporting index to {output_path}...")
        
        export_data = {
            'elements': {name: elem.to_dict() for name, elem in self.elements.items()},
            'capabilities': {
                cap_id: {
                    'capability_id': cap.capability_id,
                    'name': cap.name,
                    'description': cap.description,
                    'keywords': list(cap.keywords),
                    'implementing_functions': list(cap.implementing_functions)
                }
                for cap_id, cap in self.capabilities.items()
            },
            'call_graph': {k: list(v) for k, v in self.call_graph.items()},
            'stats': self.stats,
            'vocabulary': list(self.vocabulary),
            'idf_scores': self.idf_scores
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"  Index exported successfully")
    
    def save_binary(self, output_path: str) -> None:
        """
        Save index as binary pickle for fast loading.
        
        Args:
            output_path: Path to save the binary index
        """
        print(f"Saving binary index to {output_path}...")
        
        with open(output_path, 'wb') as f:
            pickle.dump(self, f)
        
        print("  Binary index saved successfully")
    
    @staticmethod
    def load_binary(input_path: str) -> 'TALCodeIndexer':
        """
        Load a saved binary index.
        
        Args:
            input_path: Path to the saved index
            
        Returns:
            Loaded indexer instance
        """
        print(f"Loading binary index from {input_path}...")
        
        with open(input_path, 'rb') as f:
            indexer = pickle.load(f)
        
        print(f"  Loaded {indexer.stats['total_elements']} elements")
        return indexer
    
    def print_statistics(self) -> None:
        """Print comprehensive indexing statistics."""
        print("\n" + "=" * 60)
        print("INDEXING STATISTICS")
        print("=" * 60)
        print(f"Total Elements:        {self.stats['total_elements']}")
        print(f"  Procedures:          {self.stats['total_procedures']}")
        print(f"  Variables:           {self.stats['total_variables']}")
        print(f"  Function Calls:      {self.stats['total_calls']}")
        print(f"Files Indexed:         {self.stats['files_indexed']}")
        print(f"Malformed Nodes:       {self.stats['malformed_nodes_recovered']}")
        print(f"Vocabulary Size:       {len(self.vocabulary)}")
        print(f"Business Capabilities: {len(self.capabilities)}")
        print(f"Capability Mappings:   {sum(len(e.business_capabilities) for e in self.elements.values())}")
        print("=" * 60)


def create_payment_system_capabilities() -> Dict[str, Dict]:
    """
    Create a standard set of business capabilities for payment systems.
    Returns capability definitions for easy registration.
    """
    return {
        'payment.wire.initiation': {
            'name': 'Wire Transfer Initiation',
            'description': 'Accept and validate wire transfer requests',
            'keywords': ['wire', 'transfer', 'initiate', 'process_wire', 'wire_transfer', 'drawdown']
        },
        'payment.wire.validation': {
            'name': 'Wire Transfer Validation',
            'description': 'Validate wire transfer details and parties',
            'keywords': ['validate', 'validation', 'check', 'verify', 'wire']
        },
        'compliance.ofac.screening': {
            'name': 'OFAC Sanctions Screening',
            'description': 'Screen parties against OFAC sanctions lists',
            'keywords': ['ofac', 'sanction', 'screen', 'sdn', 'watchlist', 'compliance']
        },
        'compliance.party.validation': {
            'name': 'Party Validation',
            'description': 'Validate originator and beneficiary details',
            'keywords': ['party', 'validate', 'originator', 'beneficiary', 'customer']
        },
        'payment.iso20022.messaging': {
            'name': 'ISO 20022 Message Creation',
            'description': 'Create and format ISO 20022 payment messages',
            'keywords': ['pacs', 'iso', '20022', 'message', 'format', 'pacs008', 'pacs.008']
        },
        'payment.bic.validation': {
            'name': 'BIC Code Validation',
            'description': 'Validate bank identifier codes',
            'keywords': ['bic', 'swift', 'bank', 'identifier', 'routing']
        },
        'payment.clearing.routing': {
            'name': 'Payment Clearing and Routing',
            'description': 'Route payments to clearing networks',
            'keywords': ['clear', 'clearing', 'route', 'routing', 'fedwire', 'chips', 'ach']
        },
        'payment.repair.prediction': {
            'name': 'Payment Repair Prediction',
            'description': 'Predict and execute payment repairs',
            'keywords': ['repair', 'fix', 'ace', 'exception', 'error', 'correction']
        },
        'payment.ledger.accounting': {
            'name': 'Ledger Accounting',
            'description': 'Record transactions in general ledger',
            'keywords': ['ledger', 'accounting', 'post', 'journal', 'gl', 'account']
        },
        'data.cdc.processing': {
            'name': 'Change Data Capture Processing',
            'description': 'Process CDC events from Oracle LogMiner',
            'keywords': ['cdc', 'logminer', 'oracle', 'change', 'capture', 'scn']
        }
    }


if __name__ == '__main__':
    # Example usage
    print("TAL Code Indexer - Example Usage")
    print("=" * 60)
    
    # Create indexer
    indexer = TALCodeIndexer()
    
    # Register payment system capabilities
    capabilities = create_payment_system_capabilities()
    for cap_id, cap_data in capabilities.items():
        indexer.add_business_capability(
            cap_id,
            cap_data['name'],
            cap_data['description'],
            cap_data['keywords']
        )
    
    print(f"\nRegistered {len(capabilities)} business capabilities")
    print("\nTo use this indexer:")
    print("1. Parse your TAL files with tal_proc_parser or enhanced_tal_parser")
    print("2. Call indexer.index_ast(ast_node, file_path) for each file")
    print("3. Call indexer.map_capabilities() to map code to business functions")
    print("4. Use indexer.search(query) for TF-IDF search")
    print("5. Use indexer.get_capability_functions(cap_id) to find related code")
    print("6. Export with indexer.export_index() or indexer.save_binary()")
