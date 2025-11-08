"""
AST Parser Interface for Knowledge Graph - TAL Parser
Modified to work with tal_proc_parser and enhanced TAL parser ASTs
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Set, Tuple
from pathlib import Path
import re
from dataclasses import dataclass
import logging

from knowledge_graph import (
    Entity, Relationship, EntityType, RelationType, KnowledgeGraph
)

# Import the TAL parser modules
try:
    import tal_proc_parser
    from enhanced_tal_parser import EnhancedTALParser
    TAL_PARSERS_AVAILABLE = True
except ImportError:
    TAL_PARSERS_AVAILABLE = False
    logging.warning("TAL parser modules not available")

logger = logging.getLogger(__name__)


# ============================================================================
# Abstract Parser Interface (unchanged)
# ============================================================================

class ASTParser(ABC):
    """Abstract base class for AST parsers"""
    
    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.kg = knowledge_graph
        self.file_entity_cache: Dict[str, Entity] = {}
    
    @abstractmethod
    def parse_file(self, file_path: str, ast_data: Any) -> Entity:
        """Parse a file's AST and populate the knowledge graph"""
        pass
    
    @abstractmethod
    def extract_entities(self, ast_data: Any, file_entity: Entity) -> List[Entity]:
        """Extract entities from AST"""
        pass
    
    @abstractmethod
    def extract_relationships(self, ast_data: Any, entities: List[Entity]) -> List[Relationship]:
        """Extract relationships from AST"""
        pass
    
    def get_or_create_file_entity(self, file_path: str, language: str) -> Entity:
        """Get or create a file entity"""
        if file_path in self.file_entity_cache:
            return self.file_entity_cache[file_path]
        
        path = Path(file_path)
        file_entity = Entity(
            id="",
            type=EntityType.FILE,
            name=path.name,
            qualified_name=str(path),
            file_path=str(path),
            language=language,
            metadata={'extension': path.suffix}
        )
        
        self.kg.add_entity(file_entity)
        self.file_entity_cache[file_path] = file_entity
        return file_entity
    
    def create_directory_hierarchy(self, file_path: str) -> List[Entity]:
        """Create directory hierarchy entities"""
        path = Path(file_path)
        directories = []
        
        for parent in path.parents:
            if parent == Path('.'):
                continue
            
            dir_entity = Entity(
                id="",
                type=EntityType.DIRECTORY,
                name=parent.name,
                qualified_name=str(parent),
                file_path=str(parent),
                metadata={}
            )
            self.kg.add_entity(dir_entity)
            directories.append(dir_entity)
        
        return directories


# ============================================================================
# Enhanced TAL Parser - Works with tal_proc_parser AST
# ============================================================================

class TALParser(ASTParser):
    """
    Parser for TAL (Transaction Application Language) code
    
    This parser works with the AST structure produced by:
    - tal_proc_parser.py (foundation parser)
    - enhanced_tal_parser.py (comprehensive parser)
    
    Accepts either:
    1. TALNode tree structure directly
    2. Result dictionary from EnhancedTALParser.parse_file()
    """
    
    def __init__(self, knowledge_graph: KnowledgeGraph):
        super().__init__(knowledge_graph)
        self.language = "TAL"
        self.entity_lookup: Dict[str, Entity] = {}  # Track entities by qualified name
        self.procedure_calls: List[Tuple[str, str, int]] = []  # (caller, callee, line)
    
    def parse_file(self, file_path: str, ast_data: Any) -> Entity:
        """
        Parse TAL file AST
        
        Args:
            file_path: Path to the TAL source file
            ast_data: Either a TALNode (root program node) or result dict from EnhancedTALParser
        
        Returns:
            The file entity
        """
        file_entity = self.get_or_create_file_entity(file_path, self.language)
        
        # Reset tracking for this file
        self.entity_lookup.clear()
        self.procedure_calls.clear()
        
        # Determine AST format and extract root node
        root_node = self._extract_root_node(ast_data)
        
        if root_node is None:
            logger.warning(f"No valid AST root node found for {file_path}")
            return file_entity
        
        # Extract entities from AST
        entities = self.extract_entities(root_node, file_entity)
        
        # Extract relationships from AST
        relationships = self.extract_relationships(root_node, entities)
        
        # Add to knowledge graph
        for entity in entities:
            self.kg.add_entity(entity)
        
        for relationship in relationships:
            self.kg.add_relationship(relationship)
        
        logger.info(f"Parsed TAL file: {file_path} - {len(entities)} entities, {len(relationships)} relationships")
        return file_entity
    
    def _extract_root_node(self, ast_data: Any):
        """Extract the root TALNode from various input formats"""
        if TAL_PARSERS_AVAILABLE and isinstance(ast_data, tal_proc_parser.TALNode):
            # Direct TALNode
            return ast_data
        elif isinstance(ast_data, dict):
            # Result dictionary from EnhancedTALParser
            if 'ast' in ast_data:
                return ast_data['ast']
            # Legacy format - convert to TALNode structure
            return self._convert_legacy_format(ast_data)
        return None
    
    def _convert_legacy_format(self, data: dict):
        """Convert legacy dict format to TALNode structure (for backward compatibility)"""
        if not TAL_PARSERS_AVAILABLE:
            return None
        
        # Create a program node
        program = tal_proc_parser.TALNode('program')
        
        # Add procedures
        for proc_data in data.get('procedures', []):
            proc_node = tal_proc_parser.TALNode('procedure', name=proc_data.get('name', ''))
            proc_node.attributes.update(proc_data)
            program.add_child(proc_node)
        
        return program
    
    def extract_entities(self, root_node, file_entity: Entity) -> List[Entity]:
        """
        Extract entities from TAL AST node tree
        
        Traverses the TALNode tree and creates Entity objects for:
        - Procedures
        - Subprocs
        - Variables (global, local, parameters)
        - Structures
        - Directives
        """
        entities = []
        
        # Traverse the AST tree
        self._traverse_and_extract(root_node, file_entity, entities, parent_context=None)
        
        return entities
    
    def _traverse_and_extract(self, node, file_entity: Entity, entities: List[Entity], 
                              parent_context: Optional[str] = None):
        """
        Recursively traverse TAL AST and extract entities
        
        Args:
            node: Current TALNode
            file_entity: File entity for context
            entities: List to accumulate entities
            parent_context: Qualified name of parent entity (for scoping)
        """
        if not hasattr(node, 'type'):
            return
        
        node_type = node.type
        
        # Extract procedure entities
        if node_type == 'procedure':
            proc_entity = self._extract_procedure_entity(node, file_entity, parent_context)
            if proc_entity:
                entities.append(proc_entity)
                self.entity_lookup[proc_entity.qualified_name] = proc_entity
                
                # Recurse into procedure with new context
                new_context = proc_entity.qualified_name
                for child in node.children:
                    self._traverse_and_extract(child, file_entity, entities, new_context)
        
        # Extract subprocedure entities
        elif node_type == 'subproc':
            subproc_entity = self._extract_subproc_entity(node, file_entity, parent_context)
            if subproc_entity:
                entities.append(subproc_entity)
                self.entity_lookup[subproc_entity.qualified_name] = subproc_entity
                
                # Recurse into subproc
                new_context = subproc_entity.qualified_name
                for child in node.children:
                    self._traverse_and_extract(child, file_entity, entities, new_context)
        
        # Extract variable declarations
        elif node_type == 'var_decl':
            var_entities = self._extract_variable_entities(node, file_entity, parent_context)
            entities.extend(var_entities)
            for var_entity in var_entities:
                self.entity_lookup[var_entity.qualified_name] = var_entity
        
        # Extract parameters
        elif node_type == 'parameters':
            for param_child in node.children:
                if param_child.type == 'parameter':
                    param_entity = self._extract_parameter_entity(param_child, file_entity, parent_context)
                    if param_entity:
                        entities.append(param_entity)
                        self.entity_lookup[param_entity.qualified_name] = param_entity
        
        # Extract structure definitions
        elif node_type in ['struct_decl', 'nested_struct_decl']:
            struct_entity = self._extract_structure_entity(node, file_entity, parent_context)
            if struct_entity:
                entities.append(struct_entity)
                self.entity_lookup[struct_entity.qualified_name] = struct_entity
        
        # Extract directives
        elif 'directive' in node_type:
            directive_entity = self._extract_directive_entity(node, file_entity)
            if directive_entity:
                entities.append(directive_entity)
        
        # Extract procedure calls (for relationship extraction later)
        elif node_type in ['call_stmt', 'system_function_call']:
            self._record_procedure_call(node, parent_context)
        
        # Continue traversing for other node types
        else:
            for child in node.children:
                self._traverse_and_extract(child, file_entity, entities, parent_context)
    
    def _extract_procedure_entity(self, node, file_entity: Entity, 
                                   parent_context: Optional[str]) -> Optional[Entity]:
        """Extract Entity from a procedure TALNode"""
        proc_name = node.name
        if not proc_name:
            return None
        
        # Build qualified name
        if parent_context:
            qualified_name = f"{parent_context}::{proc_name}"
        else:
            qualified_name = f"{file_entity.name}::{proc_name}"
        
        # Extract metadata from node attributes
        metadata = {
            'return_type': node.attributes.get('return_type', 'void'),
            'is_main': node.attributes.get('is_main', False),
            'is_forward': node.attributes.get('is_forward', False),
            'is_external': node.attributes.get('is_external', False),
            'attributes': node.attributes.get('attributes', [])
        }
        
        # Extract parameters
        params_node = None
        for child in node.children:
            if child.type == 'parameters':
                params_node = child
                break
        
        if params_node:
            param_names = []
            for param in params_node.children:
                if param.type == 'parameter':
                    param_names.append(param.name)
            metadata['parameters'] = param_names
        
        # Count local variables and statements
        local_count = 0
        stmt_count = 0
        for child in node.children:
            if child.type == 'local_declarations':
                local_count = len(child.children)
            elif child.type == 'statements':
                stmt_count = len(child.children)
        
        metadata['local_variable_count'] = local_count
        metadata['statement_count'] = stmt_count
        
        # Create entity
        entity = Entity(
            id="",
            type=EntityType.PROCEDURE,
            name=proc_name,
            qualified_name=qualified_name,
            file_path=file_entity.file_path,
            start_line=node.location.line if node.location else 0,
            end_line=node.location.line if node.location else 0,  # Could be improved
            language=self.language,
            metadata=metadata
        )
        
        # Create DEFINES relationship
        self.kg.add_relationship(Relationship(
            source_id=file_entity.id,
            target_id=entity.id,
            type=RelationType.DEFINES
        ))
        
        return entity
    
    def _extract_subproc_entity(self, node, file_entity: Entity, 
                                 parent_context: Optional[str]) -> Optional[Entity]:
        """Extract Entity from a subproc TALNode"""
        subproc_name = node.attributes.get('subproc_name', node.name)
        if not subproc_name:
            return None
        
        # Build qualified name
        if parent_context:
            qualified_name = f"{parent_context}::{subproc_name}"
        else:
            qualified_name = f"{file_entity.name}::{subproc_name}"
        
        # Extract metadata
        metadata = {
            'return_type': node.attributes.get('return_type', 'void'),
            'is_subproc': True
        }
        
        # Extract parameters
        params_node = None
        for child in node.children:
            if child.type == 'parameters':
                params_node = child
                break
        
        if params_node:
            param_names = []
            for param in params_node.children:
                if param.type == 'parameter':
                    param_names.append(param.name)
            metadata['parameters'] = param_names
        
        # Create entity (using PROCEDURE type for subprocs too)
        entity = Entity(
            id="",
            type=EntityType.PROCEDURE,
            name=subproc_name,
            qualified_name=qualified_name,
            file_path=file_entity.file_path,
            start_line=node.location.line if node.location else 0,
            language=self.language,
            metadata=metadata
        )
        
        return entity
    
    def _extract_variable_entities(self, node, file_entity: Entity, 
                                    parent_context: Optional[str]) -> List[Entity]:
        """Extract variable entities from var_decl node"""
        entities = []
        var_type = node.attributes.get('type', 'UNKNOWN')
        
        # Process each variable specification
        for child in node.children:
            if child.type == 'var_spec':
                var_name = child.name
                if not var_name:
                    continue
                
                # Build qualified name
                if parent_context:
                    qualified_name = f"{parent_context}::{var_name}"
                    scope = 'local'
                else:
                    qualified_name = f"{file_entity.name}::{var_name}"
                    scope = 'global'
                
                # Extract metadata
                metadata = {
                    'data_type': var_type,
                    'scope': scope,
                    'is_pointer': child.attributes.get('pointer', False),
                    'is_array': child.attributes.get('array', False)
                }
                
                if 'array_bounds' in child.attributes:
                    metadata['array_bounds'] = child.attributes['array_bounds']
                
                if 'initializer' in child.attributes:
                    metadata['initializer'] = child.attributes['initializer']
                
                # Create entity
                entity = Entity(
                    id="",
                    type=EntityType.VARIABLE,
                    name=var_name,
                    qualified_name=qualified_name,
                    file_path=file_entity.file_path,
                    start_line=node.location.line if node.location else 0,
                    language=self.language,
                    metadata=metadata
                )
                
                entities.append(entity)
        
        return entities
    
    def _extract_parameter_entity(self, node, file_entity: Entity, 
                                   parent_context: Optional[str]) -> Optional[Entity]:
        """Extract Entity from a parameter TALNode"""
        param_name = node.name
        if not param_name:
            return None
        
        # Build qualified name
        if parent_context:
            qualified_name = f"{parent_context}::{param_name}"
        else:
            qualified_name = f"{file_entity.name}::{param_name}"
        
        # Extract metadata
        param_type = node.attributes.get('type', 'UNKNOWN')
        metadata = {
            'data_type': param_type,
            'scope': 'parameter',
            'is_pointer': node.attributes.get('pointer', False)
        }
        
        # Create entity
        entity = Entity(
            id="",
            type=EntityType.VARIABLE,
            name=param_name,
            qualified_name=qualified_name,
            file_path=file_entity.file_path,
            start_line=node.location.line if node.location else 0,
            language=self.language,
            metadata=metadata
        )
        
        return entity
    
    def _extract_structure_entity(self, node, file_entity: Entity, 
                                   parent_context: Optional[str]) -> Optional[Entity]:
        """Extract Entity from a struct declaration node"""
        struct_name = node.attributes.get('struct_name', node.name)
        if not struct_name:
            return None
        
        # Build qualified name
        if parent_context:
            qualified_name = f"{parent_context}::{struct_name}"
        else:
            qualified_name = f"{file_entity.name}::{struct_name}"
        
        # Extract metadata
        metadata = {
            'is_nested': node.type == 'nested_struct_decl'
        }
        
        if 'template_params' in node.attributes:
            metadata['template_params'] = node.attributes['template_params']
        
        # Extract fields
        fields = []
        for child in node.children:
            if child.type in ['struct_field', 'nested_struct_field', 'struct_body']:
                # Process struct body or fields
                self._extract_struct_fields(child, fields)
        
        if fields:
            metadata['fields'] = fields
            metadata['field_count'] = len(fields)
        
        # Create entity
        entity = Entity(
            id="",
            type=EntityType.STRUCTURE,
            name=struct_name,
            qualified_name=qualified_name,
            file_path=file_entity.file_path,
            start_line=node.location.line if node.location else 0,
            language=self.language,
            metadata=metadata
        )
        
        return entity
    
    def _extract_struct_fields(self, node, fields: List[Dict[str, Any]]):
        """Recursively extract struct fields"""
        if node.type == 'struct_body':
            for child in node.children:
                self._extract_struct_fields(child, fields)
        elif node.type in ['struct_field', 'nested_struct_field']:
            field_info = {
                'name': node.attributes.get('field_name', node.name),
                'type': node.attributes.get('field_type', 'UNKNOWN')
            }
            
            if 'array_bounds' in node.attributes:
                field_info['array_bounds'] = node.attributes['array_bounds']
                field_info['is_array'] = True
            
            fields.append(field_info)
    
    def _extract_directive_entity(self, node, file_entity: Entity) -> Optional[Entity]:
        """Extract Entity from directive nodes"""
        directive_type = node.type.replace('_directive', '').upper()
        
        metadata = {
            'directive_type': directive_type,
            'value': node.value if hasattr(node, 'value') else ''
        }
        
        # Add specific attributes based on directive type
        if directive_type == 'PAGE' and 'title' in node.attributes:
            metadata['title'] = node.attributes['title']
        elif directive_type == 'SECTION' and 'section_name' in node.attributes:
            metadata['section_name'] = node.attributes['section_name']
        elif directive_type == 'SOURCE' and 'function_name' in node.attributes:
            metadata['function_name'] = node.attributes['function_name']
        
        # Create entity
        entity = Entity(
            id="",
            type=EntityType.DIRECTIVE,
            name=f"{directive_type}_directive",
            qualified_name=f"{file_entity.name}::{directive_type}_{node.location.line if node.location else 0}",
            file_path=file_entity.file_path,
            start_line=node.location.line if node.location else 0,
            language=self.language,
            metadata=metadata
        )
        
        return entity
    
    def _record_procedure_call(self, node, caller_context: Optional[str]):
        """Record procedure call for relationship extraction"""
        callee_name = node.attributes.get('function', '')
        if not callee_name:
            return
        
        line = node.location.line if node.location else 0
        
        if caller_context:
            self.procedure_calls.append((caller_context, callee_name, line))
    
    def extract_relationships(self, root_node, entities: List[Entity]) -> List[Relationship]:
        """
        Extract relationships from TAL AST
        
        Extracts:
        - CALLS relationships (procedure calls)
        - CONTAINS relationships (procedures containing variables)
        - USES relationships (variables used in procedures)
        """
        relationships = []
        
        # Build entity lookup by name for quick access
        entities_by_qname = {e.qualified_name: e for e in entities}
        entities_by_name = {e.name: e for e in entities}
        
        # Extract CALLS relationships from recorded calls
        for caller_qname, callee_name, line in self.procedure_calls:
            caller = entities_by_qname.get(caller_qname)
            if not caller:
                continue
            
            # Try to find callee by name (could be in same file or external)
            callee = None
            
            # First try exact match by name in current file
            for entity in entities:
                if entity.type == EntityType.PROCEDURE and entity.name == callee_name:
                    callee = entity
                    break
            
            # If found, create relationship
            if callee:
                rel = Relationship(
                    source_id=caller.id,
                    target_id=callee.id,
                    type=RelationType.CALLS,
                    metadata={'line': line}
                )
                relationships.append(rel)
        
        # Extract CONTAINS relationships (procedures contain variables/params)
        for entity in entities:
            if entity.type == EntityType.VARIABLE:
                # Find containing procedure
                qname_parts = entity.qualified_name.split('::')
                if len(qname_parts) >= 2:
                    # Try to find parent procedure
                    parent_qname = '::'.join(qname_parts[:-1])
                    parent = entities_by_qname.get(parent_qname)
                    
                    if parent and parent.type == EntityType.PROCEDURE:
                        rel = Relationship(
                            source_id=parent.id,
                            target_id=entity.id,
                            type=RelationType.CONTAINS,
                            metadata={'scope': entity.metadata.get('scope', 'unknown')}
                        )
                        relationships.append(rel)
        
        return relationships


# ============================================================================
# Convenience function to parse TAL files
# ============================================================================

def parse_tal_file_to_kg(file_path: str, knowledge_graph: KnowledgeGraph) -> Entity:
    """
    Convenience function to parse a TAL file and add to knowledge graph
    
    This function:
    1. Uses EnhancedTALParser to parse the file
    2. Feeds the AST to TALParser
    3. Populates the knowledge graph
    
    Args:
        file_path: Path to TAL source file
        knowledge_graph: KnowledgeGraph instance to populate
    
    Returns:
        File entity
    """
    if not TAL_PARSERS_AVAILABLE:
        raise ImportError("TAL parser modules not available")
    
    # Parse the file with enhanced TAL parser
    enhanced_parser = EnhancedTALParser()
    parse_result = enhanced_parser.parse_file(file_path)
    
    if not parse_result.get('success'):
        logger.error(f"Failed to parse {file_path}: {parse_result.get('error')}")
        raise ValueError(f"TAL parsing failed: {parse_result.get('error')}")
    
    # Use TALParser to extract entities and relationships
    tal_parser = TALParser(knowledge_graph)
    file_entity = tal_parser.parse_file(file_path, parse_result['ast'])
    
    return file_entity


# ============================================================================
# Example usage
# ============================================================================

if __name__ == "__main__":
    """
    Example demonstrating TAL file parsing into knowledge graph
    """
    from knowledge_graph import KnowledgeGraph
    
    # Create knowledge graph
    kg = KnowledgeGraph()
    
    # Parse a TAL file
    tal_file = "example.tal"
    
    try:
        file_entity = parse_tal_file_to_kg(tal_file, kg)
        print(f"Successfully parsed {tal_file}")
        print(f"Total entities: {len(kg.entities)}")
        print(f"Total relationships: {len(kg.relationships)}")
        
        # Show some statistics
        procedures = [e for e in kg.entities.values() if e.type == EntityType.PROCEDURE]
        variables = [e for e in kg.entities.values() if e.type == EntityType.VARIABLE]
        
        print(f"\nProcedures: {len(procedures)}")
        print(f"Variables: {len(variables)}")
        
        # Show procedure calls
        calls = [r for r in kg.relationships if r.type == RelationType.CALLS]
        print(f"Procedure calls: {len(calls)}")
        
    except Exception as e:
        print(f"Error parsing file: {e}")
