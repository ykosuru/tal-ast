"""
AST Parser Interface for Knowledge Graph - TAL Parser
Modified to export graph data to files for visualization

Features:
- Parse TAL AST into knowledge graph entities and relationships
- Handle external procedure references
- Export graph data to JSON for visualization
- Support for multiple file parsing
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Set, Tuple
from pathlib import Path
import re
from dataclasses import dataclass
import logging
import sys
import json

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


"""
AST Parser Interface for Knowledge Graph - TAL Parser
Modified to work with tal_proc_parser and enhanced TAL parser ASTs

Features:
- Parse TAL AST into knowledge graph entities and relationships
- Handle external procedure references
- Advanced search capabilities for finding functionality
- Support for multiple file parsing
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Set, Tuple
from pathlib import Path
import re
from dataclasses import dataclass
import logging
import sys

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Abstract Parser Interface
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
    
    Features:
    - Handles external procedure references
    - Creates placeholder entities for unresolved calls
    - Tracks directives in file metadata
    """
    
    def __init__(self, knowledge_graph: KnowledgeGraph):
        super().__init__(knowledge_graph)
        self.language = "TAL"
        self.entity_lookup: Dict[str, Entity] = {}
        self.procedure_calls: List[Tuple[str, str, int]] = []
        self.directives: List[Dict[str, Any]] = []
        self.external_references: List[str] = []
    
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
        self.directives.clear()
        self.external_references.clear()
        
        # Determine AST format and extract root node
        root_node = self._extract_root_node(ast_data)
        
        if root_node is None:
            logger.warning(f"No valid AST root node found for {file_path}")
            return file_entity
        
        # Extract entities from AST
        entities = self.extract_entities(root_node, file_entity)
        
        # Extract relationships from AST (handles external references)
        relationships = self.extract_relationships(root_node, entities)
        
        # Add directives to file metadata
        if self.directives:
            file_entity.metadata['directives'] = self.directives
            file_entity.metadata['directive_count'] = len(self.directives)
        
        # Add external reference info to file metadata
        if self.external_references:
            file_entity.metadata['external_references'] = self.external_references
            file_entity.metadata['external_reference_count'] = len(self.external_references)
        
        # Add to knowledge graph
        for entity in entities:
            self.kg.add_entity(entity)
        
        for relationship in relationships:
            self.kg.add_relationship(relationship)
        
        logger.info(
            f"Parsed TAL file: {file_path} - "
            f"{len(entities)} entities, {len(relationships)} relationships"
        )
        
        if self.external_references:
            logger.info(
                f"  External references: {len(self.external_references)} "
                f"({', '.join(self.external_references[:5])}{'...' if len(self.external_references) > 5 else ''})"
            )
        
        return file_entity
    
    def _extract_root_node(self, ast_data: Any):
        """Extract the root TALNode from various input formats"""
        if TAL_PARSERS_AVAILABLE and isinstance(ast_data, tal_proc_parser.TALNode):
            return ast_data
        elif isinstance(ast_data, dict):
            if 'ast' in ast_data:
                return ast_data['ast']
            return self._convert_legacy_format(ast_data)
        return None
    
    def _convert_legacy_format(self, data: dict):
        """Convert legacy dict format to TALNode structure"""
        if not TAL_PARSERS_AVAILABLE:
            return None
        
        program = tal_proc_parser.TALNode('program')
        for proc_data in data.get('procedures', []):
            proc_node = tal_proc_parser.TALNode('procedure', name=proc_data.get('name', ''))
            proc_node.attributes.update(proc_data)
            program.add_child(proc_node)
        
        return program
    
    def extract_entities(self, root_node, file_entity: Entity) -> List[Entity]:
        """Extract entities from TAL AST node tree"""
        entities = []
        self._traverse_and_extract(root_node, file_entity, entities, parent_context=None)
        return entities
    
    def _traverse_and_extract(self, node, file_entity: Entity, entities: List[Entity], 
                              parent_context: Optional[str] = None):
        """Recursively traverse TAL AST and extract entities"""
        if not hasattr(node, 'type'):
            return
        
        node_type = node.type
        
        # Extract procedure entities
        if node_type == 'procedure':
            proc_entity = self._extract_procedure_entity(node, file_entity, parent_context)
            if proc_entity:
                entities.append(proc_entity)
                self.entity_lookup[proc_entity.qualified_name] = proc_entity
                
                new_context = proc_entity.qualified_name
                for child in node.children:
                    self._traverse_and_extract(child, file_entity, entities, new_context)
        
        # Extract subprocedure entities
        elif node_type == 'subproc':
            subproc_entity = self._extract_subproc_entity(node, file_entity, parent_context)
            if subproc_entity:
                entities.append(subproc_entity)
                self.entity_lookup[subproc_entity.qualified_name] = subproc_entity
                
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
        
        # Record directives
        elif 'directive' in node_type:
            self._record_directive(node)
        
        # Extract procedure calls
        elif node_type in ['call_stmt', 'system_function_call']:
            self._record_procedure_call(node, parent_context)
        
        # Continue traversing
        else:
            for child in node.children:
                self._traverse_and_extract(child, file_entity, entities, parent_context)
    
    def _extract_procedure_entity(self, node, file_entity: Entity, 
                                   parent_context: Optional[str]) -> Optional[Entity]:
        """Extract Entity from a procedure TALNode"""
        proc_name = node.name
        if not proc_name:
            return None
        
        if parent_context:
            qualified_name = f"{parent_context}::{proc_name}"
        else:
            qualified_name = f"{file_entity.name}::{proc_name}"
        
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
            metadata['parameter_count'] = len(param_names)
        
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
        
        entity = Entity(
            id="",
            type=EntityType.PROCEDURE,
            name=proc_name,
            qualified_name=qualified_name,
            file_path=file_entity.file_path,
            start_line=node.location.line if node.location else 0,
            end_line=node.location.line if node.location else 0,
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
        
        if parent_context:
            qualified_name = f"{parent_context}::{subproc_name}"
        else:
            qualified_name = f"{file_entity.name}::{subproc_name}"
        
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
            metadata['parameter_count'] = len(param_names)
        
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
        
        for child in node.children:
            if child.type == 'var_spec':
                var_name = child.name
                if not var_name:
                    continue
                
                if parent_context:
                    qualified_name = f"{parent_context}::{var_name}"
                    scope = 'local'
                else:
                    qualified_name = f"{file_entity.name}::{var_name}"
                    scope = 'global'
                
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
        
        if parent_context:
            qualified_name = f"{parent_context}::{param_name}"
        else:
            qualified_name = f"{file_entity.name}::{param_name}"
        
        param_type = node.attributes.get('type', 'UNKNOWN')
        metadata = {
            'data_type': param_type,
            'scope': 'parameter',
            'is_pointer': node.attributes.get('pointer', False)
        }
        
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
        
        if parent_context:
            qualified_name = f"{parent_context}::{struct_name}"
        else:
            qualified_name = f"{file_entity.name}::{struct_name}"
        
        metadata = {
            'is_nested': node.type == 'nested_struct_decl'
        }
        
        if 'template_params' in node.attributes:
            metadata['template_params'] = node.attributes['template_params']
        
        fields = []
        for child in node.children:
            if child.type in ['struct_field', 'nested_struct_field', 'struct_body']:
                self._extract_struct_fields(child, fields)
        
        if fields:
            metadata['fields'] = fields
            metadata['field_count'] = len(fields)
        
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
    
    def _record_directive(self, node):
        """Record directive information for file metadata"""
        directive_type = node.type.replace('_directive', '').upper()
        
        directive_info = {
            'type': directive_type,
            'line': node.location.line if node.location else 0,
            'value': node.value if hasattr(node, 'value') and node.value else ''
        }
        
        if directive_type == 'PAGE' and 'title' in node.attributes:
            directive_info['title'] = node.attributes['title']
        elif directive_type == 'SECTION' and 'section_name' in node.attributes:
            directive_info['section_name'] = node.attributes['section_name']
        elif directive_type == 'SOURCE' and 'function_name' in node.attributes:
            directive_info['function_name'] = node.attributes['function_name']
        
        self.directives.append(directive_info)
    
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
        Handles external/unresolved procedure calls
        """
        relationships = []
        
        entities_by_qname = {e.qualified_name: e for e in entities}
        procedures_by_name = {e.name: e for e in entities if e.type == EntityType.PROCEDURE}
        
        # Extract CALLS relationships
        for caller_qname, callee_name, line in self.procedure_calls:
            caller = entities_by_qname.get(caller_qname)
            if not caller:
                continue
            
            # Try to find callee in current file
            callee = procedures_by_name.get(callee_name)
            
            if callee:
                # Internal call
                rel = Relationship(
                    source_id=caller.id,
                    target_id=callee.id,
                    type=RelationType.CALLS,
                    metadata={'line': line, 'resolved': True}
                )
                relationships.append(rel)
            else:
                # External/unresolved call - create placeholder entity
                logger.debug(f"Creating external reference for: {callee_name}")
                
                external_proc = Entity(
                    id="",
                    type=EntityType.PROCEDURE,
                    name=callee_name,
                    qualified_name=f"external::{callee_name}",
                    language=self.language,
                    metadata={
                        'is_external': True,
                        'unresolved': True,
                        'called_from': [caller.qualified_name]
                    }
                )
                
                self.kg.add_entity(external_proc)
                self.external_references.append(callee_name)
                
                rel = Relationship(
                    source_id=caller.id,
                    target_id=external_proc.id,
                    type=RelationType.CALLS,
                    metadata={
                        'line': line, 
                        'resolved': False,
                        'external': True
                    }
                )
                relationships.append(rel)
        
        # Extract CONTAINS relationships
        for entity in entities:
            if entity.type == EntityType.VARIABLE:
                qname_parts = entity.qualified_name.split('::')
                if len(qname_parts) >= 2:
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
# Knowledge Graph Search Utilities
# ============================================================================

class KnowledgeGraphSearch:
    """Advanced search capabilities for knowledge graph"""
    
    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg
    
    def search_by_name(self, search_term: str, 
                       entity_type: Optional[EntityType] = None,
                       case_sensitive: bool = False) -> List[Entity]:
        """
        Search entities by name (supports wildcards)
        
        Args:
            search_term: Term to search for (supports * and ? wildcards)
            entity_type: Filter by entity type
            case_sensitive: Case-sensitive search
        
        Returns:
            List of matching entities
        """
        pattern = search_term.replace('*', '.*').replace('?', '.')
        if not case_sensitive:
            pattern = f"(?i){pattern}"
        
        regex = re.compile(pattern)
        entities = self.kg.query_entities(entity_type=entity_type)
        
        matches = []
        for entity in entities:
            if regex.search(entity.name) or regex.search(entity.qualified_name):
                matches.append(entity)
        
        return matches
    
    def search_by_metadata(self, key: str, value: Any = None,
                          entity_type: Optional[EntityType] = None) -> List[Entity]:
        """
        Search entities by metadata
        
        Args:
            key: Metadata key to search
            value: Optional value to match
            entity_type: Filter by entity type
        
        Returns:
            List of matching entities
        """
        entities = self.kg.query_entities(entity_type=entity_type)
        
        matches = []
        for entity in entities:
            if key in entity.metadata:
                if value is None or entity.metadata[key] == value:
                    matches.append(entity)
        
        return matches
    
    def search_full_text(self, search_term: str,
                        entity_type: Optional[EntityType] = None,
                        case_sensitive: bool = False) -> List[Dict[str, Any]]:
        """
        Full-text search across entity names, qualified names, and metadata
        
        Args:
            search_term: Term to search for
            entity_type: Filter by entity type
            case_sensitive: Case-sensitive search
        
        Returns:
            List of dicts with entity and match information
        """
        pattern = re.compile(
            re.escape(search_term), 
            0 if case_sensitive else re.IGNORECASE
        )
        
        entities = self.kg.query_entities(entity_type=entity_type)
        
        results = []
        for entity in entities:
            matches = []
            
            if pattern.search(entity.name):
                matches.append({'field': 'name', 'value': entity.name})
            
            if pattern.search(entity.qualified_name):
                matches.append({'field': 'qualified_name', 'value': entity.qualified_name})
            
            for key, value in entity.metadata.items():
                value_str = str(value)
                if pattern.search(value_str):
                    matches.append({'field': f'metadata.{key}', 'value': value_str})
            
            if matches:
                results.append({
                    'entity': entity,
                    'matches': matches,
                    'match_count': len(matches)
                })
        
        results.sort(key=lambda x: x['match_count'], reverse=True)
        
        return results
    
    def find_procedures_calling(self, procedure_name: str) -> List[Entity]:
        """Find all procedures that call a given procedure"""
        target = None
        procedures = self.kg.query_entities(entity_type=EntityType.PROCEDURE)
        for proc in procedures:
            if proc.name == procedure_name:
                target = proc
                break
        
        if not target:
            return []
        
        callers = self.kg.get_neighbors(
            target.id,
            rel_type=RelationType.CALLS,
            direction="incoming"
        )
        
        return callers
    
    def find_procedures_called_by(self, procedure_name: str) -> List[Entity]:
        """Find all procedures called by a given procedure"""
        source = None
        procedures = self.kg.query_entities(entity_type=EntityType.PROCEDURE)
        for proc in procedures:
            if proc.name == procedure_name:
                source = proc
                break
        
        if not source:
            return []
        
        callees = self.kg.get_neighbors(
            source.id,
            rel_type=RelationType.CALLS,
            direction="outgoing"
        )
        
        return callees
    
    def find_related_entities(self, entity_name: str, 
                             max_depth: int = 2) -> Dict[str, Any]:
        """
        Find all entities related to a given entity (up to max_depth hops)
        
        Args:
            entity_name: Name of the entity to search from
            max_depth: Maximum relationship depth to traverse
        
        Returns:
            Dict with entity and related entities at each depth
        """
        entities = self.kg.query_entities()
        start_entity = None
        for entity in entities:
            if entity.name == entity_name:
                start_entity = entity
                break
        
        if not start_entity:
            return {'error': f'Entity "{entity_name}" not found'}
        
        visited = {start_entity.id}
        current_level = [start_entity]
        results = {
            'entity': start_entity,
            'related_by_depth': {}
        }
        
        for depth in range(1, max_depth + 1):
            next_level = []
            level_entities = []
            
            for entity in current_level:
                neighbors = self.kg.get_neighbors(entity.id, direction="both")
                
                for neighbor in neighbors:
                    if neighbor.id not in visited:
                        visited.add(neighbor.id)
                        next_level.append(neighbor)
                        level_entities.append({
                            'entity': neighbor,
                            'from': entity.name
                        })
            
            if level_entities:
                results['related_by_depth'][depth] = level_entities
            
            current_level = next_level
            if not current_level:
                break
        
        return results
    
    def find_by_functionality(self, functionality_term: str) -> Dict[str, Any]:
        """
        Find entities related to a specific functionality
        
        Example: find_by_functionality("drawdown")
        
        Args:
            functionality_term: Term to search (e.g., "drawdown", "payment", "validation")
        
        Returns:
            Dict with categorized results
        """
        results = {
            'search_term': functionality_term,
            'procedures': [],
            'variables': [],
            'structures': [],
            'related_procedures': [],
            'full_text_matches': []
        }
        
        # Search procedures
        proc_matches = self.search_by_name(
            f"*{functionality_term}*",
            entity_type=EntityType.PROCEDURE
        )
        results['procedures'] = proc_matches
        
        # Search variables
        var_matches = self.search_by_name(
            f"*{functionality_term}*",
            entity_type=EntityType.VARIABLE
        )
        results['variables'] = var_matches
        
        # Search structures
        struct_matches = self.search_by_name(
            f"*{functionality_term}*",
            entity_type=EntityType.STRUCTURE
        )
        results['structures'] = struct_matches
        
        # For each matching procedure, find callers and callees
        for proc in proc_matches:
            callers = self.kg.get_neighbors(
                proc.id,
                rel_type=RelationType.CALLS,
                direction="incoming"
            )
            
            callees = self.kg.get_neighbors(
                proc.id,
                rel_type=RelationType.CALLS,
                direction="outgoing"
            )
            
            if callers or callees:
                results['related_procedures'].append({
                    'procedure': proc,
                    'called_by': callers,
                    'calls': callees
                })
        
        # Full-text search in metadata
        full_text_matches = self.search_full_text(functionality_term)
        results['full_text_matches'] = full_text_matches
        
        return results
    
    def display_search_results(self, results: Dict[str, Any]):
        """Display search results in a readable format"""
        print(f"\n{'='*70}")
        print(f"SEARCH RESULTS FOR: '{results['search_term']}'")
        print(f"{'='*70}")
        
        # Display procedures
        procedures = results.get('procedures', [])
        print(f"\n📋 PROCEDURES ({len(procedures)}):")
        if procedures:
            for proc in procedures:
                print(f"  ├─ {proc.name}")
                print(f"  │  Location: {proc.file_path}:{proc.start_line}")
                if 'return_type' in proc.metadata:
                    print(f"  │  Returns: {proc.metadata['return_type']}")
                if 'parameter_count' in proc.metadata:
                    print(f"  │  Parameters: {proc.metadata['parameter_count']}")
                print()
        else:
            print("  (none found)")
        
        # Display variables
        variables = results.get('variables', [])
        print(f"\n📊 VARIABLES ({len(variables)}):")
        if variables:
            for var in variables[:10]:  # Limit to first 10
                data_type = var.metadata.get('data_type', 'UNKNOWN')
                scope = var.metadata.get('scope', 'unknown')
                print(f"  ├─ {var.name} ({data_type})")
                print(f"  │  Scope: {scope}")
            if len(variables) > 10:
                print(f"  └─ ... and {len(variables) - 10} more")
        else:
            print("  (none found)")
        
        # Display structures
        structures = results.get('structures', [])
        print(f"\n🏗️  STRUCTURES ({len(structures)}):")
        if structures:
            for struct in structures:
                print(f"  ├─ {struct.name}")
                if 'field_count' in struct.metadata:
                    print(f"  │  Fields: {struct.metadata['field_count']}")
        else:
            print("  (none found)")
        
        # Display related procedures
        related = results.get('related_procedures', [])
        print(f"\n🔗 RELATED PROCEDURES ({len(related)}):")
        if related:
            for item in related:
                proc = item['procedure']
                print(f"\n  {proc.name}:")
                
                if item['called_by']:
                    print(f"    Called by:")
                    for caller in item['called_by'][:5]:
                        print(f"      ← {caller.name}")
                    if len(item['called_by']) > 5:
                        print(f"      ... and {len(item['called_by']) - 5} more")
                
                if item['calls']:
                    print(f"    Calls:")
                    for callee in item['calls'][:5]:
                        print(f"      → {callee.name}")
                    if len(item['calls']) > 5:
                        print(f"      ... and {len(item['calls']) - 5} more")
        else:
            print("  (none found)")
        
        print(f"\n{'='*70}\n")


# ============================================================================
# Convenience Functions
# ============================================================================

def parse_tal_file_to_kg(file_path: str, knowledge_graph: KnowledgeGraph) -> Entity:
    """
    Parse a TAL file and add to knowledge graph
    
    Args:
        file_path: Path to TAL source file
        knowledge_graph: KnowledgeGraph instance to populate
    
    Returns:
        File entity
    """
    if not TAL_PARSERS_AVAILABLE:
        raise ImportError("TAL parser modules not available")
    
    enhanced_parser = EnhancedTALParser()
    parse_result = enhanced_parser.parse_file(file_path)
    
    if not parse_result.get('success'):
        logger.error(f"Failed to parse {file_path}: {parse_result.get('error')}")
        raise ValueError(f"TAL parsing failed: {parse_result.get('error')}")
    
    tal_parser = TALParser(knowledge_graph)
    file_entity = tal_parser.parse_file(file_path, parse_result['ast'])
    
    return file_entity


def parse_tal_directory(directory: str, knowledge_graph: KnowledgeGraph) -> List[Entity]:
    """
    Parse all TAL files in a directory
    
    Args:
        directory: Directory containing TAL files
        knowledge_graph: KnowledgeGraph instance to populate
    
    Returns:
        List of file entities
    """
    from pathlib import Path
    
    tal_files = list(Path(directory).rglob("*.tal"))
    file_entities = []
    
    for tal_file in tal_files:
        try:
            file_entity = parse_tal_file_to_kg(str(tal_file), knowledge_graph)
            file_entities.append(file_entity)
        except Exception as e:
            logger.error(f"Error parsing {tal_file}: {e}")
    
    return file_entities


# ============================================================================
# Example Usage
# ============================================================================

def main():
    """Main example demonstrating TAL parsing and search"""
    
    if len(sys.argv) < 2:
        print("Usage: python parsers.py <tal_file_or_directory> [search_term]")
        print()
        print("Examples:")
        print("  python parsers.py payment_system.tal")
        print("  python parsers.py payment_system.tal drawdown")
        print("  python parsers.py ./tal_source/")
        sys.exit(1)
    
    input_path = sys.argv[1]
    search_term = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Create knowledge graph
    kg = KnowledgeGraph(backend="networkx")
    
    # Parse file(s)
    print(f"\n{'='*70}")
    print("PARSING TAL CODE")
    print(f"{'='*70}\n")
    
    file_entities = []
    path = Path(input_path)
    
    try:
        if path.is_file():
            file_entity = parse_tal_file_to_kg(input_path, kg)
            file_entities.append(file_entity)
        elif path.is_dir():
            file_entities = parse_tal_directory(input_path, kg)
        else:
            print(f"Error: '{input_path}' not found")
            sys.exit(1)
        
        # Get statistics
        stats = kg.get_statistics()
        
        print(f"\n{'='*70}")
        print("PARSING COMPLETE")
        print(f"{'='*70}")
        print(f"Files parsed: {len(file_entities)}")
        print(f"Total entities: {stats['total_entities']}")
        print(f"Total relationships: {stats['total_relationships']}")
        
        print(f"\nEntity breakdown:")
        for entity_type, count in stats['entity_counts'].items():
            print(f"  {entity_type}: {count}")
        
        print(f"\nRelationship breakdown:")
        for rel_type, count in stats['relationship_counts'].items():
            print(f"  {rel_type}: {count}")
        
        # Show external references
        for file_entity in file_entities:
            if 'external_reference_count' in file_entity.metadata:
                ext_count = file_entity.metadata['external_reference_count']
                if ext_count > 0:
                    print(f"\nExternal references in {file_entity.name}: {ext_count}")
                    ext_refs = file_entity.metadata['external_references']
                    print(f"  {', '.join(ext_refs[:10])}{'...' if len(ext_refs) > 10 else ''}")
        
        # Perform search if search term provided
        if search_term:
            search = KnowledgeGraphSearch(kg)
            results = search.find_by_functionality(search_term)
            search.display_search_results(results)
        else:
            # Show some sample queries
            print(f"\n{'='*70}")
            print("SAMPLE QUERIES")
            print(f"{'='*70}\n")
            
            procedures = kg.query_entities(entity_type=EntityType.PROCEDURE)
            if procedures:
                print(f"Sample procedures:")
                for proc in procedures[:5]:
                    print(f"  - {proc.name}")
                    if 'is_main' in proc.metadata and proc.metadata['is_main']:
                        print(f"    [MAIN PROCEDURE]")
                
                print(f"\nTo search for functionality, run:")
                print(f"  python parsers.py {input_path} <search_term>")
                print(f"\nExample:")
                print(f"  python parsers.py {input_path} drawdown")
        
        # Optionally save to JSON
        # kg.save_to_json('knowledge_graph.json')
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# ============================================================================
# Enhanced Directory Traversal and Main Function
# ============================================================================

def parse_tal_directory_recursive(directory: str, knowledge_graph: KnowledgeGraph, 
                                   recursive: bool = True) -> Dict[str, Any]:
    """
    Parse all TAL files in a directory with detailed progress tracking
    
    Args:
        directory: Directory containing TAL files
        knowledge_graph: KnowledgeGraph instance to populate
        recursive: Whether to search subdirectories recursively
    
    Returns:
        Dict with parsing results and statistics
    """
    from pathlib import Path
    import time
    
    start_time = time.time()
    dir_path = Path(directory)
    
    # Find all TAL files
    if recursive:
        tal_files = list(dir_path.rglob("*.tal"))
    else:
        tal_files = list(dir_path.glob("*.tal"))
    
    if not tal_files:
        logger.warning(f"No .tal files found in {directory}")
        return {
            'success': False,
            'file_count': 0,
            'error': 'No TAL files found'
        }
    
    # Parse results tracking
    results = {
        'success': True,
        'file_count': len(tal_files),
        'parsed_successfully': [],
        'failed': [],
        'skipped': [],
        'total_entities': 0,
        'total_relationships': 0,
        'external_references': set(),
        'parse_time': 0
    }
    
    print(f"\n{'='*70}")
    print(f"PARSING TAL FILES")
    print(f"{'='*70}")
    print(f"Directory: {directory}")
    print(f"Files found: {len(tal_files)}")
    print(f"Recursive: {recursive}")
    print(f"{'='*70}\n")
    
    # Parse each file
    for idx, tal_file in enumerate(tal_files, 1):
        file_path_str = str(tal_file)
        print(f"[{idx}/{len(tal_files)}] Parsing: {tal_file.name}... ", end='', flush=True)
        
        try:
            file_entity = parse_tal_file_to_kg(file_path_str, knowledge_graph)
            
            # Collect external references
            if 'external_references' in file_entity.metadata:
                results['external_references'].update(file_entity.metadata['external_references'])
            
            results['parsed_successfully'].append({
                'file': file_path_str,
                'entity': file_entity
            })
            
            print("✓")
            
        except Exception as e:
            print(f"✗ ERROR: {str(e)[:50]}")
            results['failed'].append({
                'file': file_path_str,
                'error': str(e)
            })
            logger.error(f"Failed to parse {tal_file}: {e}")
    
    # Calculate final statistics
    results['parse_time'] = time.time() - start_time
    stats = knowledge_graph.get_statistics()
    results['total_entities'] = stats['total_entities']
    results['total_relationships'] = stats['total_relationships']
    results['entity_breakdown'] = stats['entity_counts']
    results['relationship_breakdown'] = stats['relationship_counts']
    
    return results


def resolve_external_references(knowledge_graph: KnowledgeGraph) -> Dict[str, Any]:
    """
    Attempt to resolve external procedure references across parsed files
    
    After parsing multiple files, some external references may now be resolvable
    because the procedure exists in another file.
    
    Args:
        knowledge_graph: Knowledge graph with parsed entities
    
    Returns:
        Dict with resolution statistics
    """
    # Find all external procedure entities
    all_entities = knowledge_graph.query_entities()
    external_procs = [e for e in all_entities if e.metadata.get('is_external')]
    real_procs = {e.name: e for e in all_entities 
                  if e.type == EntityType.PROCEDURE and not e.metadata.get('is_external')}
    
    resolution_results = {
        'total_external': len(external_procs),
        'resolved': [],
        'unresolved': []
    }
    
    for ext_proc in external_procs:
        if ext_proc.name in real_procs:
            # Found the real procedure!
            real_proc = real_procs[ext_proc.name]
            
            # Find all relationships pointing to the external entity
            relationships = knowledge_graph.query_relationships(target_id=ext_proc.id)
            
            resolution_results['resolved'].append({
                'name': ext_proc.name,
                'external_id': ext_proc.id,
                'real_id': real_proc.id,
                'real_file': real_proc.file_path,
                'caller_count': len(relationships)
            })
            
            # Update relationships to point to real procedure
            for rel in relationships:
                # Create new relationship pointing to real procedure
                new_rel = Relationship(
                    source_id=rel.source_id,
                    target_id=real_proc.id,
                    type=rel.type,
                    metadata={
                        **rel.metadata,
                        'resolved': True,
                        'originally_external': True
                    }
                )
                knowledge_graph.add_relationship(new_rel)
            
            # Mark external entity as resolved
            ext_proc.metadata['resolved'] = True
            ext_proc.metadata['resolved_to'] = real_proc.id
            ext_proc.metadata['resolved_file'] = real_proc.file_path
        else:
            resolution_results['unresolved'].append({
                'name': ext_proc.name,
                'called_from': ext_proc.metadata.get('called_from', [])
            })
    
    return resolution_results


def display_parsing_summary(results: Dict[str, Any], resolution: Dict[str, Any] = None):
    """Display comprehensive parsing summary"""
    print(f"\n{'='*70}")
    print("PARSING SUMMARY")
    print(f"{'='*70}")
    
    # File statistics
    print(f"\n📁 FILES:")
    print(f"  Total files: {results['file_count']}")
    print(f"  Parsed successfully: {len(results['parsed_successfully'])}")
    if results['failed']:
        print(f"  Failed: {len(results['failed'])}")
    if results['skipped']:
        print(f"  Skipped: {len(results['skipped'])}")
    
    # Show failed files
    if results['failed']:
        print(f"\n❌ FAILED FILES:")
        for failure in results['failed'][:5]:
            print(f"  ├─ {Path(failure['file']).name}")
            print(f"  │  Error: {failure['error'][:60]}")
        if len(results['failed']) > 5:
            print(f"  └─ ... and {len(results['failed']) - 5} more")
    
    # Entity statistics
    print(f"\n📊 ENTITIES:")
    print(f"  Total: {results['total_entities']}")
    for entity_type, count in sorted(results['entity_breakdown'].items()):
        print(f"    {entity_type}: {count}")
    
    # Relationship statistics
    print(f"\n🔗 RELATIONSHIPS:")
    print(f"  Total: {results['total_relationships']}")
    for rel_type, count in sorted(results['relationship_breakdown'].items()):
        print(f"    {rel_type}: {count}")
    
    # External references
    if results['external_references']:
        print(f"\n🌐 EXTERNAL REFERENCES: {len(results['external_references'])}")
        ext_refs = list(results['external_references'])
        print(f"  {', '.join(ext_refs[:10])}")
        if len(ext_refs) > 10:
            print(f"  ... and {len(ext_refs) - 10} more")
    
    # Resolution statistics
    if resolution:
        print(f"\n🔍 REFERENCE RESOLUTION:")
        print(f"  Total external references: {resolution['total_external']}")
        print(f"  Resolved: {len(resolution['resolved'])}")
        print(f"  Unresolved: {len(resolution['unresolved'])}")
        
        if resolution['resolved']:
            print(f"\n  ✓ Resolved references:")
            for item in resolution['resolved'][:5]:
                print(f"    ├─ {item['name']}")
                print(f"    │  Found in: {Path(item['real_file']).name}")
                print(f"    │  Callers: {item['caller_count']}")
            if len(resolution['resolved']) > 5:
                print(f"    └─ ... and {len(resolution['resolved']) - 5} more")
        
        if resolution['unresolved']:
            print(f"\n  ⚠ Unresolved references:")
            for item in resolution['unresolved'][:10]:
                print(f"    - {item['name']}")
            if len(resolution['unresolved']) > 10:
                print(f"    ... and {len(resolution['unresolved']) - 10} more")
    
    # Performance
    print(f"\n⏱️  PERFORMANCE:")
    print(f"  Total time: {results['parse_time']:.2f} seconds")
    if results['file_count'] > 0:
        print(f"  Average per file: {results['parse_time'] / results['file_count']:.2f} seconds")
    
    print(f"\n{'='*70}\n")


def export_knowledge_graph(kg: KnowledgeGraph, output_dir: str = "./output"):
    """
    Export knowledge graph in multiple formats
    
    Args:
        kg: Knowledge graph to export
        output_dir: Output directory for export files
    """
    from pathlib import Path
    import json
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    print(f"\n{'='*70}")
    print("EXPORTING KNOWLEDGE GRAPH")
    print(f"{'='*70}\n")
    
    # Export full graph to JSON
    json_file = output_path / "knowledge_graph.json"
    kg.save_to_json(str(json_file))
    print(f"✓ Full graph: {json_file}")
    
    # Export procedures summary
    procedures = kg.query_entities(entity_type=EntityType.PROCEDURE)
    procedures_data = []
    for proc in procedures:
        proc_info = {
            'name': proc.name,
            'file': proc.file_path,
            'line': proc.start_line,
            'return_type': proc.metadata.get('return_type'),
            'parameters': proc.metadata.get('parameters', []),
            'is_main': proc.metadata.get('is_main', False),
            'is_external': proc.metadata.get('is_external', False),
            'statement_count': proc.metadata.get('statement_count', 0)
        }
        procedures_data.append(proc_info)
    
    proc_file = output_path / "procedures.json"
    with open(proc_file, 'w') as f:
        json.dump(procedures_data, f, indent=2)
    print(f"✓ Procedures: {proc_file}")
    
    # Export call graph
    call_relationships = kg.query_relationships(rel_type=RelationType.CALLS)
    call_graph_data = []
    for rel in call_relationships:
        caller = kg.get_entity(rel.source_id)
        callee = kg.get_entity(rel.target_id)
        if caller and callee:
            call_graph_data.append({
                'caller': caller.name,
                'caller_file': caller.file_path,
                'callee': callee.name,
                'callee_file': callee.file_path,
                'line': rel.metadata.get('line'),
                'external': rel.metadata.get('external', False)
            })
    
    call_file = output_path / "call_graph.json"
    with open(call_file, 'w') as f:
        json.dump(call_graph_data, f, indent=2)
    print(f"✓ Call graph: {call_file}")
    
    # Export statistics
    stats = kg.get_statistics()
    stats_file = output_path / "statistics.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"✓ Statistics: {stats_file}")
    
    print(f"\n{'='*70}\n")


# ============================================================================
# Graph Export Functions (NEW)
# ============================================================================

def export_for_visualization(kg: KnowledgeGraph, 
                             include_files: bool = False,
                             include_variables: bool = False,
                             max_nodes: int = 500) -> Dict[str, Any]:
    """
    Export knowledge graph in visualization-ready format
    
    Args:
        kg: Knowledge graph to export
        include_files: Include file entities
        include_variables: Include variable entities
        max_nodes: Maximum number of nodes to export
    
    Returns:
        Dict with nodes and edges ready for visualization
    """
    all_entities = kg.query_entities()
    
    # Filter entities
    filtered_entities = []
    for entity in all_entities:
        if entity.type == EntityType.FILE and not include_files:
            continue
        if entity.type == EntityType.VARIABLE and not include_variables:
            continue
        filtered_entities.append(entity)
    
    # Limit nodes
    if len(filtered_entities) > max_nodes:
        logger.info(f"Limiting to {max_nodes} nodes (found {len(filtered_entities)})")
        procedures = [e for e in filtered_entities if e.type == EntityType.PROCEDURE]
        others = [e for e in filtered_entities if e.type != EntityType.PROCEDURE]
        filtered_entities = procedures[:int(max_nodes * 0.8)] + others[:int(max_nodes * 0.2)]
    
    # Get relationships
    all_relationships = kg.query_relationships()
    entity_ids = {e.id for e in filtered_entities}
    filtered_relationships = [r for r in all_relationships 
                             if r.source_id in entity_ids and r.target_id in entity_ids]
    
    # Convert to serializable format
    nodes = []
    for entity in filtered_entities:
        node = {
            'id': entity.id,
            'name': entity.name,
            'type': entity.type.value,
            'qualified_name': entity.qualified_name,
            'file_path': entity.file_path,
            'start_line': entity.start_line,
            'end_line': entity.end_line,
            'language': entity.language,
            'metadata': entity.metadata
        }
        nodes.append(node)
    
    edges = []
    for rel in filtered_relationships:
        edge = {
            'source': rel.source_id,
            'target': rel.target_id,
            'type': rel.type.value,
            'weight': rel.weight,
            'metadata': rel.metadata
        }
        edges.append(edge)
    
    return {
        'nodes': nodes,
        'edges': edges,
        'statistics': kg.get_statistics(),
        'metadata': {
            'total_entities': len(all_entities),
            'filtered_entities': len(filtered_entities),
            'total_relationships': len(all_relationships),
            'filtered_relationships': len(filtered_relationships),
            'include_files': include_files,
            'include_variables': include_variables
        }
    }


def export_knowledge_graph(kg: KnowledgeGraph, output_dir: str = "./output") -> str:
    """
    Export knowledge graph in multiple formats including visualization-ready JSON
    
    Args:
        kg: Knowledge graph to export
        output_dir: Output directory for export files
    
    Returns:
        Path to visualization data file
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    print(f"\n{'='*70}")
    print("EXPORTING KNOWLEDGE GRAPH")
    print(f"{'='*70}\n")
    
    # Export full graph to JSON
    json_file = output_path / "knowledge_graph.json"
    kg.save_to_json(str(json_file))
    print(f"✓ Full graph: {json_file}")
    
    # Export visualization-ready format (THIS IS THE KEY ONE)
    vis_data = export_for_visualization(kg, include_files=False, include_variables=False)
    vis_file = output_path / "graph_data.json"
    with open(vis_file, 'w') as f:
        json.dump(vis_data, f, indent=2)
    print(f"✓ Visualization data: {vis_file}")
    
    # Export procedures summary
    procedures = kg.query_entities(entity_type=EntityType.PROCEDURE)
    procedures_data = []
    for proc in procedures:
        proc_info = {
            'name': proc.name,
            'file': proc.file_path,
            'line': proc.start_line,
            'return_type': proc.metadata.get('return_type'),
            'parameters': proc.metadata.get('parameters', []),
            'is_main': proc.metadata.get('is_main', False),
            'is_external': proc.metadata.get('is_external', False),
            'statement_count': proc.metadata.get('statement_count', 0)
        }
        procedures_data.append(proc_info)
    
    proc_file = output_path / "procedures.json"
    with open(proc_file, 'w') as f:
        json.dump(procedures_data, f, indent=2)
    print(f"✓ Procedures: {proc_file}")
    
    # Export call graph
    call_relationships = kg.query_relationships(rel_type=RelationType.CALLS)
    call_graph_data = []
    for rel in call_relationships:
        caller = kg.get_entity(rel.source_id)
        callee = kg.get_entity(rel.target_id)
        if caller and callee:
            call_graph_data.append({
                'caller': caller.name,
                'caller_file': caller.file_path,
                'callee': callee.name,
                'callee_file': callee.file_path,
                'line': rel.metadata.get('line'),
                'external': rel.metadata.get('external', False)
            })
    
    call_file = output_path / "call_graph.json"
    with open(call_file, 'w') as f:
        json.dump(call_graph_data, f, indent=2)
    print(f"✓ Call graph: {call_file}")
    
    # Export statistics
    stats = kg.get_statistics()
    stats_file = output_path / "statistics.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"✓ Statistics: {stats_file}")
    
    print(f"\n{'='*70}\n")
    
    return str(vis_file)



def main():
    """
    Main function for parsing TAL directories and building knowledge graph
    
    Usage:
        python parsers.py <directory> [options]
    
    Options:
        --search <term>      Search for functionality after parsing
        --export <dir>       Export knowledge graph to directory
        --no-recursive       Don't search subdirectories
        --no-resolve         Don't resolve external references
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Parse TAL files and build knowledge graph',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Parse all TAL files in directory (recursive)
  python parsers.py ./tal_source
  
  # Parse and search for functionality
  python parsers.py ./tal_source --search drawdown
  
  # Parse and export results
  python parsers.py ./tal_source --export ./output
  
  # Parse non-recursively
  python parsers.py ./tal_source --no-recursive
        """
    )
    
    parser.add_argument('directory', help='Directory containing TAL files')
    parser.add_argument('--search', metavar='TERM', help='Search for functionality after parsing')
    parser.add_argument('--export', metavar='DIR', help='Export knowledge graph to directory')
    parser.add_argument('--no-recursive', action='store_true', help="Don't search subdirectories")
    parser.add_argument('--no-resolve', action='store_true', help="Don't resolve external references")
    parser.add_argument('--backend', default='networkx', choices=['networkx', 'kuzu'], 
                       help='Graph database backend (default: networkx)')
    
    args = parser.parse_args()
    
    # Validate directory
    dir_path = Path(args.directory)
    if not dir_path.exists():
        print(f"Error: Directory '{args.directory}' does not exist")
        sys.exit(1)
    
    if not dir_path.is_dir():
        print(f"Error: '{args.directory}' is not a directory")
        sys.exit(1)
    
    try:
        # Create knowledge graph
        print(f"\n{'='*70}")
        print("INITIALIZING KNOWLEDGE GRAPH")
        print(f"{'='*70}")
        print(f"Backend: {args.backend}")
        print(f"{'='*70}\n")
        
        kg = KnowledgeGraph(backend=args.backend)
        
        # Parse directory
        recursive = not args.no_recursive
        results = parse_tal_directory_recursive(args.directory, kg, recursive=recursive)
        
        if not results['success']:
            print(f"Error: {results.get('error', 'Unknown error')}")
            sys.exit(1)
        
        # Resolve external references
        resolution = None
        if not args.no_resolve and results['parsed_successfully']:
            print(f"\n{'='*70}")
            print("RESOLVING EXTERNAL REFERENCES")
            print(f"{'='*70}\n")
            
            resolution = resolve_external_references(kg)
            
            print(f"Resolution complete:")
            print(f"  Resolved: {len(resolution['resolved'])}")
            print(f"  Unresolved: {len(resolution['unresolved'])}")
        
        # Display summary
        display_parsing_summary(results, resolution)
        
        # Export if requested
        if args.export:
            export_knowledge_graph(kg, args.export)
        
        # Search if requested
        if args.search:
            print(f"\n{'='*70}")
            print(f"SEARCHING FOR: '{args.search}'")
            print(f"{'='*70}\n")
            
            search = KnowledgeGraphSearch(kg)
            search_results = search.find_by_functionality(args.search)
            search.display_search_results(search_results)
        else:
            # Show available search capabilities
            procedures = kg.query_entities(entity_type=EntityType.PROCEDURE)
            if procedures:
                print(f"\n{'='*70}")
                print("SAMPLE PROCEDURES")
                print(f"{'='*70}\n")
                
                # Show main procedures
                main_procs = [p for p in procedures if p.metadata.get('is_main')]
                if main_procs:
                    print("Main procedures:")
                    for proc in main_procs:
                        print(f"  ★ {proc.name} ({Path(proc.file_path).name})")
                
                # Show some regular procedures
                regular_procs = [p for p in procedures if not p.metadata.get('is_main') 
                                and not p.metadata.get('is_external')][:10]
                if regular_procs:
                    print(f"\nSample procedures:")
                    for proc in regular_procs:
                        print(f"  - {proc.name} ({Path(proc.file_path).name})")
                
                print(f"\n{'='*70}")
                print("NEXT STEPS")
                print(f"{'='*70}\n")
                print("To search for functionality:")
                print(f"  python parsers.py {args.directory} --search <term>\n")
                print("Examples:")
                print(f"  python parsers.py {args.directory} --search drawdown")
                print(f"  python parsers.py {args.directory} --search payment")
                print(f"  python parsers.py {args.directory} --search validation\n")
                
                if not args.export:
                    print("To export knowledge graph:")
                    print(f"  python parsers.py {args.directory} --export ./output\n")
        
        # Interactive mode hint
        print(f"{'='*70}")
        print("For interactive exploration, use:")
        print("  from parsers import KnowledgeGraphSearch")
        print("  search = KnowledgeGraphSearch(kg)")
        print("  results = search.find_by_functionality('your_term')")
        print(f"{'='*70}\n")
        
    except KeyboardInterrupt:
        print("\n\nParsing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)



# Note: Include all the other parser code from the original file here
# (TALParser, KnowledgeGraphSearch, etc.)
# For brevity, I'm showing just the new/modified functions



if __name__ == "__main__":
    print("Updated parsers.py with graph export functionality")
    print("Use export_knowledge_graph(kg, output_dir) to export graph data")
    main()
