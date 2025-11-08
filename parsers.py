"""
AST Parser Interface for Knowledge Graph
Provides pluggable parsers for TAL, COBOL, and other languages
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
        """
        Parse a file's AST and populate the knowledge graph
        
        Args:
            file_path: Path to the source file
            ast_data: AST data structure (format depends on parser)
        
        Returns:
            The file entity
        """
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
# TAL Parser
# ============================================================================

@dataclass
class TALProcedure:
    """Represents a TAL procedure from AST"""
    name: str
    start_line: int
    end_line: int
    parameters: List[str]
    local_variables: List[str]
    calls: List[str]
    uses_directives: List[str]  # ?SOURCE, ?SECTION, etc.


class TALParser(ASTParser):
    """Parser for TAL (Transaction Application Language) code"""
    
    def __init__(self, knowledge_graph: KnowledgeGraph):
        super().__init__(knowledge_graph)
        self.language = "TAL"
    
    def parse_file(self, file_path: str, ast_data: Any) -> Entity:
        """
        Parse TAL file AST
        
        Expected ast_data structure:
        {
            'procedures': [TALProcedure, ...],
            'globals': [...],
            'literals': [...],
            'directives': [...],
            'structures': [...]
        }
        """
        file_entity = self.get_or_create_file_entity(file_path, self.language)
        
        # Extract entities
        entities = self.extract_entities(ast_data, file_entity)
        
        # Extract relationships
        relationships = self.extract_relationships(ast_data, entities)
        
        # Add to knowledge graph
        for entity in entities:
            self.kg.add_entity(entity)
        
        for relationship in relationships:
            self.kg.add_relationship(relationship)
        
        logger.info(f"Parsed TAL file: {file_path} - {len(entities)} entities, {len(relationships)} relationships")
        return file_entity
    
    def extract_entities(self, ast_data: Dict[str, Any], file_entity: Entity) -> List[Entity]:
        """Extract entities from TAL AST"""
        entities = []
        
        # Extract procedures
        procedures = ast_data.get('procedures', [])
        for proc_data in procedures:
            if isinstance(proc_data, TALProcedure):
                proc = proc_data
            else:
                # Handle dictionary format
                proc = TALProcedure(
                    name=proc_data.get('name', ''),
                    start_line=proc_data.get('start_line', 0),
                    end_line=proc_data.get('end_line', 0),
                    parameters=proc_data.get('parameters', []),
                    local_variables=proc_data.get('local_variables', []),
                    calls=proc_data.get('calls', []),
                    uses_directives=proc_data.get('uses_directives', [])
                )
            
            proc_entity = Entity(
                id="",
                type=EntityType.PROCEDURE,
                name=proc.name,
                qualified_name=f"{file_entity.name}::{proc.name}",
                file_path=file_entity.file_path,
                start_line=proc.start_line,
                end_line=proc.end_line,
                language=self.language,
                metadata={
                    'parameters': proc.parameters,
                    'local_variables': proc.local_variables,
                    'call_count': len(proc.calls),
                    'uses_directives': proc.uses_directives
                }
            )
            entities.append(proc_entity)
            
            # Create DEFINES relationship
            self.kg.add_relationship(Relationship(
                source_id=file_entity.id,
                target_id=proc_entity.id,
                type=RelationType.DEFINES
            ))
            
            # Extract parameters as variables
            for param in proc.parameters:
                var_entity = Entity(
                    id="",
                    type=EntityType.VARIABLE,
                    name=param,
                    qualified_name=f"{proc.name}::{param}",
                    file_path=file_entity.file_path,
                    start_line=proc.start_line,
                    language=self.language,
                    metadata={'scope': 'parameter'}
                )
                entities.append(var_entity)
            
            # Extract local variables
            for var in proc.local_variables:
                var_entity = Entity(
                    id="",
                    type=EntityType.VARIABLE,
                    name=var,
                    qualified_name=f"{proc.name}::{var}",
                    file_path=file_entity.file_path,
                    language=self.language,
                    metadata={'scope': 'local'}
                )
                entities.append(var_entity)
        
        # Extract structures
        structures = ast_data.get('structures', [])
        for struct_data in structures:
            struct_entity = Entity(
                id="",
                type=EntityType.STRUCTURE,
                name=struct_data.get('name', ''),
                qualified_name=f"{file_entity.name}::{struct_data.get('name', '')}",
                file_path=file_entity.file_path,
                start_line=struct_data.get('start_line', 0),
                language=self.language,
                metadata={
                    'fields': struct_data.get('fields', []),
                    'size': struct_data.get('size', 0)
                }
            )
            entities.append(struct_entity)
        
        # Extract global variables
        globals_data = ast_data.get('globals', [])
        for global_var in globals_data:
            var_entity = Entity(
                id="",
                type=EntityType.VARIABLE,
                name=global_var.get('name', ''),
                qualified_name=f"{file_entity.name}::{global_var.get('name', '')}",
                file_path=file_entity.file_path,
                language=self.language,
                metadata={'scope': 'global', 'type': global_var.get('type', '')}
            )
            entities.append(var_entity)
        
        return entities
    
    def extract_relationships(self, ast_data: Dict[str, Any], entities: List[Entity]) -> List[Relationship]:
        """Extract relationships from TAL AST"""
        relationships = []
        
        # Build entity lookup
        entity_by_name = {e.name: e for e in entities}
        procedures_by_name = {e.name: e for e in entities if e.type == EntityType.PROCEDURE}
        
        # Extract procedure calls
        procedures = ast_data.get('procedures', [])
        for proc_data in procedures:
            if isinstance(proc_data, TALProcedure):
                proc = proc_data
            else:
                proc = TALProcedure(
                    name=proc_data.get('name', ''),
                    start_line=proc_data.get('start_line', 0),
                    end_line=proc_data.get('end_line', 0),
                    parameters=proc_data.get('parameters', []),
                    local_variables=proc_data.get('local_variables', []),
                    calls=proc_data.get('calls', []),
                    uses_directives=proc_data.get('uses_directives', [])
                )
            
            caller = procedures_by_name.get(proc.name)
            if not caller:
                continue
            
            for called_proc_name in proc.calls:
                callee = procedures_by_name.get(called_proc_name)
                if callee:
                    rel = Relationship(
                        source_id=caller.id,
                        target_id=callee.id,
                        type=RelationType.CALLS,
                        metadata={'line': proc.start_line}
                    )
                    relationships.append(rel)
        
        # Extract ?SOURCE includes
        directives = ast_data.get('directives', [])
        for directive in directives:
            if directive.get('type') == 'SOURCE':
                # Create INCLUDES relationship to source file
                included_file = directive.get('file', '')
                if included_file:
                    # This would need file entity from another parse
                    pass
        
        return relationships


# ============================================================================
# COBOL Parser
# ============================================================================

@dataclass
class COBOLParagraph:
    """Represents a COBOL paragraph"""
    name: str
    section: Optional[str]
    start_line: int
    end_line: int
    performs: List[str]  # Other paragraphs/sections performed
    calls: List[str]  # External programs called


class COBOLParser(ASTParser):
    """Parser for COBOL code"""
    
    def __init__(self, knowledge_graph: KnowledgeGraph):
        super().__init__(knowledge_graph)
        self.language = "COBOL"
    
    def parse_file(self, file_path: str, ast_data: Any) -> Entity:
        """
        Parse COBOL file AST
        
        Expected ast_data structure:
        {
            'program_id': str,
            'divisions': {...},
            'sections': [...],
            'paragraphs': [COBOLParagraph, ...],
            'data_items': [...],
            'copybooks': [...]
        }
        """
        file_entity = self.get_or_create_file_entity(file_path, self.language)
        
        # Extract entities
        entities = self.extract_entities(ast_data, file_entity)
        
        # Extract relationships
        relationships = self.extract_relationships(ast_data, entities)
        
        # Add to knowledge graph
        for entity in entities:
            self.kg.add_entity(entity)
        
        for relationship in relationships:
            self.kg.add_relationship(relationship)
        
        logger.info(f"Parsed COBOL file: {file_path} - {len(entities)} entities, {len(relationships)} relationships")
        return file_entity
    
    def extract_entities(self, ast_data: Dict[str, Any], file_entity: Entity) -> List[Entity]:
        """Extract entities from COBOL AST"""
        entities = []
        
        # Extract program as module
        program_id = ast_data.get('program_id', '')
        if program_id:
            module_entity = Entity(
                id="",
                type=EntityType.MODULE,
                name=program_id,
                qualified_name=program_id,
                file_path=file_entity.file_path,
                language=self.language,
                metadata={'type': 'program'}
            )
            entities.append(module_entity)
        
        # Extract divisions
        divisions = ast_data.get('divisions', {})
        for div_name, div_data in divisions.items():
            div_entity = Entity(
                id="",
                type=EntityType.DIVISION,
                name=div_name,
                qualified_name=f"{program_id}::{div_name}",
                file_path=file_entity.file_path,
                language=self.language,
                metadata=div_data
            )
            entities.append(div_entity)
        
        # Extract sections
        sections = ast_data.get('sections', [])
        for section_data in sections:
            section_entity = Entity(
                id="",
                type=EntityType.SECTION,
                name=section_data.get('name', ''),
                qualified_name=f"{program_id}::{section_data.get('name', '')}",
                file_path=file_entity.file_path,
                start_line=section_data.get('start_line', 0),
                end_line=section_data.get('end_line', 0),
                language=self.language,
                metadata={'division': section_data.get('division', '')}
            )
            entities.append(section_entity)
        
        # Extract paragraphs
        paragraphs = ast_data.get('paragraphs', [])
        for para_data in paragraphs:
            if isinstance(para_data, COBOLParagraph):
                para = para_data
            else:
                para = COBOLParagraph(
                    name=para_data.get('name', ''),
                    section=para_data.get('section'),
                    start_line=para_data.get('start_line', 0),
                    end_line=para_data.get('end_line', 0),
                    performs=para_data.get('performs', []),
                    calls=para_data.get('calls', [])
                )
            
            para_entity = Entity(
                id="",
                type=EntityType.PARAGRAPH,
                name=para.name,
                qualified_name=f"{program_id}::{para.section or 'main'}::{para.name}",
                file_path=file_entity.file_path,
                start_line=para.start_line,
                end_line=para.end_line,
                language=self.language,
                metadata={
                    'section': para.section,
                    'performs': para.performs,
                    'calls': para.calls
                }
            )
            entities.append(para_entity)
        
        # Extract data items
        data_items = ast_data.get('data_items', [])
        for data_item in data_items:
            var_entity = Entity(
                id="",
                type=EntityType.VARIABLE,
                name=data_item.get('name', ''),
                qualified_name=f"{program_id}::{data_item.get('name', '')}",
                file_path=file_entity.file_path,
                start_line=data_item.get('line', 0),
                language=self.language,
                metadata={
                    'level': data_item.get('level', ''),
                    'picture': data_item.get('picture', ''),
                    'usage': data_item.get('usage', ''),
                    'value': data_item.get('value', '')
                }
            )
            entities.append(var_entity)
        
        return entities
    
    def extract_relationships(self, ast_data: Dict[str, Any], entities: List[Entity]) -> List[Relationship]:
        """Extract relationships from COBOL AST"""
        relationships = []
        
        # Build entity lookup
        paragraphs_by_name = {e.name: e for e in entities if e.type == EntityType.PARAGRAPH}
        sections_by_name = {e.name: e for e in entities if e.type == EntityType.SECTION}
        
        # Extract PERFORM relationships
        paragraphs = ast_data.get('paragraphs', [])
        for para_data in paragraphs:
            if isinstance(para_data, COBOLParagraph):
                para = para_data
            else:
                para = COBOLParagraph(
                    name=para_data.get('name', ''),
                    section=para_data.get('section'),
                    start_line=para_data.get('start_line', 0),
                    end_line=para_data.get('end_line', 0),
                    performs=para_data.get('performs', []),
                    calls=para_data.get('calls', [])
                )
            
            source = paragraphs_by_name.get(para.name)
            if not source:
                continue
            
            # PERFORM relationships
            for performed in para.performs:
                target = paragraphs_by_name.get(performed) or sections_by_name.get(performed)
                if target:
                    rel = Relationship(
                        source_id=source.id,
                        target_id=target.id,
                        type=RelationType.PERFORMS,
                        metadata={'line': para.start_line}
                    )
                    relationships.append(rel)
            
            # CALL relationships (external programs)
            for called_program in para.calls:
                # Would need to link to external program entity
                pass
        
        # Extract COPY relationships
        copybooks = ast_data.get('copybooks', [])
        for copybook in copybooks:
            # Create COPIES relationship
            pass
        
        return relationships


# ============================================================================
# Generic Parser for Other Languages
# ============================================================================

class GenericParser(ASTParser):
    """Generic parser for languages with standard AST structures"""
    
    def __init__(self, knowledge_graph: KnowledgeGraph, language: str):
        super().__init__(knowledge_graph)
        self.language = language
    
    def parse_file(self, file_path: str, ast_data: Any) -> Entity:
        """Parse generic AST structure"""
        file_entity = self.get_or_create_file_entity(file_path, self.language)
        
        entities = self.extract_entities(ast_data, file_entity)
        relationships = self.extract_relationships(ast_data, entities)
        
        for entity in entities:
            self.kg.add_entity(entity)
        
        for relationship in relationships:
            self.kg.add_relationship(relationship)
        
        return file_entity
    
    def extract_entities(self, ast_data: Dict[str, Any], file_entity: Entity) -> List[Entity]:
        """Extract entities from generic AST"""
        entities = []
        
        # Extract classes
        for class_data in ast_data.get('classes', []):
            class_entity = Entity(
                id="",
                type=EntityType.CLASS,
                name=class_data.get('name', ''),
                qualified_name=f"{file_entity.name}::{class_data.get('name', '')}",
                file_path=file_entity.file_path,
                start_line=class_data.get('start_line', 0),
                end_line=class_data.get('end_line', 0),
                language=self.language,
                metadata=class_data.get('metadata', {})
            )
            entities.append(class_entity)
        
        # Extract functions
        for func_data in ast_data.get('functions', []):
            func_entity = Entity(
                id="",
                type=EntityType.FUNCTION,
                name=func_data.get('name', ''),
                qualified_name=f"{file_entity.name}::{func_data.get('name', '')}",
                file_path=file_entity.file_path,
                start_line=func_data.get('start_line', 0),
                end_line=func_data.get('end_line', 0),
                language=self.language,
                metadata=func_data.get('metadata', {})
            )
            entities.append(func_entity)
        
        return entities
    
    def extract_relationships(self, ast_data: Dict[str, Any], entities: List[Entity]) -> List[Relationship]:
        """Extract relationships from generic AST"""
        relationships = []
        
        # Extract call relationships
        for call_data in ast_data.get('calls', []):
            rel = Relationship(
                source_id=call_data['source_id'],
                target_id=call_data['target_id'],
                type=RelationType.CALLS,
                metadata=call_data.get('metadata', {})
            )
            relationships.append(rel)
        
        return relationships


# ============================================================================
# Parser Factory
# ============================================================================

class ParserFactory:
    """Factory for creating appropriate parsers"""
    
    @staticmethod
    def create_parser(language: str, knowledge_graph: KnowledgeGraph) -> ASTParser:
        """Create parser for given language"""
        language_upper = language.upper()
        
        if language_upper == "TAL":
            return TALParser(knowledge_graph)
        elif language_upper == "COBOL":
            return COBOLParser(knowledge_graph)
        else:
            return GenericParser(knowledge_graph, language)


# ============================================================================
# Batch Parser for Multiple Files
# ============================================================================

class BatchParser:
    """Parse multiple files in batch"""
    
    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.kg = knowledge_graph
        self.parsers: Dict[str, ASTParser] = {}
    
    def parse_files(self, file_data: List[Dict[str, Any]]) -> None:
        """
        Parse multiple files
        
        Args:
            file_data: List of dicts with 'file_path', 'language', and 'ast_data'
        """
        for file_info in file_data:
            file_path = file_info['file_path']
            language = file_info['language']
            ast_data = file_info['ast_data']
            
            # Get or create parser
            if language not in self.parsers:
                self.parsers[language] = ParserFactory.create_parser(language, self.kg)
            
            parser = self.parsers[language]
            
            try:
                parser.parse_file(file_path, ast_data)
            except Exception as e:
                logger.error(f"Error parsing {file_path}: {e}")
        
        logger.info(f"Batch parse complete: {len(file_data)} files")
    
    def parse_directory(self, directory: str, language: str,
                       ast_provider: callable) -> None:
        """
        Parse all files in a directory
        
        Args:
            directory: Directory path
            language: Language of files
            ast_provider: Function that takes file_path and returns AST data
        """
        from pathlib import Path
        
        path = Path(directory)
        files = list(path.rglob(f"*.{language.lower()}"))
        
        file_data = []
        for file_path in files:
            try:
                ast_data = ast_provider(str(file_path))
                file_data.append({
                    'file_path': str(file_path),
                    'language': language,
                    'ast_data': ast_data
                })
            except Exception as e:
                logger.error(f"Error getting AST for {file_path}: {e}")
        
        self.parse_files(file_data)
