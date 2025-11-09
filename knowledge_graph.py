"""
Python Knowledge Graph for Code Analysis
Inspired by GitLab Knowledge Graph, optimized for TAL and COBOL AST processing

Core Features:
- Pluggable AST parsers for TAL/COBOL/other languages
- Entity extraction (files, procedures, functions, classes, variables)
- Relationship tracking (calls, imports, dependencies, data flow)
- Multiple backend support (Kuzu, NetworkX, Neo4j)
- RAG-ready export formats
- Cypher-like query interface

HASHABILITY:
- Entity and Relationship classes are hashable for use in sets/dicts
- Hashing based on immutable identifying fields only
- Equality checks use same fields as hashing
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Set, Optional, Any, Tuple, Union
from enum import Enum
from pathlib import Path
import json
import logging
from datetime import datetime
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Core Data Models
# ============================================================================

class EntityType(Enum):
    """Entity types in the knowledge graph"""
    FILE = "file"
    DIRECTORY = "directory"
    MODULE = "module"
    PROCEDURE = "procedure"  # TAL procedures
    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    VARIABLE = "variable"
    CONSTANT = "constant"
    TYPE_DEFINITION = "type_definition"
    STRUCTURE = "structure"  # COBOL/TAL structures
    PARAGRAPH = "paragraph"  # COBOL paragraphs
    SECTION = "section"  # COBOL sections
    DIVISION = "division"  # COBOL divisions
    

class RelationType(Enum):
    """Relationship types between entities"""
    CONTAINS = "contains"  # Directory contains file, file contains procedure
    CALLS = "calls"  # Procedure/function calls another
    IMPORTS = "imports"  # Module imports another
    INHERITS = "inherits"  # Class inheritance
    REFERENCES = "references"  # Variable/data reference
    DEFINES = "defines"  # File defines procedure
    USES = "uses"  # Uses a type or variable
    DEPENDS_ON = "depends_on"  # General dependency
    PERFORMS = "performs"  # COBOL PERFORM
    COPIES = "copies"  # COBOL COPY statement
    INCLUDES = "includes"  # TAL ?SOURCE directive
    DATA_FLOW = "data_flow"  # Data flows from A to B
    

@dataclass
class Entity:
    """
    Represents a code entity in the knowledge graph
    
    HASHABILITY:
    ------------
    This class is hashable to support:
    - Use in sets for deduplication
    - Use as dictionary keys for fast lookups
    - Graph database operations
    
    Hashing is based on immutable identifying fields:
    - id: Unique identifier
    - type: Entity type (enum)
    - qualified_name: Fully qualified name
    
    Mutable fields (metadata, created_at) are NOT used in hashing to ensure
    hash stability even when these fields change.
    
    IMPORTANT: Two entities with the same id, type, and qualified_name are
    considered equal, even if they have different metadata or timestamps.
    """
    id: str
    type: EntityType
    name: str
    qualified_name: str  # Fully qualified name
    file_path: Optional[str] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    start_column: Optional[int] = None
    end_column: Optional[int] = None
    language: Optional[str] = None  # TAL, COBOL, etc.
    metadata: Dict[str, Any] = field(default_factory=dict)
    content_hash: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if isinstance(self.type, str):
            self.type = EntityType(self.type)
        if not self.id:
            self.id = self._generate_id()
    
    def __hash__(self) -> int:
        """
        Make Entity hashable based on immutable identifying fields
        
        Only uses id, type, and qualified_name for hashing because:
        - These fields uniquely identify an entity
        - They are immutable after creation
        - metadata and created_at can change over time
        
        This ensures hash stability and allows entities to be used in
        sets and as dictionary keys.
        """
        return hash((self.id, self.type, self.qualified_name))
    
    def __eq__(self, other) -> bool:
        """
        Check equality based on the same fields used for hashing
        
        Two entities are equal if they have the same id, type, and
        qualified_name, regardless of other fields like metadata.
        
        This is consistent with __hash__ to satisfy Python's requirement
        that objects which compare equal must have the same hash value.
        """
        if not isinstance(other, Entity):
            return NotImplemented
        return (
            self.id == other.id and 
            self.type == other.type and 
            self.qualified_name == other.qualified_name
        )
    
    def _generate_id(self) -> str:
        """Generate unique ID based on entity properties"""
        key = f"{self.type.value}:{self.qualified_name}:{self.file_path or ''}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['type'] = self.type.value
        data['created_at'] = self.created_at.isoformat()
        return data
    
    def __repr__(self) -> str:
        """Provide clean string representation"""
        return f"Entity(id='{self.id}', type={self.type.value}, name='{self.name}')"


@dataclass
class Relationship:
    """
    Represents a relationship between entities
    
    HASHABILITY:
    ------------
    This class is hashable to support:
    - Use in sets for deduplication
    - Use as dictionary keys for relationship lookups
    - Graph database operations
    
    Hashing is based on immutable identifying fields:
    - source_id: Source entity ID
    - target_id: Target entity ID
    - type: Relationship type (enum)
    
    Mutable fields (metadata, weight, created_at) are NOT used in hashing
    to ensure hash stability.
    
    IMPORTANT: Two relationships with the same source_id, target_id, and type
    are considered equal, even if they have different metadata or weights.
    """
    source_id: str
    target_id: str
    type: RelationType
    metadata: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if isinstance(self.type, str):
            self.type = RelationType(self.type)
    
    def __hash__(self) -> int:
        """
        Make Relationship hashable based on immutable identifying fields
        
        Only uses source_id, target_id, and type for hashing because:
        - These fields uniquely identify a relationship
        - They are immutable after creation
        - metadata, weight, and created_at can change over time
        
        This ensures hash stability and allows relationships to be used
        in sets and as dictionary keys.
        """
        return hash((self.source_id, self.target_id, self.type))
    
    def __eq__(self, other) -> bool:
        """
        Check equality based on the same fields used for hashing
        
        Two relationships are equal if they have the same source_id,
        target_id, and type, regardless of metadata or weight.
        
        This is consistent with __hash__ to satisfy Python's requirement
        that objects which compare equal must have the same hash value.
        """
        if not isinstance(other, Relationship):
            return NotImplemented
        return (
            self.source_id == other.source_id and 
            self.target_id == other.target_id and 
            self.type == other.type
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['type'] = self.type.value
        data['created_at'] = self.created_at.isoformat()
        return data
    
    def __repr__(self) -> str:
        """Provide clean string representation"""
        return f"Relationship(source='{self.source_id}', target='{self.target_id}', type={self.type.value})"


# ============================================================================
# Abstract Database Interface
# ============================================================================

class GraphDatabase(ABC):
    """Abstract interface for graph database backends"""
    
    @abstractmethod
    def add_entity(self, entity: Entity) -> None:
        """Add an entity to the graph"""
        pass
    
    @abstractmethod
    def add_relationship(self, relationship: Relationship) -> None:
        """Add a relationship to the graph"""
        pass
    
    @abstractmethod
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Retrieve an entity by ID"""
        pass
    
    @abstractmethod
    def query_entities(self, 
                      entity_type: Optional[EntityType] = None,
                      filters: Optional[Dict[str, Any]] = None) -> List[Entity]:
        """Query entities with optional filters"""
        pass
    
    @abstractmethod
    def query_relationships(self,
                          source_id: Optional[str] = None,
                          target_id: Optional[str] = None,
                          rel_type: Optional[RelationType] = None) -> List[Relationship]:
        """Query relationships"""
        pass
    
    @abstractmethod
    def get_neighbors(self, entity_id: str, 
                     rel_type: Optional[RelationType] = None,
                     direction: str = "outgoing") -> List[Entity]:
        """Get neighboring entities"""
        pass
    
    @abstractmethod
    def execute_query(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        """Execute a database-specific query"""
        pass
    
    @abstractmethod
    def export_subgraph(self, entity_ids: List[str]) -> Dict[str, Any]:
        """Export a subgraph for RAG or analysis"""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all data from the database"""
        pass
    
    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        pass


# ============================================================================
# NetworkX Implementation (In-Memory)
# ============================================================================

import networkx as nx

class NetworkXDatabase(GraphDatabase):
    """In-memory implementation using NetworkX"""
    
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.entities: Dict[str, Entity] = {}
        
    def add_entity(self, entity: Entity) -> None:
        """Add entity as a node"""
        self.entities[entity.id] = entity
        self.graph.add_node(entity.id, **entity.to_dict())
        logger.debug(f"Added entity: {entity.type.value} - {entity.name}")
    
    def add_relationship(self, relationship: Relationship) -> None:
        """Add relationship as an edge"""
        if relationship.source_id not in self.entities:
            logger.warning(f"Source entity {relationship.source_id} not found")
            return
        if relationship.target_id not in self.entities:
            logger.warning(f"Target entity {relationship.target_id} not found")
            return
            
        self.graph.add_edge(
            relationship.source_id,
            relationship.target_id,
            key=relationship.type.value,
            **relationship.to_dict()
        )
        logger.debug(f"Added relationship: {relationship.type.value}")
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Retrieve entity by ID"""
        return self.entities.get(entity_id)
    
    def query_entities(self,
                      entity_type: Optional[EntityType] = None,
                      filters: Optional[Dict[str, Any]] = None) -> List[Entity]:
        """Query entities with filters"""
        results = []
        for entity in self.entities.values():
            if entity_type and entity.type != entity_type:
                continue
            if filters:
                match = True
                for key, value in filters.items():
                    if getattr(entity, key, None) != value and entity.metadata.get(key) != value:
                        match = False
                        break
                if not match:
                    continue
            results.append(entity)
        return results
    
    def query_relationships(self,
                          source_id: Optional[str] = None,
                          target_id: Optional[str] = None,
                          rel_type: Optional[RelationType] = None) -> List[Relationship]:
        """Query relationships"""
        results = []
        edges = self.graph.edges(data=True, keys=True)
        
        for src, tgt, key, data in edges:
            if source_id and src != source_id:
                continue
            if target_id and tgt != target_id:
                continue
            if rel_type and data.get('type') != rel_type.value:
                continue
            
            rel = Relationship(
                source_id=src,
                target_id=tgt,
                type=RelationType(data['type']),
                metadata=data.get('metadata', {}),
                weight=data.get('weight', 1.0)
            )
            results.append(rel)
        return results
    
    def get_neighbors(self, entity_id: str,
                     rel_type: Optional[RelationType] = None,
                     direction: str = "outgoing") -> List[Entity]:
        """
        Get neighboring entities
        
        Note: Uses set() for deduplication, which requires Entity to be hashable
        """
        if entity_id not in self.graph:
            return []
        
        neighbors = []
        if direction in ("outgoing", "both"):
            for successor in self.graph.successors(entity_id):
                if rel_type:
                    edges = self.graph.get_edge_data(entity_id, successor)
                    if any(data.get('type') == rel_type.value for data in edges.values()):
                        neighbors.append(self.entities[successor])
                else:
                    neighbors.append(self.entities[successor])
        
        if direction in ("incoming", "both"):
            for predecessor in self.graph.predecessors(entity_id):
                if rel_type:
                    edges = self.graph.get_edge_data(predecessor, entity_id)
                    if any(data.get('type') == rel_type.value for data in edges.values()):
                        neighbors.append(self.entities[predecessor])
                else:
                    neighbors.append(self.entities[predecessor])
        
        # Remove duplicates using set (requires Entity to be hashable)
        return list(set(neighbors))
    
    def execute_query(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        """Execute a custom query (simplified implementation)"""
        # This is a simplified query interface
        # In practice, you'd implement a query parser
        raise NotImplementedError("Custom queries not implemented for NetworkX backend")
    
    def export_subgraph(self, entity_ids: List[str]) -> Dict[str, Any]:
        """Export subgraph as JSON"""
        subgraph = self.graph.subgraph(entity_ids)
        
        entities_data = [
            self.entities[nid].to_dict() 
            for nid in subgraph.nodes()
        ]
        
        relationships_data = []
        for src, tgt, key, data in subgraph.edges(data=True, keys=True):
            relationships_data.append({
                'source_id': src,
                'target_id': tgt,
                'type': data['type'],
                'metadata': data.get('metadata', {}),
                'weight': data.get('weight', 1.0)
            })
        
        return {
            'entities': entities_data,
            'relationships': relationships_data,
            'entity_count': len(entities_data),
            'relationship_count': len(relationships_data)
        }
    
    def clear(self) -> None:
        """Clear all data"""
        self.graph.clear()
        self.entities.clear()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics"""
        entity_counts = {}
        for entity in self.entities.values():
            entity_counts[entity.type.value] = entity_counts.get(entity.type.value, 0) + 1
        
        relationship_counts = {}
        for _, _, _, data in self.graph.edges(data=True, keys=True):
            rel_type = data['type']
            relationship_counts[rel_type] = relationship_counts.get(rel_type, 0) + 1
        
        return {
            'total_entities': len(self.entities),
            'total_relationships': self.graph.number_of_edges(),
            'entity_counts': entity_counts,
            'relationship_counts': relationship_counts,
            'avg_degree': sum(dict(self.graph.degree()).values()) / max(len(self.entities), 1)
        }


# ============================================================================
# Kuzu Implementation (For compatibility with GitLab KG)
# ============================================================================

try:
    import kuzu
    KUZU_AVAILABLE = True
except ImportError:
    KUZU_AVAILABLE = False
    logger.warning("Kuzu not installed. Install with: pip install kuzu")


class KuzuDatabase(GraphDatabase):
    """Kuzu graph database implementation"""
    
    def __init__(self, db_path: str = "./knowledge_graph_db"):
        if not KUZU_AVAILABLE:
            raise ImportError("Kuzu is not installed. Install with: pip install kuzu")
        
        self.db_path = Path(db_path)
        self.db_path.mkdir(exist_ok=True)
        self.db = kuzu.Database(str(self.db_path))
        self.conn = kuzu.Connection(self.db)
        self._init_schema()
    
    def _init_schema(self):
        """Initialize database schema"""
        # Create node tables for each entity type
        for entity_type in EntityType:
            try:
                self.conn.execute(f"""
                    CREATE NODE TABLE {entity_type.value} (
                        id STRING,
                        name STRING,
                        qualified_name STRING,
                        file_path STRING,
                        start_line INT64,
                        end_line INT64,
                        language STRING,
                        metadata STRING,
                        PRIMARY KEY (id)
                    )
                """)
            except Exception as e:
                logger.debug(f"Table {entity_type.value} may already exist: {e}")
        
        # Create relationship tables
        for rel_type in RelationType:
            try:
                # Create relationship table between any entity types
                self.conn.execute(f"""
                    CREATE REL TABLE {rel_type.value} (
                        FROM file TO file,
                        FROM procedure TO procedure,
                        FROM function TO function,
                        weight DOUBLE DEFAULT 1.0,
                        metadata STRING
                    )
                """)
            except Exception as e:
                logger.debug(f"Relationship {rel_type.value} may already exist: {e}")
    
    def add_entity(self, entity: Entity) -> None:
        """Add entity to Kuzu"""
        metadata_json = json.dumps(entity.metadata)
        
        query = f"""
            CREATE (:{entity.type.value} {{
                id: $id,
                name: $name,
                qualified_name: $qualified_name,
                file_path: $file_path,
                start_line: $start_line,
                end_line: $end_line,
                language: $language,
                metadata: $metadata
            }})
        """
        
        params = {
            'id': entity.id,
            'name': entity.name,
            'qualified_name': entity.qualified_name,
            'file_path': entity.file_path or '',
            'start_line': entity.start_line or 0,
            'end_line': entity.end_line or 0,
            'language': entity.language or '',
            'metadata': metadata_json
        }
        
        try:
            self.conn.execute(query, params)
        except Exception as e:
            logger.error(f"Error adding entity: {e}")
    
    def add_relationship(self, relationship: Relationship) -> None:
        """Add relationship to Kuzu"""
        metadata_json = json.dumps(relationship.metadata)
        
        query = f"""
            MATCH (a {{id: $source_id}}), (b {{id: $target_id}})
            CREATE (a)-[:{relationship.type.value} {{
                weight: $weight,
                metadata: $metadata
            }}]->(b)
        """
        
        params = {
            'source_id': relationship.source_id,
            'target_id': relationship.target_id,
            'weight': relationship.weight,
            'metadata': metadata_json
        }
        
        try:
            self.conn.execute(query, params)
        except Exception as e:
            logger.error(f"Error adding relationship: {e}")
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Retrieve entity by ID"""
        # Search across all entity tables
        for entity_type in EntityType:
            query = f"MATCH (n:{entity_type.value} {{id: $id}}) RETURN n"
            result = self.conn.execute(query, {'id': entity_id})
            
            rows = result.get_as_df()
            if not rows.empty:
                row = rows.iloc[0]['n']
                return Entity(
                    id=row['id'],
                    type=entity_type,
                    name=row['name'],
                    qualified_name=row['qualified_name'],
                    file_path=row.get('file_path'),
                    start_line=row.get('start_line'),
                    end_line=row.get('end_line'),
                    language=row.get('language'),
                    metadata=json.loads(row.get('metadata', '{}'))
                )
        return None
    
    def query_entities(self,
                      entity_type: Optional[EntityType] = None,
                      filters: Optional[Dict[str, Any]] = None) -> List[Entity]:
        """Query entities"""
        if entity_type:
            query = f"MATCH (n:{entity_type.value}) RETURN n"
        else:
            query = "MATCH (n) RETURN n"
        
        result = self.conn.execute(query)
        entities = []
        
        for row in result.get_as_df().itertuples():
            node = row.n
            entities.append(Entity(
                id=node['id'],
                type=entity_type or EntityType(node.get('type', 'file')),
                name=node['name'],
                qualified_name=node['qualified_name'],
                file_path=node.get('file_path'),
                metadata=json.loads(node.get('metadata', '{}'))
            ))
        
        return entities
    
    def query_relationships(self,
                          source_id: Optional[str] = None,
                          target_id: Optional[str] = None,
                          rel_type: Optional[RelationType] = None) -> List[Relationship]:
        """Query relationships"""
        conditions = []
        params = {}
        
        if source_id:
            conditions.append("a.id = $source_id")
            params['source_id'] = source_id
        if target_id:
            conditions.append("b.id = $target_id")
            params['target_id'] = target_id
        
        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        rel_pattern = f"[r:{rel_type.value}]" if rel_type else "[r]"
        
        query = f"""
            MATCH (a)-{rel_pattern}->(b)
            {where_clause}
            RETURN a.id as source_id, b.id as target_id, type(r) as type, r.weight as weight
        """
        
        result = self.conn.execute(query, params)
        relationships = []
        
        for row in result.get_as_df().itertuples():
            relationships.append(Relationship(
                source_id=row.source_id,
                target_id=row.target_id,
                type=RelationType(row.type),
                weight=row.weight
            ))
        
        return relationships
    
    def get_neighbors(self, entity_id: str,
                     rel_type: Optional[RelationType] = None,
                     direction: str = "outgoing") -> List[Entity]:
        """Get neighboring entities"""
        rel_pattern = f"[r:{rel_type.value}]" if rel_type else "[r]"
        
        if direction == "outgoing":
            query = f"MATCH (n {{id: $id}})-{rel_pattern}->(m) RETURN m"
        elif direction == "incoming":
            query = f"MATCH (n {{id: $id}})<-{rel_pattern}-(m) RETURN m"
        else:  # both
            query = f"MATCH (n {{id: $id}})-{rel_pattern}-(m) RETURN m"
        
        result = self.conn.execute(query, {'id': entity_id})
        neighbors = []
        
        for row in result.get_as_df().itertuples():
            node = row.m
            neighbors.append(Entity(
                id=node['id'],
                type=EntityType(node.get('type', 'file')),
                name=node['name'],
                qualified_name=node['qualified_name']
            ))
        
        return neighbors
    
    def execute_query(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        """Execute Cypher query"""
        result = self.conn.execute(query, params or {})
        return result.get_as_df().to_dict('records')
    
    def export_subgraph(self, entity_ids: List[str]) -> Dict[str, Any]:
        """Export subgraph"""
        # Implementation similar to NetworkX version
        entities = [self.get_entity(eid) for eid in entity_ids if self.get_entity(eid)]
        
        return {
            'entities': [e.to_dict() for e in entities],
            'relationships': [],
            'entity_count': len(entities)
        }
    
    def clear(self) -> None:
        """Clear database"""
        for entity_type in EntityType:
            try:
                self.conn.execute(f"MATCH (n:{entity_type.value}) DELETE n")
            except Exception:
                pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics"""
        stats = {
            'total_entities': 0,
            'entity_counts': {}
        }
        
        for entity_type in EntityType:
            try:
                result = self.conn.execute(f"MATCH (n:{entity_type.value}) RETURN COUNT(n) as count")
                count = result.get_as_df().iloc[0]['count']
                stats['entity_counts'][entity_type.value] = count
                stats['total_entities'] += count
            except Exception:
                pass
        
        return stats


# ============================================================================
# Main Knowledge Graph Class
# ============================================================================

class KnowledgeGraph:
    """Main knowledge graph interface"""
    
    def __init__(self, backend: str = "networkx", db_path: Optional[str] = None):
        """
        Initialize knowledge graph
        
        Args:
            backend: Database backend ('networkx' or 'kuzu')
            db_path: Path for persistent storage (for kuzu)
        """
        if backend == "networkx":
            self.db = NetworkXDatabase()
        elif backend == "kuzu":
            self.db = KuzuDatabase(db_path or "./knowledge_graph_db")
        else:
            raise ValueError(f"Unknown backend: {backend}")
        
        self.backend = backend
        logger.info(f"Initialized KnowledgeGraph with {backend} backend")
    
    def add_entity(self, entity: Entity) -> None:
        """Add entity to graph"""
        self.db.add_entity(entity)
    
    def add_relationship(self, relationship: Relationship) -> None:
        """Add relationship to graph"""
        self.db.add_relationship(relationship)
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID"""
        return self.db.get_entity(entity_id)
    
    def query_entities(self, **kwargs) -> List[Entity]:
        """Query entities"""
        return self.db.query_entities(**kwargs)
    
    def query_relationships(self, **kwargs) -> List[Relationship]:
        """Query relationships"""
        return self.db.query_relationships(**kwargs)
    
    def get_neighbors(self, entity_id: str, **kwargs) -> List[Entity]:
        """Get neighbors"""
        return self.db.get_neighbors(entity_id, **kwargs)
    
    def find_call_chain(self, start_entity_id: str, end_entity_id: str) -> List[List[str]]:
        """Find call chains between two entities"""
        if self.backend != "networkx":
            raise NotImplementedError("Call chain finding only implemented for NetworkX")
        
        import networkx as nx
        
        # Create a view of only CALLS relationships
        call_graph = nx.DiGraph()
        for rel in self.db.query_relationships(rel_type=RelationType.CALLS):
            call_graph.add_edge(rel.source_id, rel.target_id)
        
        try:
            paths = list(nx.all_simple_paths(call_graph, start_entity_id, end_entity_id, cutoff=10))
            return paths
        except nx.NodeNotFound:
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get graph statistics"""
        return self.db.get_statistics()
    
    def export_for_rag(self, entity_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Export graph data for RAG systems"""
        if entity_ids:
            return self.db.export_subgraph(entity_ids)
        
        # Export all entities
        entities = self.db.query_entities()
        relationships = self.db.query_relationships()
        
        return {
            'entities': [e.to_dict() for e in entities],
            'relationships': [r.to_dict() for r in relationships],
            'entity_count': len(entities),
            'relationship_count': len(relationships),
            'statistics': self.get_statistics()
        }
    
    def clear(self) -> None:
        """Clear all data"""
        self.db.clear()
    
    def save_to_json(self, filepath: str) -> None:
        """Save graph to JSON file"""
        data = self.export_for_rag()
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Saved graph to {filepath}")
    
    def load_from_json(self, filepath: str) -> None:
        """Load graph from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Add entities
        for entity_data in data['entities']:
            entity = Entity(**entity_data)
            self.add_entity(entity)
        
        # Add relationships
        for rel_data in data['relationships']:
            rel = Relationship(**rel_data)
            self.add_relationship(rel)
        
        logger.info(f"Loaded graph from {filepath}")

    def extract_call_subgraph(self, 
                             entry_procedures: List[str],
                             max_depth: int = 10,
                             exclude_procedures: Optional[Set[str]] = None,
                             include_variables: bool = True,
                             include_structures: bool = True) -> Dict[str, Any]:
        """
        Extract subgraph starting from entry point procedures
        
        Args:
            entry_procedures: List of procedure names to start from
            max_depth: Maximum call depth to traverse
            exclude_procedures: Set of procedure names to exclude (e.g., common utilities)
            include_variables: Include variables used by procedures
            include_structures: Include structure definitions
        
        Returns:
            Dict with subgraph entities and relationships
        """
        if exclude_procedures is None:
            exclude_procedures = set()
        
        # Find entry point entities
        all_procedures = self.query_entities(entity_type=EntityType.PROCEDURE)
        entry_entities = []
        for proc_name in entry_procedures:
            for proc in all_procedures:
                if proc.name == proc_name:
                    entry_entities.append(proc)
                    break
        
        if not entry_entities:
            logger.warning(f"No entry procedures found from: {entry_procedures}")
            return {
                'entities': [],
                'relationships': [],
                'entry_points': entry_procedures,
                'error': 'No entry procedures found'
            }
        
        logger.info(f"Starting subgraph extraction from {len(entry_entities)} entry points")
        
        # BFS traversal of call graph
        visited_procedures = set()
        subgraph_entities = []
        subgraph_entity_ids = set()
        current_level = entry_entities
        
        for depth in range(max_depth + 1):
            if not current_level:
                break
            
            logger.info(f"Depth {depth}: Processing {len(current_level)} procedures")
            next_level = []
            
            for proc in current_level:
                if proc.id in visited_procedures:
                    continue
                
                # Skip if in exclusion list
                if proc.name in exclude_procedures:
                    logger.debug(f"Skipping excluded procedure: {proc.name}")
                    continue
                
                visited_procedures.add(proc.id)
                subgraph_entities.append(proc)
                subgraph_entity_ids.add(proc.id)
                
                # Get called procedures (outgoing CALLS relationships)
                callees = self.get_neighbors(
                    proc.id,
                    rel_type=RelationType.CALLS,
                    direction="outgoing"
                )
                
                for callee in callees:
                    if (callee.id not in visited_procedures and 
                        callee.name not in exclude_procedures):
                        next_level.append(callee)
            
            current_level = next_level
        
        logger.info(f"Found {len(subgraph_entities)} procedures in call chain")
        
        # Include variables if requested
        if include_variables:
            for proc in subgraph_entities:
                # Get variables contained in this procedure
                variables = self.get_neighbors(
                    proc.id,
                    rel_type=RelationType.CONTAINS,
                    direction="outgoing"
                )
                for var in variables:
                    if var.type == EntityType.VARIABLE:
                        if var.id not in subgraph_entity_ids:
                            subgraph_entities.append(var)
                            subgraph_entity_ids.add(var.id)
        
        # Include structures if requested
        if include_structures:
            structures = self.query_entities(entity_type=EntityType.STRUCTURE)
            for struct in structures:
                # Check if any procedure in subgraph references this structure
                # This is a simple heuristic - you might want more sophisticated logic
                if struct.file_path in [p.file_path for p in subgraph_entities[:len(visited_procedures)]]:
                    if struct.id not in subgraph_entity_ids:
                        subgraph_entities.append(struct)
                        subgraph_entity_ids.add(struct.id)
        
        # Get all relationships between entities in subgraph
        subgraph_relationships = []
        all_relationships = self.query_relationships()
        
        for rel in all_relationships:
            if (rel.source_id in subgraph_entity_ids and 
                rel.target_id in subgraph_entity_ids):
                subgraph_relationships.append(rel)
        
        logger.info(f"Subgraph: {len(subgraph_entities)} entities, "
                   f"{len(subgraph_relationships)} relationships")
        
        return {
            'entities': subgraph_entities,
            'relationships': subgraph_relationships,
            'entry_points': entry_procedures,
            'excluded_procedures': list(exclude_procedures),
            'max_depth': max_depth,
            'statistics': {
                'total_entities': len(subgraph_entities),
                'total_relationships': len(subgraph_relationships),
                'procedure_count': len(visited_procedures),
                'variable_count': sum(1 for e in subgraph_entities if e.type == EntityType.VARIABLE),
                'structure_count': sum(1 for e in subgraph_entities if e.type == EntityType.STRUCTURE)
            }
        }
    
    def get_call_chain(self, from_procedure: str, to_procedure: str, 
                       exclude_procedures: Optional[Set[str]] = None) -> List[List[str]]:
        """
        Find all call paths from one procedure to another
        
        Args:
            from_procedure: Starting procedure name
            to_procedure: Target procedure name
            exclude_procedures: Procedures to exclude from paths
        
        Returns:
            List of call paths (each path is a list of procedure names)
        """
        if self.backend != "networkx":
            raise NotImplementedError("Call path finding only implemented for NetworkX")
        
        # Find procedure entities
        procedures = self.query_entities(entity_type=EntityType.PROCEDURE)
        start_proc = next((p for p in procedures if p.name == from_procedure), None)
        end_proc = next((p for p in procedures if p.name == to_procedure), None)
        
        if not start_proc or not end_proc:
            return []
        
        # Build filtered call graph
        import networkx as nx
        call_graph = nx.DiGraph()
        
        for rel in self.query_relationships(rel_type=RelationType.CALLS):
            source_entity = self.get_entity(rel.source_id)
            target_entity = self.get_entity(rel.target_id)
            
            if source_entity and target_entity:
                # Skip excluded procedures
                if exclude_procedures and (
                    source_entity.name in exclude_procedures or 
                    target_entity.name in exclude_procedures
                ):
                    continue
                
                call_graph.add_edge(
                    source_entity.id, 
                    target_entity.id,
                    source_name=source_entity.name,
                    target_name=target_entity.name
                )
        
        # Find all simple paths
        try:
            paths = list(nx.all_simple_paths(
                call_graph, 
                start_proc.id, 
                end_proc.id, 
                cutoff=10
            ))
            
            # Convert entity IDs to procedure names
            name_paths = []
            for path in paths:
                name_path = []
                for entity_id in path:
                    entity = self.get_entity(entity_id)
                    if entity:
                        name_path.append(entity.name)
                name_paths.append(name_path)
            
            return name_paths
        except nx.NodeNotFound:
            return []

# ============================================================================
# Hashability Tests (Run on import to verify)
# ============================================================================

def _verify_hashability():
    """Verify Entity and Relationship classes are properly hashable"""
    try:
        # Test Entity hashability
        e1 = Entity(
            id="test_1",
            type=EntityType.PROCEDURE,
            name="test_proc",
            qualified_name="file::test_proc"
        )
        e2 = Entity(
            id="test_1",
            type=EntityType.PROCEDURE,
            name="test_proc",
            qualified_name="file::test_proc",
            metadata={"different": "metadata"}  # Different metadata
        )
        
        # Test hashing
        _ = hash(e1)
        _ = hash(e2)
        
        # Test equality
        assert e1 == e2, "Entities with same ID should be equal"
        
        # Test in set
        entity_set = {e1, e2}
        assert len(entity_set) == 1, "Set should deduplicate equal entities"
        
        # Test as dict key
        entity_dict = {e1: "value"}
        assert entity_dict[e2] == "value", "Should retrieve by equal key"
        
        # Test Relationship hashability
        r1 = Relationship(
            source_id="src",
            target_id="tgt",
            type=RelationType.CALLS
        )
        r2 = Relationship(
            source_id="src",
            target_id="tgt",
            type=RelationType.CALLS,
            weight=2.0  # Different weight
        )
        
        # Test hashing
        _ = hash(r1)
        _ = hash(r2)
        
        # Test equality
        assert r1 == r2, "Relationships with same source/target/type should be equal"
        
        # Test in set
        rel_set = {r1, r2}
        assert len(rel_set) == 1, "Set should deduplicate equal relationships"
        
        logger.debug("✓ Hashability verification passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Hashability verification failed: {e}")
        return False


# Run verification on module import
_verify_hashability()
