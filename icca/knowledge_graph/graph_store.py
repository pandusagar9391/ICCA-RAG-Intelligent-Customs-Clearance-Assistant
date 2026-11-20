"""
Customs Knowledge Graph Store for ICCA-RAG.

Manages directed graph representation of customs entities and relations.
Supports:
- NetworkX DiGraph core engine with seamless pure-Python fallback.
- Bidirectional adjacency traversal and entity lookup indexing.
- JSON serialization (`save_json`, `load_json`).
- Export to GML and GraphML standard formats.
- Neo4j readiness: connection stub, schema sync, and Cypher query execution.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from icca.knowledge_graph.schema import (
    CustomsEntity,
    CustomsRelation,
    EntityType,
    RelationType,
)

logger = logging.getLogger(__name__)

# Try importing NetworkX; provide pure-Python DiGraph fallback if not installed.
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    nx = None
    HAS_NETWORKX = False


class _InMemoryNodeView:
    """Dictionary-like view for graph nodes."""
    def __init__(self, node_dict: Dict[str, Dict[str, Any]]):
        self._nodes = node_dict

    def __getitem__(self, node: str) -> Dict[str, Any]:
        return self._nodes[node]

    def __iter__(self):
        return iter(self._nodes)

    def __len__(self):
        return len(self._nodes)

    def __contains__(self, node):
        return node in self._nodes

    def items(self):
        return self._nodes.items()

    def data(self, data: Union[bool, str] = True, default: Any = None):
        if data is True:
            for k, v in self._nodes.items():
                yield k, v
        elif isinstance(data, str):
            for k, v in self._nodes.items():
                yield k, v.get(data, default)
        else:
            for k in self._nodes.keys():
                yield k


class InMemoryDiGraph:
    """
    Lightweight, high-performance pure-Python DiGraph providing NetworkX-compatible API.
    Used when NetworkX is not installed or in resource-constrained environments.
    """
    def __init__(self):
        self._succ: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._pred: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._node_data: Dict[str, Dict[str, Any]] = {}
        self.graph: Dict[str, Any] = {}

    @property
    def nodes(self) -> _InMemoryNodeView:
        return _InMemoryNodeView(self._node_data)

    @property
    def succ(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        return self._succ

    @property
    def pred(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        return self._pred

    def add_node(self, node_for_adding: str, **attr):
        node = str(node_for_adding)
        if node not in self._succ:
            self._succ[node] = {}
            self._pred[node] = {}
            self._node_data[node] = {}
        self._node_data[node].update(attr)

    def add_edge(self, u_of_edge: str, v_of_edge: str, **attr):
        u, v = str(u_of_edge), str(v_of_edge)
        if u not in self._succ:
            self.add_node(u)
        if v not in self._succ:
            self.add_node(v)
        datadict = self._succ[u].get(v, {})
        datadict.update(attr)
        self._succ[u][v] = datadict
        self._pred[v][u] = datadict

    def has_node(self, n: str) -> bool:
        return str(n) in self._node_data

    def has_edge(self, u: str, v: str) -> bool:
        u_str, v_str = str(u), str(v)
        return u_str in self._succ and v_str in self._succ[u_str]

    def successors(self, n: str):
        n_str = str(n)
        if n_str not in self._succ:
            return iter([])
        return iter(list(self._succ[n_str].keys()))

    def predecessors(self, n: str):
        n_str = str(n)
        if n_str not in self._pred:
            return iter([])
        return iter(list(self._pred[n_str].keys()))

    def neighbors(self, n: str):
        return self.successors(n)

    def in_edges(self, n: Optional[str] = None, data: bool = False):
        if n is not None:
            n_str = str(n)
            if n_str not in self._pred:
                return []
            if data:
                return [(u, n_str, self._pred[n_str][u]) for u in self._pred[n_str]]
            return [(u, n_str) for u in self._pred[n_str]]
        # All in edges
        edges = []
        for v, u_dict in self._pred.items():
            for u, d in u_dict.items():
                edges.append((u, v, d) if data else (u, v))
        return edges

    def out_edges(self, n: Optional[str] = None, data: bool = False):
        if n is not None:
            n_str = str(n)
            if n_str not in self._succ:
                return []
            if data:
                return [(n_str, v, self._succ[n_str][v]) for v in self._succ[n_str]]
            return [(n_str, v) for v in self._succ[n_str]]
        # All out edges
        edges = []
        for u, v_dict in self._succ.items():
            for v, d in v_dict.items():
                edges.append((u, v, d) if data else (u, v))
        return edges

    def number_of_nodes(self) -> int:
        return len(self._node_data)

    def number_of_edges(self) -> int:
        return sum(len(nbrs) for nbrs in self._succ.values())

    def clear(self):
        self._succ.clear()
        self._pred.clear()
        self._node_data.clear()
        self.graph.clear()


class CustomsGraphStore:
    """
    Core graph store for the Customs Knowledge Graph.
    Uses NetworkX DiGraph when available, or InMemoryDiGraph fallback.
    """

    def __init__(self, name: str = "ICCA_Customs_KG"):
        self.name = name
        self.graph = nx.DiGraph(name=name) if HAS_NETWORKX else InMemoryDiGraph()
        self.graph.graph["name"] = name

        # Entity and relation catalogs
        self.entities: Dict[str, CustomsEntity] = {}
        self.relations: List[CustomsRelation] = []

        # Secondary indexes for ultra-fast lookup
        self.type_index: Dict[str, Set[str]] = {}
        self.keyword_index: Dict[str, Set[str]] = {}

        # Neo4j integration state
        self._neo4j_driver = None
        self._neo4j_connected = False
        self._neo4j_config: Dict[str, Any] = {}

    # -------------------------------------------------------------------------
    # Graph Mutation Methods
    # -------------------------------------------------------------------------

    def add_entity(self, entity: Union[CustomsEntity, Dict[str, Any]]) -> CustomsEntity:
        """
        Add a customs entity node to the graph and update internal indexes.
        
        Args:
            entity: CustomsEntity instance or dictionary representation.
            
        Returns:
            The added CustomsEntity object.
        """
        if isinstance(entity, dict):
            entity_obj = CustomsEntity.from_dict(entity)
        elif isinstance(entity, CustomsEntity):
            entity_obj = entity
        else:
            raise TypeError(f"Expected CustomsEntity or dict, got {type(entity)}")

        e_id = entity_obj.id
        self.entities[e_id] = entity_obj

        # Add to graph structure
        node_attrs = {
            "type": str(entity_obj.type),
            "name": str(entity_obj.name),
            "description": str(entity_obj.description),
            **entity_obj.properties,
        }
        self.graph.add_node(e_id, **node_attrs)

        # Update type index
        e_type = str(entity_obj.type)
        if e_type not in self.type_index:
            self.type_index[e_type] = set()
        self.type_index[e_type].add(e_id)

        # Update keyword index
        self._index_entity_keywords(entity_obj)

        return entity_obj

    def add_relation(self, relation: Union[CustomsRelation, Dict[str, Any]]) -> CustomsRelation:
        """
        Add a directed customs relation edge to the graph.
        
        Args:
            relation: CustomsRelation instance or dictionary representation.
            
        Returns:
            The added CustomsRelation object.
        """
        if isinstance(relation, dict):
            rel_obj = CustomsRelation.from_dict(relation)
        elif isinstance(relation, CustomsRelation):
            rel_obj = relation
        else:
            raise TypeError(f"Expected CustomsRelation or dict, got {type(relation)}")

        src, tgt = rel_obj.source_id, rel_obj.target_id

        # Ensure endpoints exist as entities (create placeholders if not yet added)
        if src not in self.entities:
            placeholder_src = CustomsEntity(id=src, type="UNKNOWN", name=src)
            self.add_entity(placeholder_src)

        if tgt not in self.entities:
            placeholder_tgt = CustomsEntity(id=tgt, type="UNKNOWN", name=tgt)
            self.add_entity(placeholder_tgt)

        self.relations.append(rel_obj)

        # Edge attributes in graph
        edge_attrs = {
            "relation_type": str(rel_obj.relation_type),
            "weight": float(rel_obj.weight),
            **rel_obj.properties,
        }
        self.graph.add_edge(src, tgt, **edge_attrs)

        return rel_obj

    def _index_entity_keywords(self, entity: CustomsEntity):
        """Index search tokens from name, description, and properties."""
        text_corpus = f"{entity.id} {entity.name} {entity.description}"
        # Include list or string properties
        for k, v in entity.properties.items():
            if isinstance(v, (str, int, float)):
                text_corpus += f" {v}"
            elif isinstance(v, (list, tuple)):
                text_corpus += " " + " ".join(str(item) for item in v)

        # Tokenize (alphanumeric sequences, including dotted HS codes like 8471.30)
        tokens = re.findall(r"[a-z0-9]+(?:[\.\-][a-z0-9]+)*", text_corpus.lower())
        for token in tokens:
            token = token.strip(".-")
            if len(token) >= 2:
                if token not in self.keyword_index:
                    self.keyword_index[token] = set()
                self.keyword_index[token].add(entity.id)

    # -------------------------------------------------------------------------
    # Graph Query & Retrieval Methods
    # -------------------------------------------------------------------------

    def get_entity(self, entity_id: str) -> Optional[CustomsEntity]:
        """Retrieve an entity by its unique ID."""
        return self.entities.get(str(entity_id))

    def get_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: Optional[str] = None
    ) -> List[CustomsRelation]:
        """
        Find relations between two entities matching an optional relation_type.
        """
        src = str(source_id)
        tgt = str(target_id)
        matched = []
        for rel in self.relations:
            if rel.source_id == src and rel.target_id == tgt:
                if relation_type is None or rel.relation_type == relation_type:
                    matched.append(rel)
        return matched

    def get_neighbors(
        self,
        entity_id: str,
        direction: str = "out"
    ) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        Retrieve neighbors of an entity.
        
        Args:
            entity_id: Target entity ID.
            direction: 'out' (successors), 'in' (predecessors), or 'both'.
            
        Returns:
            List of tuples: (neighbor_id, relation_type, edge_attributes).
        """
        e_id = str(entity_id)
        if not self.graph.has_node(e_id):
            return []

        neighbors: List[Tuple[str, str, Dict[str, Any]]] = []

        # Outgoing edges: e_id -> neighbor
        if direction in ("out", "both"):
            succ_edges = self.graph.out_edges(e_id, data=True)
            for _, tgt, data in succ_edges:
                rel_type = data.get("relation_type", "RELATED_TO")
                neighbors.append((tgt, rel_type, data))

        # Incoming edges: neighbor -> e_id
        if direction in ("in", "both"):
            pred_edges = self.graph.in_edges(e_id, data=True)
            for src, _, data in pred_edges:
                rel_type = data.get("relation_type", "RELATED_TO")
                neighbors.append((src, rel_type, data))

        return neighbors

    def get_entities_by_type(self, entity_type: Union[str, EntityType]) -> List[CustomsEntity]:
        """Retrieve all entities of a specified type."""
        t_key = entity_type.value if isinstance(entity_type, EntityType) else str(entity_type)
        entity_ids = self.type_index.get(t_key, set())
        return [self.entities[eid] for eid in entity_ids if eid in self.entities]

    def find_entities_by_keyword(self, keyword: str) -> List[CustomsEntity]:
        """
        Search entities by keyword or partial term across names, IDs, and descriptions.
        """
        if not keyword:
            return []

        kw = keyword.lower().strip()
        matched_ids: Set[str] = set()

        # Direct exact or prefix hit in keyword index
        if kw in self.keyword_index:
            matched_ids.update(self.keyword_index[kw])

        # Partial token match
        for indexed_token, ids in self.keyword_index.items():
            if kw in indexed_token or indexed_token in kw:
                matched_ids.update(ids)

        # Full scan fallback for phrases
        if not matched_ids or " " in kw:
            for eid, entity in self.entities.items():
                if kw in entity.name.lower() or kw in entity.id.lower() or kw in entity.description.lower():
                    matched_ids.add(eid)

        return [self.entities[eid] for eid in matched_ids if eid in self.entities]

    def get_stats(self) -> Dict[str, Any]:
        """Return graph size, entity distribution, and relation breakdown."""
        rel_type_counts: Dict[str, int] = {}
        for rel in self.relations:
            r_type = rel.relation_type
            rel_type_counts[r_type] = rel_type_counts.get(r_type, 0) + 1

        entity_type_counts: Dict[str, int] = {
            etype: len(eids) for etype, eids in self.type_index.items()
        }

        return {
            "name": self.name,
            "engine": "networkx" if HAS_NETWORKX else "in_memory_fallback",
            "total_entities": len(self.entities),
            "total_relations": len(self.relations),
            "entities_by_type": entity_type_counts,
            "relations_by_type": rel_type_counts,
        }

    def clear(self):
        """Reset the graph store to an empty state."""
        self.graph.clear()
        self.entities.clear()
        self.relations.clear()
        self.type_index.clear()
        self.keyword_index.clear()

    # -------------------------------------------------------------------------
    # Serialization (JSON, GML, GraphML)
    # -------------------------------------------------------------------------

    def save_json(self, path: Union[str, Path]) -> None:
        """
        Serialize entire knowledge graph to a standardized JSON document.
        """
        out_path = Path(path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "schema_version": "1.0",
            "name": self.name,
            "stats": self.get_stats(),
            "entities": [entity.to_dict() for entity in self.entities.values()],
            "relations": [rel.to_dict() for rel in self.relations],
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        logger.info(f"Customs KG saved successfully to {out_path} ({len(self.entities)} entities, {len(self.relations)} relations).")

    def load_json(self, path: Union[str, Path]) -> None:
        """
        Load knowledge graph entities and relations from a JSON file.
        """
        in_path = Path(path).resolve()
        if not in_path.exists():
            raise FileNotFoundError(f"Customs KG JSON file not found at: {in_path}")

        with open(in_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        self.clear()
        self.name = payload.get("name", self.name)
        if hasattr(self.graph, "name"):
            self.graph.name = self.name

        # Load entities first
        for e_dict in payload.get("entities", []):
            self.add_entity(e_dict)

        # Load relations
        for r_dict in payload.get("relations", []):
            self.add_relation(r_dict)

        logger.info(f"Loaded {len(self.entities)} entities and {len(self.relations)} relations from {in_path}.")

    def export_gml(self, path: Union[str, Path]) -> None:
        """
        Export graph in standard GML (Graph Modelling Language) format.
        """
        out_path = Path(path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if HAS_NETWORKX:
            # Prepare clean copy suitable for NetworkX GML exporter
            clean_g = nx.DiGraph()
            for eid, entity in self.entities.items():
                clean_g.add_node(eid, label=entity.name, type=str(entity.type))
            for rel in self.relations:
                clean_g.add_edge(rel.source_id, rel.target_id, relation_type=str(rel.relation_type))
            nx.write_gml(clean_g, str(out_path))
        else:
            # Fallback pure-python standard GML writer
            lines = ["graph [", "  directed 1"]
            for eid, entity in self.entities.items():
                clean_name = entity.name.replace('"', "'")
                lines.append("  node [")
                lines.append(f'    id "{eid}"')
                lines.append(f'    label "{clean_name}"')
                lines.append(f'    type "{entity.type}"')
                lines.append("  ]")
            for rel in self.relations:
                lines.append("  edge [")
                lines.append(f'    source "{rel.source_id}"')
                lines.append(f'    target "{rel.target_id}"')
                lines.append(f'    relation_type "{rel.relation_type}"')
                lines.append(f"    weight {rel.weight}")
                lines.append("  ]")
            lines.append("]")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

        logger.info(f"Exported Customs KG to GML at {out_path}.")

    def export_graphml(self, path: Union[str, Path]) -> None:
        """
        Export graph in standard GraphML XML format.
        """
        out_path = Path(path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if HAS_NETWORKX:
            clean_g = nx.DiGraph()
            for eid, entity in self.entities.items():
                clean_g.add_node(eid, name=entity.name, type=str(entity.type))
            for rel in self.relations:
                clean_g.add_edge(rel.source_id, rel.target_id, relation_type=str(rel.relation_type))
            nx.write_graphml(clean_g, str(out_path))
        else:
            # Fallback pure-python standard GraphML writer
            xml_lines = [
                '<?xml version="1.0" encoding="UTF-8"?>',
                '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">',
                '  <key id="name" for="node" attr.name="name" attr.type="string"/>',
                '  <key id="type" for="node" attr.name="type" attr.type="string"/>',
                '  <key id="rel" for="edge" attr.name="relation_type" attr.type="string"/>',
                '  <graph id="G" edgedefault="directed">',
            ]
            for eid, entity in self.entities.items():
                esc_name = entity.name.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                xml_lines.append(f'    <node id="{eid}">')
                xml_lines.append(f'      <data key="name">{esc_name}</data>')
                xml_lines.append(f'      <data key="type">{entity.type}</data>')
                xml_lines.append('    </node>')

            for i, rel in enumerate(self.relations):
                xml_lines.append(f'    <edge id="e{i}" source="{rel.source_id}" target="{rel.target_id}">')
                xml_lines.append(f'      <data key="rel">{rel.relation_type}</data>')
                xml_lines.append('    </edge>')

            xml_lines.append('  </graph>')
            xml_lines.append('</graphml>')
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("\n".join(xml_lines))

        logger.info(f"Exported Customs KG to GraphML at {out_path}.")

    # -------------------------------------------------------------------------
    # Neo4j Enterprise Extension Points & Cypher Interface
    # -------------------------------------------------------------------------

    def connect_neo4j(
        self,
        uri: str = "bolt://localhost:7687",
        auth: Tuple[str, str] = ("neo4j", "password"),
        **kwargs
    ) -> bool:
        """
        Connect to external Neo4j instance if driver and server are available.
        Provides a seamless plug-in extension point for enterprise Neo4j graphs.
        
        Args:
            uri: Bolt or Neo4j URI (e.g., bolt://localhost:7687 or neo4j+s://...).
            auth: Tuple of (username, password).
            
        Returns:
            bool: True if successfully connected; False otherwise.
        """
        self._neo4j_config = {"uri": uri, "auth": auth, **kwargs}
        try:
            from neo4j import GraphDatabase
            self._neo4j_driver = GraphDatabase.driver(uri, auth=auth, **kwargs)
            # Verify connectivity
            with self._neo4j_driver.session() as session:
                session.run("RETURN 1 AS test")
            self._neo4j_connected = True
            logger.info(f"Successfully connected to Neo4j database at {uri}.")
            return True
        except ImportError:
            logger.warning("neo4j Python package is not installed. Neo4j connectivity disabled.")
            self._neo4j_connected = False
            return False
        except Exception as e:
            logger.warning(f"Could not connect to Neo4j instance at {uri}: {e}")
            self._neo4j_connected = False
            return False

    def sync_to_neo4j(self) -> Dict[str, Any]:
        """
        Synchronize the current in-memory graph into the connected Neo4j database.
        Executes MERGE statements for entities and relations.
        """
        if not self._neo4j_connected or self._neo4j_driver is None:
            return {
                "status": "skipped",
                "message": "Neo4j is not connected. Use connect_neo4j() first.",
                "synced_nodes": 0,
                "synced_edges": 0,
            }

        synced_nodes = 0
        synced_edges = 0

        try:
            with self._neo4j_driver.session() as session:
                # Sync nodes
                for entity in self.entities.values():
                    cypher = (
                        f"MERGE (n:`{entity.type}` {{id: $id}}) "
                        "SET n.name = $name, n.description = $desc, n += $props"
                    )
                    session.run(
                        cypher,
                        id=entity.id,
                        name=entity.name,
                        desc=entity.description,
                        props=entity.properties,
                    )
                    synced_nodes += 1

                # Sync edges
                for rel in self.relations:
                    cypher = (
                        "MATCH (a {id: $source_id}), (b {id: $target_id}) "
                        f"MERGE (a)-[r:`{rel.relation_type}`]->(b) "
                        "SET r.weight = $weight, r += $props"
                    )
                    session.run(
                        cypher,
                        source_id=rel.source_id,
                        target_id=rel.target_id,
                        weight=rel.weight,
                        props=rel.properties,
                    )
                    synced_edges += 1

            return {
                "status": "success",
                "message": f"Synced {synced_nodes} nodes and {synced_edges} edges to Neo4j.",
                "synced_nodes": synced_nodes,
                "synced_edges": synced_edges,
            }
        except Exception as e:
            logger.error(f"Error syncing graph to Neo4j: {e}")
            return {
                "status": "error",
                "message": str(e),
                "synced_nodes": synced_nodes,
                "synced_edges": synced_edges,
            }

    def query_cypher(
        self,
        cypher_query: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute arbitrary Cypher query against connected Neo4j instance.
        If Neo4j is not connected, returns an explanatory status.
        """
        if not self._neo4j_connected or self._neo4j_driver is None:
            logger.warning("Cypher query attempted without active Neo4j connection.")
            return [{
                "status": "unsupported_offline",
                "message": "Neo4j is not connected. Use connect_neo4j() with live database credentials.",
                "query": cypher_query,
            }]

        try:
            with self._neo4j_driver.session() as session:
                result = session.run(cypher_query, parameters or {})
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Failed to execute Cypher query: {e}")
            return [{"status": "error", "error": str(e)}]
